//! Resident lazy-prefix sequence for G4/byte and G8/nibble Instruction RA.
//!
//! The lazy prefix gathers from the stage-5 lookup plane while each bind doubles
//! the branch tables. At the configured width, the final gather writes
//! factor-major dense tables before releasing the lookup plane.

use std::{ffi::c_void, mem::size_of, slice};

use jolt_field::{Prime128OffsetA7F7 as AkitaField, Zero as _};
use metal::{
    objc::rc::autoreleasepool, Buffer, ComputePipelineState, FunctionConstantValues, MTLDataType,
    MTLResourceOptions, MTLSize,
};

use super::{
    encode_column_reductions, set_inline_bytes, validate_completed_command, Fp128, MetalError,
    PipelineLimits, ResidentLookupIndexPlane, SolinasMetal,
};

const G4_TABLE_VALUES: usize = 16 * 256;
const G8_TABLE_VALUES: usize = 32 * 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Geometry {
    groups: usize,
    chunk_bits: usize,
}

impl Geometry {
    fn from_table_values(values: usize) -> Result<Self, MetalError> {
        match values {
            G4_TABLE_VALUES => Ok(Self {
                groups: 4,
                chunk_bits: 8,
            }),
            G8_TABLE_VALUES => Ok(Self {
                groups: 8,
                chunk_bits: 4,
            }),
            _ => Err(MetalError::InvalidInstructionRaState(
                "instruction RA requires G4 byte or G8 nibble tables",
            )),
        }
    }

    const fn factors(self) -> usize {
        self.groups * 4
    }
    const fn bins(self) -> usize {
        1 << self.chunk_bits
    }
    const fn table_values(self) -> usize {
        self.factors() * self.bins()
    }
    const fn materialize_params(self) -> u32 {
        ((self.groups as u32) << 8) | self.chunk_bits as u32
    }
}
const SAMPLES: usize = 4;
const SIMD_WIDTH: usize = 32;
const DEFAULT_MESSAGE_THREADS: usize = 128;
const DEFAULT_MATERIALIZE_THREADS: usize = 64;
const BRANCH_THREADS: usize = 256;
const MESSAGE_WIDTH_1_PIPELINE: &str = "solinas_instruction_ra_first_message";
const NIBBLE_WIDTH_1_PIPELINE: &str = "solinas_instruction_ra_nibble_first_message";
const MESSAGE_WIDTH_2_PIPELINE: &str = "solinas_instruction_ra_message_width_2";
const MESSAGE_WIDTH_4_PIPELINE: &str = "solinas_instruction_ra_message_width_4";
const MESSAGE_WIDTH_8_PIPELINE: &str = "solinas_instruction_ra_message_width_8";
const MESSAGE_WIDE_PIPELINE: &str = "solinas_instruction_ra_message_wide";
const DOUBLE_PIPELINE: &str = "solinas_instruction_ra_double_branches";
const MATERIALIZE_WIDTH_16_PIPELINE: &str = "solinas_instruction_ra_materialize_width_16";
const MATERIALIZE_WIDE_PIPELINE: &str = "solinas_instruction_ra_materialize_wide";
const DENSE_TRANSITION_PIPELINE: &str = "solinas_instruction_ra_dense_transition";
const WEIGHT_PREFIX_PIPELINE: &str = "solinas_instruction_ra_weight_prefix";
const TERMINAL_BIND_PIPELINE: &str = "solinas_instruction_ra_bind_terminal";
const REDUCE_PIPELINE: &str = "solinas_instruction_ra_reduce";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum InstructionRaMaterializeWidth {
    W16 = 16,
    W32 = 32,
    W64 = 64,
    W128 = 128,
    W256 = 256,
    W512 = 512,
}

impl InstructionRaMaterializeWidth {
    pub const fn elements(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionRaSequenceConfig {
    pub message_threads_per_threadgroup: Option<usize>,
    pub materialize_threads_per_threadgroup: Option<usize>,
    pub materialize_width: InstructionRaMaterializeWidth,
    /// Overwrites the resident inverse map after its final gather; the plane is one-shot.
    pub reuse_inverse_for_dense: bool,
}

impl Default for InstructionRaSequenceConfig {
    fn default() -> Self {
        Self {
            message_threads_per_threadgroup: Some(DEFAULT_MESSAGE_THREADS),
            materialize_threads_per_threadgroup: Some(DEFAULT_MATERIALIZE_THREADS),
            materialize_width: InstructionRaMaterializeWidth::W16,
            reuse_inverse_for_dense: false,
        }
    }
}

#[cfg(test)]
fn instruction_ra_weight_capacities(rows: usize) -> Result<(usize, usize), MetalError> {
    if rows < 32 || !rows.is_power_of_two() {
        return Err(MetalError::InvalidInstructionRaRows(rows));
    }
    let e_out_capacity = 1usize << (rows.ilog2() / 2);
    let e_in_capacity = (rows / 2) / e_out_capacity;
    Ok((e_in_capacity, e_out_capacity))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionRaSequenceScratchLayout {
    pub branch_a_bytes: u64,
    pub branch_b_bytes: u64,
    pub dense_a_bytes: u64,
    pub dense_b_active_bytes: u64,
    pub dense_b_owned_bytes: u64,
    pub dense_b_physical_bytes: u64,
}

impl InstructionRaSequenceScratchLayout {
    pub const fn owned_bytes(self) -> u64 {
        self.branch_a_bytes + self.branch_b_bytes + self.dense_a_bytes + self.dense_b_owned_bytes
    }

    pub const fn resident_bytes_after_handoff(self) -> u64 {
        self.branch_a_bytes + self.branch_b_bytes + self.dense_a_bytes + self.dense_b_physical_bytes
    }
}

impl InstructionRaSequenceConfig {
    pub fn scratch_layout(
        self,
        rows: usize,
        chunk_table_values: usize,
    ) -> Result<InstructionRaSequenceScratchLayout, MetalError> {
        let geometry = Geometry::from_table_values(chunk_table_values)?;
        let materialize_width = self.materialize_width.elements();
        if rows < 2 * materialize_width || !rows.is_power_of_two() {
            return Err(MetalError::InvalidInstructionRaRows(rows));
        }
        if self.reuse_inverse_for_dense && materialize_width < 2 * geometry.factors() {
            return Err(MetalError::InvalidInstructionRaState(
                "the resident inverse buffer is too small for this geometry and materialization width",
            ));
        }
        let (branch_a_width, branch_b_width) = branch_capacity_widths(materialize_width);
        let branch_a_values = geometry
            .factors()
            .checked_mul(branch_a_width)
            .and_then(|values| values.checked_mul(geometry.bins()))
            .ok_or(MetalError::InputTooLong(rows))?;
        let branch_b_values = geometry
            .factors()
            .checked_mul(branch_b_width)
            .and_then(|values| values.checked_mul(geometry.bins()))
            .ok_or(MetalError::InputTooLong(rows))?;
        let dense_a_values = geometry
            .factors()
            .checked_mul(rows / materialize_width)
            .ok_or(MetalError::InputTooLong(rows))?;
        let dense_b_active_values = dense_a_values / 2;
        Ok(InstructionRaSequenceScratchLayout {
            branch_a_bytes: byte_length::<Fp128>(branch_a_values)?,
            branch_b_bytes: byte_length::<Fp128>(branch_b_values)?,
            dense_a_bytes: byte_length::<Fp128>(dense_a_values)?,
            dense_b_active_bytes: byte_length::<Fp128>(dense_b_active_values)?,
            dense_b_owned_bytes: if self.reuse_inverse_for_dense {
                0
            } else {
                byte_length::<Fp128>(dense_b_active_values)?
            },
            dense_b_physical_bytes: if self.reuse_inverse_for_dense {
                byte_length::<u32>(rows)?
            } else {
                byte_length::<Fp128>(dense_b_active_values)?
            },
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MessageParams {
    e_in_length: u32,
    e_out_length: u32,
    // Shares the common G4 first-message ABI's reserved uint2. Lazy kernels
    // interpret these words as group count and committed chunk width.
    geometry: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BranchParams {
    branch_width: u32,
    factor_count: u32,
    bins: u32,
    _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MaterializeParams {
    source_elements: u32,
    e_in_length: u32,
    e_out_length: u32,
    // The Metal materialization ABI reserves one scalar: groups << 8 | bits.
    geometry: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WeightPrefixParams {
    source_offset: u32,
    destination_offset: u32,
    destination_elements: u32,
    _reserved: u32,
}

struct Pipelines {
    width_1: ComputePipelineState,
    width_2: ComputePipelineState,
    width_4: ComputePipelineState,
    width_8: ComputePipelineState,
    wide_messages: Vec<(usize, ComputePipelineState)>,
    double: ComputePipelineState,
    materialize: ComputePipelineState,
    dense_transition: ComputePipelineState,
    weight_prefix: ComputePipelineState,
    terminal_bind: ComputePipelineState,
    reduce: ComputePipelineState,
}

struct Buffers {
    branches_a: Buffer,
    branches_b: Buffer,
    dense_a: Buffer,
    dense_b: Option<Buffer>,
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
}

pub(crate) struct InstructionRaSequenceStorage {
    context: SolinasMetal,
    config: InstructionRaSequenceConfig,
    geometry: Geometry,
    pipelines: Pipelines,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    rows: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
    message_threads_per_threadgroup: usize,
    materialize_threads_per_threadgroup: usize,
    branch_threads_per_threadgroup: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for InstructionRaSequenceStorage {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_storage"),
            device_storage_bytes(&self.buffers),
        );
        visitor.exit();
    }
}

pub struct InstructionRaSequence {
    context: SolinasMetal,
    geometry: Geometry,
    lookup_plane: Option<ResidentLookupIndexPlane>,
    pipelines: Pipelines,
    reduction_limits: PipelineLimits,
    buffers: Buffers,
    rows: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
    message_threads_per_threadgroup: usize,
    materialize_threads_per_threadgroup: usize,
    branch_threads_per_threadgroup: usize,
    materialize_width: usize,
    reuse_inverse_for_dense: bool,
    branch_width: usize,
    branches_in_a: bool,
    dense: bool,
    dense_in_a: bool,
    dense_elements: usize,
    e_in_elements: usize,
    e_out_elements: usize,
    weights_initialized: bool,
    weight_prefixes_ready: bool,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for InstructionRaSequence {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("device_storage"),
            device_storage_bytes(&self.buffers),
        );
        if let Some(plane) = &self.lookup_plane {
            visitor.visit_field(allocative::Key::new("lookup_plane"), plane);
        }
        visitor.exit();
    }
}

impl SolinasMetal {
    #[cfg(test)]
    pub(crate) fn prepare_instruction_ra_test_sequence(
        &self,
        lookups: &[u128],
        chunk_tables: &[AkitaField],
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionRaSequenceConfig,
    ) -> Result<InstructionRaSequence, MetalError> {
        let rows = lookups.len();
        if rows < 32 || !rows.is_power_of_two() {
            return Err(MetalError::InvalidInstructionRaRows(rows));
        }
        // Exercise a non-identity cycle-to-table mapping as in the stage-5 scatter.
        let mut stored = vec![[0u64; 2]; rows];
        let mut inverse = vec![0u32; rows];
        for (row, lookup) in lookups.iter().enumerate() {
            let destination = row * 17 % rows;
            stored[destination] = [*lookup as u64, (*lookup >> 64) as u64];
            inverse[row] = destination as u32;
        }
        let plane = ResidentLookupIndexPlane::from_buffers(
            super::buffer_from_slice(&self.device, &stored),
            super::buffer_from_slice(&self.device, &inverse),
            rows,
            self.device_registry_id(),
        );
        self.prepare_instruction_ra_sequence_with_plane(
            plane,
            chunk_tables,
            e_in_capacity,
            e_out_capacity,
            config,
        )
    }

    fn compile_instruction_ra_width_pipeline(
        &self,
        name: &'static str,
        width: usize,
    ) -> Result<ComputePipelineState, MetalError> {
        let _span = tracing::info_span!(
            "MetalSolinas::pipeline_compile",
            pipeline = name,
            specialized = true,
            width
        )
        .entered();
        let width = u32::try_from(width).map_err(|_| MetalError::InputTooLong(width))?;
        let constants = FunctionConstantValues::new();
        constants.set_constant_value_at_index(
            std::ptr::from_ref(&width).cast::<c_void>(),
            MTLDataType::UInt,
            0,
        );
        let function = self
            .library
            .get_function(name, Some(constants))
            .map_err(|message| MetalError::FunctionLookup { name, message })?;
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|message| MetalError::PipelineCompilation { name, message })
    }

    pub(crate) fn prepare_instruction_ra_sequence_with_plane(
        &self,
        plane: ResidentLookupIndexPlane,
        chunk_tables: &[AkitaField],
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionRaSequenceConfig,
    ) -> Result<InstructionRaSequence, MetalError> {
        let rows = plane.len();
        let storage = self.prepare_instruction_ra_sequence_storage(
            rows,
            e_in_capacity,
            e_out_capacity,
            config,
            chunk_tables.len(),
        )?;
        storage.attach(plane, chunk_tables)
    }

    pub(crate) fn prepare_instruction_ra_sequence_storage(
        &self,
        rows: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionRaSequenceConfig,
        chunk_table_values: usize,
    ) -> Result<InstructionRaSequenceStorage, MetalError> {
        let geometry = Geometry::from_table_values(chunk_table_values)?;
        let materialize_width = config.materialize_width.elements();
        if rows < 2 * materialize_width || !rows.is_power_of_two() {
            return Err(MetalError::InvalidInstructionRaRows(rows));
        }
        if config.reuse_inverse_for_dense && materialize_width < 2 * geometry.factors() {
            return Err(MetalError::InvalidInstructionRaState(
                "the resident inverse buffer is too small for this geometry and materialization width",
            ));
        }
        let covered = e_in_capacity
            .checked_mul(e_out_capacity)
            .ok_or(MetalError::InputTooLong(rows))?;
        if e_in_capacity == 0 || e_out_capacity == 0 || covered != rows / 2 {
            return Err(MetalError::InstructionRaWeightShape {
                expected: rows / 2,
                covered,
            });
        }
        let _ = u32::try_from(rows).map_err(|_| MetalError::InputTooLong(rows))?;
        let e_in_storage = weight_storage_elements(e_in_capacity)?;
        let e_out_storage = weight_storage_elements(e_out_capacity)?;

        let mut wide_messages = Vec::new();
        let mut width = 16;
        while width < materialize_width {
            wide_messages.push((
                width,
                self.compile_instruction_ra_width_pipeline(MESSAGE_WIDE_PIPELINE, width)?,
            ));
            width *= 2;
        }
        let (materialize_pipeline_name, materialize) = if materialize_width == 16 {
            (
                MATERIALIZE_WIDTH_16_PIPELINE,
                self.compile_named_pipeline(MATERIALIZE_WIDTH_16_PIPELINE)?,
            )
        } else {
            (
                MATERIALIZE_WIDE_PIPELINE,
                self.compile_instruction_ra_width_pipeline(
                    MATERIALIZE_WIDE_PIPELINE,
                    materialize_width,
                )?,
            )
        };
        let width_1_name = if geometry.chunk_bits == 4 {
            NIBBLE_WIDTH_1_PIPELINE
        } else {
            MESSAGE_WIDTH_1_PIPELINE
        };
        let pipelines = Pipelines {
            width_1: self.compile_named_pipeline(width_1_name)?,
            width_2: self.compile_named_pipeline(MESSAGE_WIDTH_2_PIPELINE)?,
            width_4: self.compile_named_pipeline(MESSAGE_WIDTH_4_PIPELINE)?,
            width_8: self.compile_named_pipeline(MESSAGE_WIDTH_8_PIPELINE)?,
            wide_messages,
            double: self.compile_named_pipeline(DOUBLE_PIPELINE)?,
            materialize,
            dense_transition: self.compile_named_pipeline(DENSE_TRANSITION_PIPELINE)?,
            weight_prefix: self.compile_named_pipeline(WEIGHT_PREFIX_PIPELINE)?,
            terminal_bind: self.compile_named_pipeline(TERMINAL_BIND_PIPELINE)?,
            reduce: self.compile_named_pipeline(REDUCE_PIPELINE)?,
        };
        let message_limits = Self::limits(&pipelines.width_1);
        let materialize_limits = Self::limits(&pipelines.materialize);
        let reduction_limits = Self::limits(&pipelines.reduce);
        for (pipeline, limits) in [
            (width_1_name, message_limits),
            (MESSAGE_WIDTH_2_PIPELINE, Self::limits(&pipelines.width_2)),
            (MESSAGE_WIDTH_4_PIPELINE, Self::limits(&pipelines.width_4)),
            (MESSAGE_WIDTH_8_PIPELINE, Self::limits(&pipelines.width_8)),
            (DOUBLE_PIPELINE, Self::limits(&pipelines.double)),
            (materialize_pipeline_name, materialize_limits),
            (
                DENSE_TRANSITION_PIPELINE,
                Self::limits(&pipelines.dense_transition),
            ),
            (
                WEIGHT_PREFIX_PIPELINE,
                Self::limits(&pipelines.weight_prefix),
            ),
            (
                TERMINAL_BIND_PIPELINE,
                Self::limits(&pipelines.terminal_bind),
            ),
            (REDUCE_PIPELINE, reduction_limits),
        ] {
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedInstructionRaExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        for (_, pipeline) in &pipelines.wide_messages {
            let limits = Self::limits(pipeline);
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedInstructionRaExecutionWidth {
                    pipeline: MESSAGE_WIDE_PIPELINE,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let message_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.message_threads_per_threadgroup,
            message_limits,
        )?;
        let materialize_threads_per_threadgroup = Self::resolve_threadgroup_width(
            config.materialize_threads_per_threadgroup,
            materialize_limits,
        )?;
        let branch_threads_per_threadgroup =
            Self::resolve_threadgroup_width(Some(BRANCH_THREADS), Self::limits(&pipelines.double))?;

        let (branch_a_width, branch_b_width) = branch_capacity_widths(materialize_width);
        let branch_a_capacity = geometry.factors() * branch_a_width * geometry.bins();
        let branch_b_capacity = geometry.factors() * branch_b_width * geometry.bins();
        let dense_capacity = geometry
            .factors()
            .checked_mul(rows / materialize_width)
            .ok_or(MetalError::InputTooLong(rows))?;
        let partial_capacity = SAMPLES
            .checked_mul(e_out_capacity)
            .ok_or(MetalError::InputTooLong(e_out_capacity))?;
        let scratch = config.scratch_layout(rows, chunk_table_values)?;
        let extra_fields = e_in_storage
            .checked_add(e_out_storage)
            .and_then(|n| {
                partial_capacity
                    .checked_mul(2)
                    .and_then(|p| n.checked_add(p))
            })
            .ok_or(MetalError::InputTooLong(rows))?;
        let owned_bytes = scratch
            .owned_bytes()
            .checked_add(byte_length::<Fp128>(extra_fields)?)
            .ok_or(MetalError::InputTooLong(rows))?;
        self.validate_additional_working_set(owned_bytes)?;
        Ok(InstructionRaSequenceStorage {
            context: self.clone(),
            geometry,
            pipelines,
            reduction_limits,
            buffers: Buffers {
                branches_a: new_buffer(self, branch_a_capacity)?,
                branches_b: new_buffer(self, branch_b_capacity)?,
                dense_a: new_buffer(self, dense_capacity)?,
                dense_b: if config.reuse_inverse_for_dense {
                    None
                } else {
                    Some(new_buffer(self, dense_capacity / 2)?)
                },
                e_in: new_buffer(self, e_in_storage)?,
                e_out: new_buffer(self, e_out_storage)?,
                partial_a: new_buffer(self, partial_capacity)?,
                partial_b: new_buffer(self, partial_capacity)?,
            },
            rows,
            e_in_capacity,
            e_out_capacity,
            message_threads_per_threadgroup,
            materialize_threads_per_threadgroup,
            branch_threads_per_threadgroup,
            config,
        })
    }
}

impl InstructionRaSequenceStorage {
    pub(crate) fn matches(
        &self,
        context: &SolinasMetal,
        rows: usize,
        e_in_capacity: usize,
        e_out_capacity: usize,
        config: InstructionRaSequenceConfig,
        chunk_table_values: usize,
    ) -> bool {
        self.context.device_registry_id() == context.device_registry_id()
            && self.rows == rows
            && self.e_in_capacity == e_in_capacity
            && self.e_out_capacity == e_out_capacity
            && self.config == config
            && self.geometry.table_values() == chunk_table_values
    }

    pub(crate) fn attach(
        self,
        plane: ResidentLookupIndexPlane,
        chunk_tables: &[AkitaField],
    ) -> Result<InstructionRaSequence, MetalError> {
        if plane.len() != self.rows {
            return Err(MetalError::InvalidInstructionRaState(
                "resident lookup plane does not match the preallocated sequence",
            ));
        }
        if chunk_tables.len() != self.geometry.table_values() {
            return Err(MetalError::InstructionRaStorageLength {
                expected: self.geometry.table_values(),
                got: chunk_tables.len(),
            });
        }
        validate_plane(&self.context, &plane)?;
        write_fields(
            &self.buffers.branches_a,
            self.geometry.table_values(),
            chunk_tables,
        )?;
        if self.buffers.dense_b.is_none() {
            let required = byte_length::<Fp128>(
                self.geometry.factors() * (self.rows / self.config.materialize_width.elements())
                    / 2,
            )?;
            let inverse = plane.cycle_to_table_major();
            if inverse.length() < required {
                return Err(MetalError::InstructionRaPlaneLength {
                    name: "cycle-to-table-major dense reuse",
                    expected: required,
                    got: inverse.length(),
                });
            }
        }

        Ok(InstructionRaSequence {
            context: self.context,
            geometry: self.geometry,
            lookup_plane: Some(plane),
            pipelines: self.pipelines,
            reduction_limits: self.reduction_limits,
            buffers: self.buffers,
            rows: self.rows,
            e_in_capacity: self.e_in_capacity,
            e_out_capacity: self.e_out_capacity,
            message_threads_per_threadgroup: self.message_threads_per_threadgroup,
            materialize_threads_per_threadgroup: self.materialize_threads_per_threadgroup,
            branch_threads_per_threadgroup: self.branch_threads_per_threadgroup,
            materialize_width: self.config.materialize_width.elements(),
            reuse_inverse_for_dense: self.config.reuse_inverse_for_dense,
            branch_width: 1,
            branches_in_a: true,
            dense: false,
            dense_in_a: true,
            dense_elements: 0,
            e_in_elements: self.e_in_capacity,
            e_out_elements: self.e_out_capacity,
            weights_initialized: false,
            weight_prefixes_ready: false,
        })
    }
}

#[cfg(feature = "allocative")]
fn device_storage_bytes(buffers: &Buffers) -> usize {
    let fixed = [
        &buffers.branches_a,
        &buffers.branches_b,
        &buffers.dense_a,
        &buffers.e_in,
        &buffers.e_out,
        &buffers.partial_a,
        &buffers.partial_b,
    ]
    .into_iter()
    .fold(0usize, |bytes, buffer| {
        bytes.saturating_add(buffer.length() as usize)
    });
    buffers.dense_b.as_ref().map_or(fixed, |buffer| {
        fixed.saturating_add(buffer.length() as usize)
    })
}

impl InstructionRaSequence {
    /// Uploads the initial split Eq tables once. Their shrinking prefixes are
    /// prepared on the device with the first round and stay resident thereafter.
    pub(crate) fn initialize_weights(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<(), MetalError> {
        if self.weights_initialized {
            return Err(MetalError::InvalidInstructionRaState(
                "instruction RA weights were initialized more than once",
            ));
        }
        self.validate_weights(self.rows / 2, e_in.len(), e_out.len())?;
        if e_in.len() != self.e_in_capacity || e_out.len() != self.e_out_capacity {
            return Err(MetalError::InvalidInstructionRaState(
                "instruction RA initial weights do not match the reserved split",
            ));
        }
        write_fields(&self.buffers.e_in, self.e_in_capacity, e_in)?;
        write_fields(&self.buffers.e_out, self.e_out_capacity, e_out)?;
        self.weights_initialized = true;
        Ok(())
    }

    pub fn message(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        if !self.weights_initialized {
            self.initialize_weights(e_in, e_out)?;
        }
        self.validate_resident_weight_lengths(e_in.len(), e_out.len(), false)?;
        if self.dense {
            return Err(MetalError::InvalidInstructionRaState(
                "the lazy prefix has already materialized",
            ));
        }
        self.write_active_weights(e_in, e_out)?;
        self.message_resident()
    }

    pub fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        self.validate_resident_weight_lengths(e_in.len(), e_out.len(), true)?;
        self.write_active_weights(e_in, e_out)?;
        self.bind_and_message_resident(challenge)
    }

    // Keep the explicit-weight API available for callers supplying arbitrary
    // weights. The production kernel calls only the resident methods below.
    fn write_active_weights(
        &self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<(), MetalError> {
        let (e_in_offset, e_out_offset) = self.resident_weight_offsets(e_in.len(), e_out.len());
        write_fields_at_offset(&self.buffers.e_in, e_in.len(), e_in, e_in_offset)?;
        write_fields_at_offset(&self.buffers.e_out, e_out.len(), e_out, e_out_offset)
    }

    pub(crate) fn message_resident(&mut self) -> Result<[AkitaField; SAMPLES], MetalError> {
        self.require_weights()?;
        self.execute_lazy(None, self.e_in_elements, self.e_out_elements)
    }

    pub(crate) fn bind_and_message_resident(
        &mut self,
        challenge: AkitaField,
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        self.require_weights()?;
        let (e_in, e_out) = self.next_weight_lengths()?;
        let message = if self.dense {
            self.execute_dense(challenge, e_in, e_out)
        } else {
            self.execute_lazy(Some(challenge), e_in, e_out)
        }?;
        self.e_in_elements = e_in;
        self.e_out_elements = e_out;
        Ok(message)
    }

    fn require_weights(&self) -> Result<(), MetalError> {
        if !self.weights_initialized {
            return Err(MetalError::InvalidInstructionRaState(
                "instruction RA resident weights have not been initialized",
            ));
        }
        Ok(())
    }

    fn next_weight_lengths(&self) -> Result<(usize, usize), MetalError> {
        if self.e_in_elements > 1 {
            Ok((self.e_in_elements / 2, self.e_out_elements))
        } else if self.e_out_elements > 1 {
            Ok((1, self.e_out_elements / 2))
        } else {
            Err(MetalError::InvalidInstructionRaState(
                "instruction RA has no remaining round after this bind",
            ))
        }
    }

    fn validate_resident_weight_lengths(
        &self,
        e_in: usize,
        e_out: usize,
        bind: bool,
    ) -> Result<(), MetalError> {
        self.require_weights()?;
        let expected = if bind {
            self.next_weight_lengths()?
        } else {
            (self.e_in_elements, self.e_out_elements)
        };
        if (e_in, e_out) != expected {
            return Err(MetalError::InvalidInstructionRaState(
                "instruction RA weights do not match the resident round split",
            ));
        }
        Ok(())
    }

    fn encode_weight_prefixes(&self, encoder: &metal::ComputeCommandEncoderRef) {
        if self.weight_prefixes_ready {
            return;
        }
        encoder.set_compute_pipeline_state(&self.pipelines.weight_prefix);
        for (buffer, capacity) in [
            (&self.buffers.e_in, self.e_in_capacity),
            (&self.buffers.e_out, self.e_out_capacity),
        ] {
            encoder.set_buffer(0, Some(buffer), 0);
            let mut source_elements = capacity;
            let mut source_offset = 0;
            while source_elements > 1 {
                let destination_elements = source_elements / 2;
                let destination_offset = source_offset + source_elements;
                let params = WeightPrefixParams {
                    source_offset: source_offset as u32,
                    destination_offset: destination_offset as u32,
                    destination_elements: destination_elements as u32,
                    _reserved: 0,
                };
                set_inline_bytes(encoder, 1, &params);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: destination_elements.div_ceil(SIMD_WIDTH) as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: SIMD_WIDTH as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                source_offset = destination_offset;
                source_elements = destination_elements;
            }
        }
    }

    fn weight_offset(capacity: usize, elements: usize) -> u64 {
        // The full table starts at zero; each following half table is packed
        // immediately after its predecessor. No round overwrites its sources.
        ((2 * capacity - 2 * elements) * size_of::<Fp128>()) as u64
    }

    fn resident_weight_offsets(&self, e_in: usize, e_out: usize) -> (u64, u64) {
        (
            Self::weight_offset(self.e_in_capacity, e_in),
            Self::weight_offset(self.e_out_capacity, e_out),
        )
    }

    pub fn read_current_tables(&self, output: &mut [AkitaField]) -> Result<(), MetalError> {
        if !self.dense {
            return Err(MetalError::InvalidInstructionRaState(
                "lazy tables cannot be read as dense tables",
            ));
        }
        let elements = self.geometry.factors() * self.dense_elements;
        if output.len() != elements {
            return Err(MetalError::InstructionRaStorageLength {
                expected: elements,
                got: output.len(),
            });
        }
        // SAFETY: materialization completed synchronously before `dense` was
        // set, and the buffer contains `elements` initialized fields.
        let values = unsafe {
            slice::from_raw_parts(
                self.dense_source_buffer()?.contents().cast::<Fp128>(),
                elements,
            )
        };
        self.context
            .validate_inputs("instruction RA dense tables", values)?;
        for (output, value) in output.iter_mut().zip(values) {
            *output = value.into_jolt_field();
        }
        Ok(())
    }

    /// Binds the last cycle variable on the device and reads only the terminal
    /// claim of each factor. Dense polynomial tables never return to the host.
    pub(crate) fn bind_terminal(
        &mut self,
        challenge: AkitaField,
    ) -> Result<Vec<AkitaField>, MetalError> {
        if !self.dense || self.dense_elements != 2 || !self.weight_prefixes_ready {
            return Err(MetalError::InvalidInstructionRaState(
                "instruction RA terminal bind requires the completed two-element round",
            ));
        }
        let factor_count = self.geometry.factors() as u32;
        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.terminal_bind);
            encoder.set_buffer(0, Some(self.dense_source_buffer()?), 0);
            encoder.set_buffer(1, Some(self.dense_destination_buffer()?), 0);
            set_inline_bytes(encoder, 2, &Fp128::from_jolt_field(&challenge));
            set_inline_bytes(encoder, 3, &factor_count);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.geometry.factors().div_ceil(SIMD_WIDTH) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: SIMD_WIDTH as u64,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            command_buffer.commit();
            crate::metal::profiling::record_submission(command_buffer);
            command_buffer.wait_until_completed();
            Ok::<(), MetalError>(())
        })?;
        validate_completed_command(command_buffer)?;
        self.dense_elements = 1;
        self.dense_in_a = !self.dense_in_a;
        let mut claims = vec![AkitaField::zero(); self.geometry.factors()];
        self.read_current_tables(&mut claims)?;
        Ok(claims)
    }

    pub const fn current_elements(&self) -> usize {
        if self.dense {
            self.dense_elements
        } else {
            self.rows / self.branch_width
        }
    }

    copy_field_getters! { pub, {
        branch_width: usize,
        is_dense => dense: bool,
    }}

    fn execute_lazy(
        &mut self,
        challenge: Option<AkitaField>,
        e_in: usize,
        e_out: usize,
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        if self.dense {
            return Err(MetalError::InvalidInstructionRaState(
                "the lazy prefix has already materialized",
            ));
        }
        let next_width = if challenge.is_some() {
            self.branch_width * 2
        } else {
            self.branch_width
        };
        if next_width > self.materialize_width {
            return Err(MetalError::InvalidInstructionRaState(
                "branch width exceeds the materialization point",
            ));
        }
        let source_elements = self.rows / next_width;
        self.validate_weights(source_elements / 2, e_in, e_out)?;
        let (e_in_offset, e_out_offset) = self.resident_weight_offsets(e_in, e_out);
        let materialize = next_width == self.materialize_width;
        let message_params = MessageParams {
            e_in_length: u32::try_from(e_in).map_err(|_| MetalError::InputTooLong(e_in))?,
            e_out_length: u32::try_from(e_out).map_err(|_| MetalError::InputTooLong(e_out))?,
            geometry: [self.geometry.groups as u32, self.geometry.chunk_bits as u32],
        };
        let materialize_params = MaterializeParams {
            source_elements: u32::try_from(source_elements)
                .map_err(|_| MetalError::InputTooLong(source_elements))?,
            e_in_length: message_params.e_in_length,
            e_out_length: message_params.e_out_length,
            geometry: self.geometry.materialize_params(),
        };
        let message_pipeline = if materialize {
            None
        } else {
            Some(self.message_pipeline(next_width)?.clone())
        };
        let plane = self
            .lookup_plane
            .as_ref()
            .ok_or(MetalError::InvalidInstructionRaState(
                "the resident lookup plane is missing",
            ))?;

        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            self.encode_weight_prefixes(encoder);
            let mut message_branches_in_a = self.branches_in_a;
            if let Some(challenge) = challenge {
                let params = BranchParams {
                    branch_width: self.branch_width as u32,
                    factor_count: self.geometry.factors() as u32,
                    bins: self.geometry.bins() as u32,
                    _reserved: 0,
                };
                encoder.set_compute_pipeline_state(&self.pipelines.double);
                encoder.set_buffer(0, Some(self.branch_source_buffer()), 0);
                encoder.set_buffer(1, Some(self.branch_destination_buffer()), 0);
                set_inline_bytes(encoder, 2, &Fp128::from_jolt_field(&challenge));
                set_inline_bytes(encoder, 3, &params);
                let elements = self.geometry.factors() * self.branch_width * self.geometry.bins();
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: elements.div_ceil(self.branch_threads_per_threadgroup) as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: self.branch_threads_per_threadgroup as u64,
                        height: 1,
                        depth: 1,
                    },
                );
                message_branches_in_a = !message_branches_in_a;
            }

            if materialize {
                encoder.set_compute_pipeline_state(&self.pipelines.materialize);
                encoder.set_buffer(0, Some(plane.lookups()), 0);
                encoder.set_buffer(1, Some(plane.cycle_to_table_major()), 0);
                encoder.set_buffer(2, Some(self.branch_buffer(message_branches_in_a)), 0);
                encoder.set_buffer(3, Some(&self.buffers.dense_a), 0);
                encoder.set_buffer(4, Some(&self.buffers.e_in), e_in_offset);
                encoder.set_buffer(5, Some(&self.buffers.e_out), e_out_offset);
                encoder.set_buffer(6, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 7, &materialize_params);
                Self::encode_message_dispatch(
                    encoder,
                    e_out,
                    self.materialize_threads_per_threadgroup,
                );
            } else {
                encoder.set_compute_pipeline_state(message_pipeline.as_ref().ok_or(
                    MetalError::InvalidInstructionRaState("the lazy message pipeline is missing"),
                )?);
                encoder.set_buffer(0, Some(plane.lookups()), 0);
                encoder.set_buffer(1, Some(plane.cycle_to_table_major()), 0);
                encoder.set_buffer(2, Some(self.branch_buffer(message_branches_in_a)), 0);
                encoder.set_buffer(3, Some(&self.buffers.e_in), e_in_offset);
                encoder.set_buffer(4, Some(&self.buffers.e_out), e_out_offset);
                encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 6, &message_params);
                Self::encode_message_dispatch(encoder, e_out, self.message_threads_per_threadgroup);
            }
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduce,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out,
                SAMPLES,
                self.reduction_limits.thread_execution_width,
            )?;
            encoder.end_encoding();
            command_buffer.commit();
            crate::metal::profiling::record_submission(command_buffer);
            command_buffer.wait_until_completed();
            Ok::<bool, MetalError>(final_in_a)
        })?;

        let message = self.finish_command(command_buffer, final_in_a)?;
        self.weight_prefixes_ready = true;
        if challenge.is_some() {
            self.branch_width = next_width;
            self.branches_in_a = !self.branches_in_a;
        }
        if materialize {
            if self.reuse_inverse_for_dense {
                let inverse = self
                    .lookup_plane
                    .as_ref()
                    .ok_or(MetalError::InvalidInstructionRaState(
                        "the resident inverse buffer is missing at materialization",
                    ))?
                    .cycle_to_table_major()
                    .clone();
                self.buffers.dense_b = Some(inverse);
            }
            self.dense = true;
            self.dense_in_a = true;
            self.dense_elements = source_elements;
            // `finish_command` observed completion before the inverse buffer was
            // repurposed and the lookup-plane ownership was released.
            let _ = self.lookup_plane.take();
        }
        Ok(message)
    }

    fn execute_dense(
        &mut self,
        challenge: AkitaField,
        e_in: usize,
        e_out: usize,
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        if !self.dense || self.dense_elements < 4 {
            return Err(MetalError::InvalidInstructionRaState(
                "dense transition needs at least four elements per factor",
            ));
        }
        self.validate_weights(self.dense_elements / 4, e_in, e_out)?;
        let (e_in_offset, e_out_offset) = self.resident_weight_offsets(e_in, e_out);
        let params = MaterializeParams {
            source_elements: u32::try_from(self.dense_elements)
                .map_err(|_| MetalError::InputTooLong(self.dense_elements))?,
            e_in_length: u32::try_from(e_in).map_err(|_| MetalError::InputTooLong(e_in))?,
            e_out_length: u32::try_from(e_out).map_err(|_| MetalError::InputTooLong(e_out))?,
            geometry: self.geometry.materialize_params(),
        };

        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let final_in_a = autoreleasepool(|| {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.dense_transition);
            encoder.set_buffer(0, Some(self.dense_source_buffer()?), 0);
            encoder.set_buffer(1, Some(self.dense_destination_buffer()?), 0);
            encoder.set_buffer(2, Some(&self.buffers.e_in), e_in_offset);
            encoder.set_buffer(3, Some(&self.buffers.e_out), e_out_offset);
            encoder.set_buffer(4, Some(&self.buffers.partial_a), 0);
            set_inline_bytes(encoder, 5, &Fp128::from_jolt_field(&challenge));
            set_inline_bytes(encoder, 6, &params);
            Self::encode_message_dispatch(encoder, e_out, self.message_threads_per_threadgroup);
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduce,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out,
                SAMPLES,
                self.reduction_limits.thread_execution_width,
            )?;
            encoder.end_encoding();
            command_buffer.commit();
            crate::metal::profiling::record_submission(command_buffer);
            command_buffer.wait_until_completed();
            Ok::<bool, MetalError>(final_in_a)
        })?;

        let message = self.finish_command(command_buffer, final_in_a)?;
        self.dense_elements /= 2;
        self.dense_in_a = !self.dense_in_a;
        Ok(message)
    }

    fn validate_weights(
        &self,
        expected: usize,
        e_in: usize,
        e_out: usize,
    ) -> Result<(), MetalError> {
        let covered = e_in
            .checked_mul(e_out)
            .ok_or(MetalError::InputTooLong(expected))?;
        if e_in == 0
            || e_out == 0
            || !e_in.is_power_of_two()
            || !e_out.is_power_of_two()
            || e_in > self.e_in_capacity
            || e_out > self.e_out_capacity
            || covered != expected
        {
            return Err(MetalError::InstructionRaWeightShape { expected, covered });
        }
        Ok(())
    }

    fn message_pipeline(&self, width: usize) -> Result<&ComputePipelineState, MetalError> {
        match width {
            1 => Ok(&self.pipelines.width_1),
            2 => Ok(&self.pipelines.width_2),
            4 => Ok(&self.pipelines.width_4),
            8 => Ok(&self.pipelines.width_8),
            _ => self
                .pipelines
                .wide_messages
                .iter()
                .find_map(|(pipeline_width, pipeline)| {
                    (*pipeline_width == width).then_some(pipeline)
                })
                .ok_or(MetalError::InvalidInstructionRaState(
                    "no lazy message pipeline for this branch width",
                )),
        }
    }

    fn encode_message_dispatch(
        encoder: &metal::ComputeCommandEncoderRef,
        groups: usize,
        threads_per_threadgroup: usize,
    ) {
        let simdgroups = threads_per_threadgroup / SIMD_WIDTH;
        encoder
            .set_threadgroup_memory_length(0, (SAMPLES * simdgroups * size_of::<Fp128>()) as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: groups as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_threadgroup as u64,
                height: 1,
                depth: 1,
            },
        );
    }

    fn finish_command(
        &mut self,
        command_buffer: &metal::CommandBufferRef,
        final_in_a: bool,
    ) -> Result<[AkitaField; SAMPLES], MetalError> {
        validate_completed_command(command_buffer)?;
        let buffer = if final_in_a {
            &self.buffers.partial_a
        } else {
            &self.buffers.partial_b
        };
        // SAFETY: the completed reduction leaves four fields at the front of
        // the selected shared buffer.
        let values = unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), SAMPLES) };
        self.context
            .validate_inputs("instruction RA lazy message", values)?;
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }

    fn branch_source_buffer(&self) -> &Buffer {
        self.branch_buffer(self.branches_in_a)
    }

    fn branch_destination_buffer(&self) -> &Buffer {
        self.branch_buffer(!self.branches_in_a)
    }

    fn branch_buffer(&self, in_a: bool) -> &Buffer {
        if in_a {
            &self.buffers.branches_a
        } else {
            &self.buffers.branches_b
        }
    }

    fn dense_source_buffer(&self) -> Result<&Buffer, MetalError> {
        if self.dense_in_a {
            Ok(&self.buffers.dense_a)
        } else {
            self.dense_b_buffer()
        }
    }

    fn dense_destination_buffer(&self) -> Result<&Buffer, MetalError> {
        if self.dense_in_a {
            self.dense_b_buffer()
        } else {
            Ok(&self.buffers.dense_a)
        }
    }

    fn dense_b_buffer(&self) -> Result<&Buffer, MetalError> {
        self.buffers
            .dense_b
            .as_ref()
            .ok_or(MetalError::InvalidInstructionRaState(
                "the dense destination buffer is missing",
            ))
    }
}

fn branch_capacity_widths(materialize_width: usize) -> (usize, usize) {
    if materialize_width == 16 {
        return (16, 16);
    }
    let materializes_in_a = materialize_width.trailing_zeros().is_multiple_of(2);
    if materializes_in_a {
        (materialize_width, materialize_width / 2)
    } else {
        (materialize_width / 2, materialize_width)
    }
}

fn validate_plane(
    context: &SolinasMetal,
    plane: &ResidentLookupIndexPlane,
) -> Result<(), MetalError> {
    let expected_lookups = byte_length::<[u64; 2]>(plane.len())?;
    let expected_inverse = byte_length::<u32>(plane.len())?;
    if plane.lookups().length() != expected_lookups {
        return Err(MetalError::InstructionRaPlaneLength {
            name: "lookup-index",
            expected: expected_lookups,
            got: plane.lookups().length(),
        });
    }
    if plane.cycle_to_table_major().length() != expected_inverse {
        return Err(MetalError::InstructionRaPlaneLength {
            name: "cycle-to-table-major",
            expected: expected_inverse,
            got: plane.cycle_to_table_major().length(),
        });
    }
    let expected = context.device.registry_id();
    let got = plane.device_registry_id();
    if got != expected {
        return Err(MetalError::InstructionRaPlaneDevice { expected, got });
    }
    Ok(())
}

fn new_buffer(context: &SolinasMetal, elements: usize) -> Result<Buffer, MetalError> {
    let bytes = byte_length::<Fp128>(elements)?;
    context.validate_buffer_length(bytes)?;
    Ok(context
        .device
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared))
}

fn write_fields(buffer: &Buffer, capacity: usize, values: &[AkitaField]) -> Result<(), MetalError> {
    write_fields_at_offset(buffer, capacity, values, 0)
}

fn write_fields_at_offset(
    buffer: &Buffer,
    capacity: usize,
    values: &[AkitaField],
    offset: u64,
) -> Result<(), MetalError> {
    if values.len() > capacity {
        return Err(MetalError::InstructionRaStorageLength {
            expected: capacity,
            got: values.len(),
        });
    }
    let end = offset
        .checked_add(byte_length::<Fp128>(capacity)?)
        .ok_or(MetalError::InputTooLong(capacity))?;
    if end > buffer.length() {
        return Err(MetalError::InvalidInstructionRaState(
            "instruction RA weight prefix exceeds its buffer",
        ));
    }
    // SAFETY: the checked buffer range contains `capacity` fields and no
    // command is using it while the host writes the selected prefix.
    let output = unsafe {
        slice::from_raw_parts_mut(
            buffer.contents().cast::<Fp128>().byte_add(offset as usize),
            capacity,
        )
    };
    for (output, value) in output.iter_mut().zip(values) {
        *output = Fp128::from_jolt_field(value);
    }
    Ok(())
}

fn byte_length<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

fn weight_storage_elements(capacity: usize) -> Result<usize, MetalError> {
    capacity
        .checked_mul(2)
        .and_then(|elements| elements.checked_sub(1))
        .filter(|elements| *elements <= u32::MAX as usize)
        .ok_or(MetalError::InputTooLong(capacity))
}

const _: () = assert!(size_of::<MessageParams>() == 16);
const _: () = assert!(size_of::<BranchParams>() == 16);
const _: () = assert!(size_of::<MaterializeParams>() == 16);
const _: () = assert!(size_of::<WeightPrefixParams>() == 16);

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test assertions")]
#[expect(clippy::unwrap_used, reason = "GPU geometry fixtures")]
mod tests {
    use jolt_field::Prime128OffsetA7F7 as AkitaField;
    use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

    use super::*;

    #[test]
    fn weight_capacities_match_the_low_to_high_gruen_split() {
        for log_t in 5..=30 {
            let gruen = GruenSplitEqPolynomial::<AkitaField>::new(
                &vec![AkitaField::zero(); log_t],
                BindingOrder::LowToHigh,
            );
            assert_eq!(
                instruction_ra_weight_capacities(1 << log_t).ok(),
                Some((gruen.e_in_current_len(), gruen.e_out_current_len()))
            );
        }
        assert!(matches!(
            instruction_ra_weight_capacities(31),
            Err(MetalError::InvalidInstructionRaRows(31))
        ));
    }

    #[test]
    fn t26_scratch_layout_tracks_materialization_width() {
        let cases = [
            (InstructionRaMaterializeWidth::W16, 1538),
            (InstructionRaMaterializeWidth::W32, 771),
            (InstructionRaMaterializeWidth::W64, 390),
            (InstructionRaMaterializeWidth::W128, 204),
            (InstructionRaMaterializeWidth::W256, 120),
            (InstructionRaMaterializeWidth::W512, 96),
        ];
        for (materialize_width, expected_mib) in cases {
            let config = InstructionRaSequenceConfig {
                materialize_width,
                ..InstructionRaSequenceConfig::default()
            };
            let layout = config
                .scratch_layout(1 << 26, G4_TABLE_VALUES)
                .expect("the T26 layout should fit");
            assert_eq!(layout.owned_bytes() >> 20, expected_mib);
        }
    }

    #[test]
    fn inverse_reuse_removes_only_the_owned_dense_destination() {
        let config = InstructionRaSequenceConfig {
            materialize_width: InstructionRaMaterializeWidth::W256,
            reuse_inverse_for_dense: true,
            ..InstructionRaSequenceConfig::default()
        };
        let layout = config
            .scratch_layout(1 << 26, G4_TABLE_VALUES)
            .expect("the T26 layout should fit");
        assert_eq!(layout.dense_b_active_bytes >> 20, 32);
        assert_eq!(layout.dense_b_owned_bytes, 0);
        assert_eq!(layout.dense_b_physical_bytes >> 20, 256);
        assert_eq!(layout.owned_bytes() >> 20, 88);
        assert_eq!(layout.resident_bytes_after_handoff() >> 20, 344);
    }

    #[test]
    fn g8_t24_w64_bounds_dense_scratch_and_inverse_reuse() {
        let config = InstructionRaSequenceConfig {
            materialize_width: InstructionRaMaterializeWidth::W64,
            ..Default::default()
        };
        let layout = config.scratch_layout(1 << 24, G8_TABLE_VALUES).unwrap();
        assert_eq!(layout.dense_a_bytes, 128 << 20);
        assert_eq!(layout.dense_b_owned_bytes, 64 << 20);
        assert_eq!(layout.branch_a_bytes + layout.branch_b_bytes, 768 << 10);
        assert_eq!(layout.owned_bytes(), (192 << 20) + (768 << 10));
        let reused = InstructionRaSequenceConfig {
            reuse_inverse_for_dense: true,
            ..config
        }
        .scratch_layout(1 << 24, G8_TABLE_VALUES)
        .unwrap();
        assert_eq!(reused.dense_b_owned_bytes, 0);
        assert_eq!(reused.dense_b_physical_bytes, 64 << 20);
        assert!(InstructionRaSequenceConfig {
            materialize_width: InstructionRaMaterializeWidth::W32,
            reuse_inverse_for_dense: true,
            ..Default::default()
        }
        .scratch_layout(1 << 24, G8_TABLE_VALUES)
        .is_err());
        assert!(config.scratch_layout(1 << 24, 123).is_err());
    }

    #[test]
    fn preallocated_storage_does_not_match_a_different_geometry() {
        let rows = 1 << 8;
        let context = SolinasMetal::for_akita().unwrap();
        let (e_in, e_out) = instruction_ra_weight_capacities(rows).unwrap();
        let config = InstructionRaSequenceConfig {
            materialize_width: InstructionRaMaterializeWidth::W64,
            ..Default::default()
        };
        let storage = context
            .prepare_instruction_ra_sequence_storage(rows, e_in, e_out, config, G4_TABLE_VALUES)
            .unwrap();
        assert!(storage.matches(&context, rows, e_in, e_out, config, G4_TABLE_VALUES));
        assert!(!storage.matches(&context, rows, e_in, e_out, config, G8_TABLE_VALUES));
    }

    #[test]
    fn width_16_rejects_inverse_reuse() {
        let config = InstructionRaSequenceConfig {
            reuse_inverse_for_dense: true,
            ..InstructionRaSequenceConfig::default()
        };
        assert!(matches!(
            config.scratch_layout(1 << 26, G4_TABLE_VALUES),
            Err(MetalError::InvalidInstructionRaState(_))
        ));
    }

    #[test]
    fn wide_materialization_pipelines_compile() {
        let rows = 1 << 10;
        let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
        let (e_in, e_out) =
            instruction_ra_weight_capacities(rows).expect("weight capacities should derive");
        let _storage = context
            .prepare_instruction_ra_sequence_storage(
                rows,
                e_in,
                e_out,
                InstructionRaSequenceConfig {
                    materialize_width: InstructionRaMaterializeWidth::W32,
                    reuse_inverse_for_dense: true,
                    ..InstructionRaSequenceConfig::default()
                },
                G4_TABLE_VALUES,
            )
            .expect("wide Instruction RA pipelines should compile");
    }

    #[test]
    fn g8_t22_w64_scratch_and_resident_weight_cache_are_bounded() {
        let config = InstructionRaSequenceConfig {
            materialize_width: InstructionRaMaterializeWidth::W64,
            ..Default::default()
        };
        let layout = config.scratch_layout(1 << 22, G8_TABLE_VALUES).unwrap();
        assert_eq!(layout.owned_bytes(), (48 << 20) + (768 << 10));
        let (e_in, e_out) = instruction_ra_weight_capacities(1 << 22).unwrap();
        let cache_values =
            weight_storage_elements(e_in).unwrap() + weight_storage_elements(e_out).unwrap();
        assert_eq!(byte_length::<Fp128>(cache_values).unwrap(), (96 << 10) - 32);
        assert!(weight_storage_elements(0).is_err());
        assert!(weight_storage_elements(usize::MAX).is_err());
    }

    #[test]
    fn resident_weight_prefixes_match_gruen_across_every_round() {
        use jolt_field::{One as _, Ring as _};

        let context = SolinasMetal::for_akita().unwrap();
        let log_t = 9;
        let rows = 1 << log_t;
        let point: Vec<_> = (0..log_t)
            .map(|index| AkitaField::from_u128(u128::MAX - (1 << 32) + 22000 - index as u128))
            .collect();
        let mut gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
        let mut sequence = context
            .prepare_instruction_ra_test_sequence(
                &vec![u128::MAX; rows],
                &vec![AkitaField::one(); G8_TABLE_VALUES],
                gruen.e_in_current_len(),
                gruen.e_out_current_len(),
                InstructionRaSequenceConfig {
                    materialize_width: InstructionRaMaterializeWidth::W64,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(sequence.message_resident().is_err());
        assert!(sequence
            .bind_and_message_resident(AkitaField::one())
            .is_err());
        assert!(sequence.bind_terminal(AkitaField::one()).is_err());
        assert_eq!(sequence.branch_width(), 1);
        let _ = sequence
            .message(gruen.e_in_current(), gruen.e_out_current())
            .unwrap();
        assert!(sequence
            .initialize_weights(gruen.e_in_current(), gruen.e_out_current())
            .is_err());

        for round in 0..log_t {
            if round > 0 {
                gruen.bind(AkitaField::from_u64(31 + round as u64));
                let before = sequence.current_elements();
                assert!(sequence
                    .bind_and_message(AkitaField::one(), &[], gruen.e_out_current())
                    .is_err());
                assert_eq!(sequence.current_elements(), before);
                let _ = sequence
                    .bind_and_message(
                        AkitaField::from_u64(31 + round as u64),
                        gruen.e_in_current(),
                        gruen.e_out_current(),
                    )
                    .unwrap();
            }
            for (buffer, capacity, expected) in [
                (
                    &sequence.buffers.e_in,
                    sequence.e_in_capacity,
                    gruen.e_in_current(),
                ),
                (
                    &sequence.buffers.e_out,
                    sequence.e_out_capacity,
                    gruen.e_out_current(),
                ),
            ] {
                let offset =
                    InstructionRaSequence::weight_offset(capacity, expected.len()) as usize;
                // SAFETY: the synchronous round completed prefix preparation,
                // and the checked cache layout includes this active Eq table.
                let cached = unsafe {
                    slice::from_raw_parts(
                        buffer.contents().cast::<Fp128>().byte_add(offset),
                        expected.len(),
                    )
                };
                let actual: Vec<AkitaField> =
                    cached.iter().map(|value| value.into_jolt_field()).collect();
                assert_eq!(actual, expected, "round {round}");
            }
        }
        assert_eq!(sequence.current_elements(), 2);
        assert_eq!(
            sequence.bind_terminal(AkitaField::one()).unwrap(),
            vec![AkitaField::one(); 32]
        );
        assert!(sequence.bind_terminal(AkitaField::one()).is_err());
    }
}
