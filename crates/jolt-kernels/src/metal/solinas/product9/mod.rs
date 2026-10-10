//! Unregistered draft: G8 Instruction Read-RAF cycle sums from resident address
//! carriers. The original Val and eight virtual RA bases stay implicit until
//! six LowToHigh binds; only T/64 factor entries are then materialized.

use std::{mem::size_of, slice};

use jolt_field::{One as _, Prime128OffsetA7F7 as AkitaField, Zero as _};
use metal::{
    objc::rc::autoreleasepool, Buffer, CompileOptions, ComputePipelineState, MTLResourceOptions,
    MTLSize, NSRange,
};
use thiserror::Error;

use super::{
    buffer_from_slice, encode_column_reductions, set_inline_bytes, validate_completed_command,
    Fp128, MetalError, PipelineLimits, ResidentLookupIndexPlane, SolinasMetal,
    AKITA_OFFSET_FFFFA7F7,
};

pub(crate) const FACTORS: usize = 9;
const WIDTH: usize = 64;
const THREADS: usize = 128;
const MAX_ROWS: usize = 1 << 24;
const PHASES: usize = 16;
const BINS: usize = 256;
const LAZY: &str = "solinas_product9_lazy_cycle";
const DENSE: &str = "solinas_product9_dense_bind";
const REDUCE: &str = "solinas_product9_reduce";

#[derive(Debug, Error)]
pub(crate) enum Product9Error {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("invalid Product9 state: {0}")]
    State(&'static str),
    #[error("invalid Product9 source rows: {0}")]
    Source(u32),
}

/// Owned aliases of the immutable address-phase source. Packed claims and
/// lookups are table-major; the inverse maps original cycle order to that order.
/// Cloning these buffers retains their allocations, not a copy of their rows.
pub(crate) struct Product9ResidentSource {
    claims: Buffer,
    plane: ResidentLookupIndexPlane,
}

impl Product9ResidentSource {
    pub(super) fn new(
        context: &SolinasMetal,
        claims: Buffer,
        plane: ResidentLookupIndexPlane,
    ) -> Result<Self, Product9Error> {
        let rows = plane.len();
        if !(128..=MAX_ROWS).contains(&rows)
            || !rows.is_power_of_two()
            || plane.device_registry_id() != context.device_registry_id()
            || claims.device().registry_id() != context.device_registry_id()
            || plane.lookups().device().registry_id() != context.device_registry_id()
            || plane.cycle_to_table_major().device().registry_id() != context.device_registry_id()
            || claims.length() != rows as u64
            || plane.lookups().length() != bytes::<[u64; 2]>(rows)?
            || plane.cycle_to_table_major().length() != bytes::<u32>(rows)?
        {
            return Err(Product9Error::State(
                "resident address carrier shape/device mismatch",
            ));
        }
        Ok(Self { claims, plane })
    }

    pub(crate) fn rows(&self) -> usize {
        self.plane.len()
    }

    #[cfg(test)]
    pub(crate) fn fixture(
        context: &SolinasMetal,
        lookups: &[u128],
        claims: &[u8],
    ) -> Result<Self, Product9Error> {
        if lookups.len() != claims.len() {
            return Err(Product9Error::State("fixture rows differ"));
        }
        let rows = lookups.len();
        if rows < 128 || !rows.is_power_of_two() || rows > MAX_ROWS {
            return Err(Product9Error::State("fixture shape invalid"));
        }
        let mut stored = vec![[0u64; 2]; rows];
        let mut packed = vec![0u8; rows];
        let mut inverse = vec![0u32; rows];
        for (cycle, lookup) in lookups.iter().enumerate() {
            let row = cycle * 17 % rows;
            stored[row] = [*lookup as u64, (*lookup >> 64) as u64];
            packed[row] = claims[cycle];
            inverse[cycle] = row as u32;
        }
        Self::new(
            context,
            buffer_from_slice(&context.device, &packed),
            ResidentLookupIndexPlane::from_buffers(
                buffer_from_slice(&context.device, &stored),
                buffer_from_slice(&context.device, &inverse),
                rows,
                context.device_registry_id(),
            ),
        )
    }
}

/// Metadata already computed by the CPU at the address/cycle boundary. The RAF
/// identity scalar includes the existing gamma^3 canonical-address contribution.
pub(crate) struct Product9CycleInput {
    pub(crate) phases: Vec<Vec<AkitaField>>,
    pub(crate) table_values: Vec<AkitaField>,
    pub(crate) raf_interleaved: AkitaField,
    pub(crate) raf_identity: AkitaField,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    source_elements: u32,
    branch_width: u32,
    e_in_length: u32,
    e_out_length: u32,
    table_count: u32,
    materialize: u32,
    reserved: u32,
}
const _: () = assert!(size_of::<Params>() == 32);

struct Pipelines {
    lazy: ComputePipelineState,
    dense: ComputePipelineState,
    reduce: ComputePipelineState,
    reduce_limits: PipelineLimits,
}

struct Buffers {
    phases: Buffer,
    values: Buffer,
    weights: Buffer,
    dense_a: Buffer,
    dense_b: Buffer,
    e_in: Buffer,
    e_out: Buffer,
    partial_a: Buffer,
    partial_b: Buffer,
    status: Buffer,
}

pub(crate) struct Product9Sequence {
    context: SolinasMetal,
    source: Option<Product9ResidentSource>,
    buffers: Buffers,
    pipelines: Pipelines,
    rows: usize,
    elements: usize,
    dense: bool,
    source_in_a: bool,
    branch_weights: Vec<AkitaField>,
    raf_interleaved: AkitaField,
    raf_identity: AkitaField,
    table_count: usize,
    e_in_capacity: usize,
    e_out_capacity: usize,
    owned_bytes: u64,
}

fn bytes<T>(elements: usize) -> Result<u64, MetalError> {
    elements
        .checked_mul(size_of::<T>())
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(MetalError::InputTooLong(elements))
}

#[cfg(test)]
pub(crate) fn dense_scratch_bytes(rows: usize) -> Result<u64, Product9Error> {
    if !(128..=MAX_ROWS).contains(&rows) || !rows.is_power_of_two() {
        return Err(Product9Error::State("rows outside bounded G8 domain"));
    }
    let fields = FACTORS
        .checked_mul(rows / WIDTH)
        .ok_or(MetalError::InputTooLong(rows))?;
    Ok(bytes::<Fp128>(fields + fields / 2)?)
}

fn pipelines(context: &SolinasMetal) -> Result<Pipelines, Product9Error> {
    // Distinct kernel names and a private library preserve Product5/RAM/bytecode
    // source and pipeline ABIs. Cached pipelines are retained across warm proofs.
    let mut cache = context
        .pipeline_cache
        .lock()
        .map_err(|_| MetalError::PipelineCachePoisoned)?;
    if [LAZY, DENSE, REDUCE]
        .iter()
        .any(|name| !cache.contains_key(&(*name, None)))
    {
        let source = format!(
            "#define SOLINAS_OFFSET {}u\n{}\n{}\n{}",
            context.offset,
            include_str!("../fp128.metal"),
            include_str!("../simd_reduce.metal"),
            include_str!("shader.metal")
        );
        let library = context
            .device
            .new_library_with_source(&source, &CompileOptions::new())
            .map_err(MetalError::LibraryCompilation)?;
        for name in [LAZY, DENSE, REDUCE] {
            let function = library
                .get_function(name, None)
                .map_err(|message| MetalError::FunctionLookup { name, message })?;
            let pipeline = context
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|message| MetalError::PipelineCompilation { name, message })?;
            if SolinasMetal::limits(&pipeline).thread_execution_width != 32 {
                return Err(Product9Error::State("Product9 requires 32-lane simdgroups"));
            }
            drop(cache.insert((name, None), pipeline));
        }
    }
    let get = |name| {
        cache
            .get(&(name, None))
            .cloned()
            .ok_or(Product9Error::State("Product9 pipeline missing"))
    };
    let reduce = get(REDUCE)?;
    Ok(Pipelines {
        lazy: get(LAZY)?,
        dense: get(DENSE)?,
        reduce_limits: SolinasMetal::limits(&reduce),
        reduce,
    })
}

impl Product9Sequence {
    pub(crate) fn new(
        context: SolinasMetal,
        source: Product9ResidentSource,
        input: Product9CycleInput,
        e_in_capacity: usize,
        e_out_capacity: usize,
    ) -> Result<Self, Product9Error> {
        let rows = source.rows();
        if context.offset != AKITA_OFFSET_FFFFA7F7
            || source.plane.device_registry_id() != context.device_registry_id()
            || input.phases.len() != PHASES
            || input.phases.iter().any(|phase| phase.len() != BINS)
            || input.table_values.len() != super::instruction_read_raf::INSTRUCTION_READ_RAF_TABLES
            || e_in_capacity == 0
            || e_out_capacity == 0
            || e_in_capacity.checked_mul(e_out_capacity) != Some(rows / 2)
        {
            return Err(Product9Error::State("G8 cycle metadata/weights mismatch"));
        }
        let dense_fields = FACTORS * (rows / WIDTH);
        let sizes = [
            bytes::<Fp128>(PHASES * BINS)?,
            bytes::<Fp128>(input.table_values.len())?,
            bytes::<Fp128>(WIDTH)?,
            bytes::<Fp128>(dense_fields)?,
            bytes::<Fp128>(dense_fields / 2)?,
            bytes::<Fp128>(e_in_capacity)?,
            bytes::<Fp128>(e_out_capacity)?,
            bytes::<Fp128>(FACTORS * e_out_capacity)?,
            bytes::<Fp128>(FACTORS * e_out_capacity)?,
            4,
        ];
        let owned_bytes = sizes
            .iter()
            .try_fold(0u64, |sum, size| sum.checked_add(*size))
            .ok_or(MetalError::InputTooLong(rows))?;
        context.validate_additional_working_set(owned_bytes)?;
        for size in sizes {
            context.validate_buffer_length(size)?;
        }
        let pipelines = pipelines(&context)?;
        for pipeline in [&pipelines.lazy, &pipelines.dense] {
            let limits = SolinasMetal::limits(pipeline);
            let _ = SolinasMetal::resolve_threadgroup_width(Some(THREADS), limits)?;
            if limits.static_threadgroup_memory_length + (FACTORS * (THREADS / 32) * 16) as u64
                > context.device.max_threadgroup_memory_length()
            {
                return Err(Product9Error::State("Product9 exceeds threadgroup memory"));
            }
        }
        let phases: Vec<_> = input
            .phases
            .iter()
            .flatten()
            .map(Fp128::from_jolt_field)
            .collect();
        let values: Vec<_> = input
            .table_values
            .iter()
            .map(Fp128::from_jolt_field)
            .collect();
        context.validate_inputs("Product9 phase tables", &phases)?;
        context.validate_inputs("Product9 table values", &values)?;
        let allocate = |size| {
            context
                .device
                .new_buffer(size, MTLResourceOptions::StorageModeShared)
        };
        let buffers = Buffers {
            phases: buffer_from_slice(&context.device, &phases),
            values: buffer_from_slice(&context.device, &values),
            weights: allocate(sizes[2]),
            dense_a: allocate(sizes[3]),
            dense_b: allocate(sizes[4]),
            e_in: allocate(sizes[5]),
            e_out: allocate(sizes[6]),
            partial_a: allocate(sizes[7]),
            partial_b: allocate(sizes[8]),
            status: allocate(sizes[9]),
        };
        Ok(Self {
            context,
            source: Some(source),
            buffers,
            pipelines,
            rows,
            elements: rows,
            dense: false,
            source_in_a: true,
            branch_weights: vec![AkitaField::one()],
            raf_interleaved: input.raf_interleaved,
            raf_identity: input.raf_identity,
            table_count: input.table_values.len(),
            e_in_capacity,
            e_out_capacity,
            owned_bytes,
        })
    }

    pub(crate) fn current_elements(&self) -> usize {
        self.elements
    }
    pub(crate) fn is_dense(&self) -> bool {
        self.dense
    }
    pub(crate) fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    pub(crate) fn message(
        &mut self,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; FACTORS], Product9Error> {
        if self.dense || self.branch_weights.len() != 1 {
            return Err(Product9Error::State(
                "initial message requested after binding",
            ));
        }
        self.execute(None, e_in, e_out)
    }

    pub(crate) fn bind_and_message(
        &mut self,
        challenge: AkitaField,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; FACTORS], Product9Error> {
        self.execute(Some(challenge), e_in, e_out)
    }

    pub(crate) fn read_current_factor_tables(
        &self,
        output: &mut [Vec<AkitaField>; FACTORS],
    ) -> Result<(), Product9Error> {
        if !self.dense || output.iter().any(|table| table.len() < self.elements) {
            return Err(Product9Error::State("dense CPU tail capacity mismatch"));
        }
        let source = if self.source_in_a {
            &self.buffers.dense_a
        } else {
            &self.buffers.dense_b
        };
        // SAFETY: every dense producing command completed and wrote exactly
        // FACTORS * elements canonical fields in this retained shared buffer.
        let values = unsafe {
            slice::from_raw_parts(source.contents().cast::<Fp128>(), FACTORS * self.elements)
        };
        self.context.validate_inputs("Product9 CPU tail", values)?;
        for (factor, table) in output.iter_mut().enumerate() {
            for (destination, value) in table
                .iter_mut()
                .zip(&values[factor * self.elements..(factor + 1) * self.elements])
            {
                *destination = value.into_jolt_field();
            }
            table.truncate(self.elements);
        }
        Ok(())
    }

    fn execute(
        &mut self,
        challenge: Option<AkitaField>,
        e_in: &[AkitaField],
        e_out: &[AkitaField],
    ) -> Result<[AkitaField; FACTORS], Product9Error> {
        let next_elements = if challenge.is_some() {
            self.elements / 2
        } else {
            self.elements
        };
        if next_elements < 2
            || (self.dense && challenge.is_none())
            || e_in.is_empty()
            || e_out.is_empty()
            || e_in.len() > self.e_in_capacity
            || e_out.len() > self.e_out_capacity
            || e_in.len().checked_mul(e_out.len()) != Some(next_elements / 2)
        {
            return Err(Product9Error::State("cycle message weights/state mismatch"));
        }
        let mut next_weights = self.branch_weights.clone();
        if !self.dense {
            if let Some(bind) = challenge {
                let width = next_weights.len();
                next_weights.resize(2 * width, AkitaField::zero());
                for offset in 0..width {
                    let weight = next_weights[offset];
                    next_weights[offset] = weight * (AkitaField::one() - bind);
                    next_weights[width + offset] = weight * bind;
                }
            }
            if next_weights.len() > WIDTH {
                return Err(Product9Error::State("lazy width exceeded"));
            }
            write_fields(&self.buffers.weights, &next_weights)?;
        }
        write_fields(&self.buffers.e_in, e_in)?;
        write_fields(&self.buffers.e_out, e_out)?;
        let materialize = !self.dense && next_weights.len() == WIDTH;
        let params = Params {
            rows: self.rows as u32,
            source_elements: if self.dense {
                self.elements
            } else {
                next_elements
            } as u32,
            branch_width: next_weights.len() as u32,
            e_in_length: e_in.len() as u32,
            e_out_length: e_out.len() as u32,
            table_count: self.table_count as u32,
            materialize: u32::from(materialize),
            reserved: 0,
        };
        let command = autoreleasepool(|| self.context.queue.new_command_buffer().to_owned());
        let final_in_a = autoreleasepool(|| {
            let blit = command.new_blit_command_encoder();
            blit.fill_buffer(&self.buffers.status, NSRange::new(0, 4), 0);
            blit.end_encoding();
            let encoder = command.new_compute_command_encoder();
            if self.dense {
                let (source, destination) = if self.source_in_a {
                    (&self.buffers.dense_a, &self.buffers.dense_b)
                } else {
                    (&self.buffers.dense_b, &self.buffers.dense_a)
                };
                encoder.set_compute_pipeline_state(&self.pipelines.dense);
                encoder.set_buffer(0, Some(source), 0);
                encoder.set_buffer(1, Some(destination), 0);
                set_inline_bytes(
                    encoder,
                    2,
                    &Fp128::from_jolt_field(
                        &challenge.ok_or(Product9Error::State("dense bind missing"))?,
                    ),
                );
                encoder.set_buffer(3, Some(&self.buffers.e_in), 0);
                encoder.set_buffer(4, Some(&self.buffers.e_out), 0);
                encoder.set_buffer(5, Some(&self.buffers.partial_a), 0);
                set_inline_bytes(encoder, 6, &params);
            } else {
                let source = self
                    .source
                    .as_ref()
                    .ok_or(Product9Error::State("lazy source missing"))?;
                encoder.set_compute_pipeline_state(&self.pipelines.lazy);
                for (index, buffer) in [
                    &source.claims,
                    source.plane.lookups(),
                    source.plane.cycle_to_table_major(),
                    &self.buffers.phases,
                    &self.buffers.values,
                    &self.buffers.weights,
                ]
                .into_iter()
                .enumerate()
                {
                    encoder.set_buffer(index as u64, Some(buffer), 0);
                }
                set_inline_bytes(encoder, 6, &Fp128::from_jolt_field(&self.raf_interleaved));
                set_inline_bytes(encoder, 7, &Fp128::from_jolt_field(&self.raf_identity));
                for (index, buffer) in [
                    &self.buffers.e_in,
                    &self.buffers.e_out,
                    &self.buffers.partial_a,
                    &self.buffers.dense_a,
                ]
                .into_iter()
                .enumerate()
                {
                    encoder.set_buffer((index + 8) as u64, Some(buffer), 0);
                }
                set_inline_bytes(encoder, 12, &params);
                encoder.set_buffer(13, Some(&self.buffers.status), 0);
            }
            encoder.set_threadgroup_memory_length(
                0,
                (FACTORS * (THREADS / 32) * size_of::<Fp128>()) as u64,
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: e_out.len() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS as u64,
                    height: 1,
                    depth: 1,
                },
            );
            let final_in_a = encode_column_reductions(
                encoder,
                &self.pipelines.reduce,
                &self.buffers.partial_a,
                &self.buffers.partial_b,
                e_out.len(),
                FACTORS,
                self.pipelines.reduce_limits.thread_execution_width,
            )?;
            encoder.end_encoding();
            command.commit();
            crate::metal::profiling::record_submission(&command);
            command.wait_until_completed();
            validate_completed_command(&command)?;
            // Reap the profiling owner before dropping aliases or handing the
            // dense tail back to the CPU; completed work stays accounted.
            let _ = crate::metal::profiling::snapshot();
            Ok::<_, Product9Error>(final_in_a)
        })?;
        // SAFETY: the producing command completed and initialized this one-word
        // shared status before any shader could atomically record bad source rows.
        let invalid = unsafe { *self.buffers.status.contents().cast::<u32>() };
        if invalid != 0 {
            return Err(Product9Error::Source(invalid));
        }
        let output = if final_in_a {
            &self.buffers.partial_a
        } else {
            &self.buffers.partial_b
        };
        // SAFETY: the completed reduction initialized the first FACTORS fields.
        let values = unsafe { slice::from_raw_parts(output.contents().cast::<Fp128>(), FACTORS) };
        self.context.validate_inputs("Product9 q grid", values)?;
        if self.dense {
            self.source_in_a = !self.source_in_a;
        }
        self.elements = next_elements;
        self.branch_weights = next_weights;
        if materialize {
            self.dense = true;
            self.source_in_a = true;
            // Only our aliases are released; another stage's retained lookup
            // plane remains owned independently and its inverse is never written.
            self.source = None;
        }
        Ok(std::array::from_fn(|index| values[index].into_jolt_field()))
    }
}

fn write_fields(buffer: &Buffer, values: &[AkitaField]) -> Result<(), Product9Error> {
    if bytes::<Fp128>(values.len())? > buffer.length() {
        return Err(Product9Error::State("field upload exceeds capacity"));
    }
    // SAFETY: the previous command completed before each upload and the exact
    // active prefix fits the uniquely owned shared destination allocation.
    let output =
        unsafe { slice::from_raw_parts_mut(buffer.contents().cast::<Fp128>(), values.len()) };
    for (out, value) in output.iter_mut().zip(values) {
        *out = Fp128::from_jolt_field(value);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
