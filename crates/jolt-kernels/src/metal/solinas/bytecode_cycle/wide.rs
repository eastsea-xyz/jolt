use std::{mem::size_of, slice};

use jolt_field::{One as _, Prime128OffsetA7F7 as AkitaField, Zero as _};
use jolt_poly::EqPolynomial;
use metal::{objc::rc::autoreleasepool, Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::super::{
    buffer_from_slice, encode_column_reductions, set_inline_bytes, validate_completed_command,
    BooleanityRows, Fp128, MetalError, PipelineLimits, SolinasMetal, AKITA_OFFSET_FFFFA7F7,
};
use super::{byte_length, BytecodeCycleSequenceConfig, SIMD_WIDTH};

const STAGES: usize = 9;
const MAX_RA: usize = 8;
const MAX_SAMPLES: usize = MAX_RA + 2;
const DENSE_TRIM_MIN_BYTES: u64 = 64 * 1024 * 1024;
const DENSE_TRIM_CAPACITY_RATIO: u64 = 4;
const FIRST_MESSAGE: &str = "solinas_bytecode_wide_first_message";
const BIND_ROOTS: &str = "solinas_bytecode_row_bind_lo_roots";
const FIRST_BIND: &str = "solinas_bytecode_wide_first_bind_message";
const TRANSITION: &str = "solinas_bytecode_wide_transition";
const TERMINAL_BIND: &str = "solinas_bytecode_wide_terminal_bind";
const REDUCE: &str = "solinas_bytecode_wide_reduce";

/// Stage roots and sparse address roots only; no T-wide host field tables.
pub(crate) struct BytecodeCycleWideInputs<'a> {
    pub stage_points: &'a [Vec<AkitaField>],
    pub stage_weights: &'a [AkitaField],
    pub entry_weight: AkitaField,
    pub ra: &'a [Vec<AkitaField>],
    pub chunk_bits: usize,
    pub address_elements: usize,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Params {
    source_elements: u32,
    lo_length: u32,
    hi_length: u32,
    ra_count: u32,
    chunk_bits: u32,
    ra_entries: u32,
    message_pairs: u32,
    threadgroups: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RootBindParams {
    source_length: u32,
    output_length: u32,
    reserved: [u32; 2],
}

struct Pipelines {
    first_message: ComputePipelineState,
    bind_roots: ComputePipelineState,
    first_bind: ComputePipelineState,
    transition: ComputePipelineState,
    terminal_bind: ComputePipelineState,
    reduce: ComputePipelineState,
}

struct RowBuffers {
    rows: BooleanityRows,
    eq_lo: Buffer,
    bound_eq_lo: Buffer,
    weighted_eq_hi: Buffer,
    ra: Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    BeforeMessage,
    BeforeFirstBind,
    Dense,
    Finished,
    Poisoned,
}

/// Factors are packed in column-major slabs. The first bind creates only
/// T/2 entries per factor; subsequent binds ping-pong through T/4 scratch.
pub(crate) struct BytecodeCycleWideSequence {
    context: SolinasMetal,
    pipelines: Pipelines,
    reduction_limits: PipelineLimits,
    row_buffers: Option<RowBuffers>,
    dense_a: Buffer,
    dense_b: Buffer,
    #[cfg(test)]
    dense_trim_min_bytes: u64,
    partial_a: Buffer,
    partial_b: Buffer,
    params: Params,
    entry_weight: AkitaField,
    message_threads: usize,
    root_bind_threads: usize,
    transition_threads: usize,
    max_threadgroups: usize,
    current_elements: usize,
    source_in_a: bool,
    phase: Phase,
}

impl SolinasMetal {
    pub(crate) fn prepare_bytecode_cycle_wide_sequence(
        &self,
        rows: BooleanityRows,
        inputs: BytecodeCycleWideInputs<'_>,
        config: BytecodeCycleSequenceConfig,
    ) -> Result<BytecodeCycleWideSequence, MetalError> {
        self.validate_booleanity_rows(&rows)?;
        if self.offset != AKITA_OFFSET_FFFFA7F7 {
            return Err(MetalError::UnexpectedSolinasOffset {
                expected: AKITA_OFFSET_FFFFA7F7,
                got: self.offset,
            });
        }
        let elements = rows.len();
        if elements < 4 || !elements.is_power_of_two() {
            return Err(MetalError::InvalidBytecodeCycleTableLength {
                minimum: 4,
                got: elements,
            });
        }
        let row_count = u32::try_from(elements).map_err(|_| MetalError::InputTooLong(elements))?;
        if config.max_threadgroups == 0 {
            return Err(MetalError::InvalidBytecodeCycleThreadgroups(0));
        }
        let ra_count = inputs.ra.len();
        if !(1..=MAX_RA).contains(&ra_count)
            || !(1..=16).contains(&inputs.chunk_bits)
            || ra_count * inputs.chunk_bits > 56
            || !inputs.address_elements.is_power_of_two()
            || (inputs.address_elements.ilog2() as usize).div_ceil(inputs.chunk_bits) != ra_count
        {
            return Err(MetalError::InvalidBytecodeCycleState(
                "wide bytecode cycle RA geometry is unsupported",
            ));
        }
        if inputs.stage_points.len() != STAGES || inputs.stage_weights.len() != STAGES {
            return Err(MetalError::BytecodeCycleRowStageCount {
                expected: STAGES,
                points: inputs.stage_points.len(),
                weights: inputs.stage_weights.len(),
            });
        }
        let log_t = elements.ilog2() as usize;
        for (stage, point) in inputs.stage_points.iter().enumerate() {
            if point.len() != log_t {
                return Err(MetalError::BytecodeCycleRowPointLength {
                    stage,
                    expected: log_t,
                    got: point.len(),
                });
            }
        }
        let ra_entries = 1usize << inputs.chunk_bits;
        for roots in inputs.ra {
            if roots.len() != ra_entries {
                return Err(MetalError::BytecodeCyclePlaneLength {
                    plane: "RA roots",
                    expected: ra_entries,
                    got: roots.len(),
                });
            }
        }
        // Validate the unmasked PC before any dispatch. Root selectors may
        // otherwise silently truncate a mapped PC above the address domain.
        // SAFETY: BooleanityRows owns five initialized column-major u64 planes
        // and preparation runs after its producing commands have completed.
        let pc_words = unsafe {
            slice::from_raw_parts(
                rows.buffer().contents().cast::<u64>().add(4 * elements),
                elements,
            )
        };
        let invalid_pc = |word: &u64| (word & ((1u64 << 56) - 1)) > inputs.address_elements as u64;
        #[cfg(feature = "parallel")]
        let has_invalid_pc = pc_words.par_iter().any(invalid_pc);
        #[cfg(not(feature = "parallel"))]
        let has_invalid_pc = pc_words.iter().any(invalid_pc);
        if has_invalid_pc {
            return Err(MetalError::InvalidBytecodeCycleState(
                "wide bytecode cycle mapped PC exceeds the address domain",
            ));
        }

        let hi_bits = (log_t - log_t / 2)
            .min(config.max_threadgroups.ilog2() as usize)
            .min(log_t - 2);
        let lo_length = 1usize << (log_t - hi_bits);
        let hi_length = 1usize << hi_bits;
        let tables = ra_count + 3;
        let samples = ra_count + 2;
        let checked_product =
            |a: usize, b: usize| a.checked_mul(b).ok_or(MetalError::InputTooLong(elements));
        let row_words = checked_product(5, elements)?;
        let _ = u32::try_from(row_words).map_err(|_| MetalError::InputTooLong(row_words))?;
        let eq_lo_elements = checked_product(STAGES, lo_length)?;
        let bound_eq_lo_elements = eq_lo_elements / 2;
        let weighted_eq_hi_elements = checked_product(STAGES, hi_length)?;
        let ra_elements = checked_product(ra_count, ra_entries)?;
        let dense_a_elements = checked_product(tables, elements / 2)?;
        let dense_b_elements = dense_a_elements / 2;
        // Exact dispatch capacities are resolved below; max_threadgroups is a
        // safe bounded reservation for every pipeline width.
        let partial_elements = checked_product(samples, config.max_threadgroups.max(hi_length))?;
        let capacities = [
            eq_lo_elements,
            bound_eq_lo_elements,
            weighted_eq_hi_elements,
            ra_elements,
            dense_a_elements,
            dense_b_elements,
            partial_elements,
            partial_elements,
        ];
        for capacity in [
            eq_lo_elements,
            weighted_eq_hi_elements,
            dense_a_elements,
            partial_elements,
        ] {
            let _ = u32::try_from(capacity).map_err(|_| MetalError::InputTooLong(capacity))?;
        }
        let planned_elements = capacities.into_iter().try_fold(0usize, |total, capacity| {
            self.validate_buffer_length(byte_length(capacity)?)?;
            total
                .checked_add(capacity)
                .ok_or(MetalError::InputTooLong(elements))
        })?;
        let planned_bytes = byte_length(planned_elements)?;
        self.validate_additional_working_set(planned_bytes)?;
        let _allocation_span = tracing::info_span!(
            "MetalBytecodeReadRafCycle::wide_allocation_plan",
            ra_factors = ra_count,
            planned_device_bytes = planned_bytes,
        )
        .entered();

        let pipelines = Pipelines {
            first_message: self.compile_named_pipeline(FIRST_MESSAGE)?,
            bind_roots: self.compile_named_pipeline(BIND_ROOTS)?,
            first_bind: self.compile_named_pipeline(FIRST_BIND)?,
            transition: self.compile_named_pipeline(TRANSITION)?,
            terminal_bind: self.compile_named_pipeline(TERMINAL_BIND)?,
            reduce: self.compile_named_pipeline(REDUCE)?,
        };
        for (pipeline, state) in [
            (FIRST_MESSAGE, &pipelines.first_message),
            (BIND_ROOTS, &pipelines.bind_roots),
            (FIRST_BIND, &pipelines.first_bind),
            (TRANSITION, &pipelines.transition),
            (TERMINAL_BIND, &pipelines.terminal_bind),
            (REDUCE, &pipelines.reduce),
        ] {
            let limits = Self::limits(state);
            if limits.thread_execution_width != SIMD_WIDTH {
                return Err(MetalError::UnsupportedBytecodeCycleExecutionWidth {
                    pipeline,
                    expected: SIMD_WIDTH,
                    got: limits.thread_execution_width,
                });
            }
        }
        let message_threads = Self::resolve_threadgroup_width(
            config.message_threads_per_threadgroup,
            Self::limits(&pipelines.first_message),
        )?;
        let transition_threads = Self::resolve_threadgroup_width(
            config.transition_threads_per_threadgroup,
            Self::limits(&pipelines.first_bind),
        )?;
        // Both transition pipelines must accept the same dispatch width.
        let _ = Self::resolve_threadgroup_width(
            Some(transition_threads),
            Self::limits(&pipelines.transition),
        )?;
        let _ = Self::resolve_threadgroup_width(
            Some(SIMD_WIDTH),
            Self::limits(&pipelines.terminal_bind),
        )?;
        let root_bind_threads =
            Self::resolve_threadgroup_width(None, Self::limits(&pipelines.bind_roots))?;
        let reduction_limits = Self::limits(&pipelines.reduce);
        let stage_roots = |stage: usize| {
            (
                EqPolynomial::<AkitaField>::evals(&inputs.stage_points[stage][hi_bits..], None),
                EqPolynomial::<AkitaField>::evals(
                    &inputs.stage_points[stage][..hi_bits],
                    Some(inputs.stage_weights[stage]),
                ),
            )
        };
        #[cfg(feature = "parallel")]
        let roots: Vec<_> = (0..STAGES).into_par_iter().map(stage_roots).collect();
        #[cfg(not(feature = "parallel"))]
        let roots: Vec<_> = (0..STAGES).map(stage_roots).collect();
        let eq_lo = roots
            .iter()
            .flat_map(|(lo, _)| lo.iter())
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        let weighted_eq_hi = roots
            .iter()
            .flat_map(|(_, hi)| hi.iter())
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        let ra = inputs
            .ra
            .iter()
            .flatten()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>();
        for (name, values) in [
            ("wide bytecode low roots", eq_lo.as_slice()),
            ("wide bytecode high roots", weighted_eq_hi.as_slice()),
            ("wide bytecode RA roots", ra.as_slice()),
        ] {
            self.validate_inputs(name, values)?;
        }
        let new_buffer = |capacity| {
            Ok::<Buffer, MetalError>(self.device.new_buffer(
                byte_length(capacity)?,
                MTLResourceOptions::StorageModeShared,
            ))
        };
        Ok(BytecodeCycleWideSequence {
            context: self.clone(),
            pipelines,
            reduction_limits,
            row_buffers: Some(RowBuffers {
                rows,
                eq_lo: buffer_from_slice(&self.device, &eq_lo),
                bound_eq_lo: new_buffer(bound_eq_lo_elements)?,
                weighted_eq_hi: buffer_from_slice(&self.device, &weighted_eq_hi),
                ra: buffer_from_slice(&self.device, &ra),
            }),
            dense_a: new_buffer(dense_a_elements)?,
            dense_b: new_buffer(dense_b_elements)?,
            #[cfg(test)]
            dense_trim_min_bytes: DENSE_TRIM_MIN_BYTES,
            partial_a: new_buffer(partial_elements)?,
            partial_b: new_buffer(partial_elements)?,
            params: Params {
                source_elements: row_count,
                lo_length: lo_length as u32,
                hi_length: hi_length as u32,
                ra_count: ra_count as u32,
                chunk_bits: inputs.chunk_bits as u32,
                ra_entries: ra_entries as u32,
                message_pairs: 0,
                threadgroups: hi_length as u32,
            },
            entry_weight: inputs.entry_weight,
            message_threads,
            root_bind_threads,
            transition_threads,
            max_threadgroups: config.max_threadgroups,
            current_elements: elements,
            source_in_a: true,
            phase: Phase::BeforeMessage,
        })
    }
}

impl BytecodeCycleWideSequence {
    pub(crate) fn message(&mut self) -> Result<[AkitaField; MAX_SAMPLES], MetalError> {
        if self.phase != Phase::BeforeMessage {
            return Err(MetalError::InvalidBytecodeCycleState(
                "wide bytecode initial message may run exactly once",
            ));
        }
        let message = self.execute_row_round(None)?;
        self.phase = Phase::BeforeFirstBind;
        Ok(message)
    }

    pub(crate) fn bind_and_message(
        &mut self,
        challenge: AkitaField,
    ) -> Result<[AkitaField; MAX_SAMPLES], MetalError> {
        match self.phase {
            Phase::BeforeMessage => Err(MetalError::InvalidBytecodeCycleState(
                "wide bytecode first bind requires the initial message",
            )),
            Phase::BeforeFirstBind => {
                let message = self.execute_row_round(Some(challenge))?;
                self.current_elements /= 2;
                self.phase = Phase::Dense;
                self.row_buffers = None;
                Ok(message)
            }
            Phase::Dense => self.execute_dense_round(challenge),
            Phase::Finished => Err(MetalError::InvalidBytecodeCycleState(
                "wide bytecode rounds are already finished",
            )),
            Phase::Poisoned => Err(MetalError::InvalidBytecodeCycleState(
                "wide bytecode destination replacement did not complete",
            )),
        }
    }

    /// The final bind stays on the device; only the terminal scalar for each
    /// factor is returned for produced-opening metadata.
    pub(crate) fn finish_bind(
        &mut self,
        challenge: AkitaField,
    ) -> Result<Vec<AkitaField>, MetalError> {
        if self.phase != Phase::Dense || self.current_elements != 2 {
            return Err(MetalError::InvalidBytecodeCycleState(
                "wide bytecode final bind requires exactly two dense entries",
            ));
        }
        let command = autoreleasepool(|| self.context.queue.new_command_buffer().to_owned());
        autoreleasepool(|| {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.terminal_bind);
            encoder.set_buffer(0, Some(self.source()), 0);
            encoder.set_buffer(1, Some(self.destination()), 0);
            set_inline_bytes(encoder, 2, &Fp128::from_jolt_field(&challenge));
            set_inline_bytes(encoder, 3, &self.params);
            Self::dispatch(encoder, 1, SIMD_WIDTH);
            encoder.end_encoding();
            command.commit();
            crate::metal::profiling::record_submission(&command);
            command.wait_until_completed();
        });
        validate_completed_command(&command)?;
        let tables = self.params.ra_count as usize + 3;
        // SAFETY: the terminal command has completed and writes exactly one
        // canonical field per factor into the preallocated destination slab.
        let values =
            unsafe { slice::from_raw_parts(self.destination().contents().cast::<Fp128>(), tables) };
        self.context
            .validate_inputs("wide bytecode terminal factors", values)?;
        let terminal = values.iter().map(|value| value.into_jolt_field()).collect();
        self.current_elements = 1;
        self.source_in_a = !self.source_in_a;
        self.phase = Phase::Finished;
        Ok(terminal)
    }

    pub(crate) fn current_elements(&self) -> usize {
        self.current_elements
    }

    pub(crate) fn is_dense(&self) -> bool {
        matches!(self.phase, Phase::Dense | Phase::Finished)
    }

    pub(crate) fn samples(&self) -> usize {
        self.params.ra_count as usize + 2
    }

    pub(crate) fn read_current_tables(
        &self,
        output: &mut [Vec<AkitaField>],
    ) -> Result<(), MetalError> {
        if !self.is_dense() {
            return Err(MetalError::InvalidBytecodeCycleState(
                "wide bytecode factors are not dense before the first bind",
            ));
        }
        let tables = self.params.ra_count as usize + 3;
        if output.len() != tables
            || output
                .iter()
                .any(|plane| plane.len() != self.current_elements)
        {
            return Err(MetalError::InvalidBytecodeCycleState(
                "wide bytecode readback has the wrong table shape",
            ));
        }
        // SAFETY: every dispatch has completed and source has `tables`
        // initialized planes of current_elements canonical fields.
        let values = unsafe {
            slice::from_raw_parts(
                self.source().contents().cast::<Fp128>(),
                tables * self.current_elements,
            )
        };
        self.context
            .validate_inputs("wide bytecode resident factors", values)?;
        for (output, values) in output
            .iter_mut()
            .zip(values.chunks_exact(self.current_elements))
        {
            for (output, value) in output.iter_mut().zip(values) {
                *output = value.into_jolt_field();
            }
        }
        Ok(())
    }

    fn execute_row_round(
        &self,
        challenge: Option<AkitaField>,
    ) -> Result<[AkitaField; MAX_SAMPLES], MetalError> {
        let rows = self
            .row_buffers
            .as_ref()
            .ok_or(MetalError::InvalidBytecodeCycleState(
                "wide bytecode row buffers were released",
            ))?;
        let command = autoreleasepool(|| self.context.queue.new_command_buffer().to_owned());
        let final_in_a = autoreleasepool(|| {
            if let Some(challenge) = challenge {
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.pipelines.bind_roots);
                encoder.set_buffer(0, Some(&rows.eq_lo), 0);
                encoder.set_buffer(1, Some(&rows.bound_eq_lo), 0);
                set_inline_bytes(encoder, 2, &Fp128::from_jolt_field(&challenge));
                let params = RootBindParams {
                    source_length: self.params.lo_length,
                    output_length: self.params.lo_length / 2,
                    reserved: [0; 2],
                };
                set_inline_bytes(encoder, 3, &params);
                Self::dispatch(
                    encoder,
                    (STAGES * params.output_length as usize).div_ceil(self.root_bind_threads),
                    self.root_bind_threads,
                );
                encoder.end_encoding();
            }
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(if challenge.is_some() {
                &self.pipelines.first_bind
            } else {
                &self.pipelines.first_message
            });
            encoder.set_buffer(0, Some(rows.rows.buffer()), 0);
            encoder.set_buffer(
                1,
                Some(if challenge.is_some() {
                    &rows.bound_eq_lo
                } else {
                    &rows.eq_lo
                }),
                0,
            );
            encoder.set_buffer(2, Some(&rows.weighted_eq_hi), 0);
            encoder.set_buffer(3, Some(&rows.ra), 0);
            let threads = if let Some(challenge) = challenge {
                encoder.set_buffer(4, Some(&self.dense_a), 0);
                encoder.set_buffer(5, Some(&self.partial_a), 0);
                set_inline_bytes(encoder, 6, &Fp128::from_jolt_field(&challenge));
                let entry = (AkitaField::one() - challenge) * self.entry_weight;
                set_inline_bytes(encoder, 7, &Fp128::from_jolt_field(&entry));
                set_inline_bytes(encoder, 8, &self.params);
                self.transition_threads
            } else {
                encoder.set_buffer(4, Some(&self.partial_a), 0);
                set_inline_bytes(encoder, 5, &Fp128::from_jolt_field(&self.entry_weight));
                set_inline_bytes(encoder, 6, &self.params);
                self.message_threads
            };
            self.encode_message_dispatch(encoder, self.params.hi_length as usize, threads, STAGES);
            let final_in_a = self.encode_reductions(encoder, self.params.hi_length as usize)?;
            encoder.end_encoding();
            command.commit();
            crate::metal::profiling::record_submission(&command);
            command.wait_until_completed();
            Ok::<bool, MetalError>(final_in_a)
        })?;
        validate_completed_command(&command)?;
        self.read_message(final_in_a)
    }

    fn execute_dense_round(
        &mut self,
        challenge: AkitaField,
    ) -> Result<[AkitaField; MAX_SAMPLES], MetalError> {
        if self.current_elements < 4 {
            return Err(MetalError::InvalidBytecodeCycleTableLength {
                minimum: 4,
                got: self.current_elements,
            });
        }
        self.trim_dense_destination(self.current_elements / 2)?;
        let pairs = self.current_elements / 4;
        let groups = self
            .max_threadgroups
            .min(pairs.div_ceil(self.transition_threads));
        let params = Params {
            source_elements: self.current_elements as u32,
            message_pairs: pairs as u32,
            threadgroups: groups as u32,
            ..self.params
        };
        let command = autoreleasepool(|| self.context.queue.new_command_buffer().to_owned());
        let final_in_a = autoreleasepool(|| {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.transition);
            encoder.set_buffer(0, Some(self.source()), 0);
            encoder.set_buffer(1, Some(self.destination()), 0);
            encoder.set_buffer(2, Some(&self.partial_a), 0);
            set_inline_bytes(encoder, 3, &Fp128::from_jolt_field(&challenge));
            set_inline_bytes(encoder, 4, &params);
            self.encode_message_dispatch(encoder, groups, self.transition_threads, 0);
            let final_in_a = self.encode_reductions(encoder, groups)?;
            encoder.end_encoding();
            command.commit();
            crate::metal::profiling::record_submission(&command);
            command.wait_until_completed();
            Ok::<bool, MetalError>(final_in_a)
        })?;
        validate_completed_command(&command)?;
        let message = self.read_message(final_in_a)?;
        self.current_elements /= 2;
        self.source_in_a = !self.source_in_a;
        Ok(message)
    }

    fn trim_dense_destination(&mut self, output_elements: usize) -> Result<(), MetalError> {
        if self.phase != Phase::Dense
            || output_elements == 0
            || output_elements != self.current_elements / 2
        {
            return Err(MetalError::InvalidBytecodeCycleState(
                "wide bytecode replacement must match the next dense output",
            ));
        }
        let tables = self.params.ra_count as usize + 3;
        let fields = tables
            .checked_mul(output_elements)
            .ok_or(MetalError::InputTooLong(output_elements))?;
        let required_bytes = byte_length(fields)?;
        self.context.validate_buffer_length(required_bytes)?;
        let old_bytes = self.destination().length();
        #[cfg(not(test))]
        let minimum_bytes = DENSE_TRIM_MIN_BYTES;
        #[cfg(test)]
        let minimum_bytes = self.dense_trim_min_bytes;
        if old_bytes < minimum_bytes || old_bytes / DENSE_TRIM_CAPACITY_RATIO < required_bytes {
            return Ok(());
        }
        // Prior rounds have completed; reap their profiling handles before
        // releasing the inactive slab and planning its replacement.
        let _ = crate::metal::profiling::snapshot();
        super::super::validate_working_set(
            self.context
                .device
                .current_allocated_size()
                .saturating_sub(old_bytes),
            required_bytes,
            self.context.device.recommended_max_working_set_size(),
        )?;

        // No fallible operation follows the retirement. Poison the sequence
        // during the non-fallible Metal allocation so unwinding cannot expose
        // the temporary partial-buffer alias as a valid destination.
        self.phase = Phase::Poisoned;
        let placeholder = self.partial_a.clone();
        let retired = if self.source_in_a {
            std::mem::replace(&mut self.dense_b, placeholder)
        } else {
            std::mem::replace(&mut self.dense_a, placeholder)
        };
        drop(retired);
        let replacement = self
            .context
            .device
            .new_buffer(required_bytes, MTLResourceOptions::StorageModeShared);
        if self.source_in_a {
            self.dense_b = replacement;
        } else {
            self.dense_a = replacement;
        }
        self.phase = Phase::Dense;
        Ok(())
    }

    fn encode_message_dispatch(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        groups: usize,
        threads: usize,
        root_elements: usize,
    ) {
        let shared_elements = root_elements + self.samples() * (threads / SIMD_WIDTH);
        encoder.set_threadgroup_memory_length(0, (shared_elements * size_of::<Fp128>()) as u64);
        Self::dispatch(encoder, groups, threads);
    }

    fn dispatch(encoder: &metal::ComputeCommandEncoderRef, groups: usize, threads: usize) {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: groups as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads as u64,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_reductions(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        groups: usize,
    ) -> Result<bool, MetalError> {
        encode_column_reductions(
            encoder,
            &self.pipelines.reduce,
            &self.partial_a,
            &self.partial_b,
            groups,
            self.samples(),
            self.reduction_limits.thread_execution_width,
        )
    }

    fn read_message(&self, in_a: bool) -> Result<[AkitaField; MAX_SAMPLES], MetalError> {
        let buffer = if in_a {
            &self.partial_a
        } else {
            &self.partial_b
        };
        // SAFETY: dispatch and recursive reductions have completed, leaving
        // one canonical field for each active sample at the buffer front.
        let values =
            unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), self.samples()) };
        self.context
            .validate_inputs("wide bytecode message", values)?;
        let mut message = [AkitaField::zero(); MAX_SAMPLES];
        for (output, value) in message.iter_mut().zip(values) {
            *output = value.into_jolt_field();
        }
        Ok(message)
    }

    fn source(&self) -> &Buffer {
        if self.source_in_a {
            &self.dense_a
        } else {
            &self.dense_b
        }
    }

    fn destination(&self) -> &Buffer {
        if self.source_in_a {
            &self.dense_b
        } else {
            &self.dense_a
        }
    }
}

const _: () = assert!(size_of::<Params>() == 32);
const _: () = assert!(size_of::<RootBindParams>() == 16);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "wide bytecode cycle Metal parity setup")]
mod tests {
    use super::*;
    use crate::metal::solinas::BooleanityRow;
    use jolt_field::Ring as _;

    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn random_field(seed: &mut u64) -> AkitaField {
        AkitaField::from_u128((u128::from(next(seed)) << 64) | u128::from(next(seed)))
    }

    fn message(tables: &[Vec<AkitaField>], samples: usize) -> Vec<AkitaField> {
        (0..samples)
            .map(|sample| {
                let t = AkitaField::from_u64(if sample == 0 { 0 } else { sample as u64 + 1 });
                (0..tables[0].len() / 2).fold(AkitaField::zero(), |sum, pair| {
                    let factor = |plane: usize| {
                        let lo = tables[plane][2 * pair];
                        lo + t * (tables[plane][2 * pair + 1] - lo)
                    };
                    sum + (3..tables.len())
                        .fold(factor(0) + factor(2) * factor(1), |value, plane| {
                            value * factor(plane)
                        })
                })
            })
            .collect()
    }

    fn bind(tables: &[Vec<AkitaField>], challenge: AkitaField) -> Vec<Vec<AkitaField>> {
        tables
            .iter()
            .map(|plane| {
                plane
                    .chunks_exact(2)
                    .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn random_wide_chunk_sequences_match_every_message_bind_and_terminal_evaluation() {
        check_random_wide_sequences(DENSE_TRIM_MIN_BYTES, false);
    }

    #[test]
    fn shrinking_inactive_wide_slabs_preserves_all_rounds_sources_and_terminal_values() {
        check_random_wide_sequences(4 * 1024, true);
    }

    fn check_random_wide_sequences(minimum_trim_bytes: u64, expect_shrink: bool) {
        let context = SolinasMetal::for_akita().unwrap();
        let mut capacity_decreases = 0;
        let mut seed = 0xfe65_7150_3020_a1bdu64;
        for (log_t, log_k, chunk_bits, max_threadgroups, threads) in [
            // The smallest admitted trace goes directly from the row bind to
            // the terminal bind. Wider groups also exercise inactive SIMD
            // groups participating uniformly in the reduction barriers.
            (2usize, 21usize, 4usize, 1, 64),
            (3, 21, 4, 3, 128),
            (6, 3, 4, 2, 32),
            (6, 13, 8, 2, 32),
            (6, 21, 8, 2, 32),
            (6, 13, 4, 2, 32),
            (6, 17, 4, 2, 32),
            (12, 21, 4, 128, 64),
            (6, 25, 4, 2, 32),
            (6, 29, 4, 2, 32),
        ] {
            if expect_shrink && log_t != 12 {
                continue;
            }
            let elements = 1usize << log_t;
            let ra_count = log_k.div_ceil(chunk_bits);
            let addresses = 1usize << log_k;
            let stage_points = (0..STAGES)
                .map(|_| {
                    (0..log_t)
                        .map(|_| random_field(&mut seed))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let stage_weights = (0..STAGES)
                .map(|_| random_field(&mut seed))
                .collect::<Vec<_>>();
            let entry_weight = random_field(&mut seed);
            let ra = (0..ra_count)
                .map(|_| {
                    let point = (0..chunk_bits)
                        .map(|_| random_field(&mut seed))
                        .collect::<Vec<_>>();
                    EqPolynomial::<AkitaField>::evals(&point, None)
                })
                .collect::<Vec<_>>();
            let rows = (0..elements)
                .map(|index| {
                    let pc = match index % 7 {
                        0 => None,
                        1 => Some(0),
                        2 => Some(addresses as u64 - 1),
                        _ => Some(next(&mut seed) & (addresses as u64 - 1)),
                    };
                    let magnitude = if index % 11 == 0 {
                        u64::MAX
                    } else {
                        next(&mut seed)
                    };
                    let increment = if index % 3 == 0 {
                        -i128::from(magnitude)
                    } else {
                        i128::from(magnitude)
                    };
                    BooleanityRow::new(0, pc, None, increment).unwrap()
                })
                .collect::<Vec<_>>();
            let stage_eq = stage_points
                .iter()
                .map(|point| EqPolynomial::<AkitaField>::evals(point, None))
                .collect::<Vec<_>>();
            let mut initial = vec![vec![AkitaField::zero(); elements]; ra_count + 3];
            for (index, row) in rows.iter().enumerate() {
                for stage in 0..STAGES {
                    initial[usize::from(stage >= 5)][index] +=
                        stage_weights[stage] * stage_eq[stage][index];
                }
                let words = row.words();
                let magnitude = AkitaField::from_u64(words[3]);
                initial[2][index] = if words[4] >> 63 == 0 {
                    magnitude
                } else {
                    -magnitude
                };
                for chunk in 0..ra_count {
                    initial[chunk + 3][index] = row.mapped_pc().map_or(AkitaField::zero(), |pc| {
                        let hot =
                            (pc >> ((ra_count - chunk - 1) * chunk_bits)) & ((1 << chunk_bits) - 1);
                        ra[chunk][hot]
                    });
                }
            }
            initial[0][0] += entry_weight;
            let resident = context.prepare_booleanity_rows(&rows).unwrap();
            let mut sequence = context
                .prepare_bytecode_cycle_wide_sequence(
                    resident,
                    BytecodeCycleWideInputs {
                        stage_points: &stage_points,
                        stage_weights: &stage_weights,
                        entry_weight,
                        ra: &ra,
                        chunk_bits,
                        address_elements: addresses,
                    },
                    BytecodeCycleSequenceConfig {
                        message_threads_per_threadgroup: Some(threads),
                        transition_threads_per_threadgroup: Some(threads),
                        max_threadgroups,
                    },
                )
                .unwrap();
            sequence.dense_trim_min_bytes = minimum_trim_bytes;
            assert!(sequence.bind_and_message(AkitaField::one()).is_err());
            assert!(sequence.finish_bind(AkitaField::one()).is_err());
            assert_eq!(sequence.current_elements(), elements);
            let samples = ra_count + 2;
            assert_eq!(
                &sequence.message().unwrap()[..samples],
                message(&initial, samples)
            );
            assert!(sequence.message().is_err());
            assert!(sequence.finish_bind(AkitaField::one()).is_err());
            assert_eq!(sequence.current_elements(), elements);
            let mut challenges = (0..log_t)
                .map(|_| random_field(&mut seed))
                .collect::<Vec<_>>();
            challenges[0] = AkitaField::zero();
            challenges[1] = AkitaField::one();
            if let Some(challenge) = challenges.get_mut(2) {
                *challenge = -AkitaField::one();
            }
            let mut expected = initial.clone();
            for challenge in challenges.iter().take(log_t - 1) {
                let before_capacity = sequence.dense_a.length() + sequence.dense_b.length();
                let active_source = (sequence.phase == Phase::Dense)
                    .then(|| (sequence.source().contents(), sequence.source().length()));
                expected = bind(&expected, *challenge);
                let values = sequence.bind_and_message(*challenge).unwrap();
                let after_capacity = sequence.dense_a.length() + sequence.dense_b.length();
                if after_capacity < before_capacity {
                    capacity_decreases += 1;
                }
                if let Some((active_pointer, active_bytes)) = active_source {
                    assert_eq!(sequence.destination().contents(), active_pointer);
                    assert_eq!(sequence.destination().length(), active_bytes);
                }
                assert_eq!(
                    &values[..samples],
                    message(&expected, samples),
                    "logK={log_k}, chunk={chunk_bits}"
                );
                let mut actual =
                    vec![vec![AkitaField::zero(); sequence.current_elements()]; ra_count + 3];
                sequence.read_current_tables(&mut actual).unwrap();
                assert_eq!(
                    actual, expected,
                    "resident bind logK={log_k}, chunk={chunk_bits}"
                );
                if sequence.current_elements() > 2 {
                    let capacities = [sequence.dense_a.length(), sequence.dense_b.length()];
                    if expect_shrink {
                        assert!(sequence.trim_dense_destination(usize::MAX).is_err());
                        assert_eq!(sequence.phase, Phase::Dense);
                        assert_eq!(
                            [sequence.dense_a.length(), sequence.dense_b.length()],
                            capacities
                        );
                    }
                    assert!(sequence.finish_bind(AkitaField::from_u64(71)).is_err());
                    assert_eq!(
                        [sequence.dense_a.length(), sequence.dense_b.length()],
                        capacities
                    );
                    let mut after =
                        vec![vec![AkitaField::zero(); sequence.current_elements()]; ra_count + 3];
                    sequence.read_current_tables(&mut after).unwrap();
                    assert_eq!(
                        after, actual,
                        "invalid finish must preserve resident factors"
                    );
                }
            }
            // An additional polynomial round cannot consume the final pair.
            assert!(sequence.bind_and_message(AkitaField::from_u64(73)).is_err());
            let mut unchanged = vec![vec![AkitaField::zero(); 2]; ra_count + 3];
            sequence.read_current_tables(&mut unchanged).unwrap();
            assert_eq!(unchanged, expected);
            let mut wrong_shape = vec![vec![AkitaField::from_u64(83); 2]; ra_count + 2];
            let before_readback = wrong_shape.clone();
            assert!(sequence.read_current_tables(&mut wrong_shape).is_err());
            assert_eq!(wrong_shape, before_readback);
            let final_tables = bind(&expected, challenges[log_t - 1]);
            let poison = AkitaField::from_u64(0xfee1_dead);
            // SAFETY: all prior commands completed. This is the unused,
            // preallocated destination slab, with room for every terminal field.
            let destination = unsafe {
                slice::from_raw_parts_mut(
                    sequence.destination().contents().cast::<Fp128>(),
                    ra_count + 3,
                )
            };
            destination.fill(Fp128::from_jolt_field(&poison));
            let terminal = sequence.finish_bind(challenges[log_t - 1]).unwrap();
            let expected_terminal = final_tables
                .iter()
                .map(|plane| plane[0])
                .collect::<Vec<_>>();
            assert_eq!(
                terminal, expected_terminal,
                "GPU terminal bind logK={log_k}, chunk={chunk_bits}"
            );
            assert_eq!(sequence.current_elements(), 1);
            let mut actual_terminal = vec![vec![AkitaField::zero(); 1]; ra_count + 3];
            sequence.read_current_tables(&mut actual_terminal).unwrap();
            assert_eq!(
                actual_terminal, final_tables,
                "terminal GPU destination must overwrite poison"
            );
            assert!(actual_terminal.iter().any(|plane| plane[0] != poison));
            assert!(sequence.finish_bind(AkitaField::from_u64(79)).is_err());
            assert!(sequence.bind_and_message(AkitaField::one()).is_err());
            let mut after = vec![vec![AkitaField::zero(); 1]; ra_count + 3];
            sequence.read_current_tables(&mut after).unwrap();
            assert_eq!(
                after, actual_terminal,
                "duplicate finish must preserve terminal factors"
            );
            let cycle_point = challenges.iter().rev().copied().collect::<Vec<_>>();
            let eq = EqPolynomial::<AkitaField>::evals(&cycle_point, None);
            for (initial, terminal) in initial.iter().zip(final_tables) {
                let oracle = initial
                    .iter()
                    .zip(&eq)
                    .fold(AkitaField::zero(), |sum, (value, weight)| {
                        sum + *value * *weight
                    });
                assert_eq!(
                    terminal[0], oracle,
                    "terminal factor oracle logK={log_k}, chunk={chunk_bits}"
                );
            }
        }
        if expect_shrink {
            assert!(
                capacity_decreases > 0,
                "small-threshold fixture must release inactive capacity"
            );
        } else {
            assert_eq!(capacity_decreases, 0);
        }
    }

    #[test]
    fn malformed_wide_shape_and_pc_fail_before_device_allocation_or_dispatch() {
        let context = SolinasMetal::for_akita().unwrap();
        let elements = 16;
        let stage_points = vec![vec![AkitaField::from_u64(3); 4]; STAGES];
        let stage_weights = vec![AkitaField::one(); STAGES];
        let ra = vec![vec![AkitaField::one(); 16]; 6];
        let rows = vec![BooleanityRow::new(0, Some(0), None, 0).unwrap(); elements];
        let resident = context.prepare_booleanity_rows(&rows).unwrap();
        let before = context.device_info().current_allocated_size;
        let snapshot = crate::metal::profiling::snapshot();
        for (factors, bits, addresses) in [(6, 0, 1 << 21), (5, 4, 1 << 21), (6, 4, 3 << 20)] {
            assert!(context
                .prepare_bytecode_cycle_wide_sequence(
                    resident.clone(),
                    BytecodeCycleWideInputs {
                        stage_points: &stage_points,
                        stage_weights: &stage_weights,
                        entry_weight: AkitaField::zero(),
                        ra: &ra[..factors],
                        chunk_bits: bits,
                        address_elements: addresses,
                    },
                    BytecodeCycleSequenceConfig::default(),
                )
                .is_err());
            assert_eq!(context.device_info().current_allocated_size, before);
            assert_eq!(crate::metal::profiling::snapshot(), snapshot);
        }
        let invalid_rows = vec![BooleanityRow::new(0, Some(1 << 21), None, 0).unwrap(); elements];
        let resident = context.prepare_booleanity_rows(&invalid_rows).unwrap();
        let before = context.device_info().current_allocated_size;
        let error = context.prepare_bytecode_cycle_wide_sequence(
            resident.clone(),
            BytecodeCycleWideInputs {
                stage_points: &stage_points,
                stage_weights: &stage_weights,
                entry_weight: AkitaField::zero(),
                ra: &ra,
                chunk_bits: 4,
                address_elements: 1 << 21,
            },
            BytecodeCycleSequenceConfig::default(),
        );
        assert!(matches!(
            error,
            Err(MetalError::InvalidBytecodeCycleState(
                "wide bytecode cycle mapped PC exceeds the address domain"
            ))
        ));
        // Keep the invalid source alive so this compares only preparation.
        assert_eq!(context.device_info().current_allocated_size, before);
        assert_eq!(crate::metal::profiling::snapshot(), snapshot);
    }
}
