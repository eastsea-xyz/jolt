use jolt_field::Prime128OffsetA7F7 as AkitaField;
#[cfg(test)]
use jolt_field::Ring as _;
use jolt_field::Zero as _;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{ProveRounds, RoundExecutionDomain, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeReadRafAddressPhase;
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::JoltWitnessPlane;

#[cfg(all(test, feature = "parallel"))]
use rayon::prelude::*;

use super::backend::MetalBackend;
use super::solinas::bytecode_read_raf_address::{
    carrier::{ADDRESS_LOG2, INNER_LOG2},
    worklist::BYTECODE_ADDRESS_PUSHFORWARD_STAGES,
    BytecodeAddressSparseStage1Carrier,
};
use super::solinas::{
    BooleanityRows, BytecodeCycleRowInputs, BytecodeCycleRowSequence, BytecodeCycleSequenceConfig,
    BytecodeCycleTablesMut, BytecodeCycleWideInputs, BytecodeCycleWideSequence, MetalError,
};
#[cfg(test)]
use crate::optimized::bytecode_read_raf::MetalBytecodeCycleInputs;
use crate::optimized::bytecode_read_raf::{
    prepare_bytecode_read_raf_address, prepare_bytecode_read_raf_address_from_pushforwards,
    prepare_metal_bytecode_cycle_shell, BytecodeCycleAlgebra, BytecodeCycleDenseState, CycleKernel,
    OptimizedBytecodeReadRafCycle,
};
#[cfg(test)]
use crate::optimized::instruction_read_raf::InstructionCycleRow;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

// The resident-radix route takes the address domain as a runtime parameter and
// indexes it through the u16 work-item ABI, so any power-of-two bytecode domain in
// this band runs on the device. Below 2^12 the CPU pushforward is already cheap;
// above 2^16 addresses no longer fit the work-item ABI.
const RESIDENT_RADIX_MIN_ADDRESS_LOG2: u32 = 12;
const RESIDENT_RADIX_MAX_ADDRESS_LOG2: u32 = 16;
const COMPACT_MAX_ADDRESS_LOG2: u32 = 21;
const COMPACT_TRACE_CUTOFF: usize = 1 << 22;

fn resident_radix_supports(address_elements: usize) -> bool {
    address_elements.is_power_of_two()
        && (RESIDENT_RADIX_MIN_ADDRESS_LOG2..=RESIDENT_RADIX_MAX_ADDRESS_LOG2)
            .contains(&address_elements.ilog2())
}

fn compact_supports(address_elements: usize) -> bool {
    address_elements.is_power_of_two()
        && (RESIDENT_RADIX_MAX_ADDRESS_LOG2 + 1..=COMPACT_MAX_ADDRESS_LOG2)
            .contains(&address_elements.ilog2())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BytecodeReadRafAddressImplementation {
    Cpu,
    AddressMajor,
}

impl BytecodeReadRafAddressImplementation {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::AddressMajor => "address_major",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafAddressMetalConfig {
    pub implementation: BytecodeReadRafAddressImplementation,
    pub trace_cutoff_elements: usize,
}

impl Default for BytecodeReadRafAddressMetalConfig {
    fn default() -> Self {
        Self {
            implementation: BytecodeReadRafAddressImplementation::Cpu,
            trace_cutoff_elements: 1 << 20,
        }
    }
}

pub(super) fn bytecode_address_stage1_topology_supported(
    witness: &dyn JoltWitnessPlane<AkitaField>,
) -> bool {
    witness.program_preprocessing().bytecode.bytecode.len() == 1usize << ADDRESS_LOG2
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BytecodeReadRafAddressRoute {
    Cpu(&'static str),
    Stage1Sparse,
    ResidentRadix,
    Compact,
}

fn select_bytecode_address_route(
    trace_elements: usize,
    address_elements: usize,
    config: BytecodeReadRafAddressMetalConfig,
    has_stage1_carrier: bool,
    has_resident_rows: bool,
) -> BytecodeReadRafAddressRoute {
    if config.implementation == BytecodeReadRafAddressImplementation::Cpu {
        return BytecodeReadRafAddressRoute::Cpu("configured_cpu");
    }
    // The compact route needs resident rows, not the Stage-1 u16 topology.
    // Keep the legacy cutoff for smaller domains while avoiding wide-domain
    // CPU scratch at the measured 2^22/2^24-row production geometries.
    if compact_supports(address_elements)
        && trace_elements >= config.trace_cutoff_elements.min(COMPACT_TRACE_CUTOFF)
    {
        return if has_resident_rows {
            BytecodeReadRafAddressRoute::Compact
        } else {
            BytecodeReadRafAddressRoute::Cpu("resident_rows")
        };
    }
    if trace_elements < config.trace_cutoff_elements {
        return BytecodeReadRafAddressRoute::Cpu("trace_cutoff");
    }
    if address_elements == 1usize << ADDRESS_LOG2 && has_stage1_carrier {
        return BytecodeReadRafAddressRoute::Stage1Sparse;
    }
    if resident_radix_supports(address_elements) {
        return if has_resident_rows {
            BytecodeReadRafAddressRoute::ResidentRadix
        } else {
            BytecodeReadRafAddressRoute::Cpu("resident_rows")
        };
    }
    BytecodeReadRafAddressRoute::Cpu("address_domain")
}

struct BytecodeAddressEqTables {
    e_lo: Vec<Vec<AkitaField>>,
    e_hi: Vec<Vec<AkitaField>>,
}

fn split_bytecode_address_eq_tables(
    stage_points: &[Vec<AkitaField>],
    log_rows: usize,
    address_elements: usize,
) -> Result<BytecodeAddressEqTables, KernelError<AkitaField>> {
    if stage_points.len() != BYTECODE_ADDRESS_PUSHFORWARD_STAGES {
        return Err(KernelError::InvariantViolation {
            reason: "bytecode address stage count is invalid",
        });
    }
    if log_rows < INNER_LOG2 as usize
        || !(resident_radix_supports(address_elements) || compact_supports(address_elements))
    {
        return Err(KernelError::InvariantViolation {
            reason: "bytecode address geometry is invalid",
        });
    }
    let hi_bits = log_rows - INNER_LOG2 as usize;
    let mut e_lo = Vec::with_capacity(stage_points.len());
    let mut e_hi = Vec::with_capacity(stage_points.len());
    for point in stage_points {
        if point.len() != log_rows {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode address stage point has the wrong variable count",
            });
        }
        e_hi.push(EqPolynomial::evals(&point[..hi_bits], None));
        e_lo.push(EqPolynomial::evals(&point[hi_bits..], None));
    }
    Ok(BytecodeAddressEqTables { e_lo, e_hi })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BytecodeReadRafMetalConfig {
    pub trace_cutoff_elements: usize,
    pub cutoff_elements: usize,
    pub dispatch: BytecodeCycleSequenceConfig,
    pub cpu_tail_algebra: BytecodeCycleAlgebra,
}

impl Default for BytecodeReadRafMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            cutoff_elements: 2,
            dispatch: BytecodeCycleSequenceConfig::default(),
            cpu_tail_algebra: BytecodeCycleAlgebra::Q10Accum,
        }
    }
}

impl BytecodeReadRafMetalConfig {
    fn validate(self) -> Result<(), MetalError> {
        if self.cutoff_elements < 2 || !self.cutoff_elements.is_power_of_two() {
            return Err(MetalError::InvalidHybridCutoff(self.cutoff_elements));
        }
        Ok(())
    }
}

#[cfg(test)]
fn exact_bytecode_cycle_input_claim(
    rows: &[InstructionCycleRow],
    inputs: &MetalBytecodeCycleInputs,
    address_elements: usize,
) -> Result<AkitaField, KernelError<AkitaField>> {
    if rows.len() < 4
        || !rows.len().is_power_of_two()
        || inputs.stage_points.len() != 9
        || inputs.stage_weights.len() != 9
        || inputs.ra.is_empty()
        || inputs
            .ra
            .iter()
            .any(|roots| roots.len() != 1 << inputs.chunk_bits)
    {
        return Err(KernelError::InvariantViolation {
            reason: "Bytecode evaluator input-claim geometry is invalid",
        });
    }
    let log_t = rows.len().ilog2() as usize;
    if inputs.stage_points.iter().any(|point| point.len() != log_t) {
        return Err(KernelError::InvariantViolation {
            reason: "Bytecode evaluator stage point has the wrong variable count",
        });
    }
    let lo_bits = log_t / 2;
    let hi_bits = log_t - lo_bits;
    let lo_length = 1usize << lo_bits;
    let roots = inputs
        .stage_points
        .iter()
        .map(|point| {
            (
                EqPolynomial::<AkitaField>::evals(&point[..hi_bits], None),
                EqPolynomial::<AkitaField>::evals(&point[hi_bits..], None),
            )
        })
        .collect::<Vec<_>>();
    let invalid_pc = |row: &InstructionCycleRow| {
        row.mapped_pc()
            .is_some_and(|mapped_pc| mapped_pc >= address_elements)
    };
    #[cfg(feature = "parallel")]
    let has_invalid_pc = rows.par_iter().any(invalid_pc);
    #[cfg(not(feature = "parallel"))]
    let has_invalid_pc = rows.iter().any(invalid_pc);
    if has_invalid_pc {
        return Err(KernelError::InvariantViolation {
            reason: "Bytecode evaluator mapped PC exceeds the address domain",
        });
    }

    let term =
        |index: usize| {
            let row = &rows[index];
            let Some(mapped_pc) = row.mapped_pc() else {
                return AkitaField::zero();
            };
            let ra = inputs.ra.iter().enumerate().fold(
                AkitaField::from_u64(1),
                |value, (chunk, roots)| {
                    let shift = (inputs.ra.len() - chunk - 1) * inputs.chunk_bits;
                    value * roots[(mapped_pc >> shift) & (roots.len() - 1)]
                },
            );
            let hi = index >> lo_bits;
            let lo = index & (lo_length - 1);
            let stage_eq = |stage: usize| roots[stage].0[hi] * roots[stage].1[lo];
            let combined = (0..5).fold(AkitaField::zero(), |sum, stage| {
                sum + inputs.stage_weights[stage] * stage_eq(stage)
            });
            let fused_combined = (5..9).fold(AkitaField::zero(), |sum, stage| {
                sum + inputs.stage_weights[stage] * stage_eq(stage)
            });
            let entry = if index == 0 {
                inputs.entry_weight
            } else {
                AkitaField::zero()
            };
            ra * (combined + row.fused_inc::<AkitaField>() * fused_combined + entry)
        };
    #[cfg(feature = "parallel")]
    let claim = (0..rows.len())
        .into_par_iter()
        .fold(AkitaField::zero, |sum, index| sum + term(index))
        .reduce(AkitaField::zero, |left, right| left + right);
    #[cfg(not(feature = "parallel"))]
    let claim = (0..rows.len()).fold(AkitaField::zero(), |sum, index| sum + term(index));
    Ok(claim)
}

impl PrepareKernel<AkitaField, BytecodeReadRafAddressPhase<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, BytecodeReadRafAddressPhase<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = BytecodeReadRafAddressPhase<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let relation = inputs.relation;
        let cpu_inputs = || ProverInputs {
            relation,
            claims: inputs.claims,
            points: inputs.points,
            challenges: inputs.challenges,
        };
        let cpu = |session: &mut ProofSession| {
            prepare_bytecode_read_raf_address(session, witness, cpu_inputs())
        };
        let config = self.config.bytecode_read_raf_address;
        let dimensions = relation.dimensions();
        let trace_elements = 1usize << dimensions.log_t();
        let route_span = tracing::info_span!(
            "MetalBytecodeReadRafAddress::route",
            cycles = trace_elements,
            requested = config.implementation.as_str(),
            realized_route = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
        );
        let _route_guard = route_span.enter();
        let address_elements = 1usize << dimensions.log_k();
        let resident_rows = session
            .state::<BooleanityRows>()
            .filter(|rows| {
                rows.len() == trace_elements && self.context.validate_booleanity_rows(rows).is_ok()
            })
            .cloned();
        let route = select_bytecode_address_route(
            trace_elements,
            address_elements,
            config,
            session
                .state::<BytecodeAddressSparseStage1Carrier>()
                .is_some(),
            resident_rows.is_some(),
        );
        if let BytecodeReadRafAddressRoute::Cpu(reason) = route {
            let _ = route_span.record("realized_route", "cpu");
            let _ = route_span.record("fallback_reason", reason);
            return Ok(Box::new(cpu(session)?));
        }
        let stage_points = relation
            .stage_cycle_points()
            .iter()
            .chain(relation.fused_inc_cycle_points())
            .cloned()
            .collect::<Vec<_>>();
        let tables =
            split_bytecode_address_eq_tables(&stage_points, dimensions.log_t(), address_elements)?;
        if route == BytecodeReadRafAddressRoute::Compact {
            let _ = route_span.record("realized_route", "resident_compact_u32_v1");
            let _ = route_span.record("fallback_reason", "none");
            let span = tracing::info_span!(
                "MetalBytecodeReadRafAddress::compact",
                work_items = tracing::field::Empty,
                waves = tracing::field::Empty,
                member_owned_bytes = tracing::field::Empty,
                gpu_ns = tracing::field::Empty,
            );
            let _guard = span.enter();
            let rows = resident_rows.ok_or(KernelError::InvariantViolation {
                reason: "compact bytecode address rows disappeared",
            })?;
            let observation = self
                .context
                .execute_bytecode_address_compact(
                    rows,
                    dimensions.log_k() as u32,
                    &tables.e_lo,
                    &tables.e_hi,
                )
                .map_err(|error| {
                    KernelError::Sumcheck(metal_error(format!(
                        "compact bytecode address execution failed: {error}"
                    )))
                })?;
            let _ = span.record("work_items", observation.work_items);
            let _ = span.record("waves", observation.waves);
            let _ = span.record("member_owned_bytes", observation.member_owned_bytes);
            let _ = span.record(
                "gpu_ns",
                u64::try_from(observation.gpu_time.as_nanos()).unwrap_or(u64::MAX),
            );
            let prepared = prepare_bytecode_read_raf_address_from_pushforwards(
                witness,
                cpu_inputs(),
                observation.pushforwards,
                observation.first_push_pc,
            )?;
            return Ok(Box::new(prepared));
        }
        let (output, entry_trace_index) = match route {
            BytecodeReadRafAddressRoute::Stage1Sparse => {
                let _ =
                    route_span.record("realized_route", "address_major_fused_stage1_grouped_v1");
                let _ = route_span.record("fallback_reason", "none");
                let prepare_span =
                    tracing::info_span!("MetalBytecodeReadRafAddress::address_major_prepare")
                        .entered();
                let carrier = session.take::<BytecodeAddressSparseStage1Carrier>().ok_or(
                    KernelError::InvariantViolation {
                        reason: "bytecode address-major carrier is missing",
                    },
                )?;
                let receipt = carrier.receipt();
                let invocation = self
                    .context
                    .prepare_bytecode_address_sparse_resident(carrier, &tables.e_lo, &tables.e_hi)
                    .map_err(|error| {
                        KernelError::Sumcheck(metal_error(format!(
                            "bytecode address-major carrier preparation failed: {error}"
                        )))
                    })?;
                drop(prepare_span);
                let _join_span =
                    tracing::info_span!("MetalBytecodeReadRafAddress::address_major_join")
                        .entered();
                let observation = invocation.execute_timed().map_err(|error| {
                    KernelError::Sumcheck(metal_error(format!(
                        "bytecode address-major completion failed: {error}"
                    )))
                })?;
                if observation.receipt != receipt {
                    return Err(KernelError::InvariantViolation {
                        reason: "bytecode address-major source identity changed",
                    });
                }
                (observation.output, receipt.first_push_pc())
            }
            BytecodeReadRafAddressRoute::ResidentRadix => {
                let _ = route_span.record("realized_route", "resident_radix_v1");
                let _ = route_span.record("fallback_reason", "none");
                let prepare_span = tracing::info_span!(
                    "MetalBytecodeReadRafAddress::resident_radix_prepare",
                    work_items = tracing::field::Empty,
                    member_owned_bytes = tracing::field::Empty,
                    count_gpu_ns = tracing::field::Empty,
                    reduce_gpu_ns = tracing::field::Empty,
                );
                let prepare_guard = prepare_span.enter();
                let rows = resident_rows.ok_or(KernelError::InvariantViolation {
                    reason: "resident bytecode address rows disappeared",
                })?;
                let invocation = self
                    .context
                    .prepare_bytecode_address_resident(
                        rows,
                        dimensions.log_k() as u32,
                        &tables.e_lo,
                        &tables.e_hi,
                    )
                    .map_err(|error| {
                        KernelError::Sumcheck(metal_error(format!(
                            "resident bytecode address preparation failed: {error}"
                        )))
                    })?;
                let observation = invocation.execute_timed().map_err(|error| {
                    KernelError::Sumcheck(metal_error(format!(
                        "resident bytecode address completion failed: {error}"
                    )))
                })?;
                let _ = prepare_span.record("work_items", observation.stats.work_items());
                let _ = prepare_span
                    .record("member_owned_bytes", observation.stats.member_owned_bytes());
                let count_gpu_ns = u64::try_from(observation.stats.count_gpu_time().as_nanos())
                    .unwrap_or(u64::MAX);
                let reduce_gpu_ns = u64::try_from(observation.stats.reduce_gpu_time().as_nanos())
                    .unwrap_or(u64::MAX);
                let _ = prepare_span.record("count_gpu_ns", count_gpu_ns);
                let _ = prepare_span.record("reduce_gpu_ns", reduce_gpu_ns);
                drop(prepare_guard);
                (observation.output, observation.first_push_pc)
            }
            BytecodeReadRafAddressRoute::Cpu(_) | BytecodeReadRafAddressRoute::Compact => {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode address CPU route escaped early dispatch",
                });
            }
        };
        let expected_fields = stage_points.len().checked_mul(address_elements).ok_or(
            KernelError::InvariantViolation {
                reason: "bytecode address-major output size overflow",
            },
        )?;
        if output.len() != expected_fields {
            return Err(KernelError::TableSizeMismatch {
                table: "Metal bytecode address pushforwards".to_owned(),
                expected: expected_fields,
                got: output.len(),
            });
        }
        let pushforwards = output
            .chunks_exact(address_elements)
            .map(<[AkitaField]>::to_vec)
            .collect::<Vec<_>>();
        let prepared = prepare_bytecode_read_raf_address_from_pushforwards(
            witness,
            cpu_inputs(),
            pushforwards,
            entry_trace_index,
        )?;
        Ok(Box::new(prepared))
    }
}

impl PrepareKernel<AkitaField, BytecodeReadRafCycle<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, BytecodeReadRafCycle<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = BytecodeReadRafCycle<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let trace_elements = 1usize << dimensions.log_t();
        let config = self.config.bytecode_read_raf_cycle;
        config
            .validate()
            .map_err(|error| metal_error(error.to_string()))?;
        let cpu_inputs = || ProverInputs {
            relation,
            claims: inputs.claims,
            points: inputs.points,
            challenges: inputs.challenges,
        };
        let fallback = |session: &mut ProofSession| {
            OptimizedBytecodeReadRafCycle::new(config.cpu_tail_algebra).prepare(
                session,
                witness,
                cpu_inputs(),
            )
        };

        let num_ra = dimensions.num_committed_ra_polys();
        let chunk_bits = relation.committed_chunk_bits();
        let route_span = tracing::info_span!(
            "MetalBytecodeReadRafCycle::route",
            cycles = trace_elements,
            address_log2 = dimensions.log_k(),
            ra_factors = num_ra,
            committed_chunk_bits = chunk_bits,
            realized_route = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
        );
        let _route_guard = route_span.enter();
        if trace_elements < config.trace_cutoff_elements
            || config.cutoff_elements > trace_elements / 2
            || !bytecode_cycle_shape_supported(
                num_ra,
                chunk_bits,
                dimensions.log_k(),
                relation.degree(),
            )
        {
            let _ = route_span.record("realized_route", "cpu");
            let _ = route_span.record("fallback_reason", "trace_or_ra_geometry");
            return fallback(session);
        }
        let Some(rows) = session.state::<BooleanityRows>().cloned() else {
            let _ = route_span.record("realized_route", "cpu");
            let _ = route_span.record("fallback_reason", "resident_rows_unavailable");
            return fallback(session);
        };
        if rows.len() != trace_elements || self.context.validate_booleanity_rows(&rows).is_err() {
            let _ = route_span.record("realized_route", "cpu");
            let _ = route_span.record("fallback_reason", "resident_rows_shape_or_device");
            return fallback(session);
        }

        let _span = tracing::info_span!("MetalBytecodeReadRafCycle::prepare").entered();
        let (cpu, metadata) =
            prepare_metal_bytecode_cycle_shell(cpu_inputs(), config.cpu_tail_algebra)?;
        let sequence =
            if config.cutoff_elements > 2 && num_ra == 2 && chunk_bits == 8 && trace_elements >= 16
            {
                self.context
                    .prepare_bytecode_cycle_row_sequence(
                        rows,
                        BytecodeCycleRowInputs {
                            stage_points: &metadata.stage_points,
                            stage_weights: &metadata.stage_weights,
                            entry_weight: metadata.entry_weight,
                            ra0: &metadata.ra[0],
                            ra1: &metadata.ra[1],
                        },
                        config.dispatch,
                    )
                    .map(BytecodeMetalSequence::Q10)
            } else {
                self.context
                    .prepare_bytecode_cycle_wide_sequence(
                        rows,
                        BytecodeCycleWideInputs {
                            stage_points: &metadata.stage_points,
                            stage_weights: &metadata.stage_weights,
                            entry_weight: metadata.entry_weight,
                            ra: &metadata.ra,
                            chunk_bits: metadata.chunk_bits,
                            address_elements: metadata.address_elements,
                        },
                        config.dispatch,
                    )
                    .map(BytecodeMetalSequence::Wide)
            };
        let sequence = match sequence {
            Ok(sequence) => sequence,
            Err(error) if bytecode_prepare_can_fallback(&error) => {
                let _ = route_span.record("realized_route", "cpu");
                let _ = route_span.record("fallback_reason", "device_preparation_unavailable");
                tracing::warn!(
                    error = %error,
                    "bytecode Metal preparation unavailable; using the optimized CPU kernel"
                );
                return fallback(session);
            }
            Err(error) => return Err(metal_error(error.to_string()).into()),
        };
        let _ = route_span.record("realized_route", sequence.route_name());
        let _ = route_span.record("fallback_reason", "none");
        Ok(Box::new(MetalBytecodeReadRafKernel::new(
            cpu,
            sequence,
            config.cutoff_elements,
        )))
    }
}

fn bytecode_cycle_shape_supported(
    num_ra: usize,
    chunk_bits: usize,
    log_k: usize,
    degree: usize,
) -> bool {
    (1..=8).contains(&num_ra)
        && (1..=16).contains(&chunk_bits)
        && num_ra * chunk_bits <= 56
        && log_k > 0
        && log_k.div_ceil(chunk_bits) == num_ra
        && degree == num_ra + 2
}

enum BytecodeMetalSequence {
    Q10(BytecodeCycleRowSequence),
    Wide(BytecodeCycleWideSequence),
}

impl BytecodeMetalSequence {
    fn current_elements(&self) -> usize {
        match self {
            Self::Q10(sequence) => sequence.current_elements(),
            Self::Wide(sequence) => sequence.current_elements(),
        }
    }

    fn is_dense(&self) -> bool {
        match self {
            Self::Q10(sequence) => sequence.is_dense(),
            Self::Wide(sequence) => sequence.is_dense(),
        }
    }

    fn tables(&self) -> usize {
        match self {
            Self::Q10(_) => 5,
            Self::Wide(sequence) => sequence.samples() + 1,
        }
    }

    fn route_name(&self) -> &'static str {
        match self {
            Self::Q10(_) => "resident_two_byte_chunks",
            Self::Wide(_) => "resident_wide_chunks",
        }
    }

    fn message(&mut self, bind: Option<AkitaField>) -> Result<Vec<AkitaField>, MetalError> {
        match self {
            Self::Q10(sequence) => match bind {
                None => sequence.message().map(|values| values.to_vec()),
                Some(challenge) => sequence
                    .bind_and_message(challenge)
                    .map(|values| values.to_vec()),
            },
            Self::Wide(sequence) => {
                let samples = sequence.samples();
                let values = match bind {
                    None => sequence.message()?,
                    Some(challenge) => sequence.bind_and_message(challenge)?,
                };
                Ok(values[..samples].to_vec())
            }
        }
    }

    fn finish_bind(&mut self, challenge: AkitaField) -> Result<Vec<AkitaField>, MetalError> {
        match self {
            Self::Wide(sequence) => sequence.finish_bind(challenge),
            Self::Q10(_) => Err(MetalError::InvalidBytecodeCycleState(
                "legacy bytecode sequence requires a configured CPU tail",
            )),
        }
    }

    fn read_current_tables(&self, output: &mut [Vec<AkitaField>]) -> Result<(), MetalError> {
        match self {
            Self::Q10(sequence) => {
                let [combined, fused_combined, fused_inc, ra0, ra1] = output else {
                    return Err(MetalError::InvalidBytecodeCycleState(
                        "two-chunk bytecode readback needs five planes",
                    ));
                };
                sequence.read_current_tables(BytecodeCycleTablesMut {
                    combined,
                    fused_combined,
                    fused_inc,
                    ra0,
                    ra1,
                })
            }
            Self::Wide(sequence) => sequence.read_current_tables(output),
        }
    }
}

pub(crate) struct MetalBytecodeReadRafKernel {
    cpu: CycleKernel<AkitaField>,
    sequence: Option<BytecodeMetalSequence>,
    host_tail: Option<Vec<Vec<AkitaField>>>,
    cutoff_elements: usize,
    next_round: usize,
    finished: bool,
}

impl MetalBytecodeReadRafKernel {
    fn new(
        cpu: CycleKernel<AkitaField>,
        sequence: BytecodeMetalSequence,
        cutoff_elements: usize,
    ) -> Self {
        let tables = sequence.tables();
        Self {
            cpu,
            sequence: Some(sequence),
            host_tail: (cutoff_elements > 2).then(|| {
                (0..tables)
                    .map(|_| vec![AkitaField::zero(); cutoff_elements])
                    .collect()
            }),
            cutoff_elements,
            next_round: 0,
            finished: false,
        }
    }

    fn validate_round(
        &self,
        bind: Option<AkitaField>,
        round: usize,
    ) -> Result<(), SumcheckError<AkitaField>> {
        if self.finished
            || round != self.next_round
            || round >= self.cpu.num_rounds()
            || bind.is_some() != (round != 0)
        {
            return Err(metal_error(
                "bytecode round index or bind disagrees with progress",
            ));
        }
        Ok(())
    }

    fn validate_finish(&self) -> Result<(), SumcheckError<AkitaField>> {
        if self.finished || self.next_round != self.cpu.num_rounds() {
            return Err(metal_error(
                "bytecode final bind requires every polynomial round exactly once",
            ));
        }
        Ok(())
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let sequence = self
            .sequence
            .as_ref()
            .ok_or_else(|| metal_error("bytecode cycle sequence disappeared before readback"))?;
        if !sequence.is_dense() {
            return Err(metal_error(
                "bytecode cycle CPU handoff requires materialized factor tables",
            ));
        }
        let elements = sequence.current_elements();
        if elements > self.cutoff_elements || self.cpu.metal_elements()? != elements {
            return Err(metal_error(
                "bytecode cycle readback exceeds the configured tail or disagrees with progress",
            ));
        }
        let readback_bytes = elements
            .checked_mul(sequence.tables() * std::mem::size_of::<AkitaField>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| metal_error("bytecode cycle readback byte count overflowed"))?;
        let _span = tracing::info_span!(
            "MetalBytecodeReadRafCycle::readback",
            bytes = readback_bytes
        )
        .entered();
        let tables = self
            .host_tail
            .as_mut()
            .ok_or_else(|| metal_error("bytecode cycle host tail was already consumed"))?;
        if tables.len() != sequence.tables() || tables.iter().any(|plane| plane.len() < elements) {
            return Err(metal_error(
                "bytecode cycle host tail has the wrong preallocated shape",
            ));
        }
        for table in tables.iter_mut() {
            table.truncate(elements);
        }
        sequence
            .read_current_tables(tables)
            .map_err(|error| metal_error(error.to_string()))?;
        let mut tables = self
            .host_tail
            .take()
            .ok_or_else(|| metal_error("bytecode cycle host tail disappeared after readback"))?;
        let ra = tables.split_off(3);
        let [combined, fused_combined, fused_inc]: [Vec<AkitaField>; 3] = tables
            .try_into()
            .map_err(|_| metal_error("bytecode readback lost coefficient planes"))?;
        self.cpu.metal_restore_dense(BytecodeCycleDenseState {
            combined,
            fused_combined,
            fused_inc,
            ra,
        })?;
        self.sequence = None;
        Ok(())
    }
}

impl ProveRounds<AkitaField> for MetalBytecodeReadRafKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn execution_domain(&self) -> RoundExecutionDomain {
        if self.sequence.as_ref().is_some_and(|sequence| {
            self.cutoff_elements == 2
                || !sequence.is_dense()
                || sequence.current_elements() > self.cutoff_elements
        }) {
            RoundExecutionDomain::Accelerator
        } else {
            RoundExecutionDomain::Host
        }
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<jolt_poly::UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        self.validate_round(bind, round)?;
        if self.sequence.as_ref().is_some_and(|sequence| {
            self.cutoff_elements > 2
                && sequence.is_dense()
                && sequence.current_elements() <= self.cutoff_elements
        }) {
            self.restore_cpu_tail()?;
            let _span = tracing::info_span!("MetalBytecodeReadRafCycle::cpu_tail").entered();
            let polynomial = self.cpu.prove_round(bind, round, previous_claim)?;
            self.next_round += 1;
            return Ok(polynomial);
        }

        if let Some(sequence) = self.sequence.as_mut() {
            let span = match (bind.is_some(), sequence.is_dense()) {
                (false, false) => {
                    tracing::info_span!("MetalBytecodeReadRafCycle::first_message")
                }
                (true, false) => tracing::info_span!("MetalBytecodeReadRafCycle::first_bind"),
                (true, true) => tracing::info_span!("MetalBytecodeReadRafCycle::dense_round"),
                (false, true) => tracing::info_span!("MetalBytecodeReadRafCycle::invalid_round"),
            };
            let _span = span.enter();
            let evals = sequence
                .message(bind)
                .map_err(|error| metal_error(error.to_string()))?;
            if bind.is_some() {
                self.cpu.metal_commit_bind(sequence.current_elements())?;
            }
            let polynomial = self.cpu.metal_message(&evals, previous_claim)?;
            self.next_round += 1;
            return Ok(polynomial);
        }

        let _span = tracing::info_span!("MetalBytecodeReadRafCycle::cpu_tail").entered();
        let polynomial = self.cpu.prove_round(bind, round, previous_claim)?;
        self.next_round += 1;
        Ok(polynomial)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.validate_finish()?;
        if self.cutoff_elements == 2 {
            let sequence = self
                .sequence
                .as_mut()
                .ok_or_else(|| metal_error("resident bytecode terminal sequence disappeared"))?;
            self.cpu.metal_validate_finish(sequence.tables())?;
            let _span = tracing::info_span!("MetalBytecodeReadRafCycle::terminal_bind").entered();
            let terminal = sequence
                .finish_bind(bind)
                .map_err(|error| metal_error(error.to_string()))?;
            self.cpu.metal_finish_values(&terminal)?;
            self.sequence = None;
            self.finished = true;
            return Ok(());
        }
        if self.sequence.is_some() {
            self.restore_cpu_tail()?;
        }
        let _span = tracing::info_span!("MetalBytecodeReadRafCycle::cpu_tail").entered();
        self.cpu.finish_rounds(bind)?;
        self.finished = true;
        Ok(())
    }
}

impl SumcheckKernel<AkitaField> for MetalBytecodeReadRafKernel {
    type Relation = BytecodeReadRafCycle<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        self.cpu.output_claims(inputs)
    }
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

fn bytecode_prepare_can_fallback(error: &MetalError) -> bool {
    matches!(
        error,
        MetalError::InputTooLong(_)
            | MetalError::BufferTooLong { .. }
            | MetalError::WorkingSetTooLarge { .. }
            | MetalError::FunctionLookup { .. }
            | MetalError::PipelineCompilation { .. }
            | MetalError::UnsupportedBytecodeCycleExecutionWidth { .. }
            | MetalError::InvalidThreadgroupWidth { .. }
    )
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal bytecode parity setup")]
mod tests {
    use jolt_field::One as _;
    use std::num::NonZeroUsize;

    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafAddressPhaseInputClaims;
    use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
    use jolt_claims::protocols::jolt::relations::instruction::{
        InstructionReadRafChallenges, InstructionReadRafInputClaims,
    };
    use jolt_lookup_tables::XLEN as RISCV_XLEN;
    use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
    use jolt_verifier::stages::stage5::InstructionReadRaf;
    use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeStagePoints;
    use jolt_verifier::stages::stage6b::bytecode_read_raf::{
        BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCyclePhaseCommittedChallenges,
        BytecodeReadRafInputClaims, READ_RAF_CYCLE_STAGES,
    };
    use jolt_witness::testing::with_sample_backend_at_geometry;
    use jolt_witness::witnesses::FusedInc;
    use jolt_witness::ProgramSource;

    use super::*;
    use crate::metal::solinas::bytecode_read_raf_address::BytecodeAddressStage1TopologyOwner;
    use crate::metal::solinas::InstructionReadRafStage1Owner;
    use crate::metal::spartan_product::SpartanProductRemainderMetalConfig;
    use crate::optimized::instruction_read_raf::{
        collect_instruction_cycle_rows, InstructionCycleRow,
    };
    use crate::optimized::parity::{probe_input_claim, run_lockstep};
    use crate::uniskip::UniskipKernel;
    use crate::ReferenceBackend;

    #[test]
    fn resident_route_covers_every_supported_bytecode_domain() {
        let config = BytecodeReadRafAddressMetalConfig {
            implementation: BytecodeReadRafAddressImplementation::AddressMajor,
            trace_cutoff_elements: 1 << 26,
        };
        let route = |address_elements: usize, carrier: bool, rows: bool| {
            select_bytecode_address_route(1 << 28, address_elements, config, carrier, rows)
        };

        assert_eq!(
            route(1 << 13, true, true),
            BytecodeReadRafAddressRoute::Stage1Sparse
        );
        for log_k in RESIDENT_RADIX_MIN_ADDRESS_LOG2..=RESIDENT_RADIX_MAX_ADDRESS_LOG2 {
            assert_eq!(
                route(1 << log_k, false, true),
                BytecodeReadRafAddressRoute::ResidentRadix,
                "log_k {log_k}"
            );
            assert_eq!(
                route(1 << log_k, false, false),
                BytecodeReadRafAddressRoute::Cpu("resident_rows"),
                "log_k {log_k}"
            );
        }
        assert_eq!(
            route(1 << 11, false, true),
            BytecodeReadRafAddressRoute::Cpu("address_domain")
        );
        assert_eq!(
            route(1 << 17, false, true),
            BytecodeReadRafAddressRoute::Compact
        );
        assert_eq!(
            route(1 << 21, false, true),
            BytecodeReadRafAddressRoute::Compact
        );
        assert_eq!(
            route(1 << 21, false, false),
            BytecodeReadRafAddressRoute::Cpu("resident_rows")
        );
        assert_eq!(
            route(1 << 22, false, true),
            BytecodeReadRafAddressRoute::Cpu("address_domain")
        );
        assert_eq!(
            route(3 << 12, false, true),
            BytecodeReadRafAddressRoute::Cpu("address_domain")
        );
    }

    #[test]
    fn bytecode_address_metal_is_opt_in() {
        assert_eq!(
            BytecodeReadRafAddressMetalConfig::default().implementation,
            BytecodeReadRafAddressImplementation::Cpu
        );
    }

    #[test]
    fn compact_production_route_starts_at_logt22_without_lowering_legacy_cutoff() {
        let config = super::super::MetalConfig::production().bytecode_read_raf_address;
        assert_eq!(
            select_bytecode_address_route(1 << 22, 1 << 21, config, false, true),
            BytecodeReadRafAddressRoute::Compact
        );
        assert_eq!(
            select_bytecode_address_route(1 << 21, 1 << 21, config, false, true),
            BytecodeReadRafAddressRoute::Cpu("trace_cutoff")
        );
        assert_eq!(
            select_bytecode_address_route(1 << 22, 1 << 16, config, false, true),
            BytecodeReadRafAddressRoute::Cpu("trace_cutoff")
        );
    }

    #[test]
    fn logk14_skips_the_stage1_carrier() {
        let log_t = 15;
        with_sample_backend_at_geometry(log_t, 14, 8, |witness| {
            assert!(!bytecode_address_stage1_topology_supported(witness));
            let backend = MetalBackend::new(super::super::MetalConfig {
                instruction_read_raf: super::super::InstructionReadRafMetalConfig {
                    address_cutoff_elements: 1 << log_t,
                    ..Default::default()
                },
                bytecode_read_raf_address: BytecodeReadRafAddressMetalConfig {
                    implementation: BytecodeReadRafAddressImplementation::AddressMajor,
                    trace_cutoff_elements: 1 << log_t,
                },
                ..Default::default()
            })
            .unwrap();
            let mut session = ProofSession::default();
            <MetalBackend as UniskipKernel<AkitaField, OuterRemainder<AkitaField>>>::prepare_witness(
                &backend,
                &mut session,
                log_t,
                witness,
            )
            .unwrap();

            assert!(session.state::<InstructionReadRafStage1Owner>().is_some());
            assert!(session
                .state::<BytecodeAddressStage1TopologyOwner>()
                .is_none());
        });
    }

    #[test]
    fn logk12_resident_route_matches_reference_without_stage1_topology() {
        resident_route_matches_reference_without_stage1_topology(12);
    }

    #[test]
    fn logk14_resident_route_matches_reference_without_stage1_topology() {
        resident_route_matches_reference_without_stage1_topology(14);
    }

    #[test]
    fn logk15_resident_route_matches_reference_without_stage1_topology() {
        resident_route_matches_reference_without_stage1_topology(15);
    }

    #[test]
    fn logk17_compact_route_matches_reference_without_stage1_topology() {
        resident_route_matches_reference_without_stage1_topology(17);
    }

    fn resident_route_matches_reference_without_stage1_topology(log_k: usize) {
        let log_t = 15;
        with_sample_backend_at_geometry(log_t, log_k, 8, |witness| {
            assert_eq!(
                witness.program_preprocessing().bytecode.bytecode.len(),
                1 << log_k
            );
            let dimensions = BytecodeReadRafDimensions::new(log_t, log_k, 2);
            let relation = BytecodeReadRafAddressPhase::new(
                dimensions,
                true,
                BytecodeStagePoints {
                    stage_cycle_points: std::array::from_fn(|stage| {
                        point(log_t, 211 + stage as u64)
                    }),
                    register_read_write_point: point(7 + log_t, 239),
                    register_val_evaluation_point: point(7 + log_t, 241),
                    fused_inc_cycle_points: (0..4)
                        .map(|stage| point(log_t, 251 + stage as u64))
                        .collect(),
                },
                0,
            );
            let challenges = BytecodeReadRafAddressPhaseChallenges {
                gamma: AkitaField::from_u64(257),
                stage1_gamma: AkitaField::from_u64(263),
                stage2_gamma: AkitaField::from_u64(269),
                stage3_gamma: AkitaField::from_u64(271),
                stage4_gamma: AkitaField::from_u64(277),
                stage5_gamma: AkitaField::from_u64(281),
            };
            let claims = LatticeReadRafAddressPhaseInputClaims::<AkitaField>::default();
            let points = LatticeReadRafAddressPhaseInputClaims::<Vec<AkitaField>>::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };
            let mut reference = <ReferenceBackend as PrepareKernel<
                AkitaField,
                BytecodeReadRafAddressPhase<AkitaField>,
            >>::prepare(
                &ReferenceBackend,
                &mut ProofSession::default(),
                witness,
                inputs(),
            )
            .unwrap();
            let claim = probe_input_claim(reference.as_mut());

            let backend = MetalBackend::new(super::super::MetalConfig {
                instruction_read_raf: super::super::InstructionReadRafMetalConfig {
                    address_cutoff_elements: 1 << log_t,
                    ..Default::default()
                },
                bytecode_read_raf_address: BytecodeReadRafAddressMetalConfig {
                    implementation: BytecodeReadRafAddressImplementation::AddressMajor,
                    trace_cutoff_elements: 1 << log_t,
                },
                ..Default::default()
            })
            .unwrap();
            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();
            let resident = backend
                .context
                .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&packed))
                .unwrap();
            let mut session = ProofSession::default();
            session.park(resident);
            assert!(session
                .state::<BytecodeAddressStage1TopologyOwner>()
                .is_none());
            assert!(session
                .state::<BytecodeAddressSparseStage1Carrier>()
                .is_none());
            let mut actual = <MetalBackend as PrepareKernel<
                AkitaField,
                BytecodeReadRafAddressPhase<AkitaField>,
            >>::prepare(&backend, &mut session, witness, inputs())
            .unwrap();

            let round_challenges = point(log_k, 307);
            run_lockstep(
                reference.as_mut(),
                actual.as_mut(),
                claim,
                &round_challenges,
            );
            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                actual.output_claims(&claims).unwrap()
            );
        });
    }

    #[test]
    fn logk12_resident_scatter_covers_high_addresses_and_split_hot_items() {
        resident_scatter_covers_high_addresses_and_split_hot_items(12);
    }

    #[test]
    fn logk14_resident_scatter_covers_high_addresses_and_split_hot_items() {
        resident_scatter_covers_high_addresses_and_split_hot_items(14);
    }

    fn resident_scatter_covers_high_addresses_and_split_hot_items(log_k: usize) {
        use crate::metal::solinas::BooleanityRow;

        let log_t = 15;
        let cycles = 1usize << log_t;
        let addresses = 1usize << log_k;
        let stage_points = (0..9)
            .map(|stage| point(log_t, 401 + stage as u64))
            .collect::<Vec<_>>();
        let tables = split_bytecode_address_eq_tables(&stage_points, log_t, addresses).unwrap();
        let mut mapped = Vec::with_capacity(cycles);
        let mut increments = Vec::with_capacity(cycles);
        let rows = (0..cycles)
            .map(|cycle| {
                let pc = if cycle < 6_000 {
                    None
                } else {
                    Some((cycle * 7_919) % (addresses - 1))
                };
                let increment = if cycle.is_multiple_of(3) {
                    -((cycle % 29 + 1) as i128)
                } else {
                    (cycle % 31 + 1) as i128
                };
                mapped.push(pc);
                increments.push(increment);
                BooleanityRow::new(0, pc.map(|value| value as u64), None, increment).unwrap()
            })
            .collect::<Vec<_>>();
        assert!(mapped
            .iter()
            .flatten()
            .copied()
            .max()
            .is_some_and(|pc| pc >= addresses / 2));

        let backend = MetalBackend::new(Default::default()).unwrap();
        let resident = backend.context.prepare_booleanity_rows(&rows).unwrap();
        let invocation = backend
            .context
            .prepare_bytecode_address_resident(resident, log_k as u32, &tables.e_lo, &tables.e_hi)
            .unwrap();
        let observation = invocation.execute_timed().unwrap();
        assert_eq!(observation.first_push_pc, 0);
        let mut populations = vec![0usize; addresses];
        for address in mapped.iter().map(|pc| pc.unwrap_or(0)) {
            populations[address] += 1;
        }
        assert!(populations[0] > 4_096);
        assert_eq!(
            observation.stats.work_items(),
            populations
                .iter()
                .map(|count| count.div_ceil(4_096))
                .sum::<usize>()
        );
        assert!(observation.stats.member_owned_bytes() < 6usize << 30);

        let mut expected = vec![AkitaField::zero(); stage_points.len() * addresses];
        for cycle in 0..cycles {
            let address = mapped[cycle].unwrap_or(0);
            let outer = cycle >> INNER_LOG2;
            let inner = cycle & ((1 << INNER_LOG2) - 1);
            for stage in 0..stage_points.len() {
                let mut contribution = tables.e_lo[stage][inner] * tables.e_hi[stage][outer];
                if stage >= 5 {
                    let magnitude = AkitaField::from_u64(increments[cycle].unsigned_abs() as u64);
                    contribution *= if increments[cycle] < 0 {
                        -magnitude
                    } else {
                        magnitude
                    };
                }
                expected[stage * addresses + address] += contribution;
            }
        }
        assert_eq!(observation.output, expected);
    }

    fn point(len: usize, seed: u64) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    #[test]
    fn direct_input_claim_rejects_pc_outside_relation_domain() {
        let rows = (0..4)
            .map(|index| {
                InstructionCycleRow::new(
                    0,
                    None,
                    false,
                    if index == 0 { 4 } else { index },
                    None,
                    FusedInc(0),
                )
            })
            .collect::<Vec<_>>();
        let inputs = MetalBytecodeCycleInputs {
            stage_points: (0..9).map(|stage| point(2, stage)).collect(),
            stage_weights: vec![AkitaField::one(); 9],
            entry_weight: AkitaField::one(),
            ra: vec![vec![AkitaField::one(); 256]; 2],
            chunk_bits: 8,
            address_elements: 4,
        };

        assert!(matches!(
            exact_bytecode_cycle_input_claim(&rows, &inputs, 4),
            Err(KernelError::InvariantViolation {
                reason: "Bytecode evaluator mapped PC exceeds the address domain"
            })
        ));
    }

    #[test]
    fn bytecode_address_production_route_matches_the_optimized_host_shell() {
        let log_t = 15;
        with_sample_backend_at_geometry(log_t, 13, 8, |witness| {
            assert_eq!(
                witness.program_preprocessing().bytecode.bytecode.len(),
                1 << 13
            );
            let dimensions = BytecodeReadRafDimensions::new(log_t, 13, 2);
            let relation = BytecodeReadRafAddressPhase::new(
                dimensions,
                true,
                BytecodeStagePoints {
                    stage_cycle_points: std::array::from_fn(|stage| {
                        point(log_t, 11 + stage as u64)
                    }),
                    register_read_write_point: point(7 + log_t, 31),
                    register_val_evaluation_point: point(7 + log_t, 37),
                    fused_inc_cycle_points: (0..4)
                        .map(|stage| point(log_t, 43 + stage as u64))
                        .collect(),
                },
                0,
            );
            let challenges = BytecodeReadRafAddressPhaseChallenges {
                gamma: AkitaField::from_u64(3),
                stage1_gamma: AkitaField::from_u64(5),
                stage2_gamma: AkitaField::from_u64(7),
                stage3_gamma: AkitaField::from_u64(11),
                stage4_gamma: AkitaField::from_u64(13),
                stage5_gamma: AkitaField::from_u64(17),
            };
            let claims = LatticeReadRafAddressPhaseInputClaims::<AkitaField>::default();
            let points = LatticeReadRafAddressPhaseInputClaims::<Vec<AkitaField>>::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut reference = <ReferenceBackend as PrepareKernel<
                AkitaField,
                BytecodeReadRafAddressPhase<AkitaField>,
            >>::prepare(
                &ReferenceBackend,
                &mut ProofSession::default(),
                witness,
                inputs(),
            )
            .unwrap();
            let claim = probe_input_claim(reference.as_mut());
            let round_challenges = point(dimensions.log_k(), 101);

            let production = MetalBackend::new(super::super::MetalConfig {
                spartan_product_remainder: SpartanProductRemainderMetalConfig {
                    trace_cutoff_elements: 1 << log_t,
                    ..Default::default()
                },
                instruction_read_raf: super::super::InstructionReadRafMetalConfig {
                    address_cutoff_elements: 1 << log_t,
                    ..Default::default()
                },
                bytecode_read_raf_address: BytecodeReadRafAddressMetalConfig {
                    implementation: BytecodeReadRafAddressImplementation::AddressMajor,
                    trace_cutoff_elements: 1 << log_t,
                },
                ..Default::default()
            })
            .unwrap();
            let mut session = ProofSession::default();
            <MetalBackend as UniskipKernel<AkitaField, OuterRemainder<AkitaField>>>::prepare_witness(
                &production,
                &mut session,
                log_t,
                witness,
            )
            .unwrap();
            let source_receipt = session
                .state::<InstructionReadRafStage1Owner>()
                .unwrap()
                .receipt();
            assert_eq!(source_receipt.row_bytes(), 40 * (1 << log_t));
            assert!(session
                .state::<BytecodeAddressStage1TopologyOwner>()
                .is_some());
            assert!(session
                .state::<BytecodeAddressSparseStage1Carrier>()
                .is_none());
            let instruction_dimensions = InstructionReadRafDimensions::new(
                log_t,
                2 * RISCV_XLEN,
                NonZeroUsize::new(4).unwrap(),
            );
            let instruction_relation = InstructionReadRaf::new(instruction_dimensions);
            let instruction_claims = InstructionReadRafInputClaims::<AkitaField>::default();
            let instruction_points = InstructionReadRafInputClaims {
                lookup_output: point(log_t, 151),
                left_lookup_operand: point(log_t, 157),
                right_lookup_operand: point(log_t, 163),
            };
            let instruction_challenges = InstructionReadRafChallenges {
                gamma: AkitaField::from_u64(167),
            };
            let instruction_inputs = ProverInputs {
                relation: &instruction_relation,
                claims: &instruction_claims,
                points: &instruction_points,
                challenges: &instruction_challenges,
            };
            let instruction_kernel = <MetalBackend as PrepareKernel<
                AkitaField,
                InstructionReadRaf<AkitaField>,
            >>::prepare(
                &production, &mut session, witness, instruction_inputs
            )
            .unwrap();
            assert!(session
                .state::<BytecodeAddressStage1TopologyOwner>()
                .is_none());
            assert!(session
                .state::<BytecodeAddressSparseStage1Carrier>()
                .is_some());
            drop(instruction_kernel);
            let mut production_actual =
                <MetalBackend as PrepareKernel<
                    AkitaField,
                    BytecodeReadRafAddressPhase<AkitaField>,
                >>::prepare(&production, &mut session, witness, inputs())
                .unwrap();
            assert!(session
                .state::<BytecodeAddressSparseStage1Carrier>()
                .is_none());
            let mut production_expected =
                prepare_bytecode_read_raf_address(&mut ProofSession::default(), witness, inputs())
                    .unwrap();
            run_lockstep(
                &mut production_expected,
                production_actual.as_mut(),
                claim,
                &round_challenges,
            );
            assert_eq!(
                production_expected.output_claims(&claims).unwrap(),
                production_actual.output_claims(&claims).unwrap()
            );
        });
    }

    #[test]
    fn terminal_metadata_requires_final_progress_and_exact_factor_shape() {
        let log_t = 3;
        let relation = BytecodeReadRafCycle::committed(BytecodeReadRafCommittedCycleInputs {
            dimensions: BytecodeReadRafDimensions::new(log_t, 13, 2),
            r_address: point(13, 19),
            stage_cycle_points: std::array::from_fn(|stage| point(log_t, 41 + stage as u64)),
            entry_bytecode_index: 17,
            committed_chunk_bits: 8,
            val_stages: (0..NUM_BYTECODE_VAL_STAGES)
                .map(|stage| AkitaField::from_u64(101 + stage as u64))
                .collect(),
        });
        let claims = BytecodeReadRafInputClaims::<AkitaField>::default();
        let points = BytecodeReadRafInputClaims::<Vec<AkitaField>>::default();
        let challenges = BytecodeReadRafCyclePhaseCommittedChallenges {
            gamma: AkitaField::from_u64(31),
        };
        let (mut shell, _) = prepare_metal_bytecode_cycle_shell(
            ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            },
            BytecodeCycleAlgebra::Q10Accum,
        )
        .unwrap();
        assert!(shell.metal_finish_values(&[AkitaField::zero(); 5]).is_err());
        assert_eq!(shell.metal_elements().unwrap(), 8);
        assert_eq!(shell.metal_rounds_bound(), 0);
        shell.metal_commit_bind(4).unwrap();
        shell.metal_commit_bind(2).unwrap();
        assert!(shell.metal_finish_values(&[AkitaField::zero(); 4]).is_err());
        assert_eq!(shell.metal_elements().unwrap(), 2);
        assert_eq!(shell.metal_rounds_bound(), 2);
        let terminal = [11, 13, 17, 19, 23].map(AkitaField::from_u64);
        shell.metal_finish_values(&terminal).unwrap();
        assert_eq!(shell.metal_rounds_bound(), log_t);
        let output = shell.output_claims(&claims).unwrap();
        assert_eq!(output.fused_inc, terminal[2]);
        assert_eq!(output.bytecode_ra, terminal[3..]);
        assert!(shell.metal_finish_values(&terminal).is_err());
        assert_eq!(shell.output_claims(&claims).unwrap(), output);
    }

    #[test]
    fn production_bytecode_cutoff_keeps_rounds_resident_and_rejects_invalid_tails() {
        assert_eq!(BytecodeReadRafMetalConfig::default().cutoff_elements, 2);
        for cutoff_elements in [0, 1, 3, 6] {
            let config = BytecodeReadRafMetalConfig {
                cutoff_elements,
                ..Default::default()
            };
            assert!(
                matches!(config.validate(), Err(MetalError::InvalidHybridCutoff(got)) if got == cutoff_elements)
            );
        }
        for cutoff_elements in [2, 4, 1 << 16] {
            assert!(BytecodeReadRafMetalConfig {
                cutoff_elements,
                ..Default::default()
            }
            .validate()
            .is_ok());
        }
        with_sample_backend_at_geometry(3, 2, 8, |witness| {
            let relation = BytecodeReadRafCycle::committed(BytecodeReadRafCommittedCycleInputs {
                dimensions: BytecodeReadRafDimensions::new(3, 13, 2),
                r_address: point(13, 19),
                stage_cycle_points: std::array::from_fn(|stage| point(3, 41 + stage as u64)),
                entry_bytecode_index: 17,
                committed_chunk_bits: 8,
                val_stages: (0..NUM_BYTECODE_VAL_STAGES)
                    .map(|stage| AkitaField::from_u64(101 + stage as u64))
                    .collect(),
            });
            let claims = BytecodeReadRafInputClaims::<AkitaField>::default();
            let points = BytecodeReadRafInputClaims::<Vec<AkitaField>>::default();
            let challenges = BytecodeReadRafCyclePhaseCommittedChallenges {
                gamma: AkitaField::from_u64(31),
            };
            let mut metal = MetalBackend::new(Default::default()).unwrap();
            // Exercise the local preparation gate even if a caller bypasses
            // MetalBackend::new's shared configuration validation.
            for cutoff in [0, 1, 3] {
                metal.config.bytecode_read_raf_cycle.cutoff_elements = cutoff;
                let before = metal.context.device_info().current_allocated_size;
                let snapshot = crate::metal::profiling::snapshot();
                let result = <MetalBackend as PrepareKernel<
                    AkitaField,
                    BytecodeReadRafCycle<AkitaField>,
                >>::prepare(
                    &metal,
                    &mut ProofSession::default(),
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                );
                assert!(
                    result.is_err(),
                    "invalid cutoff {cutoff} must fail before fallback or allocation"
                );
                assert_eq!(metal.context.device_info().current_allocated_size, before);
                assert_eq!(crate::metal::profiling::snapshot(), snapshot);
            }
        });
    }

    #[test]
    fn bytecode_cycle_route_accepts_native_wide_nibbles_and_rejects_inconsistent_shapes() {
        for (ra, bits, log_k) in [(1, 4, 4), (2, 8, 13), (3, 8, 21), (6, 4, 21), (8, 4, 31)] {
            assert!(bytecode_cycle_shape_supported(ra, bits, log_k, ra + 2));
        }
        for (ra, bits, log_k, degree) in [
            (0, 4, 21, 2),
            (9, 4, 33, 11),
            (6, 0, 21, 8),
            (6, 17, 21, 8),
            (6, 4, 21, 7),
            (3, 4, 21, 5),
            (8, 8, 56, 10),
            (1, 4, 0, 3),
        ] {
            assert!(!bytecode_cycle_shape_supported(ra, bits, log_k, degree));
        }
    }

    #[test]
    fn native_wide_bytecode_cycle_rounds_match_cpu_and_terminal_oracle() {
        use crate::optimized::instruction_read_raf::SharedInstructionRows;
        use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
        use jolt_witness::witnesses::RaChunkSelector;
        use std::sync::Arc;

        let log_t = 9;
        let mut seed = 0x9160_4cbf_e789_230du64;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // A small witness supplies the trait surface. Injected shared rows use
        // the actual native address geometry without allocating a 2^21 program.
        with_sample_backend_at_geometry(log_t, 2, 8, |witness| {
            for (log_k, chunk_bits) in [(17usize, 8usize), (21, 8), (21, 4), (13, 8)] {
                let ra_count = log_k.div_ceil(chunk_bits);
                let dimensions = BytecodeReadRafDimensions::new(log_t, log_k, ra_count);
                let address_elements = 1usize << log_k;
                let address_mask = address_elements - 1;
                let r_address = (0..log_k)
                    .map(|_| {
                        AkitaField::from_u128((u128::from(random()) << 64) | u128::from(random()))
                    })
                    .collect::<Vec<_>>();
                let packed = Arc::new(
                    (0..1usize << log_t)
                        .map(|index| {
                            let pc = match index % 7 {
                                0 => 0,
                                1 => address_mask,
                                // The u16 boundary is meaningful only in the
                                // wide fixtures; the two-chunk 13-bit case must
                                // still contain valid PCs for its own domain.
                                2 => 0xffff & address_mask,
                                3 => 0x1_0000 & address_mask,
                                _ => random() as usize & address_mask,
                            };
                            assert!(
                                pc < address_elements,
                                "fixture PC {pc} exceeds logK {log_k}"
                            );
                            let magnitude = if index % 13 == 0 { 0 } else { random() };
                            let increment = if index % 3 == 0 {
                                -i128::from(magnitude)
                            } else {
                                i128::from(magnitude)
                            };
                            InstructionCycleRow::new(
                                random().into(),
                                None,
                                false,
                                pc,
                                None,
                                FusedInc(increment),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                let relation =
                    BytecodeReadRafCycle::committed(BytecodeReadRafCommittedCycleInputs {
                        dimensions,
                        r_address: r_address.clone(),
                        stage_cycle_points: std::array::from_fn(|_| {
                            (0..log_t)
                                .map(|_| {
                                    AkitaField::from_u128(
                                        (u128::from(random()) << 64) | u128::from(random()),
                                    )
                                })
                                .collect()
                        }),
                        entry_bytecode_index: (1 << log_k) - 1,
                        committed_chunk_bits: chunk_bits,
                        val_stages: (0..NUM_BYTECODE_VAL_STAGES)
                            .map(|_| AkitaField::from_u64(random()))
                            .collect(),
                    });
                let claims = BytecodeReadRafInputClaims::<AkitaField>::default();
                let points = BytecodeReadRafInputClaims::<Vec<AkitaField>>::default();
                let challenges = BytecodeReadRafCyclePhaseCommittedChallenges {
                    gamma: AkitaField::from_u64(random()),
                };
                let inputs = || ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                };
                let mut cpu_session = ProofSession::default();
                cpu_session.park(SharedInstructionRows(Arc::clone(&packed)));
                let mut expected = OptimizedBytecodeReadRafCycle::default()
                    .prepare(&mut cpu_session, witness, inputs())
                    .unwrap();
                let metal = MetalBackend::new(super::super::MetalConfig {
                    bytecode_read_raf_cycle: BytecodeReadRafMetalConfig {
                        trace_cutoff_elements: 2,
                        dispatch: BytecodeCycleSequenceConfig {
                            message_threads_per_threadgroup: Some(32),
                            transition_threads_per_threadgroup: Some(32),
                            max_threadgroups: 8,
                        },
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .unwrap();
                let resident = metal
                    .context
                    .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&packed))
                    .unwrap();
                let mut metal_session = ProofSession::default();
                metal_session.park(resident);
                // The strong carry also makes a CPU fallback numerically honest.
                metal_session.park(SharedInstructionRows(Arc::clone(&packed)));
                let mut actual = <MetalBackend as PrepareKernel<
                    AkitaField,
                    BytecodeReadRafCycle<AkitaField>,
                >>::prepare(
                    &metal, &mut metal_session, witness, inputs()
                )
                .unwrap();
                assert_eq!(actual.execution_domain(), RoundExecutionDomain::Accelerator,
                    "native wide bytecode cycle must select Metal before round zero (logK={log_k}, chunk={chunk_bits})");
                let mut round_challenges = (0..log_t)
                    .map(|_| {
                        AkitaField::from_u128((u128::from(random()) << 64) | u128::from(random()))
                    })
                    .collect::<Vec<_>>();
                round_challenges[0] = AkitaField::zero();
                round_challenges[1] = AkitaField::one();
                round_challenges[2] = -AkitaField::one();
                let cycle_tables = relation
                    .stage_cycle_points()
                    .iter()
                    .map(|point| EqPolynomial::<AkitaField>::evals(point, None))
                    .collect::<Vec<_>>();
                let chunk_eq = committed_address_chunks(&r_address, chunk_bits)
                    .iter()
                    .map(|point| EqPolynomial::<AkitaField>::evals(point, None))
                    .collect::<Vec<_>>();
                let mut powers = vec![AkitaField::one()];
                for exponent in 1..12 {
                    powers.push(powers[exponent - 1] * challenges.gamma);
                }
                let stage_values = relation.stage_values_at_r_address().unwrap();
                let address_integer =
                    r_address
                        .iter()
                        .enumerate()
                        .fold(AkitaField::zero(), |sum, (bit, value)| {
                            sum + *value * AkitaField::from_u128(1u128 << (log_k - bit - 1))
                        });
                let mut weights = (0..5)
                    .map(|stage| powers[stage] * stage_values[stage])
                    .collect::<Vec<_>>();
                weights[0] += powers[9] * address_integer;
                weights[2] += powers[10] * address_integer;
                weights.extend((5..9).map(|stage| {
                    powers[stage]
                        * if stage < 7 {
                            stage_values[5]
                        } else {
                            AkitaField::one() - stage_values[5]
                        }
                }));
                let entry_weight = powers[11]
                    * r_address
                        .iter()
                        .fold(AkitaField::one(), |value, factor| value * *factor);
                let initial_coefficient = |index: usize, stages: std::ops::Range<usize>| {
                    stages.fold(AkitaField::zero(), |sum, stage| {
                        sum + weights[stage] * cycle_tables[stage][index]
                    })
                };
                let mut claim =
                    packed
                        .iter()
                        .enumerate()
                        .fold(AkitaField::zero(), |sum, (index, row)| {
                            let ra = (0..ra_count).fold(AkitaField::one(), |value, chunk| {
                                let selector =
                                    RaChunkSelector::new(chunk, ra_count, chunk_bits).unwrap();
                                value * chunk_eq[chunk][selector.chunk_usize(row.bytecode_pc())]
                            });
                            let combined = initial_coefficient(index, 0..5)
                                + if index == 0 {
                                    entry_weight
                                } else {
                                    AkitaField::zero()
                                };
                            sum + ra
                                * (combined
                                    + row.fused_inc::<AkitaField>()
                                        * initial_coefficient(index, 5..9))
                        });
                assert_ne!(claim, AkitaField::zero(), "nontrivial input claim");
                for round in 0..log_t {
                    assert_eq!(
                        actual.execution_domain(),
                        RoundExecutionDomain::Accelerator,
                        "all polynomial rounds must remain on Metal"
                    );
                    let bind = round
                        .checked_sub(1)
                        .map(|previous| round_challenges[previous]);
                    assert!(
                        actual.finish_rounds(AkitaField::from_u64(73)).is_err(),
                        "early finish must preserve state"
                    );
                    assert!(
                        actual.prove_round(bind, log_t, claim).is_err(),
                        "round beyond transcript must preserve state"
                    );
                    let invalid_bind = if round == 0 {
                        Some(AkitaField::one())
                    } else {
                        None
                    };
                    assert!(
                        actual.prove_round(invalid_bind, round, claim).is_err(),
                        "bind parity must preserve state"
                    );
                    if round != 0 {
                        assert!(
                            actual.prove_round(bind, round - 1, claim).is_err(),
                            "stale round must preserve state"
                        );
                    }
                    let expected_poly = expected.prove_round(bind, round, claim).unwrap();
                    let actual_poly = actual.prove_round(bind, round, claim).unwrap();
                    assert_eq!(
                        actual_poly, expected_poly,
                        "logK={log_k}, chunk={chunk_bits}, round={round}"
                    );
                    claim = expected_poly.evaluate(round_challenges[round]);
                    assert!(
                        actual.prove_round(bind, round, claim).is_err(),
                        "duplicate round must preserve state"
                    );
                }
                assert_eq!(
                    actual.execution_domain(),
                    RoundExecutionDomain::Accelerator,
                    "the terminal bind must remain on Metal"
                );
                assert!(actual
                    .prove_round(Some(AkitaField::one()), log_t, claim)
                    .is_err());
                expected.finish_rounds(round_challenges[log_t - 1]).unwrap();
                actual.finish_rounds(round_challenges[log_t - 1]).unwrap();
                let output = actual.output_claims(&claims).unwrap();
                assert!(
                    actual.finish_rounds(AkitaField::from_u64(79)).is_err(),
                    "duplicate finish must preserve terminal claims"
                );
                assert!(
                    actual.prove_round(None, 0, claim).is_err(),
                    "finished proof rejects a new round"
                );
                assert_eq!(actual.output_claims(&claims).unwrap(), output);
                assert_eq!(output, expected.output_claims(&claims).unwrap());
                let cycle_point = round_challenges.iter().rev().copied().collect::<Vec<_>>();
                let cycle_eq = EqPolynomial::<AkitaField>::evals(&cycle_point, None);
                let oracle_ra = (0..ra_count)
                    .map(|chunk| {
                        let selector = RaChunkSelector::new(chunk, ra_count, chunk_bits).unwrap();
                        packed.iter().zip(&cycle_eq).fold(
                            AkitaField::zero(),
                            |sum, (row, weight)| {
                                sum + *weight
                                    * chunk_eq[chunk][selector.chunk_usize(row.bytecode_pc())]
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                let oracle_inc = packed
                    .iter()
                    .zip(&cycle_eq)
                    .fold(AkitaField::zero(), |sum, (row, weight)| {
                        sum + *weight * row.fused_inc::<AkitaField>()
                    });
                assert_eq!(
                    output.bytecode_ra, oracle_ra,
                    "terminal RA oracle, logK={log_k}, chunk={chunk_bits}"
                );
                assert_eq!(
                    output.fused_inc, oracle_inc,
                    "terminal increment oracle, logK={log_k}, chunk={chunk_bits}"
                );
                let oracle_combined = cycle_eq
                    .iter()
                    .enumerate()
                    .fold(AkitaField::zero(), |sum, (index, weight)| {
                        sum + *weight * initial_coefficient(index, 0..5)
                    })
                    + cycle_eq[0] * entry_weight;
                let oracle_fused = cycle_eq
                    .iter()
                    .enumerate()
                    .fold(AkitaField::zero(), |sum, (index, weight)| {
                        sum + *weight * initial_coefficient(index, 5..9)
                    });
                let oracle_claim = output.bytecode_ra.iter().fold(
                    oracle_combined + oracle_inc * oracle_fused,
                    |value, factor| value * *factor,
                );
                assert_eq!(
                    claim, oracle_claim,
                    "terminal summand oracle, logK={log_k}, chunk={chunk_bits}"
                );
            }
        });
    }

    #[test]
    fn production_kernel_matches_optimized_cpu_through_handoff() {
        let log_t = 10;
        with_sample_backend_at_geometry(log_t, 13, 8, |witness| {
            let dimensions = BytecodeReadRafDimensions::new(log_t, 13, 2);
            let relation = BytecodeReadRafCycle::committed(BytecodeReadRafCommittedCycleInputs {
                dimensions,
                r_address: point(13, 19),
                stage_cycle_points: std::array::from_fn(|stage| point(log_t, 41 + stage as u64)),
                entry_bytecode_index: 17,
                committed_chunk_bits: 8,
                val_stages: (0..NUM_BYTECODE_VAL_STAGES)
                    .map(|stage| AkitaField::from_u64(101 + stage as u64))
                    .collect(),
            });
            assert_eq!(relation.stage_cycle_points().len(), READ_RAF_CYCLE_STAGES);
            let claims = BytecodeReadRafInputClaims::<AkitaField>::default();
            let points = BytecodeReadRafInputClaims::<Vec<AkitaField>>::default();
            let challenges = BytecodeReadRafCyclePhaseCommittedChallenges {
                gamma: AkitaField::from_u64(31),
            };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let (mut shell, metadata) =
                prepare_metal_bytecode_cycle_shell(inputs(), BytecodeCycleAlgebra::Q10Accum)
                    .unwrap();
            assert_eq!(shell.metal_elements().unwrap(), 1 << log_t);
            assert_eq!(shell.metal_rounds_bound(), 0);
            let _ = shell
                .metal_message(
                    &(0..4)
                        .map(|index| AkitaField::from_u64(7 + index))
                        .collect::<Vec<_>>(),
                    AkitaField::zero(),
                )
                .unwrap();
            assert!(shell.metal_commit_bind((1 << log_t) / 4).is_err());
            shell.metal_commit_bind((1 << log_t) / 2).unwrap();
            assert_eq!(shell.metal_rounds_bound(), 1);
            assert_eq!(shell.metal_elements().unwrap(), (1 << log_t) / 2);

            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();
            let claim = exact_bytecode_cycle_input_claim(&packed, &metadata, 1 << 13).unwrap();
            let mut reference = <ReferenceBackend as PrepareKernel<
                AkitaField,
                BytecodeReadRafCycle<AkitaField>,
            >>::prepare(
                &ReferenceBackend,
                &mut ProofSession::default(),
                witness,
                inputs(),
            )
            .unwrap();
            assert_eq!(claim, probe_input_claim(reference.as_mut()));
            assert_ne!(claim, AkitaField::zero());

            let mut expected = OptimizedBytecodeReadRafCycle::new(BytecodeCycleAlgebra::Q10Accum)
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let metal = MetalBackend::new(super::super::MetalConfig {
                bytecode_read_raf_cycle: BytecodeReadRafMetalConfig {
                    trace_cutoff_elements: 2,
                    cutoff_elements: 4,
                    dispatch: BytecodeCycleSequenceConfig {
                        message_threads_per_threadgroup: Some(32),
                        transition_threads_per_threadgroup: Some(32),
                        max_threadgroups: 1 << 13,
                    },
                    cpu_tail_algebra: BytecodeCycleAlgebra::Q10Accum,
                },
                ..Default::default()
            })
            .unwrap();
            let resident = metal
                .context
                .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&packed))
                .unwrap();
            let mut session = ProofSession::default();
            session.park(resident);
            let mut actual = <MetalBackend as PrepareKernel<
                AkitaField,
                BytecodeReadRafCycle<AkitaField>,
            >>::prepare(&metal, &mut session, witness, inputs())
            .unwrap();
            assert!(session.state::<BooleanityRows>().is_some());

            let mut round_challenges = point(log_t, 211);
            round_challenges[0] = AkitaField::zero();
            round_challenges[1] = AkitaField::one();
            round_challenges[2] = -AkitaField::one();
            let mut claim = claim;
            let mut nonzero_round = false;
            for round in 0..log_t {
                let bind = round
                    .checked_sub(1)
                    .map(|previous| round_challenges[previous]);
                let expected_poly = expected.prove_round(bind, round, claim).unwrap();
                let actual_poly = actual.prove_round(bind, round, claim).unwrap();
                assert_eq!(actual_poly, expected_poly, "round {round}");
                nonzero_round |= expected_poly
                    .coefficients()
                    .iter()
                    .any(|coefficient| *coefficient != AkitaField::zero());
                claim = expected_poly.evaluate(round_challenges[round]);
            }
            assert!(nonzero_round, "all round polynomials were zero");
            let final_bind = round_challenges[log_t - 1];
            expected.finish_rounds(final_bind).unwrap();
            actual.finish_rounds(final_bind).unwrap();
            assert_eq!(
                actual.output_claims(&claims).unwrap(),
                expected.output_claims(&claims).unwrap()
            );
        });
    }
}
