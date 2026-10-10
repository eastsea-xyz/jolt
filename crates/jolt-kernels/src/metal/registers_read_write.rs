use std::{cmp::Ordering, sync::Arc};

use jolt_field::{Prime128OffsetA7F7 as AkitaField, Zero as _};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, RoundExecutionDomain, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChecking, RegistersReadWriteOutputClaims,
};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::backend::MetalBackend;
use super::registers_val_evaluation::RegistersValEvaluationSource;
use super::solinas::registers_claim_reduction::RegistersClaimResidentRdPlane;
use super::solinas::registers_read_write::{
    RegistersReadWriteCycleFinish, RegistersReadWriteCycleSequence,
};
use super::solinas::{
    MetalError, PendingRegistersReadWriteStage1Pipelines, RegistersReadWriteStage1Source,
};
use crate::optimized::registers_read_write::{
    AlignedPackedRegisterRows, AlignedPackedRegisterRowsError, OptimizedRegistersReadWrite,
    PackedRegisterCycleRow, RegisterCapacityExceeded, SharedRdIndices,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[cfg(feature = "test-utils")]
mod evaluation;
#[cfg(feature = "test-utils")]
pub use evaluation::{
    RegistersReadWriteCpuEvalSample, RegistersReadWriteCpuMetalEvalFixture,
    RegistersReadWriteEvalError, RegistersReadWriteEvalResult, RegistersReadWriteMetalEvalSample,
    RegistersReadWriteRoundTiming, RegistersReadWriteShapeSnapshot,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersReadWriteMetalConfig {
    pub trace_cutoff_elements: usize,
}

impl Default for RegistersReadWriteMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 22,
        }
    }
}

type RegistersReadWriteKernelBox =
    Box<dyn SumcheckKernel<AkitaField, Relation = RegistersReadWriteChecking<AkitaField>>>;

const METAL_OPERAND_CLAIMS_LOG_T_MIN: usize = 22;

pub(crate) struct SharedPackedRegisterRows {
    pub(crate) rows: Arc<AlignedPackedRegisterRows>,
    pub(crate) cycles: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for SharedPackedRegisterRows {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(mut shared) = visitor.enter_shared(
            allocative::Key::new("packed_rows"),
            std::mem::size_of::<Arc<AlignedPackedRegisterRows>>(),
            Arc::as_ptr(&self.rows).cast(),
        ) {
            shared.visit_simple(
                allocative::Key::new("allocation"),
                self.rows.allocation_bytes(),
            );
            shared.exit();
        }
        visitor.exit();
    }
}

struct MetalRegistersReadWriteKernel {
    sequence: Vec<RegistersReadWriteCycleSequence>,
    source_owner: Option<Arc<AlignedPackedRegisterRows>>,
    continuation: Option<RegistersReadWriteKernelBox>,
    gruen: GruenSplitEqPolynomial<AkitaField>,
    r_cycle: Vec<AkitaField>,
    cycle_challenges: Vec<AkitaField>,
    bound_challenges: Vec<AkitaField>,
    gamma: AkitaField,
    log_t: usize,
    log_k: usize,
    use_metal_operand_claims: bool,
    next_round: usize,
    finished: bool,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalRegistersReadWriteKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::{gruen_heap_bytes, vec_heap_bytes};
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("sequence"),
            self.sequence
                .iter()
                .map(RegistersReadWriteCycleSequence::resident_bytes)
                .sum(),
        );
        visitor.visit_simple(allocative::Key::new("gruen"), gruen_heap_bytes(&self.gruen));
        visitor.visit_simple(
            allocative::Key::new("challenges"),
            vec_heap_bytes(&self.r_cycle)
                + vec_heap_bytes(&self.cycle_challenges)
                + vec_heap_bytes(&self.bound_challenges),
        );
        visitor.exit();
    }
}

impl MetalRegistersReadWriteKernel {
    fn validate_round(
        &self,
        bind: Option<&AkitaField>,
        round: usize,
    ) -> Result<(), SumcheckError<AkitaField>> {
        if self.finished || round != self.next_round || round >= self.num_rounds() {
            return Err(metal_error(
                "registers read-write received an out-of-order round",
            ));
        }
        if (round == 0) != bind.is_none() {
            return Err(metal_error(
                "registers read-write round has the wrong bind shape",
            ));
        }
        Ok(())
    }

    fn cycle_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if let Some(challenge) = bind {
            self.gruen.bind(challenge);
            self.cycle_challenges.push(challenge);
            self.bound_challenges.push(challenge);
        }
        let mut quadratic = [AkitaField::zero(); 2];
        for (bank, sequence) in self.sequence.iter_mut().enumerate() {
            let observation = match bind {
                Some(challenge) => sequence
                    .bind_and_message(
                        challenge,
                        self.gruen.e_in_current(),
                        self.gruen.e_out_current(),
                    )
                    .map_err(metal_runtime_error)?,
                None => sequence
                    .message(
                        self.gruen.e_in_current(),
                        self.gruen.e_out_current(),
                        self.gamma,
                    )
                    .map_err(metal_runtime_error)?,
            };
            quadratic[0] += observation.quadratic[0];
            quadratic[1] += observation.quadratic[1];
            tracing::info!(
            target: "jolt::metal",
            round,
            bank,
            wall_ns = duration_nanos(observation.wall),
            gpu_active_ns = duration_nanos(observation.gpu_active),
            prefill_gpu_active_ns = duration_nanos(observation.prefill_gpu_active),
            allocation_ns = duration_nanos(observation.allocation),
            resident_bytes = observation.resident_bytes,
            peak_transition_bytes = observation.peak_transition_bytes,
            retired_source_bytes = observation.retired_source_bytes,
            "completed registers read-write Metal cycle round"
            );
        }
        Ok(self
            .gruen
            .gruen_poly_deg_3(quadratic[0], quadratic[1], previous_claim))
    }

    fn enter_address_phase(
        &mut self,
        challenge: AkitaField,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if self.continuation.is_some() || self.cycle_challenges.len() + 1 != self.log_t {
            return Err(metal_error(
                "registers read-write reached the address phase in an invalid state",
            ));
        }
        self.cycle_challenges.push(challenge);
        self.bound_challenges.push(challenge);
        let mut roots = Vec::new();
        let mut shared_increment = None;
        for (bank, sequence) in self.sequence.iter_mut().enumerate() {
            let RegistersReadWriteCycleFinish {
                roots: bank_roots,
                increment: bank_increment,
                wall,
                gpu_active,
                allocation,
                resident_bytes,
                peak_transition_bytes,
            } = sequence.finish(challenge).map_err(metal_runtime_error)?;
            if shared_increment.is_some_and(|increment| increment != bank_increment) {
                return Err(metal_error(
                    "register read-write banks disagree on the global increment",
                ));
            }
            shared_increment = Some(bank_increment);
            roots.extend(bank_roots);
            tracing::info!(
            target: "jolt::metal",
            bank,
            wall_ns = duration_nanos(wall),
            gpu_active_ns = duration_nanos(gpu_active),
            allocation_ns = duration_nanos(allocation),
            resident_bytes,
            peak_transition_bytes,
            "completed registers read-write Metal cycle handoff"
            );
        }
        let increment = shared_increment
            .ok_or_else(|| metal_error("registers read-write has no cycle sequence"))?;
        let operand_rows = if self.use_metal_operand_claims {
            None
        } else {
            Some(
                self.source_owner
                    .as_ref()
                    .ok_or_else(|| metal_error("registers read-write lost its CPU operand rows"))?
                    .as_slice(),
            )
        };
        let mut continuation = OptimizedRegistersReadWrite::prepare_after_cycle_phase(
            self.log_t,
            self.log_k,
            &self.r_cycle,
            operand_rows,
            &self.cycle_challenges,
            roots,
            increment,
        )
        .map_err(kernel_runtime_error)?;
        let polynomial = continuation.prove_round(None, round, previous_claim)?;
        self.continuation = Some(continuation);
        Ok(polynomial)
    }

    fn continuation_mut(
        &mut self,
    ) -> Result<&mut RegistersReadWriteKernelBox, SumcheckError<AkitaField>> {
        self.continuation
            .as_mut()
            .ok_or_else(|| metal_error("registers read-write lost its CPU address continuation"))
    }
}

impl ProveRounds<AkitaField> for MetalRegistersReadWriteKernel {
    fn num_rounds(&self) -> usize {
        self.log_t + self.log_k
    }

    fn execution_domain(&self) -> RoundExecutionDomain {
        if self.next_round <= self.log_t {
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
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        self.validate_round(bind.as_ref(), round)?;
        let polynomial = match round.cmp(&self.log_t) {
            Ordering::Less => self.cycle_round(bind, round, previous_claim)?,
            Ordering::Equal => self.enter_address_phase(
                bind.ok_or_else(|| {
                    metal_error("registers read-write address phase missed the final cycle bind")
                })?,
                round,
                previous_claim,
            )?,
            Ordering::Greater => {
                let challenge = bind.ok_or_else(|| {
                    metal_error("registers read-write address round missed its bind")
                })?;
                self.bound_challenges.push(challenge);
                self.continuation_mut()?
                    .prove_round(Some(challenge), round, previous_claim)?
            }
        };
        self.next_round += 1;
        Ok(polynomial)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.finished || self.next_round != self.num_rounds() {
            return Err(metal_error(
                "registers read-write reached finish in an invalid state",
            ));
        }
        self.bound_challenges.push(bind);
        self.continuation_mut()?.finish_rounds(bind)?;
        self.finished = true;
        Ok(())
    }
}

impl SumcheckKernel<AkitaField> for MetalRegistersReadWriteKernel {
    type Relation = RegistersReadWriteChecking<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<RegistersReadWriteOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        if !self.finished || self.bound_challenges.len() != self.num_rounds() {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "registers read-write output requested before finish",
            });
        }
        let mut outputs = self
            .continuation
            .as_mut()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "registers read-write output lost its address continuation",
            })?
            .output_claims(inputs)?;
        if self.use_metal_operand_claims {
            let cycle_point = self.bound_challenges[..self.log_t]
                .iter()
                .rev()
                .copied()
                .collect::<Vec<_>>();
            let address_point = self.bound_challenges[self.log_t..]
                .iter()
                .rev()
                .copied()
                .collect::<Vec<_>>();
            let observation = self
                .sequence
                .first()
                .ok_or(SumcheckKernelError::InvariantViolation {
                    reason: "registers read-write lost its operand source sequence",
                })?
                .operand_claims(&cycle_point, &address_point)
                .map_err(|error| SumcheckKernelError::ComputeBackend {
                    backend: "metal",
                    message: error.to_string(),
                })?;
            outputs.rs1_ra = observation.claims[0];
            outputs.rs2_ra = observation.claims[1];
            tracing::info!(
                target: "jolt::metal",
                prepare_ns = duration_nanos(observation.prepare),
                wall_ns = duration_nanos(observation.wall),
                gpu_active_ns = duration_nanos(observation.gpu_active),
                "completed registers read-write Metal operand claims"
            );
        }
        Ok(outputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        if !self.finished {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "registers read-write derived table requested before finish",
            });
        }
        self.continuation
            .as_ref()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "registers read-write validation lost its address continuation",
            })?
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

impl PrepareKernel<AkitaField, RegistersReadWriteChecking<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, RegistersReadWriteChecking<AkitaField>>,
    ) -> Result<RegistersReadWriteKernelBox, KernelError<AkitaField>> {
        let dimensions = inputs.relation.register_dimensions();
        let log_t = dimensions.log_t();
        let log_k = dimensions.log_k();
        if dimensions.phase1_num_rounds() != log_t
            || log_t == 0
            || log_t >= 32
            || inputs.points.rd_write_value.len() != log_t
        {
            return Err(KernelError::Unsupported {
                reason: "Metal registers read-write requires the default cycle phase",
            });
        }
        let cycles = 1usize << log_t;
        if cycles < self.config.registers_read_write.trace_cutoff_elements {
            record_route(
                cycles,
                "metal_cycle_sequence_v1",
                "optimized_cpu",
                "trace_cutoff",
                0,
                "none",
            );
            let prepared = OptimizedRegistersReadWrite.prepare(session, witness, inputs)?;
            super::instruction_read_raf::start_instruction_read_raf_scatter(session)?;
            return Ok(prepared);
        }
        if let Some(capacity) = session.take::<RegisterCapacityExceeded>() {
            if log_t >= 25 {
                return prepare_capacity_fallback(session, witness, inputs, capacity);
            }
        }
        let needs_rd_carry = self.config.registers_val_evaluation.source
            != RegistersValEvaluationSource::Stage1Resident
            || cycles < self.config.registers_val_evaluation.trace_cutoff_elements
            || !(26..=28).contains(&log_t);
        if log_t == 28 && !needs_rd_carry {
            let resident_source = session.state::<RegistersReadWriteStage1Source>().is_some();
            let resident_rd_post = session.state::<RegistersClaimResidentRdPlane>().is_some();
            let pending_pipelines = session
                .state::<PendingRegistersReadWriteStage1Pipelines>()
                .is_some();
            if resident_source != resident_rd_post || resident_source != pending_pipelines {
                return Err(KernelError::InvariantViolation {
                    reason: "registers read-write resident source carries are incomplete",
                });
            }
            if resident_source {
                let pipeline_join = session
                    .take::<PendingRegistersReadWriteStage1Pipelines>()
                    .ok_or(KernelError::InvariantViolation {
                        reason: "registers read-write pipeline warm-up disappeared",
                    })?
                    .join()
                    .map_err(metal_prepare_error)?;
                tracing::info!(
                    target: "jolt::metal",
                    wall_ns = duration_nanos(pipeline_join),
                    "joined registers read-write Stage-1 pipeline warm-up"
                );
                let source_owner = session.take::<RegistersReadWriteStage1Source>().ok_or(
                    KernelError::InvariantViolation {
                        reason: "registers read-write resident source disappeared",
                    },
                )?;
                let rd_post = session.take::<RegistersClaimResidentRdPlane>().ok_or(
                    KernelError::InvariantViolation {
                        reason: "registers read-write resident rd-post plane disappeared",
                    },
                )?;
                let source = source_owner.device_view();
                let physical_rows = source.physical_rows;
                let _span = tracing::info_span!(
                    "MetalRegistersReadWrite::source_prepare",
                    source_kind = "stage1_fused_delta_carried_rs1_v1",
                    physical_rows,
                    witness_row_extractions = 0usize,
                    packed_owner_bytes = 0usize,
                )
                .entered();
                let sequence = self
                    .context
                    .prepare_registers_read_write_cycle_sequence_from_stage1(
                        source_owner,
                        rd_post,
                        log_t,
                        inputs.challenges.gamma,
                    )
                    .map_err(metal_prepare_error)?;
                let source_bytes = sequence.source_bytes();
                record_route(
                    cycles,
                    "metal_cycle_sequence_v1",
                    "metal_cycle_sequence_v1",
                    "none",
                    source_bytes,
                    "stage1_fused_delta_carried_rs1_v1",
                );
                #[cfg(any(test, feature = "test-utils"))]
                let _ = self
                    .test_counters
                    .registers_read_write_metal_sequences
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let kernel = Box::new(MetalRegistersReadWriteKernel {
                    sequence: vec![sequence],
                    source_owner: None,
                    continuation: None,
                    gruen: GruenSplitEqPolynomial::new(
                        &inputs.points.rd_write_value,
                        BindingOrder::LowToHigh,
                    ),
                    r_cycle: inputs.points.rd_write_value.clone(),
                    cycle_challenges: Vec::with_capacity(log_t),
                    bound_challenges: Vec::with_capacity(log_t + log_k),
                    gamma: inputs.challenges.gamma,
                    log_t,
                    log_k,
                    use_metal_operand_claims: true,
                    next_round: 0,
                    finished: false,
                });
                return Ok(kernel);
            }
        }
        let Some(access) = witness.random_access() else {
            record_route(
                cycles,
                "metal_cycle_sequence_v1",
                "optimized_cpu",
                "random_access_unavailable",
                0,
                "none",
            );
            let prepared = OptimizedRegistersReadWrite.prepare(session, witness, inputs)?;
            super::instruction_read_raf::start_instruction_read_raf_scatter(session)?;
            return Ok(prepared);
        };
        if access.cycles() < cycles {
            return Err(KernelError::InvariantViolation {
                reason: "registers read-write random-access witness is shorter than its domain",
            });
        }
        let physical_rows = access.physical_rows().min(cycles);
        let _span = tracing::info_span!(
            "MetalRegistersReadWrite::source_prepare",
            source_kind = "direct_witness",
            physical_rows,
            witness_row_extractions = physical_rows,
            packed_owner_bytes = 0,
        )
        .entered();
        let rows = match if log_t < 25 {
            AlignedPackedRegisterRows::collect_native(&access, physical_rows)
        } else {
            AlignedPackedRegisterRows::collect(&access, physical_rows, log_t >= 28)
        } {
            Ok(rows) => rows,
            Err(AlignedPackedRegisterRowsError::Capacity(capacity)) => {
                return prepare_capacity_fallback(session, witness, inputs, capacity);
            }
            Err(AlignedPackedRegisterRowsError::Witness(error)) => {
                return Err(KernelError::Witness(error));
            }
            Err(AlignedPackedRegisterRowsError::Storage(reason)) => {
                return Err(KernelError::InvariantViolation { reason });
            }
        };
        let source_owner = Arc::new(rows);
        let partitioned = source_owner.device_view().active_registers() > 64;
        let source_kind = if partitioned {
            "direct_witness_two_banks"
        } else {
            "direct_witness"
        };
        let value_config = self.config.registers_val_evaluation;
        let value_split_handoff = cycles >> (log_t / 2).saturating_sub(1);
        let packed_value_requested = log_t < 25
            && cycles >= value_config.trace_cutoff_elements
            && cycles > value_config.cutoff_elements.max(value_split_handoff)
            && (value_config.source == RegistersValEvaluationSource::WitnessUpload
                || (value_config.source == RegistersValEvaluationSource::Stage1Resident
                    && log_t < 26));
        if packed_value_requested {
            session.park(SharedPackedRegisterRows {
                rows: Arc::clone(&source_owner),
                cycles,
            });
        } else if needs_rd_carry {
            #[cfg(feature = "parallel")]
            let mut rd_indices = source_owner
                .as_slice()
                .par_iter()
                .copied()
                .map(PackedRegisterCycleRow::rd_index)
                .collect::<Vec<_>>();
            #[cfg(not(feature = "parallel"))]
            let mut rd_indices = source_owner
                .as_slice()
                .iter()
                .copied()
                .map(PackedRegisterCycleRow::rd_index)
                .collect::<Vec<_>>();
            rd_indices.resize(cycles, None);
            session.park(SharedRdIndices(rd_indices));
        }
        let source_bytes = source_owner.allocation_bytes();
        let use_metal_operand_claims = log_t >= METAL_OPERAND_CLAIMS_LOG_T_MIN;
        let continuation_source = (!use_metal_operand_claims).then(|| Arc::clone(&source_owner));
        let sequence = if partitioned {
            [0, 64]
                .into_iter()
                .map(|base| {
                    self.context
                        .prepare_registers_read_write_cycle_sequence_partition(
                            Arc::clone(&source_owner),
                            log_t,
                            inputs.challenges.gamma,
                            base,
                        )
                        .map_err(metal_prepare_error)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![self
                .context
                .prepare_registers_read_write_cycle_sequence(
                    source_owner,
                    log_t,
                    inputs.challenges.gamma,
                )
                .map_err(metal_prepare_error)?]
        };
        record_route(
            cycles,
            "metal_cycle_sequence_v1",
            "metal_cycle_sequence_v1",
            "none",
            source_bytes,
            source_kind,
        );
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
            .registers_read_write_metal_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let kernel = Box::new(MetalRegistersReadWriteKernel {
            sequence,
            source_owner: continuation_source,
            continuation: None,
            gruen: GruenSplitEqPolynomial::new(
                &inputs.points.rd_write_value,
                BindingOrder::LowToHigh,
            ),
            r_cycle: inputs.points.rd_write_value.clone(),
            cycle_challenges: Vec::with_capacity(log_t),
            bound_challenges: Vec::with_capacity(log_t + log_k),
            gamma: inputs.challenges.gamma,
            log_t,
            log_k,
            use_metal_operand_claims,
            next_round: 0,
            finished: false,
        });
        super::instruction_read_raf::start_instruction_read_raf_scatter(session)?;
        Ok(kernel)
    }
}

fn record_route(
    cycles: usize,
    requested: &'static str,
    selected: &'static str,
    fallback_reason: &'static str,
    source_bytes: usize,
    source_kind: &'static str,
) {
    let _span = tracing::info_span!(
        "MetalRegistersReadWrite::route",
        cycles,
        requested,
        selected,
        fallback_reason,
        source_bytes,
        source_kind,
    )
    .entered();
}

fn prepare_capacity_fallback(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    inputs: ProverInputs<'_, AkitaField, RegistersReadWriteChecking<AkitaField>>,
    capacity: RegisterCapacityExceeded,
) -> Result<RegistersReadWriteKernelBox, KernelError<AkitaField>> {
    tracing::info!(
        target: "jolt::metal",
        active_registers = capacity.active_registers,
        "registers read-write active domain exceeds Metal capacity; using optimized CPU"
    );
    record_route(
        1usize << inputs.relation.register_dimensions().log_t(),
        "metal_cycle_sequence_v1",
        "optimized_cpu",
        "active_register_capacity",
        0,
        "none",
    );
    let prepared = OptimizedRegistersReadWrite.prepare(session, witness, inputs)?;
    super::instruction_read_raf::start_instruction_read_raf_scatter(session)?;
    Ok(prepared)
}

fn metal_prepare_error(error: MetalError) -> KernelError<AkitaField> {
    metal_runtime_error(error).into()
}

fn kernel_runtime_error(error: KernelError<AkitaField>) -> SumcheckError<AkitaField> {
    metal_error(error.to_string())
}

fn metal_runtime_error(error: MetalError) -> SumcheckError<AkitaField> {
    metal_error(error.to_string())
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

fn duration_nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "Metal registers read-write parity setup"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::{
        ReadWriteDimensions, REGISTER_ADDRESS_BITS,
    };
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_field::Ring as _;
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage4::registers_read_write_checking::{
        RegistersReadWriteChallenges, RegistersReadWriteInputClaims,
    };
    use jolt_witness::{JoltWitnessOracle, RowSource};
    use rand_chacha::ChaCha8Rng;
    use rand_core::{RngCore, SeedableRng};

    use super::*;
    use crate::metal::MetalConfig;
    use crate::optimized::parity::run_lockstep;
    use crate::optimized::registers_read_write::test_support::{structured_fixture, TraceFixture};
    use crate::optimized::spartan_outer::prepare_metal_spartan_outer_stage1_owner_witness_rows;
    use crate::reference::ReferenceBackend;

    fn point(len: usize, seed: u64) -> Vec<AkitaField> {
        let mut random = ChaCha8Rng::seed_from_u64(seed);
        (0..len)
            .map(|_| {
                AkitaField::from_u128(
                    u128::from(random.next_u64()) | (u128::from(random.next_u64()) << 64),
                )
            })
            .collect()
    }

    #[test]
    fn native_trace_shapes_route_to_metal() {
        let config = MetalConfig::production();
        for log_t in [22, 24] {
            assert!(
                (1usize << log_t) >= config.registers_read_write.trace_cutoff_elements,
                "T{log_t} must dispatch the register read/write cycle sequence",
            );
        }
    }

    #[test]
    fn random_register_traces_match_cpu_through_all_rounds() {
        for seed in [3, 31, 101, 509] {
            check_fixture_seed(random_fixture(seed), true, false, seed);
        }
    }

    #[test]
    fn gpu_operand_claims_match_cpu_for_random_padded_traces() {
        for seed in [7, 37, 127] {
            check_fixture_with_operand_claims(random_fixture(seed), true, false, seed, true);
        }
    }

    #[test]
    fn native_76_registers_select_metal_and_match_random_cpu() {
        for (log_t, seed) in [(8, 17), (9, 43)] {
            check_fixture_at_geometry(
                wide_random_fixture(76, seed),
                true,
                false,
                seed,
                false,
                log_t,
                false,
            );
        }
    }

    #[test]
    fn native_128_registers_select_metal_and_match_random_cpu() {
        for (log_t, seed) in [(8, 23), (9, 47)] {
            check_fixture_at_geometry(
                wide_random_fixture(128, seed),
                true,
                false,
                seed,
                false,
                log_t,
                false,
            );
        }
    }

    #[test]
    fn banked_gpu_claims_cover_zero_gamma_binary_binds_and_empty_bank() {
        check_fixture_at_geometry(random_fixture(53), true, false, 53, true, 8, true);
        check_fixture_at_geometry(wide_random_fixture(128, 59), true, false, 59, true, 9, true);
    }

    fn wide_random_fixture(active: u8, seed: u64) -> TraceFixture {
        let mut fixture = TraceFixture::new();
        for register in 0..active {
            fixture.op(Some(register), None, None);
        }
        let mut random = ChaCha8Rng::seed_from_u64(seed);
        for cycle in 0..123 {
            if cycle % 11 == 0 {
                fixture.noop();
                continue;
            }
            let mut operand = || {
                let value = random.next_u32();
                (value % 7 != 0).then_some((value % u32::from(active)) as u8)
            };
            fixture.op(operand(), operand(), operand());
        }
        fixture
    }

    fn random_fixture(seed: u64) -> TraceFixture {
        let mut random = ChaCha8Rng::seed_from_u64(seed);
        let mut fixture = TraceFixture::new();
        for cycle in 0..173 {
            if cycle % 11 == 0 {
                fixture.noop();
                continue;
            }
            let mut operand = || {
                let value = random.next_u32();
                (value % 5 != 0).then_some((value % 32 + 96) as u8)
            };
            fixture.op(operand(), operand(), operand());
        }
        fixture
    }

    #[test]
    fn adapter_selects_metal_and_matches_reference() {
        check_fixture(structured_fixture(1 << 8), true, false);
    }

    #[test]
    fn adapter_handles_register_capacity_boundary() {
        for active in [64u8, 65, 72, 128] {
            check_fixture(capacity_fixture(active), true, false);
        }
    }

    #[test]
    fn adapter_consumes_stage1_capacity_marker_for_native_banks() {
        check_fixture(capacity_fixture(72), true, true);
    }

    fn capacity_fixture(active: u8) -> TraceFixture {
        let mut fixture = TraceFixture::new();
        let offset = if active == 64 { 64 } else { 0 };
        for register in 0..active {
            fixture.op(
                Some(offset + register),
                Some(offset + (register + 1) % active),
                Some(offset + (register + 2) % active),
            );
        }
        fixture
    }

    fn check_fixture(fixture: TraceFixture, uses_metal: bool, prepare_stage1: bool) {
        check_fixture_seed(fixture, uses_metal, prepare_stage1, 19);
    }

    fn check_fixture_seed(
        fixture: TraceFixture,
        uses_metal: bool,
        prepare_stage1: bool,
        seed: u64,
    ) {
        check_fixture_with_operand_claims(fixture, uses_metal, prepare_stage1, seed, false);
    }

    fn check_fixture_with_operand_claims(
        fixture: TraceFixture,
        uses_metal: bool,
        prepare_stage1: bool,
        seed: u64,
        gpu_operand_claims: bool,
    ) {
        check_fixture_at_geometry(
            fixture,
            uses_metal,
            prepare_stage1,
            seed,
            gpu_operand_claims,
            8,
            false,
        );
    }

    fn check_fixture_at_geometry(
        fixture: TraceFixture,
        uses_metal: bool,
        prepare_stage1: bool,
        seed: u64,
        gpu_operand_claims: bool,
        log_t: usize,
        edge_challenges: bool,
    ) {
        fixture.with_plane(log_t, |witness| {
            let relation = RegistersReadWriteChecking::new(ReadWriteDimensions::new(
                log_t,
                REGISTER_ADDRESS_BITS,
                log_t,
                0,
            ));
            let r_cycle = point(log_t, seed);
            let evaluate = |polynomial: JoltVirtualPolynomial| {
                let table = JoltWitnessOracle::<AkitaField>::oracle_table(
                    witness,
                    JoltPolynomialId::Virtual(polynomial),
                )
                .unwrap();
                Polynomial::new(table).evaluate(&r_cycle)
            };
            let claims = RegistersReadWriteInputClaims {
                rd_write_value: evaluate(JoltVirtualPolynomial::RdWriteValue),
                rs1_value: evaluate(JoltVirtualPolynomial::Rs1Value),
                rs2_value: evaluate(JoltVirtualPolynomial::Rs2Value),
            };
            let points = RegistersReadWriteInputClaims {
                rd_write_value: r_cycle.clone(),
                rs1_value: r_cycle.clone(),
                rs2_value: r_cycle,
            };
            let challenges = RegistersReadWriteChallenges {
                gamma: if edge_challenges {
                    AkitaField::zero()
                } else {
                    point(1, seed ^ 31)[0]
                },
            };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };
            let input_claim = claims.rd_write_value
                + challenges.gamma * claims.rs1_value
                + challenges.gamma * challenges.gamma * claims.rs2_value;
            let round_challenges = if edge_challenges {
                (0..log_t + REGISTER_ADDRESS_BITS)
                    .map(|index| AkitaField::from_u64((index % 2) as u64))
                    .collect()
            } else {
                point(log_t + REGISTER_ADDRESS_BITS, seed ^ 0x65)
            };
            let mut expected = ReferenceBackend
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let metal = MetalBackend::new(MetalConfig {
                registers_read_write: RegistersReadWriteMetalConfig {
                    trace_cutoff_elements: 2,
                },
                ..Default::default()
            })
            .unwrap();
            let mut session = ProofSession::default();
            if prepare_stage1 {
                let (_, ready) = prepare_metal_spartan_outer_stage1_owner_witness_rows(
                    &metal.context,
                    witness,
                    1 << log_t,
                    false,
                    true,
                    false,
                    false,
                    false,
                )
                .unwrap();
                crate::metal::spartan_outer::publish_instruction_read_raf_stage1(
                    &metal.context,
                    &mut session,
                    ready,
                )
                .unwrap();
                assert_eq!(
                    session
                        .state::<RegisterCapacityExceeded>()
                        .unwrap()
                        .active_registers,
                    72
                );
                assert!(session.state::<RegistersReadWriteStage1Source>().is_none());
                assert!(session
                    .state::<PendingRegistersReadWriteStage1Pipelines>()
                    .is_none());
            }
            let mut actual: RegistersReadWriteKernelBox = if gpu_operand_claims {
                let access = witness.random_access().unwrap();
                let rows = Arc::new(
                    AlignedPackedRegisterRows::collect_native(&access, access.physical_rows())
                        .unwrap(),
                );
                let sequence = [0, 64]
                    .into_iter()
                    .map(|base| {
                        metal
                            .context
                            .prepare_registers_read_write_cycle_sequence_partition(
                                Arc::clone(&rows),
                                log_t,
                                challenges.gamma,
                                base,
                            )
                            .unwrap()
                    })
                    .collect();
                Box::new(MetalRegistersReadWriteKernel {
                    sequence,
                    source_owner: None,
                    continuation: None,
                    gruen: GruenSplitEqPolynomial::new(
                        &points.rd_write_value,
                        BindingOrder::LowToHigh,
                    ),
                    r_cycle: points.rd_write_value.clone(),
                    cycle_challenges: Vec::with_capacity(log_t),
                    bound_challenges: Vec::with_capacity(log_t + REGISTER_ADDRESS_BITS),
                    gamma: challenges.gamma,
                    log_t,
                    log_k: REGISTER_ADDRESS_BITS,
                    use_metal_operand_claims: true,
                    next_round: 0,
                    finished: false,
                })
            } else {
                metal.prepare(&mut session, witness, inputs()).unwrap()
            };
            assert!(session.state::<RegisterCapacityExceeded>().is_none());
            assert_eq!(
                metal.registers_read_write_metal_sequences(),
                usize::from(uses_metal && !gpu_operand_claims)
            );
            if !gpu_operand_claims {
                assert!(session.state::<SharedRdIndices>().is_some());
            }
            assert_eq!(
                actual.execution_domain(),
                if uses_metal {
                    RoundExecutionDomain::Accelerator
                } else {
                    RoundExecutionDomain::Host
                },
            );
            run_lockstep(
                expected.as_mut(),
                actual.as_mut(),
                input_claim,
                &round_challenges,
            );
            assert_eq!(actual.execution_domain(), RoundExecutionDomain::Host);
            assert_eq!(
                expected.output_claims(&claims).unwrap(),
                actual.output_claims(&claims).unwrap()
            );
            let output_points = relation
                .derive_opening_points(&round_challenges, &points)
                .unwrap();
            actual
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        });
    }
}
