use jolt_claims::protocols::jolt::relations::instruction::InstructionRaVirtualizationChallenges;
use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::Zero as _;
use jolt_sumcheck::{ProveRounds, RoundExecutionDomain, SumcheckError};
use jolt_verifier::stages::relations::{
    SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::solinas::{
    InstructionRaSequence, InstructionRaSequenceConfig, InstructionRaSequenceStorage,
    ResidentLookupIndexPlane,
};
use crate::optimized::instruction_ra_virtualization::{
    prepare_instruction_ra_from_initialization, prepare_instruction_ra_initialization,
    OptimizedInstructionRaVirtualizationKernel,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionRaVirtualizationMetalConfig {
    pub trace_cutoff_elements: usize,
    pub cutoff_elements: usize,
    pub dispatch: InstructionRaSequenceConfig,
}

impl Default for InstructionRaVirtualizationMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 22,
            // Two keeps every round, the final bind and terminal extraction
            // on Metal. Larger values retain the optional bounded CPU tail.
            cutoff_elements: 2,
            dispatch: InstructionRaSequenceConfig::default(),
        }
    }
}

impl PrepareKernel<AkitaField, InstructionRaVirtualization<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, InstructionRaVirtualization<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = InstructionRaVirtualization<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let dimensions = inputs.relation.dimensions();
        let chunk_bits = inputs.relation.committed_chunk_bits();
        let trace_elements = 1usize.checked_shl(dimensions.log_t() as u32).ok_or(
            KernelError::InvariantViolation {
                reason: "instruction RA trace length overflow",
            },
        )?;
        let mut dispatch = self.config.instruction_ra_virtualization.dispatch;
        if chunk_bits == 4 && dispatch.materialize_width.elements() < 64 {
            dispatch.materialize_width = super::solinas::InstructionRaMaterializeWidth::W64;
        }
        let initialization = prepare_instruction_ra_initialization(inputs)?;
        let _span = tracing::info_span!(
            "MetalInstructionRaVirtualization::admission",
            rows = trace_elements,
            groups = dimensions.num_virtual_ra_polys(),
            factors_per_group = dimensions.num_committed_per_virtual(),
            chunk_bits,
            supported = initialization.supports_metal_sequence(),
        )
        .entered();
        if trace_elements
            < self
                .config
                .instruction_ra_virtualization
                .trace_cutoff_elements
            || trace_elements < 2 * dispatch.materialize_width.elements()
            || !initialization.supports_metal_sequence()
        {
            let _ = session.take::<InstructionRaSequenceStorage>();
            let _ = session.take::<ResidentLookupIndexPlane>();
            return Ok(Box::new(prepare_instruction_ra_from_initialization(
                session,
                witness,
                initialization,
            )?));
        }

        let plane =
            session
                .take::<ResidentLookupIndexPlane>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Metal Instruction RA requires the resident stage-5 lookup plane",
                })?;
        if plane.len() != trace_elements {
            return Err(KernelError::TableSizeMismatch {
                table: "resident Metal lookup-index plane".to_owned(),
                expected: trace_elements,
                got: plane.len(),
            });
        }
        if plane.device_registry_id() != self.context.device_registry_id() {
            return Err(KernelError::InvariantViolation {
                reason: "resident Metal lookup-index plane belongs to another device",
            });
        }

        let (cpu, chunk_tables) = initialization.into_offloaded()?;
        let (e_in_capacity, e_out_capacity) = {
            let (e_in, e_out) = cpu.metal_weights()?;
            (e_in.len(), e_out.len())
        };
        let sequence = {
            let storage = session
                .take::<InstructionRaSequenceStorage>()
                .filter(|storage| {
                    storage.matches(
                        &self.context,
                        trace_elements,
                        e_in_capacity,
                        e_out_capacity,
                        dispatch,
                        chunk_tables.len(),
                    )
                });
            let _span = tracing::info_span!(
                "MetalInstructionRaVirtualization::sequence_prepare",
                preallocated = storage.is_some()
            )
            .entered();
            match storage {
                Some(storage) => storage.attach(plane, &chunk_tables),
                _ => self.context.prepare_instruction_ra_sequence_with_plane(
                    plane,
                    &chunk_tables,
                    e_in_capacity,
                    e_out_capacity,
                    dispatch,
                ),
            }
            .map_err(|error| metal_error(error.to_string()))?
        };
        Ok(Box::new(MetalInstructionRaVirtualizationKernel::new(
            cpu,
            sequence,
            self.config.instruction_ra_virtualization.cutoff_elements,
        )?))
    }
}

struct MetalInstructionRaVirtualizationKernel {
    cpu: OptimizedInstructionRaVirtualizationKernel<AkitaField>,
    sequence: Option<InstructionRaSequence>,
    host_tail: Vec<AkitaField>,
    cutoff_elements: usize,
    next_round: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalInstructionRaVirtualizationKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("cpu"), &self.cpu);
        if let Some(sequence) = &self.sequence {
            visitor.visit_field(allocative::Key::new("sequence"), sequence);
        }
        visitor.visit_simple(
            allocative::Key::new("host_tail"),
            crate::backend::vec_heap_bytes(&self.host_tail),
        );
        visitor.exit();
    }
}

impl MetalInstructionRaVirtualizationKernel {
    fn new(
        cpu: OptimizedInstructionRaVirtualizationKernel<AkitaField>,
        mut sequence: InstructionRaSequence,
        cutoff_elements: usize,
    ) -> Result<Self, SumcheckError<AkitaField>> {
        if cutoff_elements < 2 || !cutoff_elements.is_power_of_two() {
            return Err(metal_error(
                "instruction RA CPU cutoff must be a power of two at least two",
            ));
        }
        let host_elements = cpu
            .metal_num_polys()
            .checked_mul(cutoff_elements)
            .ok_or_else(|| metal_error("instruction RA host-tail capacity overflow"))?;
        let (e_in, e_out) = cpu.metal_weights()?;
        sequence
            .initialize_weights(e_in, e_out)
            .map_err(|error| metal_error(error.to_string()))?;
        Ok(Self {
            cpu,
            sequence: Some(sequence),
            host_tail: vec![AkitaField::zero(); host_elements],
            cutoff_elements,
            next_round: 0,
        })
    }

    fn should_restore_cpu_tail(&self) -> bool {
        self.cutoff_elements > 2
            && self.sequence.as_ref().is_some_and(|sequence| {
                sequence.is_dense() && sequence.current_elements() <= self.cutoff_elements
            })
    }

    fn restore_cpu_tail(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let _span = tracing::info_span!("MetalInstructionRaVirtualization::readback").entered();
        let sequence = self.sequence.take().ok_or_else(|| {
            metal_error("instruction RA sequence disappeared before dense readback")
        })?;
        if !sequence.is_dense() {
            return Err(metal_error(
                "instruction RA CPU handoff requires resident dense tables",
            ));
        }
        let elements = sequence.current_elements();
        let output_len = self
            .cpu
            .metal_num_polys()
            .checked_mul(elements)
            .ok_or_else(|| metal_error("instruction RA readback length overflow"))?;
        if output_len > self.host_tail.len() {
            return Err(metal_error(
                "instruction RA readback exceeds the preallocated CPU tail",
            ));
        }
        sequence
            .read_current_tables(&mut self.host_tail[..output_len])
            .map_err(|error| metal_error(error.to_string()))?;
        self.cpu
            .metal_restore_dense(&self.host_tail[..output_len], elements)
    }
}

impl ProveRounds<AkitaField> for MetalInstructionRaVirtualizationKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn execution_domain(&self) -> RoundExecutionDomain {
        if self.sequence.is_some() && !self.should_restore_cpu_tail() {
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
        if round != self.next_round {
            return Err(metal_error("instruction RA rounds arrived out of order"));
        }
        self.cpu.metal_validate_round(bind.is_some(), round)?;
        if self.should_restore_cpu_tail() {
            self.restore_cpu_tail()?;
            let message = self.cpu.prove_round(bind, round, previous_claim)?;
            self.next_round += 1;
            return Ok(message);
        }

        if self.sequence.is_some() {
            let sequence = self
                .sequence
                .as_ref()
                .ok_or_else(|| metal_error("instruction RA sequence disappeared before round"))?;
            let _span = if sequence.is_dense() {
                tracing::info_span!(
                    "MetalInstructionRaVirtualization::dense_round",
                    elements = sequence.current_elements()
                )
                .entered()
            } else {
                tracing::info_span!(
                    "MetalInstructionRaVirtualization::lazy_round",
                    elements = sequence.current_elements(),
                    branch_width = sequence.branch_width()
                )
                .entered()
            };
            let message = if let Some(challenge) = bind {
                let message = self
                    .sequence
                    .as_mut()
                    .ok_or_else(|| metal_error("instruction RA sequence disappeared before bind"))?
                    .bind_and_message_resident(challenge)
                    .map_err(|error| metal_error(error.to_string()))?;
                self.cpu.metal_bind_offloaded(challenge)?;
                message
            } else {
                self.sequence
                    .as_mut()
                    .ok_or_else(|| {
                        metal_error("instruction RA sequence disappeared before message")
                    })?
                    .message_resident()
                    .map_err(|error| metal_error(error.to_string()))?
            };
            let message = self.cpu.metal_message(message, previous_claim)?;
            self.next_round += 1;
            return Ok(message);
        }

        let message = self.cpu.prove_round(bind, round, previous_claim)?;
        self.next_round += 1;
        Ok(message)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        if self.next_round != self.num_rounds() {
            return Err(metal_error(
                "instruction RA terminal bind arrived before every round",
            ));
        }
        self.cpu.metal_validate_final_bind()?;
        if self.sequence.is_some() && self.cutoff_elements == 2 {
            let _span =
                tracing::info_span!("MetalInstructionRaVirtualization::terminal_bind").entered();
            let claims = self
                .sequence
                .as_mut()
                .ok_or_else(|| {
                    metal_error("instruction RA sequence disappeared before terminal bind")
                })?
                .bind_terminal(bind)
                .map_err(|error| metal_error(error.to_string()))?;
            self.cpu.metal_bind_offloaded(bind)?;
            self.cpu.metal_restore_dense(&claims, 1)?;
            let _ = self.sequence.take();
            return Ok(());
        }
        if self.sequence.is_some() {
            self.restore_cpu_tail()?;
        }
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalInstructionRaVirtualizationKernel {
    type Relation = InstructionRaVirtualization<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        self.cpu.output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &InstructionRaVirtualizationChallenges<AkitaField>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        self.cpu
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "CPU and actual Metal parity fixtures")]
mod tests {
    use std::{num::NonZeroUsize, sync::Arc};

    use jolt_claims::protocols::jolt::geometry::instruction::InstructionRaVirtualizationDimensions;
    use jolt_field::{One as _, Ring as _};
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_witness::witnesses::FusedInc;

    use super::*;
    use crate::metal::solinas::{InstructionRaMaterializeWidth, SolinasMetal};
    use crate::optimized::instruction_read_raf::InstructionCycleRow;
    use crate::reference::views::eq_table;

    fn field(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn random_field(state: &mut u64) -> AkitaField {
        AkitaField::from_u128((u128::from(splitmix(state)) << 64) | u128::from(splitmix(state)))
    }

    fn gpu_cpu_parity(
        log_t: usize,
        groups: usize,
        bits: usize,
        gamma: AkitaField,
        reuse: bool,
        cutoff: usize,
        seed: Option<u64>,
    ) {
        let mut state = seed.unwrap_or(0);
        let count = groups * 4;
        let lookups: Vec<u128> = (0..1 << log_t)
            .map(|row| match row {
                0 => 0,
                1 => u128::MAX,
                2 => 15u128 << 64, // Nibble factor 15, high limb's low nibble.
                3 => 15u128 << 60, // Nibble factor 16, low limb's high nibble.
                _ if seed.is_some() => {
                    (u128::from(splitmix(&mut state)) << 64) | u128::from(splitmix(&mut state))
                }
                _ => {
                    let high = (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    let low = (row as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    (u128::from(high) << 64) | u128::from(low)
                }
            })
            .collect();
        let rows = Arc::new(
            lookups
                .iter()
                .map(|index| {
                    InstructionCycleRow::new(*index, None, false, 0, None, FusedInc::default())
                })
                .collect(),
        );
        let address: Vec<_> = (0..128)
            .map(|index| {
                if seed.is_some() {
                    random_field(&mut state)
                } else {
                    field(31 + index as u64 * 11)
                }
            })
            .collect();
        let cycle: Vec<_> = (0..log_t)
            .map(|index| {
                if seed.is_some() {
                    random_field(&mut state)
                } else {
                    field(113 + index as u64 * 17)
                }
            })
            .collect();
        let relation = InstructionRaVirtualization::new(
            InstructionRaVirtualizationDimensions::new(
                log_t,
                NonZeroUsize::new(groups).unwrap(),
                NonZeroUsize::new(4).unwrap(),
            )
            .unwrap(),
            address.clone(),
            cycle.clone(),
            bits,
        );
        let claims = SumcheckInputClaims::<AkitaField, InstructionRaVirtualization<AkitaField>> {
            instruction_ra: vec![AkitaField::zero(); groups],
        };
        let points = SumcheckInputPoints::<AkitaField, InstructionRaVirtualization<AkitaField>> {
            instruction_ra: vec![Vec::new(); groups],
        };
        let challenges = InstructionRaVirtualizationChallenges { gamma };
        let initialization = prepare_instruction_ra_initialization(ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &challenges,
        })
        .unwrap();
        assert!(initialization.supports_metal_sequence());
        let (offloaded, chunk_tables) = initialization.into_offloaded().unwrap();
        let (e_in, e_out) = offloaded.metal_weights().unwrap();
        let context = SolinasMetal::for_akita().unwrap();
        let sequence = context
            .prepare_instruction_ra_test_sequence(
                &lookups,
                &chunk_tables,
                e_in.len(),
                e_out.len(),
                InstructionRaSequenceConfig {
                    materialize_width: InstructionRaMaterializeWidth::W64,
                    reuse_inverse_for_dense: reuse,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut actual =
            MetalInstructionRaVirtualizationKernel::new(offloaded, sequence, cutoff).unwrap();
        let mut expected = OptimizedInstructionRaVirtualizationKernel::new(
            log_t, groups, 4, &address, &cycle, bits, rows, gamma,
        )
        .unwrap();

        let eq = eq_table(&cycle);
        let mask = (1u128 << bits) - 1;
        let mut claim = AkitaField::zero();
        for (row, lookup) in lookups.iter().enumerate() {
            let mut sum = AkitaField::zero();
            for group in 0..groups {
                let mut product = AkitaField::one();
                for local in 0..4 {
                    let factor = group * 4 + local;
                    let selector = ((lookup >> ((count - 1 - factor) * bits)) & mask) as usize;
                    product *= chunk_tables[factor * (1 << bits) + selector];
                }
                sum += product;
            }
            claim += eq[row] * sum;
        }
        let binds: Vec<_> = (0..log_t)
            .map(|round| match round {
                1 => AkitaField::zero(),
                2 => AkitaField::one(),
                _ if seed.is_some() => random_field(&mut state),
                _ => field(997 + 37 * round as u64),
            })
            .collect();
        let mut saw_dense = false;
        let mut saw_cpu_tail = false;
        assert!(actual.prove_round(Some(field(7)), 0, claim).is_err());
        assert!(actual.prove_round(None, 1, claim).is_err());
        assert!(actual.finish_rounds(field(7)).is_err());
        assert_eq!(actual.next_round, 0);
        assert_eq!(actual.sequence.as_ref().unwrap().branch_width(), 1);
        for round in 0..log_t {
            if cutoff == 2 {
                assert_eq!(actual.execution_domain(), RoundExecutionDomain::Accelerator);
            }
            let bind = round.checked_sub(1).map(|previous| binds[previous]);
            let oracle = expected.prove_round(bind, round, claim).unwrap();
            let gpu = actual.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                gpu.coefficients(),
                oracle.coefficients(),
                "round {round}, G{groups}, log_t {log_t}, inverse reuse {reuse}"
            );
            saw_dense |= actual
                .sequence
                .as_ref()
                .is_some_and(InstructionRaSequence::is_dense);
            saw_cpu_tail |= actual.sequence.is_none();
            claim = oracle.evaluate(binds[round]);
            assert!(actual.prove_round(bind, round, claim).is_err());
            assert_eq!(actual.next_round, round + 1);
        }
        assert!(saw_dense);
        assert_eq!(saw_cpu_tail, cutoff > 2);
        if cutoff == 2 {
            assert_eq!(actual.sequence.as_ref().unwrap().current_elements(), 2);
        }
        expected.finish_rounds(binds[log_t - 1]).unwrap();
        actual.finish_rounds(binds[log_t - 1]).unwrap();
        assert!(actual.finish_rounds(binds[log_t - 1]).is_err());
        let oracle = expected.output_claims(&claims).unwrap();
        let gpu = actual.output_claims(&claims).unwrap();
        assert_eq!(gpu.committed_instruction_ra.len(), count);
        assert_eq!(
            gpu.committed_instruction_ra,
            oracle.committed_instruction_ra
        );
        let output_points = relation.derive_opening_points(&binds, &points).unwrap();
        actual
            .validate_derived_tables(&relation, &points, &output_points, &challenges)
            .unwrap();
    }

    #[test]
    fn g8_nibble_gpu_rounds_and_terminal_claims_match_cpu() {
        for log_t in [9, 10] {
            for gamma in [AkitaField::one(), field(271)] {
                gpu_cpu_parity(log_t, 8, 4, gamma, false, 4, None);
            }
        }
    }

    #[test]
    fn g8_nibble_inverse_reuse_and_g4_byte_rounds_match_cpu() {
        gpu_cpu_parity(10, 8, 4, field(271), true, 4, None);
        gpu_cpu_parity(9, 4, 8, field(271), false, 4, None);
        gpu_cpu_parity(10, 4, 8, AkitaField::one(), true, 4, None);
    }

    #[test]
    fn random_resident_g8_and_g4_all_rounds_and_terminal_claims_match_cpu() {
        for (log_t, groups, bits, reuse, seed) in [
            (7, 8, 4, false, 0x4a6f_6c74),
            (9, 8, 4, true, 0x5a17_0022),
            (10, 8, 4, false, 0x5a17_0024),
            (11, 4, 8, true, 0x5a17_0026),
        ] {
            let mut state = seed;
            let gamma = random_field(&mut state);
            gpu_cpu_parity(log_t, groups, bits, gamma, reuse, 2, Some(seed));
        }
    }

    #[test]
    fn native_g8_shape_keeps_every_round_and_terminal_on_metal() {
        let log_t: usize = 8;
        let groups = 8;
        let mut state = 0x5a17_0024u64;
        let mut next = || {
            state = state.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
            let high = state;
            state = state.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
            (u128::from(high) << 64) | u128::from(state)
        };
        let lookups: Vec<_> = (0..1 << log_t).map(|_| next()).collect();
        let address: Vec<_> = (0..128).map(|_| AkitaField::from_u128(next())).collect();
        let cycle: Vec<_> = (0..log_t).map(|_| AkitaField::from_u128(next())).collect();
        let binds: Vec<_> = (0..log_t).map(|_| AkitaField::from_u128(next())).collect();
        let rows = Arc::new(
            lookups
                .iter()
                .map(|lookup| {
                    InstructionCycleRow::new(*lookup, None, false, 0, None, FusedInc::default())
                })
                .collect(),
        );
        let relation = InstructionRaVirtualization::new(
            InstructionRaVirtualizationDimensions::new(
                log_t,
                NonZeroUsize::new(groups).unwrap(),
                NonZeroUsize::new(4).unwrap(),
            )
            .unwrap(),
            address.clone(),
            cycle.clone(),
            4,
        );
        let claims = SumcheckInputClaims::<AkitaField, InstructionRaVirtualization<AkitaField>> {
            instruction_ra: vec![AkitaField::zero(); groups],
        };
        let points = SumcheckInputPoints::<AkitaField, InstructionRaVirtualization<AkitaField>> {
            instruction_ra: vec![Vec::new(); groups],
        };
        let challenges = InstructionRaVirtualizationChallenges { gamma: field(271) };
        let initialization = prepare_instruction_ra_initialization(ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &challenges,
        })
        .unwrap();
        let (offloaded, tables) = initialization.into_offloaded().unwrap();
        let (e_in, e_out) = offloaded.metal_weights().unwrap();
        let sequence = SolinasMetal::for_akita()
            .unwrap()
            .prepare_instruction_ra_test_sequence(
                &lookups,
                &tables,
                e_in.len(),
                e_out.len(),
                InstructionRaSequenceConfig {
                    materialize_width: InstructionRaMaterializeWidth::W64,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut actual =
            MetalInstructionRaVirtualizationKernel::new(offloaded, sequence, 2).unwrap();
        let mut expected = OptimizedInstructionRaVirtualizationKernel::new(
            log_t,
            groups,
            4,
            &address,
            &cycle,
            4,
            rows,
            challenges.gamma,
        )
        .unwrap();
        let eq = eq_table(&cycle);
        let mut claim = AkitaField::zero();
        for (row, lookup) in lookups.iter().enumerate() {
            for group in 0..groups {
                let mut product = eq[row];
                for local in 0..4 {
                    let factor = group * 4 + local;
                    let selector = ((lookup >> ((31 - factor) * 4)) & 15) as usize;
                    product *= tables[factor * 16 + selector];
                }
                claim += product;
            }
        }
        for round in 0..log_t {
            assert_eq!(
                actual.execution_domain(),
                RoundExecutionDomain::Accelerator,
                "before round {round}"
            );
            let bind = round.checked_sub(1).map(|previous| binds[previous]);
            let oracle = expected.prove_round(bind, round, claim).unwrap();
            let message = actual.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                message.coefficients(),
                oracle.coefficients(),
                "round {round}"
            );
            assert_eq!(
                actual.execution_domain(),
                RoundExecutionDomain::Accelerator,
                "after round {round}"
            );
            claim = oracle.evaluate(binds[round]);
        }
        expected.finish_rounds(binds[log_t - 1]).unwrap();
        actual.finish_rounds(binds[log_t - 1]).unwrap();
        assert_eq!(
            actual
                .output_claims(&claims)
                .unwrap()
                .committed_instruction_ra,
            expected
                .output_claims(&claims)
                .unwrap()
                .committed_instruction_ra,
        );
        let output_points = relation.derive_opening_points(&binds, &points).unwrap();
        actual
            .validate_derived_tables(&relation, &points, &output_points, &challenges)
            .unwrap();
    }

    #[test]
    fn g8_nibble_zero_batching_gamma_retains_cpu_rejection() {
        let rows = Arc::new(vec![
            InstructionCycleRow::new(
                0,
                None,
                false,
                0,
                None,
                FusedInc::default()
            );
            128
        ]);
        let result = OptimizedInstructionRaVirtualizationKernel::new(
            7,
            8,
            4,
            &vec![field(31); 128],
            &vec![field(113); 7],
            4,
            rows,
            AkitaField::zero(),
        );
        assert!(matches!(
            result,
            Err(KernelError::InvariantViolation {
                reason: "instruction RA batching gamma must be invertible"
            })
        ));
    }

    #[test]
    fn production_nibble_cutoff_admits_t22_and_t24_with_resident_terminal() {
        let config = InstructionRaVirtualizationMetalConfig::default();
        assert_eq!(config.trace_cutoff_elements, 1 << 22);
        assert!(1 << 21 < config.trace_cutoff_elements);
        assert!(1 << 24 >= config.trace_cutoff_elements);
        assert_eq!(config.cutoff_elements, 2);
    }
}
