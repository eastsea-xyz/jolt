//! Unregistered cycle adapter. Integration hooks are supplied as a temporary
//! patch: they expose already-derived pending metadata and preserve CPU claims.

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_sumcheck::{ProveRounds, RoundExecutionDomain, SumcheckError};

use super::solinas::{Product9ResidentSource, Product9Sequence, SolinasMetal};
use crate::optimized::instruction_read_raf::OptimizedInstructionReadRafKernel;

pub(crate) struct Product9CycleAdapter {
    sequence: Option<Product9Sequence>,
    host_tail: Option<[Vec<AkitaField>; 9]>,
    cutoff: usize,
}

impl Product9CycleAdapter {
    pub(crate) fn prepare(
        cpu: &mut OptimizedInstructionReadRafKernel<AkitaField>,
        context: SolinasMetal,
        source: Product9ResidentSource,
        cutoff: usize,
    ) -> Result<Self, SumcheckError<AkitaField>> {
        if !(2..=1 << 16).contains(&cutoff) || !cutoff.is_power_of_two() {
            return Err(error("Product9 CPU tail cutoff outside bounded range"));
        }
        let rows = source.rows();
        let input = cpu.metal_product9_cycle_input()?;
        let (e_in, e_out) = cpu.metal_cycle_weights()?;
        let sequence = Product9Sequence::new(context, source, input, e_in.len(), e_out.len())
            .map_err(error)?;
        let _span = tracing::info_span!(
            "MetalInstructionReadRaf::product9_prepare",
            rows,
            factors = 9usize,
            materialize_width = 64usize,
            device_owned_bytes = sequence.owned_bytes(),
            cpu_tail_elements = cutoff,
            cpu_tail_bytes = 9 * cutoff * std::mem::size_of::<AkitaField>(),
        )
        .entered();
        // Mark offloaded only after shape/device/allocation admission succeeds.
        // This preserves the pending CPU state if preparation itself fails.
        cpu.metal_begin_product9_cycle()?;
        Ok(Self {
            sequence: Some(sequence),
            host_tail: Some(std::array::from_fn(|_| vec![AkitaField::default(); cutoff])),
            cutoff,
        })
    }

    pub(crate) fn execution_domain(&self) -> RoundExecutionDomain {
        if self.sequence.as_ref().is_some_and(|sequence| {
            !sequence.is_dense() || sequence.current_elements() > self.cutoff
        }) {
            RoundExecutionDomain::Accelerator
        } else {
            RoundExecutionDomain::Host
        }
    }

    pub(crate) fn prove_round(
        &mut self,
        cpu: &mut OptimizedInstructionReadRafKernel<AkitaField>,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<jolt_poly::UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if self.sequence.as_ref().is_some_and(|sequence| {
            sequence.is_dense() && sequence.current_elements() <= self.cutoff
        }) {
            self.restore(cpu)?;
        }
        let Some(sequence) = self.sequence.as_mut() else {
            return cpu.prove_round(bind, round, previous_claim);
        };
        if let Some(challenge) = bind {
            cpu.metal_bind_offloaded(challenge)?;
        }
        let (e_in, e_out) = cpu.metal_cycle_weights()?;
        let q = match bind {
            None => sequence.message(e_in, e_out),
            Some(challenge) => sequence.bind_and_message(challenge, e_in, e_out),
        }
        .map_err(error)?;
        cpu.metal_product9_cycle_message(&q, previous_claim)
    }

    pub(crate) fn restore(
        &mut self,
        cpu: &mut OptimizedInstructionReadRafKernel<AkitaField>,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let Some(sequence) = self.sequence.take() else {
            return Ok(());
        };
        let mut tables = self
            .host_tail
            .take()
            .ok_or_else(|| error("Product9 tail already consumed"))?;
        sequence
            .read_current_factor_tables(&mut tables)
            .map_err(error)?;
        // The nine Vecs move into the CPU kernel; no second full tail copy.
        cpu.metal_restore_product9_dense(tables)
    }
}

fn error(message: impl ToString) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal-product9",
        message: message.to_string(),
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "actual G8 member parity fixture")]
mod tests {
    use std::{num::NonZeroUsize, sync::Arc};

    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafInputClaims;
    use jolt_field::{One as _, Ring as _, Zero as _};
    use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN};
    use jolt_witness::witnesses::FusedInc;

    use super::*;
    use crate::reference::views::eq_table;
    use crate::{optimized::instruction_read_raf::InstructionCycleRow, SumcheckKernel};

    fn field(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    #[test]
    fn g8_instruction_read_raf_gpu_cycle_matches_entire_cpu_member() {
        for log_t in [9, 10] {
            for gamma in [AkitaField::zero(), AkitaField::one(), field(313)] {
                member_parity(log_t, gamma);
            }
        }
    }

    fn member_parity(log_t: usize, gamma: AkitaField) {
        let rows = 1 << log_t;
        let lookups: Vec<u128> = (0..rows)
            .map(|row| match row {
                0 => 0,
                1 => u128::MAX,
                2 => 255u128 << 64,
                3 => 255u128 << 56,
                _ => {
                    let high = (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    let low = (row as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    (u128::from(high) << 64) | u128::from(low)
                }
            })
            .collect();
        let tables: Vec<_> = LookupTableKind::<XLEN>::iter().collect();
        let claims: Vec<u8> = (0..rows)
            .map(|row| {
                let table = if row % 5 == 0 {
                    0
                } else {
                    (row % tables.len() + 1) as u8
                };
                table | if row % 3 == 0 { 0x80 } else { 0 }
            })
            .collect();
        let packed: Vec<_> = lookups
            .iter()
            .zip(&claims)
            .map(|(lookup, claim)| {
                let table = usize::from(claim & 0x3f);
                InstructionCycleRow::new(
                    *lookup,
                    (table != 0).then(|| table - 1),
                    claim & 0x80 != 0,
                    0,
                    None,
                    FusedInc::default(),
                )
            })
            .collect();
        let dimensions =
            InstructionReadRafDimensions::new(log_t, 128, NonZeroUsize::new(8).unwrap());
        let reduction: Vec<_> = (0..log_t)
            .map(|index| field(101 + 7 * index as u64))
            .collect();
        let mut expected = OptimizedInstructionReadRafKernel::new(
            dimensions,
            &reduction,
            Arc::new(packed.clone()),
            gamma,
        )
        .unwrap();
        let mut actual =
            OptimizedInstructionReadRafKernel::new(dimensions, &reduction, Arc::new(packed), gamma)
                .unwrap();
        let gamma2 = gamma * gamma;
        let mut claim = AkitaField::zero();
        for ((lookup, packed), weight) in lookups.iter().zip(&claims).zip(eq_table(&reduction)) {
            let table = usize::from(packed & 0x3f);
            let value = if table == 0 {
                AkitaField::zero()
            } else {
                field(tables[table - 1].materialize_entry(*lookup))
            };
            let raf = if packed & 0x80 == 0 {
                let (left, right) = LookupBits::new(*lookup, 128).uninterleave();
                gamma * field(u64::from(left)) + gamma2 * field(u64::from(right))
            } else {
                let mut raf = gamma2 * AkitaField::from_u128(*lookup);
                if *lookup >> 64 == u128::from(u64::MAX) {
                    raf += gamma2 * gamma;
                }
                raf
            };
            claim += weight * (value + raf);
        }
        let binds: Vec<_> = (0..128 + log_t)
            .map(|round| match round {
                1 | 130 => AkitaField::zero(),
                2 | 131 => AkitaField::one(),
                _ => field(797 + 17 * round as u64),
            })
            .collect();
        let mut adapter = None;
        let mut saw_cpu_tail = false;
        for round in 0..128 + log_t {
            let bind = round.checked_sub(1).map(|previous| binds[previous]);
            let oracle = expected.prove_round(bind, round, claim).unwrap();
            let polynomial = if round < 128 {
                actual.prove_round(bind, round, claim).unwrap()
            } else {
                let mut cycle_bind = bind;
                if adapter.is_none() {
                    // The challenge preceding first cycle round binds the final
                    // address bit. It must not also become the first cycle bind.
                    let cpu_first = actual.prove_round(cycle_bind.take(), round, claim).unwrap();
                    assert_eq!(cpu_first.coefficients(), oracle.coefficients());
                    let context = SolinasMetal::for_akita().unwrap();
                    let source =
                        Product9ResidentSource::fixture(&context, &lookups, &claims).unwrap();
                    adapter = Some(
                        Product9CycleAdapter::prepare(&mut actual, context, source, 4).unwrap(),
                    );
                }
                let adapter = adapter.as_mut().unwrap();
                let polynomial = adapter
                    .prove_round(&mut actual, cycle_bind, round, claim)
                    .unwrap();
                saw_cpu_tail |= adapter.sequence.is_none();
                polynomial
            };
            assert_eq!(
                polynomial.coefficients(),
                oracle.coefficients(),
                "round {round}, logT{log_t}"
            );
            claim = oracle.evaluate(binds[round]);
        }
        assert!(saw_cpu_tail);
        adapter.as_mut().unwrap().restore(&mut actual).unwrap();
        expected.finish_rounds(*binds.last().unwrap()).unwrap();
        actual.finish_rounds(*binds.last().unwrap()).unwrap();
        let input = InstructionReadRafInputClaims::<AkitaField>::default();
        let oracle = expected.output_claims(&input).unwrap();
        let gpu = actual.output_claims(&input).unwrap();
        assert_eq!(gpu.instruction_ra.len(), 8);
        assert_eq!(gpu.instruction_ra, oracle.instruction_ra);
        assert_eq!(gpu.lookup_table_flags, oracle.lookup_table_flags);
        assert_eq!(gpu.instruction_raf_flag, oracle.instruction_raf_flag);
    }
}
