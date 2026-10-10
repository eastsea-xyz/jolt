#![expect(
    clippy::unwrap_used,
    reason = "actual Metal arithmetic parity fixtures"
)]

use jolt_field::One as _;
use jolt_field::Ring as _;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};
use jolt_witness::witnesses::{BalancedIncColumn, FusedInc};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use super::*;

fn random_field(rng: &mut impl RngCore) -> AkitaField {
    AkitaField::from_u128((u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64()))
}

fn selectors(chunk_bits: usize) -> Vec<BooleanitySelector> {
    let mut selectors = (0..128 / chunk_bits)
        .rev()
        .map(|chunk| BooleanitySelector::Lookup {
            shift: (chunk * chunk_bits) as u32,
        })
        .collect::<Vec<_>>();
    // Six nibble chunks cover the T22/T24 fixtures' bytecode and RAM
    // decompositions. The byte-sized route has the corresponding three.
    for kind in [1, 2] {
        selectors.extend((0..24 / chunk_bits).rev().map(|chunk| {
            let shift = (chunk * chunk_bits) as u32;
            if kind == 1 {
                BooleanitySelector::Bytecode { shift }
            } else {
                BooleanitySelector::Ram { shift }
            }
        }));
    }
    selectors.extend(
        (0..64 / chunk_bits).map(|chunk| BooleanitySelector::FusedInc {
            shift: (chunk * chunk_bits) as u32,
        }),
    );
    selectors.push(BooleanitySelector::FusedIncMsb);
    selectors
}

fn rows(rng: &mut impl RngCore, count: usize) -> Vec<BooleanityRow> {
    (0..count)
        .map(|index| {
            let lookup = (u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64());
            let magnitude = match index % 7 {
                0 => 0,
                1 => u64::MAX,
                _ => rng.next_u64(),
            };
            let increment = if index % 3 == 0 {
                -i128::from(magnitude)
            } else {
                i128::from(magnitude)
            };
            let pc = (index % 5 != 0).then(|| rng.next_u64() & (PACKED_PC_MASK - 1));
            let ram = (index % 4 != 0).then(|| rng.next_u64() % u64::from(u32::MAX));
            BooleanityRow::new(lookup, pc, ram, increment).unwrap()
        })
        .collect()
}

fn hot_index(row: BooleanityRow, selector: BooleanitySelector, chunk_bits: usize) -> Option<usize> {
    let words = row.words();
    let mask = (1usize << chunk_bits) - 1;
    match selector {
        BooleanitySelector::Lookup { shift } => {
            let lookup = u128::from(words[0]) | (u128::from(words[1]) << 64);
            Some((lookup >> shift) as usize & mask)
        }
        BooleanitySelector::Bytecode { shift } => {
            let pc_plus_one = words[4] & PACKED_PC_MASK;
            pc_plus_one
                .checked_sub(1)
                .map(|pc| (pc >> shift) as usize & mask)
        }
        BooleanitySelector::Ram { shift } => words[2]
            .checked_sub(1)
            .map(|ram| (ram >> shift) as usize & mask),
        BooleanitySelector::FusedInc { shift } => {
            let increment = FusedInc(if words[4] >> PACKED_INC_SIGN_SHIFT == 0 {
                i128::from(words[3])
            } else {
                -i128::from(words[3])
            });
            Some(increment.selected_row(BalancedIncColumn::Digit {
                width: chunk_bits,
                index: shift as usize / chunk_bits,
            }))
        }
        BooleanitySelector::FusedIncMsb => {
            let increment = FusedInc(if words[4] >> PACKED_INC_SIGN_SHIFT == 0 {
                i128::from(words[3])
            } else {
                -i128::from(words[3])
            });
            Some(increment.selected_row(BalancedIncColumn::Carry { width: chunk_bits }))
        }
    }
}

fn bind(tables: &mut [Vec<AkitaField>], challenge: AkitaField) {
    for table in tables {
        let half = table.len() / 2;
        for index in 0..half {
            let lo = table[2 * index];
            table[index] = lo + challenge * (table[2 * index + 1] - lo);
        }
        table.truncate(half);
    }
}

fn cpu_message(
    tables: &[Vec<AkitaField>],
    rho: &[AkitaField],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> [AkitaField; MESSAGE_LANES] {
    let mut message = [AkitaField::zero(); MESSAGE_LANES];
    for (outer, outer_weight) in e_out.iter().enumerate() {
        for (inner, inner_weight) in e_in.iter().enumerate() {
            let pair = outer * e_in.len() + inner;
            let weight = *inner_weight * *outer_weight;
            for (table, rho) in tables.iter().zip(rho) {
                let h_0 = table[2 * pair];
                let delta = table[2 * pair + 1] - h_0;
                message[0] += weight * h_0 * (h_0 - *rho);
                message[1] += weight * delta * delta;
            }
        }
    }
    message
}

fn sequence_parity(
    chunk_bits: usize,
    log_t: usize,
    width: usize,
    selectors: Vec<BooleanitySelector>,
) {
    let mut rng =
        ChaCha20Rng::seed_from_u64(0x6b00 + (chunk_bits * 100 + log_t * 10 + width) as u64);
    let rows = rows(&mut rng, 1 << log_t);
    let k = 1 << chunk_bits;
    let base_tables = (0..selectors.len() * k)
        .map(|_| random_field(&mut rng))
        .collect::<Vec<_>>();
    let rho = (0..selectors.len())
        .map(|_| random_field(&mut rng))
        .collect::<Vec<_>>();
    let mut expected = selectors
        .iter()
        .enumerate()
        .map(|(poly, selector)| {
            rows.iter()
                .map(|row| {
                    hot_index(*row, *selector, chunk_bits)
                        .map_or_else(AkitaField::zero, |hot| base_tables[poly * k + hot])
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let cycle = (0..log_t)
        .map(|_| random_field(&mut rng))
        .collect::<Vec<_>>();
    let mut eq = GruenSplitEqPolynomial::new(&cycle, BindingOrder::LowToHigh);
    let context = SolinasMetal::for_akita().unwrap();
    let mut sequence = context
        .prepare_booleanity_sequence(
            &rows,
            &selectors,
            &base_tables,
            &rho,
            k,
            eq.e_in_current_len(),
            eq.e_out_current_len(),
            BooleanitySequenceConfig {
                materialize_width: width,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(sequence.finish_bind(AkitaField::one()).is_err());
    assert!(sequence
        .read_terminal_values(&mut vec![AkitaField::zero(); selectors.len()])
        .is_err());
    let challenges = (0..log_t)
        .map(|round| match round {
            0 => AkitaField::zero(),
            1 => AkitaField::one(),
            _ => random_field(&mut rng),
        })
        .collect::<Vec<_>>();
    for round in 0..log_t {
        let actual = if round == 0 {
            sequence
                .message(eq.e_in_current(), eq.e_out_current())
                .unwrap()
        } else {
            let challenge = challenges[round - 1];
            bind(&mut expected, challenge);
            eq.bind(challenge);
            sequence
                .bind_and_message_resident(challenge, eq.e_in_current_len(), eq.e_out_current_len())
                .unwrap()
        };
        assert_eq!(
            actual,
            cpu_message(&expected, &rho, eq.e_in_current(), eq.e_out_current()),
            "chunk{chunk_bits}, logT{log_t}, W{width}, round{round}, polys{}",
            selectors.len()
        );
        assert_eq!(sequence.round_device_buffer_allocations(), 0);
        if sequence.current_elements() > 2 {
            assert!(sequence.finish_bind(AkitaField::one()).is_err());
        }
    }
    sequence.finish_bind(*challenges.last().unwrap()).unwrap();
    assert!(sequence.finish_bind(*challenges.last().unwrap()).is_err());
    let mut actual = vec![AkitaField::zero(); selectors.len()];
    sequence.read_terminal_values(&mut actual).unwrap();
    bind(&mut expected, *challenges.last().unwrap());
    assert_eq!(
        actual,
        expected.iter().flatten().copied().collect::<Vec<_>>()
    );
}

#[test]
fn booleanity_random_gpu_messages_and_terminal_values_match_cpu_for_committed_chunks() {
    for chunk_bits in [4usize, 8] {
        for (log_t, width) in [(6, 1), (7, 8), (8, 32)] {
            sequence_parity(chunk_bits, log_t, width, selectors(chunk_bits));
        }
    }
}

#[test]
fn booleanity_random_gpu_partial_simd_poly_group_matches_cpu() {
    // The last lane alone must gather the row's signed increment, including
    // magnitude and sign supplied by the other SIMD row-loading lanes.
    let mut selectors = vec![BooleanitySelector::Lookup { shift: 64 }; 32];
    selectors.push(BooleanitySelector::FusedIncMsb);
    sequence_parity(4, 6, 8, selectors);
}

#[test]
fn booleanity_rejects_wrong_committed_shape_before_sequence_allocation() {
    let context = SolinasMetal::for_akita().unwrap();
    let rows = context
        .prepare_booleanity_rows(&[BooleanityRow::default(); 64])
        .unwrap();
    let rho = [AkitaField::one()];
    let base = [AkitaField::one(); 16];
    assert!(matches!(
        context.prepare_booleanity_sequence_with_rows(
            rows.clone(),
            &[BooleanitySelector::Lookup { shift: 63 }],
            &base,
            &rho,
            16,
            4,
            8,
            BooleanitySequenceConfig::default(),
        ),
        Err(MetalError::InvalidBooleanitySelector)
    ));
    assert!(matches!(
        context.prepare_booleanity_sequence_with_rows(
            rows,
            &[BooleanitySelector::Lookup { shift: 64 }],
            &base,
            &rho,
            16,
            4,
            4,
            BooleanitySequenceConfig::default(),
        ),
        Err(MetalError::BooleanityWeightShape {
            expected: 32,
            covered: 16,
        })
    ));
}

#[test]
fn resident_weight_levels_reject_skips_before_dispatch() {
    assert!(matches!(
        next_weight_level(0, 16, 4),
        Err(MetalError::InvalidBooleanityState(_))
    ));
    assert!(matches!(
        next_weight_level(16, 8, 16),
        Err(MetalError::InvalidBooleanityState(_))
    ));
    assert!(matches!(
        next_weight_level(0, 1, 0),
        Err(MetalError::InvalidBooleanityState(_))
    ));
}
