#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "deterministic native RAM RA parity fixtures"
)]

use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_field::{One as _, Ring as _, Zero as _};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, UnivariatePoly};
use metal::foreign_types::ForeignType;

use super::*;
use crate::metal::ram_records::NO_ACCESS;
use crate::reference::views::eq_table;

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn field(&mut self) -> AkitaField {
        // Also exercise canonical residues near the modulus, rather than
        // restricting multiplication to small positive field elements.
        let value =
            AkitaField::from_u128((u128::from(self.next()) << 64) | u128::from(self.next() | 2));
        match self.next() % 4 {
            0 => -AkitaField::one(),
            1 => -AkitaField::from_u64(self.next() | 2),
            2 => -value,
            _ => value,
        }
    }
}

fn bind(values: &mut Vec<AkitaField>, challenge: AkitaField) {
    for pair in 0..values.len() / 2 {
        let lo = values[2 * pair];
        values[pair] = lo + challenge * (values[2 * pair + 1] - lo);
    }
    values.truncate(values.len() / 2);
}

fn cpu_q(tables: &[Vec<AkitaField>], e_in: &[AkitaField], e_out: &[AkitaField]) -> Vec<AkitaField> {
    let factors = tables.len();
    let mut q = vec![AkitaField::zero(); factors];
    for (x_out, outer) in e_out.iter().enumerate() {
        for (x_in, inner) in e_in.iter().enumerate() {
            let pair = x_out * e_in.len() + x_in;
            let weight = *inner * *outer;
            for (sample, result) in q.iter_mut().enumerate() {
                let product = tables.iter().fold(AkitaField::one(), |product, table| {
                    let lo = table[2 * pair];
                    let hi = table[2 * pair + 1];
                    let delta = hi - lo;
                    product
                        * if sample + 1 == factors {
                            delta
                        } else {
                            lo + AkitaField::from_u64(sample as u64 + 1) * delta
                        }
                });
                *result += weight * product;
            }
        }
    }
    q
}

fn cpu_poly(eq: &[AkitaField], tables: &[Vec<AkitaField>]) -> UnivariatePoly<AkitaField> {
    // Full dense eq is an independent oracle for the Gruen factoring and
    // cached-table contractions, including the switch from E_in to E_out.
    let evaluations = (0..tables.len() + 2)
        .map(|sample| {
            let sample = AkitaField::from_u64(sample as u64);
            (0..eq.len() / 2)
                .map(|pair| {
                    let lo = eq[2 * pair];
                    let hi = eq[2 * pair + 1];
                    tables
                        .iter()
                        .fold(lo + sample * (hi - lo), |product, table| {
                            let lo = table[2 * pair];
                            let hi = table[2 * pair + 1];
                            product * (lo + sample * (hi - lo))
                        })
                })
                .sum()
        })
        .collect::<Vec<_>>();
    UnivariatePoly::interpolate_over_integers(&evaluations)
}

fn dense_allocation_snapshot(sequence: &RamRaSequence) -> ([u64; 2], [usize; 2]) {
    (
        [
            sequence
                .buffers
                .dense_a
                .as_ref()
                .map_or(0, |buffer| buffer.length()),
            sequence
                .buffers
                .dense_b
                .as_ref()
                .map_or(0, |buffer| buffer.length()),
        ],
        [
            sequence
                .buffers
                .dense_a
                .as_ref()
                .map_or(0, |buffer| buffer.as_ptr() as usize),
            sequence
                .buffers
                .dense_b
                .as_ref()
                .map_or(0, |buffer| buffer.as_ptr() as usize),
        ],
    )
}

fn assert_rejected_without_mutation(
    sequence: &mut RamRaSequence,
    call: impl FnOnce(&mut RamRaSequence) -> Result<[AkitaField; 8], MetalError>,
) {
    let state = |sequence: &RamRaSequence| {
        (
            sequence.weights_initialized,
            sequence.e_in_elements,
            sequence.e_out_elements,
            sequence.e_in_in_a,
            sequence.e_out_in_a,
            sequence.branch_width,
            sequence.branches_in_a,
            sequence.dense,
            sequence.dense_in_a,
            sequence.dense_elements,
            dense_allocation_snapshot(sequence),
        )
    };
    let before = state(sequence);
    // Only read initialized resident tables. All sequence commands finish
    // synchronously, so these snapshots also detect writes before rejection.
    let resident = |sequence: &RamRaSequence| {
        let mut values = Vec::new();
        let mut append = |buffer: &Buffer, elements: usize| {
            // SAFETY: callers select the initialized active table lengths.
            let fields =
                unsafe { slice::from_raw_parts(buffer.contents().cast::<Fp128>(), elements) };
            values.extend(
                fields
                    .iter()
                    .map(|value| value.into_jolt_field::<AkitaField>()),
            );
        };
        append(
            sequence.branch_source_buffer(),
            sequence.factors * sequence.branch_width * sequence.bins,
        );
        if sequence.weights_initialized {
            append(
                sequence.e_in_buffer(sequence.e_in_in_a),
                sequence.e_in_elements,
            );
            append(
                sequence.e_out_buffer(sequence.e_out_in_a),
                sequence.e_out_elements,
            );
        }
        if sequence.dense {
            append(
                sequence.dense_source_buffer().unwrap(),
                sequence.factors * sequence.dense_elements,
            );
        }
        values
    };
    let before_resident = resident(sequence);
    assert!(call(sequence).is_err());
    assert_eq!(state(sequence), before);
    assert_eq!(resident(sequence), before_resident);
}

fn run_randomized_case(
    context: &SolinasMetal,
    log_t: usize,
    log_k: usize,
    chunk_bits: usize,
    seed: u64,
    activity: usize,
    check_dense_allocations: bool,
) {
    let mut random = Random(seed);
    let rows = 1usize << log_t;
    let address_mask = ((1u64 << log_k) - 1) as u32;
    let mut addresses = (0..rows)
        .map(|row| {
            let value = random.next();
            if activity == 0 || (activity == 1 && row % 5 == 0) {
                NO_ACCESS
            } else {
                value as u32 & address_mask
            }
        })
        .collect::<Vec<_>>();
    if activity != 0 {
        addresses[1] = 0;
        addresses[2] = address_mask;
        addresses[rows - 1] = 1;
    }
    let address_point = (0..log_k).map(|_| random.field()).collect::<Vec<_>>();
    let chunks = committed_address_chunks(&address_point, chunk_bits);
    let factors = chunks.len();
    if log_k == 21 && chunk_bits == 4 {
        assert_eq!(factors, 6);
        // The short high chunk is padded to four bits by the canonical
        // layout: three leading zero coordinates and one address coordinate.
        assert_eq!(&chunks[0][..3], &[AkitaField::zero(); 3]);
        assert_eq!(chunks[0][3], address_point[0]);
    }
    let bins = 1usize << chunk_bits;
    let chunk_tables = chunks
        .iter()
        .flat_map(|chunk| eq_table(chunk))
        .collect::<Vec<_>>();
    let mut expected = (0..factors)
        .map(|factor| {
            addresses
                .iter()
                .map(|address| {
                    if *address == NO_ACCESS {
                        AkitaField::zero()
                    } else {
                        let chunk = (*address >> ((factors - 1 - factor) * chunk_bits)) as usize
                            & (bins - 1);
                        chunk_tables[factor * bins + chunk]
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let columns = Arc::new(RamAccessColumns::from_direct_addresses(
        addresses,
        rows,
        1usize << log_k,
    ));
    let cycle_point = (0..log_t).map(|_| random.field()).collect::<Vec<_>>();
    let mut eq = EqPolynomial::evals(&cycle_point, None);
    let mut gruen = GruenSplitEqPolynomial::new(&cycle_point, BindingOrder::LowToHigh);
    // Keep this call compatible with 44e06d5: that implementation rejects
    // six nibble factors here before submitting a command or writing buffers.
    let mut sequence = context
        .prepare_ram_ra_sequence(
            columns,
            &chunk_tables,
            factors,
            gruen.e_in_current().len(),
            gruen.e_out_current().len(),
        )
        .expect("native committed RAM RA shape must have a direct resident sequence");
    let mut dense_allocations = vec![dense_allocation_snapshot(&sequence)];
    let challenges = (0..log_t)
        .map(|round| match round {
            0 => AkitaField::zero(),
            1 => AkitaField::one(),
            2 => -AkitaField::one(),
            _ => random.field(),
        })
        .collect::<Vec<_>>();
    let mut claim = (0..rows)
        .map(|row| {
            expected
                .iter()
                .fold(eq[row], |product, table| product * table[row])
        })
        .sum();
    for round in 0..log_t {
        assert_rejected_without_mutation(&mut sequence, |sequence| {
            sequence.finish_bind(challenges[round])
        });
        assert_rejected_without_mutation(&mut sequence, |sequence| {
            sequence.bind_and_message(challenges[round], &[], gruen.e_out_current())
        });
        if round != 0 {
            assert_rejected_without_mutation(&mut sequence, |sequence| {
                sequence.message(gruen.e_in_current(), gruen.e_out_current())
            });
        }
        let actual = if round == 0 {
            sequence
                .message(gruen.e_in_current(), gruen.e_out_current())
                .unwrap()
        } else {
            let challenge = challenges[round - 1];
            for table in &mut expected {
                bind(table, challenge);
            }
            bind(&mut eq, challenge);
            gruen.bind(challenge);
            sequence
                .bind_and_message(challenge, gruen.e_in_current(), gruen.e_out_current())
                .unwrap()
        };
        let expected_q = cpu_q(&expected, gruen.e_in_current(), gruen.e_out_current());
        assert_eq!(
            &actual[..factors],
            &expected_q,
            "q grid at round {round}, logT{log_t}"
        );
        assert!(actual[factors..]
            .iter()
            .all(|value| *value == AkitaField::zero()));
        let actual_poly = gruen.gruen_poly_from_evals(&actual[..factors], claim);
        let expected_poly = cpu_poly(&eq, &expected);
        assert_eq!(
            actual_poly.coefficients(),
            expected_poly.coefficients(),
            "round {round}, logT{log_t}, logK{log_k}, chunk{chunk_bits}"
        );
        claim = expected_poly.evaluate(challenges[round]);
        assert_eq!(sequence.is_dense(), round >= 4);
        assert_eq!(sequence.current_elements(), rows >> round);
        dense_allocations.push(dense_allocation_snapshot(&sequence));
    }
    let last = challenges[log_t - 1];
    for table in &mut expected {
        bind(table, last);
    }
    bind(&mut eq, last);
    let terminal = sequence.finish_bind(last).unwrap();
    assert_eq!(
        &terminal[..factors],
        &expected.iter().map(|table| table[0]).collect::<Vec<_>>()
    );
    assert!(terminal[factors..]
        .iter()
        .all(|value| *value == AkitaField::zero()));
    assert_eq!(sequence.current_elements(), 1);
    assert_eq!(
        expected
            .iter()
            .fold(eq[0], |product, table| product * table[0]),
        claim
    );
    assert_rejected_without_mutation(&mut sequence, |sequence| sequence.finish_bind(last));
    assert_rejected_without_mutation(&mut sequence, |sequence| {
        sequence.bind_and_message(last, gruen.e_in_current(), gruen.e_out_current())
    });
    assert_rejected_without_mutation(&mut sequence, |sequence| {
        sequence.message(gruen.e_in_current(), gruen.e_out_current())
    });
    if check_dense_allocations {
        // Check timing after every q grid, full polynomial and terminal claim
        // has matched the independent CPU oracle above. The unmodified source
        // reaches this assertion, then fails because both slabs were eager.
        assert_eq!(dense_allocations[0], ([0; 2], [0; 2]));
        let first_a = dense_allocations[5].1[0];
        let first_b = if log_t > 5 {
            dense_allocations[6].1[1]
        } else {
            0
        };
        assert_ne!(first_a, 0);
        if first_b != 0 {
            assert_ne!(first_a, first_b);
        }
        let a_bytes = (factors * (rows / MATERIALIZE_WIDTH) * size_of::<Fp128>()) as u64;
        let b_bytes = a_bytes / 2;
        for (round, &(lengths, identities)) in dense_allocations[1..].iter().enumerate() {
            assert_eq!(
                lengths,
                [
                    if round >= 4 { a_bytes } else { 0 },
                    if round >= 5 { b_bytes } else { 0 }
                ],
                "dense allocations at round {round}, logT{log_t}"
            );
            assert_eq!(
                identities,
                [
                    if round >= 4 { first_a } else { 0 },
                    if round >= 5 { first_b } else { 0 }
                ],
                "dense allocations must be reused after their first consumer"
            );
        }
        let deferred_allocation_count = usize::from(first_a != 0) + usize::from(first_b != 0);
        assert_eq!(deferred_allocation_count, if rows == 32 { 1 } else { 2 });
        assert_eq!(
            dense_allocation_snapshot(&sequence),
            *dense_allocations.last().unwrap()
        );
    }
}

#[test]
fn ram_ra_dense_slabs_are_allocated_only_at_their_first_consumers() {
    let context = SolinasMetal::for_akita().unwrap();
    for (case, (log_t, log_k, chunk_bits, activity)) in [
        (5, 8, 4, 1),
        (6, 16, 8, 1),
        (8, 24, 8, 2),
        (8, 31, 4, 1),
        (9, 21, 4, 0),
    ]
    .into_iter()
    .enumerate()
    {
        run_randomized_case(
            &context,
            log_t,
            log_k,
            chunk_bits,
            0xde1a_0000 + case as u64,
            activity,
            true,
        );
    }
}

#[test]
fn ram_ra_six_nibble_randomized_all_rounds_and_terminal_match_cpu() {
    let context = SolinasMetal::for_akita().unwrap();
    for (case, (log_t, activity)) in [(5, 1), (9, 1), (10, 2), (9, 0)].into_iter().enumerate() {
        run_randomized_case(
            &context,
            log_t,
            21,
            4,
            0xa6e7_0000 + case as u64,
            activity,
            false,
        );
    }
}

#[test]
fn ram_ra_nibble_factor_boundaries_and_legacy_bytes_match_cpu() {
    let context = SolinasMetal::for_akita().unwrap();
    for (case, (log_k, chunk_bits)) in [
        (8, 4),
        (12, 4),
        (16, 4),
        (20, 4),
        (28, 4),
        (31, 4),
        (16, 8),
        (24, 8),
    ]
    .into_iter()
    .enumerate()
    {
        run_randomized_case(
            &context,
            8,
            log_k,
            chunk_bits,
            0xb17e_0000 + case as u64,
            1,
            false,
        );
    }
}

#[test]
fn ram_ra_rejects_unsupported_shapes_before_preparing_buffers() {
    let context = SolinasMetal::for_akita().unwrap();
    let columns = Arc::new(RamAccessColumns::from_direct_addresses(
        vec![NO_ACCESS; 64],
        0,
        0,
    ));
    for (factors, bins) in [(1, 16), (9, 16), (4, 256), (6, 32), (6, 15)] {
        let result = context.prepare_ram_ra_sequence(
            Arc::clone(&columns),
            &vec![AkitaField::one(); factors * bins],
            factors,
            4,
            8,
        );
        assert!(matches!(result, Err(MetalError::InvalidRamRaState(_))));
    }
}

#[test]
fn ram_ra_terminal_bind_writes_gpu_claims_and_zeroes_unused_lanes() {
    let context = SolinasMetal::for_akita().unwrap();
    let mut random = Random(0x7e12_a6e7);
    for factors in [2, 3, 6, 8] {
        for challenge in [
            AkitaField::zero(),
            AkitaField::one(),
            -AkitaField::one(),
            random.field(),
        ] {
            let rows = 64;
            let columns = Arc::new(RamAccessColumns::from_direct_addresses(
                vec![1; rows],
                rows,
                16,
            ));
            let mut sequence = context
                .prepare_ram_ra_sequence(
                    columns,
                    &vec![AkitaField::one(); factors * 16],
                    factors,
                    4,
                    8,
                )
                .unwrap();
            // Isolate the terminal kernel with known resident factor pairs.
            // The all-round oracle above checks how real rounds reach here.
            let pairs = (0..2 * factors).map(|_| random.field()).collect::<Vec<_>>();
            let dense_a = new_field_buffer(&context, factors * rows / 16).unwrap();
            write_fields(&dense_a, factors * rows / 16, &pairs).unwrap();
            sequence.buffers.dense_a = Some(dense_a);
            write_fields(&sequence.buffers.partial_a, 8, &[-AkitaField::one(); 8]).unwrap();
            sequence.dense = true;
            sequence.dense_in_a = true;
            sequence.dense_elements = 2;
            let expected: [AkitaField; 8] = std::array::from_fn(|factor| {
                if factor < factors {
                    let lo = pairs[2 * factor];
                    lo + challenge * (pairs[2 * factor + 1] - lo)
                } else {
                    AkitaField::zero()
                }
            });
            let terminal = sequence.finish_bind(challenge).unwrap();
            assert_eq!(terminal.to_vec(), expected.to_vec());
            // SAFETY: finish_bind waited for the eight terminal writes. This
            // distinguishes a GPU fold from a host-only returned result.
            let resident = unsafe {
                slice::from_raw_parts(sequence.buffers.partial_a.contents().cast::<Fp128>(), 8)
            };
            assert_eq!(
                resident
                    .iter()
                    .map(|value| value.into_jolt_field::<AkitaField>())
                    .collect::<Vec<_>>(),
                expected.to_vec()
            );
            assert_eq!(sequence.current_elements(), 1);
            assert!(sequence.finish_bind(challenge).is_err());
        }
    }
}

#[test]
fn ram_ra_resident_weights_use_only_initial_host_payload() {
    let context = SolinasMetal::for_akita().unwrap();
    let rows = 64;
    let columns = Arc::new(RamAccessColumns::from_direct_addresses(
        vec![1; rows],
        rows,
        16,
    ));
    let point = vec![AkitaField::from_u64(37); 6];
    let mut gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let tables = vec![AkitaField::one(); 6 * 16];
    let mut sequence = context
        .prepare_ram_ra_sequence(
            columns,
            &tables,
            6,
            gruen.e_in_current().len(),
            gruen.e_out_current().len(),
        )
        .unwrap();
    assert!(sequence.finish_bind(AkitaField::one()).is_err());
    assert!(sequence
        .bind_and_message(
            AkitaField::one(),
            &[AkitaField::one(); 2],
            &[AkitaField::one(); 8]
        )
        .is_err());
    let _ = sequence
        .message(gruen.e_in_current(), gruen.e_out_current())
        .unwrap();
    assert!(sequence
        .message(gruen.e_in_current(), gruen.e_out_current())
        .is_err());
    gruen.bind(AkitaField::from_u64(19));
    // Poison the later host arrays. Dimensions are the only input after the
    // first message; the true prefix weights contract on the device.
    let result = sequence
        .bind_and_message(
            AkitaField::from_u64(19),
            &vec![AkitaField::zero(); gruen.e_in_current().len()],
            &vec![AkitaField::zero(); gruen.e_out_current().len()],
        )
        .unwrap();
    assert_eq!(result[0], AkitaField::one());
    assert_eq!(result[1], AkitaField::one());
    assert_eq!(result[4], AkitaField::one());
    assert_eq!(result[5], AkitaField::zero());
}
