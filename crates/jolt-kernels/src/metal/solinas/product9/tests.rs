#![expect(clippy::unwrap_used, reason = "actual GPU parity fixtures")]

use jolt_field::Ring as _;
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial};

use super::*;

fn field(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

fn metadata() -> Product9CycleInput {
    let phases = (0..PHASES)
        .map(|phase| {
            EqPolynomial::evals(
                &(0..8)
                    .map(|bit| field(31 + phase as u64 * 17 + bit as u64 * 7))
                    .collect::<Vec<_>>(),
                None,
            )
        })
        .collect();
    Product9CycleInput {
        phases,
        table_values: (0..super::super::instruction_read_raf::INSTRUCTION_READ_RAF_TABLES)
            .map(|table| field(211 + table as u64 * 19))
            .collect(),
        raf_interleaved: field(307),
        raf_identity: field(401),
    }
}

fn cpu_bases(
    lookups: &[u128],
    claims: &[u8],
    input: &Product9CycleInput,
) -> [Vec<AkitaField>; FACTORS] {
    std::array::from_fn(|factor| {
        lookups
            .iter()
            .zip(claims)
            .map(|(lookup, claim)| {
                if factor == 0 {
                    let table = usize::from(claim & 0x3f);
                    return if table == 0 {
                        AkitaField::zero()
                    } else {
                        input.table_values[table - 1]
                    } + if claim & 0x80 == 0 {
                        input.raf_interleaved
                    } else {
                        input.raf_identity
                    };
                }
                let phase = 2 * (factor - 1);
                let shift = 120 - 8 * phase;
                input.phases[phase][((lookup >> shift) & 255) as usize]
                    * input.phases[phase + 1][((lookup >> (shift - 8)) & 255) as usize]
            })
            .collect()
    })
}

fn bind(tables: &mut [Vec<AkitaField>; FACTORS], challenge: AkitaField) {
    for table in tables {
        let half = table.len() / 2;
        for index in 0..half {
            table[index] = table[2 * index] + challenge * (table[2 * index + 1] - table[2 * index]);
        }
        table.truncate(half);
    }
}

fn cpu_q(
    tables: &[Vec<AkitaField>; FACTORS],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> [AkitaField; FACTORS] {
    let mut output = [AkitaField::zero(); FACTORS];
    for (outer, &outer_weight) in e_out.iter().enumerate() {
        for inner in 0..e_in.len() {
            let pair = outer * e_in.len() + inner;
            let lo: [AkitaField; FACTORS] = std::array::from_fn(|factor| tables[factor][2 * pair]);
            let hi: [AkitaField; FACTORS] =
                std::array::from_fn(|factor| tables[factor][2 * pair + 1]);
            for (sample, value) in output.iter_mut().enumerate() {
                let mut product = AkitaField::one();
                for factor in 0..FACTORS {
                    let delta = hi[factor] - lo[factor];
                    product *= if sample + 1 == FACTORS {
                        delta
                    } else {
                        lo[factor] + field((sample + 1) as u64) * delta
                    };
                }
                *value += outer_weight * e_in[inner] * product;
            }
        }
    }
    output
}

#[test]
fn product9_gpu_q_and_nine_terminal_values_match_cpu_across_lazy_dense_tail() {
    for log_t in [9usize, 10] {
        let rows = 1 << log_t;
        let lookups: Vec<_> = (0..rows)
            .map(|row| match row {
                0 => 0,
                1 => u128::MAX,
                2 => 255u128 << 64,
                3 => 255u128 << 56,
                _ => {
                    ((row as u128 * 0x9e37_79b9_7f4a_7c15) << 64)
                        | ((row as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9) as u128)
                }
            })
            .collect();
        let input = metadata();
        let claims: Vec<_> = (0..rows)
            .map(|row| {
                let table = if row % 5 == 0 {
                    0
                } else {
                    row % input.table_values.len() + 1
                };
                table as u8
                    | if row % 3 == 0 { 0x80 } else { 0 }
                    | if row % 7 == 0 { 0x40 } else { 0 }
            })
            .collect();
        let mut expected = cpu_bases(&lookups, &claims, &input);
        let cycle: Vec<_> = (0..log_t).map(|bit| field(503 + bit as u64 * 11)).collect();
        let mut gruen = GruenSplitEqPolynomial::new(&cycle, BindingOrder::LowToHigh);
        let context = SolinasMetal::for_akita().unwrap();
        let source = Product9ResidentSource::fixture(&context, &lookups, &claims).unwrap();
        let mut sequence = Product9Sequence::new(
            context,
            source,
            input,
            gruen.e_in_current().len(),
            gruen.e_out_current().len(),
        )
        .unwrap();
        let challenges: Vec<_> = (0..log_t)
            .map(|round| match round {
                0 => AkitaField::zero(),
                1 => AkitaField::one(),
                _ => field(601 + round as u64 * 13),
            })
            .collect();
        let mut bound = 0;
        for round in 0..log_t {
            if sequence.is_dense() && sequence.current_elements() <= 4 {
                break;
            }
            let actual = if round == 0 {
                sequence
                    .message(gruen.e_in_current(), gruen.e_out_current())
                    .unwrap()
            } else {
                let challenge = challenges[round - 1];
                bind(&mut expected, challenge);
                gruen.bind(challenge);
                bound += 1;
                sequence
                    .bind_and_message(challenge, gruen.e_in_current(), gruen.e_out_current())
                    .unwrap()
            };
            assert_eq!(
                actual,
                cpu_q(&expected, gruen.e_in_current(), gruen.e_out_current()),
                "q grid round {round}, logT{log_t}"
            );
        }
        assert!(sequence.is_dense());
        let mut actual = std::array::from_fn(|_| vec![AkitaField::zero(); 4]);
        sequence.read_current_factor_tables(&mut actual).unwrap();
        assert_eq!(actual, expected);
        for challenge in &challenges[bound..] {
            bind(&mut actual, *challenge);
            bind(&mut expected, *challenge);
        }
        assert!(actual.iter().all(|table| table.len() == 1));
        assert_eq!(actual, expected);
    }
}

#[test]
fn product9_w64_t24_scratch_is_bounded_without_full_domain_factors() {
    assert_eq!(dense_scratch_bytes(1 << 24).unwrap(), 54 << 20);
    assert!(dense_scratch_bytes(1 << 25).is_err());
    assert!(dense_scratch_bytes(1000).is_err());
}

#[test]
fn product9_rejects_invalid_claim_table_before_reading_values() {
    let rows = 128;
    let lookups = vec![0; rows];
    let mut claims = vec![0; rows];
    claims[17] = 0x3f;
    let context = SolinasMetal::for_akita().unwrap();
    let source = Product9ResidentSource::fixture(&context, &lookups, &claims).unwrap();
    let mut sequence = Product9Sequence::new(context, source, metadata(), 8, 8).unwrap();
    assert!(matches!(
        sequence.message(&[AkitaField::one(); 8], &[AkitaField::one(); 8]),
        Err(Product9Error::Source(1))
    ));
}
