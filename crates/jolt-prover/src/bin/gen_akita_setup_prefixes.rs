//! Offline generator for the shipped Akita setup-prefix commitments
//! (`crates/jolt-akita/schedules/jolt-fp128-setup-prefixes.akr`).
//!
//! A verifier key needs the setup-prefix commitments of every schedule row its
//! shape admits. Deriving one commits a prefix of the public setup stream (up
//! to the prover's whole matrix), so they are derived here once and shipped;
//! `jolt_akita::AkitaScheme::verifier_setup_only` then derives only the
//! public-matrix prefix verification reads. The shape grid is explicit and
//! the output is canonical (slots in id order), so rerunning with the same
//! grid reproduces the file byte for byte.
//!
//! ```text
//! cargo run --release -p jolt-prover --features akita --bin gen_akita_setup_prefixes -- \
//!     list  [--log-t A..=B] [--bytecode-log A..=B] [--ram-log A..=B] [--untrusted-advice-bytes N]
//! cargo run --release -p jolt-prover --features akita --bin gen_akita_setup_prefixes -- \
//!     write <out.akr> [same grid flags]
//! ```
//!
//! `write` merges into `<out.akr>` when it exists: slots already there are
//! kept (they are checked by the `setup_prefixes` test), new ones derived.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "offline generator: fail loud, narrate progress"
)]

use std::collections::BTreeMap;
use std::ops::RangeInclusive;
use std::time::Instant;

use jolt_akita::setup_prefixes;
use jolt_akita::AkitaScheme;
use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_prover::akita::preprocessing::{
    advice_physical_num_vars_for_bytes, grouped_setup_params_for_shape,
};
use jolt_prover::ProverConfig;

/// The shape grid covered by the shipped artifact: every trace length from
/// 2^16 to 2^26, program bytecode 2^10..=2^22, RAM 2^10..=2^26, and no
/// advice or the Aether block prover's 4 MiB untrusted advice.
const DEFAULT_LOG_T: RangeInclusive<usize> = 16..=26;
const DEFAULT_BYTECODE_LOG: RangeInclusive<usize> = 10..=22;
const DEFAULT_RAM_LOG: RangeInclusive<usize> = 10..=26;
const DEFAULT_UNTRUSTED_ADVICE_BYTES: usize = 4 << 20;

struct Grid {
    log_t: RangeInclusive<usize>,
    bytecode_log: RangeInclusive<usize>,
    ram_log: RangeInclusive<usize>,
    untrusted_advice_bytes: usize,
}

fn parse_range(value: &str) -> RangeInclusive<usize> {
    let parse = |s: &str| {
        s.parse::<usize>()
            .unwrap_or_else(|_| panic!("bad number {s}"))
    };
    match value.split_once("..=") {
        Some((a, b)) => parse(a)..=parse(b),
        None => parse(value)..=parse(value),
    }
}

fn parse_grid(args: &[String]) -> Grid {
    let mut grid = Grid {
        log_t: DEFAULT_LOG_T,
        bytecode_log: DEFAULT_BYTECODE_LOG,
        ram_log: DEFAULT_RAM_LOG,
        untrusted_advice_bytes: DEFAULT_UNTRUSTED_ADVICE_BYTES,
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--log-t" => grid.log_t = parse_range(value),
            "--bytecode-log" => grid.bytecode_log = parse_range(value),
            "--ram-log" => grid.ram_log = parse_range(value),
            "--untrusted-advice-bytes" => {
                grid.untrusted_advice_bytes = value.parse().expect("advice bytes");
            }
            other => panic!("unknown flag {other}"),
        }
    }
    grid
}

/// Union of the setup-prefix slots every shape of `grid` requires.
fn required_slots(grid: &Grid) -> BTreeMap<setup_prefixes::SetupPrefixSlotId, usize> {
    let advice_vars =
        advice_physical_num_vars_for_bytes(JoltAdviceKind::Untrusted, grid.untrusted_advice_bytes)
            .expect("advice arity");
    let mut slots = BTreeMap::new();
    let mut shapes = 0usize;
    for log_t in grid.log_t.clone() {
        for ram_log in grid.ram_log.clone() {
            let config = ProverConfig::for_shape(1 << log_t, 1 << ram_log);
            for bytecode_log in grid.bytecode_log.clone() {
                for advice in [None, Some(advice_vars)] {
                    let Ok(params) = grouped_setup_params_for_shape(
                        &config,
                        1 << bytecode_log,
                        advice,
                        None,
                        &[],
                    ) else {
                        continue;
                    };
                    let ids = match AkitaScheme::required_setup_prefix_slots(&params) {
                        Ok(ids) => ids,
                        Err(error) => {
                            eprintln!(
                                "skip log_t={log_t} ram=2^{ram_log} bytecode=2^{bytecode_log} \
                                 advice={advice:?}: {error}"
                            );
                            continue;
                        }
                    };
                    shapes += 1;
                    for id in ids {
                        let footprint = setup_prefixes::setup_prefix_slot_field_elements(&id)
                            .expect("footprint");
                        let _ = slots.insert(id, footprint);
                    }
                }
            }
        }
    }
    eprintln!(
        "{shapes} shapes, {} distinct setup-prefix slots",
        slots.len()
    );
    slots
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: gen_akita_setup_prefixes list|write <out.akr> [grid flags]";
    let (mode, rest) = args.split_first().expect(usage);
    match mode.as_str() {
        "list" => {
            let slots = required_slots(&parse_grid(rest));
            for (id, footprint) in &slots {
                println!(
                    "natural_len={} n_prefix={} d={} footprint={}",
                    id.natural_len,
                    id.n_prefix().expect("n_prefix"),
                    id.d_setup(),
                    footprint
                );
            }
        }
        "write" => {
            let (out, rest) = rest.split_first().expect(usage);
            let slots = required_slots(&parse_grid(rest));
            let existing = std::fs::read(out).unwrap_or_default();
            let mut registry = setup_prefixes::decode(&existing).expect("existing artifact");
            let missing: Vec<_> = slots
                .keys()
                .filter(|id| registry.get(id).is_none())
                .cloned()
                .collect();
            eprintln!(
                "{} slots already present, deriving {}",
                slots.len() - missing.len(),
                missing.len()
            );
            // Derive smallest first, one at a time, so memory tracks each slot.
            let mut ordered = missing;
            ordered.sort_by_key(|id| slots.get(id).copied().unwrap_or(0));
            for id in ordered {
                let started = Instant::now();
                let [slot] = <[_; 1]>::try_from(
                    setup_prefixes::derive(std::slice::from_ref(&id)).expect("derive slot"),
                )
                .unwrap_or_else(|_| panic!("one slot"));
                registry.insert(slot).expect("insert slot");
                eprintln!(
                    "derived n_prefix={} natural_len={} in {:.1}s",
                    id.n_prefix().expect("n_prefix"),
                    id.natural_len,
                    started.elapsed().as_secs_f64()
                );
            }
            let bytes = setup_prefixes::encode(&registry).expect("encode");
            std::fs::write(out, &bytes).expect("write artifact");
            eprintln!(
                "wrote {} slots, {} bytes to {out}",
                registry.len(),
                bytes.len()
            );
        }
        _ => panic!("{usage}"),
    }
}
