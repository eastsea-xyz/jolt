//! The shipped setup-prefix commitments (`schedules/jolt-fp128-setup-prefixes.akr`)
//! are exactly what deriving them from the public stream produces.
//!
//! `gen_akita_setup_prefixes` (jolt-prover) writes the artifact; these tests
//! re-derive its slots independently and compare them byte for byte.

#![expect(clippy::expect_used, reason = "integration tests should fail loudly")]

use jolt_akita::setup_prefixes::{self, SetupPrefixSlotId};

/// Slots up to this committed prefix are re-derived on every test run; the
/// larger ones (up to 2^28 field elements, several GB) by the ignored test.
const ROUTINE_MAX_PREFIX: usize = 1 << 24;

fn shipped_bytes() -> Vec<u8> {
    std::fs::read(
        jolt_akita::AkitaScheduleArtifacts::packaged_directory()
            .join(setup_prefixes::ARTIFACT_FILE_NAME),
    )
    .expect("read shipped setup-prefix artifact")
}

fn rederive_and_compare(max_prefix: usize) -> usize {
    let shipped = setup_prefixes::shipped().expect("shipped registry");
    let ids: Vec<SetupPrefixSlotId> = shipped
        .iter()
        .map(|(id, _)| id.clone())
        .filter(|id| id.n_prefix().expect("n_prefix") <= max_prefix)
        .collect();
    for id in &ids {
        let derived = setup_prefixes::derive(std::slice::from_ref(id)).expect("derive slot");
        assert_eq!(derived.len(), 1);
        let shipped_slot = shipped
            .get(id)
            .cloned()
            .expect("shipped slot")
            .with_committed_ring_dimension()
            .expect("canonical ring dimension");
        assert_eq!(
            derived[0],
            shipped_slot,
            "shipped slot natural_len={} n_prefix={} differs from its derivation",
            id.natural_len,
            id.n_prefix().expect("n_prefix")
        );
    }
    ids.len()
}

#[test]
fn shipped_artifact_is_canonical_and_nonempty() {
    let bytes = shipped_bytes();
    let registry = setup_prefixes::decode(&bytes).expect("decode");
    assert!(
        !registry.is_empty(),
        "the shipped registry must carry slots"
    );
    assert_eq!(setup_prefixes::encode(&registry).expect("encode"), bytes);
    let mut truncated = bytes.clone();
    let _ = truncated.pop();
    assert!(setup_prefixes::decode(&truncated).is_err());
}

#[test]
fn shipped_small_slots_rederive_bit_for_bit() {
    assert!(rederive_and_compare(ROUTINE_MAX_PREFIX) > 0);
}

/// Every shipped slot, including the 2^28-element one (about 5 GB and a few
/// minutes): `cargo test --release -p jolt-akita --test setup_prefixes -- --ignored`.
#[test]
#[ignore = "derives setup prefixes up to 2^28 field elements"]
fn all_shipped_slots_rederive_bit_for_bit() {
    let shipped = setup_prefixes::shipped().expect("shipped registry");
    assert_eq!(rederive_and_compare(usize::MAX), shipped.len());
}
