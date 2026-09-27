//! Shipped setup-prefix commitments for verifier-only setup derivation.
//!
//! Akita's recursive setup offloading commits power-of-two prefixes of the
//! public setup stream, and a verifier key carries those commitments. They are
//! SIS commitments over prefixes as long as the prover's whole public matrix
//! (2^28 field elements for a 2^26-cycle trace), so deriving one costs the
//! verifier what setup costs the prover: committing that prefix. Instead the
//! commitments for Jolt's shapes are derived once, offline, by
//! `gen_jolt_setup_prefixes` and shipped here as data, exactly like the
//! schedule catalogs they belong to. A commitment depends only on its slot id
//! (profile and prefix length) and the fixed public seed, so one registry
//! serves every shape whose schedules use those slots.
//!
//! [`crate::AkitaScheme::verifier_setup_only`] takes each required slot from
//! this registry and derives any slot missing from it (bounded by that slot's
//! prefix, never the full prover matrix). The `setup_prefixes` test
//! re-derives shipped entries and compares them byte for byte.

use std::sync::OnceLock;

use akita_pcs::{AkitaDeserialize, AkitaSerialize};
use akita_types::sample_akita_setup_seed;
pub use akita_types::{
    setup_prefix_slot_field_elements, SetupPrefixSlotId, SetupPrefixVerifierRegistry,
    SetupPrefixVerifierSlot,
};
use jolt_openings::OpeningsError;

use crate::adapters::AkitaField;

/// Artifact file name inside the packaged `schedules/` directory.
pub const ARTIFACT_FILE_NAME: &str = "jolt-fp128-setup-prefixes.akr";

const MAGIC: &[u8; 8] = b"JAKSPR01";

const SHIPPED_BYTES: &[u8] = include_bytes!("../schedules/jolt-fp128-setup-prefixes.akr");

/// The shipped registry, decoded and validated once.
pub fn shipped() -> Result<&'static SetupPrefixVerifierRegistry<AkitaField>, OpeningsError> {
    static SHIPPED: OnceLock<Result<SetupPrefixVerifierRegistry<AkitaField>, String>> =
        OnceLock::new();
    SHIPPED
        .get_or_init(|| decode(SHIPPED_BYTES).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| OpeningsError::InvalidSetup(format!("shipped setup prefixes: {error}")))
}

/// Decodes a registry artifact. An empty artifact is an empty registry.
pub fn decode(bytes: &[u8]) -> Result<SetupPrefixVerifierRegistry<AkitaField>, OpeningsError> {
    if bytes.is_empty() {
        return Ok(SetupPrefixVerifierRegistry::new(sample_akita_setup_seed()));
    }
    let payload = bytes
        .strip_prefix(MAGIC.as_slice())
        .ok_or_else(|| OpeningsError::InvalidSetup("bad setup-prefix artifact magic".to_owned()))?;
    let mut reader = payload;
    let registry =
        SetupPrefixVerifierRegistry::<AkitaField>::deserialize_compressed(&mut reader, &())
            .map_err(|error| OpeningsError::InvalidSetup(error.to_string()))?;
    if !reader.is_empty() {
        return Err(OpeningsError::InvalidSetup(
            "trailing bytes after the setup-prefix registry".to_owned(),
        ));
    }
    if registry.setup_seed() != &sample_akita_setup_seed() {
        return Err(OpeningsError::InvalidSetup(
            "setup-prefix registry belongs to a different public matrix".to_owned(),
        ));
    }
    Ok(registry)
}

/// Encodes a registry artifact (canonical: slots in id order).
pub fn encode(
    registry: &SetupPrefixVerifierRegistry<AkitaField>,
) -> Result<Vec<u8>, OpeningsError> {
    let mut bytes = MAGIC.to_vec();
    registry
        .serialize_compressed(&mut bytes)
        .map_err(|error| OpeningsError::InvalidSetup(error.to_string()))?;
    Ok(bytes)
}

/// Derives the verifier half of setup-prefix commitments for `slot_ids` from
/// the public stream, as prover setup construction commits them.
pub fn derive(
    slot_ids: &[SetupPrefixSlotId],
) -> Result<Vec<SetupPrefixVerifierSlot<AkitaField>>, OpeningsError> {
    crate::adapters::with_backend_pool(|| {
        akita_pcs::verifier_setup::derive_setup_prefix_verifier_slots::<AkitaField>(slot_ids)
    })
    .map_err(|error| OpeningsError::InvalidSetup(error.to_string()))
}
