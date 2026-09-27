//! The verifier-only Akita setup path (`AkitaScheme::verifier_setup_only`,
//! `verifier_preprocess_full_with_advice`) against the prover-derived one.

#[cfg(feature = "akita")]
#[expect(
    clippy::expect_used,
    clippy::print_stderr,
    reason = "integration tests should fail loudly and report sizes"
)]
mod verifier_setup {
    use std::sync::Arc;

    use akita_types::{AkitaExpandedSetup, AkitaVerifierSetup, FlatMatrix};
    use jolt_akita::{setup_prefixes, AkitaBackendFlavor, AkitaField, AkitaScheme};
    use jolt_claims::protocols::jolt::JoltAdviceKind;
    use jolt_openings::CommitmentScheme;
    use jolt_prover::akita::preprocessing::{
        advice_physical_num_vars_for_bytes, grouped_setup_params_for_shape,
    };
    use jolt_prover::ProverConfig;

    /// The prover-derived backend key narrowed to `num_field_elements`: what
    /// `akita_prover::AkitaProverSetup::to_verifier_setup` produces for that
    /// capacity.
    fn narrowed(
        key: &AkitaVerifierSetup<AkitaField>,
        num_field_elements: usize,
    ) -> AkitaVerifierSetup<AkitaField> {
        let mut descriptor = key.expanded.descriptor().clone();
        descriptor.num_field_elements = num_field_elements;
        let prefix = key
            .expanded
            .shared_matrix()
            .as_field_slice()
            .get(..num_field_elements)
            .expect("verifier prefix within the prover matrix")
            .to_vec();
        AkitaVerifierSetup::from_parts(
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    descriptor,
                    FlatMatrix::from_flat_data(prefix),
                ),
            ),
            key.prefix_slots.clone(),
        )
        .expect("narrowed verifier setup")
    }

    fn assert_paths_agree(log_t: usize, ram_log: usize, bytecode_log: usize, advice: bool) {
        let untrusted = advice.then(|| {
            advice_physical_num_vars_for_bytes(JoltAdviceKind::Untrusted, 4 << 20)
                .expect("advice arity")
        });
        let config = ProverConfig::for_shape(1 << log_t, 1 << ram_log);
        let params =
            grouped_setup_params_for_shape(&config, 1 << bytecode_log, untrusted, None, &[])
                .expect("setup params");
        let shape = format!("T=2^{log_t} K=2^{ram_log} bytecode=2^{bytecode_log} advice={advice}");

        let (_, from_prover) = AkitaScheme::setup(params.clone()).expect("prover setup");
        let verifier_only = AkitaScheme::verifier_setup_only(&params).expect("verifier-only setup");
        assert_eq!(
            verifier_only, from_prover,
            "{shape}: setup metadata and catalogs"
        );

        let prover_key = from_prover
            .backend_verifier_key(AkitaBackendFlavor::OneHot)
            .expect("prover-derived key");
        let verifier_key = verifier_only
            .backend_verifier_key(AkitaBackendFlavor::OneHot)
            .expect("verifier-only key");
        let n = verifier_key.expanded.descriptor().num_field_elements;
        let full = prover_key.expanded.shared_matrix().num_field_elements();
        assert!(
            n <= full,
            "{shape}: verifier prefix {n} exceeds prover matrix {full}"
        );
        assert_eq!(
            *verifier_key,
            narrowed(prover_key, n),
            "{shape}: verifier key must equal the narrowed prover-derived key"
        );
        assert_eq!(
            verifier_key.expanded.descriptor(),
            narrowed(prover_key, n).expanded.descriptor()
        );
        eprintln!(
            "{shape}: verifier matrix {n} of {full} field elements, {} setup-prefix slots",
            verifier_key.prefix_slots.len()
        );
    }

    #[test]
    fn verifier_only_setup_equals_prover_derived_at_2_16() {
        assert_paths_agree(16, 16, 12, false);
        assert_paths_agree(16, 20, 21, true);
    }

    #[test]
    fn verifier_only_setup_equals_prover_derived_at_2_20() {
        assert_paths_agree(20, 20, 16, false);
        assert_paths_agree(20, 22, 21, true);
    }

    /// The smallest shapes with setup-claim offloading: their verifier keys
    /// carry setup-prefix commitments (taken from the shipped registry).
    #[test]
    fn verifier_only_setup_equals_prover_derived_with_prefix_slots() {
        assert_paths_agree(21, 20, 16, false);
        assert_paths_agree(22, 22, 21, true);
    }

    /// The verifier-only key for the largest block-prover shape (2^26 cycles,
    /// where the prover matrix is 2^28 field elements = 4 GiB) stays within
    /// 2^24 field elements (256 MiB) and derives no setup-prefix commitment.
    #[test]
    fn verifier_only_setup_at_2_26_is_bounded() {
        let untrusted = advice_physical_num_vars_for_bytes(JoltAdviceKind::Untrusted, 4 << 20)
            .expect("advice arity");
        let config = ProverConfig::for_shape(1 << 26, 1 << 22);
        let params = grouped_setup_params_for_shape(&config, 1 << 21, Some(untrusted), None, &[])
            .expect("setup params");
        let started = std::time::Instant::now();
        let setup = AkitaScheme::verifier_setup_only(&params).expect("verifier-only setup");
        let key = setup
            .backend_verifier_key(AkitaBackendFlavor::OneHot)
            .expect("verifier key");
        let n = key.expanded.shared_matrix().num_field_elements();
        eprintln!(
            "T=2^26: verifier matrix {n} field elements, {} setup-prefix slots, {:.2}s",
            key.prefix_slots.len(),
            started.elapsed().as_secs_f64()
        );
        assert!(
            n <= 1 << 24,
            "verifier matrix {n} exceeds 2^24 field elements"
        );
        assert_eq!(key.prefix_slots.len(), 9);
    }

    /// Every slot the Aether block prover's shapes need is shipped, so a
    /// verifier never commits a setup prefix itself.
    #[test]
    fn shipped_registry_covers_the_block_prover_shapes() {
        let shipped = setup_prefixes::shipped().expect("shipped registry");
        let untrusted = advice_physical_num_vars_for_bytes(JoltAdviceKind::Untrusted, 4 << 20)
            .expect("advice arity");
        for log_t in 16..=26 {
            for ram_log in [20, 22, 24, 26] {
                let config = ProverConfig::for_shape(1 << log_t, 1 << ram_log);
                let params =
                    grouped_setup_params_for_shape(&config, 1 << 21, Some(untrusted), None, &[])
                        .expect("setup params");
                for id in AkitaScheme::required_setup_prefix_slots(&params).expect("slot ids") {
                    assert!(
                        shipped.get(&id).is_some(),
                        "T=2^{log_t} K=2^{ram_log}: slot natural_len={} not shipped",
                        id.natural_len
                    );
                }
            }
        }
    }
}
