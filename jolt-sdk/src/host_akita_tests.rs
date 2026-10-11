use std::{cell::Cell, ffi::OsStr, rc::Weak};

use super::*;

#[test]
fn automatic_metal_is_limited_to_apple_silicon() {
    assert_eq!(
        backend_mode_for_setting(1, None, true, true),
        BackendMode::MetalFull
    );
    assert_eq!(
        backend_mode_for_setting(METAL_MAX_BYTECODE_LEN + 1, None, true, true),
        BackendMode::MetalCommitOnly
    );
    assert_eq!(
        backend_mode_for_setting(1, None, true, false),
        BackendMode::Cpu
    );
    assert_eq!(
        backend_mode_for_setting(1, None, false, true),
        BackendMode::Cpu
    );
}

#[test]
fn explicit_backend_settings_override_the_automatic_choice() {
    for (setting, expected) in [
        ("full", BackendMode::MetalFull),
        ("hybrid", BackendMode::MetalHybrid),
        ("commit", BackendMode::MetalCommitOnly),
        ("cpu", BackendMode::Cpu),
    ] {
        assert_eq!(
            backend_mode_for_setting(1, Some(setting), true, false),
            expected
        );
        assert_eq!(
            backend_mode_for_setting(1, Some(setting), false, true),
            BackendMode::Cpu
        );
    }
}

#[test]
fn failure_diagnostic_is_validated_and_cpu_ignores_it() {
    for (setting, expected) in [
        ("initialize", ForcedMetalFailure::Initialize),
        ("prove", ForcedMetalFailure::Prove),
    ] {
        assert_eq!(
            forced_metal_failure(BackendMode::MetalFull, Some(OsStr::new(setting)))
                .expect("supported diagnostic must parse"),
            Some(expected)
        );
    }
    let error = forced_metal_failure(BackendMode::MetalFull, Some(OsStr::new("invalid")))
        .expect_err("invalid diagnostic must fail before attempting a backend");
    assert_eq!(
        error.downcast_ref::<IoError>().map(IoError::kind),
        Some(ErrorKind::InvalidInput)
    );
    for setting in ["invalid", "initialize", "prove"] {
        assert_eq!(
            forced_metal_failure(BackendMode::Cpu, Some(OsStr::new(setting)))
                .expect("CPU ignores the Metal diagnostic"),
            None
        );
    }
}

#[test]
fn initialization_failure_falls_back_before_proving() {
    let cache = BackendCache::default();
    let builds = RefCell::new(Vec::new());
    let proof = with_backend_for_mode(
        &cache,
        BackendMode::MetalFull,
        1,
        Some(ForcedMetalFailure::Initialize),
        |mode| {
            builds.borrow_mut().push(mode);
            Ok(AkitaBackend::optimized())
        },
        |_| {
            assert_eq!(
                cache.borrow().as_ref().map(|(mode, _)| *mode),
                Some(BackendMode::Cpu)
            );
            Ok(b"cpu output".to_vec())
        },
    )
    .expect("forced initialization failure must retry on CPU");
    assert_eq!(proof, b"cpu output");
    assert_eq!(*builds.borrow(), [BackendMode::Cpu]);
}

#[test]
fn backend_initialization_errors_fall_back_without_a_metal_proof_attempt() {
    let cache = BackendCache::default();
    let builds = RefCell::new(Vec::new());
    let attempts = Cell::new(0);
    with_backend_for_mode(
        &cache,
        BackendMode::MetalFull,
        1,
        None,
        |mode| {
            builds.borrow_mut().push(mode);
            if mode == BackendMode::MetalFull {
                return Err(IoError::other("opaque device initialization error").into());
            }
            Ok(AkitaBackend::optimized())
        },
        |_| {
            attempts.set(attempts.get() + 1);
            Ok(())
        },
    )
    .expect("opaque Metal constructor errors must retry on CPU");
    assert_eq!(attempts.get(), 1);
    assert_eq!(*builds.borrow(), [BackendMode::MetalFull, BackendMode::Cpu]);
}

#[test]
fn runtime_failure_releases_the_failed_backend_before_cpu_retry() {
    let cache = BackendCache::default();
    let builds = RefCell::new(Vec::new());
    let failed_backend: RefCell<Option<Weak<AkitaBackend>>> = RefCell::new(None);
    let attempts = Cell::new(0);
    let proof = with_backend_for_mode(
        &cache,
        BackendMode::MetalFull,
        1,
        None,
        |mode| {
            if mode == BackendMode::Cpu {
                assert!(failed_backend
                    .borrow()
                    .as_ref()
                    .expect("the first attempt must retain a weak backend handle")
                    .upgrade()
                    .is_none());
            }
            builds.borrow_mut().push(mode);
            Ok(AkitaBackend::optimized())
        },
        |_| {
            attempts.set(attempts.get() + 1);
            if attempts.get() == 1 {
                *failed_backend.borrow_mut() = cache
                    .borrow()
                    .as_ref()
                    .map(|(_, backend)| Rc::downgrade(backend));
                return Err(IoError::other("simulated Metal runtime failure").into());
            }
            Ok(b"cpu output".to_vec())
        },
    )
    .expect("runtime failure must retry once");
    assert_eq!(proof, b"cpu output");
    assert_eq!(attempts.get(), 2);
    assert_eq!(*builds.borrow(), [BackendMode::MetalFull, BackendMode::Cpu]);
    assert_eq!(
        cache.borrow().as_ref().map(|(mode, _)| *mode),
        Some(BackendMode::Cpu)
    );
}

#[test]
fn forced_prove_failure_discards_successful_metal_output_before_retry() {
    struct Output<'a> {
        metal: bool,
        dropped: &'a Cell<bool>,
    }

    impl Drop for Output<'_> {
        fn drop(&mut self) {
            if self.metal {
                self.dropped.set(true);
            }
        }
    }

    let cache = BackendCache::default();
    let metal_output_dropped = Cell::new(false);
    let attempts = Cell::new(0);
    let output = with_backend_for_mode(
        &cache,
        BackendMode::MetalFull,
        1,
        Some(ForcedMetalFailure::Prove),
        |_| Ok(AkitaBackend::optimized()),
        |_| {
            attempts.set(attempts.get() + 1);
            let metal =
                cache.borrow().as_ref().map(|(mode, _)| *mode) == Some(BackendMode::MetalFull);
            if !metal {
                assert!(metal_output_dropped.get());
            }
            Ok(Output {
                metal,
                dropped: &metal_output_dropped,
            })
        },
    )
    .expect("forced prove failure must retry after a completed Metal call");
    assert!(!output.metal);
    assert!(metal_output_dropped.get());
    assert_eq!(attempts.get(), 2);
}

#[test]
fn cpu_errors_are_returned_without_another_retry() {
    let cache = BackendCache::default();
    let attempts = Cell::new(0);
    let error = with_backend_for_mode::<()>(
        &cache,
        BackendMode::Cpu,
        1,
        Some(ForcedMetalFailure::Prove),
        |_| Ok(AkitaBackend::optimized()),
        |_| {
            attempts.set(attempts.get() + 1);
            Err(IoError::new(ErrorKind::InvalidData, "original CPU error").into())
        },
    )
    .expect_err("CPU error must propagate");
    assert_eq!(attempts.get(), 1);
    assert_eq!(error.to_string(), "original CPU error");
    assert_eq!(
        error.downcast_ref::<IoError>().map(IoError::kind),
        Some(ErrorKind::InvalidData)
    );
}

#[test]
fn cpu_fallback_error_is_returned_after_exactly_two_attempts() {
    let cache = BackendCache::default();
    let attempts = Cell::new(0);
    let error = with_backend_for_mode::<()>(
        &cache,
        BackendMode::MetalFull,
        1,
        None,
        |_| Ok(AkitaBackend::optimized()),
        |_| {
            attempts.set(attempts.get() + 1);
            let message = if attempts.get() == 1 {
                "original Metal error"
            } else {
                "original CPU error"
            };
            Err(IoError::other(message).into())
        },
    )
    .expect_err("CPU fallback errors must propagate without a retry loop");
    assert_eq!(attempts.get(), 2);
    assert_eq!(error.to_string(), "original CPU error");
}

#[test]
fn a_later_proof_reinitializes_metal_and_reuses_a_successful_backend() {
    let cache = BackendCache::default();
    let builds = RefCell::new(Vec::new());
    for forced_failure in [Some(ForcedMetalFailure::Initialize), None, None] {
        with_backend_for_mode(
            &cache,
            BackendMode::MetalFull,
            1,
            forced_failure,
            |mode| {
                builds.borrow_mut().push(mode);
                Ok(AkitaBackend::optimized())
            },
            |_| Ok(()),
        )
        .expect("subsequent proofs must recover from a transient Metal error");
    }
    assert_eq!(*builds.borrow(), [BackendMode::Cpu, BackendMode::MetalFull]);
    assert_eq!(
        cache.borrow().as_ref().map(|(mode, _)| *mode),
        Some(BackendMode::MetalFull)
    );
}
