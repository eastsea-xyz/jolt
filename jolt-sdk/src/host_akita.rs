//! Host proving through the packed (Akita, lattice) commitment scheme.
//!
//! Enabled by the `akita` feature (optimized CPU backend) or `metal`
//! (Apple GPU commitment and PIOP kernels). The public surface mirrors the
//! Dory host path in `host_utils.rs`, so `#[jolt::provable]` output compiles
//! unchanged.
//!
//! Unlike Dory, an Akita setup is sized to one proof shape (padded trace
//! length and RAM size), which is only known after tracing. The SDK
//! preprocessing therefore keeps the shared program preprocessing plus a
//! cache of per-shape setups: the prover builds the setup for the shape it
//! derived from the trace, and the verifier builds (or reuses) the setup for
//! the shape the proof declares. Akita setups are transparent, so both sides
//! derive the same setup from the same shape. The verifier derives only its
//! own half ([`verifier_preprocess_full_with_advice`]): the public-matrix
//! prefix verification reads plus the shipped setup-prefix commitments, never
//! the prover's matrix.

use std::{
    cell::RefCell,
    collections::HashMap,
    error::Error as StdError,
    io::{Error as IoError, ErrorKind},
    rc::Rc,
    sync::{Arc, Mutex, PoisonError},
};

use jolt_prover::akita::preprocessing::{
    preprocess_committed_with_advice, preprocess_full_with_advice,
    verifier_preprocess_full_with_advice, AkitaProverPreprocessing, AkitaVerifierPreprocessing,
};
use jolt_prover::akita::JoltAkitaBackend;
use jolt_prover::telemetry;
use jolt_prover::ProverConfig;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend as WitnessTraceBackend};

use super::{
    host, program_preprocessing, BlindfoldSetup, CommitmentScheme, JoltDevice,
    JoltSharedPreprocessing, MemoryConfig, MemoryLayout, OwnedTrace, PreprocessingError,
    TraceInputs, TracerBackend,
};

pub use jolt_akita::{AkitaField as F, AkitaScheme as PCS};
pub use jolt_field::JoltField;
pub use jolt_prover::akita::preprocessing::{AkitaTranscript, AkitaVc};

pub type ProofTranscript = AkitaTranscript;
pub type VerifierPCS = PCS;
pub type VerifierVC = AkitaVc;
pub type VerifierField = F;
pub type VerifierTranscript = AkitaTranscript;
pub type RV64IMACProof = jolt_verifier::JoltProof<VerifierPCS, VerifierVC>;
pub type JoltProof = RV64IMACProof;
pub type ProgramPreprocessing = jolt_verifier::ProgramPreprocessing<PCS>;
pub type VerifierTrustedAdviceCommitment = jolt_akita::AkitaCommitment;
pub type TrustedAdviceOpeningHint = jolt_prover::akita::witness::AdviceObject<PCS>;

/// Prover and verifier preprocessing share one type under Akita: the program
/// plus the per-shape setup cache.
pub type JoltProverPreprocessing = AkitaPreprocessing;
pub type JoltVerifierPreprocessing = AkitaPreprocessing;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct ShapeKey {
    trace_length: usize,
    ram_k: usize,
    untrusted_advice: bool,
    trusted_advice: bool,
}

type SetupCache = Mutex<HashMap<ShapeKey, Arc<AkitaProverPreprocessing>>>;
type VerifierSetupCache = Mutex<HashMap<ShapeKey, Arc<AkitaVerifierPreprocessing>>>;

#[derive(Clone)]
pub struct AkitaPreprocessing {
    pub shared: JoltSharedPreprocessing,
    pub bytecode_chunk_count: Option<usize>,
    setups: Arc<SetupCache>,
    verifier_setups: Arc<VerifierSetupCache>,
}

impl AkitaPreprocessing {
    pub fn new(shared: JoltSharedPreprocessing, bytecode_chunk_count: Option<usize>) -> Self {
        Self {
            shared,
            bytecode_chunk_count,
            setups: Arc::default(),
            verifier_setups: Arc::default(),
        }
    }

    pub fn memory_layout(&self) -> &MemoryLayout {
        &self.shared.program.memory_layout
    }

    /// The Akita preprocessing (setup + verifier key) for one proof shape,
    /// built on first use and cached.
    pub fn setup_for(
        &self,
        config: &ProverConfig,
        untrusted_advice: bool,
        trusted_advice: bool,
        matrix_generator: Option<&jolt_akita::SetupMatrixGenerator<'_, F>>,
    ) -> Result<Arc<AkitaProverPreprocessing>, PreprocessingError> {
        let profile = telemetry::stage("shape.setup");
        let key = ShapeKey {
            trace_length: config.trace_length,
            ram_k: config.ram_K,
            untrusted_advice,
            trusted_advice,
        };
        let mut setups = self.setups.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(setup) = setups.get(&key) {
            profile.metadata("cache_hit", "true");
            return Ok(Arc::clone(setup));
        }
        profile.metadata("cache_hit", "false");
        let program = (*self.shared.program).clone();
        let setup = match self.bytecode_chunk_count {
            Some(chunks) => preprocess_committed_with_advice(
                program,
                config,
                chunks,
                untrusted_advice,
                trusted_advice,
                matrix_generator,
            )?,
            None => preprocess_full_with_advice(
                program,
                config,
                untrusted_advice,
                trusted_advice,
                matrix_generator,
            )?,
        };
        let setup = Arc::new(setup);
        let _ = setups.insert(key, Arc::clone(&setup));
        Ok(setup)
    }
}

impl AkitaPreprocessing {
    /// The verifier preprocessing for one proof shape, built on first use and
    /// cached.
    ///
    /// A shape this preprocessing already proved reuses the prover's verifier
    /// half. Otherwise a full program derives only the verifier setup
    /// ([`verifier_preprocess_full_with_advice`]), which never materializes
    /// the prover's public matrix. A committed program still goes through
    /// [`Self::setup_for`]: its verifier needs the program commitments, which
    /// only the prover path computes.
    pub fn verifier_setup_for(
        &self,
        config: &ProverConfig,
        untrusted_advice: bool,
        trusted_advice: bool,
    ) -> Result<Arc<AkitaVerifierPreprocessing>, PreprocessingError> {
        let key = ShapeKey {
            trace_length: config.trace_length,
            ram_k: config.ram_K,
            untrusted_advice,
            trusted_advice,
        };
        let mut verifiers = self
            .verifier_setups
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(verifier) = verifiers.get(&key) {
            return Ok(Arc::clone(verifier));
        }
        let proved = self
            .setups
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
            .map(Arc::clone);
        let verifier = match (proved, self.bytecode_chunk_count) {
            (Some(setup), _) => setup.verifier.clone(),
            (None, Some(_)) => {
                drop(verifiers);
                let setup = self.setup_for(config, untrusted_advice, trusted_advice, None)?;
                verifiers = self
                    .verifier_setups
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                setup.verifier.clone()
            }
            (None, None) => verifier_preprocess_full_with_advice(
                (*self.shared.program).clone(),
                config,
                untrusted_advice,
                trusted_advice,
            )?,
        };
        let verifier = Arc::new(verifier);
        let _ = verifiers.insert(key, Arc::clone(&verifier));
        Ok(verifier)
    }
}

pub fn preprocess_program(
    source: &mut dyn host::JoltProgramSource,
    memory_config: MemoryConfig,
    max_trace_length: usize,
    bytecode_chunk_count: Option<usize>,
) -> Result<JoltProverPreprocessing, PreprocessingError> {
    let program = program_preprocessing(source, memory_config, max_trace_length)?;
    Ok(AkitaPreprocessing::new(
        JoltSharedPreprocessing::new(program)?,
        bytecode_chunk_count,
    ))
}

pub fn verifier_preprocessing_from_prover(
    prover: &JoltProverPreprocessing,
) -> JoltVerifierPreprocessing {
    prover.clone()
}

pub fn prover_preprocessing_from_shared(
    shared: JoltSharedPreprocessing,
) -> JoltProverPreprocessing {
    AkitaPreprocessing::new(shared, None)
}

/// Akita setups are transparent and per-shape, so the supplied generators and
/// BlindFold setup are not used; the verifier derives its setup from the
/// shape each proof declares.
pub fn verifier_preprocessing_from_parts(
    shared: &JoltSharedPreprocessing,
    _generators: <PCS as CommitmentScheme>::VerifierSetup,
    _blindfold_setup: Option<BlindfoldSetup>,
) -> JoltVerifierPreprocessing {
    AkitaPreprocessing::new(shared.clone(), None)
}

pub fn commit_trusted_advice(
    _preprocessing: &JoltProverPreprocessing,
    _advice_bytes: &[u8],
) -> Result<(VerifierTrustedAdviceCommitment, TrustedAdviceOpeningHint), PreprocessingError> {
    Err(PreprocessingError::InvalidAdvice {
        reason: "trusted advice is not yet supported by the jolt-sdk Akita path".to_owned(),
    })
}

pub fn prover_memory_layout(preprocessing: &JoltProverPreprocessing) -> &MemoryLayout {
    preprocessing.memory_layout()
}

pub fn verifier_memory_layout(preprocessing: &JoltVerifierPreprocessing) -> &MemoryLayout {
    preprocessing.memory_layout()
}

type AkitaBackend = JoltAkitaBackend<F, PCS>;

/// Which kernels run on the GPU under the `metal` feature.
///
/// The Metal instruction-source ABI stores `pc + 1` in its own column, and
/// the shaders index bytecode with 32-bit PCs, so the full Metal backend
/// handles any bytecode below 2^32 entries. Larger programs fall back to
/// `MetalCommitOnly`: Metal trace commitment and opening, CPU PIOP.
/// `Hybrid` keeps the booleanity / bytecode / instruction read-raf slots on
/// the CPU and the rest on Metal.
/// `JOLT_AKITA_METAL=full|hybrid|commit|cpu` overrides the automatic choice.
/// Metal is automatic on Apple Silicon; other hosts default to CPU. A Metal
/// error discards the backend and retries the whole proof once on CPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BackendMode {
    Cpu,
    MetalFull,
    MetalHybrid,
    MetalCommitOnly,
}

/// Largest bytecode length the Metal instruction-source ABI can index.
const METAL_MAX_BYTECODE_LEN: usize = u32::MAX as usize - 1;

fn backend_mode(bytecode_len: usize) -> BackendMode {
    backend_mode_for_setting(
        bytecode_len,
        std::env::var("JOLT_AKITA_METAL").ok().as_deref(),
        cfg!(all(feature = "metal", target_os = "macos")),
        cfg!(target_arch = "aarch64"),
    )
}

fn backend_mode_for_setting(
    bytecode_len: usize,
    setting: Option<&str>,
    metal_supported: bool,
    apple_silicon: bool,
) -> BackendMode {
    if !metal_supported {
        return BackendMode::Cpu;
    }
    match setting {
        Some("full") => BackendMode::MetalFull,
        Some("hybrid") => BackendMode::MetalHybrid,
        Some("commit") => BackendMode::MetalCommitOnly,
        Some("cpu") => BackendMode::Cpu,
        _ if !apple_silicon => BackendMode::Cpu,
        _ if bytecode_len > METAL_MAX_BYTECODE_LEN => BackendMode::MetalCommitOnly,
        _ => BackendMode::MetalFull,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ForcedMetalFailure {
    Initialize,
    Prove,
}

impl ForcedMetalFailure {
    fn setting(self) -> &'static str {
        match self {
            Self::Initialize => "initialize",
            Self::Prove => "prove",
        }
    }

    fn error(self) -> Box<dyn StdError + Send + Sync> {
        telemetry::metadata("backend_forced_error", self.setting());
        IoError::other(format!(
            "forced Metal {} failure (JOLT_AKITA_METAL_FORCE_ERROR)",
            self.setting()
        ))
        .into()
    }
}

/// Qualification-only diagnostic: `prove` discards a completed Metal proof,
/// then exercises the same fresh CPU retry used for real runtime failures.
fn forced_metal_failure(
    mode: BackendMode,
    setting: Option<&std::ffi::OsStr>,
) -> Result<Option<ForcedMetalFailure>, Box<dyn StdError + Send + Sync>> {
    if mode == BackendMode::Cpu {
        return Ok(None);
    }
    match setting {
        None => Ok(None),
        Some(value) if value == "initialize" => Ok(Some(ForcedMetalFailure::Initialize)),
        Some(value) if value == "prove" => Ok(Some(ForcedMetalFailure::Prove)),
        Some(_) => Err(IoError::new(
            ErrorKind::InvalidInput,
            "JOLT_AKITA_METAL_FORCE_ERROR expects initialize|prove",
        )
        .into()),
    }
}

fn build_backend(mode: BackendMode) -> Result<AkitaBackend, Box<dyn StdError + Send + Sync>> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match mode {
            BackendMode::Cpu => Ok(AkitaBackend::optimized()),
            BackendMode::MetalFull => Ok(AkitaBackend::metal()?),
            BackendMode::MetalCommitOnly => Ok(AkitaBackend::metal_commit_only()?),
            BackendMode::MetalHybrid => {
                let cpu = AkitaBackend::optimized();
                let mut backend = AkitaBackend::metal()?;
                backend.base.instruction_read_raf = cpu.base.instruction_read_raf;
                backend.base.booleanity_address = cpu.base.booleanity_address;
                backend.base.booleanity_cycle = cpu.base.booleanity_cycle;
                backend.base.bytecode_read_raf_address = cpu.base.bytecode_read_raf_address;
                backend.base.bytecode_read_raf_cycle = cpu.base.bytecode_read_raf_cycle;
                Ok(backend)
            }
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = mode;
        Ok(AkitaBackend::optimized())
    }
}

type BackendCache = RefCell<Option<(BackendMode, Rc<AkitaBackend>)>>;

thread_local! {
    // The backend's kernel slots are not `Sync`, so it is cached per thread.
    // Metal initialization compiles pipelines and allocates device state, so
    // it is not repeated per proof.
    static BACKEND: BackendCache = const { RefCell::new(None) };
}

fn cached_backend(
    cache: &BackendCache,
    mode: BackendMode,
    bytecode_len: usize,
    build: &mut impl FnMut(BackendMode) -> Result<AkitaBackend, Box<dyn StdError + Send + Sync>>,
) -> Result<Rc<AkitaBackend>, Box<dyn StdError + Send + Sync>> {
    let mut cached = cache.borrow_mut();
    if let Some((cached_mode, backend)) = cached.as_ref() {
        if *cached_mode == mode {
            return Ok(Rc::clone(backend));
        }
    }
    // Release the previous device before allocating another backend.
    *cached = None;
    let _profile = telemetry::stage("backend.initialize");
    eprintln!("jolt-sdk: Akita prover backend {mode:?} (bytecode length {bytecode_len})");
    let backend = Rc::new(build(mode)?);
    *cached = Some((mode, Rc::clone(&backend)));
    Ok(backend)
}

fn with_backend_for_mode<R>(
    cache: &BackendCache,
    mode: BackendMode,
    bytecode_len: usize,
    forced_failure: Option<ForcedMetalFailure>,
    mut build: impl FnMut(BackendMode) -> Result<AkitaBackend, Box<dyn StdError + Send + Sync>>,
    mut f: impl FnMut(&AkitaBackend, BackendMode) -> Result<R, Box<dyn StdError + Send + Sync>>,
) -> Result<R, Box<dyn StdError + Send + Sync>> {
    telemetry::metadata("backend_requested_mode", format!("{mode:?}"));
    telemetry::metadata("backend_mode", format!("{mode:?}"));
    let attempt = (|| {
        if mode != BackendMode::Cpu && forced_failure == Some(ForcedMetalFailure::Initialize) {
            return Err(ForcedMetalFailure::Initialize.error());
        }
        let backend = cached_backend(cache, mode, bytecode_len, &mut build)?;
        let result = f(&backend, mode);
        if mode != BackendMode::Cpu
            && forced_failure == Some(ForcedMetalFailure::Prove)
            && result.is_ok()
        {
            // Drop all output from the Metal attempt before starting CPU work.
            drop(result);
            return Err(ForcedMetalFailure::Prove.error());
        }
        result
    })();
    match attempt {
        Ok(proof) => Ok(proof),
        Err(error) if mode == BackendMode::Cpu => Err(error),
        Err(error) => {
            // The attempt has returned, dropping its proof session/transcript
            // and local backend handle. Evict the remaining cached GPU owner.
            *cache.borrow_mut() = None;
            telemetry::metadata("backend_fallback_from", format!("{mode:?}"));
            telemetry::metadata("backend_fallback_error", error.to_string());
            telemetry::metadata("backend_mode", "Cpu");
            eprintln!("jolt-sdk: Akita {mode:?} failed: {error}; retrying the proof on CPU");
            let _profile = telemetry::stage("backend.cpu_retry");
            let cpu = cached_backend(cache, BackendMode::Cpu, bytecode_len, &mut build)?;
            // The closure only borrows the witness/setup. Calling prove again
            // creates a new transcript and session without cloning those inputs.
            f(&cpu, BackendMode::Cpu)
        }
    }
}

fn with_backend<R>(
    bytecode_len: usize,
    f: impl FnMut(&AkitaBackend, BackendMode) -> Result<R, Box<dyn StdError + Send + Sync>>,
) -> Result<R, Box<dyn StdError + Send + Sync>> {
    let mode = backend_mode(bytecode_len);
    let forced_failure = forced_metal_failure(
        mode,
        std::env::var_os("JOLT_AKITA_METAL_FORCE_ERROR").as_deref(),
    )?;
    BACKEND.with(|cache| {
        with_backend_for_mode(cache, mode, bytecode_len, forced_failure, build_backend, f)
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "SDK proof boundary mirrors guest inputs"
)]
pub fn prove_program(
    source: &dyn host::JoltProgramSource,
    preprocessing: &JoltProverPreprocessing,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    trusted_advice_commitment: Option<VerifierTrustedAdviceCommitment>,
    trusted_advice_hint: Option<TrustedAdviceOpeningHint>,
    advice_tape: Option<Vec<u8>>,
) -> Result<(RV64IMACProof, JoltDevice), Box<dyn StdError + Send + Sync>> {
    let _profile = telemetry::stage("sdk.prove_program");
    let program = {
        let _profile = telemetry::stage("guest_program.load");
        Arc::new(source.build_jolt_program()?)
    };
    let program_preprocessing = Arc::clone(&preprocessing.shared.program);
    let memory_layout = &program_preprocessing.memory_layout;
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
        max_trusted_advice_size: memory_layout.max_trusted_advice_size,
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        stack_size: memory_layout.stack_size,
        heap_size: memory_layout.heap_size,
        program_size: Some(memory_layout.program_size),
    };
    let trace_inputs = TraceInputs::new(
        inputs.to_vec(),
        untrusted_advice.to_vec(),
        trusted_advice.to_vec(),
        memory_config,
    )
    .with_advice_tape(advice_tape);
    #[cfg(not(feature = "field-inline"))]
    let trace = {
        let _profile = telemetry::stage("guest.execution_trace");
        TracerBackend::new().trace_compact(
            &program,
            trace_inputs,
            &program_preprocessing.bytecode,
        )?
    };
    #[cfg(feature = "field-inline")]
    let trace = {
        let _profile = telemetry::stage("guest.execution_trace");
        program.trace_with(&mut TracerBackend::new(), trace_inputs)?
    };
    #[cfg(not(feature = "field-inline"))]
    telemetry::metadata("raw_trace_rows", trace.trace.as_slice().len().to_string());
    #[cfg(all(feature = "metal", target_os = "macos", not(feature = "field-inline")))]
    if let Some(mode) = std::env::var_os("AETHER_PROVER_GPU_TRACE_REPLAY") {
        if mode != "1" {
            return Err("AETHER_PROVER_GPU_TRACE_REPLAY expects 1 for the diagnostic".into());
        }
        let _profile = telemetry::stage("trace.gpu_replay_prototype");
        let report = jolt_prover::trace_replay::diagnose(trace.trace.as_slice())?;
        telemetry::metadata("trace_gpu_replay_diagnostic", report.to_string());
        eprintln!("jolt-sdk: {report}");
    }
    #[cfg(feature = "field-inline")]
    telemetry::metadata("raw_trace_rows", trace.trace.rows().len().to_string());
    let public_io = trace.device.clone();
    #[cfg(not(feature = "field-inline"))]
    let config = {
        let _profile = telemetry::stage("configuration.derive");
        ProverConfig::derive_compact::<F>(
            trace.trace.as_slice(),
            memory_layout,
            program_preprocessing.ram.min_bytecode_address,
            program_preprocessing.ram.bytecode_words.len(),
            program_preprocessing.max_padded_trace_length,
        )?
    };
    #[cfg(feature = "field-inline")]
    let config = {
        let _profile = telemetry::stage("configuration.derive");
        ProverConfig::derive::<F>(
            trace.trace.rows(),
            memory_layout,
            program_preprocessing.ram.min_bytecode_address,
            program_preprocessing.ram.bytecode_words.len(),
            program_preprocessing.max_padded_trace_length,
        )?
    };
    telemetry::metadata("padded_trace_rows", config.trace_length.to_string());
    telemetry::metadata("ram_K", config.ram_K.to_string());
    telemetry::metadata(
        "max_padded_trace_rows",
        program_preprocessing.max_padded_trace_length.to_string(),
    );
    telemetry::metadata(
        "one_hot_chunk_bits",
        config.one_hot_config.committed_chunk_bits().to_string(),
    );
    let has_trusted_advice = trusted_advice_hint.is_some();
    if trusted_advice_commitment.is_some() != has_trusted_advice {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "trusted advice commitment and hint must be supplied together",
        )
        .into());
    }
    let has_untrusted_advice = !untrusted_advice.is_empty();
    let witness_config = JoltVmWitnessConfig::new(
        config.trace_length.ilog2() as usize,
        config.ram_K,
        config.one_hot_config,
    )
    .include_trusted_advice(has_trusted_advice)
    .include_untrusted_advice(has_untrusted_advice);
    let witness_inputs = JoltVmWitnessInputs::new(&program, &program_preprocessing, trace);
    #[cfg(not(feature = "field-inline"))]
    let witness = {
        let _profile = telemetry::stage("witness.trace_backend");
        WitnessTraceBackend::<OwnedTrace>::from_compact(witness_config, witness_inputs)
    };
    #[cfg(feature = "field-inline")]
    let witness = {
        let _profile = telemetry::stage("witness.trace_backend");
        WitnessTraceBackend::<OwnedTrace>::try_new(witness_config, witness_inputs)?
    };
    let proof = with_backend(program_preprocessing.bytecode.code_size, |backend, mode| {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let metal = if mode == BackendMode::MetalFull {
            backend.trace_commitment_backend().metal_backend()
        } else {
            None
        };
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let generate: &jolt_akita::SetupMatrixGenerator<'_, F> = &|field_count, seed| {
            // The factory runs in the dedicated backend pool. Its process
            // counters overlap the calling setup span, so this is a supplement.
            let profile = telemetry::asynchronous_stage("shape.matrix_gpu");
            let metal = metal.as_ref().ok_or_else(|| {
                jolt_akita::AkitaError::InvalidSetup(
                    "MetalFull setup has no Metal matrix producer".to_owned(),
                )
            })?;
            let output = metal
                .derive_public_matrix_prefix(field_count, seed)
                .map_err(|error| jolt_akita::AkitaError::InvalidSetup(error.to_string()))?;
            let metrics = &output.metrics;
            profile.metadata("field_count", field_count.to_string());
            profile.metadata("gpu_field_elements", metrics.gpu_field_elements.to_string());
            profile.metadata("pages", metrics.pages.to_string());
            profile.metadata("command_buffers", metrics.command_buffers.to_string());
            profile.metadata("output_bytes", metrics.output_bytes.to_string());
            profile.metadata(
                "peak_allocation_bytes",
                metrics.peak_allocation_bytes.to_string(),
            );
            profile.metadata("cpu_fallback", metrics.cpu_fallback.to_string());
            profile.metadata(
                "gpu_active_seconds",
                metrics.gpu_active_time.as_secs_f64().to_string(),
            );
            profile.metadata(
                "command_wall_seconds",
                metrics.command_wall_time.as_secs_f64().to_string(),
            );
            profile.metadata(
                "buffer_setup_seconds",
                metrics.buffer_setup_time.as_secs_f64().to_string(),
            );
            profile.metadata(
                "readback_seconds",
                metrics.readback_time.as_secs_f64().to_string(),
            );
            if let Some(reason) = &metrics.fallback_reason {
                profile.metadata("fallback_reason", reason);
            }
            Ok(output.matrix)
        };
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let matrix_generator = if mode == BackendMode::MetalFull {
            Some(generate)
        } else {
            None
        };
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let matrix_generator = {
            let _ = mode;
            None
        };
        // A failed GPU producer has not entered the shape cache; the existing
        // backend retry invokes this closure again with a fresh CPU backend.
        let setup = preprocessing.setup_for(
            &config,
            has_untrusted_advice,
            has_trusted_advice,
            matrix_generator,
        )?;
        Ok(jolt_prover::akita::prove::<
            F,
            PCS,
            VerifierVC,
            ProofTranscript,
            _,
        >(
            backend,
            &setup,
            &config,
            trusted_advice_hint.as_ref(),
            &witness,
            &public_io,
        )?)
    })?;
    Ok((proof, public_io))
}

pub fn verify_program(
    preprocessing: &JoltVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &RV64IMACProof,
    trusted_advice_commitment: Option<&VerifierTrustedAdviceCommitment>,
) -> Result<(), Box<dyn StdError + Send + Sync>> {
    let max_trace_length = preprocessing.shared.program.max_padded_trace_length;
    if !proof.trace_length.is_power_of_two() || proof.trace_length > max_trace_length {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "proof trace length is not a power of two within the program's maximum",
        )
        .into());
    }
    if !proof.ram_K.is_power_of_two() {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "proof RAM size is not a power of two",
        )
        .into());
    }
    // Only the configuration the prover derives for this shape is accepted:
    // the setup (and its cache entry) then follows from the shape alone.
    let config = ProverConfig::for_shape(proof.trace_length, proof.ram_K);
    if proof.rw_config != config.rw_config
        || proof.one_hot_config != config.one_hot_config
        || proof.trace_polynomial_order != config.trace_polynomial_order
    {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "proof declares a chunking configuration other than the canonical one for its shape",
        )
        .into());
    }
    let verifier = preprocessing.verifier_setup_for(
        &config,
        proof.untrusted_advice_commitment.is_some(),
        trusted_advice_commitment.is_some(),
    )?;
    jolt_verifier::verify::<VerifierField, VerifierPCS, VerifierVC, VerifierTranscript>(
        &verifier,
        public_io,
        proof,
        trusted_advice_commitment,
    )?;
    Ok(())
}

#[cfg(test)]
#[path = "host_akita_tests.rs"]
mod tests;
