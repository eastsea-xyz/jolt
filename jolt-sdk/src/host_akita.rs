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
//! derive the same setup from the same shape.

use std::{
    cell::RefCell,
    collections::HashMap,
    error::Error as StdError,
    io::{Error as IoError, ErrorKind},
    rc::Rc,
    sync::{Arc, Mutex, PoisonError},
};

use jolt_prover::akita::preprocessing::{
    preprocess_committed_with_advice, preprocess_full_with_advice, AkitaProverPreprocessing,
};
use jolt_prover::akita::JoltAkitaBackend;
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

#[derive(Clone)]
pub struct AkitaPreprocessing {
    pub shared: JoltSharedPreprocessing,
    pub bytecode_chunk_count: Option<usize>,
    setups: Arc<SetupCache>,
}

impl AkitaPreprocessing {
    pub fn new(shared: JoltSharedPreprocessing, bytecode_chunk_count: Option<usize>) -> Self {
        Self {
            shared,
            bytecode_chunk_count,
            setups: Arc::default(),
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
    ) -> Result<Arc<AkitaProverPreprocessing>, PreprocessingError> {
        let key = ShapeKey {
            trace_length: config.trace_length,
            ram_k: config.ram_K,
            untrusted_advice,
            trusted_advice,
        };
        let mut setups = self.setups.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(setup) = setups.get(&key) {
            return Ok(Arc::clone(setup));
        }
        let program = (*self.shared.program).clone();
        let setup = match self.bytecode_chunk_count {
            Some(chunks) => preprocess_committed_with_advice(
                program,
                config,
                chunks,
                untrusted_advice,
                trusted_advice,
            )?,
            None => preprocess_full_with_advice(program, config, untrusted_advice, trusted_advice)?,
        };
        let setup = Arc::new(setup);
        let _ = setups.insert(key, Arc::clone(&setup));
        Ok(setup)
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
/// The Metal instruction-source ABI packs `pc + 1` into 14 bits, so the
/// Metal PIOP kernels reject programs whose bytecode has 16383 or more
/// entries ("booleanity row cannot be represented by the packed Metal ABI").
/// The rows are built as early as stage 1 (Spartan outer publishes the
/// instruction read-raf source), so for such programs the automatic choice
/// is `MetalCommitOnly`: Metal trace commitment and opening, CPU PIOP.
/// `Hybrid` (booleanity / bytecode / instruction read-raf slots on CPU, the
/// rest on Metal) works at 2^23 but still hits the ABI at 2^24.
/// `JOLT_AKITA_METAL=full|hybrid|commit|cpu` overrides the automatic choice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BackendMode {
    Cpu,
    MetalFull,
    MetalHybrid,
    MetalCommitOnly,
}

/// Largest bytecode length the Metal instruction-source ABI can index.
const METAL_MAX_BYTECODE_LEN: usize = (1 << 14) - 2;

fn backend_mode(bytecode_len: usize) -> BackendMode {
    if !cfg!(all(feature = "metal", target_os = "macos")) {
        return BackendMode::Cpu;
    }
    match std::env::var("JOLT_AKITA_METAL").as_deref() {
        Ok("full") => BackendMode::MetalFull,
        Ok("hybrid") => BackendMode::MetalHybrid,
        Ok("commit") => BackendMode::MetalCommitOnly,
        Ok("cpu") => BackendMode::Cpu,
        _ if bytecode_len > METAL_MAX_BYTECODE_LEN => BackendMode::MetalCommitOnly,
        _ => BackendMode::MetalFull,
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

thread_local! {
    // The backend's kernel slots are not `Sync`, so it is cached per thread.
    // Metal initialization compiles pipelines and allocates device state, so
    // it is not repeated per proof.
    static BACKEND: RefCell<Option<(BackendMode, Rc<AkitaBackend>)>> = const { RefCell::new(None) };
}

fn with_backend<R>(
    bytecode_len: usize,
    f: impl FnOnce(&AkitaBackend) -> Result<R, Box<dyn StdError + Send + Sync>>,
) -> Result<R, Box<dyn StdError + Send + Sync>> {
    let mode = backend_mode(bytecode_len);
    let backend = BACKEND.with(|cell| -> Result<_, Box<dyn StdError + Send + Sync>> {
        let mut cached = cell.borrow_mut();
        match cached.as_ref() {
            Some((cached_mode, backend)) if *cached_mode == mode => Ok(Rc::clone(backend)),
            _ => {
                eprintln!(
                    "jolt-sdk: Akita prover backend {mode:?} (bytecode length {bytecode_len})"
                );
                let backend = Rc::new(build_backend(mode)?);
                *cached = Some((mode, Rc::clone(&backend)));
                Ok(backend)
            }
        }
    })?;
    f(&backend)
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
    let program = Arc::new(source.build_jolt_program()?);
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
    let trace = TracerBackend::new().trace_compact(
        &program,
        trace_inputs,
        &program_preprocessing.bytecode,
    )?;
    #[cfg(feature = "field-inline")]
    let trace = program.trace_with(&mut TracerBackend::new(), trace_inputs)?;
    let public_io = trace.device.clone();
    #[cfg(not(feature = "field-inline"))]
    let config = ProverConfig::derive_compact::<F>(
        trace.trace.as_slice(),
        memory_layout,
        program_preprocessing.ram.min_bytecode_address,
        program_preprocessing.ram.bytecode_words.len(),
        program_preprocessing.max_padded_trace_length,
    )?;
    #[cfg(feature = "field-inline")]
    let config = ProverConfig::derive::<F>(
        trace.trace.rows(),
        memory_layout,
        program_preprocessing.ram.min_bytecode_address,
        program_preprocessing.ram.bytecode_words.len(),
        program_preprocessing.max_padded_trace_length,
    )?;
    let has_trusted_advice = trusted_advice_hint.is_some();
    if trusted_advice_commitment.is_some() != has_trusted_advice {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            "trusted advice commitment and hint must be supplied together",
        )
        .into());
    }
    let has_untrusted_advice = !untrusted_advice.is_empty();
    let setup = preprocessing.setup_for(&config, has_untrusted_advice, has_trusted_advice)?;
    let witness_config = JoltVmWitnessConfig::new(
        config.trace_length.ilog2() as usize,
        config.ram_K,
        config.one_hot_config,
    )
    .include_trusted_advice(has_trusted_advice)
    .include_untrusted_advice(has_untrusted_advice);
    let witness_inputs = JoltVmWitnessInputs::new(&program, &program_preprocessing, trace);
    #[cfg(not(feature = "field-inline"))]
    let witness = WitnessTraceBackend::<OwnedTrace>::from_compact(witness_config, witness_inputs);
    #[cfg(feature = "field-inline")]
    let witness = WitnessTraceBackend::<OwnedTrace>::try_new(witness_config, witness_inputs)?;
    let proof = with_backend(program_preprocessing.bytecode.code_size, |backend| {
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
    let config = ProverConfig {
        trace_length: proof.trace_length,
        ram_K: proof.ram_K,
        rw_config: proof.rw_config,
        one_hot_config: proof.one_hot_config,
        trace_polynomial_order: proof.trace_polynomial_order,
    };
    let setup = preprocessing.setup_for(
        &config,
        proof.untrusted_advice_commitment.is_some(),
        trusted_advice_commitment.is_some(),
    )?;
    jolt_verifier::verify::<VerifierField, VerifierPCS, VerifierVC, VerifierTranscript>(
        &setup.verifier,
        public_io,
        proof,
        trusted_advice_commitment,
    )?;
    Ok(())
}
