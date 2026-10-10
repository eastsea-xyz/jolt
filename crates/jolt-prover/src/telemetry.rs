//! Opt-in, process-wide proof timing. CPU time includes all Rayon and worker
//! threads; asynchronous supplements describe overlap and are never added to
//! the accounted total. GPU time is the sum of completed command-buffer
//! durations, so pending buffers explicitly expose stage-boundary ambiguity.
//! The separate GPU busy total unions raw intervals across both Metal backends.
//! Memory peaks are sampled lower bounds over inclusive stage intervals;
//! overlapping gauges and peaks must never be added or subtracted.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    marker::PhantomData,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, RecvTimeoutError, Sender},
        Arc, Mutex, PoisonError,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use jolt_kernels::profiling_memory::{MemoryProbe, MemorySnapshot};
use serde::Serialize;

pub use tracing;

static ENABLED: AtomicBool = AtomicBool::new(false);
static GENERATION: AtomicU64 = AtomicU64::new(0);
static ACTIVE: Mutex<Option<ActiveProfile>> = Mutex::new(None);
const MEMORY_SAMPLE_INTERVAL: Duration = Duration::from_millis(25);

thread_local! {
    static STACK: RefCell<Vec<(u64, usize)>> = const { RefCell::new(Vec::new()) };
}

/// Durations in seconds. A CPU core-second is one core occupied for one second.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ProfileMetrics {
    pub wall_seconds: f64,
    pub cpu_core_seconds: Option<f64>,
    pub gpu_seconds: f64,
    pub akita_gpu_seconds: f64,
    pub piop_gpu_seconds: f64,
    pub gpu_command_buffers: u64,
    pub gpu_submitted_command_buffers: u64,
    pub gpu_timestamp_unavailable: u64,
    pub gpu_pending_at_start: u64,
    pub gpu_pending_at_end: u64,
}

impl ProfileMetrics {
    fn subtract(&mut self, child: Self) {
        self.wall_seconds = (self.wall_seconds - child.wall_seconds).max(0.0);
        self.cpu_core_seconds = self
            .cpu_core_seconds
            .zip(child.cpu_core_seconds)
            .map(|(parent, child)| (parent - child).max(0.0));
        self.gpu_seconds = (self.gpu_seconds - child.gpu_seconds).max(0.0);
        self.akita_gpu_seconds = (self.akita_gpu_seconds - child.akita_gpu_seconds).max(0.0);
        self.piop_gpu_seconds = (self.piop_gpu_seconds - child.piop_gpu_seconds).max(0.0);
        self.gpu_command_buffers = self
            .gpu_command_buffers
            .saturating_sub(child.gpu_command_buffers);
        self.gpu_submitted_command_buffers = self
            .gpu_submitted_command_buffers
            .saturating_sub(child.gpu_submitted_command_buffers);
        self.gpu_timestamp_unavailable = self
            .gpu_timestamp_unavailable
            .saturating_sub(child.gpu_timestamp_unavailable);
    }
}

/// Missing observations for each gauge, including partial query failures.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct MemoryUnavailableSamples {
    pub rss: u64,
    pub physical_footprint: u64,
    pub metal_allocated: u64,
}

/// Inclusive temporal observations, not causal allocations of a stage.
/// Every sampled peak is a LOWER BOUND: a spike can occur between samples.
/// Unified-memory gauges overlap and are never additive or subtractable.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ProfileMemory {
    pub sampled_peak_rss_bytes: Option<u64>,
    pub sampled_peak_physical_footprint_bytes: Option<u64>,
    /// Aggregate resources on the system-default Metal device, counted once
    /// across Akita and Jolt. This is not a buffer-only or separate VRAM gauge.
    pub sampled_peak_metal_allocated_bytes: Option<u64>,
    pub sample_count: u64,
    pub unavailable_samples: MemoryUnavailableSamples,
}

#[derive(Debug, Default, Serialize)]
pub struct ProofProfileMemory {
    #[serde(flatten)]
    pub sampled: ProfileMemory,
    /// Requested period; scheduling and boundary samples affect actual spacing.
    pub sample_interval_seconds: f64,
    pub periodic_sampler_started: bool,
    /// OS high-water marks cover the process lifetime, including earlier warm
    /// requests. They are conservative for this proof and are not stage peaks.
    pub process_peak_rss_bytes: Option<u64>,
    pub process_peak_physical_footprint_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ProfileStage {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub name: String,
    /// Supplements share process CPU/GPU counters with concurrent stages.
    /// Their durations are descriptive, and must not be summed with stages.
    pub supplement: bool,
    pub complete: bool,
    pub start_wall_seconds: f64,
    pub end_wall_seconds: f64,
    pub inclusive: ProfileMetrics,
    pub exclusive: ProfileMetrics,
    pub memory: ProfileMemory,
    pub wall_percent_of_total: f64,
    pub cpu_percent_of_total: Option<f64>,
    /// IDs of supplemental intervals that overlap this interval.
    pub overlapping_supplements: Vec<usize>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct ProofProfile {
    /// Approximate wall-clock origin for correlating external allocation logs.
    pub profile_started_unix_seconds: f64,
    /// Actual sampled gauges in chronological order; values overlap on unified memory.
    pub memory_timeline: Vec<ProfileMemorySample>,
    pub schema_version: u32,
    pub total: ProfileMetrics,
    /// Completed Metal intervals in host mach-time seconds, [start, end].
    pub gpu_command_buffer_intervals: Vec<[f64; 2]>,
    /// Union duration across both sources; overlapping buffers count once.
    pub gpu_busy_seconds: f64,
    pub memory: ProofProfileMemory,
    pub stages: Vec<ProfileStage>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct ProfileMemorySample {
    pub wall_seconds: f64,
    pub rss_bytes: Option<u64>,
    pub physical_footprint_bytes: Option<u64>,
    pub metal_allocated_bytes: Option<u64>,
}

#[derive(Clone, Copy, Default)]
struct GpuSnapshot {
    akita_seconds: f64,
    piop_seconds: f64,
    command_buffers: u64,
    submitted_command_buffers: u64,
    timestamp_unavailable: u64,
    pending_command_buffers: u64,
}

#[derive(Clone, Copy)]
struct Sample {
    wall: Instant,
    cpu_seconds: Option<f64>,
    gpu: GpuSnapshot,
}

impl Sample {
    fn capture() -> Self {
        Self {
            wall: Instant::now(),
            cpu_seconds: process_cpu_seconds(),
            gpu: gpu_snapshot(),
        }
    }

    fn elapsed(self, end: Self) -> ProfileMetrics {
        let akita_gpu_seconds = (end.gpu.akita_seconds - self.gpu.akita_seconds).max(0.0);
        let piop_gpu_seconds = (end.gpu.piop_seconds - self.gpu.piop_seconds).max(0.0);
        ProfileMetrics {
            wall_seconds: end.wall.saturating_duration_since(self.wall).as_secs_f64(),
            cpu_core_seconds: end
                .cpu_seconds
                .zip(self.cpu_seconds)
                .map(|(end, start)| (end - start).max(0.0)),
            gpu_seconds: akita_gpu_seconds + piop_gpu_seconds,
            akita_gpu_seconds,
            piop_gpu_seconds,
            gpu_command_buffers: end
                .gpu
                .command_buffers
                .saturating_sub(self.gpu.command_buffers),
            gpu_submitted_command_buffers: end
                .gpu
                .submitted_command_buffers
                .saturating_sub(self.gpu.submitted_command_buffers),
            gpu_timestamp_unavailable: end
                .gpu
                .timestamp_unavailable
                .saturating_sub(self.gpu.timestamp_unavailable),
            gpu_pending_at_start: self.gpu.pending_command_buffers,
            gpu_pending_at_end: end.gpu.pending_command_buffers,
        }
    }
}

struct ActiveStage {
    parent_id: Option<usize>,
    name: String,
    supplement: bool,
    start: Sample,
    end: Option<Sample>,
    metadata: BTreeMap<String, String>,
}

struct ActiveProfile {
    generation: u64,
    start: Sample,
    stages: Vec<ActiveStage>,
    metadata: BTreeMap<String, String>,
    memory: Option<MemorySampler>,
}

impl ActiveProfile {
    fn finish(
        mut self,
        end: Sample,
        mut gpu_command_buffer_intervals: Vec<[f64; 2]>,
    ) -> ProofProfile {
        let total = self.start.elapsed(end);
        gpu_command_buffer_intervals.retain(|interval| valid_gpu_interval(*interval));
        let gpu_busy_seconds = gpu_busy_union_seconds(&gpu_command_buffer_intervals);
        let periodic_sampler_started = self
            .memory
            .as_ref()
            .is_some_and(|sampler| sampler.worker.is_some());
        let mut samples = self.memory.take().map_or_else(Vec::new, |sampler| {
            sampler.capture_at(end.wall);
            sampler.finish()
        });
        samples.sort_unstable_by_key(|sample| sample.wall);
        let profile_started_unix_seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |duration| duration.as_secs_f64())
            - self.start.wall.elapsed().as_secs_f64();
        let memory_timeline = samples
            .iter()
            .map(|sample| ProfileMemorySample {
                wall_seconds: sample
                    .wall
                    .saturating_duration_since(self.start.wall)
                    .as_secs_f64(),
                rss_bytes: sample.snapshot.rss_bytes,
                physical_footprint_bytes: sample.snapshot.physical_footprint_bytes,
                metal_allocated_bytes: sample.snapshot.metal_allocated_bytes,
            })
            .collect();
        let memory_index = MemoryIndex::new(samples);
        let proof_memory = memory_index.range(self.start.wall, end.wall);
        let mut stages: Vec<ProfileStage> = self
            .stages
            .into_iter()
            .enumerate()
            .map(|(id, stage)| {
                let stage_end = stage.end.unwrap_or(end);
                let inclusive = stage.start.elapsed(stage_end);
                ProfileStage {
                    id,
                    parent_id: stage.parent_id,
                    name: stage.name,
                    supplement: stage.supplement,
                    complete: stage.end.is_some(),
                    start_wall_seconds: stage
                        .start
                        .wall
                        .saturating_duration_since(self.start.wall)
                        .as_secs_f64(),
                    end_wall_seconds: stage_end
                        .wall
                        .saturating_duration_since(self.start.wall)
                        .as_secs_f64(),
                    inclusive,
                    exclusive: inclusive,
                    memory: memory_index.range(stage.start.wall, stage_end.wall).sampled,
                    wall_percent_of_total: 0.0,
                    cpu_percent_of_total: None,
                    overlapping_supplements: Vec::new(),
                    metadata: stage.metadata,
                }
            })
            .collect();
        let children: Vec<_> = stages
            .iter()
            .filter(|stage| !stage.supplement)
            .filter_map(|stage| stage.parent_id.map(|parent| (parent, stage.inclusive)))
            .collect();
        for (parent, child) in children {
            if let Some(stage) = stages.get_mut(parent) {
                stage.exclusive.subtract(child);
            }
        }
        let supplements: Vec<_> = stages
            .iter()
            .filter(|stage| stage.supplement)
            .map(|stage| (stage.id, stage.start_wall_seconds, stage.end_wall_seconds))
            .collect();
        for stage in &mut stages {
            stage.wall_percent_of_total = if total.wall_seconds > 0.0 {
                100.0 * stage.exclusive.wall_seconds / total.wall_seconds
            } else {
                0.0
            };
            stage.cpu_percent_of_total = stage
                .exclusive
                .cpu_core_seconds
                .zip(total.cpu_core_seconds)
                .filter(|(_, total)| *total > 0.0)
                .map(|(stage, total)| 100.0 * stage / total);
            stage.overlapping_supplements = supplements
                .iter()
                .filter(|(id, start, end)| {
                    *id != stage.id
                        && *start < stage.end_wall_seconds
                        && *end > stage.start_wall_seconds
                })
                .map(|(id, _, _)| *id)
                .collect();
        }
        ProofProfile {
            profile_started_unix_seconds,
            memory_timeline,
            schema_version: 1,
            total,
            gpu_command_buffer_intervals,
            gpu_busy_seconds,
            memory: ProofProfileMemory {
                sampled: proof_memory.sampled,
                sample_interval_seconds: MEMORY_SAMPLE_INTERVAL.as_secs_f64(),
                periodic_sampler_started,
                process_peak_rss_bytes: proof_memory.process_peak_rss_bytes,
                process_peak_physical_footprint_bytes: proof_memory
                    .process_peak_physical_footprint_bytes,
            },
            stages,
            metadata: self.metadata,
        }
    }
}

#[derive(Clone, Copy)]
struct MemorySample {
    wall: Instant,
    snapshot: MemorySnapshot,
}

struct MemoryRecording {
    probe: MemoryProbe,
    samples: Vec<MemorySample>,
}

impl MemoryRecording {
    fn capture_at(&mut self, wall: Instant) {
        self.samples.push(MemorySample {
            wall,
            snapshot: self.probe.snapshot(),
        });
    }
}

/// Owned by the active profile. Normal finish and caller panic cleanup both
/// drop this owner, which wakes and joins the sampler before freeing its probe.
struct MemorySampler {
    recording: Arc<Mutex<MemoryRecording>>,
    stop: Sender<()>,
    worker: Option<JoinHandle<()>>,
}

impl MemorySampler {
    fn new(start: Instant) -> Self {
        let recording = Arc::new(Mutex::new(MemoryRecording {
            probe: MemoryProbe::new(),
            samples: Vec::new(),
        }));
        recording
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .capture_at(start);
        let worker_recording = Arc::clone(&recording);
        let (stop, receiver) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("jolt-memory-profile".to_owned())
            .spawn(move || {
                while let Err(RecvTimeoutError::Timeout) =
                    receiver.recv_timeout(MEMORY_SAMPLE_INTERVAL)
                {
                    worker_recording
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .capture_at(Instant::now());
                }
            })
            .ok();
        Self {
            recording,
            stop,
            worker,
        }
    }

    fn capture_at(&self, wall: Instant) {
        self.recording
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .capture_at(wall);
    }

    fn stop(&mut self) {
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }

    fn finish(mut self) -> Vec<MemorySample> {
        self.stop();
        let mut recording = self
            .recording
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let samples = std::mem::take(&mut recording.samples);
        drop(recording);
        samples
    }
}

impl Drop for MemorySampler {
    fn drop(&mut self) {
        self.stop();
    }
}

#[derive(Clone, Copy, Default)]
struct MemoryAggregate {
    sampled: ProfileMemory,
    process_peak_rss_bytes: Option<u64>,
    process_peak_physical_footprint_bytes: Option<u64>,
}

impl MemoryAggregate {
    fn from_snapshot(snapshot: MemorySnapshot) -> Self {
        Self {
            sampled: ProfileMemory {
                sampled_peak_rss_bytes: snapshot.rss_bytes,
                sampled_peak_physical_footprint_bytes: snapshot.physical_footprint_bytes,
                sampled_peak_metal_allocated_bytes: snapshot.metal_allocated_bytes,
                sample_count: 1,
                unavailable_samples: MemoryUnavailableSamples {
                    rss: u64::from(snapshot.rss_bytes.is_none()),
                    physical_footprint: u64::from(snapshot.physical_footprint_bytes.is_none()),
                    metal_allocated: u64::from(snapshot.metal_allocated_bytes.is_none()),
                },
            },
            process_peak_rss_bytes: snapshot.process_peak_rss_bytes,
            process_peak_physical_footprint_bytes: snapshot.process_peak_physical_footprint_bytes,
        }
    }

    fn merge(self, other: Self) -> Self {
        let left = self.sampled;
        let right = other.sampled;
        Self {
            sampled: ProfileMemory {
                sampled_peak_rss_bytes: left
                    .sampled_peak_rss_bytes
                    .max(right.sampled_peak_rss_bytes),
                sampled_peak_physical_footprint_bytes: left
                    .sampled_peak_physical_footprint_bytes
                    .max(right.sampled_peak_physical_footprint_bytes),
                sampled_peak_metal_allocated_bytes: left
                    .sampled_peak_metal_allocated_bytes
                    .max(right.sampled_peak_metal_allocated_bytes),
                sample_count: left.sample_count + right.sample_count,
                unavailable_samples: MemoryUnavailableSamples {
                    rss: left.unavailable_samples.rss + right.unavailable_samples.rss,
                    physical_footprint: left.unavailable_samples.physical_footprint
                        + right.unavailable_samples.physical_footprint,
                    metal_allocated: left.unavailable_samples.metal_allocated
                        + right.unavailable_samples.metal_allocated,
                },
            },
            process_peak_rss_bytes: self
                .process_peak_rss_bytes
                .max(other.process_peak_rss_bytes),
            process_peak_physical_footprint_bytes: self
                .process_peak_physical_footprint_bytes
                .max(other.process_peak_physical_footprint_bytes),
        }
    }
}

/// A range-max/count index avoids a samples-times-stages scan when PIOP emits
/// thousands of spans. Construction is O(n log n); each stage query O(log n).
#[derive(Default)]
struct MemoryIndex {
    times: Vec<Instant>,
    tree: Vec<MemoryAggregate>,
    leaf_start: usize,
}

impl MemoryIndex {
    fn new(mut samples: Vec<MemorySample>) -> Self {
        samples.sort_unstable_by_key(|sample| sample.wall);
        let leaf_start = samples.len().next_power_of_two();
        let mut tree = vec![MemoryAggregate::default(); 2 * leaf_start];
        let mut times = Vec::with_capacity(samples.len());
        for (index, sample) in samples.into_iter().enumerate() {
            times.push(sample.wall);
            tree[leaf_start + index] = MemoryAggregate::from_snapshot(sample.snapshot);
        }
        for index in (1..leaf_start).rev() {
            tree[index] = tree[2 * index].merge(tree[2 * index + 1]);
        }
        Self {
            times,
            tree,
            leaf_start,
        }
    }

    fn range(&self, start: Instant, end: Instant) -> MemoryAggregate {
        if start > end || self.times.is_empty() {
            return MemoryAggregate::default();
        }
        // Both stage boundaries are inclusive. Samples retain the same host
        // Instant origin as stage timing, including concurrent supplements.
        let mut left = self.leaf_start + self.times.partition_point(|time| *time < start);
        let mut right = self.leaf_start + self.times.partition_point(|time| *time <= end);
        let mut result = MemoryAggregate::default();
        while left < right {
            if left % 2 == 1 {
                result = result.merge(self.tree[left]);
                left += 1;
            }
            if right % 2 == 1 {
                right -= 1;
                result = result.merge(self.tree[right]);
            }
            left /= 2;
            right /= 2;
        }
        result
    }
}

/// Start a fresh profile, enabling both GPU counter sources. One profile may
/// be active per process; normal proving never starts one implicitly.
pub fn start() -> Result<(), &'static str> {
    let mut active = ACTIVE.lock().unwrap_or_else(PoisonError::into_inner);
    if active.is_some() {
        return Err("a proof profile is already active in this process");
    }
    set_gpu_enabled(true);
    let start = Sample::capture();
    let generation = GENERATION.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    let mut metadata = BTreeMap::new();
    let _ = metadata.insert(
        "cpu_clock".to_owned(),
        "CLOCK_PROCESS_CPUTIME_ID; includes all process threads".to_owned(),
    );
    let _ = metadata.insert("gpu_clock".to_owned(), "total.gpu_seconds and stage gpu_seconds sum completed Metal command-buffer durations attributed at observed completion; pending buffers can cross stage boundaries".to_owned());
    let _ = metadata.insert("gpu_busy_clock".to_owned(), "gpu_busy_seconds unions gpu_command_buffer_intervals across Akita and PIOP; raw GPUStartTime/GPUEndTime use host mach-time seconds".to_owned());
    let _ = metadata.insert("accounting".to_owned(), "sum exclusive normal stages only; asynchronous supplements share process counters and are excluded".to_owned());
    let _ = metadata.insert("memory_sampling".to_owned(), "sampled peaks are LOWER BOUNDS; boundary and requested 25 ms periodic samples use the stage host Instant clock; stage memory is inclusive temporal observation, not causal allocation".to_owned());
    let _ = metadata.insert("memory_gauges".to_owned(), "RSS, physical footprint, and Metal allocated resource bytes overlap on unified memory; never add or subtract peaks. Metal samples one cached system-default device once across Akita/Jolt, excludes other devices, and is not buffer-only or separate GPU RAM".to_owned());
    let _ = metadata.insert("memory_high_water_scope".to_owned(), "process lifetime; Darwin getrusage(RUSAGE_SELF).ru_maxrss bytes and proc_pid_rusage(RUSAGE_INFO_V4).ri_lifetime_max_phys_footprint; include previous warm requests; missing APIs remain null".to_owned());
    *active = Some(ActiveProfile {
        generation,
        start,
        stages: Vec::new(),
        metadata,
        memory: Some(MemorySampler::new(start.wall)),
    });
    ENABLED.store(true, Ordering::Release);
    Ok(())
}

/// Add metadata to the active profile; a no-op when profiling is inactive.
pub fn metadata(key: impl Into<String>, value: impl Into<String>) {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    if let Some(active) = ACTIVE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_mut()
    {
        let _ = active.metadata.insert(key.into(), value.into());
    }
}

/// Time a named, nested interval. Drop the guard at the region boundary.
pub fn stage(name: impl Into<String>) -> StageGuard {
    StageGuard::new(name, false)
}

/// Time asynchronous work as a supplementary interval. Its process CPU/GPU
/// deltas also include concurrent work and are excluded from stage totals.
pub fn asynchronous_stage(name: impl Into<String>) -> StageGuard {
    StageGuard::new(name, true)
}

/// Finish and take the profile after proof serialization, disabling counters.
/// Unclosed intervals are explicitly reported with `complete = false`.
pub fn finish() -> Option<ProofProfile> {
    if !ENABLED.swap(false, Ordering::AcqRel) {
        return None;
    }
    let end = Sample::capture();
    set_gpu_enabled(false);
    let gpu_command_buffer_intervals = take_gpu_completed_intervals();
    ACTIVE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
        .map(|active| active.finish(end, gpu_command_buffer_intervals))
}

/// A stage guard stays on its originating thread so nesting remains exact.
pub struct StageGuard {
    token: Option<(u64, usize)>,
    _thread: PhantomData<Rc<()>>,
}

impl StageGuard {
    fn new(name: impl Into<String>, supplement: bool) -> Self {
        let mut guard = Self {
            token: None,
            _thread: PhantomData,
        };
        if !ENABLED.load(Ordering::Acquire) {
            return guard;
        }
        let mut active = ACTIVE.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(active) = active.as_mut() else {
            return guard;
        };
        let start = Sample::capture();
        if let Some(memory) = &active.memory {
            memory.capture_at(start.wall);
        }
        let generation = active.generation;
        let id = active.stages.len();
        let parent_id = STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.retain(|(old_generation, _)| *old_generation == generation);
            let parent = stack.last().map(|(_, id)| *id);
            stack.push((generation, id));
            parent
        });
        let supplement = supplement
            || parent_id
                .and_then(|parent| active.stages.get(parent))
                .is_some_and(|parent| parent.supplement);
        active.stages.push(ActiveStage {
            parent_id,
            name: name.into(),
            supplement,
            start,
            end: None,
            metadata: BTreeMap::new(),
        });
        guard.token = Some((generation, id));
        guard
    }

    pub fn metadata(&self, key: impl Into<String>, value: impl Into<String>) {
        let Some((generation, id)) = self.token else {
            return;
        };
        let mut active = ACTIVE.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(active) = active
            .as_mut()
            .filter(|active| active.generation == generation)
        {
            if let Some(stage) = active.stages.get_mut(id) {
                let _ = stage.metadata.insert(key.into(), value.into());
            }
        }
    }
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        let Some((generation, id)) = self.token else {
            return;
        };
        STACK.with(|stack| {
            stack
                .borrow_mut()
                .retain(|token| *token != (generation, id));
        });
        let mut active = ACTIVE.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(active) = active
            .as_mut()
            .filter(|active| active.generation == generation)
        {
            let end = Sample::capture();
            if let Some(memory) = &active.memory {
                memory.capture_at(end.wall);
            }
            if let Some(stage) = active.stages.get_mut(id) {
                stage.end = Some(end);
            }
        }
    }
}

fn set_gpu_enabled(enabled: bool) {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        jolt_akita::metal_profiling::set_enabled(enabled);
        jolt_kernels::metal_profiling::set_enabled(enabled);
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let _ = enabled;
}

fn gpu_snapshot() -> GpuSnapshot {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let akita = jolt_akita::metal_profiling::snapshot();
        let piop = jolt_kernels::metal_profiling::snapshot();
        GpuSnapshot {
            akita_seconds: akita.gpu_seconds,
            piop_seconds: piop.gpu_seconds,
            command_buffers: akita.command_buffers + piop.command_buffers,
            submitted_command_buffers: akita.submitted_command_buffers
                + piop.submitted_command_buffers,
            timestamp_unavailable: akita.timestamp_unavailable + piop.timestamp_unavailable,
            pending_command_buffers: akita.pending_command_buffers + piop.pending_command_buffers,
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        GpuSnapshot::default()
    }
}

fn take_gpu_completed_intervals() -> Vec<[f64; 2]> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let mut intervals = jolt_akita::metal_profiling::take_completed_intervals();
        intervals.extend(jolt_kernels::metal_profiling::take_completed_intervals());
        intervals
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        Vec::new()
    }
}

fn valid_gpu_interval([start, end]: [f64; 2]) -> bool {
    start.is_finite() && end.is_finite() && start > 0.0 && end >= start
}

fn gpu_busy_union_seconds(intervals: &[[f64; 2]]) -> f64 {
    let mut intervals: Vec<_> = intervals
        .iter()
        .copied()
        .filter(|interval| valid_gpu_interval(*interval))
        .collect();
    intervals.sort_unstable_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then(left[1].total_cmp(&right[1]))
    });
    let Some((&[mut start, mut end], rest)) = intervals.split_first() else {
        return 0.0;
    };
    let mut total = 0.0;
    for &[next_start, next_end] in rest {
        if next_start <= end {
            end = end.max(next_end);
        } else {
            total += end - start;
            start = next_start;
            end = next_end;
        }
    }
    total + (end - start)
}

#[cfg(any(
    target_os = "macos",
    all(target_os = "linux", target_pointer_width = "64")
))]
fn process_cpu_seconds() -> Option<f64> {
    use std::ffi::{c_int, c_long};

    #[repr(C)]
    struct Timespec {
        seconds: c_long,
        nanoseconds: c_long,
    }
    extern "C" {
        fn clock_gettime(clock_id: c_int, time: *mut Timespec) -> c_int;
    }
    #[cfg(target_os = "macos")]
    const PROCESS_CLOCK: c_int = 12;
    #[cfg(target_os = "linux")]
    const PROCESS_CLOCK: c_int = 2;
    let mut time = Timespec {
        seconds: 0,
        nanoseconds: 0,
    };
    // SAFETY: Darwin and 64-bit Linux use two signed C longs for timespec;
    // their CLOCK_PROCESS_CPUTIME_ID constants are 12 and 2 respectively.
    // `time` is writable, properly aligned, and lives across the C call.
    let result = unsafe { clock_gettime(PROCESS_CLOCK, &raw mut time) };
    (result == 0).then_some(time.seconds as f64 + time.nanoseconds as f64 * 1e-9)
}

#[cfg(not(any(
    target_os = "macos",
    all(target_os = "linux", target_pointer_width = "64")
)))]
fn process_cpu_seconds() -> Option<f64> {
    None
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn memory_json_keeps_missing_values_null_and_flattens_profile_samples(
    ) -> Result<(), serde_json::Error> {
        let json = serde_json::to_value(ProofProfileMemory::default())?;
        assert!(json.get("sampled").is_none());
        assert!(json["sampled_peak_rss_bytes"].is_null());
        assert!(json["sampled_peak_physical_footprint_bytes"].is_null());
        assert!(json["sampled_peak_metal_allocated_bytes"].is_null());
        assert!(json["process_peak_rss_bytes"].is_null());
        assert!(json["process_peak_physical_footprint_bytes"].is_null());
        assert_eq!(json["sample_count"], 0);
        assert_eq!(json["unavailable_samples"]["rss"], 0);
        assert_eq!(json["unavailable_samples"]["physical_footprint"], 0);
        assert_eq!(json["unavailable_samples"]["metal_allocated"], 0);
        Ok(())
    }

    #[test]
    fn memory_peaks_use_inclusive_boundaries_without_adding_overlapping_gauges() {
        let origin = Instant::now();
        let sample = |seconds, rss, footprint, metal| MemorySample {
            wall: origin + Duration::from_secs(seconds),
            snapshot: MemorySnapshot {
                rss_bytes: Some(rss),
                physical_footprint_bytes: Some(footprint),
                metal_allocated_bytes: Some(metal),
                process_peak_rss_bytes: Some(1_000),
                process_peak_physical_footprint_bytes: Some(2_000),
            },
        };
        // Boundary observations may be inserted out of timestamp order by
        // concurrent stage guards and the periodic sampling thread.
        let index = MemoryIndex::new(vec![
            sample(2, 300, 400, 90),
            sample(0, 10, 20, 1),
            sample(3, 999, 888, 777),
            sample(1, 100, 200, 50),
        ]);
        let child = index.range(
            origin + Duration::from_secs(1),
            origin + Duration::from_secs(2),
        );
        assert_eq!(child.sampled.sample_count, 2);
        assert_eq!(child.sampled.sampled_peak_rss_bytes, Some(300));
        assert_eq!(
            child.sampled.sampled_peak_physical_footprint_bytes,
            Some(400)
        );
        assert_eq!(child.sampled.sampled_peak_metal_allocated_bytes, Some(90));
        assert_eq!(child.process_peak_rss_bytes, Some(1_000));
        assert_eq!(child.process_peak_physical_footprint_bytes, Some(2_000));
        assert_eq!(
            child.sampled.unavailable_samples,
            MemoryUnavailableSamples::default()
        );
        let parent = index.range(origin, origin + Duration::from_secs(2));
        assert_eq!(
            parent.sampled.sampled_peak_rss_bytes,
            child.sampled.sampled_peak_rss_bytes
        );
        assert_eq!(parent.sampled.sample_count, 3);
        let boundary = index.range(
            origin + Duration::from_secs(2),
            origin + Duration::from_secs(2),
        );
        assert_eq!(boundary.sampled.sample_count, 1);
        assert_eq!(boundary.sampled.sampled_peak_rss_bytes, Some(300));
    }

    #[test]
    fn missing_memory_queries_remain_null_and_count_each_unavailable_gauge() {
        let origin = Instant::now();
        let index = MemoryIndex::new(vec![
            MemorySample {
                wall: origin,
                snapshot: MemorySnapshot::default(),
            },
            MemorySample {
                wall: origin + Duration::from_secs(1),
                snapshot: MemorySnapshot {
                    rss_bytes: Some(0),
                    ..MemorySnapshot::default()
                },
            },
        ]);
        let memory = index.range(origin, origin + Duration::from_secs(1));
        assert_eq!(memory.sampled.sampled_peak_rss_bytes, Some(0));
        assert_eq!(memory.sampled.sampled_peak_physical_footprint_bytes, None);
        assert_eq!(memory.sampled.sampled_peak_metal_allocated_bytes, None);
        assert_eq!(memory.sampled.sample_count, 2);
        assert_eq!(memory.sampled.unavailable_samples.rss, 1);
        assert_eq!(memory.sampled.unavailable_samples.physical_footprint, 2);
        assert_eq!(memory.sampled.unavailable_samples.metal_allocated, 2);
        assert_eq!(memory.process_peak_rss_bytes, None);
        assert_eq!(memory.process_peak_physical_footprint_bytes, None);
        assert_eq!(
            index.range(origin, origin).sampled.sampled_peak_rss_bytes,
            None
        );
        assert_eq!(
            index
                .range(
                    origin + Duration::from_secs(2),
                    origin + Duration::from_secs(3)
                )
                .sampled,
            ProfileMemory::default()
        );
        assert_eq!(
            MemoryIndex::new(Vec::new()).range(origin, origin).sampled,
            ProfileMemory::default()
        );
    }

    #[test]
    fn memory_range_index_matches_observations_for_many_nested_intervals() {
        let origin = Instant::now();
        let samples: Vec<_> = (0..128)
            .rev()
            .map(|index| MemorySample {
                wall: origin + Duration::from_millis(index / 2),
                snapshot: MemorySnapshot {
                    rss_bytes: (index % 3 != 0).then_some(index * 17 % 127),
                    physical_footprint_bytes: (index % 5 != 0).then_some(index * 13 % 107),
                    metal_allocated_bytes: (index % 7 != 0).then_some(index * 11 % 101),
                    ..MemorySnapshot::default()
                },
            })
            .collect();
        let index = MemoryIndex::new(samples.clone());
        for start in 0..66 {
            for end in start..66 {
                let start = origin + Duration::from_millis(start);
                let end = origin + Duration::from_millis(end);
                let expected: Vec<_> = samples
                    .iter()
                    .filter(|sample| sample.wall >= start && sample.wall <= end)
                    .collect();
                let actual = index.range(start, end).sampled;
                assert_eq!(actual.sample_count as usize, expected.len());
                assert_eq!(
                    actual.sampled_peak_rss_bytes,
                    expected
                        .iter()
                        .filter_map(|sample| sample.snapshot.rss_bytes)
                        .max()
                );
                assert_eq!(
                    actual.sampled_peak_physical_footprint_bytes,
                    expected
                        .iter()
                        .filter_map(|sample| sample.snapshot.physical_footprint_bytes)
                        .max()
                );
                assert_eq!(
                    actual.sampled_peak_metal_allocated_bytes,
                    expected
                        .iter()
                        .filter_map(|sample| sample.snapshot.metal_allocated_bytes)
                        .max()
                );
                assert_eq!(
                    actual.unavailable_samples.rss as usize,
                    expected
                        .iter()
                        .filter(|sample| sample.snapshot.rss_bytes.is_none())
                        .count()
                );
            }
        }
    }

    #[test]
    fn exclusive_intervals_conserve_time_without_adding_async_supplements() {
        let origin = Instant::now();
        let sample = |wall, cpu| Sample {
            wall: origin + Duration::from_secs(wall),
            cpu_seconds: Some(cpu),
            gpu: GpuSnapshot {
                akita_seconds: wall as f64 * 0.25,
                piop_seconds: wall as f64 * 0.125,
                command_buffers: wall,
                submitted_command_buffers: wall,
                ..GpuSnapshot::default()
            },
        };
        let interval = |name: &str, parent_id, supplement, start, end| ActiveStage {
            parent_id,
            name: name.to_owned(),
            supplement,
            start,
            end: Some(end),
            metadata: BTreeMap::new(),
        };
        let profile = ActiveProfile {
            generation: 1,
            start: sample(0, 0.0),
            stages: vec![
                interval("proof", None, false, sample(0, 0.0), sample(10, 40.0)),
                interval("commit", Some(0), false, sample(1, 2.0), sample(6, 25.0)),
                interval("async", None, true, sample(2, 6.0), sample(8, 32.0)),
            ],
            metadata: BTreeMap::new(),
            memory: None,
        }
        .finish(sample(10, 40.0), Vec::new());
        let accounted_wall: f64 = profile
            .stages
            .iter()
            .filter(|s| !s.supplement)
            .map(|s| s.exclusive.wall_seconds)
            .sum();
        let accounted_cpu: f64 = profile
            .stages
            .iter()
            .filter(|s| !s.supplement)
            .filter_map(|s| s.exclusive.cpu_core_seconds)
            .sum();
        let accounted_gpu: f64 = profile
            .stages
            .iter()
            .filter(|s| !s.supplement)
            .map(|s| s.exclusive.gpu_seconds)
            .sum();
        assert_eq!(accounted_wall, profile.total.wall_seconds);
        assert_eq!(
            accounted_cpu,
            profile.total.cpu_core_seconds.unwrap_or_default()
        );
        assert_eq!(accounted_gpu, profile.total.gpu_seconds);
        assert!(profile.stages.iter().all(|s| s.complete));
        assert!(profile
            .stages
            .iter()
            .filter(|s| !s.supplement)
            .all(|s| s.overlapping_supplements == [2]));
    }

    #[test]
    fn gpu_busy_union_counts_overlapping_queues_once() {
        assert_eq!(gpu_busy_union_seconds(&[[12.0, 15.0], [13.0, 17.0]]), 5.0);
        assert_eq!(gpu_busy_union_seconds(&[[12.0, 15.0], [12.0, 15.0]]), 3.0);
    }

    #[test]
    fn gpu_busy_union_accepts_out_of_order_and_nested_intervals() {
        assert_eq!(
            gpu_busy_union_seconds(&[[20.0, 21.0], [12.0, 13.0], [10.0, 15.0]]),
            6.0
        );
    }

    #[test]
    fn gpu_busy_union_excludes_disjoint_queue_gaps() {
        assert_eq!(gpu_busy_union_seconds(&[[12.0, 13.0], [16.0, 17.0]]), 2.0);
        assert_eq!(gpu_busy_union_seconds(&[[12.0, 13.0], [13.0, 14.0]]), 2.0);
    }

    #[test]
    fn gpu_busy_union_ignores_invalid_timestamps() {
        assert_eq!(
            gpu_busy_union_seconds(&[
                [0.0, 1.0],
                [-1.0, 1.0],
                [3.0, 2.0],
                [f64::NAN, 2.0],
                [1.0, f64::INFINITY],
                [1.0, 1.0],
                [2.0, 4.0],
            ]),
            2.0
        );
        assert_eq!(gpu_busy_union_seconds(&[]), 0.0);
    }

    #[test]
    #[cfg(any(
        target_os = "macos",
        all(target_os = "linux", target_pointer_width = "64")
    ))]
    #[expect(
        clippy::expect_used,
        reason = "clock and thread failures must fail the profiling test"
    )]
    fn process_cpu_clock_includes_background_worker_cpu() {
        let before = process_cpu_seconds().expect("process CPU clock is supported");
        let worker_cpu = std::thread::spawn(|| {
            let before = process_cpu_seconds().expect("worker can read process CPU clock");
            let mut value = 1u64;
            for index in 0..2_000_000u64 {
                value = std::hint::black_box(
                    value.wrapping_mul(6364136223846793005).wrapping_add(index),
                );
            }
            let _ = std::hint::black_box(value);
            process_cpu_seconds().expect("worker can reread process CPU clock") - before
        })
        .join()
        .expect("background CPU worker completes");
        let all_thread_cpu =
            process_cpu_seconds().expect("process CPU clock can be reread") - before;
        assert!(worker_cpu > 0.0);
        assert!(all_thread_cpu >= worker_cpu);
    }
}
