//! Opt-in, process-wide proof timing. CPU time includes all Rayon and worker
//! threads; asynchronous supplements describe overlap and are never added to
//! the accounted total. GPU time is the sum of completed command-buffer
//! durations, so pending buffers explicitly expose stage-boundary ambiguity.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    marker::PhantomData,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Mutex, PoisonError,
    },
    time::Instant,
};

use serde::Serialize;

pub use tracing;

static ENABLED: AtomicBool = AtomicBool::new(false);
static GENERATION: AtomicU64 = AtomicU64::new(0);
static ACTIVE: Mutex<Option<ActiveProfile>> = Mutex::new(None);

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
    pub wall_percent_of_total: f64,
    pub cpu_percent_of_total: Option<f64>,
    /// IDs of supplemental intervals that overlap this interval.
    pub overlapping_supplements: Vec<usize>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct ProofProfile {
    pub schema_version: u32,
    pub total: ProfileMetrics,
    pub stages: Vec<ProfileStage>,
    pub metadata: BTreeMap<String, String>,
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
}

impl ActiveProfile {
    fn finish(self, end: Sample) -> ProofProfile {
        let total = self.start.elapsed(end);
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
            schema_version: 1,
            total,
            stages,
            metadata: self.metadata,
        }
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
    metadata.insert(
        "cpu_clock".to_owned(),
        "CLOCK_PROCESS_CPUTIME_ID; includes all process threads".to_owned(),
    );
    metadata.insert("gpu_clock".to_owned(), "sum of completed Metal GPUStartTime/GPUEndTime durations; pending buffers can cross stage boundaries".to_owned());
    metadata.insert("accounting".to_owned(), "sum exclusive normal stages only; asynchronous supplements share process counters and are excluded".to_owned());
    *active = Some(ActiveProfile {
        generation,
        start,
        stages: Vec::new(),
        metadata,
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
        active.metadata.insert(key.into(), value.into());
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
    ACTIVE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
        .map(|active| active.finish(end))
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
        let start = Sample::capture();
        let mut active = ACTIVE.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(active) = active.as_mut() else {
            return guard;
        };
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
                stage.metadata.insert(key.into(), value.into());
            }
        }
    }
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        let Some((generation, id)) = self.token else {
            return;
        };
        let end = Sample::capture();
        STACK.with(|stack| {
            stack
                .borrow_mut()
                .retain(|token| *token != (generation, id))
        });
        let mut active = ACTIVE.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(active) = active
            .as_mut()
            .filter(|active| active.generation == generation)
        {
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
        }
        .finish(sample(10, 40.0));
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
