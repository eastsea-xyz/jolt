//! Opt-in, process-wide GPU command-buffer accounting for prover stage profiles.
//!
//! Counters are cumulative. Taking a snapshot never waits for GPU work: completed
//! submissions contribute their individual GPU durations, while in-flight buffers
//! remain explicitly pending. The caller owns stage attribution for async work.
//! Raw completed intervals are retained separately for one profile's busy-time union.

use std::sync::atomic::{AtomicBool, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Cumulative GPU telemetry; subtract two snapshots to measure an interval.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuSnapshot {
    /// Sum of completed command-buffer GPU durations, excluding queue gaps.
    pub gpu_seconds: f64,
    /// Terminal command buffers observed (including failed buffers).
    pub command_buffers: u64,
    /// Terminal buffers with failed execution or unavailable GPU timestamps.
    pub timestamp_unavailable: u64,
    /// Submitted buffers not yet terminal at this snapshot.
    pub pending_command_buffers: u64,
    /// All command-buffer submissions observed while profiling was enabled.
    pub submitted_command_buffers: u64,
}

impl GpuSnapshot {
    #[cfg(any(target_os = "macos", test))]
    fn record_gpu_time(&mut self, gpu_time: Option<std::time::Duration>) {
        self.command_buffers += 1;
        if let Some(gpu_time) = gpu_time {
            self.gpu_seconds += gpu_time.as_secs_f64();
        } else {
            self.timestamp_unavailable += 1;
        }
    }
}

#[cfg(any(target_os = "macos", test))]
fn gpu_interval(start: f64, end: f64) -> Option<[f64; 2]> {
    (start.is_finite()
        && end.is_finite()
        && start > 0.0
        && end >= start
        && std::time::Duration::try_from_secs_f64(end - start).is_ok())
    .then_some([start, end])
}

/// Enable submission accounting without resetting the cumulative counters.
///
/// The prover profile's start/finish API owns this switch. Disabling accounting
/// does not discard submissions that were already in flight.
pub fn set_enabled(enabled: bool) {
    #[cfg(target_os = "macos")]
    platform::set_enabled(enabled);
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// Observe completed work without adding waits or changing command scheduling.
#[must_use]
pub fn snapshot() -> GpuSnapshot {
    #[cfg(target_os = "macos")]
    {
        platform::snapshot()
    }
    #[cfg(not(target_os = "macos"))]
    {
        GpuSnapshot::default()
    }
}

/// Drain only intervals already collected by snapshots or submissions. This does
/// not inspect pending buffers or wait, so it preserves the finish snapshot's scope.
/// Enabling a new profile clears retained intervals and isolates older submissions.
#[must_use]
pub fn take_completed_intervals() -> Vec<[f64; 2]> {
    #[cfg(target_os = "macos")]
    {
        platform::take_completed_intervals()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Vec::new()
    }
}

#[cfg(target_os = "macos")]
#[inline]
pub(crate) fn record_submission(command: &metal::CommandBufferRef) {
    if ENABLED.load(Ordering::Relaxed) {
        platform::record_submission(command);
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::sync::{Mutex, PoisonError};

    use metal::objc::{runtime::Sel, Message};
    use metal::{CommandBuffer, CommandBufferRef, MTLCommandBufferStatus};

    use super::{gpu_interval, GpuSnapshot};
    #[cfg(test)]
    use crate::metal::solinas::runtime::completed_command_gpu_time;

    static COUNTERS: Mutex<Counters> = Mutex::new(Counters::new());

    struct Counters {
        totals: GpuSnapshot,
        // Retain each submission until its terminal timestamps can be sampled.
        pending: Vec<(u64, CommandBuffer)>,
        profile_generation: u64,
        enabled: bool,
        completed_intervals: Vec<[f64; 2]>,
    }

    impl Counters {
        const fn new() -> Self {
            Self {
                totals: GpuSnapshot {
                    gpu_seconds: 0.0,
                    command_buffers: 0,
                    timestamp_unavailable: 0,
                    pending_command_buffers: 0,
                    submitted_command_buffers: 0,
                },
                pending: Vec::new(),
                profile_generation: 0,
                enabled: false,
                completed_intervals: Vec::new(),
            }
        }

        fn set_enabled(&mut self, enabled: bool) {
            if enabled && !self.enabled {
                self.profile_generation = self.profile_generation.wrapping_add(1);
                self.completed_intervals.clear();
            }
            self.enabled = enabled;
        }

        fn record_completed_interval(&mut self, generation: u64, interval: Option<[f64; 2]>) {
            self.totals.record_gpu_time(
                interval.map(|[start, end]| std::time::Duration::from_secs_f64(end - start)),
            );
            if generation == self.profile_generation {
                self.completed_intervals.extend(interval);
            }
        }

        fn take_completed_intervals(&mut self) -> Vec<[f64; 2]> {
            std::mem::take(&mut self.completed_intervals)
        }

        fn collect_completed(&mut self) {
            let mut index = 0;
            while index < self.pending.len() {
                let (generation, command) = &self.pending[index];
                let generation = *generation;
                let interval = match command.status() {
                    MTLCommandBufferStatus::Completed => completed_command_gpu_interval(command),
                    MTLCommandBufferStatus::Error => None,
                    _ => {
                        index += 1;
                        continue;
                    }
                };
                self.record_completed_interval(generation, interval);
                drop(self.pending.swap_remove(index));
            }
            self.totals.pending_command_buffers = self.pending.len() as u64;
        }
    }

    fn completed_command_gpu_interval(command: &CommandBufferRef) -> Option<[f64; 2]> {
        // SAFETY: both selectors are argument-free MTLCommandBuffer properties
        // returning NSTimeInterval (f64); the command is terminal and retained.
        let (start, end) = unsafe {
            (
                command
                    .send_message::<(), f64>(Sel::register("GPUStartTime"), ())
                    .ok()?,
                command
                    .send_message::<(), f64>(Sel::register("GPUEndTime"), ())
                    .ok()?,
            )
        };
        gpu_interval(start, end)
    }

    pub(super) fn set_enabled(enabled: bool) {
        COUNTERS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .set_enabled(enabled);
    }

    pub(super) fn record_submission(command: &CommandBufferRef) {
        let mut counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
        if !counters.enabled {
            return;
        }
        // Reap during long stages as well as at boundaries so profiling retains
        // only in-flight buffers, rather than an entire stage's resources.
        counters.collect_completed();
        let generation = counters.profile_generation;
        counters.pending.push((generation, command.to_owned()));
        counters.totals.submitted_command_buffers += 1;
    }

    pub(super) fn snapshot() -> GpuSnapshot {
        let mut counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
        counters.collect_completed();
        counters.totals
    }

    pub(super) fn take_completed_intervals() -> Vec<[f64; 2]> {
        COUNTERS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take_completed_intervals()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use metal::{objc::rc::autoreleasepool, Device, MTLResourceOptions, NSRange};

        #[test]
        fn snapshots_reap_a_submission_once_without_waiting_for_pending_work() {
            autoreleasepool(|| {
                let Some(device) = Device::system_default() else {
                    return;
                };
                let queue = device.new_command_queue();
                let command = queue.new_command_buffer();
                let mut counters = Counters::new();
                counters.set_enabled(true);
                counters
                    .pending
                    .push((counters.profile_generation, command.to_owned()));
                counters.totals.submitted_command_buffers += 1;
                counters.collect_completed();
                assert_eq!(counters.totals.pending_command_buffers, 1);
                assert_eq!(counters.totals.command_buffers, 0);

                let buffer = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
                let encoder = command.new_blit_command_encoder();
                encoder.fill_buffer(&buffer, NSRange::new(0, 64), 7);
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                counters.collect_completed();
                assert_eq!(counters.totals.command_buffers, 1);
                assert_eq!(counters.totals.pending_command_buffers, 0);
                let first = counters.totals;
                counters.collect_completed();
                assert_eq!(counters.totals, first);
                if let Ok(gpu_time) = completed_command_gpu_time(command) {
                    assert_eq!(first.gpu_seconds, gpu_time.as_secs_f64());
                    assert_eq!(first.timestamp_unavailable, 0);
                    assert_eq!(counters.take_completed_intervals().len(), 1);
                } else {
                    assert_eq!(first.timestamp_unavailable, 1);
                    assert!(counters.take_completed_intervals().is_empty());
                }
                assert!(counters.take_completed_intervals().is_empty());
            });
        }

        #[test]
        fn warm_profiles_isolate_intervals_while_counters_remain_cumulative() {
            let mut counters = Counters::new();
            counters.set_enabled(true);
            let first_generation = counters.profile_generation;
            counters.record_completed_interval(first_generation, Some([12.0, 13.0]));
            counters.set_enabled(false);
            assert_eq!(counters.take_completed_intervals(), [[12.0, 13.0]]);
            assert!(counters.take_completed_intervals().is_empty());

            // A late completion after finish is cleared by the next start.
            counters.record_completed_interval(first_generation, Some([14.0, 15.0]));
            counters.set_enabled(true);
            assert!(counters.take_completed_intervals().is_empty());
            // A prior profile's still-pending buffer counts, but cannot enter
            // this profile's interval list even if it completes during it.
            counters.record_completed_interval(first_generation, Some([16.0, 17.0]));
            counters.record_completed_interval(counters.profile_generation, Some([18.0, 19.0]));
            assert_eq!(counters.take_completed_intervals(), [[18.0, 19.0]]);
            assert_eq!(counters.totals.command_buffers, 4);
            assert_eq!(counters.totals.gpu_seconds, 4.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{gpu_interval, GpuSnapshot};
    use std::time::Duration;

    #[test]
    fn gpu_work_is_the_sum_of_durations_with_no_queue_gap_charge() {
        let mut totals = GpuSnapshot::default();
        // Buffers can execute at 12..13 and 16..17; only their two durations
        // belong to GPU work, rather than the five-second enclosing span.
        totals.record_gpu_time(Some(Duration::from_secs(1)));
        totals.record_gpu_time(Some(Duration::from_secs(1)));
        assert_eq!(totals.gpu_seconds, 2.0);
        assert_eq!(totals.command_buffers, 2);
    }

    #[test]
    fn missing_timestamps_cannot_be_mistaken_for_zero_gpu_work() {
        let mut totals = GpuSnapshot::default();
        totals.record_gpu_time(None);
        totals.record_gpu_time(Some(Duration::ZERO));
        assert_eq!(totals.command_buffers, 2);
        assert_eq!(totals.timestamp_unavailable, 1);
        assert_eq!(totals.gpu_seconds, 0.0);
    }

    #[test]
    fn invalid_or_empty_start_timestamps_are_unavailable() {
        for (start, end) in [
            (0.0, 0.0),
            (0.0, 1.0),
            (-1.0, 1.0),
            (2.0, 1.0),
            (f64::NAN, 1.0),
            (1.0, f64::INFINITY),
        ] {
            assert!(gpu_interval(start, end).is_none());
        }
        assert_eq!(gpu_interval(1.0, 1.0), Some([1.0, 1.0]));
        assert_eq!(gpu_interval(1.0, 2.0), Some([1.0, 2.0]));
    }
}
