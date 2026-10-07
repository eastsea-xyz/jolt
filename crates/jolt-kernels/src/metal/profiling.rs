//! Opt-in, process-wide GPU command-buffer accounting for prover stage profiles.
//!
//! Counters are cumulative. Taking a snapshot never waits for GPU work: completed
//! submissions contribute their individual GPU durations, while in-flight buffers
//! remain explicitly pending. The caller owns stage attribution for async work.

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

/// Enable submission accounting without resetting the cumulative counters.
///
/// The prover profile's start/finish API owns this switch. Disabling accounting
/// does not discard submissions that were already in flight.
pub fn set_enabled(enabled: bool) {
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

    use metal::{CommandBuffer, CommandBufferRef, MTLCommandBufferStatus};

    use super::GpuSnapshot;
    use crate::metal::solinas::runtime::completed_command_gpu_time;

    static COUNTERS: Mutex<Counters> = Mutex::new(Counters::new());

    struct Counters {
        totals: GpuSnapshot,
        // Retain each submission until its terminal timestamps can be sampled.
        pending: Vec<CommandBuffer>,
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
            }
        }

        fn collect_completed(&mut self) {
            let mut index = 0;
            while index < self.pending.len() {
                let command = &self.pending[index];
                let gpu_time = match command.status() {
                    MTLCommandBufferStatus::Completed => completed_command_gpu_time(command).ok(),
                    MTLCommandBufferStatus::Error => None,
                    _ => {
                        index += 1;
                        continue;
                    }
                };
                self.totals.record_gpu_time(gpu_time);
                drop(self.pending.swap_remove(index));
            }
            self.totals.pending_command_buffers = self.pending.len() as u64;
        }
    }

    pub(super) fn record_submission(command: &CommandBufferRef) {
        let mut counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
        // Reap during long stages as well as at boundaries so profiling retains
        // only in-flight buffers, rather than an entire stage's resources.
        counters.collect_completed();
        counters.pending.push(command.to_owned());
        counters.totals.submitted_command_buffers += 1;
    }

    pub(super) fn snapshot() -> GpuSnapshot {
        let mut counters = COUNTERS.lock().unwrap_or_else(PoisonError::into_inner);
        counters.collect_completed();
        counters.totals
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
                counters.pending.push(command.to_owned());
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
                } else {
                    assert_eq!(first.timestamp_unavailable, 1);
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GpuSnapshot;
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
}
