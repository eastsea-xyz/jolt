//! Host memory gauges for explicitly enabled proof profiles.
//!
//! These gauges overlap on unified-memory machines. In particular, Metal's
//! device allocation counter covers resources, not a separate GPU RAM space
//! or only buffers, and must never be added to RSS or physical footprint.

/// Instantaneous memory gauges plus process-lifetime high-water marks.
/// Unsupported or failed queries stay unavailable instead of becoming zero.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemorySnapshot {
    pub rss_bytes: Option<u64>,
    pub physical_footprint_bytes: Option<u64>,
    pub metal_allocated_bytes: Option<u64>,
    pub process_peak_rss_bytes: Option<u64>,
    pub process_peak_physical_footprint_bytes: Option<u64>,
}

/// A probe is created only when profiling starts. Cache one system-default
/// device so Akita and Jolt do not count the same device's resources twice.
/// Other devices are outside this probe's scope; no allocations are summed.
pub struct MemoryProbe {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    device: Option<metal::Device>,
}

impl MemoryProbe {
    #[must_use]
    pub fn new() -> Self {
        Self {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            device: system_default_memory_device(),
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> MemorySnapshot {
        let snapshot = process_memory_snapshot();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            use metal::objc::{runtime::Sel, Message};

            // SAFETY: currentAllocatedSize is the argument-free MTLDevice
            // NSUInteger property. The probe retains the device across this
            // query; construction checked that this selector is supported.
            let metal_allocated_bytes = self.device.as_ref().and_then(|device| unsafe {
                device
                    .send_message::<(), metal::NSUInteger>(
                        Sel::register("currentAllocatedSize"),
                        (),
                    )
                    .ok()
                    .map(|bytes| bytes as u64)
            });
            MemorySnapshot {
                metal_allocated_bytes,
                ..snapshot
            }
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            snapshot
        }
    }
}

impl Default for MemoryProbe {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn system_default_memory_device() -> Option<metal::Device> {
    use metal::objc::{
        runtime::{Sel, BOOL, YES},
        Message,
    };

    let device = metal::Device::system_default()?;
    // SAFETY: NSObject's respondsToSelector: takes one SEL and returns BOOL.
    // Checking once avoids invoking an unavailable Metal property on older
    // systems; the retained device's method support cannot change afterwards.
    let supported = unsafe {
        device.send_message::<(Sel,), BOOL>(
            Sel::register("respondsToSelector:"),
            (Sel::register("currentAllocatedSize"),),
        )
    }
    .ok()?;
    (supported == YES).then_some(device)
}

#[cfg(target_os = "macos")]
fn process_memory_snapshot() -> MemorySnapshot {
    use std::mem::MaybeUninit;

    let mut snapshot = MemorySnapshot::default();
    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage writes the native libc rusage layout into a writable,
    // aligned buffer. It is read only after successful initialization. Darwin
    // reports ru_maxrss in bytes, unlike Linux's kilobyte representation.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0 {
        // SAFETY: successful getrusage initialized the entire rusage buffer.
        let usage = unsafe { usage.assume_init() };
        snapshot.process_peak_rss_bytes = u64::try_from(usage.ru_maxrss).ok();
    }

    let mut usage = MaybeUninit::<libc::rusage_info_v4>::uninit();
    // SAFETY: the V4 flavor selects exactly libc's rusage_info_v4 layout.
    // proc_pid_rusage's opaque output pointer is cast from that writable,
    // aligned buffer; getpid takes no arguments and has no preconditions.
    let result = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            usage.as_mut_ptr().cast(),
        )
    };
    if result == 0 {
        // SAFETY: successful proc_pid_rusage initialized the V4 buffer.
        let usage = unsafe { usage.assume_init() };
        snapshot.rss_bytes = Some(usage.ri_resident_size);
        snapshot.physical_footprint_bytes = Some(usage.ri_phys_footprint);
        snapshot.process_peak_physical_footprint_bytes = Some(usage.ri_lifetime_max_phys_footprint);
    } else {
        let mut usage = MaybeUninit::<libc::rusage_info_v2>::uninit();
        // SAFETY: this older flavor writes the V2 layout into the correctly
        // sized buffer, again read only on success. Lifetime footprint stays
        // unavailable on systems that cannot provide the V4 flavor.
        let result = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V2,
                usage.as_mut_ptr().cast(),
            )
        };
        if result == 0 {
            // SAFETY: successful proc_pid_rusage initialized the V2 buffer.
            let usage = unsafe { usage.assume_init() };
            snapshot.rss_bytes = Some(usage.ri_resident_size);
            snapshot.physical_footprint_bytes = Some(usage.ri_phys_footprint);
        }
    }
    snapshot
}

#[cfg(not(target_os = "macos"))]
fn process_memory_snapshot() -> MemorySnapshot {
    MemorySnapshot::default()
}
