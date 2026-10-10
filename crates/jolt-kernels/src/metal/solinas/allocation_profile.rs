//! Opt-in host-ownership events for register payload buffers.
//!
//! `JOLT_GPU_ALLOCATION_LOG` selects an absolute JSONL path supplied by the
//! profiling runner under its lane's `tmp/` directory. No Metal handles are
//! retained here. Release events mark host ownership ending; command buffers
//! can retain a resource longer. Other Metal buffers and CPU heaps are outside
//! this diagnostic's scope.

use std::{
    fmt::Write as _,
    fs::{File, OpenOptions},
    io::Write,
    panic::Location,
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use metal::{foreign_types::ForeignType, Buffer};

static PROFILE: OnceLock<Option<AllocationProfile>> = OnceLock::new();

struct AllocationProfile {
    started: Instant,
    output: Mutex<Option<File>>,
}

pub(super) fn start() {
    let _ = profile();
}

#[track_caller]
pub(super) fn record(event: &'static str, buffer: &Buffer, reason: Option<&'static str>) {
    let Some(profile) = profile() else {
        return;
    };
    let Ok(mut output) = profile.output.lock() else {
        return;
    };
    let Some(file) = output.as_mut() else {
        return;
    };
    let site = Location::caller();
    let line = event_line(
        event,
        buffer.as_ptr() as usize,
        buffer.length(),
        profile.started.elapsed().as_secs_f64(),
        &format!("{}:{}", site.file(), site.line()),
        reason,
    );
    if let Err(error) = file.write_all(line.as_bytes()) {
        *output = None;
        tracing::warn!(%error, "Metal allocation diagnostic disabled after write failure");
    }
}

fn profile() -> Option<&'static AllocationProfile> {
    PROFILE.get_or_init(AllocationProfile::from_env).as_ref()
}

impl AllocationProfile {
    fn from_env() -> Option<Self> {
        let path = PathBuf::from(std::env::var_os("JOLT_GPU_ALLOCATION_LOG")?);
        if !path.is_absolute() {
            tracing::warn!(path = %path.display(), "Metal allocation diagnostic requires an absolute path");
            return None;
        }
        let mut file = match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => file,
            Err(error) => {
                tracing::warn!(%error, path = %path.display(), "Metal allocation diagnostic could not open output");
                return None;
            }
        };
        let started = Instant::now();
        let wall_unix_seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_secs_f64();
        let header = format!(
            "{{\"event\":\"start\",\"wall_unix_seconds\":{wall_unix_seconds:.9},\"elapsed_seconds\":0,\"scope\":\"register_private_payloads\"}}\n"
        );
        if let Err(error) = file.write_all(header.as_bytes()) {
            tracing::warn!(%error, "Metal allocation diagnostic could not write header");
            return None;
        }
        Some(Self {
            started,
            output: Mutex::new(Some(file)),
        })
    }
}

fn event_line(
    event: &str,
    id: usize,
    bytes: u64,
    elapsed_seconds: f64,
    site: &str,
    reason: Option<&str>,
) -> String {
    let mut line = format!(
        "{{\"event\":\"{event}\",\"id\":\"0x{id:x}\",\"bytes\":{bytes},\"elapsed_seconds\":{elapsed_seconds:.9},\"site\":"
    );
    append_json_string(&mut line, site);
    if let Some(reason) = reason {
        line.push_str(",\"reason\":");
        append_json_string(&mut line, reason);
    }
    line.push_str("}\n");
    line
}

fn append_json_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{0000}'..='\u{001f}' => {
                let _ = write!(output, "\\u{:04x}", u32::from(character));
            }
            _ => output.push(character),
        }
    }
    output.push('"');
}

#[cfg(test)]
mod tests {
    use super::event_line;

    #[test]
    fn allocation_events_are_jsonl_and_preserve_source_paths() {
        let site = "crates/커널/\"file\\name\"\n\r\t\u{0000}\u{001f}.rs:42";
        for event in ["allocate", "reuse", "recycle", "release", "evict"] {
            let line = event_line(event, 0xabcd, u64::MAX, 1.25, site, Some("pool_cap"));
            assert_eq!(line.lines().count(), 1);
            let parsed: serde_json::Value = serde_json::from_str(&line).unwrap_or_default();
            assert_eq!(parsed["event"], event);
            assert_eq!(parsed["id"], "0xabcd");
            assert_eq!(parsed["bytes"].as_u64(), Some(u64::MAX));
            assert_eq!(parsed["elapsed_seconds"].as_f64(), Some(1.25));
            assert_eq!(parsed["site"], site);
            assert_eq!(parsed["reason"], "pool_cap");
        }
        let line = event_line("allocate", 1, 16, 0.0, "sequence.rs:1", None);
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap_or_default();
        assert!(parsed.get("reason").is_none());
    }
}
