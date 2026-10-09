//! Opt-in partial ALU replay diagnostic. Its results never replace proof rows.
//! Compile/register only with `all(feature = "metal", target_os = "macos")`.

pub use jolt_kernels::metal::solinas::trace_replay::{
    diagnose, TraceReplayError, TraceReplayReport,
};
