//! Bounded diagnostic replay of straight-line final JoltIR ALU chunks.
//!
//! This is NOT a full RISC-V emulator: no fetch, branch, memory, advice, CSR,
//! host I/O, or unsupported inline semantics. The CPU compact trace is the oracle.
//! GPU output never enters a proof. Only static instruction descriptors and a
//! first-touch register checkpoint cross into the GPU input buffers; per-row
//! CPU operands/results and expected final registers remain on the host.

use std::{collections::BTreeMap, fmt, mem::size_of, time::Instant};

use jolt_riscv::{JoltInstructionKind as Kind, JoltTraceRow};
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use thiserror::Error;

use super::runtime::{buffer_from_slice, completed_command_gpu_time, set_inline_bytes};

pub const MAX_CHUNKS: usize = 512;
pub const MAX_CHUNK_ROWS: usize = 64;
pub const MIN_CHUNK_ROWS: usize = 8;
pub const REGISTER_COUNT: usize = 128;
pub const MAX_EXTRA_BUFFER_BYTES: usize = 32 * 1024 * 1024;
const NONE: u32 = u32::MAX;
const IMMEDIATE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Instruction {
    imm: u64,
    op: u32,
    rs1: u32,
    rs2: u32,
    rd: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Chunk {
    start: u32,
    count: u32,
    known_lo: u64,
    known_hi: u64,
    initial: [u64; REGISTER_COUNT],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Values {
    rs1: u64,
    rs2: u64,
    rd_pre: u64,
    rd_write: u64,
}

#[repr(C)]
struct Params {
    chunks: u32,
    instructions: u32,
}

const _: () = {
    assert!(size_of::<Instruction>() == 24);
    assert!(size_of::<Chunk>() == 1048);
    assert!(size_of::<Values>() == 32);
    assert!(size_of::<Params>() == 8);
};

#[derive(Debug, Error)]
pub enum TraceReplayError {
    #[error("trace replay source contract failed at physical row {row}: {detail}")]
    Source { row: usize, detail: &'static str },
    #[error("trace replay resource bound exceeded: {0}")]
    Bounds(&'static str),
    #[error("trace replay Metal error: {0}")]
    Metal(String),
    #[error("trace replay GPU rejected chunk {0}")]
    GpuRejected(usize),
    #[error("trace replay parity failed at physical row {row}, column {column}: CPU={expected:#x}, GPU={actual:#x}")]
    Parity {
        row: usize,
        column: &'static str,
        expected: u64,
        actual: u64,
    },
    #[error("trace replay process CPU clock unavailable")]
    CpuClock,
}

/// All timings include diagnostic overhead; `cpu_seconds` is process CPU time,
/// so unrelated process threads may contribute. No full-trace speedup is claimed.
#[derive(Debug, Default)]
pub struct TraceReplayReport {
    pub physical_rows: usize,
    pub supported_rows: usize,
    pub eligible_rows: usize,
    pub short_run_rows: usize,
    pub sampled_rows: usize,
    pub chunks: usize,
    pub checkpoint_registers: usize,
    pub exact_parity: bool,
    pub device_name: Option<String>,
    pub device_buffer_bytes: u64,
    pub extra_buffer_bytes_bound: usize,
    pub submitted_commands: usize,
    pub completed_commands: usize,
    pub wall_seconds: f64,
    pub cpu_seconds: f64,
    pub selection_seconds: f64,
    pub compile_seconds: f64,
    pub command_wall_seconds: f64,
    pub gpu_seconds: f64,
}

impl fmt::Display for TraceReplayReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "partial_alu_replay physical_rows={} supported_rows={} eligible_rows={} short_run_rows={} sampled_rows={} chunks={} checkpoint_registers={} exact_parity={} device={:?} device_buffer_bytes={} extra_buffer_bytes_bound={} submitted={} completed={} wall_s={:.6} cpu_s={:.6} selection_s={:.6} compile_s={:.6} command_wall_s={:.6} gpu_s={:.6} sparse_end_state=true full_riscv_emulator=false proof_output_used=false",
            self.physical_rows, self.supported_rows, self.eligible_rows, self.short_run_rows,
            self.sampled_rows, self.chunks, self.checkpoint_registers, self.exact_parity,
            self.device_name, self.device_buffer_bytes, self.extra_buffer_bytes_bound,
            self.submitted_commands, self.completed_commands, self.wall_seconds,
            self.cpu_seconds, self.selection_seconds, self.compile_seconds,
            self.command_wall_seconds, self.gpu_seconds)
    }
}

struct Prepared {
    instructions: Vec<Instruction>,
    chunks: Vec<Chunk>,
    physical_starts: Vec<usize>,
    expected_final: Vec<[u64; REGISTER_COUNT]>,
    report: TraceReplayReport,
}

fn source_error(row: usize, detail: &'static str) -> TraceReplayError {
    TraceReplayError::Source { row, detail }
}

fn instruction(
    row: &JoltTraceRow,
    physical: usize,
) -> Result<Option<Instruction>, TraceReplayError> {
    if row.is_load() || row.is_store() || row.is_noop() {
        return Ok(None);
    }
    let Some(kind) = row.instruction_kind() else {
        return Err(source_error(physical, "unknown final instruction tag"));
    };
    let (op, immediate, unary, pow_immediate) = match kind {
        Kind::Add(_) => (1, false, false, false),
        Kind::Addi(_) => (1, true, false, false),
        Kind::Sub(_) => (2, false, false, false),
        Kind::Xor(_) => (3, false, false, false),
        Kind::XorI(_) => (3, true, false, false),
        Kind::And(_) => (4, false, false, false),
        Kind::AndI(_) => (4, true, false, false),
        Kind::Or(_) => (5, false, false, false),
        Kind::OrI(_) => (5, true, false, false),
        Kind::Mul(_) => (6, false, false, false),
        Kind::MulI(_) => (6, true, false, false),
        Kind::AddW(_) => (7, false, false, false),
        Kind::AddiW(_) => (7, true, false, false),
        Kind::SubW(_) => (8, false, false, false),
        Kind::MulW(_) => (9, false, false, false),
        Kind::MulIW(_) => (9, true, false, false),
        Kind::VirtualZeroExtendWord(_) => (10, false, true, false),
        Kind::VirtualSignExtendWord(_) => (11, false, true, false),
        Kind::Pow2(_) => (12, false, true, false),
        Kind::Pow2I(_) => (12, true, true, true),
        Kind::Pow2W(_) => (13, false, true, false),
        Kind::Pow2IW(_) => (13, true, true, true),
        Kind::VirtualSrl(_) => (14, false, false, false),
        Kind::VirtualSrli(_) => (14, true, false, false),
        Kind::VirtualSra(_) => (15, false, false, false),
        Kind::VirtualSrai(_) => (15, true, false, false),
        Kind::VirtualSrlw(_) => (16, false, false, false),
        Kind::VirtualSrliw(_) => (16, true, false, false),
        Kind::VirtualSraw(_) => (17, false, false, false),
        Kind::VirtualSraiw(_) => (17, true, false, false),
        Kind::Andn(_) => (18, false, false, false),
        Kind::Slt(_) => (19, false, false, false),
        Kind::SltI(_) => (19, true, false, false),
        Kind::SltU(_) => (20, false, false, false),
        Kind::SltIU(_) => (20, true, false, false),
        _ => return Ok(None),
    };
    let rs1 = row.rs1_index().map_or(NONE, u32::from);
    let rs2 = row.rs2_index().map_or(NONE, u32::from);
    let rd = row.rd_index().map_or(NONE, u32::from);
    if (rs1 == NONE) != pow_immediate || (rs2 != NONE) != (!immediate && !unary) || rd == NONE {
        return Err(source_error(
            physical,
            "operand shape disagrees with supported instruction",
        ));
    }
    if [rs1, rs2, rd]
        .into_iter()
        .any(|r| r != NONE && r >= REGISTER_COUNT as u32)
    {
        return Err(source_error(
            physical,
            "register id exceeds the 128-slot checkpoint",
        ));
    }
    if row.imm() < i64::MIN as i128 || row.imm() > u64::MAX as i128 {
        return Err(source_error(
            physical,
            "immediate does not fit RV64 bit representation",
        ));
    }
    Ok(Some(Instruction {
        imm: row.imm() as u64,
        op: op | if immediate { IMMEDIATE } else { 0 },
        rs1,
        rs2,
        rd,
    }))
}

fn touch(
    chunk: &mut Chunk,
    current: &mut [u64; REGISTER_COUNT],
    reg: u32,
    value: u64,
    physical: usize,
) -> Result<(), TraceReplayError> {
    if reg == NONE {
        return if value == 0 {
            Ok(())
        } else {
            Err(source_error(
                physical,
                "absent operand has nonzero captured value",
            ))
        };
    }
    let index = reg as usize;
    let mask = 1u64 << (index & 63);
    let known = if index < 64 {
        &mut chunk.known_lo
    } else {
        &mut chunk.known_hi
    };
    if *known & mask == 0 {
        // The only oracle values ever uploaded: each register's FIRST touch.
        *known |= mask;
        chunk.initial[index] = value;
        current[index] = value;
    } else if current[index] != value {
        return Err(source_error(
            physical,
            "register read/pre-write differs from preceding CPU state",
        ));
    }
    Ok(())
}

fn prepare(rows: &[JoltTraceRow]) -> Result<Prepared, TraceReplayError> {
    let mut result = Prepared {
        instructions: Vec::with_capacity(MAX_CHUNKS * MAX_CHUNK_ROWS),
        chunks: Vec::with_capacity(MAX_CHUNKS),
        physical_starts: Vec::with_capacity(MAX_CHUNKS),
        expected_final: Vec::with_capacity(MAX_CHUNKS),
        report: TraceReplayReport {
            physical_rows: rows.len(),
            ..TraceReplayReport::default()
        },
    };
    // Bounded sampled-bytecode consistency table, never all-program checkpoints.
    let mut static_instructions = BTreeMap::<u64, (u64, Instruction)>::new();
    let mut run_start = 0;
    let mut run_len = 0;
    for physical in 0..=rows.len() {
        let supported = if physical < rows.len() {
            instruction(&rows[physical], physical)?.is_some()
        } else {
            false
        };
        result.report.supported_rows += usize::from(supported);
        let contiguous = run_len == 0
            || (physical < rows.len()
                && rows[physical - 1].pc().checked_add(1) == Some(rows[physical].pc()));
        if run_len != 0 && (!supported || !contiguous || run_len == MAX_CHUNK_ROWS) {
            if run_len < MIN_CHUNK_ROWS {
                result.report.short_run_rows += run_len;
            } else {
                result.report.eligible_rows += run_len;
                if result.chunks.len() < MAX_CHUNKS {
                    add_chunk(
                        &mut result,
                        rows,
                        run_start,
                        run_len,
                        &mut static_instructions,
                    )?;
                }
            }
            run_len = 0;
        }
        if supported {
            if run_len == 0 {
                run_start = physical;
            }
            run_len += 1;
        }
    }
    result.report.sampled_rows = result.instructions.len();
    result.report.chunks = result.chunks.len();
    // Conservative buffer accounting includes all host vector capacities, the
    // simultaneous shared-device copies/outputs, and 128 bytes per map entry.
    let input_bytes = result.instructions.capacity() * size_of::<Instruction>()
        + result.chunks.capacity() * size_of::<Chunk>();
    let output_bytes = result.instructions.capacity() * size_of::<Values>()
        + MAX_CHUNKS * (REGISTER_COUNT * size_of::<u64>() + size_of::<u32>());
    let host_bytes = input_bytes
        + result.physical_starts.capacity() * size_of::<usize>()
        + result.expected_final.capacity() * size_of::<[u64; REGISTER_COUNT]>()
        + static_instructions.len() * 128;
    let bound = host_bytes + input_bytes + output_bytes + size_of::<Params>();
    if bound > MAX_EXTRA_BUFFER_BYTES {
        return Err(TraceReplayError::Bounds(
            "host/device prototype buffers exceed 32 MiB",
        ));
    }
    result.report.extra_buffer_bytes_bound = bound;
    Ok(result)
}

fn add_chunk(
    result: &mut Prepared,
    rows: &[JoltTraceRow],
    physical_start: usize,
    count: usize,
    static_instructions: &mut BTreeMap<u64, (u64, Instruction)>,
) -> Result<(), TraceReplayError> {
    let mut chunk = Chunk {
        start: result.instructions.len() as u32,
        count: count as u32,
        known_lo: 1,
        known_hi: 0,
        initial: [0; REGISTER_COUNT],
    };
    let mut current = [0u64; REGISTER_COUNT];
    for (offset, row) in rows[physical_start..physical_start + count]
        .iter()
        .enumerate()
    {
        let physical = physical_start + offset;
        let ins = instruction(row, physical)?
            .ok_or_else(|| source_error(physical, "unsupported row inside selected chunk"))?;
        let descriptor = (row.unexpanded_pc(), ins);
        if let Some(previous) = static_instructions.insert(row.pc(), descriptor) {
            if previous != descriptor {
                return Err(source_error(
                    physical,
                    "one bytecode PC has conflicting static instruction descriptors",
                ));
            }
        }
        touch(&mut chunk, &mut current, ins.rs1, row.rs1_value(), physical)?;
        touch(&mut chunk, &mut current, ins.rs2, row.rs2_value(), physical)?;
        touch(
            &mut chunk,
            &mut current,
            ins.rd,
            row.rd_pre_value(),
            physical,
        )?;
        if ins.rd == 0 && row.rd_write_value() != 0 {
            return Err(source_error(physical, "CPU row writes a nonzero x0"));
        }
        current[ins.rd as usize] = row.rd_write_value();
        result.instructions.push(ins);
    }
    result.report.checkpoint_registers +=
        chunk.known_lo.count_ones() as usize + chunk.known_hi.count_ones() as usize;
    result.physical_starts.push(physical_start);
    result.expected_final.push(current);
    result.chunks.push(chunk);
    Ok(())
}

fn cpu_seconds() -> Result<f64, TraceReplayError> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes exactly this valid rusage allocation on success.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return Err(TraceReplayError::CpuClock);
    }
    // SAFETY: checked the successful return value above.
    let usage = unsafe { usage.assume_init() };
    Ok((usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) as f64
        + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) as f64 / 1_000_000.0)
}

/// Replay at most 32,768 sampled physical rows and compare every generated
/// register column and the sparse touched-register end state with the CPU
/// trace oracle. Untouched slots are checked to remain the checkpoint's zero
/// placeholders; their architectural values are intentionally not captured.
/// Fewer than eight consecutive supported rows are declined without GPU work.
pub fn diagnose(rows: &[JoltTraceRow]) -> Result<TraceReplayReport, TraceReplayError> {
    metal::objc::rc::autoreleasepool(|| diagnose_inner(rows))
}

fn diagnose_inner(rows: &[JoltTraceRow]) -> Result<TraceReplayReport, TraceReplayError> {
    let wall = Instant::now();
    let cpu_start = cpu_seconds()?;
    let selection = Instant::now();
    let mut prepared = prepare(rows)?;
    prepared.report.selection_seconds = selection.elapsed().as_secs_f64();
    if !prepared.chunks.is_empty() {
        replay_and_compare(&mut prepared, rows)?;
        prepared.report.exact_parity = true;
    }
    prepared.report.cpu_seconds = cpu_seconds()? - cpu_start;
    prepared.report.wall_seconds = wall.elapsed().as_secs_f64();
    Ok(prepared.report)
}

fn replay_and_compare(
    prepared: &mut Prepared,
    rows: &[JoltTraceRow],
) -> Result<(), TraceReplayError> {
    let device = Device::system_default()
        .ok_or_else(|| TraceReplayError::Metal("no system Metal device".into()))?;
    super::hang_watchdog::start(&device);
    prepared.report.device_name = Some(device.name().to_owned());
    let compile = Instant::now();
    let library = device
        .new_library_with_source(include_str!("trace_replay.metal"), &CompileOptions::new())
        .map_err(TraceReplayError::Metal)?;
    let function = library
        .get_function("jolt_trace_replay", None)
        .map_err(TraceReplayError::Metal)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(TraceReplayError::Metal)?;
    prepared.report.compile_seconds = compile.elapsed().as_secs_f64();
    let instruction_bytes = prepared.instructions.len() * size_of::<Instruction>();
    let checkpoint_bytes = prepared.chunks.len() * size_of::<Chunk>();
    let row_bytes = prepared.instructions.len() * size_of::<Values>();
    let final_bytes = prepared.chunks.len() * REGISTER_COUNT * size_of::<u64>();
    let status_bytes = prepared.chunks.len() * size_of::<u32>();
    let sizes = [
        instruction_bytes,
        checkpoint_bytes,
        row_bytes,
        final_bytes,
        status_bytes,
    ];
    if sizes
        .iter()
        .any(|&bytes| bytes == 0 || bytes as u64 > device.max_buffer_length())
        || sizes.iter().sum::<usize>() > MAX_EXTRA_BUFFER_BYTES
    {
        return Err(TraceReplayError::Bounds(
            "Metal buffer length/cumulative allocation limit",
        ));
    }
    let instructions = buffer_from_slice(&device, &prepared.instructions);
    let checkpoints = buffer_from_slice(&device, &prepared.chunks);
    let output = device.new_buffer(row_bytes as u64, MTLResourceOptions::StorageModeShared);
    let final_state = device.new_buffer(final_bytes as u64, MTLResourceOptions::StorageModeShared);
    let status = device.new_buffer(status_bytes as u64, MTLResourceOptions::StorageModeShared);
    prepared.report.device_buffer_bytes =
        [&instructions, &checkpoints, &output, &final_state, &status]
            .into_iter()
            .map(|buffer| buffer.length())
            .sum();
    let params = Params {
        chunks: prepared.chunks.len() as u32,
        instructions: prepared.instructions.len() as u32,
    };
    let queue = device.new_command_queue();
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for (index, buffer) in [&instructions, &checkpoints, &output, &final_state, &status]
        .into_iter()
        .enumerate()
    {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    set_inline_bytes(encoder, 5, &params);
    let width = pipeline
        .thread_execution_width()
        .min(pipeline.max_total_threads_per_threadgroup());
    if width == 0 {
        encoder.end_encoding();
        return Err(TraceReplayError::Bounds("zero pipeline execution width"));
    }
    encoder.dispatch_thread_groups(
        MTLSize {
            width: (prepared.chunks.len() as u64).div_ceil(width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    let command_wall = Instant::now();
    command.commit();
    prepared.report.submitted_commands = 1;
    crate::metal::profiling::record_submission(command);
    // Always reap the one submitted command before any buffer can be freed,
    // including failure paths. Snapshot also reaps profiling's retained handle.
    command.wait_until_completed();
    let _completed_snapshot = crate::metal::profiling::snapshot();
    prepared.report.command_wall_seconds = command_wall.elapsed().as_secs_f64();
    prepared.report.gpu_seconds = completed_command_gpu_time(command)
        .map_err(|e| TraceReplayError::Metal(e.to_string()))?
        .as_secs_f64();
    prepared.report.completed_commands = 1;
    // SAFETY: every dispatched chunk writes a status before any early return;
    // the one submitted command completed before this shared-buffer read.
    let gpu_status = unsafe {
        std::slice::from_raw_parts(status.contents().cast::<u32>(), prepared.chunks.len())
    };
    for (chunk_id, &status) in gpu_status.iter().enumerate() {
        if status != 1 {
            return Err(TraceReplayError::GpuRejected(chunk_id));
        }
    }
    // SAFETY: all chunk statuses are successful, which guarantees every row
    // and end-register slot was initialized. These integer POD structs match
    // the statically checked Metal ABI and the command has completed.
    let (gpu_rows, gpu_final) = unsafe {
        (
            std::slice::from_raw_parts(
                output.contents().cast::<Values>(),
                prepared.instructions.len(),
            ),
            std::slice::from_raw_parts(
                final_state.contents().cast::<u64>(),
                prepared.chunks.len() * REGISTER_COUNT,
            ),
        )
    };
    for (chunk_id, chunk) in prepared.chunks.iter().enumerate() {
        let physical_start = prepared.physical_starts[chunk_id];
        for offset in 0..chunk.count as usize {
            let row = &rows[physical_start + offset];
            let actual = gpu_rows[chunk.start as usize + offset];
            for (column, expected, actual) in [
                ("rs1", row.rs1_value(), actual.rs1),
                ("rs2", row.rs2_value(), actual.rs2),
                ("rd_pre", row.rd_pre_value(), actual.rd_pre),
                ("rd_write", row.rd_write_value(), actual.rd_write),
            ] {
                compare(physical_start + offset, column, expected, actual)?;
            }
        }
        for r in 0..REGISTER_COUNT {
            compare(
                physical_start + chunk.count as usize - 1,
                "final_register",
                prepared.expected_final[chunk_id][r],
                gpu_final[chunk_id * REGISTER_COUNT + r],
            )?;
        }
    }
    Ok(())
}

fn compare(
    row: usize,
    column: &'static str,
    expected: u64,
    actual: u64,
) -> Result<(), TraceReplayError> {
    if expected == actual {
        Ok(())
    } else {
        Err(TraceReplayError::Parity {
            row,
            column,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests construct explicit CPU oracle fixtures"
)]
mod tests {
    use super::*;
    use jolt_riscv::{CapturedState, JoltInstructionRow, NonMemoryState, NormalizedOperands};

    fn append(
        rows: &mut Vec<JoltTraceRow>,
        registers: &mut [u64; REGISTER_COUNT],
        kind: Kind,
        operands: NormalizedOperands,
        result: u64,
    ) {
        let state = NonMemoryState {
            rs1_value: operands.rs1.map_or(0, |r| registers[r as usize]),
            rs2_value: operands.rs2.map_or(0, |r| registers[r as usize]),
            rd_pre_value: operands.rd.map_or(0, |r| registers[r as usize]),
            rd_write_value: if operands.rd == Some(0) { 0 } else { result },
        };
        let instruction = JoltInstructionRow {
            instruction_kind: kind,
            address: 0x8000_0000 + rows.len() * 4,
            operands,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        rows.push(
            JoltTraceRow::from_components(
                CapturedState::NonMemory(state),
                &instruction,
                rows.len() as u32 + 1,
            )
            .unwrap(),
        );
        if let Some(rd) = operands.rd {
            registers[rd as usize] = state.rd_write_value;
        }
    }

    fn immediate(rs1: u8, rd: u8, imm: i128) -> NormalizedOperands {
        NormalizedOperands {
            rs1: Some(rs1),
            rs2: None,
            rd: Some(rd),
            imm,
        }
    }

    fn binary(rs1: u8, rs2: u8, rd: u8) -> NormalizedOperands {
        NormalizedOperands {
            rs1: Some(rs1),
            rs2: Some(rs2),
            rd: Some(rd),
            imm: 0,
        }
    }

    fn fixture() -> Vec<JoltTraceRow> {
        let mut rows = Vec::new();
        let mut registers = [0; REGISTER_COUNT];
        registers[127] = u64::MAX;
        append(
            &mut rows,
            &mut registers,
            Kind::ADDI,
            immediate(0, 1, -1),
            u64::MAX,
        );
        append(&mut rows, &mut registers, Kind::ADDI, immediate(1, 1, 1), 0);
        append(&mut rows, &mut registers, Kind::SUB, binary(0, 127, 2), 1);
        append(&mut rows, &mut registers, Kind::MUL, binary(127, 127, 3), 1);
        append(&mut rows, &mut registers, Kind::XOR, binary(2, 3, 4), 0);
        append(
            &mut rows,
            &mut registers,
            Kind::ANDI,
            immediate(127, 5, 255),
            255,
        );
        append(
            &mut rows,
            &mut registers,
            Kind::ORI,
            immediate(4, 6, 0x8000_0000),
            0x8000_0000,
        );
        append(
            &mut rows,
            &mut registers,
            Kind::ADDIW,
            immediate(6, 6, 0),
            0xffff_ffff_8000_0000,
        );
        append(
            &mut rows,
            &mut registers,
            Kind::VIRTUAL_SIGN_EXTEND_WORD,
            immediate(6, 7, 0),
            0xffff_ffff_8000_0000,
        );
        append(
            &mut rows,
            &mut registers,
            Kind::VIRTUAL_ZERO_EXTEND_WORD,
            immediate(6, 8, 0),
            0x8000_0000,
        );
        append(
            &mut rows,
            &mut registers,
            Kind::VirtualPow2I,
            NormalizedOperands {
                rs1: None,
                rs2: None,
                rd: Some(9),
                imm: 63,
            },
            1 << 63,
        );
        append(
            &mut rows,
            &mut registers,
            Kind::VirtualSRLI,
            immediate(9, 10, 1i128 << 63),
            1,
        );
        append(
            &mut rows,
            &mut registers,
            Kind::VirtualSRAI,
            immediate(9, 11, 8),
            0xf000_0000_0000_0000,
        );
        append(
            &mut rows,
            &mut registers,
            Kind::VirtualSRLIW,
            immediate(8, 12, 1 << 31),
            1,
        );
        append(
            &mut rows,
            &mut registers,
            Kind::VirtualSRAIW,
            immediate(8, 13, 1 << 31),
            u64::MAX,
        );
        append(
            &mut rows,
            &mut registers,
            Kind::VirtualPow2,
            immediate(2, 14, 0),
            2,
        );
        append(
            &mut rows,
            &mut registers,
            Kind::VirtualMULI,
            immediate(14, 15, -1),
            u64::MAX - 1,
        );
        append(&mut rows, &mut registers, Kind::ADDW, binary(6, 6, 16), 0);
        append(
            &mut rows,
            &mut registers,
            Kind::MULW,
            binary(127, 127, 17),
            1,
        );
        append(&mut rows, &mut registers, Kind::ADDI, immediate(0, 0, 3), 0);
        append(
            &mut rows,
            &mut registers,
            Kind::XORI,
            immediate(127, 18, -1),
            0,
        );
        append(&mut rows, &mut registers, Kind::SLT, binary(127, 2, 19), 1);
        append(&mut rows, &mut registers, Kind::SLTU, binary(127, 2, 20), 0);
        append(&mut rows, &mut registers, Kind::OR, binary(2, 3, 21), 1);
        append(
            &mut rows,
            &mut registers,
            Kind::ANDN,
            binary(127, 2, 22),
            u64::MAX - 1,
        );
        rows
    }

    #[test]
    fn first_touch_checkpoint_does_not_upload_later_cpu_results() {
        let rows = fixture();
        let prepared = prepare(&rows).unwrap();
        assert_eq!(prepared.chunks.len(), 1);
        assert_eq!(prepared.chunks[0].initial[1], 0);
        assert_eq!(prepared.chunks[0].initial[6], 0);
        assert_eq!(prepared.chunks[0].initial[127], u64::MAX);
        assert_eq!(prepared.expected_final[0][6], 0xffff_ffff_8000_0000);
        assert!(prepared.report.extra_buffer_bytes_bound <= MAX_EXTRA_BUFFER_BYTES);
    }

    #[test]
    fn unsupported_and_short_runs_decline_without_gpu_submission() {
        let report = diagnose(&[JoltTraceRow::no_op(); 16]).unwrap();
        assert_eq!(report.supported_rows, 0);
        assert_eq!(report.submitted_commands, 0);
        assert!(!report.exact_parity);
        let rows = fixture();
        let report = diagnose(&rows[..MIN_CHUNK_ROWS - 1]).unwrap();
        assert_eq!(report.short_run_rows, MIN_CHUNK_ROWS - 1);
        assert_eq!(report.sampled_rows, 0);
    }

    #[test]
    fn inconsistent_cpu_dependency_fails_before_submission() {
        let mut rows = fixture();
        let instruction = JoltInstructionRow {
            instruction_kind: Kind::ADDI,
            address: 0x8000_0004,
            operands: immediate(1, 1, 1),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        rows[1] = JoltTraceRow::from_components(
            CapturedState::NonMemory(NonMemoryState {
                rs1_value: 7,
                rs2_value: 0,
                rd_pre_value: u64::MAX,
                rd_write_value: 0,
            }),
            &instruction,
            2,
        )
        .unwrap();
        assert!(matches!(
            prepare(&rows),
            Err(TraceReplayError::Source { row: 1, .. })
        ));
    }

    #[test]
    fn register_id_128_fails_closed() {
        let instruction = JoltInstructionRow {
            instruction_kind: Kind::ADDI,
            address: 0x8000_0000,
            operands: immediate(128, 1, 1),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        let row = JoltTraceRow::from_components(CapturedState::default(), &instruction, 1).unwrap();
        assert!(matches!(
            prepare(&[row]),
            Err(TraceReplayError::Source { .. })
        ));
    }

    #[test]
    fn sampling_caps_do_not_hide_total_coverage() {
        let run = fixture();
        let rows: Vec<_> = (0..=MAX_CHUNKS).flat_map(|_| run.iter().copied()).collect();
        let prepared = prepare(&rows).unwrap();
        assert_eq!(prepared.chunks.len(), MAX_CHUNKS);
        assert_eq!(prepared.report.supported_rows, rows.len());
        assert_eq!(prepared.report.eligible_rows, rows.len());
        assert_eq!(prepared.report.sampled_rows, MAX_CHUNKS * run.len());
        assert!(prepared
            .chunks
            .iter()
            .all(|c| c.count as usize <= MAX_CHUNK_ROWS));
        assert!(prepared.report.extra_buffer_bytes_bound <= MAX_EXTRA_BUFFER_BYTES);
    }

    #[test]
    fn conflicting_sampled_bytecode_descriptor_fails_closed() {
        let mut rows = fixture();
        let mut changed = fixture();
        let instruction = JoltInstructionRow {
            instruction_kind: Kind::ADDI,
            address: changed[0].unexpanded_pc() as usize,
            operands: immediate(0, 1, -2),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        changed[0] =
            JoltTraceRow::from_components(changed[0].captured_state(), &instruction, 1).unwrap();
        rows.extend(changed);
        assert!(matches!(
            prepare(&rows),
            Err(TraceReplayError::Source {
                detail: "one bytecode PC has conflicting static instruction descriptors",
                ..
            })
        ));
    }

    #[test]
    fn metal_alu_replay_matches_dependencies_x0_immediates_and_overflow() {
        // This is a real Metal execution test, never a CPU substitute/skip.
        let rows = fixture();
        let report = diagnose(&rows).unwrap();
        assert!(report.exact_parity);
        assert_eq!(report.sampled_rows, rows.len());
        assert_eq!(report.submitted_commands, 1);
        assert_eq!(report.completed_commands, 1);
        assert!(report.device_buffer_bytes > 0);
    }

    #[test]
    fn metal_alu_replay_detects_wrong_oracle_result() {
        let mut rows = fixture();
        let index = rows.len() - 1;
        let original = rows[index];
        let instruction = JoltInstructionRow {
            instruction_kind: Kind::ANDN,
            address: original.unexpanded_pc() as usize,
            operands: binary(127, 2, 22),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        rows[index] = JoltTraceRow::from_components(
            CapturedState::NonMemory(NonMemoryState {
                rs1_value: original.rs1_value(),
                rs2_value: original.rs2_value(),
                rd_pre_value: original.rd_pre_value(),
                rd_write_value: 7,
            }),
            &instruction,
            original.pc() as u32,
        )
        .unwrap();
        assert!(matches!(
            diagnose(&rows),
            Err(TraceReplayError::Parity {
                column: "rd_write",
                ..
            })
        ));
    }

    #[test]
    fn metal_alu_replay_rejects_invalid_word_shift_mask() {
        let mut rows = fixture();
        let mut registers = [0; REGISTER_COUNT];
        // Start a fresh independent run so its deliberately invalid shift mask
        // cannot be hidden by any earlier CPU-derived data dependency.
        rows.push(JoltTraceRow::no_op());
        let start = rows.len();
        for _ in 0..8 {
            append(
                &mut rows,
                &mut registers,
                Kind::VirtualSRLIW,
                immediate(0, 1, 0),
                0,
            );
        }
        assert!(start > 0);
        assert!(matches!(
            diagnose(&rows),
            Err(TraceReplayError::GpuRejected(1))
        ));
    }
}
