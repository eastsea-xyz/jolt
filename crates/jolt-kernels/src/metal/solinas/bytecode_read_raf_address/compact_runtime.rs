//! Large-domain pushforwards: O(K) counts, O(T) row indices, bounded item waves.
//! Host work is prefix metadata only; all nine field sums execute on Metal.

use std::{
    mem::{size_of, size_of_val},
    slice,
    time::Duration,
};

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use metal::{objc::rc::autoreleasepool, Buffer, MTLResourceOptions, MTLSize, NSRange};

use super::{
    carrier::{AddressMajorShape, INNER_LOG2},
    worklist::{BYTECODE_ADDRESS_PUSHFORWARD_STAGES, BYTECODE_ADDRESS_WORK_ITEM_ROWS},
    worklist_runtime::{
        field_bytes, flatten_tables, shader_count, to_u64, validate_pipeline, validate_table_shape,
        BytecodeAddressSparseParams, BytecodeAddressSparseRuntimeError,
    },
};
use crate::metal::solinas::{
    buffer_from_slice, completed_command_gpu_time, set_inline_bytes, BooleanityRows, Fp128,
    SolinasMetal,
};

pub(crate) const COMPACT_WAVE_ITEMS: usize = 1 << 18;
const MAX_MEMBER_BYTES: usize = 6 << 30;
const THREADS: usize = 256;
const WORKER_THREADS: usize = 128;
const ITEMS_PER_GROUP: usize = 4;
const COUNT: &str = "solinas_bytecode_address_compact_count";
const SCATTER: &str = "solinas_bytecode_address_compact_scatter";
const WORKER: &str = "solinas_bytecode_address_compact_worker";
const REDUCE: &str = "solinas_bytecode_address_compact_reduce";

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompactItem {
    address: u32,
    start: u32,
    count: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompactRange {
    address: u32,
    begin: u32,
    end: u32,
}

const _: [(); 16] = [(); size_of::<CompactItem>()];
const _: [(); 12] = [(); size_of::<CompactRange>()];

#[derive(Default)]
struct WaveCursor {
    address: usize,
    consumed: u32,
}

struct Wave {
    items: Vec<CompactItem>,
    ranges: Vec<CompactRange>,
}

impl WaveCursor {
    fn next(&mut self, offsets: &[u32], limit: usize) -> Wave {
        let mut wave = Wave {
            items: Vec::new(),
            ranges: Vec::new(),
        };
        while self.address + 1 < offsets.len() && wave.items.len() < limit {
            let population = offsets[self.address + 1] - offsets[self.address];
            if self.consumed == population {
                self.address += 1;
                self.consumed = 0;
                continue;
            }
            let begin = wave.items.len() as u32;
            while self.consumed < population && wave.items.len() < limit {
                let count =
                    (population - self.consumed).min(BYTECODE_ADDRESS_WORK_ITEM_ROWS as u32);
                wave.items.push(CompactItem {
                    address: self.address as u32,
                    start: offsets[self.address] + self.consumed,
                    count,
                    reserved: 0,
                });
                self.consumed += count;
            }
            wave.ranges.push(CompactRange {
                address: self.address as u32,
                begin,
                end: wave.items.len() as u32,
            });
        }
        wave
    }
}

pub(crate) struct BytecodeAddressCompactObservation {
    pub(crate) pushforwards: Vec<Vec<AkitaField>>,
    pub(crate) first_push_pc: usize,
    pub(crate) work_items: usize,
    pub(crate) waves: usize,
    pub(crate) member_owned_bytes: usize,
    pub(crate) gpu_time: Duration,
}

impl SolinasMetal {
    pub(crate) fn execute_bytecode_address_compact(
        &self,
        rows: BooleanityRows,
        log_addresses: u32,
        e_lo: &[Vec<AkitaField>],
        e_hi: &[Vec<AkitaField>],
    ) -> Result<BytecodeAddressCompactObservation, BytecodeAddressSparseRuntimeError> {
        self.execute_bytecode_address_compact_inner(
            rows,
            log_addresses,
            e_lo,
            e_hi,
            COMPACT_WAVE_ITEMS,
        )
    }

    #[cfg(test)]
    pub(crate) fn execute_bytecode_address_compact_with_wave_limit(
        &self,
        rows: BooleanityRows,
        log_addresses: u32,
        e_lo: &[Vec<AkitaField>],
        e_hi: &[Vec<AkitaField>],
        wave_limit: usize,
    ) -> Result<BytecodeAddressCompactObservation, BytecodeAddressSparseRuntimeError> {
        self.execute_bytecode_address_compact_inner(rows, log_addresses, e_lo, e_hi, wave_limit)
    }

    fn execute_bytecode_address_compact_inner(
        &self,
        rows: BooleanityRows,
        log_addresses: u32,
        e_lo: &[Vec<AkitaField>],
        e_hi: &[Vec<AkitaField>],
        wave_limit: usize,
    ) -> Result<BytecodeAddressCompactObservation, BytecodeAddressSparseRuntimeError> {
        self.validate_booleanity_rows(&rows)?;
        let cycles = rows.len();
        if !(17..=21).contains(&log_addresses)
            || wave_limit == 0
            || wave_limit > COMPACT_WAVE_ITEMS
            || cycles < 1 << INNER_LOG2
            || !cycles.is_power_of_two()
        {
            return Err(BytecodeAddressSparseRuntimeError::InvalidState);
        }
        let shape = AddressMajorShape::new(cycles.ilog2(), log_addresses, INNER_LOG2)?;
        let addresses = shape.addresses()?;
        validate_table_shape("compact E_lo", e_lo, shape.inner_length()?)?;
        validate_table_shape("compact E_hi", e_hi, shape.outer_length()?)?;
        let flat_lo = flatten_tables(e_lo);
        let flat_hi = flatten_tables(e_hi);
        self.validate_inputs("compact E_lo", &flat_lo)?;
        self.validate_inputs("compact E_hi", &flat_hi)?;
        let sizes = [
            addresses.checked_mul(4), // Shared counts reused as scatter cursors.
            Some(8),                  // invalid count and first PC.
            cycles.checked_mul(4),    // Original row indices, no wide row copies.
            field_bytes(flat_lo.len()).ok(),
            field_bytes(flat_hi.len()).ok(),
            field_bytes(BYTECODE_ADDRESS_PUSHFORWARD_STAGES * wave_limit).ok(),
            wave_limit.checked_mul(size_of::<CompactItem>()),
            wave_limit.checked_mul(size_of::<CompactRange>()),
            field_bytes(BYTECODE_ADDRESS_PUSHFORWARD_STAGES * addresses).ok(),
        ];
        let sizes: Vec<usize> = sizes.into_iter().collect::<Option<_>>().ok_or(
            BytecodeAddressSparseRuntimeError::SizeOverflow("compact buffers"),
        )?;
        let member_owned_bytes = sizes
            .iter()
            .try_fold(0usize, |sum, bytes| sum.checked_add(*bytes))
            .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                "compact working set",
            ))?;
        if member_owned_bytes > MAX_MEMBER_BYTES {
            return Err(BytecodeAddressSparseRuntimeError::ResidentStorageTooLarge {
                bytes: member_owned_bytes,
                maximum: MAX_MEMBER_BYTES,
            });
        }
        self.validate_additional_working_set(to_u64("compact working set", member_owned_bytes)?)?;
        for bytes in &sizes {
            self.validate_buffer_length(to_u64("compact buffer", *bytes)?)?;
        }
        let count_pipeline = self.compile_named_pipeline(COUNT)?;
        let scatter_pipeline = self.compile_named_pipeline(SCATTER)?;
        let worker_pipeline = self.compile_named_pipeline(WORKER)?;
        let reduce_pipeline = self.compile_named_pipeline(REDUCE)?;
        for (name, pipeline, threads) in [
            (COUNT, &count_pipeline, THREADS),
            (SCATTER, &scatter_pipeline, THREADS),
            (WORKER, &worker_pipeline, WORKER_THREADS),
            (REDUCE, &reduce_pipeline, THREADS),
        ] {
            validate_pipeline(name, Self::limits(pipeline), threads)?;
        }
        let shared = |bytes: usize| {
            self.device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
        };
        let private = |bytes: usize| {
            self.device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModePrivate)
        };
        let counts = shared(sizes[0]);
        let status = shared(sizes[1]);
        let indices = private(sizes[2]);
        let lo_buffer = buffer_from_slice(&self.device, &flat_lo);
        let hi_buffer = buffer_from_slice(&self.device, &flat_hi);
        drop((flat_lo, flat_hi));
        let partials = private(sizes[5]);
        let items = shared(sizes[6]);
        let ranges = shared(sizes[7]);
        let output = shared(sizes[8]);
        let mut params = BytecodeAddressSparseParams {
            physical_rows: shader_count("compact rows", cycles)?,
            addresses: shader_count("compact addresses", addresses)?,
            inner_length: shader_count("compact inner", shape.inner_length()?)?,
            outer_length: shader_count("compact outer", shape.outer_length()?)?,
            work_items: 0,
            stages: BYTECODE_ADDRESS_PUSHFORWARD_STAGES as u32,
            base_stages: 5,
            reserved: 0,
        };
        let mut gpu_time = autoreleasepool(|| {
            let command = self.queue.new_command_buffer();
            let blit = command.new_blit_command_encoder();
            for buffer in [&counts, &status, &output] {
                blit.fill_buffer(buffer, NSRange::new(0, buffer.length()), 0);
            }
            blit.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&count_pipeline);
            encoder.set_buffer(0, Some(rows.buffer()), 0);
            encoder.set_buffer(1, Some(&counts), 0);
            encoder.set_buffer(2, Some(&status), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_thread_groups(groups(cycles, THREADS), threads(THREADS));
            encoder.end_encoding();
            command.commit();
            crate::metal::profiling::record_submission(command);
            command.wait_until_completed();
            completed_command_gpu_time(command)
        })?;
        let status_values = read_u32(&status, 2)?;
        let first_push_pc = status_values[1] as usize;
        if status_values[0] != 0 || first_push_pc >= addresses {
            return Err(BytecodeAddressSparseRuntimeError::InvalidResidentState {
                invalid_rows: status_values[0],
            });
        }
        let offsets = prefix_counts(read_u32(&counts, addresses)?, params.physical_rows)?;
        // SAFETY: counts is uniquely owned, exactly K u32s, and the count
        // command completed. No device accesses overlap this host rewrite.
        unsafe { slice::from_raw_parts_mut(counts.contents().cast::<u32>(), addresses) }
            .copy_from_slice(&offsets[..addresses]);
        gpu_time += autoreleasepool(|| {
            let command = self.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&scatter_pipeline);
            encoder.set_buffer(0, Some(rows.buffer()), 0);
            encoder.set_buffer(1, Some(&counts), 0);
            encoder.set_buffer(2, Some(&indices), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_thread_groups(groups(cycles, THREADS), threads(THREADS));
            encoder.end_encoding();
            command.commit();
            crate::metal::profiling::record_submission(command);
            command.wait_until_completed();
            completed_command_gpu_time(command)
        })?;
        if read_u32(&counts, addresses)? != &offsets[1..] {
            return Err(BytecodeAddressSparseRuntimeError::InvalidState);
        }
        let mut cursor = WaveCursor::default();
        let mut work_items = 0;
        let mut waves = 0;
        loop {
            let wave = cursor.next(&offsets, wave_limit);
            if wave.items.is_empty() {
                break;
            }
            write_shared(&items, &wave.items)?;
            write_shared(&ranges, &wave.ranges)?;
            params.work_items = wave.items.len() as u32;
            params.reserved = wave.ranges.len() as u32;
            gpu_time += autoreleasepool(|| {
                let command = self.queue.new_command_buffer();
                let worker = command.new_compute_command_encoder();
                worker.set_compute_pipeline_state(&worker_pipeline);
                for (index, buffer) in [
                    &indices,
                    rows.buffer(),
                    &items,
                    &lo_buffer,
                    &hi_buffer,
                    &partials,
                ]
                .into_iter()
                .enumerate()
                {
                    worker.set_buffer(index as u64, Some(buffer), 0);
                }
                set_inline_bytes(worker, 6, &params);
                worker.dispatch_thread_groups(
                    groups(wave.items.len(), ITEMS_PER_GROUP),
                    threads(WORKER_THREADS),
                );
                worker.end_encoding();
                let reduce = command.new_compute_command_encoder();
                reduce.set_compute_pipeline_state(&reduce_pipeline);
                for (index, buffer) in [&partials, &ranges, &output].into_iter().enumerate() {
                    reduce.set_buffer(index as u64, Some(buffer), 0);
                }
                set_inline_bytes(reduce, 3, &params);
                reduce.dispatch_thread_groups(
                    groups(
                        wave.ranges.len() * BYTECODE_ADDRESS_PUSHFORWARD_STAGES,
                        THREADS,
                    ),
                    threads(THREADS),
                );
                reduce.end_encoding();
                command.commit();
                crate::metal::profiling::record_submission(command);
                command.wait_until_completed();
                completed_command_gpu_time(command)
            })?;
            work_items += wave.items.len();
            waves += 1;
        }
        // Release all scratch before constructing the nine host vectors. No
        // flattened host output copy coexists with those vectors.
        drop((
            counts, status, indices, lo_buffer, hi_buffer, partials, items, ranges, offsets, rows,
        ));
        let fields = BYTECODE_ADDRESS_PUSHFORWARD_STAGES * addresses;
        // SAFETY: output is shared, holds exactly `fields` initialized field
        // elements, and every producing command has completed.
        let values = unsafe { slice::from_raw_parts(output.contents().cast::<Fp128>(), fields) };
        self.validate_inputs("compact bytecode output", values)?;
        let pushforwards = values
            .chunks_exact(addresses)
            .map(|stage| {
                stage
                    .iter()
                    .map(|value| (*value).into_jolt_field())
                    .collect()
            })
            .collect();
        Ok(BytecodeAddressCompactObservation {
            pushforwards,
            first_push_pc,
            work_items,
            waves,
            member_owned_bytes,
            gpu_time,
        })
    }
}

fn prefix_counts(counts: &[u32], rows: u32) -> Result<Vec<u32>, BytecodeAddressSparseRuntimeError> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut total = 0u32;
    offsets.push(total);
    for count in counts {
        total =
            total
                .checked_add(*count)
                .ok_or(BytecodeAddressSparseRuntimeError::SizeOverflow(
                    "compact prefix",
                ))?;
        offsets.push(total);
    }
    if total != rows {
        return Err(BytecodeAddressSparseRuntimeError::InvalidResidentSummary {
            expected_rows: rows as usize,
            summarized_rows: total as usize,
        });
    }
    Ok(offsets)
}

fn read_u32(buffer: &Buffer, len: usize) -> Result<&[u32], BytecodeAddressSparseRuntimeError> {
    if buffer.length() != (len * size_of::<u32>()) as u64 {
        return Err(BytecodeAddressSparseRuntimeError::InvalidState);
    }
    // SAFETY: exact shared-buffer length and completed producing commands.
    Ok(unsafe { slice::from_raw_parts(buffer.contents().cast::<u32>(), len) })
}

fn write_shared<T: Copy>(
    buffer: &Buffer,
    values: &[T],
) -> Result<(), BytecodeAddressSparseRuntimeError> {
    if size_of_val(values) as u64 > buffer.length() {
        return Err(BytecodeAddressSparseRuntimeError::InvalidState);
    }
    // SAFETY: destination has sufficient shared storage and the preceding
    // command completed; each wave exclusively owns its host write window.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), buffer.contents().cast::<T>(), values.len());
    }
    Ok(())
}

fn groups(elements: usize, width: usize) -> MTLSize {
    MTLSize {
        width: elements.div_ceil(width) as u64,
        height: 1,
        depth: 1,
    }
}
fn threads(width: usize) -> MTLSize {
    MTLSize {
        width: width as u64,
        height: 1,
        depth: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_prefix_rejects_overflow_and_population_mismatch() {
        assert!(prefix_counts(&[u32::MAX, 1], u32::MAX).is_err());
        assert!(prefix_counts(&[1, 2], 4).is_err());
        assert_eq!(
            prefix_counts(&[0, 3, 0, 2], 5).unwrap_or_default(),
            [0, 0, 3, 3, 5]
        );
    }

    #[test]
    fn compact_waves_cover_every_row_once_across_a_hot_address_seam() {
        let offsets = prefix_counts(&[0, 12_001, 2], 12_003).unwrap_or_default();
        assert_eq!(offsets, [0, 0, 12_001, 12_003]);
        let mut cursor = WaveCursor::default();
        let mut seen = Vec::new();
        let mut wave_count = 0;
        loop {
            let wave = cursor.next(&offsets, 2);
            if wave.items.is_empty() {
                break;
            }
            assert!(wave.items.len() <= 2);
            assert!(wave
                .ranges
                .windows(2)
                .all(|pair| pair[0].address < pair[1].address));
            for item in wave.items {
                seen.extend(item.start..item.start + item.count);
            }
            wave_count += 1;
        }
        assert_eq!(wave_count, 2);
        assert_eq!(seen, (0..12_003).collect::<Vec<_>>());
    }
}
