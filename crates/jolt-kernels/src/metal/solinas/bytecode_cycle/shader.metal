#define BYTECODE_CYCLE_TABLES 5u
#define BYTECODE_CYCLE_SAMPLES 4u

struct BytecodeCycleParams {
    uint source_elements;
    uint message_pairs;
    uint threadgroups;
    uint reserved;
};

inline void bytecode_cycle_grid_from_anchors(
    SolinasFp128 at_zero,
    SolinasFp128 at_one,
    SolinasFp128 leading,
    thread SolinasFp128* grid)
{
    SolinasFp128 second_difference = solinas_add(leading, leading);
    SolinasFp128 delta_two = solinas_add(
        solinas_sub(at_one, at_zero),
        second_difference);
    grid[0] = at_zero;
    grid[1] = solinas_add(at_one, delta_two);
    SolinasFp128 delta_three = solinas_add(delta_two, second_difference);
    grid[2] = solinas_add(grid[1], delta_three);
    grid[3] = solinas_add(solinas_add(grid[2], delta_three), second_difference);
}

inline void bytecode_cycle_q10(
    thread const SolinasFp128* lo,
    thread const SolinasFp128* hi,
    thread SolinasFp128* q)
{
    SolinasFp128 ra_grid[BYTECODE_CYCLE_SAMPLES];
    SolinasFp128 ra_leading = solinas_mul_wide(
        solinas_sub(hi[3], lo[3]),
        solinas_sub(hi[4], lo[4]));
    bytecode_cycle_grid_from_anchors(
        solinas_mul_wide(lo[3], lo[4]),
        solinas_mul_wide(hi[3], hi[4]),
        ra_leading,
        ra_grid);

    SolinasFp128 coefficient_grid[BYTECODE_CYCLE_SAMPLES];
    SolinasFp128 coefficient_zero = solinas_add(
        lo[0],
        solinas_mul_wide(lo[2], lo[1]));
    SolinasFp128 coefficient_one = solinas_add(
        hi[0],
        solinas_mul_wide(hi[2], hi[1]));
    SolinasFp128 coefficient_leading = solinas_mul_wide(
        solinas_sub(hi[2], lo[2]),
        solinas_sub(hi[1], lo[1]));
    bytecode_cycle_grid_from_anchors(
        coefficient_zero,
        coefficient_one,
        coefficient_leading,
        coefficient_grid);

    for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
        q[sample] = solinas_mul_wide(ra_grid[sample], coefficient_grid[sample]);
    }
}

inline void bytecode_cycle_finish_block(
    thread SolinasFp128* lanes,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint group,
    uint groups,
    uint lane_in_simd,
    uint simdgroup,
    uint simdgroups)
{
    for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
        SolinasFp128 sum = solinas_simd_sum_32(lanes[sample]);
        if (lane_in_simd == 0) {
            shared[sample * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
            SolinasFp128 sum = lane_in_simd < simdgroups
                ? shared[sample * simdgroups + lane_in_simd]
                : solinas_zero();
            sum = solinas_simd_sum_32(sum);
            if (lane_in_simd == 0) {
                partials[sample * groups + group] = sum;
            }
        }
    }
}

inline SolinasFp128 bytecode_cycle_load(
    uint table,
    uint index,
    device const SolinasFp128* table_zero,
    device const SolinasFp128* table_one,
    device const SolinasFp128* table_two,
    device const SolinasFp128* table_three,
    device const SolinasFp128* table_four)
{
    switch (table) {
        case 0u: return table_zero[index];
        case 1u: return table_one[index];
        case 2u: return table_two[index];
        case 3u: return table_three[index];
        default: return table_four[index];
    }
}

inline void bytecode_cycle_store(
    uint table,
    uint index,
    SolinasFp128 value,
    device SolinasFp128* table_zero,
    device SolinasFp128* table_one,
    device SolinasFp128* table_two,
    device SolinasFp128* table_three,
    device SolinasFp128* table_four)
{
    switch (table) {
        case 0u: table_zero[index] = value; break;
        case 1u: table_one[index] = value; break;
        case 2u: table_two[index] = value; break;
        case 3u: table_three[index] = value; break;
        default: table_four[index] = value; break;
    }
}

kernel void solinas_bytecode_cycle_q10_message(
    device const SolinasFp128* table_zero [[buffer(0)]],
    device const SolinasFp128* table_one [[buffer(1)]],
    device const SolinasFp128* table_two [[buffer(2)]],
    device const SolinasFp128* table_three [[buffer(3)]],
    device const SolinasFp128* table_four [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant BytecodeCycleParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint local_index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[BYTECODE_CYCLE_SAMPLES];
    for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    uint pair = group * threads + local_index;
    uint stride = params.threadgroups * threads;
    for (; pair < params.message_pairs; pair += stride) {
        SolinasFp128 lo[BYTECODE_CYCLE_TABLES];
        SolinasFp128 hi[BYTECODE_CYCLE_TABLES];
        for (uint table = 0; table < BYTECODE_CYCLE_TABLES; table++) {
            uint source = 2u * pair;
            lo[table] = bytecode_cycle_load(
                table, source, table_zero, table_one, table_two, table_three, table_four);
            hi[table] = bytecode_cycle_load(
                table, source + 1u, table_zero, table_one, table_two, table_three, table_four);
        }
        SolinasFp128 q[BYTECODE_CYCLE_SAMPLES];
        bytecode_cycle_q10(lo, hi, q);
        for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
            lanes[sample] = solinas_add(lanes[sample], q[sample]);
        }
    }

    bytecode_cycle_finish_block(
        lanes,
        partials,
        shared,
        group,
        params.threadgroups,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}

kernel void solinas_bytecode_cycle_q10_transition(
    device const SolinasFp128* source_zero [[buffer(0)]],
    device const SolinasFp128* source_one [[buffer(1)]],
    device const SolinasFp128* source_two [[buffer(2)]],
    device const SolinasFp128* source_three [[buffer(3)]],
    device const SolinasFp128* source_four [[buffer(4)]],
    device SolinasFp128* bound_zero [[buffer(5)]],
    device SolinasFp128* bound_one [[buffer(6)]],
    device SolinasFp128* bound_two [[buffer(7)]],
    device SolinasFp128* bound_three [[buffer(8)]],
    device SolinasFp128* bound_four [[buffer(9)]],
    device SolinasFp128* partials [[buffer(10)]],
    constant SolinasFp128& challenge [[buffer(11)]],
    constant BytecodeCycleParams& params [[buffer(12)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint local_index [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[BYTECODE_CYCLE_SAMPLES];
    for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    uint pair = group * threads + local_index;
    uint stride = params.threadgroups * threads;
    for (; pair < params.message_pairs; pair += stride) {
        SolinasFp128 lo[BYTECODE_CYCLE_TABLES];
        SolinasFp128 hi[BYTECODE_CYCLE_TABLES];
        for (uint table = 0; table < BYTECODE_CYCLE_TABLES; table++) {
            uint source = 4u * pair;
            SolinasFp128 value_zero = bytecode_cycle_load(
                table, source, source_zero, source_one, source_two, source_three, source_four);
            SolinasFp128 value_one = bytecode_cycle_load(
                table, source + 1u, source_zero, source_one, source_two, source_three, source_four);
            SolinasFp128 value_two = bytecode_cycle_load(
                table, source + 2u, source_zero, source_one, source_two, source_three, source_four);
            SolinasFp128 value_three = bytecode_cycle_load(
                table, source + 3u, source_zero, source_one, source_two, source_three, source_four);
            lo[table] = solinas_add(
                value_zero,
                solinas_mul_wide(challenge, solinas_sub(value_one, value_zero)));
            hi[table] = solinas_add(
                value_two,
                solinas_mul_wide(challenge, solinas_sub(value_three, value_two)));
            uint destination = 2u * pair;
            bytecode_cycle_store(
                table,
                destination,
                lo[table],
                bound_zero,
                bound_one,
                bound_two,
                bound_three,
                bound_four);
            bytecode_cycle_store(
                table,
                destination + 1u,
                hi[table],
                bound_zero,
                bound_one,
                bound_two,
                bound_three,
                bound_four);
        }
        SolinasFp128 q[BYTECODE_CYCLE_SAMPLES];
        bytecode_cycle_q10(lo, hi, q);
        for (uint sample = 0; sample < BYTECODE_CYCLE_SAMPLES; sample++) {
            lanes[sample] = solinas_add(lanes[sample], q[sample]);
        }
    }

    bytecode_cycle_finish_block(
        lanes,
        partials,
        shared,
        group,
        params.threadgroups,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}

// The wide path preserves the separate linear RA factors. Multiplying roots
// before binding would change the multilinear extension and the proof.
#define BYTECODE_WIDE_STAGES 9u
#define BYTECODE_WIDE_MAX_RA 8u
#define BYTECODE_WIDE_MAX_TABLES (BYTECODE_WIDE_MAX_RA + 3u)
#define BYTECODE_WIDE_MAX_SAMPLES (BYTECODE_WIDE_MAX_RA + 2u)

struct BytecodeWideParams {
    uint source_elements;
    uint lo_length;
    uint hi_length;
    uint ra_count;
    uint chunk_bits;
    uint ra_entries;
    uint message_pairs;
    uint threadgroups;
};

inline SolinasFp128 bytecode_wide_bind(
    SolinasFp128 lo,
    SolinasFp128 hi,
    SolinasFp128 challenge)
{
    return solinas_add(lo, solinas_mul_wide(challenge, solinas_sub(hi, lo)));
}

inline void bytecode_wide_coefficients(
    uint x_lo,
    uint lo_length,
    device const SolinasFp128* eq_lo,
    threadgroup const SolinasFp128* weighted_hi,
    thread SolinasFp128* factors)
{
    factors[0] = solinas_zero();
    factors[1] = solinas_zero();
    for (uint stage = 0u; stage < BYTECODE_WIDE_STAGES; stage++) {
        uint column = stage < 5u ? 0u : 1u;
        factors[column] = solinas_add(factors[column], solinas_mul_wide(
            weighted_hi[stage], eq_lo[stage * lo_length + x_lo]));
    }
}

inline void bytecode_wide_row_factors(
    device const ulong* rows,
    uint row,
    device const SolinasFp128* ra,
    constant BytecodeWideParams& params,
    thread SolinasFp128* factors)
{
    ulong magnitude = booleanity_row_word(rows, params.source_elements, 3u, row);
    ulong flags = booleanity_row_word(rows, params.source_elements, 4u, row);
    SolinasFp128 increment = solinas_zero();
    increment.limb[0] = (uint)magnitude;
    increment.limb[1] = (uint)(magnitude >> 32);
    factors[2] = (flags >> 63) != 0ul
        ? solinas_sub(solinas_zero(), increment)
        : increment;
    ulong plus_one = flags & BOOLEANITY_SOURCE_PC_MASK;
    ulong pc = plus_one == 0ul ? 0ul : plus_one - 1ul;
    for (uint chunk = 0u; chunk < params.ra_count; chunk++) {
        uint shift = (params.ra_count - chunk - 1u) * params.chunk_bits;
        uint hot = (uint)(pc >> shift) & (params.ra_entries - 1u);
        factors[chunk + 3u] = plus_one == 0ul
            ? solinas_zero()
            : ra[chunk * params.ra_entries + hot];
    }
}

inline void bytecode_wide_accumulate(
    thread SolinasFp128* values,
    thread SolinasFp128* deltas,
    uint ra_count,
    thread SolinasFp128* lanes)
{
    uint tables = ra_count + 3u;
    uint samples = ra_count + 2u;
    for (uint table = 0u; table < tables; table++) {
        deltas[table] = solinas_sub(deltas[table], values[table]);
    }
    for (uint sample = 0u; sample < samples; sample++) {
        // Each factor is linear in t. Advance 0 -> 2 -> 3 -> ... using
        // additions, leaving full-width multiplication for the summand.
        if (sample != 0u) {
            for (uint table = 0u; table < tables; table++) {
                values[table] = solinas_add(values[table], deltas[table]);
                if (sample == 1u) {
                    values[table] = solinas_add(values[table], deltas[table]);
                }
            }
        }
        SolinasFp128 value = solinas_add(values[0], solinas_mul_wide(values[2], values[1]));
        for (uint chunk = 0u; chunk < ra_count; chunk++) {
            value = solinas_mul_wide(value, values[chunk + 3u]);
        }
        lanes[sample] = solinas_add(lanes[sample], value);
    }
}

inline void bytecode_wide_finish(
    thread SolinasFp128* lanes,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint samples,
    uint group,
    uint groups,
    uint lane,
    uint simdgroup,
    uint simdgroups)
{
    for (uint sample = 0u; sample < samples; sample++) {
        SolinasFp128 sum = solinas_simd_sum_32(lanes[sample]);
        if (lane == 0u) {
            shared[sample * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        for (uint sample = 0u; sample < samples; sample++) {
            SolinasFp128 sum = lane < simdgroups
                ? shared[sample * simdgroups + lane]
                : solinas_zero();
            sum = solinas_simd_sum_32(sum);
            if (lane == 0u) {
                partials[sample * groups + group] = sum;
            }
        }
    }
}

kernel void solinas_bytecode_wide_first_message(
    device const ulong* rows [[buffer(0)]],
    device const SolinasFp128* eq_lo [[buffer(1)]],
    device const SolinasFp128* weighted_eq_hi [[buffer(2)]],
    device const SolinasFp128* ra [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant SolinasFp128& entry_weight [[buffer(5)]],
    constant BytecodeWideParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint local [[thread_index_in_threadgroup]],
    uint x_hi [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (local < BYTECODE_WIDE_STAGES) {
        shared[local] = weighted_eq_hi[local * params.hi_length + x_hi];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    SolinasFp128 lanes[BYTECODE_WIDE_MAX_SAMPLES];
    for (uint sample = 0u; sample < params.ra_count + 2u; sample++) {
        lanes[sample] = solinas_zero();
    }
    for (uint pair = local; pair < params.lo_length / 2u; pair += threads) {
        uint x_lo = 2u * pair;
        uint row = x_hi * params.lo_length + x_lo;
        SolinasFp128 lo[BYTECODE_WIDE_MAX_TABLES];
        SolinasFp128 hi[BYTECODE_WIDE_MAX_TABLES];
        bytecode_wide_coefficients(x_lo, params.lo_length, eq_lo, shared, lo);
        bytecode_wide_coefficients(x_lo + 1u, params.lo_length, eq_lo, shared, hi);
        if (row == 0u) {
            lo[0] = solinas_add(lo[0], entry_weight);
        }
        bytecode_wide_row_factors(rows, row, ra, params, lo);
        bytecode_wide_row_factors(rows, row + 1u, ra, params, hi);
        bytecode_wide_accumulate(lo, hi, params.ra_count, lanes);
    }
    bytecode_wide_finish(lanes, partials, shared + BYTECODE_WIDE_STAGES,
        params.ra_count + 2u, x_hi, params.hi_length, lane, simdgroup, threads / 32u);
}

kernel void solinas_bytecode_wide_first_bind_message(
    device const ulong* rows [[buffer(0)]],
    device const SolinasFp128* bound_eq_lo [[buffer(1)]],
    device const SolinasFp128* weighted_eq_hi [[buffer(2)]],
    device const SolinasFp128* ra [[buffer(3)]],
    device SolinasFp128* bound [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant SolinasFp128& challenge [[buffer(6)]],
    constant SolinasFp128& bound_entry_weight [[buffer(7)]],
    constant BytecodeWideParams& params [[buffer(8)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint local [[thread_index_in_threadgroup]],
    uint x_hi [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    if (local < BYTECODE_WIDE_STAGES) {
        shared[local] = weighted_eq_hi[local * params.hi_length + x_hi];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    SolinasFp128 lanes[BYTECODE_WIDE_MAX_SAMPLES];
    for (uint sample = 0u; sample < params.ra_count + 2u; sample++) {
        lanes[sample] = solinas_zero();
    }
    uint bound_lo_length = params.lo_length / 2u;
    uint bound_elements = params.source_elements / 2u;
    for (uint pair = local; pair < bound_lo_length / 2u; pair += threads) {
        uint destination = x_hi * bound_lo_length + 2u * pair;
        uint source = x_hi * params.lo_length + 4u * pair;
        SolinasFp128 lo[BYTECODE_WIDE_MAX_TABLES];
        SolinasFp128 hi[BYTECODE_WIDE_MAX_TABLES];
        bytecode_wide_coefficients(2u * pair, bound_lo_length, bound_eq_lo, shared, lo);
        bytecode_wide_coefficients(2u * pair + 1u, bound_lo_length, bound_eq_lo, shared, hi);
        if (destination == 0u) {
            lo[0] = solinas_add(lo[0], bound_entry_weight);
        }
        SolinasFp128 zero[BYTECODE_WIDE_MAX_TABLES];
        SolinasFp128 one[BYTECODE_WIDE_MAX_TABLES];
        bytecode_wide_row_factors(rows, source, ra, params, zero);
        bytecode_wide_row_factors(rows, source + 1u, ra, params, one);
        for (uint table = 2u; table < params.ra_count + 3u; table++) {
            lo[table] = bytecode_wide_bind(zero[table], one[table], challenge);
        }
        bytecode_wide_row_factors(rows, source + 2u, ra, params, zero);
        bytecode_wide_row_factors(rows, source + 3u, ra, params, one);
        for (uint table = 2u; table < params.ra_count + 3u; table++) {
            hi[table] = bytecode_wide_bind(zero[table], one[table], challenge);
        }
        for (uint table = 0u; table < params.ra_count + 3u; table++) {
            bound[table * bound_elements + destination] = lo[table];
            bound[table * bound_elements + destination + 1u] = hi[table];
        }
        bytecode_wide_accumulate(lo, hi, params.ra_count, lanes);
    }
    bytecode_wide_finish(lanes, partials, shared + BYTECODE_WIDE_STAGES,
        params.ra_count + 2u, x_hi, params.hi_length, lane, simdgroup, threads / 32u);
}

kernel void solinas_bytecode_wide_transition(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* bound [[buffer(1)]],
    device SolinasFp128* partials [[buffer(2)]],
    constant SolinasFp128& challenge [[buffer(3)]],
    constant BytecodeWideParams& params [[buffer(4)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint local [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[BYTECODE_WIDE_MAX_SAMPLES];
    for (uint sample = 0u; sample < params.ra_count + 2u; sample++) {
        lanes[sample] = solinas_zero();
    }
    uint bound_elements = params.source_elements / 2u;
    uint stride = params.threadgroups * threads;
    for (uint pair = group * threads + local; pair < params.message_pairs; pair += stride) {
        SolinasFp128 lo[BYTECODE_WIDE_MAX_TABLES];
        SolinasFp128 hi[BYTECODE_WIDE_MAX_TABLES];
        for (uint table = 0u; table < params.ra_count + 3u; table++) {
            uint offset = table * params.source_elements + 4u * pair;
            lo[table] = bytecode_wide_bind(source[offset], source[offset + 1u], challenge);
            hi[table] = bytecode_wide_bind(source[offset + 2u], source[offset + 3u], challenge);
            uint destination = table * bound_elements + 2u * pair;
            bound[destination] = lo[table];
            bound[destination + 1u] = hi[table];
        }
        bytecode_wide_accumulate(lo, hi, params.ra_count, lanes);
    }
    bytecode_wide_finish(lanes, partials, shared, params.ra_count + 2u,
        group, params.threadgroups, lane, simdgroup, threads / 32u);
}

struct BytecodeWideReductionParams {
    uint input_count;
    uint output_count;
    uint samples;
    uint reserved;
};

kernel void solinas_bytecode_wide_terminal_bind(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* terminal [[buffer(1)]],
    constant SolinasFp128& challenge [[buffer(2)]],
    constant BytecodeWideParams& params [[buffer(3)]],
    uint table [[thread_position_in_grid]])
{
    if (table < params.ra_count + 3u) {
        terminal[table] = bytecode_wide_bind(
            source[2u * table], source[2u * table + 1u], challenge);
    }
}

kernel void solinas_bytecode_wide_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant BytecodeWideReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint sample = 0u; sample < params.samples; sample++) {
        SolinasFp128 value = gid < params.input_count
            ? input[sample * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane == 0u) {
            output[sample * params.output_count + gid / 32u] = value;
        }
    }
}
