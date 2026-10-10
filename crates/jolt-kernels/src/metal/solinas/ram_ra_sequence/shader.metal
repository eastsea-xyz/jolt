// Direct cycle-order RAM RA virtualization for byte and nibble chunks. The
// factor tables, lazy branches, split-eq weights and dense tail stay resident.

#define RAM_RA_MAX_FACTORS 8u
#define RAM_RA_SAMPLES RAM_RA_MAX_FACTORS

struct RamRaMessageParams {
    uint e_in_length;
    uint e_out_length;
    uint factor_count;
    uint chunk_bits;
};

struct RamRaBranchParams {
    uint branch_width;
    uint factor_count;
    uint bins;
    uint reserved;
};

struct RamRaMaterializeParams {
    uint source_elements;
    uint e_in_length;
    uint e_out_length;
    uint factor_count;
    uint chunk_bits;
    uint reserved_0;
    uint reserved_1;
    uint reserved_2;
};

struct RamRaReductionParams {
    uint input_count;
    uint output_count;
    uint columns;
    uint reserved;
};

constant uint ram_ra_branch_width [[function_constant(20)]];

inline uint ram_ra_address_chunk(
    uint address, uint factor, uint factor_count, uint chunk_bits)
{
    return (address >> ((factor_count - 1u - factor) * chunk_bits))
        & ((1u << chunk_bits) - 1u);
}

inline bool ram_ra_gather(
    uint pair,
    uint branch_width,
    uint factor_count,
    uint chunk_bits,
    device const uint* addresses,
    device const SolinasFp128* branches,
    thread InstructionRaLinear* factors)
{
    SolinasFp128 lo[RAM_RA_MAX_FACTORS];
    SolinasFp128 hi[RAM_RA_MAX_FACTORS];
    for (uint factor = 0; factor < factor_count; factor++) {
        lo[factor] = solinas_zero();
        hi[factor] = solinas_zero();
    }

    bool active = false;
    uint original = 2u * pair * branch_width;
    for (uint offset = 0; offset < branch_width; offset++) {
        uint lo_address = addresses[original + offset];
        uint hi_address = addresses[original + branch_width + offset];
        if (lo_address != 0xffffffffu) {
            active = true;
            for (uint factor = 0; factor < factor_count; factor++) {
                uint table = (factor * branch_width + offset) * (1u << chunk_bits);
                lo[factor] = solinas_add(
                    lo[factor],
                    branches[table + ram_ra_address_chunk(
                        lo_address, factor, factor_count, chunk_bits)]);
            }
        }
        if (hi_address != 0xffffffffu) {
            active = true;
            for (uint factor = 0; factor < factor_count; factor++) {
                uint table = (factor * branch_width + offset) * (1u << chunk_bits);
                hi[factor] = solinas_add(
                    hi[factor],
                    branches[table + ram_ra_address_chunk(
                        hi_address, factor, factor_count, chunk_bits)]);
            }
        }
    }

    for (uint factor = 0; factor < factor_count; factor++) {
        factors[factor].at_one = hi[factor];
        factors[factor].at_infinity = solinas_sub(hi[factor], lo[factor]);
    }
    return active;
}

inline void ram_ra_product(
    thread const InstructionRaLinear* factors,
    uint factor_count,
    thread SolinasFp128* q)
{
    for (uint sample = 0; sample < RAM_RA_SAMPLES; sample++) {
        q[sample] = solinas_zero();
    }

    // Pair linear factors first. A quadratic's finite differences let us
    // advance through 1..factor_count-1 using additions, rather than multiply
    // all factors independently at every sample. The last sample is infinity.
    uint pairs = factor_count / 2u;
    SolinasFp128 values[RAM_RA_MAX_FACTORS / 2u];
    SolinasFp128 steps[RAM_RA_MAX_FACTORS / 2u];
    SolinasFp128 curvature[RAM_RA_MAX_FACTORS / 2u];
    SolinasFp128 leading = solinas_zero();
    leading.limb[0] = 1u;
    for (uint pair = 0; pair < pairs; pair++) {
        InstructionRaQuadratic quadratic = instruction_ra_quadratic(
            factors[2u * pair], factors[2u * pair + 1u]);
        values[pair] = quadratic.at_one;
        steps[pair] = solinas_sub(quadratic.at_two, quadratic.at_one);
        curvature[pair] = solinas_add(quadratic.at_infinity, quadratic.at_infinity);
        leading = solinas_mul_wide(leading, quadratic.at_infinity);
    }
    bool odd = (factor_count & 1u) != 0u;
    SolinasFp128 odd_value = factors[factor_count - 1u].at_one;
    if (odd) {
        leading = solinas_mul_wide(leading, factors[factor_count - 1u].at_infinity);
    }
    for (uint sample = 0; sample + 1u < factor_count; sample++) {
        SolinasFp128 product = values[0];
        for (uint pair = 1; pair < pairs; pair++) {
            product = solinas_mul_wide(product, values[pair]);
        }
        if (odd) {
            product = solinas_mul_wide(product, odd_value);
            odd_value = solinas_add(odd_value, factors[factor_count - 1u].at_infinity);
        }
        q[sample] = product;
        for (uint pair = 0; pair < pairs; pair++) {
            values[pair] = solinas_add(values[pair], steps[pair]);
            steps[pair] = solinas_add(steps[pair], curvature[pair]);
        }
    }
    q[factor_count - 1u] = leading;
}

inline void ram_ra_finish_block(
    thread SolinasFp128* lanes,
    SolinasFp128 e_out,
    device SolinasFp128* partials,
    threadgroup SolinasFp128* shared,
    uint x_out,
    uint e_out_length,
    uint lane_in_simd,
    uint simdgroup,
    uint simdgroups)
{
    for (uint sample = 0; sample < RAM_RA_SAMPLES; sample++) {
        SolinasFp128 sum = solinas_simd_sum_32(lanes[sample]);
        if (lane_in_simd == 0) {
            shared[sample * simdgroups + simdgroup] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        for (uint sample = 0; sample < RAM_RA_SAMPLES; sample++) {
            SolinasFp128 sum = lane_in_simd < simdgroups
                ? shared[sample * simdgroups + lane_in_simd]
                : solinas_zero();
            sum = solinas_simd_sum_32(sum);
            if (lane_in_simd == 0) {
                partials[sample * e_out_length + x_out] = solinas_mul_wide(e_out, sum);
            }
        }
    }
}

kernel void solinas_ram_ra_lazy_message(
    device const uint* addresses [[buffer(0)]],
    device const SolinasFp128* branches [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant RamRaMessageParams& params [[buffer(5)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[RAM_RA_SAMPLES];
    for (uint sample = 0; sample < RAM_RA_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    for (uint x_in = x_in_thread; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        InstructionRaLinear factors[RAM_RA_MAX_FACTORS];
        if (!ram_ra_gather(
                pair,
                ram_ra_branch_width,
                params.factor_count,
                params.chunk_bits,
                addresses,
                branches,
                factors)) {
            continue;
        }
        SolinasFp128 q[RAM_RA_SAMPLES];
        ram_ra_product(factors, params.factor_count, q);
        for (uint sample = 0; sample < params.factor_count; sample++) {
            lanes[sample] = solinas_add(
                lanes[sample],
                solinas_mul_wide(e_in[x_in], q[sample]));
        }
    }

    ram_ra_finish_block(
        lanes,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}

kernel void solinas_ram_ra_double_branches(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    constant SolinasFp128& challenge [[buffer(2)]],
    constant RamRaBranchParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint per_factor = params.branch_width * params.bins;
    uint elements = params.factor_count * per_factor;
    if (gid >= elements) {
        return;
    }
    uint factor = gid / per_factor;
    uint within = gid - factor * per_factor;
    uint destination_base = factor * 2u * per_factor;
    SolinasFp128 value = source[gid];
    SolinasFp128 one = solinas_zero();
    one.limb[0] = 1u;
    destination[destination_base + within] = solinas_mul_wide(
        solinas_sub(one, challenge), value);
    destination[destination_base + per_factor + within] =
        solinas_mul_wide(challenge, value);
}

kernel void solinas_ram_ra_materialize_width_16(
    device const uint* addresses [[buffer(0)]],
    device const SolinasFp128* branches [[buffer(1)]],
    device SolinasFp128* dense [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant RamRaMaterializeParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[RAM_RA_SAMPLES];
    for (uint sample = 0; sample < RAM_RA_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }

    for (uint x_in = x_in_thread; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        InstructionRaLinear factors[RAM_RA_MAX_FACTORS];
        bool active = ram_ra_gather(
            pair, 16u, params.factor_count, params.chunk_bits, addresses, branches, factors);
        for (uint factor = 0; factor < params.factor_count; factor++) {
            uint destination = factor * params.source_elements + 2u * pair;
            dense[destination] = solinas_sub(
                factors[factor].at_one,
                factors[factor].at_infinity);
            dense[destination + 1u] = factors[factor].at_one;
        }
        if (!active) {
            continue;
        }
        SolinasFp128 q[RAM_RA_SAMPLES];
        ram_ra_product(factors, params.factor_count, q);
        for (uint sample = 0; sample < params.factor_count; sample++) {
            lanes[sample] = solinas_add(
                lanes[sample],
                solinas_mul_wide(e_in[x_in], q[sample]));
        }
    }

    ram_ra_finish_block(
        lanes,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}

kernel void solinas_ram_ra_dense_transition(
    device const SolinasFp128* tables [[buffer(0)]],
    device SolinasFp128* bound [[buffer(1)]],
    device const SolinasFp128* e_in [[buffer(2)]],
    device const SolinasFp128* e_out [[buffer(3)]],
    device SolinasFp128* partials [[buffer(4)]],
    constant SolinasFp128& challenge [[buffer(5)]],
    constant RamRaMaterializeParams& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint x_in_thread [[thread_index_in_threadgroup]],
    uint x_out [[threadgroup_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[RAM_RA_SAMPLES];
    for (uint sample = 0; sample < RAM_RA_SAMPLES; sample++) {
        lanes[sample] = solinas_zero();
    }
    uint bound_elements = params.source_elements / 2u;

    for (uint x_in = x_in_thread; x_in < params.e_in_length; x_in += threads) {
        uint pair = x_out * params.e_in_length + x_in;
        InstructionRaLinear factors[RAM_RA_MAX_FACTORS];
        for (uint factor = 0; factor < params.factor_count; factor++) {
            uint source = factor * params.source_elements + 4u * pair;
            SolinasFp128 lo_0 = tables[source];
            SolinasFp128 hi_0 = tables[source + 1u];
            SolinasFp128 lo_1 = tables[source + 2u];
            SolinasFp128 hi_1 = tables[source + 3u];
            SolinasFp128 bound_0 = solinas_add(
                lo_0,
                solinas_mul_wide(challenge, solinas_sub(hi_0, lo_0)));
            SolinasFp128 bound_1 = solinas_add(
                lo_1,
                solinas_mul_wide(challenge, solinas_sub(hi_1, lo_1)));
            uint destination = factor * bound_elements + 2u * pair;
            bound[destination] = bound_0;
            bound[destination + 1u] = bound_1;
            factors[factor].at_one = bound_1;
            factors[factor].at_infinity = solinas_sub(bound_1, bound_0);
        }
        SolinasFp128 q[RAM_RA_SAMPLES];
        ram_ra_product(factors, params.factor_count, q);
        for (uint sample = 0; sample < params.factor_count; sample++) {
            lanes[sample] = solinas_add(
                lanes[sample],
                solinas_mul_wide(e_in[x_in], q[sample]));
        }
    }

    ram_ra_finish_block(
        lanes,
        e_out[x_out],
        partials,
        shared,
        x_out,
        params.e_out_length,
        lane_in_simd,
        simdgroup,
        threads / 32u);
}

kernel void solinas_ram_ra_contract_eq(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    constant uint& elements [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < elements) {
        destination[gid] = solinas_add(source[2u * gid], source[2u * gid + 1u]);
    }
}

kernel void solinas_ram_ra_terminal_bind(
    device const SolinasFp128* tables [[buffer(0)]],
    device SolinasFp128* terminal [[buffer(1)]],
    constant SolinasFp128& challenge [[buffer(2)]],
    constant uint& factor_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= RAM_RA_MAX_FACTORS) {
        return;
    }
    if (gid >= factor_count) {
        terminal[gid] = solinas_zero();
        return;
    }
    SolinasFp128 lo = tables[2u * gid];
    SolinasFp128 hi = tables[2u * gid + 1u];
    terminal[gid] = solinas_add(
        lo,
        solinas_mul_wide(challenge, solinas_sub(hi, lo)));
}

kernel void solinas_ram_ra_reduce(
    device const SolinasFp128* input [[buffer(0)]],
    device SolinasFp128* output [[buffer(1)]],
    constant RamRaReductionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane_in_simd [[thread_index_in_simdgroup]])
{
    for (uint sample = 0; sample < params.columns; sample++) {
        SolinasFp128 value = gid < params.input_count
            ? input[sample * params.input_count + gid]
            : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane_in_simd == 0) {
            output[sample * params.output_count + gid / 32u] = value;
        }
    }
}
