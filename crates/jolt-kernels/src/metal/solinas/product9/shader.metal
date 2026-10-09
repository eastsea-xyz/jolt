// Independent G8 Instruction Read-RAF cycle library, after fp128 and simd_reduce.
#define PRODUCT9_FACTORS 9u
#define PRODUCT9_PHASE_BINS 256u

struct Product9Lookup { ulong2 limbs; };
struct Product9Params {
    uint rows;
    uint source_elements;
    uint branch_width;
    uint e_in_length;
    uint e_out_length;
    uint table_count;
    uint materialize;
    uint reserved;
};
struct Product9ReduceParams { uint input_count; uint output_count; uint2 reserved; };

inline uint product9_byte(Product9Lookup lookup, uint shift) {
    return (uint)((shift < 64u ? lookup.limbs[0] : lookup.limbs[1]) >> (shift & 63u)) & 255u;
}

inline void product9_base(
    uint cycle, device const uchar* claims, device const Product9Lookup* lookups,
    device const uint* inverse, device const SolinasFp128* phases,
    device const SolinasFp128* values, SolinasFp128 raf_interleaved,
    SolinasFp128 raf_identity, constant Product9Params& params,
    device atomic_uint* invalid, thread SolinasFp128* factors)
{
    uint row = inverse[cycle];
    if (row >= params.rows) {
        atomic_fetch_add_explicit(invalid, 1u, memory_order_relaxed);
        for (uint i = 0; i < PRODUCT9_FACTORS; i++) factors[i] = solinas_zero();
        return;
    }
    uchar packed = claims[row];
    // Bit 6 is topology rank, not part of the canonical claim table index.
    uint table = (uint)packed & 0x3fu;
    if (table > params.table_count) {
        atomic_fetch_add_explicit(invalid, 1u, memory_order_relaxed);
        for (uint i = 0; i < PRODUCT9_FACTORS; i++) factors[i] = solinas_zero();
        return;
    }
    factors[0] = solinas_add(table == 0u ? solinas_zero() : values[table - 1u],
        (packed & 0x80u) == 0u ? raf_interleaved : raf_identity);
    Product9Lookup lookup = lookups[row];
    for (uint factor = 0; factor < 8u; factor++) {
        uint phase = 2u * factor;
        uint shift = 120u - 8u * phase;
        factors[factor + 1u] = solinas_mul_wide(
            phases[phase * PRODUCT9_PHASE_BINS + product9_byte(lookup, shift)],
            phases[(phase + 1u) * PRODUCT9_PHASE_BINS + product9_byte(lookup, shift - 8u)]);
    }
}

inline SolinasFp128 product9_product(thread const SolinasFp128* values) {
    SolinasFp128 result = values[0];
    for (uint factor = 1; factor < PRODUCT9_FACTORS; factor++)
        result = solinas_mul_wide(result, values[factor]);
    return result;
}

inline void product9_accumulate(
    thread SolinasFp128* lo, thread SolinasFp128* hi, SolinasFp128 inner_weight,
    thread SolinasFp128* lanes)
{
    SolinasFp128 evals[PRODUCT9_FACTORS];
    SolinasFp128 steps[PRODUCT9_FACTORS];
    // Weight Val alone: eq belongs outside the nine-factor product.
    lo[0] = solinas_mul_wide(lo[0], inner_weight);
    hi[0] = solinas_mul_wide(hi[0], inner_weight);
    for (uint factor = 0; factor < PRODUCT9_FACTORS; factor++) {
        evals[factor] = hi[factor];
        steps[factor] = solinas_sub(hi[factor], lo[factor]);
    }
    for (uint sample = 0; sample < PRODUCT9_FACTORS - 1u; sample++) {
        lanes[sample] = solinas_add(lanes[sample], product9_product(evals));
        for (uint factor = 0; factor < PRODUCT9_FACTORS; factor++)
            evals[factor] = solinas_add(evals[factor], steps[factor]);
    }
    lanes[PRODUCT9_FACTORS - 1u] = solinas_add(
        lanes[PRODUCT9_FACTORS - 1u], product9_product(steps));
}

inline void product9_finish(
    thread SolinasFp128* lanes, SolinasFp128 outer_weight,
    device SolinasFp128* partials, threadgroup SolinasFp128* shared,
    uint outer, uint outer_count, uint lane, uint simdgroup, uint simdgroups)
{
    for (uint sample = 0; sample < PRODUCT9_FACTORS; sample++) {
        SolinasFp128 value = solinas_simd_sum_32(lanes[sample]);
        if (lane == 0) shared[sample * simdgroups + simdgroup] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        for (uint sample = 0; sample < PRODUCT9_FACTORS; sample++) {
            SolinasFp128 value = lane < simdgroups
                ? shared[sample * simdgroups + lane] : solinas_zero();
            value = solinas_simd_sum_32(value);
            if (lane == 0) partials[sample * outer_count + outer] =
                solinas_mul_wide(outer_weight, value);
        }
    }
}

kernel void solinas_product9_lazy_cycle(
    device const uchar* claims [[buffer(0)]],
    device const Product9Lookup* lookups [[buffer(1)]],
    device const uint* inverse [[buffer(2)]],
    device const SolinasFp128* phases [[buffer(3)]],
    device const SolinasFp128* values [[buffer(4)]],
    device const SolinasFp128* weights [[buffer(5)]],
    constant SolinasFp128& raf_interleaved [[buffer(6)]],
    constant SolinasFp128& raf_identity [[buffer(7)]],
    device const SolinasFp128* e_in [[buffer(8)]],
    device const SolinasFp128* e_out [[buffer(9)]],
    device SolinasFp128* partials [[buffer(10)]],
    device SolinasFp128* dense [[buffer(11)]],
    constant Product9Params& params [[buffer(12)]],
    device atomic_uint* invalid [[buffer(13)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint inner_thread [[thread_index_in_threadgroup]], uint outer [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[PRODUCT9_FACTORS];
    for (uint sample = 0; sample < PRODUCT9_FACTORS; sample++) lanes[sample] = solinas_zero();
    for (uint inner = inner_thread; inner < params.e_in_length; inner += threads) {
        uint pair = outer * params.e_in_length + inner;
        SolinasFp128 lo[PRODUCT9_FACTORS], hi[PRODUCT9_FACTORS];
        for (uint factor = 0; factor < PRODUCT9_FACTORS; factor++)
            lo[factor] = hi[factor] = solinas_zero();
        for (uint offset = 0; offset < params.branch_width; offset++) {
            SolinasFp128 base_lo[PRODUCT9_FACTORS], base_hi[PRODUCT9_FACTORS];
            product9_base(2u * pair * params.branch_width + offset,
                claims, lookups, inverse, phases, values, raf_interleaved, raf_identity,
                params, invalid, base_lo);
            product9_base((2u * pair + 1u) * params.branch_width + offset,
                claims, lookups, inverse, phases, values, raf_interleaved, raf_identity,
                params, invalid, base_hi);
            for (uint factor = 0; factor < PRODUCT9_FACTORS; factor++) {
                lo[factor] = solinas_add(lo[factor], solinas_mul_wide(weights[offset], base_lo[factor]));
                hi[factor] = solinas_add(hi[factor], solinas_mul_wide(weights[offset], base_hi[factor]));
            }
        }
        if (params.materialize != 0u) {
            for (uint factor = 0; factor < PRODUCT9_FACTORS; factor++) {
                dense[factor * params.source_elements + 2u * pair] = lo[factor];
                dense[factor * params.source_elements + 2u * pair + 1u] = hi[factor];
            }
        }
        product9_accumulate(lo, hi, e_in[inner], lanes);
    }
    product9_finish(lanes, e_out[outer], partials, shared,
        outer, params.e_out_length, lane, simdgroup, threads / 32u);
}

kernel void solinas_product9_dense_bind(
    device const SolinasFp128* source [[buffer(0)]],
    device SolinasFp128* destination [[buffer(1)]],
    constant SolinasFp128& challenge [[buffer(2)]],
    device const SolinasFp128* e_in [[buffer(3)]],
    device const SolinasFp128* e_out [[buffer(4)]],
    device SolinasFp128* partials [[buffer(5)]],
    constant Product9Params& params [[buffer(6)]],
    threadgroup SolinasFp128* shared [[threadgroup(0)]],
    uint inner_thread [[thread_index_in_threadgroup]], uint outer [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    SolinasFp128 lanes[PRODUCT9_FACTORS];
    for (uint sample = 0; sample < PRODUCT9_FACTORS; sample++) lanes[sample] = solinas_zero();
    uint bound_elements = params.source_elements / 2u;
    for (uint inner = inner_thread; inner < params.e_in_length; inner += threads) {
        uint pair = outer * params.e_in_length + inner;
        SolinasFp128 lo[PRODUCT9_FACTORS], hi[PRODUCT9_FACTORS];
        for (uint factor = 0; factor < PRODUCT9_FACTORS; factor++) {
            uint index = factor * params.source_elements + 4u * pair;
            lo[factor] = solinas_add(source[index], solinas_mul_wide(
                challenge, solinas_sub(source[index + 1u], source[index])));
            hi[factor] = solinas_add(source[index + 2u], solinas_mul_wide(
                challenge, solinas_sub(source[index + 3u], source[index + 2u])));
            destination[factor * bound_elements + 2u * pair] = lo[factor];
            destination[factor * bound_elements + 2u * pair + 1u] = hi[factor];
        }
        product9_accumulate(lo, hi, e_in[inner], lanes);
    }
    product9_finish(lanes, e_out[outer], partials, shared,
        outer, params.e_out_length, lane, simdgroup, threads / 32u);
}

kernel void solinas_product9_reduce(
    device const SolinasFp128* input [[buffer(0)]], device SolinasFp128* output [[buffer(1)]],
    constant Product9ReduceParams& params [[buffer(2)]], uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint sample = 0; sample < PRODUCT9_FACTORS; sample++) {
        SolinasFp128 value = gid < params.input_count
            ? input[sample * params.input_count + gid] : solinas_zero();
        value = solinas_simd_sum_32(value);
        if (lane == 0) output[sample * params.output_count + gid / 32u] = value;
    }
}
