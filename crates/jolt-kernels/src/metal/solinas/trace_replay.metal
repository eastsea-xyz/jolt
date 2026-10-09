#include <metal_stdlib>
using namespace metal;

// Standalone diagnostic library. No proof kernels or field arithmetic required.
struct ReplayInstruction {
    ulong imm;
    uint op, rs1, rs2, rd;
};
struct ReplayChunk {
    uint start, count;
    ulong known_lo, known_hi;
    ulong initial[128];
};
struct ReplayValues { ulong rs1, rs2, rd_pre, rd_write; };
struct ReplayParams { uint chunks, instructions; };

inline ulong replay_sext32(ulong value) {
    return (value & 0xfffffffful) | ((value & 0x80000000ul) ? 0xffffffff00000000ul : 0ul);
}

inline uint replay_ctz64(ulong value) {
    uint low = uint(value);
    return low != 0u ? ctz(low) : 32u + ctz(uint(value >> 32));
}

kernel void jolt_trace_replay(
    device const ReplayInstruction* instructions [[buffer(0)]],
    device const ReplayChunk* chunks [[buffer(1)]],
    device ReplayValues* output [[buffer(2)]],
    device ulong* final_registers [[buffer(3)]],
    device uint* status [[buffer(4)]],
    constant ReplayParams& params [[buffer(5)]],
    uint chunk_id [[thread_position_in_grid]]) {
    if (chunk_id >= params.chunks) return;
    status[chunk_id] = 2; // fail closed until every row and final state is written
    const device ReplayChunk& chunk = chunks[chunk_id];
    if (chunk.count < 8 || chunk.count > 64 || chunk.start > params.instructions ||
        chunk.count > params.instructions - chunk.start) return;
    ulong registers[128];
    bool known[128];
    for (uint r = 0; r < 128; ++r) {
        registers[r] = chunk.initial[r];
        known[r] = (((r < 64 ? chunk.known_lo : chunk.known_hi) >> (r & 63)) & 1ul) != 0;
        if (!known[r] && registers[r] != 0) return;
    }
    if (!known[0] || registers[0] != 0) return;
    for (uint i = 0; i < chunk.count; ++i) {
        ReplayInstruction ins = instructions[chunk.start + i];
        uint op = ins.op & 255u;
        bool immediate = (ins.op & 256u) != 0;
        if ((ins.op & ~511u) || ins.rd >= 128 || !known[ins.rd]) return;
        bool has_rs1 = ins.rs1 != 0xffffffffu;
        bool has_rs2 = ins.rs2 != 0xffffffffu;
        if ((has_rs1 && (ins.rs1 >= 128 || !known[ins.rs1])) ||
            (has_rs2 && (ins.rs2 >= 128 || !known[ins.rs2]))) return;
        // Match the host's operand contract without using any oracle values.
        bool pow_immediate = immediate && (op == 12 || op == 13);
        bool unary = op >= 10 && op <= 13;
        if (has_rs1 == pow_immediate || has_rs2 != (!immediate && !unary)) return;
        ulong x = has_rs1 ? registers[ins.rs1] : 0ul;
        ulong rs2 = has_rs2 ? registers[ins.rs2] : 0ul;
        ulong y = immediate ? ins.imm : rs2;
        ulong value;
        switch (op) {
            case 1: value = x + y; break;
            case 2: value = x - y; break;
            case 3: value = x ^ y; break;
            case 4: value = x & y; break;
            case 5: value = x | y; break;
            case 6: value = x * y; break;
            case 7: value = replay_sext32(x + y); break;
            case 8: value = replay_sext32(x - y); break;
            case 9: value = replay_sext32(x * y); break;
            case 10: value = x & 0xfffffffful; break;
            case 11: value = replay_sext32(x); break;
            case 12: value = 1ul << ((pow_immediate ? y : x) & 63ul); break;
            case 13: value = 1ul << ((pow_immediate ? y : x) & 31ul); break;
            // Final JoltIR right shifts use trailing-zero bitmask encoding.
            case 14: value = x >> (replay_ctz64(y) & 63u); break;
            case 15: value = as_type<ulong>(as_type<long>(x) >> (replay_ctz64(y) & 63u)); break;
            case 16: {
                uint shift = replay_ctz64(y);
                if (shift >= 32) return;
                value = replay_sext32(ulong(uint(x) >> shift));
                break;
            }
            case 17: {
                uint shift = replay_ctz64(y);
                if (shift >= 32) return;
                value = replay_sext32(ulong(as_type<uint>(as_type<int>(uint(x)) >> shift)));
                break;
            }
            case 18: value = x & ~y; break;
            case 19: value = as_type<long>(x) < as_type<long>(y) ? 1ul : 0ul; break;
            case 20: value = x < y ? 1ul : 0ul; break;
            default: return;
        }
        ulong pre = registers[ins.rd];
        // x0 remains zero, including its captured post-write value.
        ulong post = ins.rd == 0 ? 0ul : value;
        output[chunk.start + i] = ReplayValues{x, rs2, pre, post};
        registers[ins.rd] = post;
    }
    for (uint r = 0; r < 128; ++r) final_registers[chunk_id * 128 + r] = registers[r];
    status[chunk_id] = 1;
}
