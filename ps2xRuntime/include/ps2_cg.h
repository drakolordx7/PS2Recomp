#ifndef PS2_CG_H
#define PS2_CG_H
// Support header for the "register locals" code generator mode (PS2X_CODEGEN=locals, ps2xRecomp).
//
// Generated functions in this mode keep the low 64 bits of every guest GPR they touch in C++ locals (gprN; gprhN
// holds the upper 64 bits for registers used by 128-bit ops). The locals are loaded from ctx->r[] at function
// entry (and resume), written back (PS2_FLUSH) before anything that can observe ctx (calls, syscalls, stubs,
// returns, yields), and reloaded (PS2_RELOAD) after calls and stubs.
//
// Everything is prefixed L_/ps2Cg so classic-mode and locals-mode functions can share a translation unit.
// This header is header-only on purpose: no runtime library changes.

#include "ps2_runtime_macros.h"
#include "ps2_runtime.h"

#include <cstdint>
#include <cstdlib>

#if defined(_MSC_VER)
#define PS2CG_NOINLINE __declspec(noinline)
#define PS2CG_FORCEINLINE __forceinline
#else
#define PS2CG_NOINLINE __attribute__((noinline))
#define PS2CG_FORCEINLINE inline __attribute__((always_inline))
#endif

extern bool g_ps2GuestUnwinding;

// ---- GPR locals --------------------------------------------------------------------------------------------------
// Reg 0 reads are the constants gpr0/gprh0 declared by the function prologue; writes to reg 0 are rewritten to
// PS2_DISCARD by the recompiler.
#define L_GPR_U32(n) (static_cast<uint32_t>(gpr##n))
#define L_GPR_S32(n) (static_cast<int32_t>(gpr##n))
#define L_GPR_U64(n) (static_cast<uint64_t>(gpr##n))
#define L_GPR_S64(n) (static_cast<int64_t>(gpr##n))
#define L_GPR_VEC(n) _mm_set_epi64x(static_cast<long long>(gprh##n), static_cast<long long>(gpr##n))

#define L_SET_GPR_U32(n, v) do { gpr##n = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(v))); } while (0)
#define L_SET_GPR_S32(n, v) L_SET_GPR_U32(n, v)
#define L_SET_GPR_ZE32(n, v) do { gpr##n = static_cast<uint64_t>(static_cast<uint32_t>(v)); } while (0)
#define L_SET_GPR_U64(n, v) do { gpr##n = static_cast<uint64_t>(static_cast<int64_t>(v)); } while (0)
#define L_SET_GPR_S64(n, v) L_SET_GPR_U64(n, v)
#define L_SET_GPR_VEC(n, v)                                                    \
    do                                                                         \
    {                                                                          \
        const __m128i _lv = (v);                                               \
        gpr##n = static_cast<uint64_t>(_mm_cvtsi128_si64(_lv));                \
        gprh##n = static_cast<uint64_t>(_mm_extract_epi64(_lv, 1));            \
    } while (0)
#define PS2_DISCARD(v) ((void)(v))

// ctx <-> local transfers. ctx->r[n] is a __m128i; the halves are accessed as two uint64_t.
#define PS2_CG_LO(ctx, n) (reinterpret_cast<uint64_t *>(&(ctx)->r[n])[0])
#define PS2_CG_HI(ctx, n) (reinterpret_cast<uint64_t *>(&(ctx)->r[n])[1])

// ---- guest memory ------------------------------------------------------------------------------------------------
// RAM window [0, 32 MB) is never special, so it is one compare and one typed access. Everything else (mirrors,
// kseg0/1, MMIO, scratchpad, VU memory) takes the old path in an out-of-line function.
PS2CG_NOINLINE inline uint8_t ps2CgRd8Slow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a)
{
    return PS2Runtime::isSpecialAddress(a) ? rt->Load8(rdram, ctx, a) : Ps2FastRead8(rdram, a);
}
PS2CG_NOINLINE inline uint16_t ps2CgRd16Slow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a)
{
    return PS2Runtime::isSpecialAddress(a) ? rt->Load16(rdram, ctx, a) : Ps2FastRead16(rdram, a);
}
PS2CG_NOINLINE inline uint32_t ps2CgRd32Slow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a)
{
    return PS2Runtime::isSpecialAddress(a) ? rt->Load32(rdram, ctx, a) : Ps2FastRead32(rdram, a);
}
PS2CG_NOINLINE inline uint64_t ps2CgRd64Slow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a)
{
    return PS2Runtime::isSpecialAddress(a) ? rt->Load64(rdram, ctx, a) : Ps2FastRead64(rdram, a);
}
PS2CG_NOINLINE inline __m128i ps2CgRd128Slow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a)
{
    return PS2Runtime::isSpecialAddress(a) ? rt->Load128(rdram, ctx, a) : Ps2FastRead128(rdram, a);
}
PS2CG_NOINLINE inline void ps2CgWr8Slow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a, uint8_t v)
{
    if (PS2Runtime::isSpecialAddress(a)) rt->Store8(rdram, ctx, a, v); else Ps2FastWrite8(rdram, a, v);
}
PS2CG_NOINLINE inline void ps2CgWr16Slow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a, uint16_t v)
{
    if (PS2Runtime::isSpecialAddress(a)) rt->Store16(rdram, ctx, a, v); else Ps2FastWrite16(rdram, a, v);
}
PS2CG_NOINLINE inline void ps2CgWr32Slow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a, uint32_t v)
{
    if (PS2Runtime::isSpecialAddress(a)) rt->Store32(rdram, ctx, a, v); else Ps2FastWrite32(rdram, a, v);
}
PS2CG_NOINLINE inline void ps2CgWr64Slow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a, uint64_t v)
{
    if (PS2Runtime::isSpecialAddress(a)) rt->Store64(rdram, ctx, a, v); else Ps2FastWrite64(rdram, a, v);
}
PS2CG_NOINLINE inline void ps2CgWr128Slow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a, __m128i v)
{
    if (PS2Runtime::isSpecialAddress(a)) rt->Store128(rdram, ctx, a, v); else Ps2FastWrite128(rdram, a, v);
}

#define PS2CG_RD(NAME, TYPE, BYTES)                                                                                  \
    PS2CG_FORCEINLINE TYPE ps2Cg##NAME(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a)                 \
    {                                                                                                                \
        if (a <= PS2_RAM_SIZE - (BYTES)) [[likely]]                                                                  \
        {                                                                                                            \
            return *reinterpret_cast<const TYPE *>(rdram + a);                                                       \
        }                                                                                                            \
        return ps2Cg##NAME##Slow(rdram, ctx, rt, a);                                                                 \
    }
PS2CG_RD(Rd8, uint8_t, 1u)
PS2CG_RD(Rd16, uint16_t, 2u)
PS2CG_RD(Rd32, uint32_t, 4u)
PS2CG_RD(Rd64, uint64_t, 8u)
#undef PS2CG_RD

PS2CG_FORCEINLINE __m128i ps2CgRd128(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a)
{
    if (a <= PS2_RAM_SIZE - 16u) [[likely]]
    {
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(rdram + a));
    }
    return ps2CgRd128Slow(rdram, ctx, rt, a);
}

#define PS2CG_WR(NAME, TYPE, BYTES)                                                                                  \
    PS2CG_FORCEINLINE void ps2Cg##NAME(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a, TYPE v)         \
    {                                                                                                                \
        if (a <= PS2_RAM_SIZE - (BYTES)) [[likely]]                                                                  \
        {                                                                                                            \
            *reinterpret_cast<TYPE *>(rdram + a) = v;                                                                \
            return;                                                                                                  \
        }                                                                                                            \
        ps2Cg##NAME##Slow(rdram, ctx, rt, a, v);                                                                     \
    }
PS2CG_WR(Wr8, uint8_t, 1u)
PS2CG_WR(Wr16, uint16_t, 2u)
PS2CG_WR(Wr32, uint32_t, 4u)
PS2CG_WR(Wr64, uint64_t, 8u)
#undef PS2CG_WR

PS2CG_FORCEINLINE void ps2CgWr128(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t a, __m128i v)
{
    if (a <= PS2_RAM_SIZE - 16u) [[likely]]
    {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(rdram + a), v);
        return;
    }
    ps2CgWr128Slow(rdram, ctx, rt, a, v);
}

#define L_READ8(addr) ps2CgRd8(rdram, ctx, runtime, static_cast<uint32_t>(addr))
#define L_READ16(addr) ps2CgRd16(rdram, ctx, runtime, static_cast<uint32_t>(addr))
#define L_READ32(addr) ps2CgRd32(rdram, ctx, runtime, static_cast<uint32_t>(addr))
#define L_READ64(addr) ps2CgRd64(rdram, ctx, runtime, static_cast<uint32_t>(addr))
#define L_READ128(addr) ps2CgRd128(rdram, ctx, runtime, static_cast<uint32_t>(addr))
#define L_WRITE8(addr, val) ps2CgWr8(rdram, ctx, runtime, static_cast<uint32_t>(addr), static_cast<uint8_t>(val))
#define L_WRITE16(addr, val) ps2CgWr16(rdram, ctx, runtime, static_cast<uint32_t>(addr), static_cast<uint16_t>(val))
#define L_WRITE32(addr, val) ps2CgWr32(rdram, ctx, runtime, static_cast<uint32_t>(addr), static_cast<uint32_t>(val))
#define L_WRITE64(addr, val) ps2CgWr64(rdram, ctx, runtime, static_cast<uint32_t>(addr), static_cast<uint64_t>(val))
#define L_WRITE128(addr, val) ps2CgWr128(rdram, ctx, runtime, static_cast<uint32_t>(addr), (val))

// ---- EE checkpoint budget and calls ------------------------------------------------------------------------------
// The old code charged EeScheduler::checkpointDue on every guest call (8 cycles) and loop back edge (32 cycles).
// Locals mode subtracts the same charge from a plain counter and only calls the scheduler when it runs out; the
// accumulated charge is then passed on in one go. Events posted to the scheduler are noticed at that point, i.e.
// within a few microseconds of guest time. The refill (256 cycles = 32 calls) is a compromise: 1024 was as fast but
// made the movie player hang (the IOP file-read completion never arrives when the IOP is advanced in >=1024-cycle
// slices with KZ_IPU on; 64..512 boot fine), so the IOP is still advanced in small slices. PS2X_EE_BUDGET overrides the refill (cycles); PS2X_STRESS_YIELD forces
// a scheduler checkpoint on every check so the yield/resume stress test still exercises every call and back edge.
inline int32_t g_ps2EeBudget = 256;

inline int32_t ps2CgBudgetRefill()
{
    static const int32_t refill = []() -> int32_t
    {
        const char *stress = std::getenv("PS2X_STRESS_YIELD");
        if (stress && std::strtoul(stress, nullptr, 10) != 0ul)
            return 1;
        const char *v = std::getenv("PS2X_EE_BUDGET");
        if (v && std::strtol(v, nullptr, 10) > 0)
            return static_cast<int32_t>(std::strtol(v, nullptr, 10));
        return 256;
    }();
    return refill;
}

// Called when the budget went negative. Returns true when the guest must yield (g_ps2GuestUnwinding is set).
PS2CG_NOINLINE inline bool ps2CgBudgetSlow(PS2Runtime *rt)
{
#ifdef PS2_CG_CLASSIC_CHECKPOINTS
    return rt->eeCheckpointDue(); // measurement variant: charge the scheduler on every back edge like the old code
#endif
    const int32_t refill = ps2CgBudgetRefill();
    const int32_t consumed = refill - g_ps2EeBudget;
    g_ps2EeBudget = refill;
    return rt->eeCheckpointDue(static_cast<uint32_t>(consumed > 0 ? consumed : 1));
}

// Old dispatchGuestBranch epilogue (callee returned): decides whether execution continues in the caller.
PS2CG_NOINLINE inline bool ps2CgAfterCallSlow(PS2Runtime *rt, R5900Context *ctx, uint32_t target, uint32_t fall)
{
    if (rt->isStopRequested() || ctx->pc == 0u || g_ps2GuestUnwinding)
        return false;
    if (ctx->pc == target)
        ctx->pc = fall;
    if (ctx->pc != fall)
    {
        g_ps2GuestUnwinding = true; // non-local return (longjmp, tail jump): the scheduler continues at ctx->pc
        return false;
    }
    return true;
}

PS2CG_NOINLINE inline bool ps2CgCallSlow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t target,
                                         uint32_t src, uint32_t fall, bool indirect)
{
    return rt->dispatchGuestBranch(rdram, ctx, target, src, fall,
                                   indirect ? PS2Runtime::GuestBranchKind::IndirectCall
                                            : PS2Runtime::GuestBranchKind::DirectCall,
                                   indirect ? "JALR" : "JAL");
}

// JAL / JALR. ctx->pc == target and every modified GPR are already in ctx. Returns false when the caller must
// return at once (yield, unwinding callee, missing function handled by dispatchGuestBranch).
PS2CG_FORCEINLINE bool ps2CgCall(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t target, uint32_t src,
                                 uint32_t fall, bool indirect)
{
#ifdef PS2_CG_CLASSIC_CHECKPOINTS
    return ps2CgCallSlow(rdram, ctx, rt, target, src, fall, indirect); // measurement variant: old dispatch path
#else
    if ((g_ps2EeBudget -= 8) < 0) [[unlikely]]
    {
        if (ps2CgBudgetSlow(rt))
            return false;
    }
    const uint32_t slot = (target - g_ps2RecompiledFunctionTableBase) >> 2;
    if (((target & 3u) != 0u) | (slot >= g_ps2RecompiledFunctionTableSlotCount)) [[unlikely]]
        return ps2CgCallSlow(rdram, ctx, rt, target, src, fall, indirect);
    const PS2Runtime::RecompiledFunction fn = g_ps2RecompiledFunctionTable[slot];
    if (fn == nullptr) [[unlikely]]
        return ps2CgCallSlow(rdram, ctx, rt, target, src, fall, indirect);
    fn(rdram, ctx, rt);
    if (!g_ps2GuestUnwinding && ctx->pc == fall) [[likely]]
        return true;
    return ps2CgAfterCallSlow(rt, ctx, target, fall);
#endif
}

// Loop back edge: charge the budget, and when it runs out flush and let the scheduler decide. Evaluates to true
// when the function has to return (ctx->pc must already be the loop target).
#ifdef PS2_CG_CLASSIC_CHECKPOINTS
#define PS2_BACKEDGE_DUE() (true)
#else
#define PS2_BACKEDGE_DUE() ((g_ps2EeBudget -= 32) < 0)
#endif

#endif // PS2_CG_H
