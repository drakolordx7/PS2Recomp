#ifndef PS2_CG_DSP_H
#define PS2_CG_DSP_H
// Guest control flow without the scheduler (PS2X_CODEGEN_DSP=1 with PS2X_CODEGEN=locals, patch 0023).
//
// Included by generated code in addition to ps2_cg.h, which stays untouched (a change there rebuilds every build tree).
//
// 1. Tail transfers. When a generated function leaves through a jump to another function or fragment (a conditional
//    branch out of its range, a `j` to something that is not a function start, an unresolved `jr $reg`, falling off
//    its end into the next fragment), it stores ctx->pc = target | 1 (ps2DspTail) and returns. The low bit marks
//    "this is a tail jump, not a return". Guest code addresses are word aligned, so it cannot be mistaken for one.
//    The nearest enclosing call site (ps2DspCall) sees pc != its return address, recognises the mark and calls the
//    target itself in a loop (ps2DspAfterCallSlow): the callee's frame is gone (it was a tail jump), the caller's frames
//    stay alive, so the eventual `jr $ra` back to the caller is an ordinary return and no frame has to be re-entered
//    through the scheduler. Before this, every such transfer set g_ps2GuestUnwinding, every frame returned to the
//    scheduler, and the scheduler re-entered the target and then, one dispatch per level, every caller's resume label.
//    Only marked pcs are followed. A pc that is neither the return address nor marked is a non-local return (longjmp,
//    thread switch): the old unwinding protocol applies unchanged.
//    The scheduler clears the mark on the pc of a function that returned to it (EeScheduler::run).
//    Budget and yields: each followed tail jump is charged like a dispatch (8 cycles against g_ps2EeBudget); when the
//    budget runs out the checkpoint runs, and a yield leaves ctx->pc = target (unmarked) with the registers already
//    flushed, exactly what the scheduler expects to resume.
// 2. Self-recursive returns. `jal` to an address inside the same function is a goto, so the matching `jr $ra` returned
//    to the scheduler, which re-entered the function at the return label. A `jr $ra` in such a function now jumps to
//    the label directly when the target is one of the function's own recursive return addresses.

#include "ps2_cg.h"

// Counters (the scheduler prints them with PS2X_SCHED_STATS): 0 = tail jumps followed by a call site, 1 = self-recursive
// returns kept local, 2 = calls that returned into the caller after at least one tail jump, 3 = unresolved targets.
inline uint64_t g_ps2DspStat[4] = {0, 0, 0, 0};

#define PS2_TAIL_MARK 1u

// ctx->pc = target, marked as a tail jump. Registers must already be flushed.
PS2CG_FORCEINLINE void ps2DspTail(R5900Context *ctx, uint32_t target)
{
    ctx->pc = target | PS2_TAIL_MARK;
}

// Slow path of a call: the callee returned with ctx->pc != the return address, or the scheduler is unwinding. Follows
// tail jumps until the guest returns to `fall` (returns true: the caller continues), yields or leaves non-locally
// (returns false with g_ps2GuestUnwinding set, as the old ps2CgAfterCallSlow did).
PS2CG_NOINLINE inline bool ps2DspAfterCallSlow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t target,
                                               uint32_t fall)
{
    uint32_t entryPc = target;
    bool followed = false;
    for (;;)
    {
        if (rt->isStopRequested() || ctx->pc == 0u || g_ps2GuestUnwinding)
            return false;
        if ((ctx->pc & PS2_TAIL_MARK) == 0u)
        {
            if (ctx->pc == entryPc)
                ctx->pc = fall;
            if (ctx->pc == fall)
            {
                g_ps2DspStat[2] += followed ? 1u : 0u;
                return true;
            }
            g_ps2GuestUnwinding = true; // non-local return (longjmp, thread switch): the scheduler continues at ctx->pc
            return false;
        }
        const uint32_t pc = ctx->pc & ~PS2_TAIL_MARK;
        ctx->pc = pc;
        if (pc == fall)
        {
            g_ps2DspStat[2] += followed ? 1u : 0u;
            return true;
        }
        if ((g_ps2EeBudget -= 8) < 0) [[unlikely]]
        {
            if (ps2CgBudgetSlow(rt))
                return false; // yield: pc = target, registers flushed, scheduler resumes there
        }
        const uint32_t slot = (pc - g_ps2RecompiledFunctionTableBase) >> 2;
        PS2Runtime::RecompiledFunction fn = nullptr;
        if (((pc & 3u) == 0u) & (slot < g_ps2RecompiledFunctionTableSlotCount))
            fn = g_ps2RecompiledFunctionTable[slot];
        if (fn == nullptr) [[unlikely]]
        {
            ++g_ps2DspStat[3];
            g_ps2GuestUnwinding = true; // the scheduler reports the missing function
            return false;
        }
        ++g_ps2DspStat[0];
        followed = true;
        entryPc = pc;
        fn(rdram, ctx, rt);
    }
}

// The scheduler is the outermost call site: it enters the function it resumes or dispatches directly (after a yield, the
// innermost frame), so a tail jump made there has no ps2DspCall above it. The scheduler follows marked tail jumps the same
// way (EeScheduler::run): each one is charged like a dispatch, a yield or an unwind ends the loop, anything else (missing
// target, stop) leaves ctx->pc unmarked for the scheduler's own dispatch checks.
PS2CG_NOINLINE inline void ps2DspSchedulerFollow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt)
{
    while ((ctx->pc & PS2_TAIL_MARK) != 0u)
    {
        const uint32_t pc = ctx->pc & ~PS2_TAIL_MARK;
        ctx->pc = pc;
        if (g_ps2GuestUnwinding || pc == 0u || rt->isStopRequested())
            return;
        if ((g_ps2EeBudget -= 8) < 0) [[unlikely]]
        {
            if (ps2CgBudgetSlow(rt))
                return;
        }
        const uint32_t slot = (pc - g_ps2RecompiledFunctionTableBase) >> 2;
        PS2Runtime::RecompiledFunction fn = nullptr;
        if (((pc & 3u) == 0u) & (slot < g_ps2RecompiledFunctionTableSlotCount))
            fn = g_ps2RecompiledFunctionTable[slot];
        if (fn == nullptr) [[unlikely]]
            return;
        ++g_ps2DspStat[0];
        fn(rdram, ctx, rt);
    }
}

// JAL / JALR; same contract as ps2CgCall (ps2_cg.h). ctx->pc == target and every modified GPR are already in ctx.
PS2CG_FORCEINLINE bool ps2DspCall(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t target, uint32_t src,
                                  uint32_t fall, bool indirect)
{
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
    return ps2DspAfterCallSlow(rdram, ctx, rt, target, fall);
}

// Unresolved `jr $reg` to another function or fragment (jump tables that were not resolved, function pointers, tail
// calls through a register). Registers are flushed. Leaves through the caller's trampoline when the target exists;
// otherwise the runtime reports it exactly as before. Returns nothing: the generated function returns afterwards.
PS2CG_FORCEINLINE void ps2DspTailIndirect(uint8_t *rdram, R5900Context *ctx, PS2Runtime *rt, uint32_t target,
                                          uint32_t src)
{
    const uint32_t slot = (target - g_ps2RecompiledFunctionTableBase) >> 2;
    if (((target & 3u) == 0u) & (slot < g_ps2RecompiledFunctionTableSlotCount) &&
        g_ps2RecompiledFunctionTable[slot] != nullptr) [[likely]]
    {
        ps2DspTail(ctx, target);
        return;
    }
    (void)rt->dispatchGuestBranch(rdram, ctx, target, src, 0u, PS2Runtime::GuestBranchKind::IndirectJump, "JR");
}

#endif // PS2_CG_DSP_H
