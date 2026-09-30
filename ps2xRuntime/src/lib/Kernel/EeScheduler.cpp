#include "runtime/ps2_host_gs.h"
#include "runtime/ps2_timeline.h"
#include <atomic>
#include <cstdlib>
#include "runtime/ee_scheduler.h"

#include "ps2_log.h"
#include "../ps2_iop_post.h"
#include "../ps2_vif1_worker.h"
#include "ps2_runtime_macros.h"
#include "ps2_cg_dsp.h" // g_ps2DspStat, PS2_TAIL_MARK (includes ps2_cg.h)
#include "ps2_cg.h" // g_ps2EeBudget: the guest code's checkpoint budget (the dispatch loop charges through it as well)

#include <algorithm>
#include <string_view>
#include <intrin.h>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <cstdio>

extern bool g_ps2GuestUnwinding; // ps2_runtime.cpp

namespace
{
    // Number of events in m_events, readable without the lock: processPendingEvents runs on nearly every scheduler
    // iteration and used to lock twice, allocate a deque and read the host clock even with nothing to do.
    std::atomic<uint32_t> g_queuedEvents{0};
}

namespace
{
    // Read once at startup. Function-local statics with dynamic initialisers cost a thread-safe-init check (a TLS load)
    // on every call under MSVC, and these functions run at every checkpoint.
    const bool g_wallClockPacing = []() {
        const char *v = std::getenv("PS2X_WALLCLOCK");
        return !(v && v[0] == '0');
    }();
    // PS2X_DISPATCH_BUDGET=0: charge every scheduler dispatch with a checkpoint of its own (the old behaviour) instead of
    // through the guest budget.
    const bool g_dispatchViaBudget = []() {
        const char *v = std::getenv("PS2X_DISPATCH_BUDGET");
        return !(v && v[0] == '0');
    }();
    // EE cycle clock scale (patch 0025, docs/findings.md "Frame pipeline timeline"). Multiplies the cycles the guest code's
    // checkpoints charge to the EE cycle clock, the way PCSX2's "EE cycle rate" does. The vblank event needs both its cycle
    // deadline and its host deadline. The charged estimate runs ~2x faster than wall time on this host, so a frame whose
    // estimated work is more than one vblank period's cycles (2.46 M at 120 Hz) spans two vblanks although the host executed it
    // in less than one, and the EE thread sleeps in processDueDeadlines for the rest of the second vblank.
    //   PS2X_EE_CYCLE_SCALE unset or "auto" (default): 60 Hz / vblank rate, at most 1: the game gets the same estimated
    //     cycles per vblank as at its native 60 Hz, which is what makes the guest see the same timing per vblank at any rate
    //     (timers, IOP time, spin-loop iterations per vblank), the same way kz_timing scales the frame timer.
    //   PS2X_EE_CYCLE_SCALE=1: the previous behaviour. <f>: a fixed factor. 0: the clock follows wall time only.
    // Q16 fixed point; 65536 = off. Set from ps2SetVblankPeriodMicros() (auto) or the environment, before the EE thread runs.
    std::atomic<uint32_t> g_eeCycleScaleQ16{65536u};
    bool g_eeCycleScaleAuto = true;
    double g_eeCycleScaleFixed = 1.0;
    const bool g_eeCycleScaleInit = []() {
        const char *v = std::getenv("PS2X_EE_CYCLE_SCALE");
        if (v && *v && std::string_view(v) != "auto")
        {
            g_eeCycleScaleAuto = false;
            g_eeCycleScaleFixed = std::atof(v);
            g_eeCycleScaleQ16.store(g_eeCycleScaleFixed <= 0.0 ? 0u : static_cast<uint32_t>(std::min(g_eeCycleScaleFixed, 16.0) * 65536.0),
                                    std::memory_order_relaxed);
        }
        return true;
    }();
    const uint32_t g_stressYield = []() {
        const char *v = std::getenv("PS2X_STRESS_YIELD");
        return v ? static_cast<uint32_t>(std::strtoul(v, nullptr, 10)) : 0u;
    }();
}

#define SCHED_STAT(expr) do { if (g_schedStats.on) { expr; } } while (0)

namespace
{
    // PS2X_SCHED_STATS=1: every 10 s a few lines with the scheduler's per-second rates (yields by reason, dispatches, events),
    // the rdtsc split of run() time between guest functions and the scheduler around them, and host time slept in vblank
    // pacing. =2 adds per-pc histograms of what the scheduler dispatched (skews the time split: a hash map update per
    // dispatch). Compile with -DPS2X_SCHED_STATS_HOT for the per-checkpoint counters too (checkpoint count, estimated vs
    // wall-clock-floor cycles, cost of the timer and IOP parts of accountCycles). All counters are EE-thread only.
    struct SchedStats
    {
        bool on = false;
        bool hist = false; // PS2X_SCHED_STATS=2: also per-pc histograms (a hash map update per dispatch: skews the time split)
        // False for PS2X_SCHED_STATS=3: only the vblank pacing sleep is accumulated and reported every 10 s (nothing per loop
        // iteration or dispatch; =1 reads the host clock and does rdtsc/counter work on every dispatch, i.e. ~40 ns x 1 M/s in a
        // build that dispatches 1 M times per second, which inflates exactly the thing an A/B of dispatch counts measures).
        bool perDispatch = false;
        uint64_t checkpoints = 0, checkpointCycles = 0;
        uint64_t yieldPending = 0, yieldDeadline = 0, yieldSlice = 0, yieldStress = 0;
        std::unordered_map<uint32_t, uint64_t> pcHist, exitHist;
        // Why each dispatch happened, judged from how the previous guest entry ended (kinds: 0 = scheduler-originated
        // (new thread, invocation, event), 1 = yield (a checkpoint gave control back), 2 = return-resume (the guest
        // returned through $ra to a call's return address: a caller resumed after an unwind), 3 = jump (any other pc:
        // tail jump, fall into the next fragment, indirect jump). pairHist (=2 only): (previous entry pc << 32 | this pc).
        uint64_t reasonCount[4] = {0, 0, 0, 0};
        std::unordered_map<uint64_t, uint64_t> pairHist[4];
        uint32_t lastEntryPc = 0, lastExitPc = 0;
        int lastThread = -1;
        bool lastValid = false;
        uint64_t yieldsAtEntry = 0;
        // rdtsc time split: inside guest functions vs the scheduler loop around them, for dispatches in the guest's
        // vsync/flag wait loop (FUN_0014fd90 and its entry fragments, 0x14fd90..0x15008c) and everywhere else.
        // Guest clock: cycles charged by the checkpoint estimate vs cycles added by the wall-clock floor, and host time
        // slept waiting for a vblank's host deadline (processDueDeadlines) although its guest cycle had been reached.
        uint64_t estCycles = 0, clampCycles = 0, pacingSleepTsc = 0, pacingSleeps = 0;
        uint64_t acctCalls = 0, acctTsc = 0, timersTsc = 0, iopTsc = 0; // PS2X_SCHED_STATS_HOT: accountCycles and its two callees
        uint64_t tscInGuestSpin = 0, tscInGuestOther = 0, tscLoopSpin = 0, tscLoopOther = 0, tscPrevExit = 0;
        uint64_t transfers = 0;
        uint64_t loopIters = 0, dispatches = 0, pendingCalls = 0, events = 0, deadlineEvents = 0, timerIrqs = 0, idleWaits = 0;
        std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    };
    SchedStats g_schedStats;
    const bool g_schedStatsInit = []() {
        const char *v = std::getenv("PS2X_SCHED_STATS");
        g_schedStats.on = v && *v && *v != '0';
        g_schedStats.hist = v && *v == '2';
        g_schedStats.perDispatch = g_schedStats.on && *v != '3';
        return true;
    }();
    void schedStatsReport()
    {
        SchedStats &st = g_schedStats;
        const auto now = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(now - st.last).count();
        if (secs < 10.0)
            return;
        const double k = 1.0 / secs;
        std::fprintf(stderr,
                     "[sched-stats] %.1fs per second: checkpoints=%.0f (avg %.1f cycles) yields: pending=%.0f deadline=%.0f slice=%.0f stress=%.0f "
                     "| loop=%.0f dispatch=%.0f pendingCalls=%.0f events=%.0f deadlineEvents=%.0f timerIrq=%.0f idleWaits=%.0f\n",
                     secs, st.checkpoints * k, st.checkpoints ? double(st.checkpointCycles) / double(st.checkpoints) : 0.0,
                     st.yieldPending * k, st.yieldDeadline * k, st.yieldSlice * k, st.yieldStress * k, st.loopIters * k,
                     st.dispatches * k, st.pendingCalls * k, st.events * k, st.deadlineEvents * k, st.timerIrqs * k, st.idleWaits * k);
        auto top = [](const std::unordered_map<uint32_t, uint64_t> &h, const char *label, double k) {
            std::vector<std::pair<uint64_t, uint32_t>> v;
            for (const auto &e : h)
                v.emplace_back(e.second, e.first);
            std::sort(v.rbegin(), v.rend());
            std::fprintf(stderr, "[sched-stats]   %s:", label);
            for (size_t i = 0; i < v.size() && i < 12; ++i)
                std::fprintf(stderr, " %06x:%.0f", v[i].second, v[i].first * k);
            std::fprintf(stderr, "\n");
        };
        std::fprintf(stderr, "[sched-stats]   guest control flow kept off the scheduler per second (ps2_cg_dsp.h): tail jumps followed=%.0f "
                             "recursive returns local=%.0f calls returned after a followed tail=%.0f missing tail targets=%.0f\n",
                     g_ps2DspStat[0] * k, g_ps2DspStat[1] * k, g_ps2DspStat[2] * k, g_ps2DspStat[3] * k);
        std::memset(g_ps2DspStat, 0, sizeof(g_ps2DspStat));
        std::fprintf(stderr, "[sched-stats]   dispatch reasons per second: scheduler=%.0f yield=%.0f return-resume=%.0f jump=%.0f\n",
                     st.reasonCount[0] * k, st.reasonCount[1] * k, st.reasonCount[2] * k, st.reasonCount[3] * k);
        if (st.hist)
        {
            static const char *const kindName[4] = {"scheduler", "yield", "return-resume", "jump"};
            for (int kind = 1; kind < 4; ++kind)
            {
                std::vector<std::pair<uint64_t, uint64_t>> v;
                for (const auto &e : st.pairHist[kind])
                    v.emplace_back(e.second, e.first);
                std::sort(v.rbegin(), v.rend());
                std::fprintf(stderr, "[sched-stats]   %s (prev entry->pc):", kindName[kind]);
                for (size_t i = 0; i < v.size() && i < 14; ++i)
                    std::fprintf(stderr, " %06x->%06x:%.0f", static_cast<uint32_t>(v[i].second >> 32), static_cast<uint32_t>(v[i].second), v[i].first * k);
                std::fprintf(stderr, "\n");
            }
        }
        top(st.pcHist, "dispatch pcs", k);
        top(st.exitHist, "exit pcs (ctx->pc after return)", k);
        std::fprintf(stderr, "[sched-stats]   transfers(exceptions)=%.0f/s\n", st.transfers * k);
        if (st.acctCalls != 0u)
            std::fprintf(stderr, "[sched-stats]   accountCycles, EE timers + IOP clock part: %.0f calls/s, %.1f ns each (EE timers %.1f ns, IOP clock %.1f ns; rdtsc: each figure includes ~10 ns of its own overhead)\n",
                         st.acctCalls * k, st.acctTsc / 3.187 / st.acctCalls, st.timersTsc / 3.187 / st.acctCalls, st.iopTsc / 3.187 / st.acctCalls);
        std::fprintf(stderr,
                     "[sched-stats]   guest clock per second: estimated %.1f Mcycles, added by the wall-clock floor %.1f Mcycles (EE clock 294.9 M/s); "
                     "slept for vblank pacing %.1f ms in %.0f waits\n",
                     st.estCycles * k / 1e6, st.clampCycles * k / 1e6, st.pacingSleepTsc * k / 3.187e6, st.pacingSleeps * k);
        {
            const uint64_t total = st.tscInGuestSpin + st.tscInGuestOther + st.tscLoopSpin + st.tscLoopOther;
            const double t = total ? 100.0 / double(total) : 0.0;
            std::fprintf(stderr,
                         "[sched-stats]   time in run(): guest wait-loop %.1f%%, guest other %.1f%%, scheduler around wait-loop dispatches %.1f%%, "
                         "scheduler around other dispatches %.1f%% (rdtsc, run() only; %.0f Mtsc/s)\n",
                         st.tscInGuestSpin * t, st.tscInGuestOther * t, st.tscLoopSpin * t, st.tscLoopOther * t, total * k / 1e6);
        }
        const bool hist = st.hist, perDispatch = st.perDispatch;
        st = SchedStats{};
        st.on = true;
        st.hist = hist;
        st.perDispatch = perDispatch;
    }
}

namespace
{
    // currentThread() runs on every EE checkpoint (accountCycles -> currentContext); cache the hash lookup.
    // unordered_map node addresses are stable; the cache is dropped whenever a thread is erased.
    const EeScheduler *g_cachedScheduler = nullptr;
    int g_cachedThreadId = -1;
    GuestThread *g_cachedThread = nullptr;
    void dropThreadCache() { g_cachedScheduler = nullptr; }
}


// Scheduler-level trace ring: every guest entry (thread/invocation, pc, ra, sp) and what it came back with.
// Printed when the scheduler hits a pc with no recompiled function.
namespace
{
    struct SchedTrace
    {
        char what; // 'E' enter, 'X' exit, 'I' invocation pushed, 'D' invocation done
        int thread;
        uint32_t depth, pc, ra, sp;
    };
    SchedTrace g_schedTrace[256];
    uint32_t g_schedTracePos = 0;
    inline void schedTrace(char what, int thread, size_t depth, const R5900Context &c)
    {
        g_schedTrace[g_schedTracePos++ & 255u] = SchedTrace{what, thread, static_cast<uint32_t>(depth), c.pc,
                                                            getRegU32(&c, 31), getRegU32(&c, 29)};
    }
    void dumpSchedTrace()
    {
        std::fprintf(stderr, "[sched-trace] oldest first:\n");
        for (uint32_t i = 0; i < 256u; ++i)
        {
            const SchedTrace &t = g_schedTrace[(g_schedTracePos + i) & 255u];
            if (!t.what)
                continue;
            std::fprintf(stderr, "  %c th=%d d=%u pc=0x%x ra=0x%x sp=0x%x\n", t.what, t.thread, t.depth, t.pc, t.ra, t.sp);
        }
    }
}

namespace
{
    constexpr int KE_OK = 0;
    constexpr int KE_ERROR = -1;
    constexpr int KE_ILLEGAL_PRIORITY = -403;
    constexpr int KE_ILLEGAL_THID = -406;
    constexpr int KE_UNKNOWN_THID = -407;
    constexpr int KE_UNKNOWN_SEMID = -408;
    constexpr int KE_UNKNOWN_EVFID = -409;
    constexpr int KE_DORMANT = -413;
    constexpr int KE_NOT_DORMANT = -414;
    constexpr int KE_NOT_SUSPEND = -415;
    constexpr int KE_NOT_WAIT = -416;
    constexpr int KE_RELEASE_WAIT = -418;
    constexpr int KE_SEMA_ZERO = -419;
    constexpr int KE_SEMA_OVF = -420;
    constexpr int KE_EVF_COND = -421;
    constexpr int KE_WAIT_DELETE = -425;

    constexpr uint32_t WEF_OR = 0x01u;
    constexpr uint32_t WEF_CLEAR = 0x10u;
    constexpr uint32_t WEF_CLEAR_ALL = 0x20u;
    std::atomic<uint32_t> g_vblankPeriodMicros{16667u};
    std::chrono::microseconds vblankPeriod()
    {
        return std::chrono::microseconds(g_vblankPeriodMicros.load(std::memory_order_relaxed));
    }
    constexpr auto kVBlankDuration = std::chrono::microseconds(500);
    constexpr uint64_t kAlarmTickMicroseconds = 64u;
    constexpr uint32_t kDebugPublishDispatchInterval = 4096u;

    constexpr uint64_t microsecondsToEeCycles(uint64_t microseconds)
    {
        return (microseconds * EeScheduler::kEeClockHz + 999999ull) / 1000000ull;
    }

    std::chrono::nanoseconds eeCyclesToHostDuration(uint64_t cycles)
    {
        constexpr uint64_t kNanosecondsPerSecond = 1000000000ull;
        const uint64_t wholeSeconds = cycles / EeScheduler::kEeClockHz;
        const uint64_t remainingCycles = cycles % EeScheduler::kEeClockHz;
        const uint64_t remainingNanoseconds = (remainingCycles * kNanosecondsPerSecond + EeScheduler::kEeClockHz - 1u) / EeScheduler::kEeClockHz;
        return std::chrono::seconds(wholeSeconds) + std::chrono::nanoseconds(remainingNanoseconds);
    }

    uint64_t vblankPeriodCycles()
    {
        return microsecondsToEeCycles(g_vblankPeriodMicros.load(std::memory_order_relaxed));
    }
    constexpr uint64_t kVBlankDurationCycles = microsecondsToEeCycles(500u);
    constexpr uint64_t kAlarmTickCycles = microsecondsToEeCycles(kAlarmTickMicroseconds);

    template <typename Map>
    int allocatePositiveId(int &nextId, const Map &objects)
    {
        const int first = std::max(1, nextId);
        int candidate = first;
        do
        {
            if (!objects.contains(candidate))
            {
                nextId = (candidate == std::numeric_limits<int>::max()) ? 1 : candidate + 1;
                return candidate;
            }
            candidate = (candidate == std::numeric_limits<int>::max()) ? 1 : candidate + 1;
        } while (candidate != first);
        return 0;
    }
}

EeScheduler::EeScheduler(PS2Runtime &runtime)
    : m_runtime(runtime)
{
}

EeScheduler::~EeScheduler()
{
    requestStop();
}

void EeScheduler::reset(uint8_t *rdram, const R5900Context &mainContext)
{
    m_executorThread = std::this_thread::get_id();
    m_rdram = rdram;
    m_readyQueues = {};
    m_threads.clear();
    dropThreadCache();
    m_semaphores.clear();
    m_eventFlags.clear();
    m_alarms.clear();
    m_intcHandlers.clear();
    m_dmacHandlers.clear();
    m_nextThreadId = kFirstThreadId;
    m_nextInvocationThreadId = -1;
    m_nextSemaphoreId = 1;
    m_nextEventFlagId = 1;
    m_nextAlarmId = 1;
    m_nextIntcHandlerId = 1;
    m_nextDmacHandlerId = 1;
    m_intcHeadOrder = 0;
    m_intcTailOrder = 1000;
    m_dmacHeadOrder = 0;
    m_dmacTailOrder = 1000;
    m_enabledIntcMask = 0xFFFFFFFFu;
    m_enabledDmacMask = 0xFFFFFFFFu;
    m_currentThreadId = 0;
    m_rescheduleRequested = false;
    m_timeSliceExpired = false;
    m_insideInterrupt = false;
    m_pendingEeTimerInterrupts = 0u;
    m_eeCycle = 0u;
    m_sliceEndCycle = kDefaultTimeSliceCycles;
    m_stopRequested.store(false, std::memory_order_release);
    m_checkpointPending.store(false, std::memory_order_release);
    m_debugPublishCountdown = 0u;
    {
        std::lock_guard lock(m_eventMutex);
        m_events.clear();
        g_queuedEvents.store(0u, std::memory_order_release);
        m_deadlines.clear();
        m_pendingInvocations.clear();
    }
    m_eventSequence = 0;
    m_invocationSequence = 0;
    m_vsyncTick = 0;
    m_vsyncFlagAddress = 0;
    m_vsyncTickAddress = 0;
    m_gsVSyncCallback = 0;
    m_gsVSyncCallbackGp = 0;
    m_gsVSyncCallbackSp = 0;
    m_runtime.memory().gs().vsyncTick.store(0u, std::memory_order_release);
    m_runtime.memory().resetEeTimers();

    GuestThread main{};
    main.id = kMainThreadId;
    main.context = mainContext;
    main.entry = mainContext.pc;
    // $sp is live execution state, not the stable initial stack descriptor
    // returned by ReferThreadStatus. SetupThread records that metadata.
    main.stack = 0u;
    main.gp = getRegU32(&mainContext, 28);
    main.initialPriority = 0;
    main.currentPriority = 0;
    main.status = EeThreadStatus::Ready;
    m_threads.emplace(main.id, std::move(main));
    m_readyQueues[0].push_back(kMainThreadId);
    scheduleEvent(m_eeCycle + vblankPeriodCycles(),
                  std::chrono::steady_clock::now() + vblankPeriod(),
                  EeEvent{EeEventType::VBlankStart, 0, 0});
    publishSnapshot();
}

void EeScheduler::run()
{
    assertExecutor();
    m_running.store(true, std::memory_order_release);

    // processPendingEvents ran twice per iteration (and iterations are ~1M/s in gameplay) although it has work only when
    // an event was posted, a cycle deadline or EE timer interrupt is due, a reschedule is requested, or the checkpoint
    // flag is set (the flag is what it clears). Everything it looks at is tested here without the call.
    // PS2X_EVENT_PUMP_ALWAYS=1 restores the unconditional call (A/B).
    static const bool alwaysPump = [] {
        const char *v = std::getenv("PS2X_EVENT_PUMP_ALWAYS");
        return v && *v && *v != '0';
    }();
    const auto eventPumpNeeded = [this]() noexcept {
        if (alwaysPump)
            return true;
        if (m_checkpointPending.load(std::memory_order_relaxed) || m_rescheduleRequested ||
            m_pendingEeTimerInterrupts != 0u || g_queuedEvents.load(std::memory_order_relaxed) != 0u)
            return true;
        const uint64_t nextDeadline = m_nextDeadlineCycle.load(std::memory_order_relaxed);
        return nextDeadline != 0u && m_eeCycle >= nextDeadline;
    };

    while (!m_stopRequested.load(std::memory_order_acquire))
    {
        if (g_schedStats.perDispatch)
        {
            ++g_schedStats.loopIters;
            schedStatsReport();
        }
        if (eventPumpNeeded())
            processPendingEvents();
        if (m_stopRequested.load(std::memory_order_acquire))
        {
            break;
        }

        if (m_currentThreadId == 0)
        {
            GuestThread *next = selectReady();
            if (!next && m_pendingInvocations.empty())
            {
                copyMainContextToRuntime();
                publishIdleDebugContext();
                publishSnapshot();
                SCHED_STAT(++g_schedStats.idleWaits);
                waitForEvent();
                SCHED_STAT(g_schedStats.tscPrevExit = 0u);
                continue;
            }
            if (next)
            {
                makeRunning(*next);
            }
            else
            {
                GuestThread *owner = &acquireInvocationThread();
                GuestInvocation invocation = std::move(m_pendingInvocations.front());
                m_pendingInvocations.pop_front();
                owner->status = EeThreadStatus::Running;
                m_currentThreadId = owner->id;
                renewTimeSlice();
                if (getRegU32(&invocation.context, 29) == 0u)
                {
                    SET_GPR_U32(&invocation.context, 29, invocationStackTop());
                }
                ps2tl::rec(ps2tl::INV_B, invocation.context.pc, static_cast<uint32_t>(invocation.kind), static_cast<uint32_t>(m_eeCycle));
                owner->invocations.push_back(std::move(invocation));
            }
        }

        GuestThread *running = currentThread();
        assert(running != nullptr);
        if (running->resumeCompletion)
        {
            auto completion = std::move(running->resumeCompletion);
            running->resumeCompletion = {};
            try
            {
                g_ps2GuestUnwinding = false;
                completion(running->activeContext());
            }
            catch (const EeDispatcherTransfer &)
            {
            }
            if (m_currentThreadId == 0)
            {
                continue;
            }
        }
        R5900Context &context = running->activeContext();
        if (m_debugPublishCountdown == 0u)
        {
            copyMainContextToRuntime();
            publishSnapshot();
            m_debugPublishCountdown = kDebugPublishDispatchInterval - 1u;
        }
        else
        {
            --m_debugPublishCountdown;
        }

        // Debug display only (pc/ra/sp/gp of the running thread): every 8th dispatch is plenty.
        static uint32_t debugPublishTick = 0u;
        if ((++debugPublishTick & 7u) == 0u)
            publishDebugContext(context);

        if (context.pc == 0u)
        {
            if (!running->invocations.empty())
            {
                GuestInvocation completed = std::move(running->invocations.back());
                running->invocations.pop_back();
                ps2tl::rec(ps2tl::INV_E, completed.context.pc, static_cast<uint32_t>(completed.kind), static_cast<uint32_t>(m_eeCycle));
                schedTrace('D', running->id, running->invocations.size(), running->activeContext());
                if (completed.onComplete)
                {
                    try
                    {
                        completed.onComplete(completed.context, running->activeContext());
                    }
                    catch (const EeDispatcherTransfer &)
                    {
                    }
                }
                continue;
            }
            makeDormant(*running);
            m_currentThreadId = 0;
            copyMainContextToRuntime();
            publishIdleDebugContext();
            publishSnapshot();
            continue;
        }

        // Pending invocations (interrupts, SIF commands, stream callbacks) run on a thread that is executing its own
        // code, including guest code an HLE function called on its behalf (HleCall): a libmpeg nodata callback that
        // blocks on a file read needs the SIF command that completes the read.
        if (!m_pendingInvocations.empty() &&
            (running->invocations.empty() || running->invocations.back().kind == GuestInvocationKind::HleCall))
        {
            GuestInvocation invocation = std::move(m_pendingInvocations.front());
            m_pendingInvocations.pop_front();
            if (getRegU32(&invocation.context, 29) == 0u)
            {
                SET_GPR_U32(&invocation.context, 29, invocationStackTop());
            }
            ps2tl::rec(ps2tl::INV_B, invocation.context.pc, static_cast<uint32_t>(invocation.kind), static_cast<uint32_t>(m_eeCycle));
            running->invocations.push_back(std::move(invocation));
            schedTrace('I', running->id, running->invocations.size(), context);
            continue;
        }

        // One probe of the dense function table instead of hasFunction + lookupFunction (two out-of-line calls; the
        // latter also records the dispatch history, a debug aid for missing-function reports).
        PS2Runtime::RecompiledFunction function = nullptr;
        {
            const uint32_t slot = (context.pc - g_ps2RecompiledFunctionTableBase) >> 2;
            if ((context.pc & 3u) == 0u && slot < g_ps2RecompiledFunctionTableSlotCount)
                function = g_ps2RecompiledFunctionTable[slot];
        }
        if (function == nullptr)
        {
            if (!running->invocations.empty())
            {
                context.pc = 0u;
            }
            else
            {
                dumpSchedTrace();
                m_runtime.reportMissingFunction(m_rdram,
                                                &context,
                                                context.pc,
                                                context.pc,
                                                PS2Runtime::GuestBranchKind::DirectJump,
                                                "EE scheduler");
                makeDormant(*running);
                m_currentThreadId = 0;
            }
            continue;
        }

        // A dispatch costs kGuestDispatchCycles like a call does, and is charged the same way: against the guest's
        // checkpoint budget (ps2_cg.h), with the accumulated charge passed to checkpointDue when it runs out. A checkpoint
        // per dispatch (~1M/s in gameplay: every entry fragment and every return after an unwind comes through here) was
        // the largest part of the scheduler's cost. Total cycles charged are the same; events and deadlines are noticed at
        // the top of the loop, so nothing waits for the checkpoint.
        if (g_dispatchViaBudget)
        {
            if ((g_ps2EeBudget -= static_cast<int32_t>(kGuestDispatchCycles)) < 0)
            {
                const int32_t refill = ps2CgBudgetRefill();
                const int32_t consumed = refill - g_ps2EeBudget;
                g_ps2EeBudget = refill;
                if (checkpointDue(static_cast<uint32_t>(consumed > 0 ? consumed : 1)))
                {
                    continue;
                }
            }
        }
        else if (checkpointDue(kGuestDispatchCycles))
        {
            continue;
        }

        try
        {
            uint64_t statT0 = 0;
            bool statSpin = false;
            if (g_schedStats.perDispatch)
            {
                ++g_schedStats.dispatches;
                if (g_schedStats.hist)
                    ++g_schedStats.pcHist[context.pc];
                {
                    SchedStats &st = g_schedStats;
                    unsigned kind = 0;
                    uint32_t prevEntry = 0;
                    if (st.lastValid && st.lastThread == running->id && st.lastExitPc == context.pc)
                    {
                        prevEntry = st.lastEntryPc;
                        const uint64_t y = st.yieldPending + st.yieldDeadline + st.yieldSlice + st.yieldStress;
                        kind = 3;
                        if (y != st.yieldsAtEntry)
                            kind = 1;
                        else if (context.pc >= 8u && context.pc < 0x2000000u)
                        {
                            uint32_t w = 0;
                            std::memcpy(&w, m_rdram + context.pc - 8u, 4);
                            // pc is a call's return address (jal / jalr two words before) and the guest returned through
                            // $ra (a callee returned to a caller that had been unwound). A branch that merely lands on such
                            // an address is a jump.
                            if (((w >> 26) == 3u || ((w >> 26) == 0u && (w & 0x3fu) == 9u)) && getRegU32(&context, 31) == context.pc)
                                kind = 2;
                        }
                    }
                    ++st.reasonCount[kind];
                    if (st.hist)
                        ++st.pairHist[kind][(static_cast<uint64_t>(prevEntry) << 32) | context.pc];
                    st.lastEntryPc = context.pc;
                    st.yieldsAtEntry = st.yieldPending + st.yieldDeadline + st.yieldSlice + st.yieldStress;
                    st.lastValid = false;
                }
                statSpin = context.pc >= 0x14fd90u && context.pc < 0x15008cu;
                statT0 = __rdtsc();
                if (g_schedStats.tscPrevExit != 0u)
                    (statSpin ? g_schedStats.tscLoopSpin : g_schedStats.tscLoopOther) += statT0 - g_schedStats.tscPrevExit;
            }
            m_insideInterrupt = !running->invocations.empty() && running->invocations.back().kind == GuestInvocationKind::Interrupt;
            m_guestExecuting.store(true, std::memory_order_release);
            schedTrace('E', running->id, running->invocations.size(), context);
            g_ps2GuestUnwinding = false;
            function(m_rdram, &context, &m_runtime);
            ps2DspSchedulerFollow(m_rdram, &context, &m_runtime);
            context.pc &= ~PS2_TAIL_MARK; // a tail jump the guest's call sites did not follow (ps2_cg_dsp.h): dispatch its target
            schedTrace('X', running->id, running->invocations.size(), context);
            if (g_schedStats.perDispatch)
            {
                g_schedStats.lastExitPc = context.pc;
                g_schedStats.lastThread = running->id;
                g_schedStats.lastValid = true;
                if (g_schedStats.hist)
                    ++g_schedStats.exitHist[context.pc];
                g_schedStats.tscPrevExit = __rdtsc();
                (statSpin ? g_schedStats.tscInGuestSpin : g_schedStats.tscInGuestOther) += g_schedStats.tscPrevExit - statT0;
            }
            m_guestExecuting.store(false, std::memory_order_release);
            m_insideInterrupt = false;
        }
        catch (const EeDispatcherTransfer &)
        {
            schedTrace('T', running->id, running->invocations.size(), context);
            SCHED_STAT(++g_schedStats.transfers);
            m_guestExecuting.store(false, std::memory_order_release);
            m_insideInterrupt = false;
        }
        catch (...)
        {
            m_guestExecuting.store(false, std::memory_order_release);
            m_running.store(false, std::memory_order_release);
            publishSnapshot();
            throw;
        }

        if (eventPumpNeeded())
            processPendingEvents();
        if (m_rescheduleRequested && m_currentThreadId != 0)
        {
            GuestThread *preempted = currentThread();
            assert(preempted != nullptr);
            enqueueReady(*preempted, !m_timeSliceExpired);
            m_currentThreadId = 0;
            m_rescheduleRequested = false;
            m_timeSliceExpired = false;
        }
    }

    m_guestExecuting.store(false, std::memory_order_release);
    m_running.store(false, std::memory_order_release);
    copyMainContextToRuntime();
    publishSnapshot();
}

void EeScheduler::requestStop()
{
    m_stopRequested.store(true, std::memory_order_release);
    m_checkpointPending.store(true, std::memory_order_release);
    m_eventCv.notify_all();
}

void EeScheduler::postEvent(EeEvent event)
{
    if (event.type == EeEventType::Stop)
    {
        requestStop();
        return;
    }

    {
        std::lock_guard lock(m_eventMutex);
        m_events.push_back(event);
        g_queuedEvents.fetch_add(1u, std::memory_order_release);
        m_checkpointPending.store(true, std::memory_order_release);
    }
    m_eventCv.notify_one();
}

bool EeScheduler::checkpointDue(uint32_t cycles) noexcept
{
    if (const uint32_t scale = g_eeCycleScaleQ16.load(std::memory_order_relaxed); scale != 65536u) [[unlikely]]
        cycles = static_cast<uint32_t>((static_cast<uint64_t>(cycles) * scale) >> 16); // accountCycles charges at least 1
    accountCycles(cycles);
#ifdef PS2X_SCHED_STATS_HOT // per-checkpoint counters cost a load and a branch each: compiled in only on request
    SCHED_STAT(++g_schedStats.checkpoints; g_schedStats.checkpointCycles += cycles);
#endif

    // PS2X_STRESS_YIELD=N: yield to the scheduler at every Nth checkpoint (debug aid for resume bugs).
    const uint32_t stressYield = g_stressYield;
    static uint32_t stressCount = 0;
    if (stressYield != 0u && ++stressCount >= stressYield)
    {
        stressCount = 0;
        SCHED_STAT(++g_schedStats.yieldStress);
        return true;
    }

    if (m_checkpointPending.load(std::memory_order_acquire) ||
        m_stopRequested.load(std::memory_order_acquire))
    {
        SCHED_STAT(++g_schedStats.yieldPending);
        return true;
    }

    const uint64_t nextEventCycle = m_nextDeadlineCycle.load(std::memory_order_acquire);
    if (nextEventCycle != 0u && m_eeCycle >= nextEventCycle)
    {
        m_checkpointPending.store(true, std::memory_order_release);
        SCHED_STAT(++g_schedStats.yieldDeadline);
        return true;
    }

    if (m_eeCycle < m_sliceEndCycle)
    {
        return false;
    }

    const GuestThread *running = currentThread();
    if (running != nullptr && hasReadyAtOrAbovePriority(running->currentPriority))
    {
        m_rescheduleRequested = true;
        m_timeSliceExpired = true;
        SCHED_STAT(++g_schedStats.yieldSlice);
        return true;
    }

    renewTimeSlice();
    return false;
}

namespace
{
    // COP0 Count runs at the EE clock (294.912 MHz). Recompiled `mfc0 Count` reads ctx->cop0_count, which nothing
    // else advances, so games that time themselves with Count (fades, frame timers) would see time stand still.
    // Wall-clock time since the first call, in EE cycles (294.912 MHz).
    uint64_t hostCop0Count64()
    {
        static const auto start = std::chrono::steady_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        return static_cast<uint64_t>(static_cast<double>(ns) * 0.294912);
    }
}

void EeScheduler::accountCycles(uint32_t cycles) noexcept
{
    // The EE cycle clock never falls behind wall time: checkpoints only charge an estimate per block, so recompiled
    // code (and especially guest spin-waits) would otherwise let emulated time lag real time, and every cycle-gated
    // event (vblank, EE timers, IOP time) would run slow. Faster-than-real-time execution still accrues cycles.
    const bool wallClock = g_wallClockPacing;
    // Reading the host clock costs ~20-30 ns and this runs on nearly every guest call: sample it every 64th call.
    static uint64_t wallCycles = 0;
    static uint32_t wallSampleCountdown = 0;
    if (wallSampleCountdown-- == 0u)
    {
        wallCycles = static_cast<uint64_t>(hostCop0Count64());
        wallSampleCountdown = 63u;
    }
    uint64_t next = m_eeCycle + std::max<uint64_t>(1u, cycles);
#ifdef PS2X_SCHED_STATS_HOT
    if (g_schedStats.on)
    {
        g_schedStats.estCycles += std::max<uint64_t>(1u, cycles);
        if (wallClock && next < wallCycles)
            g_schedStats.clampCycles += wallCycles - next;
    }
#endif
    if (wallClock && next < wallCycles)
        next = wallCycles;
    const uint64_t elapsed = next - m_eeCycle;
    m_eeCycle = next;
    // Inline fast path of currentContext(): the thread cache (currentThread) is valid at nearly every checkpoint.
    GuestThread *const self = (g_cachedScheduler == this && g_cachedThreadId == m_currentThreadId && g_cachedThread != nullptr)
                                  ? g_cachedThread
                                  : currentThread();
    if (self != nullptr)
        (self->invocations.empty() ? self->context : self->invocations.back().context).cop0_count = static_cast<uint32_t>(wallCycles);
#ifdef PS2X_SCHED_STATS_HOT
    const uint64_t statA0 = g_schedStats.on ? __rdtsc() : 0u;
#endif
    m_pendingEeTimerInterrupts |= m_runtime.memory().advanceEeTimers(elapsed);
#ifdef PS2X_SCHED_STATS_HOT
    const uint64_t statA1 = g_schedStats.on ? __rdtsc() : 0u;
#endif
    m_runtime.advanceIopEeCycles(elapsed);
#ifdef PS2X_SCHED_STATS_HOT
    if (g_schedStats.on)
    {
        const uint64_t statA2 = __rdtsc();
        ++g_schedStats.acctCalls;
        g_schedStats.timersTsc += statA1 - statA0;
        g_schedStats.iopTsc += statA2 - statA1;
        g_schedStats.acctTsc += statA2 - statA0;
    }
#endif
    if (m_pendingEeTimerInterrupts != 0u)
    {
        m_checkpointPending.store(true, std::memory_order_release);
    }
}

bool EeScheduler::isExecutingGuest() const noexcept
{
    return m_guestExecuting.load(std::memory_order_acquire);
}

void EeScheduler::setupCurrentThread(uint32_t stack, uint32_t stackSize, uint32_t gp)
{
    assertExecutor();
    GuestThread *target = currentThread();
    if (!target)
    {
        return;
    }

    target->stack = stack;
    target->stackSize = stackSize;
    target->gp = gp;
    publishSnapshot();
}

int EeScheduler::createThread(const EeThreadCreateParams &params)
{
    assertExecutor();
    if (params.priority < 1 || params.priority >= kPriorityCount)
    {
        return KE_ILLEGAL_PRIORITY;
    }

    const int id = allocateThreadId();
    if (id == 0)
    {
        return KE_ERROR;
    }

    GuestThread thread{};
    thread.id = id;
    thread.entry = params.entry;
    thread.stack = params.stack;
    thread.stackSize = params.stackSize;
    thread.gp = params.gp;
    thread.attr = params.attr;
    thread.option = params.option;
    thread.initialPriority = params.priority;
    thread.currentPriority = params.priority;
    thread.status = EeThreadStatus::Dormant;
    m_threads.emplace(id, std::move(thread));
    publishSnapshot();
    return id;
}

int EeScheduler::deleteThread(int id, uint32_t &ownedStack)
{
    assertExecutor();
    ownedStack = 0;
    if (id <= kMainThreadId)
    {
        return KE_ILLEGAL_THID;
    }
    auto it = m_threads.find(id);
    if (it == m_threads.end())
    {
        return KE_UNKNOWN_THID;
    }
    if (it->second.status != EeThreadStatus::Dormant)
    {
        return KE_NOT_DORMANT;
    }
    if (it->second.ownsStack)
    {
        ownedStack = it->second.stack;
    }
    m_threads.erase(it);
    dropThreadCache();
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::startThread(int id, uint32_t arg, const R5900Context &caller, bool interruptSafe)
{
    assertExecutor();
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status != EeThreadStatus::Dormant)
    {
        return KE_NOT_DORMANT;
    }

    target->context = R5900Context{};
    target->context.pc = target->entry;
    target->arg = arg;
    target->suspendCount = 0;
    target->wakeupCount = 0;
    target->wait = {};
    SET_GPR_U32(&target->context, 4, arg);
    SET_GPR_U32(&target->context, 28, target->gp != 0u ? target->gp : getRegU32(&caller, 28));
    const uint32_t stackTop = target->stack != 0u
                                  ? (target->stack + target->stackSize) & ~0xFu
                                  : getRegU32(&caller, 29);
    SET_GPR_U32(&target->context, 29, stackTop);
    SET_GPR_U32(&target->context, 31, 0u);
    enqueueReady(*target);
    requestPreemptionIfHigher(*target, interruptSafe);
    publishSnapshot();
    return KE_OK;
}

[[noreturn]] void EeScheduler::exitCurrent(bool deleteThreadRecord)
{
    assertExecutor();
    GuestThread *exiting = currentThread();
    assert(exiting != nullptr);
    const int id = exiting->id;
    const uint32_t ownedStack = deleteThreadRecord && exiting->ownsStack ? exiting->stack : 0u;
    makeDormant(*exiting);
    m_currentThreadId = 0;
    if (deleteThreadRecord && id != kMainThreadId)
    {
        m_threads.erase(id);
        dropThreadCache();
    }
    if (ownedStack != 0u)
    {
        m_runtime.guestFree(ownedStack);
    }
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

int EeScheduler::terminateThread(int id, uint32_t &ownedStack, bool interruptSafe)
{
    assertExecutor();
    ownedStack = 0;
    if (id == 0 || id == m_currentThreadId)
    {
        return KE_ILLEGAL_THID;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status == EeThreadStatus::Dormant)
    {
        return KE_DORMANT;
    }
    if (target->ownsStack)
    {
        ownedStack = target->stack;
        target->ownsStack = false;
    }
    makeDormant(*target);
    (void)interruptSafe;
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::suspendThread(int id, bool interruptSafe)
{
    assertExecutor();
    if (id == 0)
    {
        id = m_currentThreadId;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status == EeThreadStatus::Dormant)
    {
        return KE_DORMANT;
    }

    ++target->suspendCount;
    switch (target->status)
    {
    case EeThreadStatus::Running:
        target->status = EeThreadStatus::Suspended;
        m_currentThreadId = 0;
        m_rescheduleRequested = true;
        break;
    case EeThreadStatus::Ready:
        removeReady(*target);
        target->status = EeThreadStatus::Suspended;
        break;
    case EeThreadStatus::Waiting:
        target->status = EeThreadStatus::WaitingSuspended;
        break;
    case EeThreadStatus::WaitingSuspended:
    case EeThreadStatus::Suspended:
        break;
    case EeThreadStatus::Dormant:
        break;
    }
    if (interruptSafe && m_insideInterrupt)
    {
        m_rescheduleRequested = true;
    }
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::resumeThread(int id, bool interruptSafe)
{
    assertExecutor();
    if (id == 0)
    {
        return KE_ILLEGAL_THID;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->suspendCount == 0)
    {
        return KE_NOT_SUSPEND;
    }
    --target->suspendCount;
    if (target->suspendCount != 0)
    {
        return KE_OK;
    }
    if (target->status == EeThreadStatus::WaitingSuspended)
    {
        target->status = EeThreadStatus::Waiting;
    }
    else if (target->status == EeThreadStatus::Suspended)
    {
        enqueueReady(*target);
        requestPreemptionIfHigher(*target, interruptSafe);
    }
    publishSnapshot();
    return KE_OK;
}

void EeScheduler::sleepCurrent()
{
    assertExecutor();
    GuestThread *self = currentThread();
    assert(self != nullptr);
    if (self->wakeupCount != 0u)
    {
        --self->wakeupCount;
        setReturnS32(&self->activeContext(), KE_OK);
        return;
    }
    blockCurrent(EeWaitState{EeWaitReason::Sleep, std::monostate{}});
}

int EeScheduler::wakeupThread(int id, bool interruptSafe)
{
    assertExecutor();
    if (id == 0 || id == m_currentThreadId)
    {
        return KE_ILLEGAL_THID;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status == EeThreadStatus::Dormant)
    {
        return KE_DORMANT;
    }
    if ((target->status == EeThreadStatus::Waiting || target->status == EeThreadStatus::WaitingSuspended) &&
        target->wait.reason == EeWaitReason::Sleep)
    {
        makeReady(*target, KE_OK, interruptSafe);
    }
    else
    {
        ++target->wakeupCount;
    }
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::cancelWakeup(int id)
{
    assertExecutor();
    if (id == 0)
    {
        id = m_currentThreadId;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    const int old = static_cast<int>(target->wakeupCount);
    target->wakeupCount = 0;
    publishSnapshot();
    return old;
}

int EeScheduler::changePriority(int id, int priority, bool interruptSafe, int &oldPriority)
{
    assertExecutor();
    if (priority < 1 || priority >= kPriorityCount)
    {
        return KE_ILLEGAL_PRIORITY;
    }
    if (id == 0)
    {
        id = m_currentThreadId;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    oldPriority = target->currentPriority;
    if (target->status == EeThreadStatus::Ready)
    {
        removeReady(*target);
        target->currentPriority = priority;
        enqueueReady(*target);
        requestPreemptionIfHigher(*target, interruptSafe);
    }
    else
    {
        target->currentPriority = priority;
        if (target->status == EeThreadStatus::Running)
        {
            for (int p = 0; p < target->currentPriority; ++p)
            {
                if (!m_readyQueues[p].empty())
                {
                    m_rescheduleRequested = true;
                    break;
                }
            }
        }
    }
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::rotateReadyQueue(int priority, bool interruptSafe)
{
    assertExecutor();
    if (priority == 0)
    {
        const GuestThread *self = currentThread();
        priority = self ? self->currentPriority : 0;
    }
    if (priority < 0 || priority >= kPriorityCount)
    {
        return KE_ILLEGAL_PRIORITY;
    }

    GuestThread *self = currentThread();
    if (self && self->currentPriority == priority)
    {
        enqueueReady(*self);
        m_currentThreadId = 0;
        m_rescheduleRequested = true;
    }
    else
    {
        auto &queue = m_readyQueues[priority];
        if (queue.size() > 1u)
        {
            const int head = queue.front();
            queue.pop_front();
            queue.push_back(head);
        }
    }
    (void)interruptSafe;
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::releaseWait(int id, bool interruptSafe)
{
    assertExecutor();
    if (id == 0)
    {
        return KE_ILLEGAL_THID;
    }
    GuestThread *target = thread(id);
    if (!target)
    {
        return KE_UNKNOWN_THID;
    }
    if (target->status != EeThreadStatus::Waiting && target->status != EeThreadStatus::WaitingSuspended)
    {
        return KE_NOT_WAIT;
    }
    removeFromWaitObject(*target);
    makeReady(*target, KE_RELEASE_WAIT, interruptSafe);
    publishSnapshot();
    return KE_OK;
}

void EeScheduler::transferIfRequested(bool interruptSafe)
{
    assertExecutor();
    if (interruptSafe || m_insideInterrupt || !m_rescheduleRequested)
    {
        return;
    }
    if (m_currentThreadId != 0)
    {
        GuestThread *self = currentThread();
        assert(self != nullptr);
        enqueueReady(*self, true);
        m_currentThreadId = 0;
    }
    m_rescheduleRequested = false;
    m_timeSliceExpired = false;
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

int EeScheduler::createSemaphore(int initCount, int maxCount, uint32_t attr, uint32_t option)
{
    assertExecutor();
    if (maxCount <= 0 || initCount < 0 || initCount > maxCount)
    {
        return KE_ERROR;
    }
    const int id = allocatePositiveId(m_nextSemaphoreId, m_semaphores);
    if (id == 0)
    {
        return KE_ERROR;
    }
    EeSemaphore semaphore{};
    semaphore.id = id;
    semaphore.count = initCount;
    semaphore.maxCount = maxCount;
    semaphore.initCount = initCount;
    semaphore.attr = attr;
    semaphore.option = option;
    m_semaphores.emplace(id, std::move(semaphore));
    publishSnapshot();
    return id;
}

int EeScheduler::deleteSemaphore(int id, bool interruptSafe)
{
    assertExecutor();
    auto it = m_semaphores.find(id);
    if (it == m_semaphores.end())
    {
        return KE_UNKNOWN_SEMID;
    }
    std::deque<int> waiters = std::move(it->second.waiters);
    m_semaphores.erase(it);
    for (const int threadId : waiters)
    {
        if (GuestThread *waiter = thread(threadId))
        {
            makeReady(*waiter, KE_WAIT_DELETE, interruptSafe);
        }
    }
    publishSnapshot();
    return id;
}

int EeScheduler::signalSemaphore(int id, bool interruptSafe)
{
    assertExecutor();
    EeSemaphore *object = semaphore(id);
    if (!object)
    {
        return KE_UNKNOWN_SEMID;
    }
    if (!object->waiters.empty())
    {
        const int waiterId = object->waiters.front();
        object->waiters.pop_front();
        GuestThread *waiter = thread(waiterId);
        assert(waiter != nullptr);
        makeReady(*waiter, id, interruptSafe);
        publishSnapshot();
        return id;
    }
    if (object->count == object->maxCount)
    {
        return KE_SEMA_OVF;
    }
    ++object->count;
    publishSnapshot();
    return id;
}

int EeScheduler::pollSemaphore(int id)
{
    assertExecutor();
    EeSemaphore *object = semaphore(id);
    if (!object)
    {
        return KE_UNKNOWN_SEMID;
    }
    if (object->count == 0)
    {
        return KE_SEMA_ZERO;
    }
    --object->count;
    publishSnapshot();
    return id;
}

void EeScheduler::waitSemaphore(int id)
{
    assertExecutor();
    EeSemaphore *object = semaphore(id);
    if (!object)
    {
        GuestThread *self = currentThread();
        assert(self != nullptr);
        setReturnS32(&self->activeContext(), KE_UNKNOWN_SEMID);
        return;
    }
    if (object->count != 0)
    {
        --object->count;
        GuestThread *self = currentThread();
        assert(self != nullptr);
        setReturnS32(&self->activeContext(), id);
        publishSnapshot();
        return;
    }
    GuestThread *self = currentThread();
    assert(self != nullptr);
    object->waiters.push_back(self->id);
    blockCurrent(EeWaitState{EeWaitReason::Semaphore, EeSemaphoreWait{id}});
}

int EeScheduler::createEventFlag(uint32_t initialBits, uint32_t attr, uint32_t option)
{
    assertExecutor();
    const int id = allocatePositiveId(m_nextEventFlagId, m_eventFlags);
    if (id == 0)
    {
        return KE_ERROR;
    }
    EeEventFlag flag{};
    flag.id = id;
    flag.attr = attr;
    flag.option = option;
    flag.initBits = initialBits;
    flag.bits = initialBits;
    m_eventFlags.emplace(id, std::move(flag));
    publishSnapshot();
    return id;
}

int EeScheduler::deleteEventFlag(int id, bool interruptSafe)
{
    assertExecutor();
    auto it = m_eventFlags.find(id);
    if (it == m_eventFlags.end())
    {
        return KE_UNKNOWN_EVFID;
    }
    std::deque<int> waiters = std::move(it->second.waiters);
    m_eventFlags.erase(it);
    for (const int threadId : waiters)
    {
        if (GuestThread *waiter = thread(threadId))
        {
            makeReady(*waiter, KE_WAIT_DELETE, interruptSafe);
        }
    }
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::setEventFlag(int id, uint32_t bits, bool interruptSafe)
{
    assertExecutor();
    EeEventFlag *flag = eventFlag(id);
    if (!flag)
    {
        return KE_UNKNOWN_EVFID;
    }
    flag->bits |= bits;
    finishEventWaiters(*flag, interruptSafe);
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::clearEventFlag(int id, uint32_t mask)
{
    assertExecutor();
    EeEventFlag *flag = eventFlag(id);
    if (!flag)
    {
        return KE_UNKNOWN_EVFID;
    }
    flag->bits &= mask;
    publishSnapshot();
    return KE_OK;
}

int EeScheduler::pollEventFlag(int id, uint32_t bits, uint32_t mode, uint32_t &observedBits)
{
    assertExecutor();
    EeEventFlag *flag = eventFlag(id);
    if (!flag)
    {
        return KE_UNKNOWN_EVFID;
    }
    if (!eventCondition(flag->bits, bits, mode))
    {
        return KE_EVF_COND;
    }
    observedBits = flag->bits;
    if ((mode & WEF_CLEAR_ALL) != 0u)
    {
        flag->bits = 0;
    }
    else if ((mode & WEF_CLEAR) != 0u)
    {
        flag->bits &= ~bits;
    }
    publishSnapshot();
    return KE_OK;
}

void EeScheduler::waitEventFlag(int id, uint32_t bits, uint32_t mode, uint32_t resultAddress)
{
    assertExecutor();
    EeEventFlag *flag = eventFlag(id);
    GuestThread *self = currentThread();
    assert(self != nullptr);
    if (!flag)
    {
        setReturnS32(&self->activeContext(), KE_UNKNOWN_EVFID);
        return;
    }
    if (eventCondition(flag->bits, bits, mode))
    {
        const uint32_t observed = flag->bits;
        writeGuestU32(resultAddress, observed);
        if ((mode & WEF_CLEAR_ALL) != 0u)
        {
            flag->bits = 0;
        }
        else if ((mode & WEF_CLEAR) != 0u)
        {
            flag->bits &= ~bits;
        }
        setReturnS32(&self->activeContext(), KE_OK);
        publishSnapshot();
        return;
    }
    flag->waiters.push_back(self->id);
    blockCurrent(EeWaitState{EeWaitReason::EventFlag,
                             EeEventFlagWait{id, bits, mode, resultAddress}});
}

int EeScheduler::setAlarm(uint16_t ticks,
                          uint32_t handler,
                          uint32_t argument,
                          uint32_t gp,
                          uint32_t sp)
{
    assertExecutor();
    if (handler == 0u || !m_runtime.hasFunction(handler))
    {
        return KE_ERROR;
    }
    const int id = allocatePositiveId(m_nextAlarmId, m_alarms);
    if (id == 0)
    {
        return KE_ERROR;
    }
    m_alarms.emplace(id, EeAlarm{id, ticks, handler, argument, gp, sp});
    const uint64_t tickCount = ticks == 0u ? 1u : static_cast<uint64_t>(ticks);
    scheduleEvent(m_eeCycle + tickCount * kAlarmTickCycles,
                  std::chrono::steady_clock::now() + std::chrono::microseconds(tickCount * kAlarmTickMicroseconds),
                  EeEvent{EeEventType::Alarm, static_cast<uint32_t>(id), 0});
    return id;
}

int EeScheduler::cancelAlarm(int id)
{
    assertExecutor();
    if (m_alarms.erase(id) == 0u)
    {
        return KE_ERROR;
    }
    {
        std::lock_guard lock(m_eventMutex);
        std::erase_if(m_deadlines, [id](const ScheduledEvent &scheduled)
                      { return scheduled.event.type == EeEventType::Alarm &&
                               scheduled.event.id == static_cast<uint32_t>(id); });
        updateNextDeadline();
    }
    return KE_OK;
}

void EeScheduler::queueInvocation(GuestInvocation invocation)
{
    assertExecutor();
    invocation.sequence = ++m_invocationSequence;
    m_pendingInvocations.push_back(std::move(invocation));
    m_checkpointPending.store(true, std::memory_order_release);
}

[[noreturn]] void EeScheduler::invokeCurrent(GuestInvocation invocation)
{
    assertExecutor();
    GuestThread *owner = currentThread();
    assert(owner != nullptr);
    if (getRegU32(&invocation.context, 29) == 0u)
    {
        SET_GPR_U32(&invocation.context, 29, invocationStackTop());
    }
    invocation.sequence = ++m_invocationSequence;
    owner->invocations.push_back(std::move(invocation));
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

[[noreturn]] void EeScheduler::invokeCurrentSequence(std::vector<GuestInvocation> invocations)
{
    assertExecutor();
    GuestThread *owner = currentThread();
    assert(owner != nullptr);
    assert(!invocations.empty());
    for (auto it = invocations.rbegin(); it != invocations.rend(); ++it)
    {
        if (getRegU32(&it->context, 29) == 0u)
        {
            SET_GPR_U32(&it->context, 29, invocationStackTop());
        }
        it->sequence = ++m_invocationSequence;
        owner->invocations.push_back(std::move(*it));
    }
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

bool EeScheduler::hasInvocation(GuestInvocationKind kind, uint64_t tag) const
{
    const GuestThread *owner = currentThread();
    if (!owner)
    {
        return false;
    }
    return std::any_of(owner->invocations.begin(), owner->invocations.end(),
                       [kind, tag](const GuestInvocation &invocation)
                       {
                           return invocation.kind == kind && invocation.tag == tag;
                       });
}

uint32_t EeScheduler::invocationStackTop()
{
    assertExecutor();
    const GuestThread *owner = currentThread();
    if (!owner)
    {
        throw std::logic_error("EE invocation stack requested without a current guest context");
    }
    const size_t depth = owner ? owner->invocations.size() : 0u;
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(owner->id)) << 32u) |
                         static_cast<uint32_t>(depth);
    const auto existing = m_invocationStackTops.find(key);
    if (existing != m_invocationStackTops.end())
    {
        return existing->second;
    }
    constexpr uint32_t kInvocationStackSize = 0x4000u;
    const uint32_t top = m_runtime.reserveAsyncCallbackStack(kInvocationStackSize, 16u);
    if (top == 0u)
    {
        throw std::runtime_error("EE invocation stack space exhausted");
    }
    m_invocationStackTops.emplace(key, top);
    return top;
}

int EeScheduler::addIrqHandler(bool dmac,
                               uint32_t cause,
                               uint32_t handler,
                               bool append,
                               uint32_t argument,
                               uint32_t gp,
                               uint32_t sp)
{
    assertExecutor();
    auto &handlers = dmac ? m_dmacHandlers : m_intcHandlers;
    int &nextId = dmac ? m_nextDmacHandlerId : m_nextIntcHandlerId;
    const int id = allocatePositiveId(nextId, handlers);
    if (id == 0)
    {
        return KE_ERROR;
    }
    int &head = dmac ? m_dmacHeadOrder : m_intcHeadOrder;
    int &tail = dmac ? m_dmacTailOrder : m_intcTailOrder;
    handlers.emplace(id,
                     EeIrqHandler{id,
                                  cause,
                                  handler,
                                  argument,
                                  gp,
                                  sp,
                                  true,
                                  append ? ++tail : --head});
    return id;
}

int EeScheduler::removeIrqHandler(bool dmac, uint32_t cause, int id)
{
    assertExecutor();
    auto &handlers = dmac ? m_dmacHandlers : m_intcHandlers;
    auto it = handlers.find(id);
    if (it != handlers.end() && it->second.cause == cause)
    {
        handlers.erase(it);
    }
    return KE_OK;
}

int EeScheduler::setIrqHandlerEnabled(bool dmac, int id, bool enabled)
{
    assertExecutor();
    auto &handlers = dmac ? m_dmacHandlers : m_intcHandlers;
    auto it = handlers.find(id);
    if (it != handlers.end())
    {
        it->second.enabled = enabled;
    }
    return KE_OK;
}

int EeScheduler::setIrqCauseEnabled(bool dmac, uint32_t cause, bool enabled)
{
    assertExecutor();
    if (cause < 32u)
    {
        uint32_t &mask = dmac ? m_enabledDmacMask : m_enabledIntcMask;
        if (enabled)
        {
            mask |= 1u << cause;
        }
        else
        {
            mask &= ~(1u << cause);
        }
    }
    return KE_OK;
}

void EeScheduler::dispatchIrq(bool dmac, uint32_t cause)
{
    assertExecutor();
    const uint32_t mask = dmac ? m_enabledDmacMask : m_enabledIntcMask;
    if (cause < 32u && (mask & (1u << cause)) == 0u)
    {
        return;
    }
    const auto &handlers = dmac ? m_dmacHandlers : m_intcHandlers;
    std::vector<EeIrqHandler> matching;
    for (const auto &[id, handler] : handlers)
    {
        (void)id;
        if (handler.enabled && handler.cause == cause && handler.handler != 0u &&
            m_runtime.hasFunction(handler.handler))
        {
            matching.push_back(handler);
        }
    }
    if (dmac)
        ps2tl::rec(ps2tl::DMAC_IRQ, cause, static_cast<uint32_t>(matching.size()), static_cast<uint32_t>(m_eeCycle));
    std::sort(matching.begin(), matching.end(), [](const EeIrqHandler &left, const EeIrqHandler &right)
              { return left.order < right.order; });
    for (const EeIrqHandler &handler : matching)
    {
        GuestInvocation invocation{};
        invocation.kind = GuestInvocationKind::Interrupt;
        invocation.context.pc = handler.handler;
        SET_GPR_U32(&invocation.context, 4, cause);
        SET_GPR_U32(&invocation.context, 5, handler.argument);
        SET_GPR_U32(&invocation.context, 28, handler.gp);
        SET_GPR_U32(&invocation.context, 29, 0u);
        SET_GPR_U32(&invocation.context, 31, 0u);
        queueInvocation(std::move(invocation));
    }
}

void EeScheduler::setVSyncFlag(uint32_t flagAddress, uint32_t tickAddress)
{
    assertExecutor();
    m_vsyncFlagAddress = flagAddress;
    m_vsyncTickAddress = tickAddress;
    writeGuestU32(flagAddress, 0u);
    if (tickAddress != 0u)
    {
        const uint32_t physical = tickAddress & 0x1FFFFFFFu;
        if (m_rdram && physical <= PS2_RAM_SIZE - sizeof(uint64_t))
        {
            const uint64_t zero = 0u;
            std::memcpy(m_rdram + physical, &zero, sizeof(zero));
        }
    }
}

uint64_t EeScheduler::currentVSyncTick() const noexcept
{
    return m_vsyncTick;
}

uint32_t EeScheduler::setGsVSyncCallback(uint32_t callback, uint32_t gp, uint32_t sp)
{
    assertExecutor();
    (void)sp;
    const uint32_t previous = m_gsVSyncCallback;
    m_gsVSyncCallback = callback;
    m_gsVSyncCallbackGp = gp;
    m_gsVSyncCallbackSp = 0u;
    return previous;
}

[[noreturn]] void EeScheduler::waitVSync(uint64_t afterTick, int fixedResult, std::function<void(R5900Context &)> completion)
{
    blockCurrent(EeWaitState{
        EeWaitReason::VSync,
        EeVSyncWait{afterTick, fixedResult},
        std::move(completion)});
}

void EeScheduler::completeVSync(uint64_t tick)
{
    assertExecutor();
    std::vector<int> completed;
    for (const auto &[id, candidate] : m_threads)
    {
        if ((candidate.status == EeThreadStatus::Waiting || candidate.status == EeThreadStatus::WaitingSuspended) &&
            candidate.wait.reason == EeWaitReason::VSync &&
            std::get<EeVSyncWait>(candidate.wait.payload).afterTick < tick)
        {
            completed.push_back(id);
        }
    }
    std::sort(completed.begin(), completed.end());
    for (const int id : completed)
    {
        GuestThread *waiter = thread(id);
        assert(waiter != nullptr);
        const EeVSyncWait wait = std::get<EeVSyncWait>(waiter->wait.payload);
        const int result = wait.fixedResult >= 0
                               ? wait.fixedResult
                               : static_cast<int>((tick - 1u) & 1u);
        makeReady(*waiter, result, false);
    }
    publishSnapshot();
}

void EeScheduler::completeExternalWait(uint32_t type, uint64_t token, int result)
{
    assertExecutor();
    std::vector<int> completed;
    for (const auto &[id, candidate] : m_threads)
    {
        if ((candidate.status != EeThreadStatus::Waiting && candidate.status != EeThreadStatus::WaitingSuspended) ||
            (candidate.wait.reason != EeWaitReason::External &&
             candidate.wait.reason != EeWaitReason::Mpeg))
        {
            continue;
        }
        const auto &external = std::get<EeExternalWait>(candidate.wait.payload);
        if (external.type == type && external.token == token)
        {
            completed.push_back(id);
        }
    }
    std::sort(completed.begin(), completed.end());
    for (const int id : completed)
    {
        GuestThread *waiter = thread(id);
        assert(waiter != nullptr);
        makeReady(*waiter, result, false);
    }
    publishSnapshot();
}

[[noreturn]] void EeScheduler::waitExternal(EeWaitReason reason,
                                            uint32_t type,
                                            uint64_t token,
                                            std::function<void(R5900Context &)> completion)
{
    EeWaitState wait{reason, EeExternalWait{type, token}, std::move(completion)};
    blockCurrent(std::move(wait));
}

GuestThread *EeScheduler::thread(int id)
{
    auto it = m_threads.find(id);
    return it == m_threads.end() ? nullptr : &it->second;
}

const GuestThread *EeScheduler::thread(int id) const
{
    auto it = m_threads.find(id);
    return it == m_threads.end() ? nullptr : &it->second;
}

EeSemaphore *EeScheduler::semaphore(int id)
{
    auto it = m_semaphores.find(id);
    return it == m_semaphores.end() ? nullptr : &it->second;
}

const EeSemaphore *EeScheduler::semaphore(int id) const
{
    auto it = m_semaphores.find(id);
    return it == m_semaphores.end() ? nullptr : &it->second;
}

EeEventFlag *EeScheduler::eventFlag(int id)
{
    auto it = m_eventFlags.find(id);
    return it == m_eventFlags.end() ? nullptr : &it->second;
}

const EeEventFlag *EeScheduler::eventFlag(int id) const
{
    auto it = m_eventFlags.find(id);
    return it == m_eventFlags.end() ? nullptr : &it->second;
}

GuestThread *EeScheduler::currentThread()
{
    if (g_cachedScheduler != this || g_cachedThreadId != m_currentThreadId || g_cachedThread == nullptr)
    {
        g_cachedThread = thread(m_currentThreadId);
        g_cachedThreadId = m_currentThreadId;
        g_cachedScheduler = this;
    }
    return g_cachedThread;
}

const GuestThread *EeScheduler::currentThread() const
{
    return thread(m_currentThreadId);
}

int EeScheduler::currentThreadId() const noexcept
{
    return m_currentThreadId;
}

R5900Context *EeScheduler::currentContext()
{
    GuestThread *self = currentThread();
    return self ? &self->activeContext() : nullptr;
}

uint8_t *EeScheduler::rdram() const noexcept
{
    return m_rdram;
}

void EeScheduler::bindMainContextForSyscall(R5900Context &ctx, uint8_t *rdram)
{
    if (m_executorThread == std::thread::id{})
    {
        reset(rdram, ctx);
        GuestThread *main = selectReady();
        assert(main != nullptr);
        makeRunning(*main);
        return;
    }
    assertExecutor();
    m_rdram = rdram;
    if (m_currentThreadId == 0)
    {
        GuestThread *main = thread(kMainThreadId);
        assert(main != nullptr);
        assert(main->status == EeThreadStatus::Ready);
        removeReady(*main);
        makeRunning(*main);
    }
}

EeKernelSnapshot EeScheduler::snapshot() const
{
    std::lock_guard lock(m_snapshotMutex);
    return m_snapshot;
}

void EeScheduler::publishSnapshot()
{
    EeKernelSnapshot next{};
    next.sequence = ++m_snapshotSequence;
    next.eeCycle = m_eeCycle;
    next.sliceEndCycle = m_sliceEndCycle;
    next.nextEventCycle = m_nextDeadlineCycle.load(std::memory_order_acquire);
    next.runningThreadId = m_currentThreadId;
    next.threads.reserve(m_threads.size());
    for (const auto &[id, item] : m_threads)
    {
        if (id < 0)
        {
            continue;
        }
        EeThreadSnapshot snapshot{};
        snapshot.id = id;
        const R5900Context &context = item.activeContext();
        snapshot.pc = context.pc;
        snapshot.ra = getRegU32(&context, 31);
        snapshot.sp = getRegU32(&context, 29);
        snapshot.contextGp = getRegU32(&context, 28);
        snapshot.entry = item.entry;
        snapshot.stack = item.stack;
        snapshot.stackSize = item.stackSize;
        snapshot.gp = item.gp;
        snapshot.initialPriority = item.initialPriority;
        snapshot.currentPriority = item.currentPriority;
        snapshot.status = item.status;
        snapshot.waitReason = item.wait.reason;
        snapshot.waitId = waitObjectId(item.wait);
        snapshot.suspendCount = item.suspendCount;
        snapshot.wakeupCount = item.wakeupCount;
        snapshot.invocationDepth = static_cast<uint32_t>(item.invocations.size());
        next.threads.push_back(snapshot);
    }
    std::sort(next.threads.begin(), next.threads.end(), [](const auto &left, const auto &right)
              { return left.id < right.id; });
    next.semaphores.reserve(m_semaphores.size());
    for (const auto &[id, item] : m_semaphores)
    {
        next.semaphores.push_back(EeSemaphoreSnapshot{id,
                                                      item.count,
                                                      item.maxCount,
                                                      static_cast<uint32_t>(item.waiters.size())});
    }
    std::sort(next.semaphores.begin(), next.semaphores.end(), [](const auto &left, const auto &right)
              { return left.id < right.id; });
    next.eventFlags.reserve(m_eventFlags.size());
    for (const auto &[id, item] : m_eventFlags)
    {
        next.eventFlags.push_back(EeEventFlagSnapshot{id,
                                                      item.bits,
                                                      item.initBits,
                                                      item.attr,
                                                      static_cast<uint32_t>(item.waiters.size())});
    }
    std::sort(next.eventFlags.begin(), next.eventFlags.end(), [](const auto &left, const auto &right)
              { return left.id < right.id; });
    {
        std::lock_guard lock(m_snapshotMutex);
        m_snapshot = std::move(next);
    }
}

void EeScheduler::assertExecutor() const
{
    assert(m_executorThread == std::this_thread::get_id());
}

int EeScheduler::allocateThreadId()
{
    for (int attempts = 0; attempts <= kLastThreadId - kFirstThreadId; ++attempts)
    {
        const int candidate = m_nextThreadId;
        m_nextThreadId = candidate == kLastThreadId ? kFirstThreadId : candidate + 1;
        if (!m_threads.contains(candidate))
        {
            return candidate;
        }
    }
    return 0;
}

GuestThread &EeScheduler::acquireInvocationThread()
{
    for (auto &[id, candidate] : m_threads)
    {
        if (id < 0 && candidate.status == EeThreadStatus::Dormant && candidate.invocations.empty())
        {
            return candidate;
        }
    }

    GuestThread dispatcher{};
    dispatcher.id = m_nextInvocationThreadId--;
    dispatcher.initialPriority = 0;
    dispatcher.currentPriority = 0;
    dispatcher.status = EeThreadStatus::Dormant;
    return m_threads.emplace(dispatcher.id, std::move(dispatcher)).first->second;
}

void EeScheduler::enqueueReady(GuestThread &item, bool front)
{
    assert(item.currentPriority >= 0 && item.currentPriority < kPriorityCount);
    item.status = EeThreadStatus::Ready;
    auto &queue = m_readyQueues[item.currentPriority];
    if (front)
    {
        queue.push_front(item.id);
    }
    else
    {
        queue.push_back(item.id);
    }
}

void EeScheduler::removeReady(GuestThread &item)
{
    if (item.status != EeThreadStatus::Ready)
    {
        return;
    }
    auto &queue = m_readyQueues[item.currentPriority];
    auto it = std::find(queue.begin(), queue.end(), item.id);
    assert(it != queue.end());
    queue.erase(it);
}

GuestThread *EeScheduler::selectReady()
{
    for (auto &queue : m_readyQueues)
    {
        if (queue.empty())
        {
            continue;
        }
        const int id = queue.front();
        queue.pop_front();
        GuestThread *selected = thread(id);
        assert(selected != nullptr);
        assert(selected->status == EeThreadStatus::Ready);
        return selected;
    }
    return nullptr;
}

void EeScheduler::makeRunning(GuestThread &item)
{
    assert(m_currentThreadId == 0);
    assert(item.status == EeThreadStatus::Ready);
    item.status = EeThreadStatus::Running;
    m_currentThreadId = item.id;
    renewTimeSlice();
}

void EeScheduler::makeDormant(GuestThread &item)
{
    removeReady(item);
    removeFromWaitObject(item);
    item.status = EeThreadStatus::Dormant;
    item.wait = {};
    item.resumeCompletion = {};
    item.suspendCount = 0;
    item.wakeupCount = 0;
    item.invocations.clear();
}

void EeScheduler::removeFromWaitObject(GuestThread &item)
{
    const int id = item.id;
    if (item.wait.reason == EeWaitReason::Semaphore)
    {
        const int objectId = std::get<EeSemaphoreWait>(item.wait.payload).id;
        if (EeSemaphore *object = semaphore(objectId))
        {
            auto it = std::find(object->waiters.begin(), object->waiters.end(), id);
            if (it != object->waiters.end())
            {
                object->waiters.erase(it);
            }
        }
    }
    else if (item.wait.reason == EeWaitReason::EventFlag)
    {
        const int objectId = std::get<EeEventFlagWait>(item.wait.payload).id;
        if (EeEventFlag *object = eventFlag(objectId))
        {
            auto it = std::find(object->waiters.begin(), object->waiters.end(), id);
            if (it != object->waiters.end())
            {
                object->waiters.erase(it);
            }
        }
    }
    item.wait = {};
}

void EeScheduler::blockCurrent(EeWaitState wait)
{
    GuestThread *self = currentThread();
    assert(self != nullptr);
    self->wait = std::move(wait);
    self->status = self->suspendCount == 0 ? EeThreadStatus::Waiting : EeThreadStatus::WaitingSuspended;
    m_currentThreadId = 0;
    publishSnapshot();
    throw EeDispatcherTransfer{};
}

void EeScheduler::makeReady(GuestThread &item, int result, bool interruptSafe)
{
    auto completion = std::move(item.wait.completion);
    item.wait = {};
    setReturnS32(&item.activeContext(), result);
    item.resumeCompletion = std::move(completion);
    if (item.suspendCount != 0)
    {
        item.status = EeThreadStatus::Suspended;
        return;
    }
    enqueueReady(item);
    requestPreemptionIfHigher(item, interruptSafe);
}

void EeScheduler::requestPreemptionIfHigher(const GuestThread &readyThread, bool interruptSafe)
{
    const GuestThread *running = currentThread();
    if (!running || readyThread.currentPriority >= running->currentPriority)
    {
        return;
    }
    m_rescheduleRequested = true;
    if (interruptSafe || m_insideInterrupt)
    {
        m_checkpointPending.store(true, std::memory_order_release);
    }
}

void EeScheduler::applyPendingPreemption()
{
    if (!m_rescheduleRequested)
    {
        return;
    }
    if (m_currentThreadId == 0)
    {
        m_rescheduleRequested = false;
        m_timeSliceExpired = false;
        return;
    }
    GuestThread *self = currentThread();
    assert(self != nullptr);
    enqueueReady(*self, !m_timeSliceExpired);
    m_currentThreadId = 0;
    m_rescheduleRequested = false;
    m_timeSliceExpired = false;
}

void EeScheduler::processPendingEvents()
{
    assertExecutor();
    SCHED_STAT(++g_schedStats.pendingCalls);
    processDueDeadlines();
    const uint32_t timerInterrupts = m_pendingEeTimerInterrupts;
    m_pendingEeTimerInterrupts = 0u;
    for (uint32_t timer = 0u; timer < 4u; ++timer)
    {
        if ((timerInterrupts & (1u << timer)) != 0u)
        {
            SCHED_STAT(++g_schedStats.timerIrqs);
            dispatchIrq(false, 9u + timer);
        }
    }
    if (g_queuedEvents.load(std::memory_order_acquire) != 0u)
    {
        std::deque<EeEvent> pending;
        {
            std::lock_guard lock(m_eventMutex);
            pending.swap(m_events);
            g_queuedEvents.store(0u, std::memory_order_release);
        }
        for (const EeEvent &event : pending)
        {
            processEvent(event);
        }
    }

    {
        // Lock-free: an event posted concurrently re-sets m_checkpointPending after pushing (postEvent), and the
        // re-check below catches one posted between our read and our store.
        const uint64_t nextEventCycle = m_nextDeadlineCycle.load(std::memory_order_acquire);
        const bool cycleEventDue = nextEventCycle != 0u && m_eeCycle >= nextEventCycle;
        const bool pendingWork = g_queuedEvents.load(std::memory_order_acquire) != 0u || cycleEventDue ||
                                 m_stopRequested.load(std::memory_order_acquire);
        m_checkpointPending.store(pendingWork, std::memory_order_release);
        if (!pendingWork && g_queuedEvents.load(std::memory_order_acquire) != 0u)
            m_checkpointPending.store(true, std::memory_order_release);
    }
    applyPendingPreemption();
}

void EeScheduler::processDueDeadlines()
{
    // Fast path: nothing is due before the earliest deadline cycle (kept by updateNextDeadline under the lock).
    {
        const uint64_t next = m_nextDeadlineCycle.load(std::memory_order_acquire);
        if (next == 0u || m_eeCycle < next)
            return;
    }
    for (;;)
    {
        std::vector<ScheduledEvent> due;
        std::chrono::steady_clock::time_point pacingDeadline{};
        {
            std::unique_lock lock(m_eventMutex);
            const auto now = std::chrono::steady_clock::now();
            for (const ScheduledEvent &item : m_deadlines)
            {
                if (item.deadlineCycle <= m_eeCycle &&
                    (pacingDeadline == std::chrono::steady_clock::time_point{} ||
                     item.hostDeadline < pacingDeadline))
                {
                    pacingDeadline = item.hostDeadline;
                }
            }

            if (pacingDeadline == std::chrono::steady_clock::time_point{})
            {
                updateNextDeadline();
                return;
            }

            if (now < pacingDeadline)
            {
                const uint64_t statSleep0 = g_schedStats.on ? __rdtsc() : 0u;
                ps2tl::rec(ps2tl::SLEEP_B, 0u, 0u, static_cast<uint32_t>(m_eeCycle));
                m_eventCv.wait_until(lock, pacingDeadline, [this]()
                                     { return !m_events.empty() ||
                                              m_stopRequested.load(std::memory_order_acquire); });
                ps2tl::rec(ps2tl::SLEEP_E, 0u, m_events.empty() ? 0u : 1u, static_cast<uint32_t>(m_eeCycle)); // b=1: woken by a posted event
                if (g_schedStats.on)
                {
                    g_schedStats.pacingSleepTsc += __rdtsc() - statSleep0;
                    ++g_schedStats.pacingSleeps;
                    if (!g_schedStats.perDispatch)
                        schedStatsReport(); // =3: the report is driven from the (100-200/s) pacing sleeps
                }
                if (!m_events.empty() || m_stopRequested.load(std::memory_order_acquire))
                {
                    updateNextDeadline();
                    return;
                }
            }

            const auto pacedNow = std::chrono::steady_clock::now();
            auto firstFuture = std::partition(m_deadlines.begin(), m_deadlines.end(),
                                              [this, pacedNow](const ScheduledEvent &item)
                                              { return item.deadlineCycle <= m_eeCycle &&
                                                       item.hostDeadline <= pacedNow; });
            due.insert(due.end(),
                       std::make_move_iterator(m_deadlines.begin()),
                       std::make_move_iterator(firstFuture));
            m_deadlines.erase(m_deadlines.begin(), firstFuture);
            updateNextDeadline();
        }

        std::sort(due.begin(), due.end(), [](const ScheduledEvent &left, const ScheduledEvent &right)
                  {
                      if (left.deadlineCycle != right.deadlineCycle)
                      {
                          return left.deadlineCycle < right.deadlineCycle;
                      }
                      if (left.event.type != right.event.type)
                      {
                          return left.event.type < right.event.type;
                      }
                      if (left.event.id != right.event.id)
                      {
                          return left.event.id < right.event.id;
                      }
                      return left.sequence < right.sequence; });

        if (due.empty())
        {
            return;
        }

        for (ScheduledEvent &scheduled : due)
        {
            if (scheduled.event.type == EeEventType::VBlankStart)
            {
                scheduleEvent(scheduled.deadlineCycle + kVBlankDurationCycles,
                              scheduled.hostDeadline + kVBlankDuration,
                              EeEvent{EeEventType::VBlankEnd, 0, m_vsyncTick + 1u});
                scheduleEvent(scheduled.deadlineCycle + vblankPeriodCycles(),
                              scheduled.hostDeadline + vblankPeriod(),
                              EeEvent{EeEventType::VBlankStart, 0, 0});
            }
            processEvent(scheduled.event);
        }
    }
}

void EeScheduler::processEvent(const EeEvent &event)
{
    SCHED_STAT(++g_schedStats.events);
    switch (event.type)
    {
    case EeEventType::Stop:
        requestStop();
        break;
    case EeEventType::VBlankStart:
        ++m_vsyncTick;
        ps2tl::rec(ps2tl::VBLANK, static_cast<uint32_t>(m_vsyncTick), 0u, static_cast<uint32_t>(m_eeCycle));
        m_runtime.memory().gs().vsyncTick.store(m_vsyncTick, std::memory_order_release);
        if ((m_vsyncTick & 1u) != 0u)
        {
            m_runtime.memory().gs().csr.fetch_or(0x2000ull, std::memory_order_acq_rel);
        }
        else
        {
            m_runtime.memory().gs().csr.fetch_and(~0x2000ull, std::memory_order_acq_rel);
        }
        writeGuestU32(m_vsyncFlagAddress, 1u);
        if (m_vsyncTickAddress != 0u)
        {
            const uint32_t physical = m_vsyncTickAddress & 0x1FFFFFFFu;
            if (m_rdram && physical <= PS2_RAM_SIZE - sizeof(uint64_t))
            {
                std::memcpy(m_rdram + physical, &m_vsyncTick, sizeof(m_vsyncTick));
            }
        }
        m_vsyncFlagAddress = 0u;
        m_vsyncTickAddress = 0u;
        // Threaded VIF1/GIF: the frame's GIF packets must all be in the host GS before its vsync marker: wait for the
        // worker, unless the host's vsync hook orders itself behind the worker's queue (PS2HostGs::vsyncOrdered). The
        // worker's D/T-stop enables follow the guest's FBRST, sampled here.
        if (ps2Vif1Active())
        {
            if (!ps2HostGs().vsyncOrdered || ps2Vif1ForceVsyncBarrier())
                ps2Vif1Barrier(Vif1BarrierReason::Vsync);
            if (const R5900Context *fbrstCtx = currentContext())
                ps2Vif1SetFbrst(fbrstCtx->vu0_fbrst);
        }
        if (const auto hostVsync = ps2HostGs().vsync)
            hostVsync(m_vsyncTick, (m_vsyncTick & 1u) != 0u ? 1 : 0);
        completeVSync(m_vsyncTick);
        if (m_gsVSyncCallback != 0u && m_runtime.hasFunction(m_gsVSyncCallback))
        {
            GuestInvocation invocation{};
            invocation.kind = GuestInvocationKind::GsCallback;
            invocation.context.pc = m_gsVSyncCallback;
            SET_GPR_U32(&invocation.context, 4, static_cast<uint32_t>(m_vsyncTick));
            SET_GPR_U32(&invocation.context, 28, m_gsVSyncCallbackGp);
            SET_GPR_U32(&invocation.context, 29, m_gsVSyncCallbackSp);
            SET_GPR_U32(&invocation.context, 31, 0u);
            queueInvocation(std::move(invocation));
        }
        dispatchIrq(false, 2u);
        break;
    case EeEventType::ExternalWake:
        completeExternalWait(event.id, event.value, KE_OK);
        break;
    case EeEventType::VBlankEnd:
        dispatchIrq(false, 3u);
        break;
    case EeEventType::Dmac:
        // Wake-up for work the IOP thread posted (threaded IOP mode): apply it here, on the EE thread.
        if (event.id == kIopPostedWorkEventId)
            ps2IopDrainPosted(m_runtime);
        else if (event.id == kVif1WorkerEventId)
            ps2Vif1ApplyEvent(m_runtime); // VIF1/GIF worker finished a DMA job: CHCR/D_STAT/DMAC handlers
        break;
    case EeEventType::Alarm:
    {
        auto it = m_alarms.find(static_cast<int>(event.id));
        if (it == m_alarms.end())
        {
            break;
        }
        const EeAlarm alarm = it->second;
        m_alarms.erase(it);
        GuestInvocation invocation{};
        invocation.kind = GuestInvocationKind::Alarm;
        invocation.context.pc = alarm.handler;
        SET_GPR_U32(&invocation.context, 4, static_cast<uint32_t>(alarm.id));
        SET_GPR_U32(&invocation.context, 5, static_cast<uint32_t>(alarm.ticks));
        SET_GPR_U32(&invocation.context, 6, alarm.argument);
        SET_GPR_U32(&invocation.context, 28, alarm.gp);
        SET_GPR_U32(&invocation.context, 29, 0u);
        SET_GPR_U32(&invocation.context, 31, 0u);
        queueInvocation(std::move(invocation));
        break;
    }
    }
}

void EeScheduler::finishEventWaiters(EeEventFlag &flag, bool interruptSafe)
{
    for (auto it = flag.waiters.begin(); it != flag.waiters.end();)
    {
        GuestThread *waiter = thread(*it);
        assert(waiter != nullptr);
        const EeEventFlagWait wait = std::get<EeEventFlagWait>(waiter->wait.payload);
        if (!eventCondition(flag.bits, wait.bits, wait.mode))
        {
            ++it;
            continue;
        }
        const uint32_t observed = flag.bits;
        writeGuestU32(wait.resultAddress, observed);
        if ((wait.mode & WEF_CLEAR_ALL) != 0u)
        {
            flag.bits = 0;
        }
        else if ((wait.mode & WEF_CLEAR) != 0u)
        {
            flag.bits &= ~wait.bits;
        }
        it = flag.waiters.erase(it);
        makeReady(*waiter, KE_OK, interruptSafe);
    }
}

bool EeScheduler::eventCondition(uint32_t current, uint32_t requested, uint32_t mode)
{
    return (mode & WEF_OR) != 0u ? (current & requested) != 0u
                                 : (current & requested) == requested;
}

int EeScheduler::waitObjectId(const EeWaitState &wait)
{
    switch (wait.reason)
    {
    case EeWaitReason::Semaphore:
        return std::get<EeSemaphoreWait>(wait.payload).id;
    case EeWaitReason::EventFlag:
        return std::get<EeEventFlagWait>(wait.payload).id;
    default:
        return 0;
    }
}

void EeScheduler::writeGuestU32(uint32_t address, uint32_t value)
{
    if (address == 0u)
    {
        return;
    }
    const uint32_t physical = address & 0x1FFFFFFFu;
    if (!m_rdram || physical > PS2_RAM_SIZE - sizeof(value))
    {
        return;
    }
    std::memcpy(m_rdram + physical, &value, sizeof(value));
}

void EeScheduler::waitForEvent()
{
    std::unique_lock lock(m_eventMutex);
    if (!m_events.empty() || m_stopRequested.load(std::memory_order_acquire))
    {
        return;
    }
    const uint64_t timerCycles = m_runtime.memory().cyclesUntilNextEeTimerInterrupt();
    const bool hasTimerDeadline = timerCycles != std::numeric_limits<uint64_t>::max();
    if (m_deadlines.empty() && !hasTimerDeadline)
    {
        m_eventCv.wait(lock, [this]()
                       { return !m_events.empty() || m_stopRequested.load(std::memory_order_acquire); });
        return;
    }

    uint64_t deadlineCycle = 0u;
    auto hostDeadline = std::chrono::steady_clock::time_point::max();
    if (!m_deadlines.empty())
    {
        const auto next = std::min_element(m_deadlines.begin(), m_deadlines.end(),
                                           [](const ScheduledEvent &left, const ScheduledEvent &right)
                                           {
                                               if (left.deadlineCycle != right.deadlineCycle)
                                               {
                                                   return left.deadlineCycle < right.deadlineCycle;
                                               }
                                               return left.sequence < right.sequence;
                                           });
        deadlineCycle = next->deadlineCycle;
        hostDeadline = next->hostDeadline;
    }
    if (hasTimerDeadline)
    {
        const auto timerHostDeadline = std::chrono::steady_clock::now() + eeCyclesToHostDuration(timerCycles);
        if (timerHostDeadline < hostDeadline)
        {
            deadlineCycle = m_eeCycle + timerCycles;
            hostDeadline = timerHostDeadline;
        }
    }

    ps2tl::rec(ps2tl::SLEEP_B, 1u, 0u, static_cast<uint32_t>(m_eeCycle));
    const bool signaled = m_eventCv.wait_until(lock, hostDeadline, [this]()
                                               { return !m_events.empty() ||
                                                        m_stopRequested.load(std::memory_order_acquire); });
    ps2tl::rec(ps2tl::SLEEP_E, 1u, signaled ? 1u : 0u, static_cast<uint32_t>(m_eeCycle));
    if (!signaled)
    {
        const uint64_t elapsed = deadlineCycle > m_eeCycle ? deadlineCycle - m_eeCycle : 0u;
        lock.unlock();
        uint64_t remaining = elapsed;
        while (remaining > 0u)
        {
            const uint32_t step = static_cast<uint32_t>(std::min<uint64_t>(remaining, std::numeric_limits<uint32_t>::max()));
            accountCycles(step);
            remaining -= step;
        }
        m_checkpointPending.store(true, std::memory_order_release);
    }
}

void EeScheduler::scheduleEvent(uint64_t deadlineCycle,
                                std::chrono::steady_clock::time_point hostDeadline,
                                EeEvent event)
{
    {
        std::lock_guard lock(m_eventMutex);
        m_deadlines.push_back(ScheduledEvent{deadlineCycle, hostDeadline, event, ++m_eventSequence});
        updateNextDeadline();
    }
    m_eventCv.notify_one();
}

void EeScheduler::updateNextDeadline()
{
    if (m_deadlines.empty())
    {
        m_nextDeadlineCycle.store(0u, std::memory_order_release);
        return;
    }
    const auto it = std::min_element(m_deadlines.begin(), m_deadlines.end(),
                                     [](const ScheduledEvent &left, const ScheduledEvent &right)
                                     {
                                         if (left.deadlineCycle != right.deadlineCycle)
                                         {
                                             return left.deadlineCycle < right.deadlineCycle;
                                         }
                                         return left.sequence < right.sequence;
                                     });
    m_nextDeadlineCycle.store(it->deadlineCycle, std::memory_order_release);
}

bool EeScheduler::hasReadyAtOrAbovePriority(int priority) const
{
    const int last = std::clamp(priority, 0, kPriorityCount - 1);
    for (int p = 0; p <= last; ++p)
    {
        if (!m_readyQueues[static_cast<size_t>(p)].empty())
        {
            return true;
        }
    }
    return false;
}

void EeScheduler::renewTimeSlice()
{
    m_sliceEndCycle = m_eeCycle + kDefaultTimeSliceCycles;
    m_timeSliceExpired = false;
}

void EeScheduler::copyMainContextToRuntime()
{
    const GuestThread *main = thread(kMainThreadId);
    if (main)
    {
        m_runtime.m_cpuContext = main->context;
    }
}

void EeScheduler::publishDebugContext(const R5900Context &context)
{
    m_runtime.m_debugPc.store(context.pc, std::memory_order_relaxed);
    m_runtime.m_debugRa.store(getRegU32(&context, 31), std::memory_order_relaxed);
    m_runtime.m_debugSp.store(getRegU32(&context, 29), std::memory_order_relaxed);
    m_runtime.m_debugGp.store(getRegU32(&context, 28), std::memory_order_relaxed);
}

void EeScheduler::publishIdleDebugContext()
{
    // Temporary IRQ/RPC/alarm invocations deliberately return to PC=0. Once
    // the scheduler is idle, show a real EE thread context instead of leaving
    // the debugger pinned to that completed dispatcher frame.
    const GuestThread *selected = thread(kMainThreadId);
    if (!selected)
    {
        for (const auto &[id, candidate] : m_threads)
        {
            if (id > 0 && candidate.status != EeThreadStatus::Dormant)
            {
                selected = &candidate;
                break;
            }
        }
    }

    if (selected)
    {
        publishDebugContext(selected->activeContext());
    }
}

void ps2SetVblankPeriodMicros(uint32_t micros)
{
    g_vblankPeriodMicros.store(micros < 1000u ? 1000u : micros, std::memory_order_relaxed);
    (void)g_eeCycleScaleInit;
    if (g_eeCycleScaleAuto)
    {
        const double s = std::clamp(static_cast<double>(micros < 1000u ? 1000u : micros) / 16667.0, 0.125, 1.0);
        g_eeCycleScaleQ16.store(static_cast<uint32_t>(s * 65536.0), std::memory_order_relaxed);
    }
    const uint32_t q = g_eeCycleScaleQ16.load(std::memory_order_relaxed);
    if (q != 65536u)
        std::fprintf(stderr, "[sched] EE cycle clock scale %.3f (%s)\n", q / 65536.0,
                     g_eeCycleScaleAuto ? "auto: 60 Hz / vblank rate; PS2X_EE_CYCLE_SCALE=1 restores the old clock" : "PS2X_EE_CYCLE_SCALE");
}

uint32_t ps2VblankPeriodMicros()
{
    return g_vblankPeriodMicros.load(std::memory_order_relaxed);
}
