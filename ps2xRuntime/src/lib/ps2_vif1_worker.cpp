#include "ps2_vif1_worker.h"
#include "runtime/ps2_timeline.h"
#include <cstring>
#include <unordered_map>

#include "ps2_runtime.h"
#include "runtime/ee_scheduler.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/ps2_host_gs.h"
#include "runtime/ps2_host_vu.h"
#include "runtime/ps2_memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <immintrin.h>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
    using Clock = std::chrono::steady_clock;

    constexpr uint32_t kVif1Channel = 0x10009000u;
    constexpr uint32_t kGifChannel = 0x1000A000u;
    constexpr uint32_t kDStat = 0x1000E010u;

    struct Job
    {
        std::vector<std::vector<uint8_t>> gif;  // PATH3 packets, submitted (undrained) first, as the synchronous code did
        std::vector<std::vector<uint8_t>> vif1; // buffers for processVIF1Data, in order
        bool hadGif = false;
        bool hadVif1 = false;
        uint8_t kind = 1;     // timeline: 1 DMA job, 2 call (MSCAL/MSCNT from the EE), 3 GS half of a vblank
        uint32_t bytes = 0;   // timeline
        std::function<void()> call; // if set: run this on the worker instead (MSCAL from the EE thread)
    };

    struct Completion
    {
        bool hadGif;
        bool hadVif1;
    };

    struct Stats
    {
        bool enabled = false;
        std::atomic<uint64_t> jobs{0};
        std::atomic<uint64_t> jobBytes{0};
        std::atomic<uint64_t> dispatchNs{0}; // EE: transfer -> job conversion + enqueue
        std::atomic<uint64_t> walkNs{0};     // EE: DMA chain walk + snapshot copy (in the DMA-start store)
        std::atomic<uint64_t> mpgs{0};       // MPG uploads
        std::atomic<uint64_t> mpgsSame{0};   // ... that did not change VU1 code memory
        std::atomic<uint64_t> workerNs{0};   // worker: time inside jobs
        std::atomic<uint64_t> vuNs{0};       // worker: time inside the host VU1 (MSCAL/MSCNT, includes XGKICK -> GS)
        std::atomic<uint64_t> gifNs{0};      // worker: time inside the host GS packet hook
        std::atomic<uint64_t> barriers[static_cast<size_t>(Vif1BarrierReason::Count)]{};
        std::atomic<uint64_t> barrierWaits[static_cast<size_t>(Vif1BarrierReason::Count)]{};
        std::atomic<uint64_t> barrierNs[static_cast<size_t>(Vif1BarrierReason::Count)]{};
        std::atomic<uint64_t> busyReads{0};
        std::atomic<uint64_t> syncFallbacks{0};
    };

    struct Worker
    {
        std::atomic<bool> active{false};
        PS2Runtime *runtime = nullptr;
        PS2Memory *memory = nullptr;
        bool disabledByEnv = false;
        Stats stats;

        std::thread thread;
        bool started = false;
        uint32_t mxcsr = 0;

        std::mutex mutex; // queue + idle signalling
        std::condition_variable cv;
        std::condition_variable idleCv;
        std::deque<Job> queue;
        bool stop = false;
        std::atomic<uint32_t> pending{0};     // jobs in the queue (written under mutex, read by the spinning worker)
        uint64_t enqueued = 0;                // EE thread only (written under mutex)
        std::atomic<uint64_t> finished{0};    // worker

        std::mutex doneMutex;
        std::vector<Completion> done;
        std::atomic<uint64_t> doneCount{0}; // completions pushed by the worker
        uint64_t applied = 0;               // EE thread only: completions applied
        int outstanding[3] = {0, 0, 0};     // EE thread only, by channel index (1 VIF1, 2 GIF)

        std::atomic<uint32_t> fbrst{0};
        uint32_t vpuStat = 0; // worker only: sink for the host VU1's VPU_STAT bits
        uint32_t delayUs = 0; // PS2X_VIF1_DELAY_US: sleep before every job (stress test)
    };

    Worker g;
    thread_local bool t_onWorker = false;

    uint64_t nowNs()
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
    }

    void raiseDStat(PS2Memory &m, uint32_t channelBit)
    {
        uint32_t dstat = m.m_ioRegisters.count(kDStat) ? m.m_ioRegisters[kDStat] : 0u;
        dstat |= (1u << channelBit);
        const uint32_t status = dstat & 0x3FFu;
        const uint32_t mask = (dstat >> 16) & 0x3FFu;
        if ((status & mask) != 0u)
            dstat |= (1u << 31);
        else
            dstat &= ~(1u << 31);
        m.m_ioRegisters[kDStat] = dstat;
    }

    void printStats(uint64_t elapsedMs)
    {
        Stats &s = g.stats;
        std::fprintf(stderr,
                     "[vif1-thread] t=%llums jobs=%llu bytes=%llu mpg=%llu(same %llu) walk=%.2fms dispatch=%.2fms worker=%.2fms (vu1 %.2fms, hostgs %.2fms) busyReads=%llu syncFallbacks=%llu barriers:",
                     static_cast<unsigned long long>(elapsedMs), static_cast<unsigned long long>(s.jobs.exchange(0)),
                     static_cast<unsigned long long>(s.jobBytes.exchange(0)),
                     static_cast<unsigned long long>(s.mpgs.exchange(0)),
                     static_cast<unsigned long long>(s.mpgsSame.exchange(0)), s.walkNs.exchange(0) / 1e6,
                     s.dispatchNs.exchange(0) / 1e6,
                     s.workerNs.exchange(0) / 1e6, s.vuNs.exchange(0) / 1e6, s.gifNs.exchange(0) / 1e6,
                     static_cast<unsigned long long>(s.busyReads.exchange(0)),
                     static_cast<unsigned long long>(s.syncFallbacks.exchange(0)));
        static const char *names[] = {"vu1mem", "vifreg", "gifstat", "viffifo", "gifsubmit", "sync", "vsync", "other"};
        for (size_t i = 0; i < static_cast<size_t>(Vif1BarrierReason::Count); ++i)
        {
            const uint64_t n = s.barriers[i].exchange(0);
            const uint64_t w = s.barrierWaits[i].exchange(0);
            const uint64_t ns = s.barrierNs[i].exchange(0);
            if (n)
                std::fprintf(stderr, " %s=%llu(wait %llu, %.2fms)", names[i], static_cast<unsigned long long>(n),
                             static_cast<unsigned long long>(w), ns / 1e6);
        }
        std::fprintf(stderr, "\n");
    }

    void runJob(Job &job)
    {
        PS2Memory &m = *g.memory;
        if (job.call)
        {
            job.call();
            return;
        }
        for (const auto &buf : job.gif)
            m.submitGifPacket(GifPathId::Path3, buf.data(), static_cast<uint32_t>(buf.size()), false);
        for (const auto &buf : job.vif1)
            m.processVIF1Data(buf.data(), static_cast<uint32_t>(buf.size()));
        if (m.m_gifArbiter)
            m.m_gifArbiter->drain();
    }

    void workerMain()
    {
        t_onWorker = true;
        _mm_setcsr(g.mxcsr);
        ps2Vif1TuneWorkerThread();
        if (const auto bind = ps2HostVu1().bindWorkerThread)
            bind();
        const uint64_t statsStart = nowNs();
        uint64_t lastPrint = statsStart;
        for (;;)
        {
            Job job;
            // The game often waits for a channel to go idle before it starts the next DMA (loading screens do this
            // hundreds of times a second), so a job's latency matters: spin briefly for the next job before sleeping.
            for (int spin = 0; spin < 4000 && g.pending.load(std::memory_order_acquire) == 0; ++spin)
                _mm_pause();
            {
                std::unique_lock<std::mutex> lock(g.mutex);
                g.cv.wait(lock, [] { return g.stop || !g.queue.empty(); });
                if (g.queue.empty())
                    return; // stop requested and nothing left
                job = std::move(g.queue.front());
                g.queue.pop_front();
                g.pending.fetch_sub(1, std::memory_order_release);
            }
            if (g.delayUs)
            {
                // stress test: a slow worker (busy wait: sleep_for granularity is the 15.6 ms timer tick)
                const uint64_t until = nowNs() + static_cast<uint64_t>(g.delayUs) * 1000ull;
                while (nowNs() < until)
                    _mm_pause();
            }
            const uint64_t t0 = g.stats.enabled ? nowNs() : 0u;
            ps2tl::rec(ps2tl::JOB_B, job.kind, job.bytes);
            try
            {
                runJob(job);
            }
            catch (const std::exception &e)
            {
                std::fprintf(stderr, "[vif1-thread] job failed: %s\n", e.what());
            }
            catch (...)
            {
                std::fprintf(stderr, "[vif1-thread] job failed (unknown exception)\n");
            }
            if (g.stats.enabled)
            {
                const uint64_t t1 = nowNs();
                g.stats.workerNs.fetch_add(t1 - t0, std::memory_order_relaxed);
                if (t1 - lastPrint >= 10'000'000'000ull)
                {
                    lastPrint = t1;
                    printStats((t1 - statsStart) / 1'000'000ull);
                }
            }
            ps2tl::rec(ps2tl::JOB_E, job.kind, job.bytes);
            for (auto &buf : job.gif)
                ps2ChainBufferRecycle(std::move(buf));
            for (auto &buf : job.vif1)
                ps2ChainBufferRecycle(std::move(buf));
            job.gif.clear();
            job.vif1.clear();
            const bool completes = job.hadGif || job.hadVif1;
            if (completes)
            {
                {
                    std::lock_guard<std::mutex> lock(g.doneMutex);
                    g.done.push_back(Completion{job.hadGif, job.hadVif1});
                }
                g.doneCount.fetch_add(1, std::memory_order_release);
            }
            {
                std::lock_guard<std::mutex> lock(g.mutex);
                g.finished.fetch_add(1, std::memory_order_release);
            }
            g.idleCv.notify_all();
            if (completes && g.runtime)
            {
                ps2tl::rec(ps2tl::DONE_POST, job.hadVif1 ? 1u : 0u, job.hadGif ? 1u : 0u);
                g.runtime->postEeEvent(EeEvent{EeEventType::Dmac, kVif1WorkerEventId, 0u});
            }
        }
    }

    void ensureStarted()
    {
        if (g.started)
            return;
        g.started = true;
        g.mxcsr = _mm_getcsr(); // the game thread's FTZ/DAZ setting
        g.thread = std::thread(workerMain);
        std::atexit([] { ps2Vif1WorkerShutdown(); });
    }

    void enqueue(Job &&job)
    {
        ensureStarted();
        {
            std::lock_guard<std::mutex> lock(g.mutex);
            g.queue.push_back(std::move(job));
            ++g.enqueued;
            g.pending.fetch_add(1, std::memory_order_release);
        }
        g.cv.notify_one();
    }

    // EE thread. Applies finished jobs: what the synchronous processPendingTransfers() did after the data was consumed.
    void applyCompletions(uint32_t source)
    {
        std::vector<Completion> local;
        {
            std::lock_guard<std::mutex> lock(g.doneMutex);
            local.swap(g.done);
        }
        if (local.empty())
            return;
        ps2tl::Span tlSpan(ps2tl::APPLY_B, ps2tl::APPLY_E, static_cast<uint32_t>(local.size()), source);
        PS2Memory &m = *g.memory;
        for (const Completion &c : local)
        {
            ++g.applied;
            if (c.hadGif)
            {
                raiseDStat(m, 2u);
                m.queueCompletedDmacCause(2u);
                if (--g.outstanding[2] <= 0)
                {
                    g.outstanding[2] = 0;
                    m.m_ioRegisters[kGifChannel + 0x00u] &= ~0x100u;
                    m.m_ioRegisters[kGifChannel + 0x20u] = 0u;
                }
            }
            if (c.hadVif1)
            {
                raiseDStat(m, 1u);
                m.queueCompletedDmacCause(1u);
                if (--g.outstanding[1] <= 0)
                {
                    g.outstanding[1] = 0;
                    m.m_ioRegisters[kVif1Channel + 0x00u] &= ~0x100u;
                    m.m_ioRegisters[kVif1Channel + 0x20u] = 0u;
                }
            }
        }
    }

    // Splits a normal-mode (MADR/QWC) transfer into the chunks the synchronous code processed one by one, copying the
    // source now: the game may overwrite the buffer after the DMA start.
    void snapshotNormal(PS2Memory &m, const PS2Memory::PendingTransfer &p, bool gifMode, uint64_t &bytesOut,
                        std::vector<std::vector<uint8_t>> &out)
    {
        if (p.qwc == 0)
            return;
        const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
        const uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
        uint32_t srcPhys = 0;
        try
        {
            srcPhys = m.translateAddress(p.srcAddr);
        }
        catch (const std::exception &)
        {
            return;
        }
        const uint8_t *base = p.fromScratchpad ? m.m_scratchpad : m.m_rdram;
        const uint32_t limit = p.fromScratchpad ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
        uint32_t left = sizeBytes;
        while (gifMode ? left >= 16u : left > 0u)
        {
            if (srcPhys >= limit)
                srcPhys = 0;
            uint32_t chunk = left;
            if (srcPhys + chunk > limit)
                chunk = limit - srcPhys;
            if (chunk == 0)
                break;
            out.emplace_back(base + srcPhys, base + srcPhys + chunk);
            bytesOut += chunk;
            if (gifMode)
            {
                m.m_seenGifCopy = true;
                m.m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
            }
            left -= chunk;
            srcPhys += chunk;
        }
    }
}

void ps2Vif1WorkerBind(PS2Runtime &runtime, PS2Memory &memory)
{
    // Rebinding (a new memory image) stops the previous worker first.
    ps2Vif1WorkerShutdown();
    const char *v = std::getenv("PS2X_VIF1_THREAD");
    g.disabledByEnv = v && *v == '0';
    const PS2HostVu1 &vu = ps2HostVu1();
    const bool hostVu = vu.mscal != nullptr && vu.mscnt != nullptr;
    g.runtime = &runtime;
    g.memory = &memory;
    const char *dl = std::getenv("PS2X_VIF1_DELAY_US");
    g.delayUs = dl ? static_cast<uint32_t>(std::strtoul(dl, nullptr, 10)) : 0u;
    const char *st = std::getenv("PS2X_VIF1_STATS");
    g.stats.enabled = st && *st && *st != '0';
    g.active.store(!g.disabledByEnv && hostVu, std::memory_order_release);
    std::fprintf(stderr, "[vif1-thread] %s%s\n", g.active.load() ? "threaded VIF1/VU1/GIF" : "synchronous VIF1/VU1/GIF",
                 g.disabledByEnv ? " (PS2X_VIF1_THREAD=0)" : (hostVu ? "" : " (no host VU1)"));
}

bool ps2Vif1Active()
{
    return g.active.load(std::memory_order_relaxed);
}

bool ps2Vif1ForceVsyncBarrier()
{
    static const bool force = []
    {
        const char *v = std::getenv("PS2X_VIF1_VSYNC_SYNC");
        return v && *v && *v != '0';
    }();
    return force;
}

bool ps2Vif1OnWorkerThread()
{
    return t_onWorker;
}

bool ps2Vif1DispatchAsync(PS2Memory &m)
{
    if (!g.active.load(std::memory_order_relaxed) || t_onWorker)
        return false;
    if (!m.m_pendingVif0Transfers.empty())
    {
        // VIF0 (VU0 memory, EE thread) keeps the synchronous path: it also drains the arbiter, so it needs the worker idle.
        if (g.stats.enabled)
            g.stats.syncFallbacks.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (m.m_pendingGifTransfers.empty() && m.m_pendingVif1Transfers.empty())
        return true;

    const uint64_t t0 = g.stats.enabled ? nowNs() : 0u;

    // GIF_STAT.FQC as the synchronous code reported it (EE-side register).
    uint32_t observedGifQwc = 0u;
    for (const auto &transfer : m.m_pendingGifTransfers)
    {
        const uint64_t transferQwc = !transfer.chainData.empty() ? (transfer.chainData.size() / 16u) : transfer.qwc;
        observedGifQwc = static_cast<uint32_t>(std::min<uint64_t>(16u, static_cast<uint64_t>(observedGifQwc) + transferQwc));
    }
    if (observedGifQwc != 0u)
    {
        constexpr uint32_t kGifStat = 0x10003020u;
        constexpr uint32_t kGifFqcMask = 0x1F000000u;
        uint32_t &gifStat = m.m_ioRegisters[kGifStat];
        gifStat = (gifStat & ~kGifFqcMask) | (observedGifQwc << 24u);
    }

    Job job;
    uint64_t bytes = 0;
    job.hadGif = !m.m_pendingGifTransfers.empty();
    job.hadVif1 = !m.m_pendingVif1Transfers.empty();
    for (auto &p : m.m_pendingGifTransfers)
    {
        if (!p.chainData.empty())
        {
            m.m_seenGifCopy = true;
            m.m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
            bytes += p.chainData.size();
            job.gif.push_back(std::move(p.chainData));
        }
        else if (p.qwc > 0)
            snapshotNormal(m, p, true, bytes, job.gif);
    }
    for (auto &p : m.m_pendingVif1Transfers)
    {
        if (!p.chainData.empty())
        {
            bytes += p.chainData.size();
            job.vif1.push_back(std::move(p.chainData));
        }
        else if (p.qwc > 0)
            snapshotNormal(m, p, false, bytes, job.vif1);
    }
    m.m_pendingGifTransfers.clear();
    m.m_pendingVif1Transfers.clear();
    job.bytes = static_cast<uint32_t>(bytes);

    if (job.hadGif)
        ++g.outstanding[2];
    if (job.hadVif1)
        ++g.outstanding[1];
    ps2tl::rec(ps2tl::D1_KICK, static_cast<uint32_t>(bytes), (job.hadGif ? 2u : 0u) | (job.hadVif1 ? 1u : 0u),
               ps2tl::on() && g.runtime ? static_cast<uint32_t>(g.runtime->eeScheduler().eeCycleForTrace()) : 0u);
    enqueue(std::move(job));

    if (g.stats.enabled)
    {
        g.stats.jobs.fetch_add(1, std::memory_order_relaxed);
        g.stats.jobBytes.fetch_add(bytes, std::memory_order_relaxed);
        g.stats.dispatchNs.fetch_add(nowNs() - t0, std::memory_order_relaxed);
    }
    return true;
}

void ps2Vif1Barrier(Vif1BarrierReason reason)
{
    if (!g.active.load(std::memory_order_relaxed) || t_onWorker || !g.started)
        return;
    const size_t ri = static_cast<size_t>(reason);
    if (g.stats.enabled)
        g.stats.barriers[ri].fetch_add(1, std::memory_order_relaxed);
    const uint64_t target = g.enqueued;
    if (g.finished.load(std::memory_order_acquire) != target)
    {
        const uint64_t t0 = g.stats.enabled ? nowNs() : 0u;
        ps2tl::rec(ps2tl::BARRIER_B, static_cast<uint32_t>(reason));
        for (int spin = 0; spin < 2000 && g.finished.load(std::memory_order_acquire) != target; ++spin)
            _mm_pause();
        if (g.finished.load(std::memory_order_acquire) != target)
        {
            std::unique_lock<std::mutex> lock(g.mutex);
            g.idleCv.wait(lock, [target] { return g.finished.load(std::memory_order_acquire) == target; });
        }
        ps2tl::rec(ps2tl::BARRIER_E, static_cast<uint32_t>(reason));
        if (g.stats.enabled)
        {
            g.stats.barrierWaits[ri].fetch_add(1, std::memory_order_relaxed);
            g.stats.barrierNs[ri].fetch_add(nowNs() - t0, std::memory_order_relaxed);
        }
    }
    applyCompletions(2u);
}

void ps2Vif1PollCompletions()
{
    if (!g.active.load(std::memory_order_relaxed) || t_onWorker || !g.started)
        return;
    if (g.doneCount.load(std::memory_order_acquire) == g.applied)
        return;
    applyCompletions(3u);
}

bool ps2Vif1ChannelBusy(uint32_t channelBase)
{
    if (!g.active.load(std::memory_order_relaxed) || !g.started)
        return false;
    const int idx = channelBase == kVif1Channel ? 1 : (channelBase == kGifChannel ? 2 : 0);
    if (idx == 0)
        return false;
    const bool busy = g.outstanding[idx] > 0;
    if (ps2tl::on())
    {
        static bool lastBusy[3] = {false, false, false}; // EE thread only
        if (busy != lastBusy[idx])
        {
            lastBusy[idx] = busy;
            ps2tl::rec(busy ? ps2tl::CHCR_BUSY : ps2tl::CHCR_IDLE, static_cast<uint32_t>(idx));
        }
    }
    if (busy && g.stats.enabled)
        g.stats.busyReads.fetch_add(1, std::memory_order_relaxed);
    return busy;
}

void ps2Vif1ApplyEvent(PS2Runtime &runtime)
{
    if (!g.active.load(std::memory_order_relaxed) || !g.memory)
        return;
    applyCompletions(1u);
    runtime.drainCompletedDmacHandlers(g.memory->getRDRAM());
}

void ps2Vif1SetFbrst(uint32_t fbrst)
{
    g.fbrst.store(fbrst, std::memory_order_relaxed);
}

namespace
{
    bool runOnWorker(std::function<void()> fn)
    {
        if (!g.active.load(std::memory_order_relaxed))
            return false;
        if (t_onWorker)
        {
            fn();
            return true;
        }
        Job job;
        job.kind = 2;
        job.call = std::move(fn);
        enqueue(std::move(job));
        ps2Vif1Barrier(Vif1BarrierReason::Other);
        return true;
    }
}

bool ps2Vif1StatsEnabled()
{
    return g.stats.enabled;
}

uint64_t ps2Vif1NowNs()
{
    return nowNs();
}

void ps2Vif1StatMpg(bool same)
{
    g.stats.mpgs.fetch_add(1, std::memory_order_relaxed);
    if (same)
        g.stats.mpgsSame.fetch_add(1, std::memory_order_relaxed);
}

void ps2Vif1StatAddWalkNs(uint64_t ns)
{
    g.stats.walkNs.fetch_add(ns, std::memory_order_relaxed);
}

namespace
{
    struct ChainPool
    {
        std::mutex mutex;
        std::vector<std::vector<uint8_t>> free;
        std::atomic<size_t> lastBytes[3] = {}; // by channel: 0 VIF0, 1 VIF1, 2 GIF
    };
    ChainPool g_chainPool;

    // PS2X_CHAIN_POOL=0: allocate every chain buffer from scratch (the pre-pool behaviour), for A/B measurements.
    bool chainPoolEnabled()
    {
        static const bool on = []
        {
            const char *v = std::getenv("PS2X_CHAIN_POOL");
            return !(v && *v == '0');
        }();
        return on;
    }

    int chainChannelIndex(uint32_t channelBase)
    {
        return channelBase == 0x10009000u ? 1 : (channelBase == 0x1000A000u ? 2 : 0);
    }
}

std::vector<uint8_t> ps2ChainBufferAcquire(uint32_t channelBase)
{
    std::vector<uint8_t> buf;
    if (!chainPoolEnabled())
        return buf;
    {
        std::lock_guard<std::mutex> lock(g_chainPool.mutex);
        auto &pool = g_chainPool.free;
        if (!pool.empty())
        {
            size_t best = 0;
            for (size_t i = 1; i < pool.size(); ++i)
                if (pool[i].capacity() > pool[best].capacity())
                    best = i;
            buf = std::move(pool[best]);
            pool[best] = std::move(pool.back());
            pool.pop_back();
        }
    }
    buf.clear();
    const size_t hint = g_chainPool.lastBytes[chainChannelIndex(channelBase)].load(std::memory_order_relaxed);
    const size_t want = hint + hint / 8 + 4096;
    if (buf.capacity() < want)
        buf.reserve(want);
    return buf;
}

void ps2ChainBufferNoteSize(uint32_t channelBase, size_t bytes)
{
    g_chainPool.lastBytes[chainChannelIndex(channelBase)].store(bytes, std::memory_order_relaxed);
}

namespace
{
    struct VerifyState
    {
        std::mutex mutex;
        struct Entry
        {
            std::vector<ChainVerifySeg> segs;
            std::chrono::steady_clock::time_point t0;
        };
        std::unordered_map<const uint8_t *, Entry> pending;
        uint64_t jobs = 0, segsChecked[8] = {}, segsBad[8] = {}, bytesChecked[8] = {}, bytesBad[8] = {};
        uint64_t badJobs = 0;
        double latSumMs = 0, latMaxMs = 0;
        std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    };
    VerifyState g_verify;
}

bool ps2ChainVerifyEnabled()
{
    static const bool on = []
    {
        const char *v = std::getenv("PS2X_CHAIN_VERIFY");
        return v && *v && *v != '0';
    }();
    return on;
}

void ps2ChainVerifyRegister(const uint8_t *bufData, std::vector<ChainVerifySeg> &&segs)
{
    std::lock_guard<std::mutex> lock(g_verify.mutex);
    auto &e = g_verify.pending[bufData];
    e.segs = std::move(segs);
    e.t0 = std::chrono::steady_clock::now();
}

void ps2ChainVerifyCheck(const uint8_t *bufData)
{
    VerifyState::Entry e;
    {
        std::lock_guard<std::mutex> lock(g_verify.mutex);
        auto it = g_verify.pending.find(bufData);
        if (it == g_verify.pending.end())
            return;
        e = std::move(it->second);
        g_verify.pending.erase(it);
    }
    const auto now = std::chrono::steady_clock::now();
    const double latMs = std::chrono::duration<double, std::milli>(now - e.t0).count();
    uint64_t sc[8] = {}, sb[8] = {}, bc[8] = {}, bb[8] = {};
    bool bad = false;
    for (const auto &sg : e.segs)
    {
        const uint32_t k = sg.id & 7u;
        ++sc[k];
        bc[k] += sg.n;
        if (std::memcmp(sg.src, bufData + sg.dst, sg.n) != 0)
        {
            ++sb[k];
            bb[k] += sg.n;
            bad = true;
        }
    }
    std::lock_guard<std::mutex> lock(g_verify.mutex);
    ++g_verify.jobs;
    g_verify.badJobs += bad ? 1 : 0;
    g_verify.latSumMs += latMs;
    g_verify.latMaxMs = std::max(g_verify.latMaxMs, latMs);
    for (int i = 0; i < 8; ++i)
    {
        g_verify.segsChecked[i] += sc[i];
        g_verify.segsBad[i] += sb[i];
        g_verify.bytesChecked[i] += bc[i];
        g_verify.bytesBad[i] += bb[i];
    }
    if (now - g_verify.last > std::chrono::seconds(10))
    {
        static const char *names[8] = {"refe0", "cnt1", "next2", "ref3", "refs4", "call5", "ret6", "end7"};
        std::fprintf(stderr, "[chainverify] jobs=%llu jobsWithChange=%llu kick->done latency avg=%.2fms max=%.2fms\n",
                     (unsigned long long)g_verify.jobs, (unsigned long long)g_verify.badJobs,
                     g_verify.jobs ? g_verify.latSumMs / g_verify.jobs : 0.0, g_verify.latMaxMs);
        for (int i = 0; i < 8; ++i)
            if (g_verify.segsChecked[i])
                std::fprintf(stderr, "[chainverify]   %-6s segs=%llu changed=%llu bytes=%llu changedBytes=%llu\n", names[i],
                             (unsigned long long)g_verify.segsChecked[i], (unsigned long long)g_verify.segsBad[i],
                             (unsigned long long)g_verify.bytesChecked[i], (unsigned long long)g_verify.bytesBad[i]);
        std::fflush(stderr);
        g_verify.jobs = g_verify.badJobs = 0;
        g_verify.latSumMs = g_verify.latMaxMs = 0;
        for (int i = 0; i < 8; ++i)
            g_verify.segsChecked[i] = g_verify.segsBad[i] = g_verify.bytesChecked[i] = g_verify.bytesBad[i] = 0;
        g_verify.last = now;
    }
}

void ps2ChainBufferRecycle(std::vector<uint8_t> &&buf)
{
    if (ps2ChainVerifyEnabled() && !buf.empty())
        ps2ChainVerifyCheck(buf.data());
    const size_t cap = buf.capacity();
    if (!chainPoolEnabled() || cap < 4096 || cap > (256u << 20))
        return; // let it go: tiny buffers are cheap, huge ones should not stay resident
    buf.clear();
    std::lock_guard<std::mutex> lock(g_chainPool.mutex);
    if (g_chainPool.free.size() < 8)
        g_chainPool.free.push_back(std::move(buf));
}

void ps2Vif1StatAddGifNs(uint64_t ns)
{
    g.stats.gifNs.fetch_add(ns, std::memory_order_relaxed);
}

bool ps2Vif1Mscal(uint32_t startPc, uint32_t top, uint32_t itop)
{
    return runOnWorker(
        [=]
        {
            const PS2HostVu1 &hv = ps2HostVu1();
            const uint64_t t0 = g.stats.enabled ? nowNs() : 0u;
            hv.mscal(startPc, top, itop, g.fbrst.load(std::memory_order_relaxed), g.memory->getVU1CodeGeneration(),
                     &g.vpuStat);
            if (g.stats.enabled)
                g.stats.vuNs.fetch_add(nowNs() - t0, std::memory_order_relaxed);
        });
}

bool ps2Vif1Mscnt(uint32_t top, uint32_t itop)
{
    return runOnWorker(
        [=]
        {
            const PS2HostVu1 &hv = ps2HostVu1();
            const uint64_t t0 = g.stats.enabled ? nowNs() : 0u;
            hv.mscnt(top, itop, g.fbrst.load(std::memory_order_relaxed), g.memory->getVU1CodeGeneration(), &g.vpuStat);
            if (g.stats.enabled)
                g.stats.vuNs.fetch_add(nowNs() - t0, std::memory_order_relaxed);
        });
}

void ps2RunAfterQueuedGif(void (*fn)(void *), void *arg)
{
    if (!g.active.load(std::memory_order_relaxed) || t_onWorker)
    {
        fn(arg);
        return;
    }
    Job job;
    job.kind = 3;
    job.call = [fn, arg] { fn(arg); };
    enqueue(std::move(job)); // no completion: the EE does not wait
}

void ps2FlushQueuedGif()
{
    if (!g.started || t_onWorker)
        return;
    std::unique_lock<std::mutex> lock(g.mutex);
    const uint64_t target = g.enqueued;
    g.idleCv.wait(lock, [target] { return g.finished.load(std::memory_order_acquire) >= target; });
}

void ps2Vif1WorkerShutdown()
{
    if (!g.started)
        return;
    {
        std::lock_guard<std::mutex> lock(g.mutex);
        g.stop = true;
    }
    g.cv.notify_all();
    if (g.thread.joinable() && std::this_thread::get_id() != g.thread.get_id())
        g.thread.join();
    g.started = false;
    g.stop = false;
    g.queue.clear();
    g.enqueued = 0;
    g.finished.store(0);
    g.done.clear();
    g.pending.store(0);
    g.doneCount.store(0);
    g.applied = 0;
    g.outstanding[0] = g.outstanding[1] = g.outstanding[2] = 0;
}
