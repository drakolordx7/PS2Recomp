#include "ps2x/iop/iop_subsystem.h"

#include "iop_service.h"
#include "iop_module_manager.h"
#include "emulator/iop_emulator.h"
#include "module_factories.h"
#include "ps2x/iop/ps2_path.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <xmmintrin.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#endif

namespace ps2x::iop
{
    namespace
    {
        thread_local bool t_onIopThread = false;

        uint64_t nowNs()
        {
            return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now().time_since_epoch())
                                             .count());
        }

        bool envFlag(const char *name, bool defaultValue)
        {
            const char *v = std::getenv(name);
            if (!v || !*v)
                return defaultValue;
            return !(v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F');
        }

        // Threaded mode granularity. The IOP thread runs the IOP in chunks of kChunkEeCycles (4096 IOP cycles, ~110 us
        // of IOP time) so EE-side callers waiting for the lock get in quickly, and only wakes for work when at least
        // kMinBacklogEeCycles (2048 IOP cycles, ~55 us) of EE time are unconsumed.
        constexpr uint64_t kChunkEeCycles = 4096u * 8u;
        constexpr uint64_t kMinBacklogEeCycles = 2048u * 8u;
    }

    bool onIopThread() noexcept
    {
        return t_onIopThread;
    }

    class IopSubsystem::Impl
    {
    public:
        explicit Impl(IopHost &hostRef)
            : host(hostRef),
              emulator(hostRef)
        {
            statsEnabled = envFlag("PS2X_IOP_THREAD_STATS", false);
            rpcStats.enabled = envFlag("PS2X_IOP_RPC_STATS", false);
            rpcAsyncEnabled = envFlag("PS2X_IOP_RPC_ASYNC", true);
            coreServices.emplace_back(detail::createMcservService(host));
            coreServices.emplace_back(detail::createDbcmanService(host));
            coreServices.emplace_back(detail::createLibSdService(host));
            refreshServiceModuleKeys();
            rebuildRoutes();
        }

        bool serviceActive(const detail::IopService &service) const
        {
            return moduleManager.isLoaded(service.moduleAliases());
        }

        void refreshServiceModuleKeys()
        {
            std::vector<std::string> keys;
            for (const auto &service : coreServices)
            {
                for (std::string_view alias : service->moduleAliases())
                    keys.emplace_back(alias);
            }
            moduleManager.setServiceModuleKeys(std::move(keys));
        }

        void rebuildRoutes()
        {
            routes.clear();
            lastError.clear();
            for (const auto &service : coreServices)
            {
                if (!serviceActive(*service))
                    continue;
                for (const uint32_t sid : service->sids())
                {
                    if (!routes.emplace(sid, service.get()).second)
                    {
                        std::ostringstream out;
                        out << "duplicate IOP SID 0x" << std::hex << sid << " in core services";
                        lastError = out.str();
                        routes.clear();
                        return;
                    }
                }
            }
        }

        void recordLoadOutcome(std::string_view path, bool hle)
        {
            constexpr size_t maxOutcomes = 32u;
            if (loadOutcomes.size() >= maxOutcomes || !loggedLoadPaths.emplace(path).second)
                return;
            std::string message = hle ? "[IOP:HLE] fallback module='" : "[IOP:load-failed] module='";
            message.append(path);
            message += hle ? "' physical IRX unavailable; using registered HLE provider"
                           : "' no HLE provider accepted the module; physical IRX was not loaded";
            loadOutcomes.push_back(message);
            host.log(hle ? LogLevel::Info : LogLevel::Warning, message);
        }

        ~Impl()
        {
            stopThread();
        }

        // ---- threaded mode ------------------------------------------------------------------------------------------
        // Every public IopSubsystem entry point takes execMutex (recursive: handleRpc -> HLE service -> host ->
        // readIopMemory re-enters). The IOP thread holds it while it runs the IOP. execWaiters lets the IOP thread
        // hand the lock over between chunks: std::mutex is not fair, and without it a tight unlock/lock loop would
        // starve the EE thread.
        struct ExecGuard
        {
            Impl *impl = nullptr;
            explicit ExecGuard(const Impl &owner) noexcept
            {
                if (owner.threaded.load(std::memory_order_relaxed))
                {
                    impl = const_cast<Impl *>(&owner);
                    impl->lockExec();
                }
            }
            ~ExecGuard()
            {
                if (impl)
                    impl->execMutex.unlock();
            }
            ExecGuard(const ExecGuard &) = delete;
            ExecGuard &operator=(const ExecGuard &) = delete;
        };

        void lockExec()
        {
            eeCalls.fetch_add(1, std::memory_order_relaxed);
            execWaiters.fetch_add(1, std::memory_order_acq_rel);
            if (statsEnabled)
            {
                const uint64_t t0 = nowNs();
                execMutex.lock();
                const uint64_t waited = nowNs() - t0;
                eeWaitNs.fetch_add(waited, std::memory_order_relaxed);
                if (waited > 1000000u)
                    eeWaitsOver1ms.fetch_add(1, std::memory_order_relaxed);
                uint64_t prev = eeMaxWaitNs.load(std::memory_order_relaxed);
                while (waited > prev && !eeMaxWaitNs.compare_exchange_weak(prev, waited, std::memory_order_relaxed))
                {
                }
            }
            else
            {
                execMutex.lock();
            }
            execWaiters.fetch_sub(1, std::memory_order_acq_rel);
        }

        void startThread()
        {
            std::lock_guard<std::mutex> lock(threadStartMutex);
            if (threadStarted.load(std::memory_order_acquire) || !threaded.load(std::memory_order_acquire))
                return;
            stopRequested.store(false, std::memory_order_release);
#if defined(_WIN32)
            struct Start
            {
                static unsigned __stdcall entry(void *arg)
                {
                    static_cast<Impl *>(arg)->threadMain();
                    return 0;
                }
            };
            // 4 MB: IOP guest callbacks nest (callFunction re-entrancy) and the emulator keeps CPU states on the stack.
            threadHandle = reinterpret_cast<void *>(_beginthreadex(nullptr, 4u << 20, &Start::entry, this, 0, nullptr));
            threadStarted.store(threadHandle != nullptr, std::memory_order_release);
#else
            worker = std::thread([this]() { threadMain(); });
            threadStarted.store(true, std::memory_order_release);
#endif
        }

        void stopThread()
        {
            std::lock_guard<std::mutex> lock(threadStartMutex);
            if (!threadStarted.load(std::memory_order_acquire))
                return;
            stopRequested.store(true, std::memory_order_release);
#if defined(_WIN32)
            if (threadHandle)
            {
                WaitForSingleObject(static_cast<HANDLE>(threadHandle), INFINITE);
                CloseHandle(static_cast<HANDLE>(threadHandle));
                threadHandle = nullptr;
            }
#else
            if (worker.joinable())
                worker.join();
#endif
            threadStarted.store(false, std::memory_order_release);
        }

        // Sleep ~100 us (a high-resolution waitable timer on Windows; the default timer tick is 15.6 ms).
        static void idleWait(void *timer)
        {
#if defined(_WIN32)
            if (timer)
            {
                LARGE_INTEGER due;
                due.QuadPart = -1000; // 100 us, relative, in 100 ns units
                if (SetWaitableTimer(static_cast<HANDLE>(timer), &due, 0, nullptr, nullptr, FALSE))
                {
                    WaitForSingleObject(static_cast<HANDLE>(timer), 20);
                    return;
                }
            }
            Sleep(1);
#else
            (void)timer;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
#endif
        }

        void threadMain()
        {
            t_onIopThread = true;
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
            // The EE thread runs with FTZ|DAZ (denormals flushed, like the PS2's FPU); SPU2 mixing moved here from
            // there, so keep its floating-point behaviour identical.
            _mm_setcsr(_mm_getcsr() | 0x8040u);
#endif
            void *timer = nullptr;
#if defined(_WIN32)
            if (const HMODULE kernel = GetModuleHandleW(L"kernel32.dll"))
            {
                using SetDesc = HRESULT(WINAPI *)(HANDLE, PCWSTR);
                if (const auto setDesc = reinterpret_cast<SetDesc>(GetProcAddress(kernel, "SetThreadDescription")))
                    setDesc(GetCurrentThread(), L"ps2x IOP+SPU2");
            }
            // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x2 (Windows 10 1803+); falls back to Sleep(1) if unsupported.
            timer = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002u, TIMER_ALL_ACCESS);
#endif
            uint64_t consumedLocal = 0;
            uint64_t lastStatNs = nowNs();
            uint64_t statRunNs = 0;
            while (!stopRequested.load(std::memory_order_acquire))
            {
                const uint64_t target = published.load(std::memory_order_acquire);
                const uint64_t backlog = target - consumedLocal;
                if (backlog < kMinBacklogEeCycles)
                {
                    if (rpcPending.load(std::memory_order_acquire) != 0u)
                    {
                        std::lock_guard<std::recursive_mutex> lock(execMutex);
                        runQueuedRpcs();
                    }
                    else
                    {
                        idleWait(timer);
                    }
                    continue;
                }
                uint64_t maxLag = maxLagCycles.load(std::memory_order_relaxed);
                while (backlog > maxLag && !maxLagCycles.compare_exchange_weak(maxLag, backlog, std::memory_order_relaxed))
                {
                }
                while (consumedLocal < target && !stopRequested.load(std::memory_order_acquire))
                {
                    const uint64_t chunk = std::min<uint64_t>(target - consumedLocal, kChunkEeCycles);
                    {
                        std::lock_guard<std::recursive_mutex> lock(execMutex);
                        const uint64_t t0 = nowNs();
                        emulator.runEeCycles(chunk);
                        runQueuedRpcs();
                        const uint64_t dt = nowNs() - t0;
                        runNs.fetch_add(dt, std::memory_order_relaxed);
                        statRunNs += dt;
                    }
                    consumedLocal += chunk;
                    consumed.store(consumedLocal, std::memory_order_release);
                    // Let EE-side callers that queued for the lock in before taking it again.
                    while (execWaiters.load(std::memory_order_acquire) > 0 &&
                           !stopRequested.load(std::memory_order_acquire))
                    {
#if defined(_WIN32)
                        SwitchToThread();
#else
                        std::this_thread::yield();
#endif
                    }
                }
                if (statsEnabled)
                {
                    const uint64_t now = nowNs();
                    if (now - lastStatNs >= 5000000000ull)
                    {
                        const double wall = static_cast<double>(now - lastStatNs);
                        const uint64_t pub = published.load(std::memory_order_relaxed);
                        // IOP time already run ahead of the consumed EE clock (outside-thread waits inside RPCs and
                        // module starts advance the IOP on the EE thread), in ms of IOP time.
                        int64_t leadCycles = 0;
                        {
                            std::lock_guard<std::recursive_mutex> lock(execMutex);
                            leadCycles = static_cast<int64_t>(emulator.cycles()) -
                                         static_cast<int64_t>(consumedLocal / 8u);
                        }
                        std::fprintf(stderr,
                                     "[iop-thread] busy %.1f%% of wall, lag now %.2f ms (max %.2f ms), IOP lead %.2f ms, "
                                     "EE calls %llu, EE lock wait total %.2f ms max %.3f ms, waits >1 ms: %llu\n",
                                     100.0 * static_cast<double>(statRunNs) / wall,
                                     static_cast<double>(pub - consumedLocal) / 294912.0,
                                     static_cast<double>(maxLagCycles.load(std::memory_order_relaxed)) / 294912.0,
                                     static_cast<double>(leadCycles) / 36864.0,
                                     static_cast<unsigned long long>(eeCalls.load(std::memory_order_relaxed)),
                                     static_cast<double>(eeWaitNs.load(std::memory_order_relaxed)) / 1.0e6,
                                     static_cast<double>(eeMaxWaitNs.load(std::memory_order_relaxed)) / 1.0e6,
                                     static_cast<unsigned long long>(eeWaitsOver1ms.load(std::memory_order_relaxed)));
                        lastStatNs = now;
                        statRunNs = 0;
                    }
                }
            }
#if defined(_WIN32)
            if (timer)
                CloseHandle(static_cast<HANDLE>(timer));
#endif
        }

        // PS2X_IOP_RPC_STATS=1: per (sid, function, nowait, end function) call counts and costs, printed every 10 s.
        struct RpcStats
        {
            struct Row
            {
                uint64_t calls = 0, waitNs = 0, runNs = 0, maxRunNs = 0, instructions = 0, sendBytes = 0, unhandled = 0;
                uint64_t async = 0;
            };
            bool enabled = false;
            std::mutex mutex;
            std::map<uint64_t, Row> rows;
            uint64_t lastPrintNs = 0;
            uint64_t startNs = 0;
            uint64_t queueNs = 0, maxQueueNs = 0, queued = 0; // offloaded calls: submit -> start of server function
            static uint64_t key(const RpcRequest &r)
            {
                return (static_cast<uint64_t>(r.sid) << 32) | (static_cast<uint64_t>(r.function & 0xFFFFu) << 2) |
                       ((r.mode & 1u) << 1) | (r.endFunction != 0u ? 1u : 0u);
            }
            void record(const RpcRequest &r, bool handled, uint64_t waitNs, uint64_t runNs, uint64_t instructions,
                        bool async)
            {
                std::lock_guard<std::mutex> lock(mutex);
                Row &row = rows[key(r)];
                if (row.calls == 0)
                    ids[key(r)] = {r.sid, r.function, r.mode, r.endFunction != 0u};
                ++row.calls;
                row.async += async ? 1 : 0;
                row.waitNs += waitNs;
                row.runNs += runNs;
                row.maxRunNs = std::max(row.maxRunNs, runNs);
                row.instructions += instructions;
                row.sendBytes += r.send.size;
                row.unhandled += handled ? 0 : 1;
                const uint64_t now = nowNs();
                if (startNs == 0)
                    startNs = lastPrintNs = now;
                if (now - lastPrintNs >= 10000000000ull)
                {
                    print(now);
                    lastPrintNs = now;
                }
            }
            struct Id
            {
                uint32_t sid, function, mode;
                bool end;
            };
            std::map<uint64_t, Id> ids;
            void print(uint64_t now)
            {
                std::fprintf(stderr, "[iop-rpc] t=%.0fs cumulative (calls, async, wait ms, run ms, avg run us, max run us, avg insns, avg send B, unhandled)\n",
                             static_cast<double>(now - startNs) / 1e9);
                std::vector<std::pair<uint64_t, uint64_t>> order;
                for (const auto &[k, row] : rows)
                    order.emplace_back(row.runNs + row.waitNs, k);
                std::sort(order.rbegin(), order.rend());
                uint64_t totalWait = 0, totalRun = 0, totalCalls = 0;
                for (const auto &[k, row] : rows)
                {
                    totalWait += row.waitNs;
                    totalRun += row.runNs;
                    totalCalls += row.calls;
                }
                std::fprintf(stderr, "[iop-rpc]   total calls %llu wait %.1f ms run %.1f ms; offloaded %llu, queue latency avg %.1f us max %.1f us\n",
                             static_cast<unsigned long long>(totalCalls), totalWait / 1e6, totalRun / 1e6,
                             static_cast<unsigned long long>(queued),
                             queued ? static_cast<double>(queueNs) / 1e3 / static_cast<double>(queued) : 0.0,
                             static_cast<double>(maxQueueNs) / 1e3);
                int shown = 0;
                for (const auto &[cost, k] : order)
                {
                    if (++shown > 14)
                        break;
                    const Row &row = rows[k];
                    const Id &id = ids[k];
                    std::fprintf(stderr,
                                 "[iop-rpc]   sid=%08x fn=%04x %s%s calls=%llu async=%llu wait=%.1fms run=%.1fms avg=%.1fus max=%.1fus insns=%.0f send=%.0fB unh=%llu\n",
                                 id.sid, id.function, (id.mode & 1u) ? "nowait" : "block", id.end ? "+end" : "",
                                 static_cast<unsigned long long>(row.calls), static_cast<unsigned long long>(row.async),
                                 row.waitNs / 1e6, row.runNs / 1e6, row.runNs / 1e3 / static_cast<double>(row.calls),
                                 row.maxRunNs / 1e3, static_cast<double>(row.instructions) / static_cast<double>(row.calls),
                                 static_cast<double>(row.sendBytes) / static_cast<double>(row.calls),
                                 static_cast<unsigned long long>(row.unhandled));
                }
            }
        };
        RpcStats rpcStats;

        // ---- offloaded nowait RPC calls (see IopSubsystem::submitRpc) ------------------------------------------------
        std::mutex rpcQueueMutex;
        std::deque<IopSubsystem::AsyncRpc> rpcQueue;
        std::atomic<uint32_t> rpcPending{0};
        std::mutex offloadMutex;
        std::unordered_set<uint32_t> offloadSids; // sids known to have an emulated server (positive answers only)
        bool rpcAsyncEnabled = true;

        void clearOffloadCache()
        {
            std::lock_guard<std::mutex> lock(offloadMutex);
            offloadSids.clear();
        }

        // Runs every queued call, oldest first. Caller holds execMutex (IOP thread, or an EE-side entry point that
        // must stay ordered behind the queue). A `done` that throws would lose the completion; it must not.
        void runQueuedRpcs()
        {
            while (rpcPending.load(std::memory_order_acquire) != 0u)
            {
                IopSubsystem::AsyncRpc call;
                {
                    std::lock_guard<std::mutex> lock(rpcQueueMutex);
                    if (rpcQueue.empty())
                        return;
                    call = std::move(rpcQueue.front());
                    rpcQueue.pop_front();
                    rpcPending.fetch_sub(1u, std::memory_order_release);
                }
                RpcResult result{};
                const uint64_t t0 = rpcStats.enabled ? nowNs() : 0u;
                const uint64_t ins0 = rpcStats.enabled ? emulator.instructions() : 0u;
                if (rpcStats.enabled && call.submitNs != 0u)
                {
                    std::lock_guard<std::mutex> statLock(rpcStats.mutex);
                    const uint64_t latency = t0 - call.submitNs;
                    rpcStats.queueNs += latency;
                    rpcStats.maxQueueNs = std::max(rpcStats.maxQueueNs, latency);
                    ++rpcStats.queued;
                }
                try
                {
                    result = emulator.handleRpc(call.request, &call.send);
                }
                catch (...)
                {
                    result = RpcResult{};
                }
                if (rpcStats.enabled)
                    rpcStats.record(call.request, result.handled, 0u, nowNs() - t0, emulator.instructions() - ins0, true);
                if (call.done)
                    call.done(std::move(result));
            }
        }

        mutable std::recursive_mutex execMutex;
        std::atomic<int> execWaiters{0};
        std::atomic<bool> threaded{false};
        std::atomic<bool> threadStarted{false};
        std::atomic<bool> stopRequested{false};
        std::mutex threadStartMutex;
        std::atomic<uint64_t> published{0};
        std::atomic<uint64_t> consumed{0};
        std::atomic<uint64_t> maxLagCycles{0};
        std::atomic<uint64_t> runNs{0};
        std::atomic<uint64_t> eeCalls{0};
        std::atomic<uint64_t> eeWaitNs{0};
        std::atomic<uint64_t> eeMaxWaitNs{0};
        std::atomic<uint64_t> eeWaitsOver1ms{0};
        bool statsEnabled = false;
#if defined(_WIN32)
        void *threadHandle = nullptr;
#else
        std::thread worker;
#endif

        IopHost &host;
        detail::ServiceList coreServices;
        std::unordered_map<uint32_t, detail::IopService *> routes;
        std::vector<std::string> loadOutcomes;
        std::unordered_set<std::string> loggedLoadPaths;
        std::string lastError;
        detail::IopModuleManager moduleManager;
        detail::IopEmulator emulator;
    };

    IopSubsystem::IopSubsystem(IopHost &host)
        : m_impl(std::make_unique<Impl>(host))
    {
    }

    IopSubsystem::~IopSubsystem() = default;
    IopSubsystem::IopSubsystem(IopSubsystem &&) noexcept = default;
    IopSubsystem &IopSubsystem::operator=(IopSubsystem &&) noexcept = default;

    void IopSubsystem::reset()
    {
        const Impl::ExecGuard guard(*m_impl);
        m_impl->runQueuedRpcs();
        m_impl->clearOffloadCache();
        m_impl->moduleManager.reset();
        m_impl->loadOutcomes.clear();
        m_impl->loggedLoadPaths.clear();
        for (auto &service : m_impl->coreServices)
        {
            if (service)
            {
                service->reset();
            }
        }
        m_impl->emulator.reset();
        m_impl->refreshServiceModuleKeys();
        m_impl->rebuildRoutes();
    }

    ModuleLoadResult IopSubsystem::loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
    {
        const Impl::ExecGuard guard(*m_impl);
        m_impl->runQueuedRpcs();
        m_impl->clearOffloadCache();
        const ParsedPs2Path parsed = parsePs2Path(path);
        if (!parsed)
            return {true, -1, -1};

        if (parsed.device != Ps2PathDevice::Rom0)
        {
            ModuleLoadResult physical = m_impl->emulator.loadModule(path, arguments, argumentSize);
            if (physical.moduleId > 0)
            {
                m_impl->moduleManager.observePhysicalLoad(physical.moduleId, path);
                m_impl->rebuildRoutes();
                return physical;
            }
        }

        ModuleLoadResult hle = m_impl->moduleManager.loadHle(path);
        if (hle.moduleId > 0)
        {
            m_impl->rebuildRoutes();
            if (parsed.device != Ps2PathDevice::Rom0)
                m_impl->recordLoadOutcome(path, true);
        }
        else
        {
            m_impl->recordLoadOutcome(path, false);
        }
        return hle;
    }

    ModuleLoadResult IopSubsystem::loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
    {
        const Impl::ExecGuard guard(*m_impl);
        m_impl->runQueuedRpcs();
        m_impl->clearOffloadCache();
        return m_impl->emulator.loadModuleBuffer(guestAddress, arguments, argumentSize);
    }

    bool IopSubsystem::stopModule(int32_t moduleId, int32_t *result)
    {
        const Impl::ExecGuard guard(*m_impl);
        m_impl->runQueuedRpcs();
        m_impl->clearOffloadCache();
        if (m_impl->moduleManager.stopHle(moduleId, result))
        {
            m_impl->rebuildRoutes();
            return true;
        }
        if (!m_impl->emulator.stopModule(moduleId, result))
            return false;
        m_impl->moduleManager.observePhysicalStop(moduleId);
        m_impl->rebuildRoutes();
        return true;
    }

    void IopSubsystem::runEeCycles(uint64_t eeCycles) noexcept
    {
        const Impl::ExecGuard guard(*m_impl);
        m_impl->emulator.runEeCycles(eeCycles);
    }

    RpcAbi IopSubsystem::selectRpcAbi(const RpcAbiRequest &request) const
    {
        // No IOP lock: this runs on every SifCallRpc, and with offloaded RPCs the IOP thread can hold the lock for
        // milliseconds. It only reads the (constant) service list and which HLE modules are loaded; the latter changes
        // only in the EE-side module operations, which run on the calling (EE) thread.
        for (const auto &service : m_impl->coreServices)
        {
            if (service && m_impl->serviceActive(*service))
            {
                const RpcAbi selected = service->selectRpcAbi(request);
                if (selected != RpcAbi::RuntimeDefault)
                {
                    return selected;
                }
            }
        }
        return RpcAbi::RuntimeDefault;
    }

    bool IopSubsystem::canBindRpc(uint32_t sid) const noexcept
    {
        const Impl::ExecGuard guard(*m_impl);
        if (m_impl->routes.find(sid) != m_impl->routes.end())
        {
            return true;
        }
        return m_impl->emulator.hasRpcServer(sid);
    }

    RpcResult IopSubsystem::handleRpc(const RpcRequest &request)
    {
        const uint64_t statT0 = m_impl->rpcStats.enabled ? nowNs() : 0u;
        const Impl::ExecGuard guard(*m_impl);
        const uint64_t statT1 = m_impl->rpcStats.enabled ? nowNs() : 0u;
        m_impl->runQueuedRpcs();
        const uint64_t statIns0 = m_impl->rpcStats.enabled ? m_impl->emulator.instructions() : 0u;
        const auto route = m_impl->routes.find(request.sid);
        detail::IopService *hle = route != m_impl->routes.end() ? route->second : nullptr;

        RpcResult emulated = m_impl->emulator.handleRpc(request);
        if (m_impl->rpcStats.enabled)
        {
            m_impl->rpcStats.record(request, emulated.handled, statT1 - statT0, nowNs() - statT1,
                                    m_impl->emulator.instructions() - statIns0, false);
        }
        if (emulated.handled || !hle)
        {
            return emulated;
        }
        return hle->handleRpc(request);
    }

    bool IopSubsystem::canOffloadRpc(uint32_t sid)
    {
        if (!m_impl->rpcAsyncEnabled || !m_impl->threadStarted.load(std::memory_order_acquire) ||
            !m_impl->threaded.load(std::memory_order_acquire) || t_onIopThread)
            return false;
        {
            std::lock_guard<std::mutex> lock(m_impl->offloadMutex);
            if (m_impl->offloadSids.count(sid) != 0u)
                return true;
        }
        // First call (or no emulated server yet): ask the emulator under the IOP lock. Module operations clear the
        // cache under the same lock, so a positive answer cannot outlive its module.
        const Impl::ExecGuard guard(*m_impl);
        if (!m_impl->emulator.hasRpcServer(sid))
            return false;
        std::lock_guard<std::mutex> lock(m_impl->offloadMutex);
        m_impl->offloadSids.insert(sid);
        return true;
    }

    bool IopSubsystem::submitRpc(AsyncRpc call)
    {
        if (!m_impl->rpcAsyncEnabled || !m_impl->threadStarted.load(std::memory_order_acquire))
            return false;
        if (m_impl->rpcStats.enabled)
            call.submitNs = nowNs();
        std::lock_guard<std::mutex> lock(m_impl->rpcQueueMutex);
        m_impl->rpcQueue.push_back(std::move(call));
        m_impl->rpcPending.fetch_add(1u, std::memory_order_release);
        return true;
    }

    void IopSubsystem::onSifTransfer(const SifTransfer &transfer)
    {
        const Impl::ExecGuard guard(*m_impl);
        for (auto &service : m_impl->coreServices)
        {
            if (service && m_impl->serviceActive(*service))
            {
                service->onSifTransfer(transfer);
            }
        }
        m_impl->emulator.onSifTransfer(transfer);
    }

    uint32_t IopSubsystem::allocateMemory(uint32_t size, uint32_t alignment)
    {
        const Impl::ExecGuard guard(*m_impl);
        return m_impl->emulator.allocateMemory(size, alignment);
    }

    bool IopSubsystem::freeMemory(uint32_t address)
    {
        const Impl::ExecGuard guard(*m_impl);
        return m_impl->emulator.freeMemory(address);
    }

    bool IopSubsystem::readMemory(uint32_t address, void *destination, size_t size) const
    {
        const Impl::ExecGuard guard(*m_impl);
        return m_impl->emulator.readMemory(address, destination, size);
    }

    bool IopSubsystem::writeMemory(uint32_t address, const void *source, size_t size)
    {
        const Impl::ExecGuard guard(*m_impl);
        return m_impl->emulator.writeMemory(address, source, size);
    }

    bool IopSubsystem::zeroMemory(uint32_t address, size_t size)
    {
        const Impl::ExecGuard guard(*m_impl);
        return m_impl->emulator.zeroMemory(address, size);
    }

    bool IopSubsystem::isMemoryRange(uint32_t address, size_t size) const
    {
        return m_impl->emulator.isMemoryRange(address, size);
    }

    DebugSnapshot IopSubsystem::debugSnapshot() const
    {
        const Impl::ExecGuard guard(*m_impl);
        DebugSnapshot snapshot;
        snapshot.emulatorCycles = m_impl->emulator.cycles();
        snapshot.emulatorInstructions = m_impl->emulator.instructions();
        snapshot.emulatorLoadedModules = m_impl->emulator.loadedModuleCount();
        snapshot.emulatorThreads = m_impl->emulator.threadCount();
        snapshot.emulatorRpcServers = m_impl->emulator.rpcServerCount();
        snapshot.diagnostics = m_impl->loadOutcomes;
        if (!m_impl->lastError.empty())
        {
            snapshot.diagnostics.push_back(m_impl->lastError);
        }

        for (const auto &service : m_impl->coreServices)
        {
            DebugService row;
            row.name = service->name();
            row.sids.assign(service->sids().begin(), service->sids().end());
            row.active = m_impl->serviceActive(*service);
            service->appendDebugMetrics(row.metrics);
            snapshot.services.push_back(std::move(row));
        }
        return snapshot;
    }

    void IopSubsystem::enableThread(bool enable)
    {
        if (enable)
        {
            m_impl->threaded.store(true, std::memory_order_release);
            return;
        }
        m_impl->stopThread();
        m_impl->threaded.store(false, std::memory_order_release);
    }

    bool IopSubsystem::threadEnabled() const noexcept
    {
        return m_impl->threaded.load(std::memory_order_acquire);
    }

    void IopSubsystem::publishEeCycles(uint64_t totalEeCycles) noexcept
    {
        if (!m_impl->threaded.load(std::memory_order_relaxed))
            return;
        m_impl->published.store(totalEeCycles, std::memory_order_release);
        if (!m_impl->threadStarted.load(std::memory_order_relaxed))
            m_impl->startThread();
    }

    IopSubsystem::ThreadStats IopSubsystem::threadStats() const
    {
        ThreadStats stats;
        stats.published = m_impl->published.load(std::memory_order_relaxed);
        stats.consumed = m_impl->consumed.load(std::memory_order_relaxed);
        stats.maxLagCycles = m_impl->maxLagCycles.load(std::memory_order_relaxed);
        stats.runNs = m_impl->runNs.load(std::memory_order_relaxed);
        stats.eeCalls = m_impl->eeCalls.load(std::memory_order_relaxed);
        stats.eeWaitNs = m_impl->eeWaitNs.load(std::memory_order_relaxed);
        stats.eeMaxWaitNs = m_impl->eeMaxWaitNs.load(std::memory_order_relaxed);
        return stats;
    }
}
