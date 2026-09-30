#pragma once

#include "ps2x/iop/iop_host.h"
#include "ps2x/iop/iop_types.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ps2x::iop
{
    // True on the thread that runs the IOP in threaded mode.
    [[nodiscard]] bool onIopThread() noexcept;

    class IopSubsystem
    {
    public:
        explicit IopSubsystem(IopHost &host);
        ~IopSubsystem();

        IopSubsystem(const IopSubsystem &) = delete;
        IopSubsystem &operator=(const IopSubsystem &) = delete;
        IopSubsystem(IopSubsystem &&) noexcept;
        IopSubsystem &operator=(IopSubsystem &&) noexcept;

        void reset();

        [[nodiscard]] ModuleLoadResult loadModule(std::string_view path, const void *arguments = nullptr, uint32_t argumentSize = 0);
        [[nodiscard]] ModuleLoadResult loadModuleBuffer(uint32_t guestAddress, const void *arguments = nullptr, uint32_t argumentSize = 0);
        [[nodiscard]] bool stopModule(int32_t moduleId, int32_t *result = nullptr);
        void runEeCycles(uint64_t eeCycles) noexcept;

        [[nodiscard]] RpcAbi selectRpcAbi(const RpcAbiRequest &request) const;
        [[nodiscard]] bool canBindRpc(uint32_t sid) const noexcept;
        [[nodiscard]] RpcResult handleRpc(const RpcRequest &request);

        // ---- Offloaded (nowait) RPC calls, threaded mode only ---------------------------------------------------------
        // A nowait call to an emulated IOP RPC server does not have to run its server function on the calling
        // thread: the IOP runs it asynchronously on real hardware. canOffloadRpc(sid) is true when the IOP thread is
        // running and an emulated (physical IRX) server is registered for sid (cheap after the first positive answer;
        // negative answers are never cached). submitRpc() then queues the call: the IOP thread runs the server function
        // in FIFO order (between its time slices, holding the IOP lock, outside any IOP thread exactly like the inline
        // path), writes the reply into the guest receive buffer, and calls `done` on the thread that ran it, still with
        // the lock held (the host is expected to post the completion to the EE). `send` is a snapshot of the send
        // buffer taken at call time (the DMA of the real SIF). A blocking handleRpc() (and reset/module operations)
        // first runs everything queued before it, so calls stay ordered.
        struct AsyncRpc
        {
            RpcRequest request;
            std::vector<uint8_t> send;
            std::function<void(RpcResult)> done;
            uint64_t submitNs = 0; // PS2X_IOP_RPC_STATS: submit time (queue latency)
        };
        [[nodiscard]] bool canOffloadRpc(uint32_t sid);
        [[nodiscard]] bool submitRpc(AsyncRpc call);

        void onSifTransfer(const SifTransfer &transfer);

        // Physical IOP RAM access shared by the emulator, SIF DMA, and HLE services. Addresses are IOP addresses.
        [[nodiscard]] uint32_t allocateMemory(uint32_t size, uint32_t alignment = 16u);
        [[nodiscard]] bool freeMemory(uint32_t address);
        [[nodiscard]] bool readMemory(uint32_t address, void *destination, size_t size) const;
        [[nodiscard]] bool writeMemory(uint32_t address, const void *source, size_t size);
        [[nodiscard]] bool zeroMemory(uint32_t address, size_t size);
        [[nodiscard]] bool isMemoryRange(uint32_t address, size_t size) const;

        [[nodiscard]] DebugSnapshot debugSnapshot() const;

        // ---- Threaded mode ------------------------------------------------------------------------------------------
        // enableThread(true) serialises every public entry point above with one recursive lock and lets a host thread
        // run the IOP (and, through the host SPU2 hooks, SPU2 mixing) off the caller's thread. The thread is started
        // by the first publishEeCycles(). It consumes the published EE cycles (IOP time = EE / 8, exactly as
        // runEeCycles) and never runs ahead of them. runEeCycles() must not be used while the thread runs.
        // Everything the IOP does that touches the host runs on the IOP thread (or, inside handleRpc/loadModule/...,
        // on the calling thread while it holds the lock), so hosts must make IopHost callbacks thread-safe.
        // Without enableThread() every entry point is unlocked and unchanged.
        void enableThread(bool enable);
        [[nodiscard]] bool threadEnabled() const noexcept;
        // Total EE cycles elapsed so far (monotonic). Cheap: one atomic store; may be called from the hot path.
        void publishEeCycles(uint64_t totalEeCycles) noexcept;

        struct ThreadStats
        {
            uint64_t published = 0;     // EE cycles published
            uint64_t consumed = 0;      // EE cycles run by the IOP thread
            uint64_t maxLagCycles = 0;  // largest published - consumed seen when the IOP thread woke (EE cycles)
            uint64_t runNs = 0;         // wall time the IOP thread spent running the IOP (lock held)
            uint64_t eeCalls = 0;       // EE-side entry points that took the lock
            uint64_t eeWaitNs = 0;      // ... and the time they waited for it (only measured with PS2X_IOP_THREAD_STATS)
            uint64_t eeMaxWaitNs = 0;
        };
        [[nodiscard]] ThreadStats threadStats() const;

    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
