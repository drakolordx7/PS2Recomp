#include <cstdlib>
#include "iop_emulator.h"
#include "imports/iop_cdvd.h"
#include "core/iop_cpu.h"
#include "imports/iop_heaplib.h"
#include "imports/iop_imports.h"
#include "imports/iop_intrman.h"
#include "imports/iop_ioman.h"
#include "core/iop_kernel.h"
#include "imports/iop_loadcore.h"
#include "core/iop_memory.h"
#include "services/iop_module_loader.h"
#include "services/iop_rpc.h"
#include "imports/iop_stdio.h"
#include "imports/iop_sysclib.h"
#include "imports/iop_sysmem.h"
#include "imports/iop_timrman.h"
#include "imports/iop_vblank.h"
#include "iop_emulator_const.h"
#include "ps2x/iop/iop_host_spu2.h"
#include "iop_trace.h"

#include <cstdio>

#include <algorithm>
#include <cctype>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <utility>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kRamSize = IopMemory::RamSize;
        constexpr uint32_t kKernelHeapBase = IopMemory::HeapBase;
        constexpr uint32_t kKernelHeapLimit = IopMemory::HeapLimit;
        constexpr uint32_t kCallStackBase = kKernelHeapLimit;
        constexpr uint32_t kCallStackLimit = 0x001FFF00u;
        constexpr uint32_t kCallStackSize = 0x2000u;
        constexpr uint32_t kCallStackCapacity = (kCallStackLimit - kCallStackBase) / kCallStackSize;
        constexpr uint64_t kCdvdCompletionCycles = 128u;
        constexpr int kIrqSpu2 = 9;        // SPU2 IRQ (INTC line 9)
        constexpr int kIrqDmaSpu2Core0 = 0x24; // IOP DMA ch4 end
        constexpr int kIrqDmaSpu2Core1 = 0x28; // IOP DMA ch7 end

        uint32_t physicalAddress(uint32_t address)
        {
            return IopMemory::physicalAddress(address);
        }

        int32_t sign16(uint32_t value)
        {
            return static_cast<int16_t>(value & 0xFFFFu);
        }

        bool iequals(std::string_view lhs, std::string_view rhs)
        {
            if (lhs.size() != rhs.size())
                return false;
            for (size_t i = 0; i < lhs.size(); ++i)
            {
                if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
                    std::tolower(static_cast<unsigned char>(rhs[i])))
                    return false;
            }
            return true;
        }

    }

    class IopEmulator::Impl final : public IopGuestExecutor
    {
    public:
        using CpuState = IopCpuState;

        struct Module
        {
            int id = 0;
            std::string path;
            std::string name;
            uint32_t base = 0;
            uint32_t size = 0;
            uint32_t entry = 0;
            uint32_t gp = 0;
            bool resident = false;
        };

        struct GuestCallback
        {
            uint32_t function = 0;
            uint32_t gp = 0;
        };

        struct ScheduledGuestCallback
        {
            uint32_t function = 0u;
            uint32_t gp = 0u;
            uint32_t argument = 0u;
        };

        explicit Impl(IopHost &hostRef)
            : host(hostRef),
              sysmem(host, memory),
              kernel(memory),
              cdvd(host, memory, kernel),
              vblank(kernel),
              rpc(host, memory, kernel),
              sysclib(memory),
              stdio(host, memory),
              heaplib(memory),
              intrman(memory),
              timrman(),
              ioman(memory),
              cpuCore(memory),
              imports(memory),
              loadcore(memory, imports)
        {
            memory.setCycleSource(&totalCycles, &spu2CycleEpoch);
            reset();
        }

        void reset()
        {
            spu2CycleEpoch += totalCycles; // host SPU2 time stays monotonic across IOP resets
            (void)takeSpu2Pending();
            memory.reset();
            kernel.reset();
            modules.clear();
            imports.reset();
            rpc.reset();
            cdvd.reset();
            intrman.reset();
            timrman.reset();
            ioman.reset();
            pendingDmaInterrupts.clear();
            pendingGuestCallbacks.clear();
            nextModuleId = 1;
            moduleCursor = kModuleLoadBase;
            totalCycles = 0;
            totalInstructions = 0;
            eeCycleCarry = 0;
            activeCpu = nullptr;
            lastError.clear();
            servicingDmaInterrupts = false;
            servicingGuestCallbacks = false;
            callDepth = 0u;
            secrMcCommandHandler = {};
            secrMcDevIdHandler = {};
            checkKelfPathCallback = {};
        }

        uint8_t read8(uint32_t address) const
        {
            return memory.read8(address);
        }

        uint16_t read16(uint32_t address) const
        {
            return memory.read16(address);
        }

        uint32_t read32(uint32_t address) const
        {
            return memory.read32(address);
        }

        void write8(uint32_t address, uint8_t value)
        {
            memory.write8(address, value);
            schedulePendingDma();
        }

        void write16(uint32_t address, uint16_t value)
        {
            memory.write16(address, value);
            schedulePendingDma();
        }

        void write32(uint32_t address, uint32_t value)
        {
            memory.write32(address, value);
            schedulePendingDma();
        }

        void schedulePendingDma()
        {
            if (const auto dma = memory.takeDmaStart())
                pendingDmaInterrupts[dma->irq] = totalCycles + dma->delayCycles;
            if (const uint32_t spu2 = takeSpu2Pending())
                scheduleSpu2Interrupts(spu2);
        }

        // Module start routines and RPC server functions run synchronously (callFunction), outside any IOP thread,
        // so the kernel cannot block them: WaitEventFlag used to return at once. libsd waits that way for its
        // SPU2 DMA (sceSdVoiceTransStatus(ch, SD_TRANS_STATUS_WAIT)); returning early left its "transfer done" flag
        // to be set by the later DMA interrupt, so the next status poll saw a finished transfer too early and every
        // following sceSdVoiceTrans was refused as busy (Killzone's sound banks never reached SPU2 RAM).
        // Instead, advance IOP time to the next pending interrupt / guest callback / host SPU2 event and service them
        // until the flag is set. With nothing pending (only another thread could set the flag), or after 1 s of IOP
        // time, give up and return as before.
        bool waitEventFlagOutsideThread(int id, uint32_t bits, uint32_t mode)
        {
            if (servicingDmaInterrupts)
                return kernel.eventFlagSatisfied(id, bits, mode);
            const uint64_t limit = totalCycles + kIopClockHz;
            while (!kernel.eventFlagSatisfied(id, bits, mode))
            {
                uint64_t next = UINT64_MAX;
                for (const auto &[irq, completionCycle] : pendingDmaInterrupts)
                    next = std::min(next, completionCycle);
                if (!pendingGuestCallbacks.empty())
                    next = std::min(next, pendingGuestCallbacks.begin()->first);
                const IopHostSpu2 &spu2 = iopHostSpu2();
                if (spu2.nextEvent)
                {
                    const uint64_t event = spu2.nextEvent();
                    if (event != UINT64_MAX)
                        next = std::min(next, event > spu2CycleEpoch ? event - spu2CycleEpoch : 0u);
                }
                if (next == UINT64_MAX || totalCycles >= limit)
                    return false;
                totalCycles = std::min(limit, std::max(totalCycles + 1u, next));
                advanceSpu2();
                servicePendingDmaInterrupts();
                servicePendingGuestCallbacks();
            }
            return true;
        }

        // Host SPU2 (ps2x/iop/iop_host_spu2.h): interrupts it queued become due now.
        void scheduleSpu2Interrupts(uint32_t pending)
        {
            if (pending & 1u)
                pendingDmaInterrupts[kIrqSpu2] = totalCycles;
            if ((pending & 2u) && memory.completeSpu2Dma(0))
                pendingDmaInterrupts[kIrqDmaSpu2Core0] = totalCycles;
            if ((pending & 4u) && memory.completeSpu2Dma(1))
                pendingDmaInterrupts[kIrqDmaSpu2Core1] = totalCycles;
        }

        void advanceSpu2()
        {
            const IopHostSpu2 &spu2 = iopHostSpu2();
            if (spu2.advance)
                spu2.advance(spu2CycleEpoch + totalCycles);
            schedulePendingDma();
        }

        // While every thread sleeps, do not skip past the SPU2's next event (DMA progress) or more than maxIdleStep,
        // so its interrupts reach the guest on time.
        uint64_t limitIdleSkip(uint64_t nextWake) const
        {
            const IopHostSpu2 &spu2 = iopHostSpu2();
            if (!spu2.advance)
                return nextWake;
            nextWake = std::min(nextWake, totalCycles + std::max<uint64_t>(spu2.maxIdleStep, 1u));
            if (spu2.nextEvent)
            {
                const uint64_t event = spu2.nextEvent();
                if (event != UINT64_MAX)
                {
                    const uint64_t local = event > spu2CycleEpoch ? event - spu2CycleEpoch : 0u;
                    nextWake = std::min(nextWake, std::max(local, totalCycles + 1u));
                }
            }
            return nextWake;
        }

        bool readRam(uint32_t address, void *destination, size_t size) const
        {
            return memory.readRam(address, destination, size);
        }

        bool writeRam(uint32_t address, const void *source, size_t size)
        {
            return memory.writeRam(address, source, size);
        }

        bool zeroRam(uint32_t address, size_t size)
        {
            return memory.zeroRam(address, size);
        }

        bool isHardwareAddress(uint32_t phys) const
        {
            return memory.isHardwareAddress(phys);
        }

        uint32_t allocate(uint32_t size, uint32_t alignment = 16u, std::optional<uint32_t> fixed = std::nullopt)
        {
            return memory.allocate(size, alignment, fixed);
        }

        bool freeAllocation(uint32_t address)
        {
            return memory.freeAllocation(address);
        }

        void log(LogLevel level, std::string_view text)
        {
            host.log(level, text);
        }

        bool checkInterrupt(CpuState &cpu)
        {
            const uint32_t status = cpu.cop0[12];
            if ((status & 1u) == 0u)
                return false;
            if ((status & 0x2u) != 0u)
                return false;
            const bool pending = memory.interruptControl() != 0u && (memory.interruptStatus() & memory.interruptMask()) != 0u;
            if (!pending)
                return false;
            cpu.cop0[13] |= 0x400u;
            cpuCore.raiseException(cpu, 0u, cpu.pc, false);
            return true;
        }

        enum class ImportDisposition
        {
            Handled,
            JumpToGuest,
            Missing,
        };

        ImportDisposition dispatchImport(const IopImportCall &call, CpuState &cpu)
        {
            const uint32_t a0 = cpu.gpr[4];
            auto setV0 = [&](uint32_t value)
            {
                cpu.gpr[2] = value;
            };

            if (iequals(call.library, "sysmem") && sysmem.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "cdvdman") && cdvd.dispatchImport(call.ordinal, cpu))
            {
                if (const auto callback = cdvd.takeCompletionCallback())
                {
                    pendingGuestCallbacks.emplace(
                        totalCycles + kCdvdCompletionCycles,
                        ScheduledGuestCallback{
                            callback->address,
                            callback->gp,
                            callback->reason,
                        });
                }
                return ImportDisposition::Handled;
            }

            if (iequals(call.library, "loadcore") && loadcore.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "thbase") || iequals(call.library, "threadman"))
            {
                if (iopSyncTrace() && (call.ordinal == 24u || call.ordinal == 33u) && !kernel.inThread())
                    std::fprintf(stderr, "[iop-sync] %s outside a thread, ra=0x%x\n",
                                 call.ordinal == 24u ? "SleepThread" : "DelayThread", cpu.gpr[31]);
                return kernel.dispatchThreadImport(call.ordinal, cpu, totalCycles)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thsemap"))
            {
                if ((call.ordinal == 8u || call.ordinal == 9u) && !kernel.inThread())
                {
                    if (iopSyncTrace() && kernel.semaphoreCount(static_cast<int>(a0)) == 0)
                        std::fprintf(stderr, "[iop-sync] %s(%d) outside a thread, count=0, ra=0x%x cycle=%llu%s\n",
                                     call.ordinal == 8u ? "WaitSema" : "PollSema", static_cast<int>(a0), cpu.gpr[31],
                                     static_cast<unsigned long long>(totalCycles), kernel.describeThreads().c_str());
                    if (call.ordinal == 8u)
                        (void)waitSemaphoreOutsideThread(static_cast<int>(a0));
                }
                return kernel.dispatchSemaphoreImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thevent"))
            {
                if (call.ordinal == 10u && !kernel.inThread()) // WaitEventFlag outside a thread
                    (void)waitEventFlagOutsideThread(static_cast<int>(a0), cpu.gpr[5], cpu.gpr[6]);
                return kernel.dispatchEventImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "sifcmd"))
            {
                return rpc.dispatchSifCmdImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "intrman") && intrman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "secrman"))
            {
                switch (call.ordinal)
                {
                case 4: // SecrSetMcCommandHandler
                    secrMcCommandHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                case 5: // SecrSetMcDevIDHandler
                    secrMcDevIdHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                default:
                    break;
                }
            }
            if (iequals(call.library, "modload") && call.ordinal == 13u)
            {
                checkKelfPathCallback = {a0, cpu.gpr[28]};
                setV0(0);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "ioman") && ioman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sifman"))
            {
                return rpc.dispatchSifManImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "vblank") && vblank.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "timrman") && timrman.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "dmacman"))
            {
                setV0(0);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "stdio") && stdio.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sysclib"))
            {
                return sysclib.dispatchImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "heaplib") && heaplib.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            const uint32_t target = imports.resolve(call.library, call.ordinal, call.version);
            if (target != 0u)
            {
                cpu.pc = target;
                cpu.branchPending = false;
                return ImportDisposition::JumpToGuest;
            }

            std::ostringstream out;
            out << "[IOP] unhandled import " << call.library << ':' << call.ordinal
                << " version=0x" << std::hex << call.version << " pc=0x" << cpu.pc;
            log(LogLevel::Warning, out.str());
            setV0(0);
            return ImportDisposition::Missing;
        }

        bool step(CpuState &cpu)
        {
            if (cpu.stopped)
                return false;
            if (cpu.pc == kThreadReturnSentinel || cpu.pc == kCallReturnSentinel)
            {
                cpu.stopped = true;
                return false;
            }
            if (physicalAddress(cpu.pc) >= kRamSize)
            {
                std::ostringstream out;
                out << "[IOP] execution outside RAM pc=0x" << std::hex << cpu.pc;
                log(LogLevel::Error, out.str());
                cpu.stopped = true;
                return false;
            }
            if (checkInterrupt(cpu))
                return true;

            if (const auto import = imports.decode(cpu.pc))
            {
                const ImportDisposition disposition = dispatchImport(*import, cpu);
                ++totalInstructions;
                ++totalCycles;
                if (disposition == ImportDisposition::JumpToGuest)
                    return true;
                cpu.pc = cpu.gpr[31];
                cpu.branchPending = false;
                return !cpu.stopped;
            }

            const bool running = cpuCore.executeInstruction(cpu);
            schedulePendingDma();
            ++totalInstructions;
            ++totalCycles;
            return running;
        }

        uint32_t runCpu(CpuState &cpu, uint32_t instructionBudget)
        {
            CpuState *previous = activeCpu;
            activeCpu = &cpu;
            const uint64_t start = totalInstructions;
            while (!cpu.stopped && !cpu.yielded && totalInstructions - start < instructionBudget)
            {
                if (!step(cpu))
                    break;
                if (!servicingDmaInterrupts && !pendingDmaInterrupts.empty())
                    servicePendingDmaInterrupts();
                if (!servicingGuestCallbacks && !pendingGuestCallbacks.empty())
                    servicePendingGuestCallbacks();
            }
            activeCpu = previous;
            return static_cast<uint32_t>(totalInstructions - start);
        }

        uint32_t callFunction(uint32_t address,
                              uint32_t a0,
                              uint32_t a1,
                              uint32_t a2,
                              uint32_t a3,
                              uint32_t gp,
                              uint32_t budget = kMaxCallInstructions)
        {
            struct CallDepthGuard
            {
                uint32_t &depth;
                ~CallDepthGuard() { --depth; }
            };

            const uint32_t depth = callDepth++;
            const CallDepthGuard depthGuard{callDepth};
            CpuState cpu{};
            cpu.pc = address;
            cpu.gpr[4] = a0;
            cpu.gpr[5] = a1;
            cpu.gpr[6] = a2;
            cpu.gpr[7] = a3;
            cpu.gpr[28] = gp;
            if (depth < kCallStackCapacity)
            {
                const uint32_t stackTop = kCallStackLimit - depth * kCallStackSize;
                cpu.gpr[29] = stackTop - 32u;
            }
            else if (activeCpu && activeCpu->gpr[29] > kCallStackBase + kStackGuardBytes)
            {
                // Extremely deep re-entrancy borrows unused space below the
                // suspended caller's live frame. Stack growth remains away
                // from the caller, so its saved registers stay intact.
                cpu.gpr[29] = (activeCpu->gpr[29] - kStackGuardBytes) & ~15u;
            }
            else
            {
                cpu.gpr[29] = kCallStackBase - 32u;
            }
            cpu.gpr[31] = kCallReturnSentinel;
            runCpu(cpu, budget);
            return cpu.gpr[2];
        }

        uint32_t executeGuestFunction(uint32_t address,
                                      uint32_t a0,
                                      uint32_t a1,
                                      uint32_t a2,
                                      uint32_t a3,
                                      uint32_t gp) override
        {
            return callFunction(address, a0, a1, a2, a3, gp);
        }

        uint32_t executeGuestFunctionWithBudget(uint32_t address,
                                                uint32_t a0,
                                                uint32_t a1,
                                                uint32_t a2,
                                                uint32_t a3,
                                                uint32_t gp,
                                                uint32_t instructionBudget) override
        {
            return callFunction(address, a0, a1, a2, a3, gp, instructionBudget);
        }

        // Not that good to use exception handling for control flow but will do for now
        void servicePendingDmaInterrupts()
        {
            if (servicingDmaInterrupts || pendingDmaInterrupts.empty())
                return;

            servicingDmaInterrupts = true;

            std::vector<int> completed;
            for (auto it = pendingDmaInterrupts.begin(); it != pendingDmaInterrupts.end();)
            {
                if (it->second > totalCycles)
                {
                    ++it;
                    continue;
                }
                completed.push_back(it->first);
                it = pendingDmaInterrupts.erase(it);
            }
            try
            {
                for (const int irq : completed)
                    (void)intrman.dispatchInterrupt(irq, *this);
            }
            catch (...)
            {
                servicingDmaInterrupts = false;
                throw;
            }
            servicingDmaInterrupts = false;
        }

        void servicePendingGuestCallbacks()
        {
            if (servicingGuestCallbacks || pendingGuestCallbacks.empty())
                return;

            std::vector<ScheduledGuestCallback> callbacks;
            for (auto it = pendingGuestCallbacks.begin(); it != pendingGuestCallbacks.end();)
            {
                if (it->first > totalCycles)
                    break;
                callbacks.push_back(it->second);
                it = pendingGuestCallbacks.erase(it);
            }
            if (callbacks.empty())
                return;

            servicingGuestCallbacks = true;
            try
            {
                for (const ScheduledGuestCallback &callback : callbacks)
                {
                    if (callback.function != 0u)
                    {
                        (void)callFunction(callback.function,
                                           callback.argument,
                                           0u,
                                           0u,
                                           0u,
                                           callback.gp,
                                           100000u);
                    }
                }
            }
            catch (...)
            {
                servicingGuestCallbacks = false;
                throw;
            }
            servicingGuestCallbacks = false;
        }

        // PS2X_IOP_SYNC_TRACE: once per IOP second, the thread states and the words listed in PS2X_IOP_WATCH
        // (comma-separated IOP addresses).
        uint64_t nextTraceCycle = 0;
        bool inScheduler = false; // runCycles or an outside-thread wait is running IOP threads
        void traceHeartbeat()
        {
            if (totalCycles < nextTraceCycle)
                return;
            nextTraceCycle = totalCycles + kIopClockHz;
            std::string line = "[iop-sync] cycle=" + std::to_string(totalCycles) + kernel.describeThreads();
            if (const char *watch = std::getenv("PS2X_IOP_WATCH"))
            {
                char item[48];
                for (const char *p = watch; *p;)
                {
                    char *end = nullptr;
                    const unsigned long address = std::strtoul(p, &end, 0);
                    if (end == p)
                        break;
                    std::snprintf(item, sizeof(item), " %lx=%x", address, memory.read32(static_cast<uint32_t>(address)));
                    line += item;
                    p = *end == ',' ? end + 1 : end;
                }
            }
            std::fprintf(stderr, "%s\n", line.c_str());
        }

        // Earliest IOP cycle at which anything can happen while every thread is idle (thread wake, DMA/callback
        // completion, timer, SPU2 event or idle-step bound), valid until something external touches the IOP.
        // runCycles is called on nearly every EE checkpoint with a few cycles; while the IOP sleeps this lets those
        // calls just advance the clock instead of re-scanning every event source. 0 = unknown.
        uint64_t idleUntil = 0;

        void runCycles(uint64_t cycles) noexcept
        {
            static const bool fastPath = []() {
                const char *v = std::getenv("PS2X_IOP_IDLE_FASTPATH");
                return !(v && v[0] == '0');
            }();
            if (fastPath && idleUntil != 0 && totalCycles + cycles < idleUntil)
            {
                totalCycles += cycles; // exactly what the idle branch below would do
                return;
            }
            idleUntil = 0;
            if (iopSyncTrace())
                traceHeartbeat();
            const SchedulerGuard guard{inScheduler};
            try
            {
                const uint64_t target = totalCycles + cycles;
                while (totalCycles < target)
                    idleUntil = scheduleStep(target);
                advanceSpu2();
            }
            catch (...)
            {
                // Runtime scheduling must never throw through EeScheduler::accountCycles().
            }
        }

        struct SchedulerGuard
        {
            bool &flag;
            const bool previous;
            explicit SchedulerGuard(bool &f) : flag(f), previous(f) { flag = true; }
            ~SchedulerGuard() { flag = previous; }
        };

        // One scheduler step toward `target`: service due interrupts/callbacks/timers, then run the best ready
        // thread for a slice or, with every thread idle, advance the clock to the next event (at most `target`).
        // Returns the idle wake cycle (the idle fast path bound), 0 when a thread ran.
        uint64_t scheduleStep(uint64_t target)
        {
            advanceSpu2();
            servicePendingDmaInterrupts();
            servicePendingGuestCallbacks();
            timrman.serviceDue(totalCycles, *this);
            IopThread *next = kernel.beginNextReady(totalCycles);
            if (!next)
            {
                uint64_t nextWake = kernel.nextWakeCycle(UINT64_MAX);
                for (const auto &[irq, completionCycle] : pendingDmaInterrupts)
                    nextWake = std::min(nextWake, completionCycle);
                if (!pendingGuestCallbacks.empty())
                    nextWake = std::min(nextWake, pendingGuestCallbacks.begin()->first);
                nextWake = timrman.nextEventCycle(nextWake);
                nextWake = limitIdleSkip(nextWake);
                const uint64_t idleWake = nextWake > totalCycles ? nextWake : 0;
                totalCycles = std::max(totalCycles + 1u, std::min(target, nextWake));
                return idleWake;
            }
            const uint64_t before = totalCycles;
            runCpu(next->cpu, static_cast<uint32_t>(std::min<uint64_t>(kDefaultSlice, target - totalCycles)));
            kernel.endTimeslice(*next, kThreadReturnSentinel);
            if (totalCycles == before)
                ++totalCycles;
            return 0;
        }

        // RPC server functions and module start routines run synchronously (callFunction), outside any IOP thread.
        // WaitSema there used to return at once without taking the semaphore, so the caller ran its critical section
        // while the IOP thread holding the semaphore was preempted inside its own. Killzone's PFILE_R.IRX: the RPC
        // appends a read request to the streaming queue while the streaming thread is between `head = cur->next`
        // and `if (head == 0) tail = 0`; the tail is then lost, the next request is never linked, and the EE waits
        // forever for its completion (boot/loading hang at 0x14A028).
        // On a real IOP the RPC server thread blocks and the holder runs until SignalSema. Do the same: run the IOP
        // threads (and due interrupts/timers) until the semaphore is signalled, then take it. Not from an interrupt
        // handler or guest callback, nor while the scheduler is already running a thread; gives up after 1 s of IOP
        // time. PS2X_IOP_OUTSIDE_SEMA_WAIT=0 restores the old behaviour.
        bool waitSemaphoreOutsideThread(int id)
        {
            static const bool enabled = []() {
                const char *v = std::getenv("PS2X_IOP_OUTSIDE_SEMA_WAIT");
                return !(v && v[0] == '0');
            }();
            if (kernel.semaphoreCount(id) != 0)
                return true; // available, or an unknown id (WaitSema reports it)
            if (!enabled || inScheduler || servicingDmaInterrupts || servicingGuestCallbacks)
                return false;
            const SchedulerGuard guard{inScheduler};
            const uint64_t start = totalCycles;
            const uint64_t limit = totalCycles + kIopClockHz;
            kernel.setOutsideSemaphoreWait(id);
            try
            {
                while (kernel.semaphoreCount(id) == 0 && totalCycles < limit)
                    (void)scheduleStep(limit);
            }
            catch (...)
            {
                kernel.setOutsideSemaphoreWait(0);
                throw;
            }
            kernel.setOutsideSemaphoreWait(0);
            const bool available = kernel.semaphoreCount(id) > 0;
            if (iopSyncTrace() || !available)
                std::fprintf(stderr, "[iop-sync] WaitSema(%d) outside a thread waited %llu cycles: %s\n", id,
                             static_cast<unsigned long long>(totalCycles - start), available ? "taken" : "TIMED OUT");
            return available;
        }

        ModuleLoadResult loadImage(std::string path, std::span<const uint8_t> image, const void *arguments, uint32_t argumentSize)
        {
            ModuleLoadResult result{true, -1, -1};
            const IopImageLoadResult loaded = IopModuleLoader::load(image, memory, moduleCursor);
            moduleCursor = loaded.nextModuleCursor;
            if (!loaded)
            {
                if (loaded.error == IopImageLoadError::InvalidElf)
                    log(LogLevel::Error, "[IOP] rejected invalid/non-MIPS IRX ELF");
                else if (loaded.error == IopImageLoadError::ArenaExhausted)
                    log(LogLevel::Error, "[IOP] module arena exhausted");
                return result;
            }
            if (!loaded.relocationsComplete)
                log(LogLevel::Warning, "[IOP] one or more IRX relocations were unsupported");

            Module module;
            module.id = nextModuleId++;
            module.path = std::move(path);
            const size_t slash = module.path.find_last_of("/\\:");
            module.name = slash == std::string::npos ? module.path : module.path.substr(slash + 1u);
            module.base = loaded.base;
            module.size = loaded.size;
            module.entry = loaded.entry;
            module.gp = loaded.gp;

            uint32_t args = 0u;
            if (arguments && argumentSize)
            {
                args = allocate(argumentSize + 1u, 16u);
                if (args)
                {
                    writeRam(args, arguments, argumentSize);
                    write8(args + argumentSize, 0u);
                }
            }
            const uint32_t startResult = callFunction(module.entry, argumentSize, args, 0u, 0u, module.gp);
            if (args)
                freeAllocation(args);
            module.resident = startResult == 0u || startResult == 2u;
            result.moduleId = module.id;
            result.startResult = static_cast<int32_t>(startResult);
            modules[module.id] = std::move(module);

            std::ostringstream out;
            out << "[IOP] loaded IRX id=" << result.moduleId
                << " entry=0x" << std::hex << modules[result.moduleId].entry
                << " base=0x" << modules[result.moduleId].base
                << " start=" << std::dec << result.startResult;
            log(LogLevel::Info, out.str());
            return result;
        }

        ModuleLoadResult loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readWholeHostFile(host, path, image))
            {
                log(LogLevel::Warning, std::string("[IOP] failed to open IRX '") + std::string(path) + "'");
                return {true, -1, -1};
            }
            return loadImage(std::string(path), image, arguments, argumentSize);
        }

        ModuleLoadResult loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readElfFromGuest(host, guestAddress, image))
                return {true, -1, -1};
            std::ostringstream tag;
            tag << "buffer@0x" << std::hex << guestAddress;
            return loadImage(tag.str(), image, arguments, argumentSize);
        }

        bool stopModule(int32_t moduleId, int32_t *result)
        {
            auto it = modules.find(moduleId);
            if (it == modules.end())
                return false;
            // A removable IRX normally exposes a stop entry through module metadata. We do not guess it; terminate owned execution and release the image cleanly.
            kernel.terminateThreadsInRange(it->second.base, it->second.size);
            rpc.removeServersInRange(it->second.base, it->second.size);
            imports.eraseRange(it->second.base, it->second.size);
            modules.erase(it);
            kernel.cleanupDeadThreads();
            if (result)
                *result = 0;
            return true;
        }

        IopHost &host;
        IopMemory memory;
        IopSysmem sysmem;
        IopKernel kernel;
        IopCdvd cdvd;
        IopVblank vblank;
        IopRpcBridge rpc;
        IopSysclib sysclib;
        IopStdio stdio;
        IopHeaplib heaplib;
        IopIntrman intrman;
        IopTimrman timrman;
        IopIoman ioman;
        IopCpuCore cpuCore;
        IopImportRegistry imports;
        IopLoadcore loadcore;
        std::map<int, Module> modules;
        std::map<int, uint64_t> pendingDmaInterrupts;
        std::multimap<uint64_t, ScheduledGuestCallback> pendingGuestCallbacks;
        uint32_t nextModuleId = 1;
        uint32_t moduleCursor = kModuleLoadBase;
        uint64_t totalCycles = 0;
        uint64_t spu2CycleEpoch = 0; // cycles run before the last reset (host SPU2 clock)
        uint64_t totalInstructions = 0;
        uint64_t eeCycleCarry = 0;
        CpuState *activeCpu = nullptr;
        std::string lastError;
        bool servicingDmaInterrupts = false;
        bool servicingGuestCallbacks = false;
        uint32_t callDepth = 0u;
        GuestCallback secrMcCommandHandler;
        GuestCallback secrMcDevIdHandler;
        GuestCallback checkKelfPathCallback;
    };

    IopEmulator::IopEmulator(IopHost &host)
        : m_impl(std::make_unique<Impl>(host))
    {
    }

    IopEmulator::~IopEmulator() = default;

    void IopEmulator::reset()
    {
        m_impl->idleUntil = 0;
        m_impl->reset();
    }

    ModuleLoadResult IopEmulator::loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
    {
        m_impl->idleUntil = 0;
        return m_impl->loadModule(path, arguments, argumentSize);
    }

    ModuleLoadResult IopEmulator::loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
    {
        m_impl->idleUntil = 0;
        return m_impl->loadModuleBuffer(guestAddress, arguments, argumentSize);
    }

    bool IopEmulator::stopModule(int32_t moduleId, int32_t *result)
    {
        m_impl->idleUntil = 0;
        return m_impl->stopModule(moduleId, result);
    }

    void IopEmulator::runEeCycles(uint64_t eeCycles) noexcept
    {
        const uint64_t total = m_impl->eeCycleCarry + eeCycles;
        const uint64_t iopCycles = total / 8u;
        m_impl->eeCycleCarry = total % 8u;
        if (iopCycles)
            m_impl->runCycles(iopCycles);
    }

    RpcResult IopEmulator::handleRpc(const RpcRequest &request, const std::vector<uint8_t> *sendPayload)
    {
        m_impl->idleUntil = 0;
        return m_impl->rpc.handleRpc(request, *m_impl, sendPayload);
    }

    bool IopEmulator::hasRpcServer(uint32_t sid) const noexcept
    {
        return m_impl->rpc.hasServer(sid);
    }

    void IopEmulator::onSifTransfer(const SifTransfer &transfer)
    {
        m_impl->idleUntil = 0;
        m_impl->rpc.onSifTransfer(transfer);
    }

    uint32_t IopEmulator::allocateMemory(uint32_t size, uint32_t alignment)
    {
        m_impl->idleUntil = 0;
        return m_impl->memory.allocate(size, alignment);
    }

    bool IopEmulator::freeMemory(uint32_t address)
    {
        m_impl->idleUntil = 0;
        return m_impl->memory.freeAllocation(address);
    }

    bool IopEmulator::readMemory(uint32_t address, void *destination, size_t size) const
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.readRam(address, destination, size);
    }

    bool IopEmulator::writeMemory(uint32_t address, const void *source, size_t size)
    {
        m_impl->idleUntil = 0;
        return isMemoryRange(address, size) &&
               m_impl->memory.writeRam(address, source, size);
    }

    bool IopEmulator::zeroMemory(uint32_t address, size_t size)
    {
        m_impl->idleUntil = 0;
        return isMemoryRange(address, size) &&
               m_impl->memory.zeroRam(address, size);
    }

    bool IopEmulator::isMemoryRange(uint32_t address, size_t size) const
    {
        const bool physicalSegment = address < IopMemory::RamSize;
        const bool cachedSegment = address >= 0x80000000u && address < 0x80200000u;
        const bool uncachedSegment = address >= 0xA0000000u && address < 0xA0200000u;
        if (!physicalSegment && !cachedSegment && !uncachedSegment)
            return false;
        const uint32_t physical = IopMemory::physicalAddress(address);
        return physical <= IopMemory::RamSize && size <= IopMemory::RamSize - physical;
    }

    uint64_t IopEmulator::cycles() const noexcept
    {
        return m_impl->totalCycles;
    }

    uint64_t IopEmulator::instructions() const noexcept
    {
        return m_impl->totalInstructions;
    }

    uint32_t IopEmulator::loadedModuleCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->modules.size());
    }

    uint32_t IopEmulator::threadCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->kernel.threadCount());
    }

    uint32_t IopEmulator::rpcServerCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->rpc.serverCount());
    }

}
