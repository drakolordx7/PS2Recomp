#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

class PS2Memory;
class PS2Runtime;

// VIF1 / VU1 / GIF worker thread (PS2X_VIF1_THREAD, MTVU-style).
//
// A DMA start on channel 1 (VIF1) or 2 (GIF) used to run synchronously inside the guest's CHCR store: chain walk ->
// VIF1 unpack / MPG / MSCAL (host VU1) -> XGKICK -> GIF arbiter -> host GS. With the worker, the EE thread still walks
// the DMA chain (that snapshots the source data, so the game may reuse its buffers as soon as the store returns), then
// hands the buffers to this thread as one job. The worker owns from then on: VIF1 registers and state, VU1 code/data
// memory, the host VU1 (all VU1 execution must happen on this thread once it exists), the GIF arbiter and PATH1/2/3
// ordering. Jobs run in FIFO order, so the GS sees exactly the order the synchronous code produced.
//
// EE <-> worker rules (implemented in ps2_memory.cpp / ps2_vif1_interpreter.cpp / EeScheduler.cpp):
//   * CHCR.STR of D1/D2 reads busy until the worker finished the job; completion (CHCR/QWC/D_STAT, DMAC handlers) is
//     applied on the EE thread, either lazily by a register read or by the scheduler event kVif1WorkerEventId.
//   * Anything on the EE thread that touches worker-owned state first waits for the worker to go idle (barrier):
//     VU1 code/data access, VIF1 register writes, GIF_STAT reads, VIF1 FIFO writes, direct GIF submissions, the public
//     processPendingTransfers(), and the guest vblank (so a frame's GIF packets are all queued in the host GS before
//     the vsync marker).
constexpr uint32_t kVif1WorkerEventId = 0x56494631u; // 'VIF1' (EeEventType::Dmac wake-up)

enum class Vif1BarrierReason : uint8_t
{
    Vu1Memory,
    VifRegister,
    GifStat,
    VifFifo,
    GifSubmit,
    SyncTransfer,
    Vsync,
    Other,
    Count
};

// Threaded mode is fixed for the life of the process by ps2Vif1WorkerBind(): PS2X_VIF1_THREAD != 0 and a host VU1
// (its JIT code is tied to the thread that first runs it, so it must never migrate).
void ps2Vif1WorkerBind(PS2Runtime &runtime, PS2Memory &memory);
bool ps2Vif1Active();          // threaded mode is on
// true: the scheduler waits for the worker at every vblank even if the host's vsync hook orders itself
// (PS2X_VIF1_VSYNC_SYNC=1; A/B switch).
bool ps2Vif1ForceVsyncBarrier();
bool ps2Vif1OnWorkerThread();  // calling thread is the worker

// EE thread: moves the pending GIF/VIF1 transfers into a worker job. Returns false when the caller has to process
// them synchronously (threaded mode off, or VIF0 transfers pending).
bool ps2Vif1DispatchAsync(PS2Memory &memory);

// EE thread: waits until the worker is idle and applies finished jobs. Cheap no-op when threaded mode is off, the
// worker is idle, or the caller is the worker.
void ps2Vif1Barrier(Vif1BarrierReason reason);
// EE thread: applies finished jobs without waiting.
void ps2Vif1PollCompletions();
// EE thread: after ps2Vif1PollCompletions(), true while a job started on this channel (0x10009000 / 0x1000A000) runs.
bool ps2Vif1ChannelBusy(uint32_t channelBase);
// EE thread, scheduler event kVif1WorkerEventId.
void ps2Vif1ApplyEvent(PS2Runtime &runtime);
// EE thread: FBRST as the VU1 D/T-bit stop enables see it (sampled by the scheduler each vblank).
void ps2Vif1SetFbrst(uint32_t fbrst);

// MSCAL / MSCNT of the VIF1 stream: run the host VU1 on the worker thread (directly when already there, else as a
// blocking call on the worker). Returns false when threaded mode is off (caller uses the runtime callback).
bool ps2Vif1Mscal(uint32_t startPc, uint32_t top, uint32_t itop);
bool ps2Vif1Mscnt(uint32_t top, uint32_t itop);

void ps2Vif1WorkerShutdown();

// ps2_vif1_worker_win.cpp: priority / core placement of the worker thread (calling thread).
void ps2Vif1TuneWorkerThread();

// PS2X_VIF1_STATS=1 accounting for time the worker spends handing GIF packets to the host GS.
bool ps2Vif1StatsEnabled();
void ps2Vif1StatAddGifNs(uint64_t ns);
void ps2Vif1StatAddWalkNs(uint64_t ns); // EE: DMA chain walk + snapshot copy (in the DMA-start store)
uint64_t ps2Vif1NowNs();
void ps2Vif1StatMpg(bool sameAsVuMemory); // an MPG upload (same = identical to the code already in VU1 memory)

// DMA chain buffers. A gameplay frame is one ~3 MB VIF1 chain; allocating, growing (vector insert per tag) and freeing
// such a buffer every frame cost page faults and reallocation copies on the EE thread. The walk takes a buffer from
// this pool (empty, with the capacity of the previous chain of that channel); whoever consumed it gives it back
// (the worker after a job, the synchronous path after processing). Thread safe.
std::vector<uint8_t> ps2ChainBufferAcquire(uint32_t channelBase);
void ps2ChainBufferNoteSize(uint32_t channelBase, size_t bytes);
void ps2ChainBufferRecycle(std::vector<uint8_t> &&buf);

// Completed-DMAC causes queued by PS2Memory and not yet consumed (0 = nothing to drain). Lets PS2Runtime::Store* skip the
// drain (and its mutex) after every MMIO/scratchpad store. Defined in ps2_memory.cpp.
extern std::atomic<uint32_t> g_ps2CompletedDmacPending;

// PS2X_CHAIN_VERIFY=1 (measurement aid for a zero-copy DMA chain): the walk registers every copied segment with its host
// source pointer; when the buffer is recycled (job finished) the copy is compared with the live source and the
// mismatches (source overwritten by the guest between the CHCR store and job completion) are counted per DMAtag id.
struct ChainVerifySeg
{
    const uint8_t *src;
    uint32_t n;
    uint32_t dst;
    uint32_t id;
};
bool ps2ChainVerifyEnabled();
void ps2ChainVerifyRegister(const uint8_t *bufData, std::vector<ChainVerifySeg> &&segs);
void ps2ChainVerifyCheck(const uint8_t *bufData);
