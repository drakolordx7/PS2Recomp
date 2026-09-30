#pragma once

#include <cstdint>

// Optional host VU1 (e.g. a recompiling VU). When set before PS2Memory::initialize(), VU1 code/data memory is the
// host's (never freed by the runtime), and MSCAL/MSCNT from VIF1 run the host VU instead of the built-in interpreter.
// codeGeneration increments whenever VU1 code memory was written (MPG or direct stores) since the runtime started;
// the host uses it to invalidate recompiled programs. vpuStat is VU0 VPU_STAT; the host updates the VU1 bits.
// Called on the EE game thread, or, with the VIF1 worker thread (PS2X_VIF1_THREAD, default on), on that worker thread:
// then every mscal/mscnt call and every XGKICK it produces runs on the worker, never on the EE thread.
struct PS2HostVu1
{
    uint8_t *codeMem = nullptr; // 16 KB
    uint8_t *dataMem = nullptr; // 16 KB
    void (*mscal)(uint32_t startPcBytes, uint32_t top, uint32_t itop, uint32_t fbrst, uint64_t codeGeneration, uint32_t *vpuStat) = nullptr;
    void (*mscnt)(uint32_t top, uint32_t itop, uint32_t fbrst, uint64_t codeGeneration, uint32_t *vpuStat) = nullptr;
    // Optional: called once on the VIF1 worker thread before it runs its first job (per-thread host VU setup).
    void (*bindWorkerThread)() = nullptr;
};

void ps2SetHostVu1(const PS2HostVu1 &hooks);
const PS2HostVu1 &ps2HostVu1();
