#pragma once

#include <cstdint>

struct R5900Context;

// Optional host VU0 micro mode (e.g. a recompiling VU). When set before PS2Memory::initialize(), VU0 code/data memory
// is the host's (never freed by the runtime), and VCALLMS/VCALLMSR run the host VU0 instead of the built-in
// interpreter. The hook is only used while the runtime's VU0 memory is the host's (installed before initialize()).
//
// callms runs the microprogram at startPcBytes to completion, synchronously. VU0's registers are the EE's COP2
// registers in ctx (vu0_vf, vi, vu0_acc, vu0_q, vu0_i, vu0_r, vu0_status, vu0_mac_flags, vu0_clip_flags, vu0_fbrst);
// the host reads them before the run and writes the results back (plus vu0_tpc/vu0_pc and the VU0 bits of
// vu0_vpu_stat). codeGeneration increments whenever VU0 code memory was written (VIF0 MPG or EE stores); the host
// uses it to invalidate recompiled programs. Called on the EE game thread.
struct PS2HostVu0
{
    uint8_t *codeMem = nullptr; // 4 KB
    uint8_t *dataMem = nullptr; // 4 KB
    void (*callms)(R5900Context *ctx, uint32_t startPcBytes, uint64_t codeGeneration) = nullptr;
};

void ps2SetHostVu0(const PS2HostVu0 &hooks);
const PS2HostVu0 &ps2HostVu0();
