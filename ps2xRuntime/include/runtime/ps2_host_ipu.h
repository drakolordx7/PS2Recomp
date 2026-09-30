#pragma once

#include <cstdint>

// Optional host IPU (MPEG decoder), e.g. kzipu. When set, PS2Memory routes to it: IPU registers
// (0x10002000-0x1000203F), DMA channels 3/4 (IPU_FROM 0x1000B000, IPU_TO 0x1000B400; `handles` decides), 64-bit
// accesses to those, the IPU FIFO ports (0x10007000 out, 0x10007010 in), the DMAC master enable (D_CTRL / D_ENABLEW)
// and a poll on D_STAT reads. Called on the EE game thread.
struct PS2HostIpu
{
    bool (*handles)(uint32_t physAddr) = nullptr;
    uint32_t (*read32)(uint32_t physAddr) = nullptr;
    void (*write32)(uint32_t physAddr, uint32_t value) = nullptr;
    uint64_t (*read64)(uint32_t physAddr) = nullptr;
    void (*write64)(uint32_t physAddr, uint64_t value) = nullptr;
    void (*fifoWrite)(const void *qword) = nullptr; // 16 bytes into the input FIFO
    void (*fifoRead)(void *qword) = nullptr;        // 16 bytes from the output FIFO
    void (*dmacEnable)(bool enabled) = nullptr;
    void (*poll)() = nullptr;
};

void ps2SetHostIpu(const PS2HostIpu &hooks);
const PS2HostIpu &ps2HostIpu();

// Provided by the runtime (valid after PS2Memory::initialize): completes DMA channel `channel` the way the DMAC does
// (clears CHCR.STR in the runtime's register mirror, sets D_STAT bit `channel` and CIS, queues the cause for the game's
// DMAC handlers; the host then calls PS2Runtime::drainCompletedDmacHandlers).
void ps2HostCompleteDmac(uint32_t channel);
