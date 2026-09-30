#pragma once

#include <cstdint>

// Optional host SPU2 (e.g. kzspu2). When installed (before the IOP runs), the IOP emulator routes to it:
//  - every access to the SPU2 register block, IOP physical 0x1F900000-0x1F9007FF (32-bit accesses are split into
//    two 16-bit ones, 8-bit writes are read-modify-write on the 16-bit register);
//  - IOP DMA channel 4 (SPU2 core 0, 0x1F8010C0 MADR / C4 BCR / C8 CHCR) and channel 7 (core 1, 0x1F801500..): a
//    CHCR write with TR (bit 24) set calls dmaStart. CHCR.TR then stays set until the host calls
//    iopSpu2DmaComplete(core); MADR reads during the transfer come from dmaMadr;
//  - time: advance() after every IOP time slice and at the end of each runCycles(), and nextEvent() bounds how far
//    the IOP skips ahead while all threads sleep.
// Every call passes the IOP cycle counter (36.864 MHz). It is monotonic across IOP resets.
// All hooks run on the thread that runs the IOP: the EE thread, or with PS2X_IOP_THREAD the IOP thread (and, while
// it holds the IOP lock, the EE thread inside an RPC / module load). Calls are always serialised, never concurrent.
// Without hooks, the old stub behaviour is unchanged.
namespace ps2x::iop
{
    struct IopHostSpu2
    {
        uint16_t (*read16)(uint32_t physAddr, uint64_t iopCycle) = nullptr;
        void (*write16)(uint32_t physAddr, uint16_t value, uint64_t iopCycle) = nullptr;
        // core 0 = DMA ch4, 1 = ch7. `ram` points at MADR inside IOP RAM and stays valid; `halfwords` = BCR
        // (block size * block count) words * 2, clamped to the end of IOP RAM. toSpu2 = CHCR bit 0 (IOP -> SPU2).
        void (*dmaStart)(int core, uint16_t *ram, uint32_t madr, uint32_t halfwords, bool toSpu2, uint64_t iopCycle) = nullptr;
        uint32_t (*dmaMadr)(int core) = nullptr;               // optional: MADR while the transfer runs
        void (*advance)(uint64_t iopCycle) = nullptr;           // optional: run the SPU2 up to iopCycle
        uint64_t (*nextEvent)() = nullptr;                      // optional: IOP cycle (same monotonic count) of the
                                                                // next SPU2 event; UINT64_MAX = none
        uint64_t maxIdleStep = 768u * 32u;                      // longest idle skip while hooks are installed
    };

    void iopSetHostSpu2(const IopHostSpu2 &hooks);
    const IopHostSpu2 &iopHostSpu2();

    // Called by the host from inside a hook (or any time on the IOP thread). Queued, then dispatched by the emulator
    // as IOP interrupts: SPU2 IRQ -> intrman line 9; DMA end -> clears CHCR.TR, MADR = end, interrupt 0x24 / 0x28.
    void iopSpu2RaiseIrq();
    void iopSpu2DmaComplete(int core);
}
