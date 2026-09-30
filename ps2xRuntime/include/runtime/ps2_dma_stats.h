#pragma once

#include <atomic>
#include <cstdint>

// Debug counters for the DMAC / VIF1 / VU1 path (PS2X_DMA_STATS=1). Written on the EE game thread, read by the
// headless heartbeat, which prints the per-interval difference. Not included by generated code.
struct PS2DmaStats
{
    using C = std::atomic<uint64_t>;
    // DMA starts per channel (0 VIF0, 1 VIF1, 2 GIF, 3 fromIPU, 4 toIPU, 5 SIF0, 6 SIF1, 7 SIF2, 8 fromSPR, 9 toSPR)
    // and CHCR.MOD (0 normal, 1 chain, 2 interleave, 3 invalid).
    C dmaStart[10][4]{};
    C dmaUnhandled[10][4]{}; // starts no emulation path consumed (CHCR.STR still reads back as done)
    C sprChainTags{};        // DMAtags walked by toSPR/fromSPR chain transfers
    C vif1Chains{};
    C vif1Tags{};
    C vif1Bytes{};
    C vif1ChainTagLimit{}; // chains cut at the tag limit
    C vif1SprTagAddr{};    // REF/REFS/NEXT/CALL tags whose ADDR has the SPR bit
    C vif1Unpack[16]{};    // by VN<<2 | VL
    C vif1UnpackMasked{};
    C vif1UnpackDropped{}; // UNPACK whose data ran past the buffer
    C vif1UnpackQw{};      // qwords written to VU1 memory
    C vif1Mpg{};
    C vif1Mscal{};
    C vif1Mscnt{};
    C vif1Direct{};
    C vif1DirectQw{};
    C vif1Unknown{}; // undefined VIFcodes
    C vif1Stmod[4]{};
    C vu1Xgkick{};
    C vu1XgkickBytes{};
    C vu1Cycles{};
    C vu1XgkickPrim[8]{}; // PRIM type of the first PRE tag in an XGKICK packet
};

PS2DmaStats &ps2DmaStats();
bool ps2DmaStatsEnabled();
