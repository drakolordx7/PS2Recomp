// Based on Blackline Interactive implementation
#include "runtime/ps2_memory.h"
#include "runtime/ps2_dma_stats.h"
#include "ps2_vif1_worker.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

enum VIFCmd : uint8_t
{
    VIF_NOP = 0x00,
    VIF_STCYCL = 0x01,
    VIF_OFFSET = 0x02,
    VIF_BASE = 0x03,
    VIF_ITOP = 0x04,
    VIF_STMOD = 0x05,
    VIF_MSKPATH3 = 0x06,
    VIF_MARK = 0x07,
    VIF_FLUSHE = 0x10,
    VIF_FLUSH = 0x11,
    VIF_FLUSHA = 0x13,
    VIF_MSCAL = 0x14,
    VIF_MSCALF = 0x15,
    VIF_MSCNT = 0x17,
    VIF_STMASK = 0x20,
    VIF_STROW = 0x30,
    VIF_STCOL = 0x31,
    VIF_MPG = 0x4A,
    VIF_DIRECT = 0x50,
    VIF_DIRECTHL = 0x51,
};

namespace
{
    constexpr uint8_t kGifFmtImage = 2u;

    // PS2X_VIF1_DUMP=<dir>: writes every VIF1 buffer (the bytes processVIF1Data gets: a whole DMA chain with its TTE
    // VIFcodes, or a normal-mode block) as vif1_NNNNN.bin, starting PS2X_VIF1_DUMP_AFTER seconds after the first
    // VIF1 transfer, at most PS2X_VIF1_DUMP_MAX (default 400) buffers. With the first one it also writes the VU1
    // micro memory (vu1code.bin) and the scratchpad (spr.bin); vu1code.bin is rewritten whenever MPG changes it.
    struct Vif1Dump
    {
        std::string dir;
        double after = 0.0;
        uint32_t max = 400u;
        uint32_t written = 0u;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    };

    Vif1Dump *vif1Dump()
    {
        static Vif1Dump *dump = []() -> Vif1Dump *
        {
            const char *dir = std::getenv("PS2X_VIF1_DUMP");
            if (!dir || !*dir)
                return nullptr;
            auto *d = new Vif1Dump();
            d->dir = dir;
            if (const char *v = std::getenv("PS2X_VIF1_DUMP_AFTER"))
                d->after = std::atof(v);
            if (const char *v = std::getenv("PS2X_VIF1_DUMP_MAX"))
                d->max = static_cast<uint32_t>(std::atoi(v));
            return d;
        }();
        return dump;
    }

    // PS2X_VIF_UNPACK=legacy keeps PS2Recomp's original VIF1 UNPACK code (A/B comparison).
    bool legacyVif1Unpack()
    {
        static const bool legacy = []
        {
            const char *v = std::getenv("PS2X_VIF_UNPACK");
            return v && std::strcmp(v, "legacy") == 0;
        }();
        return legacy;
    }

    // PS2X_VIF_UNPACK_FAST=0 turns the plain-UNPACK fast path off (A/B measurements).
    bool unpackFastEnabled()
    {
        static const bool on = []
        {
            const char *v = std::getenv("PS2X_VIF_UNPACK_FAST");
            return !(v && *v == '0');
        }();
        return on;
    }

    // PS2X_VIF_UNPACK_MASKED_FAST=0 keeps masked UNPACKs on the generic loop (A/B measurements).
    bool unpackMaskedFastEnabled()
    {
        static const bool on = []
        {
            const char *v = std::getenv("PS2X_VIF_UNPACK_MASKED_FAST");
            return !(v && *v == '0');
        }();
        return on;
    }

    // PS2X_MPG_SKIP_SAME=0: bump the VU1 code generation on every MPG, even when the bytes did not change.
    bool mpgSkipSameEnabled()
    {
        static const bool on = []
        {
            const char *v = std::getenv("PS2X_MPG_SKIP_SAME");
            return !(v && *v == '0');
        }();
        return on;
    }

    // Bytes of source data per vector, by VN<<2|VL (PCSX2 nVifT).
    constexpr uint8_t kVifGsize[16] = {4, 2, 1, 0, 8, 4, 2, 0, 12, 6, 3, 0, 16, 8, 4, 2};

    // ---- UNPACK fast path ------------------------------------------------------------------------------------------
    // Plain UNPACK (no mask, STMOD 0, not fill mode: WL <= CL) is nearly all of Killzone's ~8700 UNPACKs per frame. The
    // generic loop below does per-element bounds checks and a per-field mask/mode switch for every quadword; here each
    // vector is one SSE load/convert/store, for the vectors whose source lies entirely inside the data (the rest, the
    // tail, still goes through the generic loop, so out-of-range reads stay zero). V3 reads four elements like V4 (W is
    // the first element of the next vector), so V3-n loads the same bytes as V4-n.
    struct ConvV4_32
    {
        static __m128i load(const uint8_t *p) { return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p)); }
    };
    struct ConvV2_32
    {
        static __m128i load(const uint8_t *p)
        {
            return _mm_shuffle_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(p)), _MM_SHUFFLE(1, 0, 1, 0));
        }
    };
    struct ConvS_32
    {
        static __m128i load(const uint8_t *p)
        {
            uint32_t v;
            std::memcpy(&v, p, 4);
            return _mm_set1_epi32(static_cast<int32_t>(v));
        }
    };
    template <bool Usn>
    struct ConvV4_16
    {
        static __m128i load(const uint8_t *p)
        {
            const __m128i h = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p));
            return Usn ? _mm_cvtepu16_epi32(h) : _mm_cvtepi16_epi32(h);
        }
    };
    template <bool Usn>
    struct ConvV2_16
    {
        static __m128i load(const uint8_t *p)
        {
            uint32_t v;
            std::memcpy(&v, p, 4);
            const __m128i h = _mm_cvtsi32_si128(static_cast<int32_t>(v));
            return _mm_shuffle_epi32(Usn ? _mm_cvtepu16_epi32(h) : _mm_cvtepi16_epi32(h), _MM_SHUFFLE(1, 0, 1, 0));
        }
    };
    template <bool Usn>
    struct ConvS_16
    {
        static __m128i load(const uint8_t *p)
        {
            uint16_t v;
            std::memcpy(&v, p, 2);
            return _mm_set1_epi32(Usn ? static_cast<int32_t>(v) : static_cast<int32_t>(static_cast<int16_t>(v)));
        }
    };
    template <bool Usn>
    struct ConvV4_8
    {
        static __m128i load(const uint8_t *p)
        {
            uint32_t v;
            std::memcpy(&v, p, 4);
            const __m128i b = _mm_cvtsi32_si128(static_cast<int32_t>(v));
            return Usn ? _mm_cvtepu8_epi32(b) : _mm_cvtepi8_epi32(b);
        }
    };
    template <bool Usn>
    struct ConvV2_8
    {
        static __m128i load(const uint8_t *p)
        {
            uint16_t v;
            std::memcpy(&v, p, 2);
            const __m128i b = _mm_cvtsi32_si128(static_cast<int32_t>(v));
            return _mm_shuffle_epi32(Usn ? _mm_cvtepu8_epi32(b) : _mm_cvtepi8_epi32(b), _MM_SHUFFLE(1, 0, 1, 0));
        }
    };
    template <bool Usn>
    struct ConvS_8
    {
        static __m128i load(const uint8_t *p)
        {
            return _mm_set1_epi32(Usn ? static_cast<int32_t>(*p) : static_cast<int32_t>(static_cast<int8_t>(*p)));
        }
    };
    struct ConvV4_5
    {
        static __m128i load(const uint8_t *p)
        {
            uint16_t v;
            std::memcpy(&v, p, 2);
            return _mm_set_epi32(static_cast<int32_t>((v & 0x8000u) >> 8), static_cast<int32_t>((v & 0x7C00u) >> 7),
                                 static_cast<int32_t>((v & 0x03E0u) >> 2), static_cast<int32_t>((v & 0x001Fu) << 3));
        }
    };

    template <class Conv>
    uint32_t unpackPlainRun(uint8_t *vuMem, uint32_t vuMemSize, uint32_t &addr, uint32_t &vcl, uint32_t &src,
                            uint32_t count, uint32_t vsize, uint32_t wl, uint32_t skipBytes, const uint8_t *data)
    {
        uint32_t a = addr, c = vcl, s = src;
        const uint32_t wrap = vuMemSize - 16u;
        for (uint32_t i = 0; i < count; ++i)
        {
            _mm_storeu_si128(reinterpret_cast<__m128i *>(vuMem + (a & wrap)), Conv::load(data + s));
            a += 16u;
            ++c;
            s += vsize;
            if (c >= wl)
            {
                a += skipBytes;
                c = 0u;
            }
        }
        addr = a;
        vcl = c;
        src = s;
        return count;
    }

    // Masked UNPACK, STMOD 0, not fill mode (~15 % of Killzone's UNPACKs): per write cycle (min(vcl, 3)) every field is
    // either the data, the row register, the column register of that cycle, or write-protected (keeps the VU memory
    // word), so a vector is (data & sel) | (dest & keep) | fixed with per-cycle constants built once per UNPACK.
    struct MaskConsts
    {
        __m128i data[4], keep[4], fixed[4];
    };

    MaskConsts buildMaskConsts(const VIFRegisters &regs)
    {
        MaskConsts mc;
        for (uint32_t c = 0; c < 4u; ++c)
        {
            alignas(16) uint32_t d[4], k[4], x[4];
            for (uint32_t f = 0; f < 4u; ++f)
            {
                const uint32_t m = (regs.mask >> (c * 8u + f * 2u)) & 3u;
                d[f] = m == 0u ? ~0u : 0u;
                k[f] = m == 3u ? ~0u : 0u;
                x[f] = m == 1u ? regs.row[f] : (m == 2u ? regs.col[c] : 0u);
            }
            mc.data[c] = _mm_load_si128(reinterpret_cast<const __m128i *>(d));
            mc.keep[c] = _mm_load_si128(reinterpret_cast<const __m128i *>(k));
            mc.fixed[c] = _mm_load_si128(reinterpret_cast<const __m128i *>(x));
        }
        return mc;
    }

    template <class Conv>
    uint32_t unpackMaskedRun(uint8_t *vuMem, uint32_t vuMemSize, uint32_t &addr, uint32_t &vcl, uint32_t &src,
                             uint32_t count, uint32_t vsize, uint32_t wl, uint32_t skipBytes, const uint8_t *data,
                             const MaskConsts &mc)
    {
        uint32_t a = addr, c = vcl, s = src;
        const uint32_t wrap = vuMemSize - 16u;
        for (uint32_t i = 0; i < count; ++i)
        {
            const uint32_t mcyc = c < 3u ? c : 3u;
            __m128i *dest = reinterpret_cast<__m128i *>(vuMem + (a & wrap));
            __m128i r = _mm_or_si128(_mm_and_si128(Conv::load(data + s), mc.data[mcyc]), mc.fixed[mcyc]);
            r = _mm_or_si128(r, _mm_and_si128(_mm_loadu_si128(dest), mc.keep[mcyc]));
            _mm_storeu_si128(dest, r);
            a += 16u;
            ++c;
            s += vsize;
            if (c >= wl)
            {
                a += skipBytes;
                c = 0u;
            }
        }
        addr = a;
        vcl = c;
        src = s;
        return count;
    }

    // Number of leading vectors handled (updates addr/vcl/src); 0 if none.
    uint32_t unpackPlain(uint32_t vnvl, bool usn, uint8_t *vuMem, uint32_t vuMemSize, uint32_t &addr, uint32_t &vcl,
                         uint32_t &src, uint32_t num, uint32_t wl, uint32_t skipBytes, const uint8_t *data,
                         uint32_t dataBytes)
    {
        const uint32_t vsize = kVifGsize[vnvl];
        const uint32_t vn = vnvl >> 2;
        const uint32_t elem = vnvl == 15u ? 2u : (4u >> (vnvl & 3u));
        const uint32_t need = (vn == 2u) ? vsize + elem : vsize; // V3 also reads the first element of the next vector
        if (dataBytes < need)
            return 0u;
        const uint32_t count = std::min<uint32_t>(num, 1u + (dataBytes - need) / vsize);
#define KZ_UNPACK_RUN(CONV) return unpackPlainRun<CONV>(vuMem, vuMemSize, addr, vcl, src, count, vsize, wl, skipBytes, data)
#define KZ_UNPACK_RUN_USN(CONV) \
    if (usn) \
        KZ_UNPACK_RUN(CONV<true>); \
    KZ_UNPACK_RUN(CONV<false>)
        switch (vnvl)
        {
        case 0: KZ_UNPACK_RUN(ConvS_32);
        case 1: KZ_UNPACK_RUN_USN(ConvS_16);
        case 2: KZ_UNPACK_RUN_USN(ConvS_8);
        case 4: KZ_UNPACK_RUN(ConvV2_32);
        case 5: KZ_UNPACK_RUN_USN(ConvV2_16);
        case 6: KZ_UNPACK_RUN_USN(ConvV2_8);
        case 8: KZ_UNPACK_RUN(ConvV4_32);
        case 9: KZ_UNPACK_RUN_USN(ConvV4_16);
        case 10: KZ_UNPACK_RUN_USN(ConvV4_8);
        case 12: KZ_UNPACK_RUN(ConvV4_32);
        case 13: KZ_UNPACK_RUN_USN(ConvV4_16);
        case 14: KZ_UNPACK_RUN_USN(ConvV4_8);
        case 15: KZ_UNPACK_RUN(ConvV4_5);
        default: return 0u;
        }
#undef KZ_UNPACK_RUN_USN
#undef KZ_UNPACK_RUN
    }

    uint32_t unpackMasked(uint32_t vnvl, bool usn, uint8_t *vuMem, uint32_t vuMemSize, uint32_t &addr, uint32_t &vcl,
                          uint32_t &src, uint32_t num, uint32_t wl, uint32_t skipBytes, const uint8_t *data,
                          uint32_t dataBytes, const VIFRegisters &regs)
    {
        const uint32_t vsize = kVifGsize[vnvl];
        const uint32_t vn = vnvl >> 2;
        const uint32_t elem = vnvl == 15u ? 2u : (4u >> (vnvl & 3u));
        const uint32_t need = (vn == 2u) ? vsize + elem : vsize;
        if (dataBytes < need)
            return 0u;
        const uint32_t count = std::min<uint32_t>(num, 1u + (dataBytes - need) / vsize);
        const MaskConsts mc = buildMaskConsts(regs);
#define KZ_MASKED_RUN(CONV) return unpackMaskedRun<CONV>(vuMem, vuMemSize, addr, vcl, src, count, vsize, wl, skipBytes, data, mc)
#define KZ_MASKED_RUN_USN(CONV)     if (usn)         KZ_MASKED_RUN(CONV<true>);     KZ_MASKED_RUN(CONV<false>)
        switch (vnvl)
        {
        case 0: KZ_MASKED_RUN(ConvS_32);
        case 1: KZ_MASKED_RUN_USN(ConvS_16);
        case 2: KZ_MASKED_RUN_USN(ConvS_8);
        case 4: KZ_MASKED_RUN(ConvV2_32);
        case 5: KZ_MASKED_RUN_USN(ConvV2_16);
        case 6: KZ_MASKED_RUN_USN(ConvV2_8);
        case 8: KZ_MASKED_RUN(ConvV4_32);
        case 9: KZ_MASKED_RUN_USN(ConvV4_16);
        case 10: KZ_MASKED_RUN_USN(ConvV4_8);
        case 12: KZ_MASKED_RUN(ConvV4_32);
        case 13: KZ_MASKED_RUN_USN(ConvV4_16);
        case 14: KZ_MASKED_RUN_USN(ConvV4_8);
        case 15: KZ_MASKED_RUN(ConvV4_5);
        default: return 0u;
        }
#undef KZ_MASKED_RUN_USN
#undef KZ_MASKED_RUN
    }

    // VIF1 UNPACK with PCSX2's semantics (Vif_Unpack.cpp: vifUnpackSetup, _nVifUnpackLoop, UNPACK_S/V2/V4/V4_5,
    // writeXYZW):
    //  - WL = 0 means 256; fill mode is CL < WL. In fill mode the source pointer only advances for the first CL
    //    writes of a block; the other writes see the next (not yet consumed) vector, normally masked to row/col.
    //  - S-xx broadcasts; V2 writes x,y,x,y; V3 reads four elements (W = the next vector's first element);
    //    V4-5 expands 5:5:5:1 to 8-bit channels (<<3) and ignores STMOD.
    //  - STMOD 1 adds the row, 2 adds and stores the row, 3 stores the data into the row.
    //  - The mask row is chosen by the write-cycle counter (min(cl, 3)), the column register the same way.
    //  - The destination address wraps in VU1 memory (0x3FF0).
    // data/dataBytes is the source stream for this UNPACK; reads past it return zero.
    void vif1UnpackPcsx2(uint8_t *vuMem, uint32_t vuMemSize, VIFRegisters &regs, uint32_t cmd, const uint8_t *data,
                         uint32_t dataBytes, bool allowFast = true)
    {
        const uint32_t upk = (cmd >> 24) & 0x1Fu; // mask bit | VN | VL
        const uint32_t vnvl = upk & 0xFu;
        const bool doMask = (upk & 0x10u) != 0u;
        const bool usn = (cmd & 0x4000u) != 0u;
        const uint32_t vsize = kVifGsize[vnvl];
        if (vsize == 0u)
            return; // invalid unpack (PCSX2 only warns)
        uint32_t num = (cmd >> 16) & 0xFFu;
        if (num == 0u)
            num = 256u;
        const uint32_t cl = regs.cycle & 0xFFu;
        const uint32_t wl = ((regs.cycle >> 8) & 0xFFu) ? ((regs.cycle >> 8) & 0xFFu) : 256u;
        const bool isFill = cl < wl;
        const uint32_t mode = (vnvl == 15u) ? 0u : (regs.mode & 3u);
        uint32_t addr = cmd & 0x3FFu;
        if ((cmd & 0x8000u) != 0u)
            addr += regs.tops;
        addr = (addr << 4) & 0x3FF0u;
        const uint32_t skipBytes = isFill ? 0u : (cl - wl) * 16u;
        const uint32_t vl = vnvl & 3u;
        const uint32_t vn = vnvl >> 2;

        auto element = [&](uint32_t offset, uint32_t index) -> uint32_t
        {
            const uint32_t bytes = 4u >> vl; // 4, 2, 1
            const uint32_t at = offset + index * bytes;
            if (at + bytes > dataBytes)
                return 0u;
            if (vl == 0u)
            {
                uint32_t v;
                std::memcpy(&v, data + at, 4);
                return v;
            }
            if (vl == 1u)
            {
                uint16_t v;
                std::memcpy(&v, data + at, 2);
                return usn ? static_cast<uint32_t>(v) : static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(v)));
            }
            const uint8_t v = data[at];
            return usn ? static_cast<uint32_t>(v) : static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(v)));
        };

        uint32_t src = 0u;  // byte offset of the current source vector
        uint32_t vcl = 0u;  // write-cycle counter (PCSX2 vif.cl)
        uint32_t n = 0u;
        if (allowFast && unpackFastEnabled() && !isFill && mode == 0u)
        {
            if (!doMask)
                n = unpackPlain(vnvl, usn, vuMem, vuMemSize, addr, vcl, src, num, wl, skipBytes, data, dataBytes);
            else if (unpackMaskedFastEnabled())
                n = unpackMasked(vnvl, usn, vuMem, vuMemSize, addr, vcl, src, num, wl, skipBytes, data, dataBytes, regs);
        }
        for (; n < num; ++n)
        {
            uint32_t in[4];
            if (vnvl == 15u)
            {
                uint16_t v = 0;
                if (src + 2u <= dataBytes)
                    std::memcpy(&v, data + src, 2);
                in[0] = (v & 0x001Fu) << 3;
                in[1] = (v & 0x03E0u) >> 2;
                in[2] = (v & 0x7C00u) >> 7;
                in[3] = (v & 0x8000u) >> 8;
            }
            else if (vn == 0u)
            {
                in[0] = in[1] = in[2] = in[3] = element(src, 0);
            }
            else if (vn == 1u)
            {
                in[0] = in[2] = element(src, 0);
                in[1] = in[3] = element(src, 1);
            }
            else // V3 and V4 read four elements
            {
                for (uint32_t c = 0; c < 4u; ++c)
                    in[c] = element(src, c);
            }

            uint32_t *dest = reinterpret_cast<uint32_t *>(vuMem + (addr & (vuMemSize - 16u)));
            const uint32_t maskCycle = vcl < 3u ? vcl : 3u;
            for (uint32_t f = 0; f < 4u; ++f)
            {
                const uint32_t m = doMask ? (regs.mask >> (maskCycle * 8u + f * 2u)) & 3u : 0u;
                switch (m)
                {
                case 0:
                    switch (mode)
                    {
                    case 1: dest[f] = in[f] + regs.row[f]; break;
                    case 2: regs.row[f] += in[f]; dest[f] = regs.row[f]; break;
                    case 3: regs.row[f] = in[f]; dest[f] = in[f]; break;
                    default: dest[f] = in[f]; break;
                    }
                    break;
                case 1: dest[f] = regs.row[f]; break;
                case 2: dest[f] = regs.col[maskCycle]; break;
                default: break; // write protect
                }
            }

            addr += 16u;
            ++vcl;
            if (isFill)
            {
                if (vcl <= cl)
                    src += vsize;
                else if (vcl == wl)
                    vcl = 0u;
            }
            else
            {
                src += vsize;
                if (vcl >= wl)
                {
                    addr += skipBytes;
                    vcl = 0u;
                }
            }
        }
        regs.num = 0u;
    }

    // PS2X_VIF_UNPACK_CHECK=<n>: runs every n-th UNPACK twice, with and without the fast path, from the same VU1 memory /
    // VIF state, and reports differences (count and the first few) on stderr.
    uint32_t unpackCheckEvery()
    {
        static const uint32_t every = []
        {
            const char *v = std::getenv("PS2X_VIF_UNPACK_CHECK");
            return v ? static_cast<uint32_t>(std::strtoul(v, nullptr, 10)) : 0u;
        }();
        return every;
    }

    void checkedUnpack(uint8_t *vuMem, uint32_t vuMemSize, VIFRegisters &regs, uint32_t cmd, const uint8_t *data,
                       uint32_t dataBytes)
    {
        static uint64_t counter = 0, checked = 0, mismatches = 0;
        static std::vector<uint8_t> before, fastOut;
        if ((counter++ % unpackCheckEvery()) != 0u)
        {
            vif1UnpackPcsx2(vuMem, vuMemSize, regs, cmd, data, dataBytes, true);
            return;
        }
        before.assign(vuMem, vuMem + vuMemSize);
        const VIFRegisters regsBefore = regs;
        vif1UnpackPcsx2(vuMem, vuMemSize, regs, cmd, data, dataBytes, true);
        fastOut.assign(vuMem, vuMem + vuMemSize);
        const VIFRegisters regsFast = regs;
        std::memcpy(vuMem, before.data(), vuMemSize);
        regs = regsBefore;
        vif1UnpackPcsx2(vuMem, vuMemSize, regs, cmd, data, dataBytes, false);
        ++checked;
        const bool same = std::memcmp(fastOut.data(), vuMem, vuMemSize) == 0 &&
                          std::memcmp(&regsFast, &regs, sizeof(regs)) == 0;
        if (!same)
        {
            if (++mismatches <= 8)
                std::fprintf(stderr, "[vif1-unpack-check] MISMATCH cmd=%08x cycle=%04x mode=%u mask=%08x bytes=%u\n", cmd,
                             regsBefore.cycle, regsBefore.mode, regsBefore.mask, dataBytes);
        }
        if ((checked % 200000u) == 0u)
            std::fprintf(stderr, "[vif1-unpack-check] checked=%llu mismatches=%llu\n",
                         static_cast<unsigned long long>(checked), static_cast<unsigned long long>(mismatches));
    }

    void writeFile(const std::string &path, const void *data, size_t size)
    {
        if (FILE *f = std::fopen(path.c_str(), "wb"))
        {
            std::fwrite(data, 1, size, f);
            std::fclose(f);
        }
    }

    uint32_t pendingGifImageQwc(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data || sizeBytes < 16u)
            return 0u;

        uint32_t offset = 0u;
        while (offset + 16u <= sizeBytes)
        {
            uint64_t tagLo = 0u;
            std::memcpy(&tagLo, data + offset, sizeof(tagLo));
            offset += 16u;

            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t payloadBytes = 0u;
            if (flg == 0u) // PACKED
            {
                payloadBytes = static_cast<uint64_t>(nloop) * nreg * 16ull;
            }
            else if (flg == 1u) // REGLIST, padded to a quadword
            {
                payloadBytes = static_cast<uint64_t>(nloop) * nreg * 8ull;
                payloadBytes = (payloadBytes + 15ull) & ~15ull;
            }
            else if (flg == kGifFmtImage)
            {
                payloadBytes = static_cast<uint64_t>(nloop) * 16ull;
                const uint64_t availableBytes = sizeBytes - offset;
                if (payloadBytes > availableBytes)
                {
                    return static_cast<uint32_t>((payloadBytes - availableBytes) / 16ull);
                }
            }
            else
            {
                return 0u;
            }

            if (payloadBytes > static_cast<uint64_t>(sizeBytes - offset))
                return 0u;
            offset += static_cast<uint32_t>(payloadBytes);
        }

        return 0u;
    }
}

void PS2Memory::processVIF0Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF0Data(m_rdram + srcPhys, sizeBytes);
}

void PS2Memory::processVIF0Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (sizeBytes == 0u)
        return;

    uint32_t pos = 0;
    while (pos + 4 <= sizeBytes)
    {
        uint32_t cmd = 0u;
        std::memcpy(&cmd, data + pos, sizeof(cmd));
        pos += 4u;

        const uint8_t opcode = static_cast<uint8_t>((cmd >> 24) & 0x7Fu);
        const uint16_t imm = static_cast<uint16_t>(cmd & 0xFFFFu);
        const uint8_t num = static_cast<uint8_t>((cmd >> 16) & 0xFFu);
        const bool irq = (cmd & 0x80000000u) != 0u;

        vif0_regs.code = cmd;
        vif0_regs.num = num;
        if (irq)
            vif0_regs.stat |= (1u << 11);

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif0_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            vif0_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif0_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif0_regs.mark = imm;
            vif0_regs.stat |= (1u << 6);
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4u > sizeBytes)
                break;
            std::memcpy(&vif0_regs.mask, data + pos, sizeof(vif0_regs.mask));
            pos += 4u;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.row, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.col, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            const uint32_t destAddr = static_cast<uint32_t>(imm & 0x1FFu) * 8u;
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            uint32_t copyBytes = 0u;
            if (m_vu0Code && destAddr < PS2_VU0_CODE_SIZE && mpgBytes > 0u)
            {
                copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU0_CODE_SIZE)
                    copyBytes = PS2_VU0_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    std::memcpy(m_vu0Code + destAddr, data + pos, copyBytes);
                    markVU0CodeModified();
                }
            }

            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if ((opcode & 0x60u) == 0x60u)
        {
            const uint8_t vn = static_cast<uint8_t>((opcode >> 2) & 0x3u);
            const uint8_t vl = static_cast<uint8_t>(opcode & 0x3u);
            const int components = static_cast<int>(vn) + 1;
            int bitsPerComponent = 32;
            switch (vl)
            {
            case 0:
                bitsPerComponent = 32;
                break;
            case 1:
                bitsPerComponent = 16;
                break;
            case 2:
                bitsPerComponent = 8;
                break;
            case 3:
                bitsPerComponent = (vn == 3u) ? 4 : 16;
                break;
            default:
                break;
            }
            const int bitsPerVector = (vl == 3u && vn == 3u) ? 16 : (components * bitsPerComponent);
            uint32_t bytesPerVector = static_cast<uint32_t>((bitsPerVector + 7) / 8);
            const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            uint32_t cl = vif0_regs.cycle & 0xFFu;
            uint32_t wl = (vif0_regs.cycle >> 8) & 0xFFu;
            if (cl == 0u)
                cl = 1u;
            if (wl == 0u)
                wl = 1u;
            uint32_t sourceVectorCount = writeVectorCount;
            if (cl < wl)
            {
                const uint32_t fullBlocks = writeVectorCount / wl;
                uint32_t remainder = writeVectorCount % wl;
                if (remainder > cl)
                    remainder = cl;
                sourceVectorCount = fullBlocks * cl + remainder;
            }
            uint32_t totalBytes = sourceVectorCount * bytesPerVector;
            totalBytes = (totalBytes + 3u) & ~3u;

            if (m_vu0Data && pos + totalBytes <= sizeBytes && vl == 0u)
            {
                uint32_t vuAddr = static_cast<uint32_t>(imm & 0x3FFu);
                if ((imm & 0x8000u) != 0u)
                    vuAddr = (vuAddr + (vif0_regs.tops & 0x3FFu)) & 0x3FFu;
                const uint8_t *srcBase = data + pos;
                uint32_t srcIndex = 0u;
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t cyclePos = writeIndex % wl;
                    const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);
                    uint32_t destVec = (cl >= wl) ? ((vuAddr + (writeIndex / wl) * cl + cyclePos) & 0x3FFu)
                                                  : ((vuAddr + writeIndex) & 0x3FFu);
                    const uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU0_DATA_SIZE)
                    {
                        if (sourceAvailable && srcIndex < sourceVectorCount)
                            ++srcIndex;
                        continue;
                    }
                    if (!sourceAvailable || srcIndex >= sourceVectorCount)
                        continue;
                    const uint8_t *srcVec = srcBase + srcIndex * bytesPerVector;
                    ++srcIndex;
                    uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                    std::memcpy(lanes, m_vu0Data + destOff, sizeof(lanes));
                    const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                    for (uint32_t c = 0; c < limit; ++c)
                    {
                        uint32_t scalar = 0u;
                        std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                        lanes[c] = scalar;
                    }
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(m_vu0Data + destOff), _mm_loadu_si128(reinterpret_cast<const __m128i *>(lanes)));
                }
            }
            pos += totalBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else
        {
            break;
        }
    }
}

void PS2Memory::processVIF1Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF1Data(m_rdram + srcPhys, sizeBytes);
}

// Forwards the data of a VIF1 DIRECT/DIRECTHL to GIF PATH2. The GS frontends consume each forwarded packet on its own,
// starting with a GIFtag, while on hardware PATH2 is one continuous GIF stream: a DIRECT may end with an IMAGE GIFtag
// whose pixel data arrives in a later DIRECT (typically "MARK; DIRECT n" in the next DMAtag's TTE words). Such
// continuation data is re-wrapped here in a synthesized IMAGE tag. Only data that really is DIRECT payload is
// wrapped; the VIFcodes between the DIRECTs are parsed as VIFcodes (they used to be taken as the first 8 bytes of
// the image, shifting every uploaded texture by two words).
void PS2Memory::forwardVif1DirectData(const uint8_t *data, uint32_t sizeBytes, bool directHl)
{
    while (sizeBytes >= 16u)
    {
        if (m_vif1PendingPath2ImageQwc != 0u)
        {
            const uint32_t chunkQw = std::min<uint32_t>(m_vif1PendingPath2ImageQwc, sizeBytes / 16u);
            std::vector<uint8_t> imagePacket(16u + static_cast<size_t>(chunkQw) * 16u, 0u);
            const uint64_t imageTag =
                static_cast<uint64_t>(chunkQw & 0x7FFFu) |
                ((m_vif1PendingPath2ImageQwc == chunkQw) ? (1ull << 15) : 0ull) |
                (static_cast<uint64_t>(kGifFmtImage) << 58);
            std::memcpy(imagePacket.data(), &imageTag, sizeof(imageTag));
            std::memcpy(imagePacket.data() + 16u, data, static_cast<size_t>(chunkQw) * 16u);
            submitGifPacket(GifPathId::Path2, imagePacket.data(), static_cast<uint32_t>(imagePacket.size()), true,
                            m_vif1PendingPath2DirectHl);
            m_vif1PendingPath2ImageQwc -= chunkQw;
            if (m_vif1PendingPath2ImageQwc == 0u)
                m_vif1PendingPath2DirectHl = false;
            data += chunkQw * 16u;
            sizeBytes -= chunkQw * 16u;
            continue;
        }

        submitGifPacket(GifPathId::Path2, data, sizeBytes, true, directHl);
        const uint32_t pendingImageQw = pendingGifImageQwc(data, sizeBytes);
        if (pendingImageQw != 0u)
        {
            m_vif1PendingPath2ImageQwc = pendingImageQw;
            m_vif1PendingPath2DirectHl = directHl;
        }
        break;
    }
}

void PS2Memory::processVIF1Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (sizeBytes == 0u)
        return;
    // VIF1 state belongs to the worker thread while it has work (no-op when this is the worker).
    ps2Vif1Barrier(Vif1BarrierReason::Other);

    if (Vif1Dump *dump = vif1Dump())
    {
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - dump->start).count();
        if (t >= dump->after && dump->written < dump->max)
        {
            char name[64];
            std::snprintf(name, sizeof(name), "/vif1_%05u.bin", dump->written);
            writeFile(dump->dir + name, data, sizeBytes);
            if (dump->written == 0u)
            {
                writeFile(dump->dir + "/spr.bin", m_scratchpad, PS2_SCRATCHPAD_SIZE);
                writeFile(dump->dir + "/vu1code.bin", m_vu1Code, PS2_VU1_CODE_SIZE);
            }
            ++dump->written;
        }
    }

    const bool stats = ps2DmaStatsEnabled();
    PS2DmaStats &st = ps2DmaStats();
    if (stats)
        st.vif1Bytes.fetch_add(sizeBytes, std::memory_order_relaxed);
    auto count = [&](std::atomic<uint64_t> &c, uint64_t n = 1u)
    {
        if (stats)
            c.fetch_add(n, std::memory_order_relaxed);
    };

    uint32_t pos = 0;

    while (pos + 4 <= sizeBytes)
    {
        if (m_vif1PendingDirectQwc != 0u)
        {
            // Continuation of a DIRECT that was cut off at the end of the previous buffer: raw GIF data, no VIFcodes.
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            if (availableQw == 0u)
            {
                break;
            }

            const uint32_t chunkQw = std::min<uint32_t>(m_vif1PendingDirectQwc, availableQw);
            forwardVif1DirectData(data + pos, chunkQw * 16u, m_vif1PendingDirectHl);
            pos += chunkQw * 16u;
            m_vif1PendingDirectQwc -= chunkQw;
            continue;
        }

        uint32_t cmd;
        memcpy(&cmd, data + pos, 4);
        pos += 4;

        uint8_t opcode = (cmd >> 24) & 0x7F;
        uint16_t imm = cmd & 0xFFFF;
        uint8_t num = (cmd >> 16) & 0xFF;
        const bool irq = (cmd & 0x80000000u) != 0u;

        // Track most-recent command for VIFn_CODE emulation.
        vif1_regs.code = cmd;
        vif1_regs.num = num;
        if (irq)
            vif1_regs.stat |= (1u << 11); // INT

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif1_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_OFFSET)
        {
            // VIF double-buffer setup. OFFSET clears DBF and resets TOPS to BASE.
            // Do not rewrite BASE from the previous TOPS value.
            vif1_regs.ofst = imm & 0x3FFu;
            vif1_regs.tops = vif1_regs.base & 0x3FFu;
            vif1_regs.stat &= ~(1u << 7); // clear DBF
            continue;
        }
        else if (opcode == VIF_BASE)
        {
            // BASE only updates the base register. TOPS changes on OFFSET/MSCAL.
            vif1_regs.base = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            // ITOP VIFcode writes pending ITOPS; VU XITOP observes it after MSCAL/MSCNT.
            vif1_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif1_regs.mode = imm & 3u;
            count(st.vif1Stmod[imm & 3u]);
            continue;
        }
        else if (opcode == VIF_MSKPATH3)
        {
            // VIF command docs: MSKPATH3 uses IMMEDIATE bit 15.
            const bool wasMasked = m_path3Masked;
            m_path3Masked = (imm & 0x8000u) != 0u;
            if (wasMasked && !m_path3Masked)
                flushMaskedPath3Packets();
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif1_regs.mark = imm;
            vif1_regs.stat |= (1u << 6); // MRK
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_MSCAL || opcode == VIF_MSCALF)
        {
            uint32_t startPC = (uint32_t)imm * 8u;

            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF

            count(st.vif1Mscal);
            // Threaded mode: the host VU1 only ever runs on the worker thread (see ps2_vif1_worker.h).
            if (!ps2Vif1Mscal(startPC, runTop, runItop) && m_vu1MscalCallback)
                m_vu1MscalCallback(startPC, runTop, runItop);
            continue;
        }
        else if (opcode == VIF_MSCNT)
        {
            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF

            count(st.vif1Mscnt);
            if (!ps2Vif1Mscnt(runTop, runItop) && m_vu1MscntCallback)
                m_vu1MscntCallback(runTop, runItop);
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4 > sizeBytes)
                break;
            uint32_t maskValue = 0;
            std::memcpy(&maskValue, data + pos, sizeof(maskValue));
            vif1_regs.mask = maskValue;
            pos += 4;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.row, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.col, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            count(st.vif1Mpg);
            uint32_t destAddr = (uint32_t)imm * 8u;
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            if (m_vu1Code && destAddr < PS2_VU1_CODE_SIZE && mpgBytes > 0)
            {
                uint32_t copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU1_CODE_SIZE)
                    copyBytes = PS2_VU1_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    // The game re-sends microcode all the time; an upload that changes nothing must not invalidate the
                    // host VU's recompiled programs (each generation change makes the next MSCALs search the program
                    // lists again).
                    const bool same = std::memcmp(m_vu1Code + destAddr, data + pos, copyBytes) == 0;
                    if (!same || !mpgSkipSameEnabled())
                    {
                        std::memcpy(m_vu1Code + destAddr, data + pos, copyBytes);
                        markVU1CodeModified();
                    }
                    if (ps2Vif1StatsEnabled())
                        ps2Vif1StatMpg(same);
                }
            }
            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if (opcode == VIF_DIRECT || opcode == VIF_DIRECTHL)
        {
            uint32_t qwCount = imm;
            if (qwCount == 0)
                qwCount = 65536;
            const uint32_t requestedQw = qwCount;
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            const bool truncated = qwCount > availableQw;
            if (qwCount > availableQw)
                qwCount = availableQw;

            const bool directHl = (opcode == VIF_DIRECTHL);
            count(st.vif1Direct);
            count(st.vif1DirectQw, requestedQw);
            if (truncated)
            {
                // The rest of this DIRECT's data starts the next buffer (see m_vif1PendingDirectQwc).
                m_vif1PendingDirectQwc = requestedQw - qwCount;
                m_vif1PendingDirectHl = directHl;
            }
            if (qwCount > 0)
            {
                forwardVif1DirectData(data + pos, qwCount * 16u, directHl);
            }

            pos += qwCount * 16;
            if (truncated)
            {
                pos = sizeBytes;
                break;
            }
            continue;
        }
        else if ((opcode & 0x60) == 0x60)
        {
            uint8_t vn = (opcode >> 2) & 0x3;
            uint8_t vl = opcode & 0x3;
            const bool maskEnable = (opcode & 0x10u) != 0u;
            int components = vn + 1;
            int bitsPerComponent = 32;
            switch (vl)
            {
            case 0:
                bitsPerComponent = 32;
                break;
            case 1:
                bitsPerComponent = 16;
                break;
            case 2:
                bitsPerComponent = 8;
                break;
            case 3:
                bitsPerComponent = (vn == 3) ? 4 : 16;
                break;
            default:
                break;
            }
            int bitsPerVector = (vl == 3 && vn == 3) ? 16 : (components * bitsPerComponent);
            uint32_t bytesPerVector = (bitsPerVector + 7) / 8;
            // UNPACK semantics: NUM is 8-bit and NUM==0 means 256 vectors (writes).
            const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);

            // STCYCL controls write cycles for UNPACK.
            uint32_t cl = vif1_regs.cycle & 0xFFu;
            uint32_t wl = (vif1_regs.cycle >> 8) & 0xFFu;
            if (cl == 0u)
                cl = 1u;
            if (wl == 0u)
                wl = 1u;

            uint32_t sourceVectorCount = writeVectorCount;
            if (cl < wl)
            {
                const uint32_t fullBlocks = writeVectorCount / wl;
                uint32_t remainder = writeVectorCount % wl;
                if (remainder > cl)
                    remainder = cl;
                sourceVectorCount = fullBlocks * cl + remainder;
            }

            uint32_t totalBytes = sourceVectorCount * bytesPerVector;
            totalBytes = (totalBytes + 3) & ~3u;

            uint32_t vuAddr = (uint32_t)imm & 0x3FFu;
            if ((imm & 0x8000u) != 0u)
                vuAddr = (vuAddr + (vif1_regs.tops & 0x3FFu)) & 0x3FFu;

            const bool zeroExtend = (imm & 0x4000u) != 0u;
            count(st.vif1Unpack[(vn << 2) | vl]);
            if (maskEnable)
                count(st.vif1UnpackMasked);
            if (pos + totalBytes > sizeBytes)
                count(st.vif1UnpackDropped);
            else
                count(st.vif1UnpackQw, writeVectorCount);

            if (!legacyVif1Unpack())
            {
                // PCSX2 source size: fill mode reads CL vectors per WL block (WL = 0 means 256).
                const uint32_t pcl = vif1_regs.cycle & 0xFFu;
                const uint32_t pwl = ((vif1_regs.cycle >> 8) & 0xFFu) ? ((vif1_regs.cycle >> 8) & 0xFFu) : 256u;
                const uint32_t srcVectors = (pcl < pwl) ? pcl * (writeVectorCount / pwl) + std::min(writeVectorCount % pwl, pcl)
                                                        : writeVectorCount;
                const uint32_t srcBytes = (srcVectors * kVifGsize[opcode & 0xFu] + 3u) & ~3u;
                if (m_vu1Data && pos + srcBytes <= sizeBytes)
                {
                    if (unpackCheckEvery() == 0u)
                        vif1UnpackPcsx2(m_vu1Data, PS2_VU1_DATA_SIZE, vif1_regs, cmd, data + pos, srcBytes);
                    else
                        checkedUnpack(m_vu1Data, PS2_VU1_DATA_SIZE, vif1_regs, cmd, data + pos, srcBytes);
                }
                pos += srcBytes;
                if (pos > sizeBytes)
                    break;
                continue;
            }

            if (m_vu1Data && totalBytes > 0 && pos + totalBytes <= sizeBytes)
            {
                const uint8_t *srcBase = data + pos;
                uint32_t srcIndex = 0u;
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t cyclePos = writeIndex % wl;
                    const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);

                    uint32_t destVec = 0;
                    if (cl >= wl)
                    {
                        destVec = (vuAddr + (writeIndex / wl) * cl + cyclePos) & 0x3FFu;
                    }
                    else
                    {
                        destVec = (vuAddr + writeIndex) & 0x3FFu;
                    }

                    uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU1_DATA_SIZE)
                    {
                        if (sourceAvailable && srcIndex < sourceVectorCount)
                            ++srcIndex;
                        continue;
                    }

                    uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                    std::memcpy(lanes, m_vu1Data + destOff, sizeof(lanes));
                    uint32_t decompressed[4] = {lanes[0], lanes[1], lanes[2], lanes[3]};
                    bool decoded = false;

                    const uint8_t *srcVec = nullptr;
                    if (sourceAvailable && srcIndex < sourceVectorCount)
                    {
                        srcVec = srcBase + srcIndex * bytesPerVector;
                        ++srcIndex;
                        decoded = true;
                    }

                    auto extend16 = [&](uint16_t raw) -> uint32_t
                    {
                        if (zeroExtend)
                            return static_cast<uint32_t>(raw);
                        return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(raw)));
                    };

                    auto extend8 = [&](uint8_t raw) -> uint32_t
                    {
                        if (zeroExtend)
                            return static_cast<uint32_t>(raw);
                        return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(raw)));
                    };

                    bool handledFormat = true;
                    if (!decoded)
                    {
                        handledFormat = false;
                    }
                    else if (vl == 0u)
                    {
                        if (components == 1)
                        {
                            uint32_t scalar = 0;
                            std::memcpy(&scalar, srcVec, sizeof(scalar));
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                uint32_t scalar = 0;
                                std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                                decompressed[c] = scalar;
                            }
                        }
                    }
                    else if (vl == 1u)
                    {
                        if (components == 1)
                        {
                            uint16_t raw = 0;
                            std::memcpy(&raw, srcVec, sizeof(raw));
                            const uint32_t scalar = extend16(raw);
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                uint16_t raw = 0;
                                std::memcpy(&raw, srcVec + c * 2u, sizeof(raw));
                                decompressed[c] = extend16(raw);
                            }
                        }
                    }
                    else if (vl == 2u)
                    {
                        if (components == 1)
                        {
                            const uint32_t scalar = extend8(srcVec[0]);
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                decompressed[c] = extend8(srcVec[c]);
                            }
                        }
                    }
                    else if (vl == 3u && vn == 3u)
                    {
                        // V4-5: packed color-like format in a single 16-bit value.
                        uint16_t packed = 0;
                        std::memcpy(&packed, srcVec, sizeof(packed));
                        decompressed[0] = packed & 0x1Fu;
                        decompressed[1] = (packed >> 5) & 0x1Fu;
                        decompressed[2] = (packed >> 10) & 0x1Fu;
                        decompressed[3] = (packed >> 15) & 0x01u;
                    }
                    else
                    {
                        handledFormat = false;
                    }

                    // Unknown compressed format fallback: preserve legacy raw-copy behavior.
                    if (!handledFormat && decoded && !maskEnable && (vif1_regs.mode == 0u || vif1_regs.mode == 3u))
                    {
                        uint32_t copyBytes = (bytesPerVector < 16u) ? bytesPerVector : 16u;
                        std::memcpy(m_vu1Data + destOff, srcVec, copyBytes);
                        continue;
                    }

                    const bool canAdd = (vl != 3u || vn != 3u);
                    const uint32_t mode = vif1_regs.mode & 3u;
                    const uint32_t colIdx = (cyclePos > 3u) ? 3u : cyclePos;
                    const uint32_t maskCycle = (cyclePos > 3u) ? 3u : cyclePos;

                    for (uint32_t field = 0u; field < 4u; ++field)
                    {
                        uint32_t maskSpec = 0u;
                        if (maskEnable)
                        {
                            const uint32_t shift = ((maskCycle * 4u) + field) * 2u;
                            maskSpec = (vif1_regs.mask >> shift) & 0x3u;
                        }

                        // In fill-write cycles with suspended source reads, treat raw-data selections as row-fill.
                        if (!decoded && maskSpec == 0u)
                            maskSpec = 1u;

                        uint32_t writeVal = lanes[field];
                        if (maskSpec == 0u)
                        {
                            if (handledFormat)
                            {
                                writeVal = decompressed[field];
                                if (canAdd && (mode == 1u || mode == 2u))
                                {
                                    writeVal = writeVal + vif1_regs.row[field];
                                    if (mode == 2u)
                                        vif1_regs.row[field] = writeVal;
                                }
                            }
                        }
                        else if (maskSpec == 1u)
                        {
                            writeVal = vif1_regs.row[field];
                        }
                        else if (maskSpec == 2u)
                        {
                            writeVal = vif1_regs.col[colIdx];
                        }
                        else
                        {
                            continue; // write-protect
                        }

                        lanes[field] = writeVal;
                    }

                    std::memcpy(m_vu1Data + destOff, lanes, sizeof(lanes));
                }
            }
            pos += totalBytes;

            if (pos > sizeBytes)
                break;
            continue;
        }
        else
        {
            count(st.vif1Unknown);
            continue;
        }
    }
}
