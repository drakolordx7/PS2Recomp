#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/ps2_host_gs.h"
#include "../ps2_vif1_worker.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>

namespace
{
    PS2HostGs g_hostGs{};
}

void ps2SetHostGs(const PS2HostGs &hooks)
{
    g_hostGs = hooks;
}

namespace
{
    // Exclusive host GS: the built-in frontend still has to see A+D writes to SIGNAL / FINISH / LABEL (0x60-0x62),
    // because they set CSR bits the game polls (e.g. Killzone waits for CSR.FINISH at boot).
    bool packetWritesGsEvent(const uint8_t *data, uint32_t size)
    {
        const uint32_t qwords = size / 16u;
        uint32_t q = 0;
        while (q < qwords)
        {
            uint64_t lo, hi;
            std::memcpy(&lo, data + q * 16u, 8);
            std::memcpy(&hi, data + q * 16u + 8u, 8);
            ++q;
            const uint32_t nloop = static_cast<uint32_t>(lo & 0x7FFFu);
            const uint32_t flg = static_cast<uint32_t>((lo >> 58) & 3u);
            uint32_t nreg = static_cast<uint32_t>((lo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            if (flg == 0u)
            {
                // Only A+D descriptors (0xE) can address SIGNAL/FINISH/LABEL: when none of the tag's NREG descriptor
                // nibbles is 0xE the whole PACKED body is skipped without looking at it (nearly every XGKICK tag).
                const uint64_t used = nreg >= 16u ? ~0ull : ((1ull << (4u * nreg)) - 1ull);
                const uint64_t x = (hi ^ 0xEEEEEEEEEEEEEEEEull) | ~used; // a nibble is 0 where the descriptor is 0xE
                const bool anyAd = (((x - 0x1111111111111111ull) & ~x & 0x8888888888888888ull) != 0ull);
                if (!anyAd)
                {
                    const uint64_t body = static_cast<uint64_t>(nloop) * nreg;
                    if (body > qwords - q)
                        return false;
                    q += static_cast<uint32_t>(body);
                    continue;
                }
                for (uint32_t l = 0; l < nloop; ++l)
                    for (uint32_t r = 0; r < nreg; ++r, ++q)
                    {
                        if (q >= qwords)
                            return false;
                        if (((hi >> (4u * r)) & 0xFu) == 0xEu)
                        {
                            const uint8_t addr = data[q * 16u + 8u];
                            if (addr >= 0x60u && addr <= 0x62u)
                                return true;
                        }
                    }
            }
            else if (flg == 1u)
                q += (nloop * nreg + 1u) / 2u;
            else
                q += nloop;
        }
        return false;
    }

    // PS2X_GIF_SCAN_CHECK=1: compares the fast scan above with the plain loop for every packet.
    bool packetWritesGsEventPlain(const uint8_t *data, uint32_t size)
    {
        const uint32_t qwords = size / 16u;
        uint32_t q = 0;
        while (q < qwords)
        {
            uint64_t lo, hi;
            std::memcpy(&lo, data + q * 16u, 8);
            std::memcpy(&hi, data + q * 16u + 8u, 8);
            ++q;
            const uint32_t nloop = static_cast<uint32_t>(lo & 0x7FFFu);
            const uint32_t flg = static_cast<uint32_t>((lo >> 58) & 3u);
            uint32_t nreg = static_cast<uint32_t>((lo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            if (flg == 0u)
            {
                for (uint32_t l = 0; l < nloop; ++l)
                    for (uint32_t r = 0; r < nreg; ++r, ++q)
                    {
                        if (q >= qwords)
                            return false;
                        if (((hi >> (4u * r)) & 0xFu) == 0xEu)
                        {
                            const uint8_t addr = data[q * 16u + 8u];
                            if (addr >= 0x60u && addr <= 0x62u)
                                return true;
                        }
                    }
            }
            else if (flg == 1u)
                q += (nloop * nreg + 1u) / 2u;
            else
                q += nloop;
        }
        return false;
    }

    bool gifScanCheckEnabled()
    {
        static const bool on = []
        {
            const char *v = std::getenv("PS2X_GIF_SCAN_CHECK");
            return v && *v && *v != '0';
        }();
        return on;
    }

    // PS2X_GIF_DIRECT=0 queues every packet like before; by default a PATH1/PATH2 packet that arrives while nothing is
    // queued goes straight to the host GS (see GifArbiter::submit).
    bool gifDirectEnabled()
    {
        static const bool on = []
        {
            const char *v = std::getenv("PS2X_GIF_DIRECT");
            return !(v && *v == '0');
        }();
        return on;
    }

    // One packet to the host GS, and to the built-in GS when it has to see it (was the body of GifArbiter::drain).
    void emitPacket(const std::function<void(const uint8_t *, uint32_t)> &processFn, GifPathId pathId, const uint8_t *data,
                    uint32_t size)
    {
        if (g_hostGs.gifPacket)
        {
            if (ps2Vif1StatsEnabled() && ps2Vif1OnWorkerThread())
            {
                const auto t0 = std::chrono::steady_clock::now();
                g_hostGs.gifPacket(static_cast<int>(pathId), data, size);
                ps2Vif1StatAddGifNs(static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
            }
            else
                g_hostGs.gifPacket(static_cast<int>(pathId), data, size);
        }
        const bool hostOnly = g_hostGs.gifPacket && g_hostGs.exclusive;
        if (!hostOnly)
        {
            processFn(data, size);
            return;
        }
        const bool writesEvent = packetWritesGsEvent(data, size);
        if (gifScanCheckEnabled() && writesEvent != packetWritesGsEventPlain(data, size))
            std::fprintf(stderr, "[gif-arbiter] SCAN MISMATCH size=%u fast=%d\n", size, writesEvent ? 1 : 0);
        if (writesEvent)
            processFn(data, size);
    }
}

const PS2HostGs &ps2HostGs()
{
    return g_hostGs;
}

GifArbiter::GifArbiter(ProcessPacketFn processFn)
    : m_processFn(std::move(processFn))
{
}

bool GifArbiter::isImagePacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes < 16u)
        return false;

    uint64_t tagLo = 0;
    std::memcpy(&tagLo, data, sizeof(tagLo));
    const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
    return flg == 2u;
}

void GifArbiter::submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl)
{
    if (!data || sizeBytes < 16 || !m_processFn)
        return;

    // Nothing queued and not a PATH3 packet: it would come out of drain() first and unchanged, so skip the queue (its
    // heap copy, the sort and the second pass). PATH3 packets stay queued: they sort behind PATH1/PATH2 ones and are
    // subject to the IMAGE rules below.
    if (pathId != GifPathId::Path3 && m_queue.empty() && gifDirectEnabled())
    {
        emitPacket(m_processFn, pathId, data, sizeBytes);
        return;
    }

    GifArbiterPacket pkt;
    pkt.pathId = pathId;
    pkt.path2DirectHl = (pathId == GifPathId::Path2) && path2DirectHl;
    pkt.path3Image = (pathId == GifPathId::Path3) && isImagePacket(data, sizeBytes);
    pkt.data.resize(sizeBytes);
    std::memcpy(pkt.data.data(), data, sizeBytes);
    m_queue.push_back(std::move(pkt));
}

void GifArbiter::drain()
{
    if (!m_processFn || m_queue.empty())
        return;

    std::stable_sort(m_queue.begin(), m_queue.end(),
                     [](const GifArbiterPacket &a, const GifArbiterPacket &b)
                     {
                         // DIRECTHL cannot preempt PATH3 IMAGE transfers.
                         if (a.path2DirectHl != b.path2DirectHl || a.path3Image != b.path3Image)
                         {
                             if (a.path3Image && b.path2DirectHl)
                                 return true;
                             if (a.path2DirectHl && b.path3Image)
                                 return false;
                         }
                         return pathPriority(a.pathId) < pathPriority(b.pathId);
                     });

    for (size_t i = 0; i < m_queue.size(); ++i)
    {
        auto &pkt = m_queue[i];
        if (!pkt.data.empty())
            emitPacket(m_processFn, pkt.pathId, pkt.data.data(), static_cast<uint32_t>(pkt.data.size()));
    }
    m_queue.clear();
}

uint8_t GifArbiter::pathPriority(GifPathId id)
{
    return static_cast<uint8_t>(id);
}
