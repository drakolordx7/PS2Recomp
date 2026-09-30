#pragma once
// Frame-pipeline timeline recorder (patch 0025). Off unless KZ_TIMELINE=<file.csv> is set.
//
//   KZ_TIMELINE=<file.csv>  record events with a QPC-based nanosecond clock, write the CSV when the window ends
//   KZ_TL_START=<s>         window start, seconds since the runtime started (default 160)
//   KZ_TL_END=<s>           window end (default 200); the file is written by the first event after it
//
// Every event is one 24-byte record in a preallocated array (one relaxed fetch_add, no locks), so the recorder itself
// costs ~30 ns per event and no allocation. Nothing here is included by generated code.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2tl
{
    enum Lane : uint8_t { LaneEE = 0, LaneWorker = 1, LaneGS = 2, LaneOther = 3 };

    // id, lane, name. Pairs are *_B / *_E (begin / end).
#define PS2TL_EVENTS(X)                                                                                              \
    X(VBLANK, LaneEE)            /* a=vsync tick, b=guest vblank rate flag */                                        \
    X(SLEEP_B, LaneEE)           /* a=0 pacing sleep in processDueDeadlines, 1 idle wait */                          \
    X(SLEEP_E, LaneEE)                                                                                               \
    X(BARRIER_B, LaneEE)         /* a=Vif1BarrierReason; only logged when the worker was busy */                     \
    X(BARRIER_E, LaneEE)                                                                                             \
    X(APPLY_B, LaneEE)           /* applyCompletions: a=1 scheduler event, 2 barrier, 3 register poll */             \
    X(APPLY_E, LaneEE)           /* a=completions applied */                                                         \
    X(DMAC_IRQ, LaneEE)          /* dispatchIrq(dmac): a=cause, b=handlers queued */                                 \
    X(INV_B, LaneEE)             /* guest invocation (interrupt/callback) starts: a=pc, b=kind */                    \
    X(INV_E, LaneEE)                                                                                                 \
    X(D1_KICK, LaneEE)           /* DMA start reached the worker: a=bytes, b=(gif<<1|vif1) */                       \
    X(CHCR_BUSY, LaneEE)         /* first busy read of D1/D2 CHCR since the last idle read: a=channel */             \
    X(CHCR_IDLE, LaneEE)         /* first idle read after a busy one: a=channel */                                   \
    X(G_ENTER, LaneEE)           /* hooked guest function entry: a=address, b=$ra (limiter: vsync counter) */ \
    X(G_EXIT, LaneEE)            /* returned without unwinding: a=address, b as above */ \
    X(JOB_B, LaneWorker)         /* a=1 DMA job, 2 call (MSCAL), 3 vsync/GS call; b=bytes */                         \
    X(JOB_E, LaneWorker)                                                                                             \
    X(DONE_POST, LaneWorker)     /* completion posted to the EE */                                                   \
    X(VSYNC_B, LaneWorker)       /* kz runVsync (GS half of the vblank) */                                           \
    X(VSYNC_E, LaneWorker)                                                                                           \
    X(GS_THROTTLE_B, LaneWorker) /* kzgsVsync waiting for the GS thread (maxQueuedFrames) */                         \
    X(GS_THROTTLE_E, LaneWorker)                                                                                     \
    X(GS_RING_B, LaneWorker)     /* kzgs ring full: producer waits */                                                \
    X(GS_RING_E, LaneWorker)                                                                                         \
    X(GS_IDLE_B, LaneGS)         /* unused */ \
    X(GS_IDLE_E, LaneGS)         /* a = GS thread idle time in us since the previous frame (logged when a frame starts) */ \
    X(GS_VSYNC_B, LaneGS)        /* GSvsync (render + present of one frame) */                                       \
    X(GS_VSYNC_E, LaneGS)                                                                                            \
    X(GS_QUEUED, LaneWorker)     /* frames queued in kzgs after this vsync was pushed */                            \
    X(IOP_B, LaneOther)                                                                                              \
    X(IOP_E, LaneOther)

    enum Id : uint16_t
    {
#define X(n, l) n,
        PS2TL_EVENTS(X)
#undef X
        Count
    };

    struct Rec
    {
        uint64_t ns;
        uint32_t a, b, c;
        uint16_t id;
        uint16_t pad;
    };

    struct State
    {
        std::atomic<bool> enabled{false};
        std::atomic<bool> recording{false};
        std::atomic<bool> written{false};
        std::atomic<uint64_t> n{0};
        Rec *buf = nullptr;
        uint64_t cap = 0;
        uint64_t epochNs = 0;
        uint64_t startNs = 0, endNs = 0;
        char path[512] = {};
    };

    inline State g_state;

    inline uint64_t rawNs()
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    inline bool on() { return g_state.enabled.load(std::memory_order_relaxed); }

    inline void write();

    // Called when the runtime starts executing: seconds in KZ_TL_START/END count from here (the "t=" of the heartbeat).
    inline void init()
    {
        const char *p = std::getenv("KZ_TIMELINE");
        if (!p || !*p)
            return;
        State &s = g_state;
        std::snprintf(s.path, sizeof(s.path), "%s", p);
        const char *a = std::getenv("KZ_TL_START");
        const char *b = std::getenv("KZ_TL_END");
        const double t0 = a && *a ? std::atof(a) : 160.0;
        const double t1 = b && *b ? std::atof(b) : 200.0;
        s.epochNs = rawNs();
        s.startNs = s.epochNs + static_cast<uint64_t>(t0 * 1e9);
        s.endNs = s.epochNs + static_cast<uint64_t>(t1 * 1e9);
        s.cap = 8ull << 20; // 8 M records = 192 MB virtual, only touched as far as used
        s.buf = static_cast<Rec *>(std::malloc(s.cap * sizeof(Rec)));
        if (!s.buf)
            return;
        s.enabled.store(true, std::memory_order_release);
        std::fprintf(stderr, "[timeline] recording t=%.1f..%.1f s to %s\n", t0, t1, s.path);
    }

    inline void rec(Id id, uint32_t a = 0, uint32_t b = 0, uint32_t c = 0)
    {
        State &s = g_state;
        if (!s.enabled.load(std::memory_order_relaxed))
            return;
        const uint64_t t = rawNs();
        if (t < s.startNs)
            return;
        if (t > s.endNs)
        {
            write();
            return;
        }
        const uint64_t i = s.n.fetch_add(1, std::memory_order_relaxed);
        if (i >= s.cap)
            return;
        s.buf[i] = Rec{t - s.epochNs, a, b, c, static_cast<uint16_t>(id), 0};
    }

    inline void write()
    {
        State &s = g_state;
        if (s.written.exchange(true))
            return;
        static const char *names[] = {
#define X(n, l) #n,
            PS2TL_EVENTS(X)
#undef X
        };
        static const uint8_t lanes[] = {
#define X(n, l) static_cast<uint8_t>(l),
            PS2TL_EVENTS(X)
#undef X
        };
        static const char *laneNames[] = {"EE", "WORKER", "GS", "OTHER"};
        const uint64_t n = std::min<uint64_t>(s.n.load(), s.cap);
        // Records from different threads are not in time order in the array (index order = fetch_add order, close to it).
        std::FILE *f = std::fopen(s.path, "w");
        if (!f)
            return;
        std::fprintf(f, "t_us,lane,event,a,b,c\n");
        for (uint64_t i = 0; i < n; ++i)
        {
            const Rec &r = s.buf[i];
            if (r.id >= Count)
                continue;
            std::fprintf(f, "%.3f,%s,%s,%u,%u,%u\n", r.ns / 1000.0, laneNames[lanes[r.id]], names[r.id], r.a, r.b, r.c);
        }
        std::fclose(f);
        s.enabled.store(false, std::memory_order_release);
        std::fprintf(stderr, "[timeline] wrote %llu events to %s\n", static_cast<unsigned long long>(n), s.path);
    }

    // Function-scope helper for begin/end pairs.
    struct Span
    {
        Id e;
        uint32_t a;
        Span(Id b, Id e_, uint32_t a_ = 0, uint32_t b_ = 0) : e(e_), a(a_) { rec(b, a_, b_); }
        ~Span() { rec(e, a); }
    };
} // namespace ps2tl
