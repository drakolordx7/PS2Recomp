// Host thread tuning for the VIF1 worker thread (kept out of ps2_vif1_worker.cpp: windows.h and the raylib names that
// ps2_runtime.h drags in do not mix).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0A00
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>

// The worker does the render-feeding work of a whole frame, so it must not be throttled:
//  * priority above normal, execution-speed throttling (EcoQoS) off;
//  * PS2X_VIF1_CORES=perf additionally restricts it to the CPU sets of the highest efficiency class on hybrid CPUs
//    (P-cores). Off by default: on an i9-12900K it made no measurable difference (43.9 vs 48.0 in-game frames/s with
//    and without it, both single runs), so placement is left to the OS.
void ps2Vif1TuneWorkerThread()
{
    HANDLE th = GetCurrentThread();
    SetThreadPriority(th, THREAD_PRIORITY_ABOVE_NORMAL);

    THREAD_POWER_THROTTLING_STATE throttle = {};
    throttle.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    throttle.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    throttle.StateMask = 0; // do not throttle
    SetThreadInformation(th, ThreadPowerThrottling, &throttle, sizeof(throttle));

    const char *v = std::getenv("PS2X_VIF1_CORES");
    if (!v || std::strcmp(v, "perf") != 0)
        return;
    ULONG len = 0;
    GetSystemCpuSetInformation(nullptr, 0, &len, GetCurrentProcess(), 0);
    if (len == 0)
        return;
    std::vector<uint8_t> buf(len);
    if (!GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data()), len, &len,
                                    GetCurrentProcess(), 0))
        return;
    int maxClass = 0, minClass = 255;
    for (ULONG off = 0; off < len;)
    {
        const auto *e = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION *>(buf.data() + off);
        if (e->Type == CpuSetInformation)
        {
            maxClass = std::max<int>(maxClass, e->CpuSet.EfficiencyClass);
            minClass = std::min<int>(minClass, e->CpuSet.EfficiencyClass);
        }
        off += e->Size;
    }
    if (maxClass == minClass)
        return;
    std::vector<ULONG> ids;
    for (ULONG off = 0; off < len;)
    {
        const auto *e = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION *>(buf.data() + off);
        if (e->Type == CpuSetInformation && e->CpuSet.EfficiencyClass == maxClass)
            ids.push_back(e->CpuSet.Id);
        off += e->Size;
    }
    if (!ids.empty())
        SetThreadSelectedCpuSets(th, ids.data(), static_cast<ULONG>(ids.size()));
}
#else
void ps2Vif1TuneWorkerThread() {}
#endif
