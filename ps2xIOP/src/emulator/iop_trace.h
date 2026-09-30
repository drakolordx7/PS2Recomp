#pragma once

#include <cstdint>
#include <cstdlib>

namespace ps2x::iop::detail
{
    // PS2X_IOP_SYNC_TRACE (debug aid, stderr): 1 = contended semaphore waits outside a thread plus a once per IOP
    // second thread/semaphore dump (and PS2X_IOP_WATCH words); 2 = also every RPC request and IOP->EE SIF command.
    inline int iopSyncTraceLevel() noexcept
    {
        static const int level = []()
        {
            const char *v = std::getenv("PS2X_IOP_SYNC_TRACE");
            return v ? std::atoi(v) : 0;
        }();
        return level;
    }

    inline bool iopSyncTrace() noexcept { return iopSyncTraceLevel() > 0; }
}
