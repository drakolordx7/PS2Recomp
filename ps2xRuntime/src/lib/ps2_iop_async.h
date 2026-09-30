#pragma once

#include "ps2x/iop/iop_types.h"

#include <functional>

namespace ps2x::iop
{
    class IopSubsystem;
}

// Offloading of nowait SIF RPC calls to the IOP thread (threaded IOP mode, PS2X_IOP_RPC_ASYNC). The runtime installs
// its subsystem once; SifCallRpc asks ps2IopRpcSubmit() to queue a nowait call. On success the IOP thread runs the
// server function later, writes the reply into the guest receive buffer, and `eeDone` runs on the EE thread (from the
// posted-work queue, ordered with the IOP's SIF commands) with the result. On false nothing was queued and the caller
// runs the call inline as before.
void ps2IopRpcInstall(ps2x::iop::IopSubsystem *subsystem);
[[nodiscard]] bool ps2IopRpcSubmit(const ps2x::iop::RpcRequest &request,
                                   std::function<void(const ps2x::iop::RpcResult &)> eeDone);
