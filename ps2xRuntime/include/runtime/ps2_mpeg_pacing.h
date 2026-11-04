#pragma once

#include <cstdint>

// Movie clock of the libmpeg HLE (Kernel/Stubs/MPEG.cpp), for games that decide themselves when to fetch the next
// picture. Such a game asks sceMpegGetPicture for a new picture when its own clock says so (Killzone: every other
// guest vblank, which assumes 59.94 Hz fields); when the host runs the guest vblank faster, it asks too often and the
// HLE has to park the thread in GetPicture until the picture is due, which stalls everything else that thread does.
// A hook can ask here instead and only fetch a picture when it is due.
namespace ps2_stubs
{
    // True when the next picture of this sceMpeg handle is due at `vsyncTick` (EeScheduler::currentVSyncTick), or when
    // the HLE has no clock for it yet (nothing served since the handle was created or reset): fetch it now.
    bool mpegPictureDue(uint32_t mpegAddr, uint64_t vsyncTick);
}
