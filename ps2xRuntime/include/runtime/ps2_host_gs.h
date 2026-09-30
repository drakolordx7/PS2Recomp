#pragma once

#include <cstdint>

// Optional host GS (e.g. a hardware renderer) fed alongside the built-in GS frontend.
// gifPacket: every GIF packet in arbiter order, with its PATH (1..3); sizeBytes is a multiple of 16.
// vsync: guest VBlankStart, after CSR.FIELD was updated; field = interlace field being displayed (0/1).
// Both are called on the EE game thread.
struct PS2HostGs
{
    void (*gifPacket)(int path, const uint8_t *data, uint32_t sizeBytes) = nullptr;
    void (*vsync)(uint64_t tick, int field) = nullptr;
    // true: the host GS is the only consumer of GIF packets; the built-in frontend no longer processes them (saves
    // its VRAM uploads, register state and vertex work on the EE thread). Only valid for games that never read GS
    // memory back through the built-in GS (local->host transfers).
    bool exclusive = false;
    // true: `vsync` splits its work. Whatever has to happen on the EE thread at the vblank (guest memory patches) runs
    // in the hook itself; the part that talks to the GS runs through ps2RunAfterQueuedGif(), so with the VIF1 worker
    // thread the EE does not have to wait for the worker to finish the frame's GIF packets first. false: the runtime
    // waits for the worker before calling `vsync` (the hook may then do everything inline).
    bool vsyncOrdered = false;
};

void ps2SetHostGs(const PS2HostGs &hooks);
const PS2HostGs &ps2HostGs();

// Calls fn(arg) after every GIF packet queued so far (PATH1/2/3, from DMA started before this call) has been handed to
// the host GS: inline on the calling thread when the VIF1/VU1/GIF work is synchronous (or when called from the worker
// itself), otherwise later on the VIF1 worker thread, in order with the GIF work. The caller does not wait; it must keep
// `arg` alive until fn ran (fn usually deletes it).
void ps2RunAfterQueuedGif(void (*fn)(void *), void *arg);
// Blocks until the worker has run everything queued so far (any thread except the worker). Teardown helper.
void ps2FlushQueuedGif();

// Guest vblank period (default 16667 us, NTSC). The host may raise the vblank rate (e.g. for high refresh output);
// takes effect from the next scheduled vblank.
void ps2SetVblankPeriodMicros(uint32_t micros);
uint32_t ps2VblankPeriodMicros();
