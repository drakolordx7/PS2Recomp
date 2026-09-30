#pragma once

#include <cstdint>

// Where the runtime keeps its own guest-memory allocations (HLE buffers via guestMalloc) and the stacks it runs
// interrupt handlers / callbacks on. By default they follow the game's SetupHeap region and the top of RAM, which is
// fine for games that use the kernel heap, but collides with games whose own allocator owns all memory from the end of
// the ELF to the top of RAM. A host can move both into a region the game never touches (e.g. the unused EE kernel area).
// Pass 0 for a limit to keep the default. Call before PS2Runtime::loadELF.
void ps2SetRuntimeArena(uint32_t heapBase, uint32_t heapLimit, uint32_t stackFloor, uint32_t stackTop);
