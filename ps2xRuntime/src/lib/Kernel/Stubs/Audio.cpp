#include "Common.h"
#include "Audio.h"
#include "../../ps2_iop_transport.h"

#include <chrono>
#include <thread>

namespace ps2_stubs
{
    namespace
    {
        constexpr uint32_t kLibSdCmdSetParam = 0x8010u;
        constexpr uint32_t kLibSdCmdVoiceTrans = 0x80D0u;
        constexpr uint32_t kLibSdCmdBlockTrans = 0x80E0u;
        constexpr uint32_t kLibSdCmdVoiceTransStatus = 0x80F0u;
        constexpr uint32_t kLibSdCmdBlockTransStatus = 0x8100u;
        constexpr uint32_t kAudioPositionMask = 0x00FFFFFFu;
        constexpr uint32_t kAudioTransferUnit = 1024u;
        constexpr uint32_t kLibSdCoreCount = 2u;
        constexpr uint32_t kLibSdTransModeIo = 0x08u;
        constexpr uint32_t kLibSdTransDirectionMask = 0x03u;
        constexpr uint32_t kLibSdTransStop = 0x02u;
        constexpr uint32_t kLibSdTransLoop = 0x10u;

        struct VoiceTransferState
        {
            uint32_t sourceAddress = 0u;
            uint32_t destinationAddress = 0u;
            uint32_t size = 0u;
            uint16_t mode = 0u;
            bool completed = true;
        };

        struct BlockTransferState
        {
            uint32_t base = 0u;
            uint32_t size = 0u;
            uint32_t pauseBase = 0u;
            uint32_t offset = 0u;
            uint32_t statusTraceCount = 0u;
            uint16_t mode = 0u;
            bool active = false;
            bool loop = false;
        };

        struct AudioStubState
        {
            bool initialized = false;
            std::array<VoiceTransferState, kLibSdCoreCount> voiceTransfers{};
            std::array<BlockTransferState, kLibSdCoreCount> blockTransfers{};
        };

        std::mutex g_audio_stub_mutex;
        AudioStubState g_audio_stub_state;

        void resetAudioStubStateUnlocked()
        {
            g_audio_stub_state = {};
        }

        // ---- libsd remote (sceSdRemote / sceSdRemoteInit) over the real SIF RPC -----------------------------------
        // The recompiler binds the SDK's EE-side libsd client (sdr.c) to these stubs. With an emulated SDRDRV.IRX behind
        // sid 0x80000701 the calls must reach it: a game that drives the SPU2 itself (Killzone's movie player uploads
        // the FMV ADPCM and starts its voices this way) is otherwise silent. The code below is the client's argument
        // packing (read from the game's copy of the library): the six arguments after the command go to a 0x40-byte
        // buffer from word 1 (word 0 is overwritten by the reply), the call is a blocking sceSifCallRpc with the
        // command as the RPC function number, and the first word of the reply is the return value.
        constexpr uint32_t kLibSdSid = 0x80000701u;
        constexpr uint32_t kSdrBufSize = 0x40u;

        struct SdrRemoteState
        {
            uint32_t client = 0u;  // t_SifRpcClientData (0x40 bytes)
            uint32_t buffer = 0u;  // send/reply buffer (0x40 bytes)
            uint32_t scratch = 0u; // private stack for the SifCallRpc stack-argument reads
            bool bound = false;
        };
        SdrRemoteState g_sdr;

        bool sdrAllocate(PS2Runtime *runtime)
        {
            if (g_sdr.client != 0u)
                return true;
            const uint32_t base = runtime->guestMalloc(0x200u, 64u);
            if (base == 0u)
                return false;
            g_sdr.client = base;
            g_sdr.buffer = base + 0x80u;
            g_sdr.scratch = base + 0x100u;
            return true;
        }

        uint32_t sdrRead(uint8_t *rdram, uint32_t addr)
        {
            uint32_t v = 0u;
            if (const uint8_t *p = getConstMemPtr(rdram, addr))
                std::memcpy(&v, p, sizeof(v));
            return v;
        }

        void sdrWrite(uint8_t *rdram, uint32_t addr, uint32_t v)
        {
            if (uint8_t *p = getMemPtr(rdram, addr))
                std::memcpy(p, &v, sizeof(v));
        }

        void sdrCall(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t cmd,
                     uint32_t sendBuf, uint32_t sendSize, uint32_t recvBuf, uint32_t recvSize)
        {
            R5900Context c = *ctx;
            SET_GPR_U32(&c, 4, g_sdr.client);
            SET_GPR_U32(&c, 5, cmd);
            SET_GPR_U32(&c, 6, 0u); // blocking
            SET_GPR_U32(&c, 7, sendBuf);
            SET_GPR_U32(&c, 8, sendSize);
            SET_GPR_U32(&c, 9, recvBuf);
            SET_GPR_U32(&c, 10, recvSize);
            SET_GPR_U32(&c, 11, 0u); // no end function
            if (uint8_t *sp = getMemPtr(rdram, g_sdr.scratch))
                std::memset(sp, 0, 0x40u);
            SET_GPR_U32(&c, 29, g_sdr.scratch);
            ps2_syscalls::SifCallRpc(rdram, &c, runtime);
        }

        // Returns true when the call went to the emulated SDRDRV.
        bool sdrRemote(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
        {
            if (!g_sdr.bound)
                return false;
            uint32_t cmd = getRegU32(ctx, 5);
            // The six arguments after the command: a2, a3, t0..t3 (the SDK function takes all eight in registers).
            const uint32_t a[6] = {getRegU32(ctx, 6), getRegU32(ctx, 7), getRegU32(ctx, 8),
                                   getRegU32(ctx, 9), getRegU32(ctx, 10), getRegU32(ctx, 11)};
            uint8_t *buf = getMemPtr(rdram, g_sdr.buffer);
            if (!buf)
                return false;
            std::memset(buf, 0, kSdrBufSize);
            sdrWrite(rdram, g_sdr.buffer, g_sdr.buffer);
            for (uint32_t i = 0; i < 6u; ++i)
                sdrWrite(rdram, g_sdr.buffer + 4u + i * 4u, a[i]);

            uint32_t result = 0u;
            if (cmd == 0x8130u || cmd == 0x81A0u || cmd == 0x81B0u)
            {
                cmd |= a[0];
                sdrCall(rdram, ctx, runtime, cmd, a[1], kSdrBufSize, g_sdr.buffer, 0x40u);
                result = sdrRead(rdram, g_sdr.buffer);
            }
            else if (cmd == 0x8140u)
            {
                cmd |= a[0];
                sdrCall(rdram, ctx, runtime, cmd, g_sdr.buffer, kSdrBufSize, a[1], 0x40u);
            }
            else if (cmd == 0x81C0u || cmd == 0x81D0u)
            {
                // sceSdProcBatch / sceSdProcBatchEx: batch table a[0] with a[1]+1 entries of 8 bytes, replies to a[2]
                // (a[3] bytes), the extended form also passes a[4] in the table's second word.
                if (uint8_t *b = getMemPtr(rdram, a[0] + 2u))
                {
                    const uint16_t n = static_cast<uint16_t>(a[1]);
                    std::memcpy(b, &n, sizeof(n));
                }
                if (cmd == 0x81D0u)
                    sdrWrite(rdram, a[0] + 4u, a[4]);
                uint32_t rb = a[2], rs = a[3];
                if (rb == 0u)
                {
                    rb = g_sdr.buffer;
                    rs = 4u;
                }
                sdrCall(rdram, ctx, runtime, cmd, a[0], (a[1] + 1u) * 8u, rb, rs);
                result = a[2] != 0u ? sdrRead(rdram, a[2]) : 0u;
            }
            else
            {
                uint32_t sendBuf = g_sdr.buffer, sendSize = kSdrBufSize;
                if (cmd - 0x9000u <= 0xF0u)
                {
                    sendBuf = a[0];
                    sendSize = a[1];
                }
                sdrCall(rdram, ctx, runtime, cmd, sendBuf, sendSize, g_sdr.buffer, 0x10u);
                result = sdrRead(rdram, g_sdr.buffer);
            }
            setReturnU32(ctx, result);
            return true;
        }

        uint32_t currentBlockStatus(const BlockTransferState &transfer)
        {
            const uint32_t position = (transfer.base + transfer.offset) & kAudioPositionMask;
            const uint32_t halfSize = transfer.size / 2u;
            const uint32_t bank = (transfer.loop && halfSize != 0u && transfer.offset >= halfSize) ? 1u : 0u;
            return (bank << 24u) | position;
        }
    }

    void resetAudioStubState()
    {
        std::lock_guard<std::mutex> lock(g_audio_stub_mutex);
        resetAudioStubStateUnlocked();
    }

    void sceSdCallBack(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSdCallBack", rdram, ctx, runtime);
    }

    void sceSdRemote(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (sdrRemote(rdram, ctx, runtime))
            return;

        const uint32_t cmd = getRegU32(ctx, 5);
        const uint32_t cmdArg0 = getRegU32(ctx, 6);
        const uint32_t cmdArg1 = getRegU32(ctx, 7);
        const uint32_t sp = getRegU32(ctx, 29);
        const uint32_t arg4Reg = getRegU32(ctx, 8);
        const uint32_t arg5Reg = getRegU32(ctx, 9);
        const uint32_t arg6Reg = getRegU32(ctx, 10);
        const uint32_t arg4Stk = FAST_READ32(sp + 0x10u);
        const uint32_t arg5Stk = FAST_READ32(sp + 0x14u);
        const uint32_t arg6Stk = FAST_READ32(sp + 0x18u);
        const uint32_t arg4 = (arg4Reg != 0u) ? arg4Reg : arg4Stk;
        const uint32_t arg5 = (arg5Reg != 0u) ? arg5Reg : arg5Stk;
        const uint32_t arg6 = (arg6Reg != 0u) ? arg6Reg : arg6Stk;

        std::lock_guard<std::mutex> lock(g_audio_stub_mutex);
        g_audio_stub_state.initialized = true;
        const uint32_t core = cmdArg0 & (kLibSdCoreCount - 1u);
        VoiceTransferState &voiceTransfer = g_audio_stub_state.voiceTransfers[core];
        BlockTransferState &blockTransfer = g_audio_stub_state.blockTransfers[core];
        uint32_t returnValue = 0u;

        if (cmd == kLibSdCmdVoiceTrans)
        {
            voiceTransfer.sourceAddress = arg4;
            voiceTransfer.destinationAddress = arg5;
            voiceTransfer.size = arg6;
            voiceTransfer.mode = static_cast<uint16_t>(cmdArg1);
            voiceTransfer.completed = true;

            // sceSdVoiceTrans returns the transferred byte count for DMA
            // mode and zero for its synchronous programmed-I/O mode.
            returnValue = ((voiceTransfer.mode & kLibSdTransModeIo) != 0u)
                              ? 0u
                              : voiceTransfer.size;

            PS2_IF_AGRESSIVE_LOGS({
                std::cerr << "[Audio:VoiceTrans] core=" << core
                          << " src=0x" << std::hex << voiceTransfer.sourceAddress
                          << " dest=0x" << voiceTransfer.destinationAddress
                          << " size=0x" << voiceTransfer.size
                          << " mode=0x" << voiceTransfer.mode
                          << std::dec << std::endl;
            });
        }
        else if (cmd == kLibSdCmdBlockTrans)
        {
            const uint32_t direction = cmdArg1 & kLibSdTransDirectionMask;
            if (direction == kLibSdTransStop)
            {
                returnValue = currentBlockStatus(blockTransfer);
                blockTransfer = {};
            }
            else if (arg4 != 0u && arg5 != 0u)
            {
                blockTransfer.base = arg4 & kAudioPositionMask;
                blockTransfer.size = arg5;
                blockTransfer.pauseBase =
                    ((arg6 != 0u) ? arg6 : arg4) & kAudioPositionMask;
                blockTransfer.mode = static_cast<uint16_t>(cmdArg1);
                blockTransfer.loop = (cmdArg1 & kLibSdTransLoop) != 0u;

                const uint32_t pauseOffset =
                    (blockTransfer.pauseBase - blockTransfer.base) & kAudioPositionMask;
                blockTransfer.offset = (pauseOffset < blockTransfer.size) ? pauseOffset : 0u;
                blockTransfer.statusTraceCount = 0u;
                blockTransfer.active = true;
                returnValue = 0u;
            }
            else
            {
                returnValue = std::numeric_limits<uint32_t>::max();
            }

            PS2_IF_AGRESSIVE_LOGS({
                std::cerr << "[Audio:BlockTrans] core=" << core
                          << " active=" << blockTransfer.active
                          << " base=0x" << std::hex << blockTransfer.base
                          << " size=0x" << blockTransfer.size
                          << " pause=0x" << blockTransfer.pauseBase
                          << " offset=0x" << blockTransfer.offset
                          << std::dec << std::endl;
            });
        }
        else if (cmd == kLibSdCmdVoiceTransStatus)
        {
            returnValue = voiceTransfer.completed ? 1u : 0u;
        }
        else if (cmd == kLibSdCmdBlockTransStatus)
        {
            if (blockTransfer.active && blockTransfer.size != 0u)
            {
                blockTransfer.offset = (blockTransfer.offset + kAudioTransferUnit) % blockTransfer.size;
                if (blockTransfer.statusTraceCount < 32u)
                {
                    PS2_IF_AGRESSIVE_LOGS({
                        std::cerr << "[Audio:BlockStatus] core=" << core
                                  << " status=0x" << std::hex << currentBlockStatus(blockTransfer)
                                  << " offset=0x" << blockTransfer.offset
                                  << " size=0x" << blockTransfer.size
                                  << std::dec << std::endl;
                    });
                    ++blockTransfer.statusTraceCount;
                }
            }
            returnValue = currentBlockStatus(blockTransfer);
        }
        else if (cmd == kLibSdCmdSetParam)
        {
            (void)cmdArg0;
            (void)cmdArg1;
            returnValue = 0u;
        }

        setReturnU32(ctx, returnValue);
    }

    void sceSdRemoteInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        // Bind to the emulated SDRDRV when there is one (the library's init loops on sceSifBindRpc until the server
        // shows up; the IOP thread keeps running meanwhile). Without one the old stub model stays in charge.
        g_sdr.bound = false;
        static const bool s_remoteEnabled = [] { const char *v = std::getenv("PS2X_SDR_REMOTE"); return !(v && v[0] == '0'); }();
        if (s_remoteEnabled && PS2IopTransport::canBindRpc(runtime, kLibSdSid) && sdrAllocate(runtime))
        {
            for (int attempt = 0; attempt < 2000 && !g_sdr.bound; ++attempt)
            {
                R5900Context c = *ctx;
                SET_GPR_U32(&c, 4, g_sdr.client);
                SET_GPR_U32(&c, 5, kLibSdSid);
                SET_GPR_U32(&c, 6, 0u);
                ps2_syscalls::SifBindRpc(rdram, &c, runtime);
                if (sdrRead(rdram, g_sdr.client + 0x24u) != 0u)
                    g_sdr.bound = true;
                else
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }

        std::lock_guard<std::mutex> lock(g_audio_stub_mutex);
        resetAudioStubStateUnlocked();
        g_audio_stub_state.initialized = true;
        setReturnS32(ctx, 0);
    }

    void sceSdTransToIOP(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSdTransToIOP", rdram, ctx, runtime);
    }

    void sceSSyn_BreakAtick(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_BreakAtick", rdram, ctx, runtime);
    }

    void sceSSyn_ClearBreakAtick(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_ClearBreakAtick", rdram, ctx, runtime);
    }

    void sceSSyn_SendExcMsg(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SendExcMsg", rdram, ctx, runtime);
    }

    void sceSSyn_SendNrpnMsg(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SendNrpnMsg", rdram, ctx, runtime);
    }

    void sceSSyn_SendRpnMsg(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SendRpnMsg", rdram, ctx, runtime);
    }

    void sceSSyn_SendShortMsg(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SendShortMsg", rdram, ctx, runtime);
    }

    void sceSSyn_SetChPriority(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetChPriority", rdram, ctx, runtime);
    }

    void sceSSyn_SetMasterVolume(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetMasterVolume", rdram, ctx, runtime);
    }

    void sceSSyn_SetOutPortVolume(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetOutPortVolume", rdram, ctx, runtime);
    }

    void sceSSyn_SetOutputAssign(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetOutputAssign", rdram, ctx, runtime);
    }

    void sceSSyn_SetOutputMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceSSyn_SetPortMaxPoly(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetPortMaxPoly", rdram, ctx, runtime);
    }

    void sceSSyn_SetPortVolume(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetPortVolume", rdram, ctx, runtime);
    }

    void sceSSyn_SetTvaEnvMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSSyn_SetTvaEnvMode", rdram, ctx, runtime);
    }

    void sceSynthesizerAmpProcI(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAmpProcI", rdram, ctx, runtime);
    }

    void sceSynthesizerAmpProcNI(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAmpProcNI", rdram, ctx, runtime);
    }

    void sceSynthesizerAssignAllNoteOff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAssignAllNoteOff", rdram, ctx, runtime);
    }

    void sceSynthesizerAssignAllSoundOff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAssignAllSoundOff", rdram, ctx, runtime);
    }

    void sceSynthesizerAssignHoldChange(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAssignHoldChange", rdram, ctx, runtime);
    }

    void sceSynthesizerAssignNoteOff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAssignNoteOff", rdram, ctx, runtime);
    }

    void sceSynthesizerAssignNoteOn(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerAssignNoteOn", rdram, ctx, runtime);
    }

    void sceSynthesizerCalcEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerCalcEnv", rdram, ctx, runtime);
    }

    void sceSynthesizerCalcPortamentPitch(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerCalcPortamentPitch", rdram, ctx, runtime);
    }

    void sceSynthesizerCalcTvfCoefAll(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerCalcTvfCoefAll", rdram, ctx, runtime);
    }

    void sceSynthesizerCalcTvfCoefF0(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerCalcTvfCoefF0", rdram, ctx, runtime);
    }

    void sceSynthesizerCent2PhaseInc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerCent2PhaseInc", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeEffectSend(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeEffectSend", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeHsPanpot(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeHsPanpot", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeNrpnCutOff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeNrpnCutOff", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeNrpnLfoDepth(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeNrpnLfoDepth", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeNrpnLfoRate(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeNrpnLfoRate", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeOutAttrib(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeOutAttrib", rdram, ctx, runtime);
    }

    void sceSynthesizerChangeOutVol(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangeOutVol", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePanpot(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePanpot", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartBendSens(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartBendSens", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartExpression(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartExpression", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartHsExpression(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartHsExpression", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartHsPitchBend(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartHsPitchBend", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartModuration(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartModuration", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartPitchBend(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartPitchBend", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePartVolume(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePartVolume", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePortamento(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePortamento", rdram, ctx, runtime);
    }

    void sceSynthesizerChangePortamentoTime(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerChangePortamentoTime", rdram, ctx, runtime);
    }

    void sceSynthesizerClearKeyMap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerClearKeyMap", rdram, ctx, runtime);
    }

    void sceSynthesizerClearSpr(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerClearSpr", rdram, ctx, runtime);
    }

    void sceSynthesizerCopyOutput(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerCopyOutput", rdram, ctx, runtime);
    }

    void sceSynthesizerDmaFromSPR(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerDmaFromSPR", rdram, ctx, runtime);
    }

    void sceSynthesizerDmaSpr(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerDmaSpr", rdram, ctx, runtime);
    }

    void sceSynthesizerDmaToSPR(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerDmaToSPR", rdram, ctx, runtime);
    }

    void sceSynthesizerGetPartial(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerGetPartial", rdram, ctx, runtime);
    }

    void sceSynthesizerGetPartOutLevel(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerGetPartOutLevel", rdram, ctx, runtime);
    }

    void sceSynthesizerGetSampleParam(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerGetSampleParam", rdram, ctx, runtime);
    }

    void sceSynthesizerHsMessage(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerHsMessage", rdram, ctx, runtime);
    }

    void sceSynthesizerLfoNone(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerLfoNone", rdram, ctx, runtime);
    }

    void sceSynthesizerLfoProc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerLfoProc", rdram, ctx, runtime);
    }

    void sceSynthesizerLfoSawDown(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerLfoSawDown", rdram, ctx, runtime);
    }

    void sceSynthesizerLfoSawUp(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerLfoSawUp", rdram, ctx, runtime);
    }

    void sceSynthesizerLfoSquare(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerLfoSquare", rdram, ctx, runtime);
    }

    void sceSynthesizerReadNoise(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerReadNoise", rdram, ctx, runtime);
    }

    void sceSynthesizerReadNoiseAdd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerReadNoiseAdd", rdram, ctx, runtime);
    }

    void sceSynthesizerReadSample16(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerReadSample16", rdram, ctx, runtime);
    }

    void sceSynthesizerReadSample16Add(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerReadSample16Add", rdram, ctx, runtime);
    }

    void sceSynthesizerReadSample8(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerReadSample8", rdram, ctx, runtime);
    }

    void sceSynthesizerReadSample8Add(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerReadSample8Add", rdram, ctx, runtime);
    }

    void sceSynthesizerResetPart(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerResetPart", rdram, ctx, runtime);
    }

    void sceSynthesizerRestorDma(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerRestorDma", rdram, ctx, runtime);
    }

    void sceSynthesizerSelectPatch(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSelectPatch", rdram, ctx, runtime);
    }

    void sceSynthesizerSendShortMessage(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSendShortMessage", rdram, ctx, runtime);
    }

    void sceSynthesizerSetMasterVolume(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetMasterVolume", rdram, ctx, runtime);
    }

    void sceSynthesizerSetRVoice(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetRVoice", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupDma(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupDma", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupLfo(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupLfo", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupMidiModuration(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupMidiModuration", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupMidiPanpot(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupMidiPanpot", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupNewNoise(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupNewNoise", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupReleaseEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupReleaseEnv", rdram, ctx, runtime);
    }

    void sceSynthesizerSetuptEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetuptEnv", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupTruncateTvaEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupTruncateTvaEnv", rdram, ctx, runtime);
    }

    void sceSynthesizerSetupTruncateTvfPitchEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerSetupTruncateTvfPitchEnv", rdram, ctx, runtime);
    }

    void sceSynthesizerTonegenerator(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerTonegenerator", rdram, ctx, runtime);
    }

    void sceSynthesizerTransposeMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerTransposeMatrix", rdram, ctx, runtime);
    }

    void sceSynthesizerTvfProcI(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerTvfProcI", rdram, ctx, runtime);
    }

    void sceSynthesizerTvfProcNI(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerTvfProcNI", rdram, ctx, runtime);
    }

    void sceSynthesizerWaitDmaFromSPR(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerWaitDmaFromSPR", rdram, ctx, runtime);
    }

    void sceSynthesizerWaitDmaToSPR(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthesizerWaitDmaToSPR", rdram, ctx, runtime);
    }

    void sceSynthsizerGetDrumPatch(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthsizerGetDrumPatch", rdram, ctx, runtime);
    }

    void sceSynthsizerGetMeloPatch(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthsizerGetMeloPatch", rdram, ctx, runtime);
    }

    void sceSynthsizerLfoNoise(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthsizerLfoNoise", rdram, ctx, runtime);
    }

    void sceSynthSizerLfoTriangle(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSynthSizerLfoTriangle", rdram, ctx, runtime);
    }
}
