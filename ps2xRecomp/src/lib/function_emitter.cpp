#include "ps2recomp/Emitters/function_emitter.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/gif_dma_kick_analyzer.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/r5900_decoder.h"
#include "ps2recomp/recompiler_reporter.h"
#include "ps2recomp/types.h"
#include "ps2recomp/control_flow_utils.h"

#include <algorithm>
#include <cctype>
#include <fmt/format.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace ps2recomp
{
    namespace
    {

        // ---- register-locals rewrite ---------------------------------------------------------------------------
        struct LocalsUsage
        {
            std::set<int> used;    // GPRs read or written (low 64 bits)
            std::set<int> written; // GPRs written (flushed back to ctx)
            std::set<int> hiUsed;  // GPRs used by 128-bit ops (upper 64 bits kept as well)
            std::set<int> hiWritten;
        };

        bool isIdentStart(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
        bool isIdentChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

        // Rewrites the GPR / memory macros of a function body to their locals-mode forms and records which
        // registers are touched:
        //   GPR_U32(ctx, 4)           -> L_GPR_U32(4)
        //   SET_GPR_S32(ctx, 4, v)    -> L_SET_GPR_S32(4, v)          (reg 0: PS2_DISCARD(v))
        //   READ32(a) / WRITE32(a,v)  -> L_READ32(a) / L_WRITE32(a,v)
        std::string rewriteForLocals(const std::string &in, LocalsUsage &use, const std::string &functionName)
        {
            static const std::unordered_set<std::string> kGet = {"GPR_U32", "GPR_S32", "GPR_U64", "GPR_S64", "GPR_VEC"};
            static const std::unordered_set<std::string> kSet = {"SET_GPR_U32", "SET_GPR_S32", "SET_GPR_ZE32",
                                                                 "SET_GPR_U64", "SET_GPR_S64", "SET_GPR_VEC"};
            static const std::unordered_set<std::string> kMem = {"READ8", "READ16", "READ32", "READ64", "READ128",
                                                                 "WRITE8", "WRITE16", "WRITE32", "WRITE64", "WRITE128"};
            std::string out;
            out.reserve(in.size() + in.size() / 8);
            const size_t n = in.size();
            size_t i = 0;
            auto skipSpaces = [&](size_t &k)
            {
                while (k < n && (in[k] == ' ' || in[k] == '\t'))
                    ++k;
            };
            auto fail = [&](const std::string &what, size_t at)
            {
                throw std::runtime_error("locals codegen: " + what + " in " + functionName + " near: " +
                                         in.substr(at, 60));
            };
            while (i < n)
            {
                const char c = in[i];
                if (!isIdentStart(c))
                {
                    out.push_back(c);
                    ++i;
                    continue;
                }
                size_t j = i;
                while (j < n && isIdentChar(in[j]))
                    ++j;
                const std::string ident = in.substr(i, j - i);
                const bool get = kGet.contains(ident);
                const bool set = kSet.contains(ident);
                if (get || set)
                {
                    size_t k = j;
                    skipSpaces(k);
                    if (k >= n || in[k] != '(')
                        fail("expected '(' after " + ident, i);
                    ++k;
                    skipSpaces(k);
                    if (in.compare(k, 3, "ctx") != 0)
                        fail("expected ctx in " + ident, i);
                    k += 3;
                    skipSpaces(k);
                    if (k >= n || in[k] != ',')
                        fail("expected ',' in " + ident, i);
                    ++k;
                    skipSpaces(k);
                    size_t d = k;
                    while (d < n && std::isdigit(static_cast<unsigned char>(in[d])))
                        ++d;
                    if (d == k)
                        fail("non-literal register in " + ident, i);
                    const int reg = std::atoi(in.substr(k, d - k).c_str());
                    if (reg < 0 || reg > 31)
                        fail("register out of range", i);
                    k = d;
                    const bool vec = ident.size() >= 3 && ident.compare(ident.size() - 3, 3, "VEC") == 0;
                    if (set)
                    {
                        skipSpaces(k);
                        if (k >= n || in[k] != ',')
                            fail("expected value in " + ident, i);
                        ++k;
                        skipSpaces(k);
                        if (reg == 0)
                        {
                            out += "PS2_DISCARD(";
                        }
                        else
                        {
                            out += "L_" + ident + "(" + std::to_string(reg) + ", ";
                            use.used.insert(reg);
                            use.written.insert(reg);
                            if (vec)
                            {
                                use.hiUsed.insert(reg);
                                use.hiWritten.insert(reg);
                            }
                        }
                    }
                    else
                    {
                        out += "L_" + ident + "(" + std::to_string(reg);
                        use.used.insert(reg);
                        if (vec && reg != 0)
                            use.hiUsed.insert(reg);
                    }
                    i = k;
                    continue;
                }
                if (kMem.contains(ident))
                {
                    out += "L_" + ident;
                    i = j;
                    continue;
                }
                out += ident;
                i = j;
            }
            if (out.find("ctx->r[") != std::string::npos)
                fail("direct ctx->r[] access", out.find("ctx->r["));
            return out;
        }

        std::string localsPrologue(const LocalsUsage &use)
        {
            std::string p;
            if (use.used.contains(0))
            {
                p += "    constexpr uint64_t gpr0 = 0; constexpr uint64_t gprh0 = 0; (void)gpr0; (void)gprh0;\n";
            }
            for (int r : use.used)
            {
                if (r == 0)
                    continue;
                p += fmt::format("    uint64_t gpr{} = PS2_CG_LO(ctx, {});\n", r, r);
                if (use.hiUsed.contains(r))
                    p += fmt::format("    uint64_t gprh{} = PS2_CG_HI(ctx, {});\n", r, r);
            }
            p += "#define PS2_FLUSH() do { \\\n";
            for (int r : use.written)
            {
                p += fmt::format("        PS2_CG_LO(ctx, {}) = gpr{}; \\\n", r, r);
                if (use.hiWritten.contains(r))
                    p += fmt::format("        PS2_CG_HI(ctx, {}) = gprh{}; \\\n", r, r);
            }
            p += "    } while (0)\n";
            p += "#define PS2_RELOAD() do { \\\n";
            for (int r : use.used)
            {
                if (r == 0)
                    continue;
                p += fmt::format("        gpr{} = PS2_CG_LO(ctx, {}); \\\n", r, r);
                if (use.hiUsed.contains(r))
                    p += fmt::format("        gprh{} = PS2_CG_HI(ctx, {}); \\\n", r, r);
            }
            p += "    } while (0)\n";
            return p;
        }

        Instruction makeSyntheticDelaySlot(uint32_t address)
        {
            Instruction inst{};
            inst.address = address;
            inst.raw = 0;
            inst.opcode = OPCODE_SPECIAL;
            inst.function = SPECIAL_SLL;
            return inst;
        }
    }

    FunctionEmitter::FunctionEmitter(CodeGenerator &codeGenerator)
        : m_codeGenerator(codeGenerator)
    {
    }

    std::string FunctionEmitter::emit(
        const Function &function,
        const std::vector<Instruction> &instructions,
        bool useHeaders)
    {
        CodeGenerator &cg = m_codeGenerator;
        std::stringstream ss;
        cg.m_currentFunctionName = function.name;

        // Register-locals mode (PS2X_CODEGEN=locals): see ps2_cg.h. The function body is generated with the classic
        // macros first, then rewritten to use C++ locals for the guest GPRs.
        const bool locals = cg.usesLocals(function.start);
        cg.m_localsMode = locals;
        cg.m_dspMode = locals && cg.m_dspEnabled;
        cg.m_curSelfReturns.clear();
        cg.m_curDelaySlotAddrs.clear();
        if (locals)
        {
            for (size_t i = 0; i + 1 < instructions.size(); ++i)
            {
                if (instructions[i].hasDelaySlot && instructions[i + 1].address == instructions[i].address + 4u)
                {
                    cg.m_curDelaySlotAddrs.insert(instructions[i].address + 4u);
                }
            }
        }

        if (useHeaders)
        {
            ss << "#include <stdexcept>\n";
            ss << "#include \"ps2_runtime_macros.h\"\n";
            ss << "#include \"ps2_runtime.h\"\n";
            ss << "#include <ps2_recompiled_functions.h>\n";
            ss << "#include <ps2_recompiled_stubs.h>\n\n";
            ss << "#include \"ps2_syscalls.h\"\n";
            ss << "#include \"ps2_stubs.h\"\n\n";
            if (locals)
            {
                ss << "#include \"ps2_cg.h\"\n";
                if (cg.m_dspMode)
                {
                    ss << "#include \"ps2_cg_dsp.h\"\n";
                }
                ss << "\n";
            }
            ss << "#ifdef PS2_FUNCTION_LOG_TRACKER\n";
            ss << "#include \"ps2_log.h\"\n";
            ss << "#endif\n\n";
        }

        CodeGenerator::AnalysisResult analysisResult = cg.collectInternalBranchTargets(function, instructions);
        std::vector<uint32_t> resumeTargets(analysisResult.resumeEntryPoints.begin(),
                                            analysisResult.resumeEntryPoints.end());
        auto resumeIt = cg.m_resumeEntryTargetsByOwner.find(function.start);
        if (resumeIt != cg.m_resumeEntryTargetsByOwner.end())
        {
            resumeTargets.insert(resumeTargets.end(), resumeIt->second.begin(), resumeIt->second.end());
        }
        std::sort(resumeTargets.begin(), resumeTargets.end());
        resumeTargets.erase(std::unique(resumeTargets.begin(), resumeTargets.end()), resumeTargets.end());
        for (uint32_t target : resumeTargets)
        {
            analysisResult.entryPoints.insert(target);
        }

        const std::unordered_set<uint32_t> &internalTargets = analysisResult.entryPoints;
        if (cg.m_dspMode)
        {
            // A `jal` into this same function is a goto (control_flow_emitter emitInternalTarget); its return address is
            // a label here. The matching `jr $ra` jumps to it directly (ps2_cg_dsp.h) instead of returning to the scheduler.
            for (const Instruction &inst : instructions)
            {
                if (inst.opcode != OPCODE_JAL)
                    continue;
                const uint32_t callTarget = buildAbsoluteJumpTarget(inst.address, inst.target);
                const uint32_t ret = inst.address + 8u;
                if (internalTargets.contains(callTarget) && internalTargets.contains(ret) &&
                    !cg.m_curDelaySlotAddrs.contains(ret))
                {
                    cg.m_curSelfReturns.insert(ret);
                }
            }
        }
        ConstantRegisterState constantRegisters;
        GifDmaKickPlan gifDmaKickPlan{};
        ss << "// Function: " << function.name << "\n";
        ss << "// Address: 0x" << std::hex << function.start << " - 0x" << function.end << std::dec << "\n";

        std::string sanitizedName = cg.getFunctionName(function.start);
        if (sanitizedName.empty())
        {
            std::stringstream nameBuilder;
            nameBuilder << "Errorfunc_" << std::hex << function.start;
            sanitizedName = nameBuilder.str();
        }

        ss << "void " << sanitizedName << "(uint8_t* rdram, R5900Context* ctx, PS2Runtime *runtime) {\n";
        ss << "#ifdef PS2_FUNCTION_LOG_TRACKER\n";
        ss << "    PS_LOG_ENTRY(\"" << sanitizedName << "\");\n";
        ss << "#endif\n";
        ss << "\n";

        // Resume switch (goes after the locals are loaded in locals mode).
        std::stringstream switchText;
        if (!resumeTargets.empty())
        {
            // A call enters at the function start, which is not a resume label: test it first so that a function with many
            // resume labels (folded blocks, PS2X_CODEGEN_FOLD) does not walk the switch on every call.
            if (cg.m_dspMode)
            {
                switchText << "    if (ctx->pc != 0x" << std::hex << function.start << "u) [[unlikely]]\n" << std::dec;
            }
            switchText << "    switch (ctx->pc) {\n";
            for (uint32_t target : resumeTargets)
            {
                switchText << "        case 0x" << std::hex << target << "u: goto label_" << target << ";\n"
                           << std::dec;
            }
            switchText << "        default: break;\n";
            switchText << "    }\n\n";
        }

        std::stringstream body;
        if (!locals)
        {
            body << "    ctx->pc = 0x" << std::hex << function.start << "u;\n"
                 << std::dec;
            body << "\n";
        }

        bool lastInstructionWasControlFlow = false;

        for (size_t i = 0; i < instructions.size(); ++i)
        {
            const Instruction &inst = instructions[i];
            lastInstructionWasControlFlow = inst.hasDelaySlot;

            if (internalTargets.contains(inst.address))
            {
                constantRegisters.clear();
                body << "label_" << std::hex << inst.address << std::dec << ":\n";
            }

            if (cg.m_emitInstructionComments)
            {
                body << "    // 0x" << std::hex << inst.address << ": 0x" << inst.raw << std::dec;
                std::string disassembly = R5900Decoder::disassembleInstruction(inst);
                if (!disassembly.empty())
                {
                    body << "  " << disassembly;
                }
                body << "\n";
            }

            try
            {
                if (inst.hasDelaySlot)
                {
                    const bool hasDecodedDelaySlot =
                        i + 1 < instructions.size() &&
                        instructions[i + 1].address == inst.address + 4u;

                    Instruction syntheticDelaySlot{};
                    const Instruction *delaySlot = nullptr;
                    if (hasDecodedDelaySlot)
                    {
                        delaySlot = &instructions[i + 1];
                    }
                    else
                    {
                        syntheticDelaySlot = makeSyntheticDelaySlot(inst.address + 4u);
                        delaySlot = &syntheticDelaySlot;
                    }

                    if (hasDecodedDelaySlot && internalTargets.contains(delaySlot->address))
                    {
                        body << "label_" << std::hex << delaySlot->address << std::dec << ":\n";
                    }

                    if (gifDmaKickPlan.valid &&
                        gifDmaKickPlan.completesInDelaySlot &&
                        gifDmaKickPlan.branchIndex == i &&
                        hasDecodedDelaySlot)
                    {
                        body << cg.handleBranchDelaySlots(
                            inst,
                            *delaySlot,
                            function,
                            analysisResult,
                            gifDmaDelaySlotOverride(*delaySlot, gifDmaKickPlan, cg.m_emitInstructionComments));
                        gifDmaKickPlan = {};
                    }
                    else
                    {
                        body << cg.handleBranchDelaySlots(inst, *delaySlot, function, analysisResult);
                    }

                    if (hasDecodedDelaySlot)
                    {
                        ++i; // Skip delay slot instruction (handled inside branch logic)
                    }
                    constantRegisters.clear();
                }
                else
                {
                    if (!gifDmaKickPlan.valid)
                    {
                        gifDmaKickPlan = tryBuildGifDmaKickPlan(instructions, i, constantRegisters, internalTargets);
                    }

                    if (gifDmaKickPlan.suppresses(i))
                    {
                        const size_t slot = gifDmaKickPlan.slotFor(i);
                        emitGifDmaCapture(body, gifDmaKickPlan, slot, "    ");

                        if (gifDmaKickPlan.completesAt(i))
                        {
                            body << "    ctx->pc = 0x" << std::hex << inst.address << "u;\n"
                                 << std::dec;
                            body << "    " << gifDmaKickCall(gifDmaKickPlan) << "\n";
                            gifDmaKickPlan = {};
                        }

                        updateConstantRegisters(inst, constantRegisters);
                        continue;
                    }

                    const MemoryAccessHint memoryHint = resolveMemoryAccessHint(inst, constantRegisters);
                    std::string code = cg.translateInstruction(inst, memoryHint);
                    if (!locals || CodeGenerator::codeNeedsPc(code))
                    {
                        body << "    ctx->pc = 0x" << std::hex << inst.address << "u;\n"
                             << std::dec;
                    }
                    if (locals)
                    {
                        code = cg.wrapSync(code);
                    }
                    body << "    " << code;
                    if (inst.isMmio)
                    {
                        body << " // MMIO: 0x" << std::hex << inst.mmioAddress << std::dec;
                    }
                    body << "\n";

                    updateConstantRegisters(inst, constantRegisters);
                }
            }
            catch (const std::exception &e)
            {
                if (cg.m_reporter)
                {
                    std::ostringstream msg;
                    msg << "translation failed: " << e.what() << " raw=0x" << std::hex << inst.raw;
                    cg.m_reporter->errorAt("codegen", function.name, inst.address, msg.str());
                }

                throw;
            }
        }

        if (!locals)
        {
            // Fallthrough with no terminating branch: publish the next PC so the EE dispatcher does not re-enter this function.
            if (!instructions.empty() && !lastInstructionWasControlFlow)
            {
                body << "    ctx->pc = 0x" << std::hex << function.end << "u;\n"
                     << std::dec;
            }
            ss << switchText.str() << body.str();
            ss << "}\n";
            return ss.str();
        }

        // Locals mode: falling off the end publishes the next pc and writes the registers back.
        if (!instructions.empty())
        {
            body << "    PS2_FLUSH();\n";
            if (cg.m_dspMode)
            {
                // Falling off the end continues in the next function or fragment: a tail jump.
                body << "    ps2DspTail(ctx, 0x" << std::hex << function.end << "u);\n"
                     << std::dec;
            }
            else
            {
                body << "    ctx->pc = 0x" << std::hex << function.end << "u;\n"
                     << std::dec;
            }
        }

        LocalsUsage use;
        std::string rewritten = rewriteForLocals(body.str(), use, function.name);
        ss << localsPrologue(use);
        ss << "\n"
           << switchText.str() << rewritten;
        ss << "#undef PS2_FLUSH\n#undef PS2_RELOAD\n";
        ss << "}\n";
        return ss.str();
    }
}
