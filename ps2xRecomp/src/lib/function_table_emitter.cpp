#include "ps2recomp/Emitters/function_table_emitter.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/ps2_recompiler.h"
#include "ps2recomp/recompiler_reporter.h"
#include "ps2recomp/types.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ps2recomp
{
    FunctionTableEmitter::FunctionTableEmitter(CodeGenerator &codeGenerator)
        : m_codeGenerator(codeGenerator)
    {
    }

    std::string FunctionTableEmitter::emit(const std::vector<Function> &functions, const std::map<uint32_t, std::string> &stubs)
    {
        (void)stubs;

        CodeGenerator &cg = m_codeGenerator;
        std::vector<std::pair<uint32_t, std::string>> entries;
        std::unordered_set<uint32_t> registeredAddresses;

        auto addEntry = [&](uint32_t address, const std::string &name)
        {
            if (name.empty())
            {
                return;
            }
            if ((address & 3u) != 0u)
            {
                std::ostringstream oss;
                oss << "Unaligned function table entry for " << name << " at 0x" << std::hex << address;

                if (cg.m_reporter)
                {
                    cg.m_reporter->errorAt("function-table", name, address, oss.str());
                }

                throw std::runtime_error(oss.str());
            }
            if (!registeredAddresses.insert(address).second)
            {
                return;
            }
            entries.emplace_back(address, name);
        };

        std::vector<std::pair<uint32_t, std::string>> normalFunctions;
        std::vector<std::pair<uint32_t, std::string>> stubFunctions;
        std::vector<std::pair<uint32_t, std::string>> systemCallFunctions;
        std::vector<std::pair<uint32_t, std::string>> libraryFunctions;

        for (const auto &function : functions)
        {
            if (!function.isRecompiled && !function.isStub && !function.isSkipped)
                continue;

            std::string generatedName = cg.getFunctionName(function.start);

            if (function.isSkipped)
            {
                libraryFunctions.emplace_back(function.start, generatedName);
            }
            else if (function.isStub)
            {
                const auto target = PS2Recompiler::resolveStubTarget(function.name);
                if (target == StubTarget::Syscall)
                {
                    systemCallFunctions.emplace_back(function.start, generatedName);
                }
                else
                {
                    stubFunctions.emplace_back(function.start, generatedName);
                }
            }
            else
            {
                normalFunctions.emplace_back(function.start, generatedName);
            }
        }

        if (cg.m_bootstrapInfo.valid)
        {
            std::string entryTarget = cg.m_bootstrapInfo.entryName;
            if (entryTarget.empty())
            {
                entryTarget = cg.getFunctionName(cg.m_bootstrapInfo.entry);
            }
            if (entryTarget.empty())
            {
                throw std::runtime_error("No entry function name available for registration.");
            }
            addEntry(cg.m_bootstrapInfo.entry, entryTarget);
        }

        for (const auto &[address, name] : normalFunctions)
        {
            addEntry(address, name);
        }

        for (const auto &[ownerStart, targets] : cg.m_resumeEntryTargetsByOwner)
        {
            const std::string ownerName = cg.getFunctionName(ownerStart);
            if (ownerName.empty())
            {
                continue;
            }

            for (uint32_t target : targets)
            {
                addEntry(target, ownerName);
            }
        }

        for (const auto &[address, name] : stubFunctions)
        {
            addEntry(address, name);
        }
        for (const auto &[address, name] : systemCallFunctions)
        {
            addEntry(address, name);
        }
        for (const auto &[address, name] : libraryFunctions)
        {
            addEntry(address, name);
        }

        std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        uint32_t tableBase = 0u;
        uint32_t tableEnd = 0u;
        if (!entries.empty())
        {
            tableBase = entries.front().first & ~3u;
            tableEnd = (entries.back().first + 4u + 3u) & ~3u;
        }
        // The set's own range; the register function checks it against the table it is given.
        const uint32_t entriesEnd = tableEnd;
        if (cg.m_tableEndMin != 0u)
        {
            const uint32_t endMin = static_cast<uint32_t>((static_cast<uint64_t>(cg.m_tableEndMin) + 3u) & ~uint64_t{3});
            if (endMin < tableBase)
            {
                std::ostringstream oss;
                oss << "table_end_min 0x" << std::hex << cg.m_tableEndMin << " is below the table base 0x" << tableBase;
                throw std::runtime_error(oss.str());
            }
            tableEnd = std::max(tableEnd, endMin);
        }
        const uint32_t slotCount = (tableEnd - tableBase) >> 2;

        std::stringstream ss;
        ss << "#include \"ps2_runtime.h\"\n";
        ss << "#include <ps2_recompiled_functions.h>\n";
        ss << "#include \"ps2_stubs.h\"\n";
        ss << "#include <ps2_recompiled_stubs.h>\n";
        ss << "#include \"ps2_syscalls.h\"\n";

        if (!cg.m_tableRegisterFunction.empty())
        {
            // Register-function mode: no table globals and no static initializer. The runtime owns the table and
            // calls this function for the set it selected; slots are relative to the base it passes.
            const std::string &fn = cg.m_tableRegisterFunction;
            ss << "#include <cstdio>\n";
            ss << "#include <cstdlib>\n\n";
            ss << "extern const uint32_t " << fn << "End = 0x" << std::hex << tableEnd << "u;\n\n";
            ss << "void " << fn << "(PS2Runtime::RecompiledFunction* table, uint32_t base, uint32_t slotCount) {\n";
            ss << "    if (slotCount != 0u && table == nullptr) {\n";
            ss << "        std::fprintf(stderr, \"" << fn << ": null function table\\n\");\n";
            ss << "        std::abort();\n";
            ss << "    }\n";
            ss << "    for (uint32_t i = 0; i < slotCount; ++i) {\n";
            ss << "        table[i] = nullptr;\n";
            ss << "    }\n";
            if (!entries.empty())
            {
                ss << "    if ((base & 3u) != 0u || base > 0x" << tableBase << "u || ((0x" << entriesEnd
                   << "u - base) >> 2) > slotCount) {\n";
                ss << "        std::fprintf(stderr, \"" << fn << ": set range 0x" << tableBase << "-0x" << entriesEnd
                   << " does not fit a table at base 0x%x with %u slots\\n\", base, slotCount);\n";
                ss << "        std::abort();\n";
                ss << "    }\n";
            }
            for (const auto &[address, name] : entries)
            {
                ss << "    table[(0x" << address << "u - base) >> 2] = " << name << ";\n";
            }
            ss << "}\n";
            return ss.str();
        }

        ss << "\n";
        ss << "extern const uint32_t g_ps2RecompiledFunctionTableBase = 0x" << std::hex << tableBase << "u;\n";
        ss << "extern const uint32_t g_ps2RecompiledFunctionTableEnd = 0x" << std::hex << tableEnd << "u;\n";
        ss << "extern const uint32_t g_ps2RecompiledFunctionTableSlotCount = " << std::dec << slotCount << "u;\n";
        ss << "PS2Runtime::RecompiledFunction g_ps2RecompiledFunctionTable[" << std::dec << (slotCount == 0u ? 1u : slotCount) << "u] = {};\n\n";

        ss << "namespace {\n";
        ss << "struct GeneratedFunctionTableInitializer {\n";
        ss << "    GeneratedFunctionTableInitializer() {\n";
        for (const auto &[address, name] : entries)
        {
            const uint32_t slot = (address - tableBase) >> 2;
            ss << "        g_ps2RecompiledFunctionTable[" << std::dec << slot << "] = " << name
               << "; // 0x" << std::hex << address << std::dec << "\n";
        }
        ss << "    }\n";
        ss << "};\n";
        ss << "static const GeneratedFunctionTableInitializer g_generatedFunctionTableInitializer;\n";
        ss << "}\n";

        return ss.str();
    }
}
