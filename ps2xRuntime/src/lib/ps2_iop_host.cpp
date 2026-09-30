#include "ps2_iop_host.h"
#include "ps2_iop_post.h"
#include "ps2_iop_async.h"

#include "ps2_runtime.h"
#include "ps2x/iop/iop_subsystem.h"
#include "ps2_stubs.h"
#include "Kernel/Stubs/SIF.h"
#include "runtime/ps2_memory.h"
#include "Kernel/Stubs/MemoryCard.h"
#include "Kernel/Syscalls/Common.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <utility>

#if !defined(_WIN32)
#include <sys/types.h>
#endif

namespace
{
    // The adapter whose posted-work queue EeScheduler drains (ps2_iop_post.h). One runtime per process.
    std::atomic<PS2IopHostAdapter *> g_postAdapter{nullptr};
    std::atomic<ps2x::iop::IopSubsystem *> g_asyncSubsystem{nullptr};
}

void ps2IopRpcInstall(ps2x::iop::IopSubsystem *subsystem)
{
    g_asyncSubsystem.store(subsystem, std::memory_order_release);
}

bool ps2IopRpcSubmit(const ps2x::iop::RpcRequest &request, std::function<void(const ps2x::iop::RpcResult &)> eeDone)
{
    ps2x::iop::IopSubsystem *const subsystem = g_asyncSubsystem.load(std::memory_order_acquire);
    PS2IopHostAdapter *const adapter = g_postAdapter.load(std::memory_order_acquire);
    if (!subsystem || !adapter || !eeDone)
        return false;
    if (!subsystem->canOffloadRpc(request.sid))
        return false;
    ps2x::iop::IopSubsystem::AsyncRpc call;
    call.request = request;
    call.send.resize(request.send.size);
    if (request.send.size != 0u && !adapter->readGuest(request.send.address, call.send.data(), call.send.size()))
        return false;
    call.done = [adapter, eeDone = std::move(eeDone)](ps2x::iop::RpcResult result)
    {
        adapter->postWork([eeDone, result]() { eeDone(result); });
    };
    return subsystem->submitRpc(std::move(call));
}

void PS2IopHostAdapter::postWork(std::function<void()> work)
{
    bool wake = false;
    {
        std::lock_guard<std::mutex> lock(m_postMutex);
        PostedSifCommand posted;
        posted.work = std::move(work);
        m_posted.push_back(std::move(posted));
        if (!m_postWakeQueued)
        {
            m_postWakeQueued = true;
            wake = true;
        }
    }
    m_postedTotal.fetch_add(1, std::memory_order_relaxed);
    if (wake)
        m_runtime.postEeEvent(EeEvent{EeEventType::Dmac, kIopPostedWorkEventId, 0u});
}

void PS2IopHostAdapter::setPostSifCommands(bool enable)
{
    m_postSifCommands.store(enable, std::memory_order_release);
    if (enable)
        g_postAdapter.store(this, std::memory_order_release);
}

PS2IopHostAdapter::CallScope::CallScope(PS2IopHostAdapter &owner, R5900Context *context, uint8_t *rdram)
    : m_lock(owner.m_callMutex),
      m_owner(&owner),
      m_previousContext(owner.m_activeContext),
      m_previousRdram(owner.m_activeRdram),
      m_previousToken(owner.m_activeToken)
{
    m_token = owner.m_nextToken++;
    if (m_token == 0)
    {
        m_token = owner.m_nextToken++;
    }
    owner.m_activeContext = context;
    owner.m_activeRdram = rdram;
    owner.m_activeToken = m_token;
}

PS2IopHostAdapter::CallScope::~CallScope()
{
    release();
}

PS2IopHostAdapter::CallScope::CallScope(CallScope &&other) noexcept
    : m_lock(std::move(other.m_lock)),
      m_owner(std::exchange(other.m_owner, nullptr)),
      m_previousContext(other.m_previousContext),
      m_previousRdram(other.m_previousRdram),
      m_previousToken(other.m_previousToken),
      m_token(other.m_token)
{
}

PS2IopHostAdapter::CallScope &PS2IopHostAdapter::CallScope::operator=(CallScope &&other) noexcept
{
    if (this != &other)
    {
        release();
        m_lock = std::move(other.m_lock);
        m_owner = std::exchange(other.m_owner, nullptr);
        m_previousContext = other.m_previousContext;
        m_previousRdram = other.m_previousRdram;
        m_previousToken = other.m_previousToken;
        m_token = other.m_token;
    }
    return *this;
}

void PS2IopHostAdapter::CallScope::release()
{
    if (m_owner)
    {
        m_owner->m_activeContext = m_previousContext;
        m_owner->m_activeRdram = m_previousRdram;
        m_owner->m_activeToken = m_previousToken;
        m_owner = nullptr;
    }
}

PS2IopHostAdapter::PS2IopHostAdapter(PS2Runtime &runtime)
    : m_runtime(runtime)
{
}

PS2IopHostAdapter::~PS2IopHostAdapter()
{
    PS2IopHostAdapter *self = this;
    g_postAdapter.compare_exchange_strong(self, nullptr);
    std::lock_guard<std::mutex> lock(m_hostFileMutex);
    for (auto &[handle, file] : m_hostFiles)
    {
        (void)handle;
        if (file.stream)
        {
            std::fclose(file.stream);
        }
    }
    m_hostFiles.clear();
}

PS2IopHostAdapter::CallScope PS2IopHostAdapter::enterCall(R5900Context *context, uint8_t *rdram)
{
    return CallScope(*this, context, rdram);
}

bool PS2IopHostAdapter::guestRange(uint32_t address, size_t size, uint8_t *&begin) const
{
    begin = nullptr;
    // The IOP thread never runs inside an EE call scope (m_activeRdram belongs to the EE thread).
    uint8_t *const rdram = (m_activeRdram && !ps2x::iop::onIopThread())
                               ? m_activeRdram
                               : m_runtime.memory().getRDRAM();
    if (!rdram)
    {
        return false;
    }
    if (size == 0)
    {
        begin = getMemPtr(rdram, address);
        return begin != nullptr;
    }
    if (size - 1 > std::numeric_limits<uint32_t>::max() - address)
    {
        return false;
    }
    uint8_t *const first = getMemPtr(rdram, address);
    uint8_t *const last = getMemPtr(rdram, address + static_cast<uint32_t>(size - 1));
    if (!first || !last || last < first || static_cast<size_t>(last - first) != size - 1)
    {
        return false;
    }
    begin = first;
    return true;
}

bool PS2IopHostAdapter::readGuest(uint32_t address, void *destination, size_t size) const
{
    if (!destination && size != 0)
    {
        return false;
    }
    uint8_t *source = nullptr;
    if (!guestRange(address, size, source))
    {
        return false;
    }
    if (size != 0)
    {
        std::memcpy(destination, source, size);
    }
    return true;
}

bool PS2IopHostAdapter::writeGuest(uint32_t address, const void *source, size_t size)
{
    if (!source && size != 0)
    {
        return false;
    }
    uint8_t *destination = nullptr;
    if (!guestRange(address, size, destination))
    {
        return false;
    }
    if (size != 0)
    {
        uint8_t *const rdram = m_activeRdram ? m_activeRdram : m_runtime.memory().getRDRAM();
        ps2TraceGuestRangeWrite(rdram, address, static_cast<uint32_t>(size), "IopHost::writeGuest", nullptr);
        std::memcpy(destination, source, size);
    }
    return true;
}

bool PS2IopHostAdapter::zeroGuest(uint32_t address, size_t size)
{
    uint8_t *destination = nullptr;
    if (!guestRange(address, size, destination))
    {
        return false;
    }
    if (size != 0)
    {
        uint8_t *const rdram = m_activeRdram ? m_activeRdram : m_runtime.memory().getRDRAM();
        ps2TraceGuestRangeWrite(rdram, address, static_cast<uint32_t>(size), "IopHost::zeroGuest", nullptr);
        std::memset(destination, 0, size);
    }
    return true;
}

bool PS2IopHostAdapter::normalizeGuestAddress(uint32_t address, uint32_t &normalized) const
{
    bool scratchpad = false;
    if (!ps2ResolveGuestPointer(address, normalized, scratchpad) || scratchpad)
    {
        normalized = 0;
        return false;
    }
    return true;
}

bool PS2IopHostAdapter::readIopMemory(uint32_t address, void *destination, size_t size) const
{
    return m_runtime.readIopMemory(address, destination, size);
}

bool PS2IopHostAdapter::writeIopMemory(uint32_t address, const void *source, size_t size)
{
    return m_runtime.writeIopMemory(address, source, size);
}

bool PS2IopHostAdapter::zeroIopMemory(uint32_t address, size_t size)
{
    return m_runtime.zeroIopMemory(address, size);
}

bool PS2IopHostAdapter::normalizeIopAddress(uint32_t address, uint32_t &normalized) const
{
    if (!m_runtime.isIopMemoryRange(address, 0u))
    {
        normalized = 0u;
        return false;
    }
    normalized = address & 0x1FFFFFFFu;
    return true;
}

uint32_t PS2IopHostAdapter::allocateIopHandle(ps2x::iop::IopHandleKind kind)
{
    uint8_t *const rdram = m_activeRdram
                               ? m_activeRdram
                               : m_runtime.memory().getRDRAM();
    if (!rdram)
    {
        return 0;
    }
    return kind == ps2x::iop::IopHandleKind::RpcPacket
               ? rpcAllocPacketAddr(rdram)
               : rpcAllocServerAddr(rdram);
}

uint32_t PS2IopHostAdapter::allocateGuest(uint32_t size, uint32_t alignment)
{
    return m_runtime.guestMalloc(size, alignment);
}

void PS2IopHostAdapter::freeGuest(uint32_t address)
{
    m_runtime.guestFree(address);
}

void PS2IopHostAdapter::audioCommand(uint32_t sid,
                                     uint32_t function,
                                     ps2x::iop::GuestBuffer send,
                                     ps2x::iop::GuestBuffer receive)
{
    uint8_t *sendPointer = nullptr;
    uint8_t *receivePointer = nullptr;
    if (send.address && !guestRange(send.address, send.size, sendPointer))
    {
        sendPointer = nullptr;
    }
    if (receive.address && !guestRange(receive.address, receive.size, receivePointer))
    {
        receivePointer = nullptr;
    }
    m_runtime.audioBackend().onSoundCommand(sid,
                                            function,
                                            sendPointer,
                                            send.size,
                                            receivePointer,
                                            receive.size);
}

std::string PS2IopHostAdapter::hostPath(ps2x::iop::HostPathKind kind) const
{
    const PS2Runtime::IoPaths &paths = PS2Runtime::getIoPaths();
    switch (kind)
    {
    case ps2x::iop::HostPathKind::CdRoot:
        return paths.cdRoot.string();
    case ps2x::iop::HostPathKind::CdImage:
        return paths.cdImage.string();
    case ps2x::iop::HostPathKind::HostRoot:
        return paths.hostRoot.string();
    case ps2x::iop::HostPathKind::MemoryCardRoot:
        return paths.mcRoot.string();
    case ps2x::iop::HostPathKind::ElfDirectory:
    default:
        return paths.elfDirectory.string();
    }
}

std::string PS2IopHostAdapter::translateGuestPath(std::string_view path) const
{
    const PS2Runtime::IoPaths &paths = PS2Runtime::getIoPaths();
    const PS2VfsMounts mounts{paths.hostRoot, paths.cdRoot, paths.mcRoot};
    std::filesystem::path hostPath;
    if (!m_runtime.vfs().resolveHostPath(path, mounts, hostPath))
        return {};
    return hostPath.string();
}

uint64_t PS2IopHostAdapter::openHostFile(std::string_view path)
{
    if (path.empty())
    {
        return 0u;
    }

    const std::filesystem::path hostPath{std::string(path)};
#if defined(_WIN32)
    std::FILE *stream = ::_wfopen(hostPath.c_str(), L"rb");
#else
    std::FILE *stream = std::fopen(hostPath.string().c_str(), "rb");
#endif
    if (!stream)
    {
        return 0u;
    }

    std::error_code error;
    const uint64_t size = std::filesystem::file_size(hostPath, error);
    if (error)
    {
        std::fclose(stream);
        return 0u;
    }

    std::lock_guard<std::mutex> lock(m_hostFileMutex);
    uint64_t handle = m_nextHostFileHandle++;
    if (handle == 0u)
    {
        handle = m_nextHostFileHandle++;
    }
    m_hostFiles.emplace(handle, HostFile{stream, size});
    return handle;
}

bool PS2IopHostAdapter::hostFileSize(uint64_t handle, uint64_t &size) const
{
    size = 0u;
    std::lock_guard<std::mutex> lock(m_hostFileMutex);
    const auto it = m_hostFiles.find(handle);
    if (it == m_hostFiles.end() || !it->second.stream)
    {
        return false;
    }
    size = it->second.size;
    return true;
}

bool PS2IopHostAdapter::readHostFile(uint64_t handle,
                                     uint64_t offset,
                                     void *destination,
                                     size_t size,
                                     size_t &bytesRead)
{
    bytesRead = 0u;
    if (!destination && size != 0u)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_hostFileMutex);
    const auto it = m_hostFiles.find(handle);
    if (it == m_hostFiles.end() || !it->second.stream)
    {
        return false;
    }

#if defined(_WIN32)
    if (offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        _fseeki64(it->second.stream, static_cast<int64_t>(offset), SEEK_SET) != 0)
#else
    if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        fseeko(it->second.stream, static_cast<off_t>(offset), SEEK_SET) != 0)
#endif
    {
        return false;
    }

    bytesRead = size == 0u
                    ? 0u
                    : std::fread(destination, 1u, size, it->second.stream);
    if (bytesRead < size && std::ferror(it->second.stream))
    {
        std::clearerr(it->second.stream);
        return false;
    }
    return true;
}

void PS2IopHostAdapter::closeHostFile(uint64_t handle)
{
    std::FILE *stream = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_hostFileMutex);
        const auto it = m_hostFiles.find(handle);
        if (it == m_hostFiles.end())
        {
            return;
        }
        stream = it->second.stream;
        m_hostFiles.erase(it);
    }
    if (stream)
    {
        std::fclose(stream);
    }
}

int32_t PS2IopHostAdapter::memoryCard(const ps2x::iop::MemoryCardRequest &request)
{
    using Handler = void (*)(uint8_t *, R5900Context *, PS2Runtime *);
    Handler handler = nullptr;
    switch (request.operation)
    {
    case ps2x::iop::MemoryCardOperation::Init:
        handler = ps2_stubs::sceMcInit;
        break;
    case ps2x::iop::MemoryCardOperation::GetInfo:
        handler = ps2_stubs::sceMcGetInfo;
        break;
    case ps2x::iop::MemoryCardOperation::Open:
        handler = ps2_stubs::sceMcOpen;
        break;
    case ps2x::iop::MemoryCardOperation::Close:
        handler = ps2_stubs::sceMcClose;
        break;
    case ps2x::iop::MemoryCardOperation::Seek:
        handler = ps2_stubs::sceMcSeek;
        break;
    case ps2x::iop::MemoryCardOperation::Read:
        handler = ps2_stubs::sceMcRead;
        break;
    case ps2x::iop::MemoryCardOperation::Write:
        handler = ps2_stubs::sceMcWrite;
        break;
    case ps2x::iop::MemoryCardOperation::Flush:
        handler = ps2_stubs::sceMcFlush;
        break;
    case ps2x::iop::MemoryCardOperation::Chdir:
        handler = ps2_stubs::sceMcChdir;
        break;
    case ps2x::iop::MemoryCardOperation::GetDir:
        handler = ps2_stubs::sceMcGetDir;
        break;
    case ps2x::iop::MemoryCardOperation::SetFileInfo:
        handler = ps2_stubs::sceMcSetFileInfo;
        break;
    case ps2x::iop::MemoryCardOperation::Delete:
        handler = ps2_stubs::sceMcDelete;
        break;
    case ps2x::iop::MemoryCardOperation::Format:
        handler = ps2_stubs::sceMcFormat;
        break;
    case ps2x::iop::MemoryCardOperation::Unformat:
        handler = ps2_stubs::sceMcUnformat;
        break;
    case ps2x::iop::MemoryCardOperation::Mkdir:
        handler = ps2_stubs::sceMcMkdir;
        break;
    }
    if (!handler)
    {
        return -1;
    }

    R5900Context context{};
    for (size_t i = 0; i < 4u; ++i)
    {
        setRegU32(&context, static_cast<int>(4 + i), request.arguments[i]);
    }

    // EE n32 ABI: the fifth argument travels in $t0, matching the sceMc* stubs.
    setRegU32(&context, 8, request.arguments[4]);

    handler(m_activeRdram ? m_activeRdram : m_runtime.memory().getRDRAM(),
            &context,
            &m_runtime);
    return ps2_stubs::getMemoryCardDebugSnapshot().lastResult;
}

bool PS2IopHostAdapter::hasGuestFunction(uint32_t address) const
{
    return m_runtime.hasFunction(address);
}

bool PS2IopHostAdapter::invokeGuestFunction(uint64_t callToken,
                                            uint32_t address,
                                            uint32_t a0,
                                            uint32_t a1,
                                            uint32_t a2,
                                            uint32_t a3,
                                            uint32_t *resultAddress)
{
    (void)callToken;
    (void)address;
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    if (resultAddress)
    {
        *resultAddress = 0u;
    }
    return false;
}

bool PS2IopHostAdapter::sendSifCommand(uint32_t commandId,
                                       const void *packet,
                                       size_t packetSize)
{
    if (m_postSifCommands.load(std::memory_order_acquire))
    {
        // Threaded IOP: queue the packet (in order) for the EE thread; dispatchSifCommand allocates guest memory and
        // queues an EE invocation, which only the EE thread may do. Returns true like a delivered command (the IOP
        // ignores the result: a command without an EE handler is still a completed DMA).
        if (!packet || packetSize < 16u || packetSize > 112u)
            return false;
        bool wake = false;
        {
            std::lock_guard<std::mutex> lock(m_postMutex);
            PostedSifCommand posted;
            posted.commandId = commandId;
            posted.packet.assign(static_cast<const uint8_t *>(packet), static_cast<const uint8_t *>(packet) + packetSize);
            m_posted.push_back(std::move(posted));
            if (!m_postWakeQueued)
            {
                m_postWakeQueued = true;
                wake = true;
            }
        }
        m_postedTotal.fetch_add(1, std::memory_order_relaxed);
        if (wake)
            m_runtime.postEeEvent(EeEvent{EeEventType::Dmac, kIopPostedWorkEventId, 0u});
        return true;
    }
    uint8_t *const rdram = m_activeRdram
                               ? m_activeRdram
                               : m_runtime.memory().getRDRAM();
    return ps2_stubs::dispatchSifCommand(rdram,
                                         &m_runtime,
                                         commandId,
                                         packet,
                                         packetSize);
}

void PS2IopHostAdapter::drainPosted()
{
    // EE thread. Commands posted while this runs get their own wake-up event.
    std::vector<PostedSifCommand> batch;
    {
        std::lock_guard<std::mutex> lock(m_postMutex);
        batch.swap(m_posted);
        m_postWakeQueued = false;
    }
    if (batch.empty())
        return;
    static const bool stats = []() {
        const char *v = std::getenv("PS2X_IOP_THREAD_STATS");
        return v && *v && v[0] != '0';
    }();
    if (stats)
    {
        static uint64_t drains = 0, commands = 0, maxBatch = 0;
        static auto last = std::chrono::steady_clock::now();
        ++drains;
        commands += batch.size();
        maxBatch = std::max<uint64_t>(maxBatch, batch.size());
        const auto now = std::chrono::steady_clock::now();
        if (now - last >= std::chrono::seconds(5))
        {
            last = now;
            std::fprintf(stderr, "[iop-thread] EE drains %llu, SIF commands %llu, max batch %llu\n",
                         static_cast<unsigned long long>(drains), static_cast<unsigned long long>(commands),
                         static_cast<unsigned long long>(maxBatch));
        }
    }
    uint8_t *const rdram = m_runtime.memory().getRDRAM();
    for (PostedSifCommand &command : batch)
    {
        if (command.work)
        {
            command.work();
            continue;
        }
        (void)ps2_stubs::dispatchSifCommand(rdram,
                                            &m_runtime,
                                            command.commandId,
                                            command.packet.data(),
                                            command.packet.size());
    }
}

void ps2IopDrainPosted(PS2Runtime &runtime)
{
    (void)runtime;
    if (PS2IopHostAdapter *adapter = g_postAdapter.load(std::memory_order_acquire))
        adapter->drainPosted();
}

void PS2IopHostAdapter::log(ps2x::iop::LogLevel level, std::string_view message)
{
    const char *prefix = "[ps2xIOP]";
    if (level == ps2x::iop::LogLevel::Warning)
    {
        prefix = "[ps2xIOP:warning]";
    }
    else if (level == ps2x::iop::LogLevel::Error)
    {
        prefix = "[ps2xIOP:error]";
    }
    std::cerr << prefix << ' ' << message << std::endl;
}
