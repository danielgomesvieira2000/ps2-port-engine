/*
 * The IOP's network libraries - Sony's inet, netcnf and inetctl - on the
 * host's sockets, for modules that run in the interpreter (iop_vm.h) on top
 * of them: EA's DirtySock module (DRTYSCKF.IRX) first.
 *
 * inet's model is not BSD's. A connection is created from a parameter block
 * (sceInetCreate) and then opened (sceInetOpen): a datagram endpoint is
 * bound, an active one connects, a passive one waits for a peer and then *is*
 * that peer's connection. Calls take a timeout in ms: -1 waits for good, 0
 * does not wait. A call that has to wait blocks its IOP thread and makes the
 * same call again a millisecond later (Outcome::Retry), so the EE never
 * waits on the network.
 *
 * What is known comes from DirtySock's calls (the LAN project's notes,
 * docs/reseau/drtysckf.md); what it never calls is answered with an error
 * and logged once.
 *
 * Names: PS2HOST_INET_HOST=<address> answers every lookup of a name that is
 * not an address with that address (a game's servers are long gone; on a LAN
 * a stand-in serves them). Without it, names go to the host's resolver.
 *
 * Also here: the few general exports such a module needs that no other
 * module has asked for (modload's SearchModuleByName, cdvdman's power-off
 * callback).
 */

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using Socket = SOCKET;
constexpr Socket kNoSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
using Socket = int;
constexpr Socket kNoSocket = -1;
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "iop_vm_internal.h"

namespace ps2host {
namespace iop {

namespace {

// sceInetParam.type
constexpr s32 kDgram = 0, kConnect = 1, kListen = 2, kRaw = 3;
// Results. -501 ends DirtySock's receive thread after an abort (0x30A4 in
// the module); the others only need to be negative.
constexpr s32 kError = -1, kAborted = -501, kTimeout = -502, kNoConnection = -503;
// sceInetRecv/Send flags
constexpr u32 kFin = 4;
// sceInetControl(1): the connection's information, 0x4C bytes; its state
// word: DirtySock takes 7 to 13 as connected.
constexpr u32 kGetInfo = 1, kInfoSize = 0x4C;
constexpr u32 kStateNone = 0, kStateOpening = 3, kStateEstablished = 7, kStatePeerClosed = 8;
constexpr u64 kPollUs = 1000;

void Log(const std::string& text) { std::fprintf(stderr, "[inet] %s\n", text.c_str()); }

// PS2HOST_INET_LOG=1: every transfer, its size and its first bytes as text.
void Traffic(const char* what, s32 cid, const char* data, int size) {
    static const bool on = std::getenv("PS2HOST_INET_LOG") != nullptr;
    if (!on) return;
    std::string text = std::string(what) + " cid " + std::to_string(cid) + ": " + std::to_string(size) + " bytes";
    if (data && size > 0) {
        text += " |";
        for (int i = 0; i < size && i < 64; ++i) {
            const unsigned char ch = static_cast<unsigned char>(data[i]);
            text += ch >= 32 && ch < 127 ? char(ch) : '.';
        }
        text += "|";
    }
    Log(text);
}

bool Started() {
    static const bool ok = [] {
#ifdef _WIN32
        WSADATA data;
        return WSAStartup(MAKEWORD(2, 2), &data) == 0;
#else
        return true;
#endif
    }();
    return ok;
}

bool WouldBlock() {
#ifdef _WIN32
    const int e = WSAGetLastError();
    return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS || e == WSAEALREADY;
#else
    return errno == EWOULDBLOCK || errno == EAGAIN || errno == EINPROGRESS || errno == EALREADY;
#endif
}

void CloseSocket(Socket s) {
    if (s == kNoSocket) return;
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

void NonBlocking(Socket s) {
#ifdef _WIN32
    u_long on = 1;
    ioctlsocket(s, FIONBIO, &on);
#else
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK);
#endif
}

void ReuseAddress(Socket s) {
    int on = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&on), sizeof on);
}

// The IPv4 in DirtySock's sceInet structures is a plain number in the IOP's
// own order (127.0.0.1 is 0x7F000001), as the ports are: an address the lobby
// gives as text reached the host reversed (1.0.0.127) when it was taken for
// network order. Host <-> guest is the same swap both ways.
u32 GuestIp(u32 ip_network) { return ntohl(ip_network); }
u32 HostIp(u32 guest) { return htonl(guest); }

// PS2HOST_INET_BIND=<address>: this instance's own address, which its
// connections leave from (two instances on one PC: 127.0.0.1, 127.0.0.2).
u32 BindAddress() {
    static const u32 ip = [] {
        in_addr a{};
        const char* text = std::getenv("PS2HOST_INET_BIND");
        return text && inet_pton(AF_INET, text, &a) == 1 ? u32(a.s_addr) : u32(INADDR_ANY);
    }();
    return ip;
}

sockaddr_in Address(u32 ip_network, u16 port) {
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = ip_network;
    a.sin_port = htons(port);
    return a;
}

std::string Text(u32 ip_network, u16 port) {
    const u8* b = reinterpret_cast<const u8*>(&ip_network);
    char text[32];
    std::snprintf(text, sizeof text, "%u.%u.%u.%u:%u", b[0], b[1], b[2], b[3], unsigned(port));
    return text;
}

// A listening socket, shared by the passive connections waiting on its port.
struct Listener {
    Socket s = kNoSocket;
    int users = 0;
};

struct Connection {
    s32 type = kDgram;
    s32 local_port = -1;
    u32 remote_ip = 0;              // network order
    u16 remote_port = 0;
    Socket s = kNoSocket;
    u32 state = kStateNone;
    bool opened = false, failed = false, aborted = false, peer_closed = false;
    bool listening = false;          // a passive one, holding its port's listener
};

struct Inet {
    std::map<s32, Connection> connections;
    std::map<u16, Listener> listeners;
    std::map<u32, u64> deadlines;    // thread -> when its waiting call times out
    s32 next = 1;
};

Inet& N() {
    static Inet n;
    return n;
}

Connection* Find(s32 cid) {
    auto found = N().connections.find(cid);
    return found == N().connections.end() ? nullptr : &found->second;
}

void ReleaseListener(Connection& k) {
    if (!k.listening) return;
    k.listening = false;
    auto found = N().listeners.find(u16(k.local_port));
    if (found != N().listeners.end() && --found->second.users <= 0) {
        CloseSocket(found->second.s);
        N().listeners.erase(found);
    }
}

// Moves a connection on: an active one's connect completing, a passive
// one's peer arriving.
void Progress(Connection& k) {
    if (!k.opened || k.failed || k.state == kStateEstablished || k.state == kStatePeerClosed) return;
    if (k.type == kConnect && k.s != kNoSocket) {
        fd_set write_set, error_set;
        FD_ZERO(&write_set);
        FD_ZERO(&error_set);
        FD_SET(k.s, &write_set);
        FD_SET(k.s, &error_set);
        timeval none{0, 0};
        if (select(int(k.s + 1), nullptr, &write_set, &error_set, &none) > 0) {
            int error = 0;
            socklen_t size = sizeof error;
            getsockopt(k.s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&error), &size);
            if (FD_ISSET(k.s, &error_set) || error) {
                k.failed = true;
                k.state = kStateNone;
                Log("connect to " + Text(k.remote_ip, k.remote_port) + " failed");
            } else {
                k.state = kStateEstablished;
                Log("connected to " + Text(k.remote_ip, k.remote_port));
            }
        }
    } else if (k.type == kListen && k.listening) {
        Listener& l = N().listeners[u16(k.local_port)];
        sockaddr_in peer{};
        socklen_t size = sizeof peer;
        const Socket s = accept(l.s, reinterpret_cast<sockaddr*>(&peer), &size);
        if (s != kNoSocket) {
            NonBlocking(s);
            k.s = s;
            k.remote_ip = peer.sin_addr.s_addr;
            k.remote_port = ntohs(peer.sin_port);
            k.state = kStateEstablished;
            ReleaseListener(k);
            Log("accepted " + Text(k.remote_ip, k.remote_port) + " on port " + std::to_string(k.local_port));
        }
    }
}

// The calling thread waits a millisecond and makes the call again, unless
// `ms` (0: never; -1: for good) has run out since it first waited.
Outcome WaitOrGiveUp(Vm& vm, Context& c, s32 ms, s32 give_up) {
    Thread* t = vm.Current();
    const u64 now = vm.NowUs();
    if (ms == 0 || !t) {
        N().deadlines.erase(vm.current);
        Return(c, u32(give_up));
        return Outcome::Return;
    }
    auto [at, fresh] = N().deadlines.try_emplace(t->id, ms < 0 ? ~0ull : now + u64(ms) * 1000);
    (void)fresh;
    if (now >= at->second) {
        N().deadlines.erase(at);
        Return(c, u32(kTimeout));
        return Outcome::Return;
    }
    t->until_us = now + kPollUs;
    vm.Block(c, Wait::Delay, 0);
    return Outcome::Retry;
}

Outcome Done(Vm& vm, Context& c, s32 value) {
    N().deadlines.erase(vm.current);
    Return(c, u32(value));
    return Outcome::Return;
}

// ------------------------------------------------------------------ inet

// sceInetCreate(param): type +0, local port +4 (-1: any), remote address
// +8 (a sceInetAddress: its IPv4 at +0xC, see GuestIp), remote port +0x18.
Outcome Create(Vm& vm, Context& c) {
    if (!Started()) return Done(vm, c, kError);
    const u32 p = Arg(vm, c, 0);
    Connection k;
    k.type = s32(vm.Read(p, 4));
    k.local_port = s32(vm.Read(p + 4, 4));
    k.remote_ip = HostIp(vm.Read(p + 0xC, 4));
    k.remote_port = u16(vm.Read(p + 0x18, 4));
    if (k.type < kDgram || k.type > kRaw) return Done(vm, c, kError);
    const s32 cid = N().next++;
    N().connections[cid] = k;
    return Done(vm, c, cid);
}

// sceInetOpen(cid, ms)
Outcome Open(Vm& vm, Context& c) {
    const s32 cid = s32(Arg(vm, c, 0)), ms = s32(Arg(vm, c, 1));
    Connection* k = Find(cid);
    if (!k) return Done(vm, c, kNoConnection);
    if (!k->opened) {
        k->opened = true;
        if (k->type == kRaw) {
            // Raw IP (DirtySock's pings, ICMP) needs privileges a game should
            // not ask for: the connection opens, what is sent goes nowhere,
            // nothing ever arrives - pings go unanswered, nothing fails.
            Log("a raw connection: sends are dropped, nothing is received");
            k->state = kStateEstablished;
            return Done(vm, c, 0);
        }
        if (k->type == kDgram) {
            k->s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
            if (k->s == kNoSocket) return Done(vm, c, kError);
            int on = 1;
            setsockopt(k->s, SOL_SOCKET, SO_BROADCAST, reinterpret_cast<const char*>(&on), sizeof on);
            ReuseAddress(k->s);
            const sockaddr_in a = Address(INADDR_ANY, k->local_port > 0 ? u16(k->local_port) : 0);
            if (bind(k->s, reinterpret_cast<const sockaddr*>(&a), sizeof a) != 0) {
                Log("udp bind to port " + std::to_string(k->local_port) + " failed");
                k->failed = true;
                return Done(vm, c, kError);
            }
            NonBlocking(k->s);
            k->state = kStateEstablished;
            sockaddr_in bound{};
            socklen_t bound_size = sizeof bound;
            getsockname(k->s, reinterpret_cast<sockaddr*>(&bound), &bound_size);
            Log("udp cid " + std::to_string(cid) + " on port " + std::to_string(ntohs(bound.sin_port)));
            return Done(vm, c, 0);
        }
        if (k->type == kConnect) {
            k->s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (k->s == kNoSocket) return Done(vm, c, kError);
            NonBlocking(k->s);
            if (k->local_port > 0 || BindAddress() != INADDR_ANY) {
                ReuseAddress(k->s);
                const sockaddr_in local = Address(BindAddress(), k->local_port > 0 ? u16(k->local_port) : 0);
                bind(k->s, reinterpret_cast<const sockaddr*>(&local), sizeof local);
            }
            const sockaddr_in a = Address(k->remote_ip, k->remote_port);
            Log("connecting to " + Text(k->remote_ip, k->remote_port));
            if (connect(k->s, reinterpret_cast<const sockaddr*>(&a), sizeof a) == 0) {
                k->state = kStateEstablished;
            } else if (WouldBlock()) {
                k->state = kStateOpening;
            } else {
                k->failed = true;
                return Done(vm, c, kError);
            }
        } else {                                                // kListen
            const u16 port = u16(k->local_port > 0 ? k->local_port : 0);
            Listener& l = N().listeners[port];
            if (l.s == kNoSocket) {
                l.s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                ReuseAddress(l.s);
                const sockaddr_in a = Address(INADDR_ANY, port);
                if (l.s == kNoSocket || bind(l.s, reinterpret_cast<const sockaddr*>(&a), sizeof a) != 0 ||
                    listen(l.s, 8) != 0) {
                    Log("listen on port " + std::to_string(port) + " failed");
                    CloseSocket(l.s);
                    N().listeners.erase(port);
                    k->failed = true;
                    return Done(vm, c, kError);
                }
                NonBlocking(l.s);
                Log("listening on port " + std::to_string(port));
            }
            ++l.users;
            k->listening = true;
            k->state = kStateOpening;
        }
    }
    Progress(*k);
    if (k->failed) return Done(vm, c, kError);
    if (k->aborted) return Done(vm, c, kAborted);
    if (k->state == kStateEstablished) return Done(vm, c, 0);
    if (ms == 0) return Done(vm, c, 0);                   // opening goes on; Control tells
    return WaitOrGiveUp(vm, c, ms, 0);
}

// sceInetClose(cid, ms)
Outcome Close(Vm& vm, Context& c) {
    const s32 cid = s32(Arg(vm, c, 0));
    Connection* k = Find(cid);
    if (!k) return Done(vm, c, kNoConnection);
    ReleaseListener(*k);
    CloseSocket(k->s);
    N().connections.erase(cid);
    return Done(vm, c, 0);
}

// sceInetAbort(cid, flags): calls waiting on it end with kAborted.
Outcome Abort(Vm& vm, Context& c) {
    Connection* k = Find(s32(Arg(vm, c, 0)));
    if (!k) return Done(vm, c, kNoConnection);
    k->aborted = true;
    return Done(vm, c, 0);
}

// sceInetRecv(cid, buffer, length, &flags, ms)
Outcome Recv(Vm& vm, Context& c) {
    const s32 cid = s32(Arg(vm, c, 0));
    const u32 buffer = Arg(vm, c, 1), length = Arg(vm, c, 2), flags = Arg(vm, c, 3);
    const s32 ms = s32(Arg(vm, c, 4));
    Connection* k = Find(cid);
    if (!k) return Done(vm, c, kNoConnection);
    if (k->aborted) return Done(vm, c, kAborted);
    Progress(*k);
    if (k->failed) return Done(vm, c, kError);
    if (flags) vm.Write(flags, 0, 4);
    if (k->peer_closed) {
        if (flags) vm.Write(flags, kFin, 4);
        return Done(vm, c, 0);
    }
    if (k->state == kStateEstablished && k->s != kNoSocket) {
        std::vector<char> data(length ? length : 1);
        const int got = recv(k->s, data.data(), int(length), 0);
        if (got > 0) {
            Traffic("recv", cid, data.data(), got);
            std::memcpy(vm.At(buffer, u32(got)), data.data(), u32(got));
            return Done(vm, c, got);
        }
        if (got == 0 && length) {
            Traffic("recv end", cid, nullptr, 0);
            k->peer_closed = true;
            k->state = kStatePeerClosed;
            if (flags) vm.Write(flags, kFin, 4);
            return Done(vm, c, 0);
        }
        if (!WouldBlock()) {
            k->failed = true;
            return Done(vm, c, kError);
        }
    }
    return WaitOrGiveUp(vm, c, ms, 0);
}

// sceInetSend(cid, buffer, length, &flags, ms): flags' FIN shuts the
// sending side once the data is sent.
Outcome Send(Vm& vm, Context& c) {
    const s32 cid = s32(Arg(vm, c, 0));
    const u32 buffer = Arg(vm, c, 1), length = Arg(vm, c, 2), flags = Arg(vm, c, 3);
    const s32 ms = s32(Arg(vm, c, 4));
    Connection* k = Find(cid);
    if (!k) return Done(vm, c, kNoConnection);
    if (k->aborted) return Done(vm, c, kAborted);
    Progress(*k);
    if (k->failed) return Done(vm, c, kError);
    if (k->type == kRaw) return Done(vm, c, s32(length));         // dropped
    const bool fin = flags && (vm.Read(flags, 4) & kFin);
    if (k->state != kStateEstablished && k->state != kStatePeerClosed)
        return WaitOrGiveUp(vm, c, ms, 0);
    int sent = 0;
    if (length) {
        sent = send(k->s, reinterpret_cast<const char*>(vm.At(buffer, length)), int(length), 0);
        Traffic("send", cid, reinterpret_cast<const char*>(vm.At(buffer, length)), sent);
        if (sent < 0) {
            if (WouldBlock()) return WaitOrGiveUp(vm, c, ms, 0);
            k->failed = true;
            return Done(vm, c, kError);
        }
    }
    if (fin && u32(sent) == length) {
#ifdef _WIN32
        shutdown(k->s, SD_SEND);
#else
        shutdown(k->s, SHUT_WR);
#endif
    }
    return Done(vm, c, sent);
}

// sceInetRecvFrom(cid, buffer, length, &flags, &address, &port, ms): the
// sender's address as a sceInetAddress (its IPv4 at +4), its port in host order.
Outcome RecvFrom(Vm& vm, Context& c) {
    const s32 cid = s32(Arg(vm, c, 0));
    const u32 buffer = Arg(vm, c, 1), length = Arg(vm, c, 2), flags = Arg(vm, c, 3);
    const u32 address = Arg(vm, c, 4), port = Arg(vm, c, 5);
    const s32 ms = s32(Arg(vm, c, 6));
    Connection* k = Find(cid);
    if (!k) return Done(vm, c, kNoConnection);
    if (k->aborted) return Done(vm, c, kAborted);
    if (k->type == kRaw) return WaitOrGiveUp(vm, c, ms, 0);        // nothing ever arrives
    if (k->failed || k->s == kNoSocket) return Done(vm, c, kError);
    if (flags) vm.Write(flags, 0, 4);
    std::vector<char> data(length ? length : 1);
    sockaddr_in from{};
    socklen_t size = sizeof from;
    const int got = recvfrom(k->s, data.data(), int(length), 0, reinterpret_cast<sockaddr*>(&from), &size);
    if (got >= 0) {
        std::memcpy(vm.At(buffer, u32(got)), data.data(), u32(got));
        if (address) {
            vm.Write(address, 0, 4);
            vm.Write(address + 4, GuestIp(from.sin_addr.s_addr), 4);
            vm.Write(address + 8, 0, 4);
            vm.Write(address + 12, 0, 4);
        }
        if (port) vm.Write(port, ntohs(from.sin_port), 4);
        Traffic(("recvfrom " + Text(from.sin_addr.s_addr, ntohs(from.sin_port))).c_str(), cid, data.data(), got);
        return Done(vm, c, got);
    }
    if (!WouldBlock()) {
#ifdef _WIN32
        if (WSAGetLastError() == WSAECONNRESET) return WaitOrGiveUp(vm, c, ms, 0);  // an ICMP from a closed port
#endif
        return Done(vm, c, kError);
    }
    return WaitOrGiveUp(vm, c, ms, 0);
}

// sceInetSendTo(cid, buffer, length, &flags, &address, port, ms)
Outcome SendTo(Vm& vm, Context& c) {
    const s32 cid = s32(Arg(vm, c, 0));
    const u32 buffer = Arg(vm, c, 1), length = Arg(vm, c, 2);
    const u32 address = Arg(vm, c, 4), port = Arg(vm, c, 5);
    const s32 ms = s32(Arg(vm, c, 6));
    Connection* k = Find(cid);
    if (!k) return Done(vm, c, kNoConnection);
    if (k->aborted) return Done(vm, c, kAborted);
    if (k->type == kRaw) return Done(vm, c, s32(length));         // dropped
    if (k->failed || k->s == kNoSocket) return Done(vm, c, kError);
    const sockaddr_in to = Address(HostIp(vm.Read(address + 4, 4)), u16(port));
    const int sent = sendto(k->s, reinterpret_cast<const char*>(vm.At(buffer, length)), int(length), 0,
                            reinterpret_cast<const sockaddr*>(&to), sizeof to);
    if (sent >= 0) {
        Traffic(("sendto " + Text(to.sin_addr.s_addr, u16(port))).c_str(), cid,
                reinterpret_cast<const char*>(vm.At(buffer, length)), sent);
        return Done(vm, c, sent);
    }
    if (WouldBlock()) return WaitOrGiveUp(vm, c, ms, 0);
    Log("sendto " + Text(to.sin_addr.s_addr, u16(port)) + " failed");
    return Done(vm, c, kError);
}

// sceInetControl(cid, code, buffer, length). Code 1: the connection's
// information (0x4C bytes): local address +0x10 (IPv4 at +0x14), local port
// +0x20, remote address +0x24 (IPv4 at +0x28), remote port +0x34, state +0x38.
Outcome Control(Vm& vm, Context& c) {
    const s32 cid = s32(Arg(vm, c, 0));
    const u32 code = Arg(vm, c, 1), buffer = Arg(vm, c, 2), length = Arg(vm, c, 3);
    Connection* k = Find(cid);
    if (!k) return Done(vm, c, kNoConnection);
    if (code != kGetInfo || length < kInfoSize) {
        static std::set<u32> seen;
        if (seen.insert(code).second)
            Log("sceInetControl code " + std::to_string(code) + " (length " + std::to_string(length) +
                ") is not known");
        return Done(vm, c, kError);
    }
    Progress(*k);
    for (u32 i = 0; i < kInfoSize; i += 4) vm.Write(buffer + i, 0, 4);
    u32 local_ip = 0;
    u16 local_port = k->local_port > 0 ? u16(k->local_port) : 0;
    if (k->s != kNoSocket) {
        sockaddr_in a{};
        socklen_t size = sizeof a;
        if (getsockname(k->s, reinterpret_cast<sockaddr*>(&a), &size) == 0) {
            local_ip = a.sin_addr.s_addr;
            local_port = ntohs(a.sin_port);
        }
    }
    vm.Write(buffer + 0x14, GuestIp(local_ip), 4);
    vm.Write(buffer + 0x20, local_port, 4);
    vm.Write(buffer + 0x28, GuestIp(k->remote_ip), 4);
    vm.Write(buffer + 0x34, k->remote_port, 4);
    vm.Write(buffer + 0x38, k->failed ? kStateNone : k->state, 4);
    return Done(vm, c, 0);
}

// Read at each lookup, not once: a port may choose the host at run time
// (Burnout 3's LAN: the lobby found on the LAN, or its own).
u32 LanHost() {
    const char* text = std::getenv("PS2HOST_INET_HOST");
    in_addr a{};
    if (text && inet_pton(AF_INET, text, &a) == 1) return u32(a.s_addr);
    return 0u;
}

// sceInetName2Address(flags, &address, name, ms, retries): the IPv4 at
// address +4.
Outcome Name2Address(Vm& vm, Context& c) {
    const u32 address = Arg(vm, c, 1);
    const std::string name = vm.String(Arg(vm, c, 2), 256);
    if (!Started()) return Done(vm, c, kError);
    u32 ip = 0;
    in_addr literal{};
    if (inet_pton(AF_INET, name.c_str(), &literal) == 1) {
        ip = literal.s_addr;
    } else if (LanHost()) {
        ip = LanHost();
    } else {
        addrinfo hints{}, *found = nullptr;
        hints.ai_family = AF_INET;
        if (getaddrinfo(name.c_str(), nullptr, &hints, &found) == 0 && found) {
            ip = reinterpret_cast<sockaddr_in*>(found->ai_addr)->sin_addr.s_addr;
            freeaddrinfo(found);
        }
    }
    Log("name " + name + " -> " + (ip ? Text(ip, 0) : std::string("not found")));
    if (!ip) return Done(vm, c, kError);
    for (u32 i = 0; i < 16; i += 4) vm.Write(address + i, 0, 4);
    vm.Write(address + 4, GuestIp(ip), 4);
    return Done(vm, c, 0);
}

// This machine's IPv4 on the LAN: the one a UDP socket "connected" outward
// would send from (nothing is sent).
u32 LocalAddress() {
    if (!Started()) return 0;
    const Socket s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == kNoSocket) return 0;
    const sockaddr_in out = Address(LanHost() ? LanHost() : htonl(0x0A000001u), 9);
    sockaddr_in a{};
    socklen_t size = sizeof a;
    u32 ip = 0;
    if (connect(s, reinterpret_cast<const sockaddr*>(&out), sizeof out) == 0 &&
        getsockname(s, reinterpret_cast<sockaddr*>(&a), &size) == 0)
        ip = a.sin_addr.s_addr;
    CloseSocket(s);
    return ip ? ip : htonl(0x7F000001u);
}

// sceInetGetInterfaceList(ids, max): one interface, id 1.
Outcome GetInterfaceList(Vm& vm, Context& c) {
    const u32 ids = Arg(vm, c, 0), max = Arg(vm, c, 1);
    if (!max) return Done(vm, c, 0);
    vm.Write(ids, 1, 4);
    return Done(vm, c, 1);
}

// sceInetInterfaceControl(id, code, buffer, length). Code 8: the interface's
// flags, a word - DirtySock's network start with "$USE-EXISTING$" (0x1258 in
// the module) takes the first with bit 1 set as the one that is up. Code 9:
// the interface's address, a sceInetAddress (IPv4 at +4).
Outcome InterfaceControl(Vm& vm, Context& c) {
    const u32 code = Arg(vm, c, 1), buffer = Arg(vm, c, 2), length = Arg(vm, c, 3);
    if (code == 8 && length >= 4) {
        vm.Write(buffer, 0x3, 4);
        return Done(vm, c, 0);
    }
    // The network device's own controls (DirtySock's 'plug', 0x18A8 in its
    // module): 0x80000100 must answer 1 for the next to be asked, and
    // 0x80030000's answer is the link's state - here always up (1).
    if (code == 0x80000100u || code == 0x80030000u) return Done(vm, c, 1);
    if (code == 9 && length >= 8) {
        for (u32 i = 0; i < length && i < 16; i += 4) vm.Write(buffer + i, 0, 4);
        vm.Write(buffer + 4, GuestIp(LocalAddress()), 4);
        return Done(vm, c, 0);
    }
    static std::set<u32> seen;
    if (seen.insert(code).second)
        Log("sceInetInterfaceControl code " + std::to_string(code) + " (length " + std::to_string(length) +
            ") is not known");
    return Done(vm, c, kError);
}

// Logs a call once per (library, export, caller) and answers `value`.
Library Logged(const char* library, u32 index, const char* name, u32 value) {
    return [library, index, name, value](Vm& vm, Context& c) -> Outcome {
        static std::set<std::tuple<std::string, u32, u32>> seen;
        const u32 caller = c.r[31] - 8;
        if (seen.insert({library, index, caller}).second)
            std::fprintf(stderr, "[iop] %s#%u %s(%08X, %08X, %08X, %08X) from %08X -> %08X\n",
                         library, unsigned(index), name, c.r[4], c.r[5], c.r[6], c.r[7],
                         caller, value);
        (void)vm;
        Return(c, value);
        return Outcome::Return;
    };
}

}  // namespace

const LibraryTable& NetworkLibraries() {
    static const LibraryTable table = {
        // inet, by Sony's order: read from DirtySock's calls
        {{"inet", 4}, Name2Address},
        {{"inet", 6}, Create},
        {{"inet", 7}, Open},
        {{"inet", 8}, Close},
        {{"inet", 9}, Recv},
        {{"inet", 10}, Send},
        {{"inet", 11}, Abort},
        {{"inet", 12}, RecvFrom},
        {{"inet", 13}, SendTo},
        {{"inet", 15}, Control},
        {{"inet", 24}, GetInterfaceList},
        {{"inet", 25}, InterfaceControl},
        {{"inet", 30}, Logged("inet", 30, "inet#30", u32(kError))},
        // netcnf, inetctl: what DRTYSCKF imports; meanings still to be traced
        {{"netcnf", 4}, Logged("netcnf", 4, "netcnf#4", u32(kError))},
        {{"netcnf", 5}, Logged("netcnf", 5, "netcnf#5", u32(kError))},
        {{"netcnf", 6}, Logged("netcnf", 6, "netcnf#6", u32(kError))},
        {{"inetctl", 4}, Logged("inetctl", 4, "inetctl#4", u32(kError))},
        {{"inetctl", 6}, Logged("inetctl", 6, "inetctl#6", u32(kError))},
        {{"inetctl", 7}, Logged("inetctl", 7, "inetctl#7", u32(kError))},
        {{"inetctl", 8}, Logged("inetctl", 8, "inetctl#8", u32(kError))},
        {{"inetctl", 9}, Logged("inetctl", 9, "inetctl#9", u32(kError))},
        {{"inetctl", 10}, Logged("inetctl", 10, "inetctl#10", u32(kError))},
        {{"inetctl", 11}, Logged("inetctl", 11, "inetctl#11", u32(kError))},
        // general exports DRTYSCKF needs
        {{"modload", 22}, Logged("modload", 22, "SearchModuleByName", 1)},
        {{"cdvdman", 74}, Logged("cdvdman", 74, "sceCdPowerOff", 1)},
        {{"cdvdman", 78}, Logged("cdvdman", 78, "sceCdPOffCallback", 0)},
        {{"ioman", 31}, Logged("ioman", 31, "ioman#31", 0)},
    };
    return table;
}

}  // namespace iop
}  // namespace ps2host
