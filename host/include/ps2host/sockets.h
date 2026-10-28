/*
 * Sockets with Winsock's names on every system: on Windows winsock2.h
 * itself, elsewhere the BSD sockets under the names the host's network code
 * (libnet.cpp) and the ports' (a LAN lobby) were written with - SOCKET,
 * closesocket, ioctlsocket, WSAGetLastError and its codes. SockLen is the
 * length accept() and getsockname() take (int on Windows).
 */

#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using SockLen = int;
#else

#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

using SOCKET = int;
using SockLen = socklen_t;
using u_long = unsigned long;
constexpr SOCKET INVALID_SOCKET = -1;
constexpr int SOCKET_ERROR = -1;
#define WSAEWOULDBLOCK EWOULDBLOCK
#define WSAEINPROGRESS EINPROGRESS
#define WSAEALREADY EALREADY
#define WSAECONNRESET ECONNRESET
#define WSAECONNABORTED ECONNABORTED
#define WSAEADDRINUSE EADDRINUSE
#define SD_SEND SHUT_WR
#define SD_BOTH SHUT_RDWR
#define MAKEWORD(a, b) ((unsigned short)(((a) & 0xFF) | (((b) & 0xFF) << 8)))
struct WSADATA {};
inline int WSAStartup(unsigned short, WSADATA*) { return 0; }
inline int WSACleanup() { return 0; }
inline int WSAGetLastError() { return errno == EAGAIN ? EWOULDBLOCK : errno; }
inline int closesocket(SOCKET s) { return close(s); }
inline int ioctlsocket(SOCKET s, long command, u_long* argument) {
    if (command == long(FIONBIO)) {
        const int flags = fcntl(s, F_GETFL, 0);
        return fcntl(s, F_SETFL, *argument ? flags | O_NONBLOCK : flags & ~O_NONBLOCK);
    }
    int value = int(*argument);
    const int result = ioctl(s, command, &value);
    *argument = u_long(value);
    return result;
}

#endif
