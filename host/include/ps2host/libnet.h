/*
 * Sony's libnet (the inet stack's EE client: sceInetCreate, sceInetOpen,
 * sceInetSendTo...) answered on the EE with the PC's sockets.
 *
 * A game's libnet client sends each call to its IOP server through MSIFRPC,
 * which a port does not run; with its wrappers' addresses (the functions that
 * pack a call for procedure 1 Create, 2 Open, 3 Close, 4 Recv, 5 Send, 13
 * RecvFrom, 14 SendTo, 15 Abort, 19 Control, 20 Poll) Register answers them
 * with non-blocking Winsock sockets: UDP for datagram (and raw: pings that go
 * nowhere) sockets, TCP for active and passive ones. Nothing waits: a call
 * with nothing to give returns at once.
 */

#ifndef PS2HOST_LIBNET_H
#define PS2HOST_LIBNET_H

#include <cstdint>

namespace ps2host {
namespace libnet {

/* The addresses of the game's libnet client wrappers (0: not answered). */
struct Client {
    std::uint32_t create = 0, open = 0, close = 0, recv = 0, send = 0, recv_from = 0, send_to = 0, abort = 0,
                  control = 0, poll = 0;
};

struct Options {
    /* An environment variable holding this instance's IPv4 address for its
       sockets (two instances on one PC: 127.0.0.1 and 127.0.0.2); unset, any. */
    const char* address_env = "PS2HOST_LAN_IP";
    /* An environment variable that, set, logs each datagram (its first bytes). */
    const char* dump_env = "PS2HOST_LAN_DUMP";
    /* The calls told in the log (and a traffic summary every 2 seconds). */
    bool (*debug)() = nullptr;
};

void Register(const Client& client, const Options& options = {});

}  // namespace libnet
}  // namespace ps2host

#endif  /* PS2HOST_LIBNET_H */
