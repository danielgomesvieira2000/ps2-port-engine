/*
 * This PC's networks, for a port's LAN play: each IPv4 address of an
 * interface that is up - not the loopback, not a self-assigned 169.254
 * address (no network answered it). A player on two (the real LAN and a VPN,
 * Wi-Fi and a cable) chooses the one the others are on.
 *
 * Burnout 3's LAN lobby (its port's lan_lobby.cpp) lists them the same way.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ps2host {

struct LocalNetwork {
    std::uint32_t address = 0;      // host order: 192.168.1.20 is 0xC0A80114
    std::uint32_t mask = 0;
    std::string name;               // the interface's: "Wi-Fi", "Ethernet 2", "eth0"
    bool gateway = false;           // a default route through it (Windows): the PC's own network, likely

    std::string Address() const;    // "192.168.1.20"
    std::uint32_t Broadcast() const { return address | ~mask; }
};

/* The networks, those with a gateway first (a VPN's or a virtual machine's
   adapter after), each in the order the system gives. */
std::vector<LocalNetwork> LocalNetworks();

}  // namespace ps2host
