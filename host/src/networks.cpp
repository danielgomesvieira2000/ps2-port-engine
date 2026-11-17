/*
 * This PC's networks (ps2host/networks.h): GetAdaptersAddresses on Windows,
 * getifaddrs elsewhere.
 */

#include "ps2host/networks.h"

#include <algorithm>

#include "ps2host/sockets.h"
#ifdef _WIN32
#include <iphlpapi.h>
#else
#include <ifaddrs.h>
#include <net/if.h>
#endif

namespace ps2host {

namespace {

bool SelfAssigned(std::uint32_t address) { return (address >> 16) == 0xA9FEu; }

}  // namespace

std::string LocalNetwork::Address() const {
    return std::to_string(address >> 24) + "." + std::to_string((address >> 16) & 0xFF) + "." +
           std::to_string((address >> 8) & 0xFF) + "." + std::to_string(address & 0xFF);
}

#ifdef _WIN32
std::vector<LocalNetwork> LocalNetworks() {
    std::vector<LocalNetwork> out;
    ULONG size = 16384;
    std::vector<unsigned char> buffer;
    ULONG result = ERROR_BUFFER_OVERFLOW;
    for (int tries = 0; tries < 3 && result == ERROR_BUFFER_OVERFLOW; ++tries) {
        buffer.resize(size);
        result = GetAdaptersAddresses(AF_INET,
                                      GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER |
                                          GAA_FLAG_INCLUDE_GATEWAYS,
                                      nullptr, reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data()), &size);
    }
    if (result != NO_ERROR) return out;
    for (const IP_ADAPTER_ADDRESSES* a = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data()); a; a = a->Next) {
        if (a->OperStatus != IfOperStatusUp || a->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;
        std::string name;
        for (const wchar_t* w = a->FriendlyName; w && *w; ++w) name += *w >= 32 && *w < 127 ? char(*w) : '?';
        for (const IP_ADAPTER_UNICAST_ADDRESS* u = a->FirstUnicastAddress; u; u = u->Next) {
            if (u->Address.lpSockaddr->sa_family != AF_INET) continue;
            const std::uint32_t ip = ntohl(reinterpret_cast<const sockaddr_in*>(u->Address.lpSockaddr)->sin_addr.s_addr);
            if (SelfAssigned(ip)) continue;
            const unsigned bits = u->OnLinkPrefixLength;
            LocalNetwork n;
            n.address = ip;
            n.mask = bits == 0 ? 0 : bits >= 32 ? 0xFFFFFFFFu : ~0u << (32 - bits);
            n.name = name;
            n.gateway = a->FirstGatewayAddress != nullptr;
            out.push_back(n);
        }
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const LocalNetwork& x, const LocalNetwork& y) { return x.gateway && !y.gateway; });
    return out;
}
#else
std::vector<LocalNetwork> LocalNetworks() {
    std::vector<LocalNetwork> out;
    ifaddrs* all = nullptr;
    if (getifaddrs(&all) != 0) return out;
    for (const ifaddrs* a = all; a; a = a->ifa_next) {
        if (!a->ifa_addr || a->ifa_addr->sa_family != AF_INET || !a->ifa_netmask) continue;
        if (!(a->ifa_flags & IFF_UP) || (a->ifa_flags & IFF_LOOPBACK)) continue;
        const std::uint32_t ip = ntohl(reinterpret_cast<const sockaddr_in*>(a->ifa_addr)->sin_addr.s_addr);
        if (SelfAssigned(ip)) continue;
        LocalNetwork n;
        n.address = ip;
        n.mask = ntohl(reinterpret_cast<const sockaddr_in*>(a->ifa_netmask)->sin_addr.s_addr);
        n.name = a->ifa_name ? a->ifa_name : "?";
        out.push_back(n);
    }
    freeifaddrs(all);
    return out;
}
#endif

}  // namespace ps2host
