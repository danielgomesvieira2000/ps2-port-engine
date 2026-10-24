/*
 * Logitech's IOP modules with nothing attached (ps2host/logitech.h).
 */

#include "ps2host/logitech.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ps2host/host.h"
#include "ps2host/run.h"

namespace ps2host {
namespace logitech {
namespace {

constexpr u32 kNoDevice = 0x80000001u;      // the libraries test bit 31 only
u32 g_dev_version = 0, g_aud_version = 0;

/* LGDEVW.IRX. Burnout 3's EE library, lgDevInit at 0x004438E0, calls
   procedure 12 with a 0x240-byte buffer and reads the module's version from
   the answer's word at +4: its own exactly, or it warns for any other of the
   same major version and halts for anything else ("wrong version of
   lgDev.irx").
     procedure 1   lgDevEnumerate(index, description*): the answer's first word
                   is the result, an error when bit 31 is set; the caller asks
                   for index after index until one fails, so with no device
                   every index fails.
   Other procedures are noted and answered with zeros. */
std::vector<u8> LgDev(u32 procedure, const std::vector<u8>& data) {
    std::vector<u8> answer(data.size() < 8 ? 8 : data.size(), 0);
    if (procedure == 12) {
        std::memcpy(&answer[4], &g_dev_version, 4);
        return answer;
    }
    if (procedure == 1) {
        std::memcpy(&answer[0], &kNoDevice, 4);
        return answer;
    }
    char text[96];
    std::snprintf(text, sizeof text, "IOP: lgDev procedure %u is not implemented (zeros)",
                  unsigned(procedure));
    Note("lgdev " + std::to_string(procedure), text);
    return answer;
}

/* LGAUD.IRX: the headset (Burnout 3's online voice chat). Its EE library,
   lgAudInit (Burnout 3's at 0x002E0838), calls procedure 0x10 with 0x20
   bytes and reads the module's version from the answer's word at +8, as
   lgDev's; procedure 1, lgAudEnumerate, as lgDev's. */
std::vector<u8> LgAud(u32 procedure, const std::vector<u8>& data) {
    std::vector<u8> answer(data.size() < 0x30 ? 0x30 : data.size(), 0);
    if (procedure == 0x10) {
        std::memcpy(&answer[8], &g_aud_version, 4);
        return answer;
    }
    if (procedure == 1) {
        std::memcpy(&answer[0], &kNoDevice, 4);
        return answer;
    }
    char text[96];
    std::snprintf(text, sizeof text, "IOP: lgAud procedure %u is not implemented (zeros)",
                  unsigned(procedure));
    Note("lgaud " + std::to_string(procedure), text);
    return answer;
}

}  // namespace

void RegisterNoDevices(std::uint32_t version) {
    g_dev_version = version;
    RegisterRpcServer(0x046D046Du, LgDev);
}

void RegisterNoHeadsets(std::uint32_t version) {
    g_aud_version = version;
    RegisterRpcServer(0x50494C42u, LgAud);
}

}  // namespace logitech
}  // namespace ps2host
