/*
 * The IOP's module loader (server 0x80000006, loadfile), on the host.
 *
 * Nothing runs on an IOP here: what a module would provide, the host provides,
 * one RPC server at a time. So loading a module succeeds without running
 * anything - it is noted, given a module ID, and reported as having started
 * and stayed resident - and the servers it would have registered answer from
 * the host when the game binds them (or are noted as missing).
 *
 * The protocol, as Burnout 3's EE-side library (0x00113578-0x00113F98) speaks it:
 *   procedure 0xFF  the module's version, 4 characters: the library compares it
 *                   with the release it was built for and refuses to load
 *                   anything otherwise. "2800" is SDK release 2.8.
 *   procedure 0     load a module from a path. The request:
 *                     0x000  argument length       (the answer: module ID)
 *                     0x004  -                     (the answer: start result)
 *                     0x008  path, 252 bytes ("cdrom0:\\IOP\\USBD.IRX;1")
 *                     0x104  arguments, 252 bytes
 * Any other procedure is noted and fails.
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ps2host/disc.h"
#include "ps2host/host.h"
#include "ps2host/iop_vm.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

// The release the IOP's loadfile answers: until the game reboots the IOP with
// a replacement image, the one Burnout 3's ROM-booted IOP was taken to have.
char g_version[4] = {'2', '8', '0', '0'};
constexpr s32 kNoSuchProcedure = -1;
constexpr u32 kResidentEnd = 0;     // a module's start result: started, stays loaded

std::vector<u8> Words(s32 first, u32 second) {
    std::vector<u8> out(8);
    std::memcpy(&out[0], &first, 4);
    std::memcpy(&out[4], &second, 4);
    return out;
}

// A module the port runs (iop_vm.h): read from the disc and started.
std::vector<u8> RunFromDisc(s32 id, const std::string& path, const std::vector<u8>& data) {
    std::string disc_path = path;
    const std::size_t colon = disc_path.find(':');
    if (colon != std::string::npos) disc_path = disc_path.substr(colon + 1);
    Disc::File file;
    if (!GuestDisc().IsOpen() || !GuestDisc().Find(disc_path, &file))
        throw Unimplemented{"IOP: module " + path + " is to run, but is not on the disc"};
    std::vector<u8> image((file.size + kSectorSize - 1) / kSectorSize * kSectorSize);
    GuestDisc().Read(file.lsn, u32(image.size() / kSectorSize), image.data());
    image.resize(file.size);
    s32 arg_length = 0;
    std::memcpy(&arg_length, &data[0], 4);
    std::vector<u8> args;
    if (arg_length > 0 && data.size() >= 0x104 + std::size_t(arg_length))
        args.assign(data.begin() + 0x104, data.begin() + 0x104 + arg_length);
    s32 result = 0;
    if (!iop::LoadModule(path, image, args, &result))
        throw Unimplemented{"IOP: " + path + " is not a module the interpreter can load"};
    return Words(id, u32(result));
}

std::string Hex(u32 value) {
    char text[16];
    std::snprintf(text, sizeof text, "0x%08X", value);
    return text;
}

// A module the EE put in IOP memory (sceSifLoadModuleBuffer): its ELF image -
// as long as its program and section headers say - and the name its .iopmod
// section gives it ("sio2man"); empty when it is not a module.
std::vector<u8> ModuleInIopMemory(u32 address, std::string* name) {
    Memory& m = GuestMemory();
    auto u16at = [&](u32 o) { return u16(m.Read(kIopRamBase + address + o, 2)); };
    auto u32at = [&](u32 o) { return u32(m.Read(kIopRamBase + address + o, 4)); };
    name->clear();
    if (address >= kIopRamSize || u32at(0) != 0x464C457Fu) return {};
    const u32 phoff = u32at(0x1C), shoff = u32at(0x20);
    const u16 phentsize = u16at(0x2A), phnum = u16at(0x2C), shentsize = u16at(0x2E),
              shnum = u16at(0x30);
    u32 size = shoff + u32(shnum) * shentsize;
    for (u16 i = 0; i < phnum; ++i) {
        const u32 ph = phoff + u32(i) * phentsize;
        size = std::max(size, u32at(ph + 4) + u32at(ph + 16));
    }
    for (u16 i = 0; i < shnum; ++i) {
        const u32 sh = shoff + u32(i) * shentsize;
        size = std::max(size, u32at(sh + 16) + (u32at(sh + 4) == 8 ? 0 : u32at(sh + 20)));
    }
    if (size > kIopRamSize - address) return {};
    std::vector<u8> image(size);
    for (u32 i = 0; i < size; ++i) image[i] = u8(m.Read(kIopRamBase + address + i, 1));
    // .iopmod (type 0x70000080): module info, entry, gp, text, data and BSS
    // sizes, a 16-bit version, then the name.
    for (u16 i = 0; i < shnum; ++i) {
        const u32 sh = shoff + u32(i) * shentsize;
        if (u32at(sh + 4) != 0x70000080u) continue;
        const u32 at = u32at(sh + 16) + 0x1A;
        for (u32 k = at; k < size && image[k]; ++k) *name += char(image[k]);
    }
    // A module with no name (Resident Evil Outbreak's sound driver) is named by
    // its size, so a port can still ask for it to run: "unnamed-38664".
    if (name->empty()) *name = "unnamed-" + std::to_string(size);
    return image;
}

std::vector<u8> LoadFile(u32 procedure, const std::vector<u8>& data) {
    static s32 next_id = 1;
    if (procedure == 0xFF) return std::vector<u8>(g_version, g_version + 4);
    if (procedure == 0 && data.size() >= 0x104) {
        const char* begin = reinterpret_cast<const char*>(&data[8]);
        const std::string path(begin, strnlen(begin, 252));
        const s32 id = next_id++;
        if (iop::RunsModule(path)) return RunFromDisc(id, path, data);
        Note("module " + path, "IOP: module " + path + " loaded as " + std::to_string(id) +
                                   " (its services are the host's)");
        return Words(id, kResidentEnd);
    }
    if (procedure == 6 && data.size() >= 8) {
        // A module from IOP memory: its address, the arguments' length, the
        // arguments from 0x104 - answered as procedure 0 is.
        u32 address = 0;
        s32 arg_length = 0;
        std::memcpy(&address, &data[0], 4);
        std::memcpy(&arg_length, &data[4], 4);
        address &= kIopRamSize - 1;
        std::string name;
        const std::vector<u8> image = ModuleInIopMemory(address, &name);
        if (image.empty()) {
            Note("module buffer " + std::to_string(address),
                 "IOP: no module at IOP address " + Hex(address) + " to load (failed)");
            return Words(-1, 0);
        }
        const s32 id = next_id++;
        if (iop::RunsModule(name)) {
            std::vector<u8> args;
            if (arg_length > 0 && data.size() >= 0x104 + std::size_t(arg_length))
                args.assign(data.begin() + 0x104, data.begin() + 0x104 + arg_length);
            s32 result = 0;
            if (!iop::LoadModule(name, image, args, &result))
                throw Unimplemented{"IOP: " + name + " is not a module the interpreter can load"};
            return Words(id, u32(result));
        }
        Note("module " + name, "IOP: module " + name +
                                   " from IOP memory at " + Hex(address) + " loaded as " +
                                   std::to_string(id) + " (its services are the host's)");
        return Words(id, kResidentEnd);
    }
    Note("loadfile " + std::to_string(procedure),
         "IOP: loadfile procedure " + std::to_string(procedure) + " is not implemented (failed)");
    return Words(kNoSuchProcedure, 0);
}

}  // namespace

void InstallLoadFile() { RegisterRpcServer(0x80000006u, LoadFile); }

// An IOP replacement image (IOPRPnnn.IMG) holds its modules, each with its
// library tag - eight characters of name, four of release: "loadfile2340".
// Its loadfile's release is what loadfile answers from then on.
void UseIopImage(const std::string& path) {
    std::string disc_path = path;
    const std::size_t colon = disc_path.find(':');
    if (colon != std::string::npos) disc_path = disc_path.substr(colon + 1);
    std::vector<u8> image;
    if (!GuestDisc().IsOpen() || !GuestDisc().ReadFile(disc_path, &image)) {
        Note("iop image " + path, "IOP: reboot image " + path + " is not on the disc");
        return;
    }
    static const char kTag[] = "loadfile";
    for (std::size_t i = 0; i + 12 <= image.size(); ++i) {
        if (std::memcmp(&image[i], kTag, 8) != 0) continue;
        bool digits = true;
        for (int k = 8; k < 12; ++k) digits = digits && image[i + k] >= '0' && image[i + k] <= '9';
        if (!digits) continue;
        std::memcpy(g_version, &image[i + 8], 4);
        Note("iop image " + path, "IOP: rebooted with " + path + ": loadfile " +
                                      std::string(g_version, 4));
        return;
    }
    Note("iop image " + path, "IOP: reboot image " + path + " has no loadfile tag");
}

const char* IopRelease() { return g_version; }

}  // namespace ps2host
