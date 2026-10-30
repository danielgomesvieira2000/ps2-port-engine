/* Hang reports (ps2host/hang_report.h). */

#include "ps2host/hang_report.h"

#include <cstdio>

#include "ps2host/iop_vm.h"
#include "ps2host/memory.h"

namespace ps2host {

namespace {
void (*g_extra)() = nullptr;
}  // namespace

void SaveMemories(const char* ee_path, const char* iop_path) {
    Memory& m = GuestMemory();
    if (std::FILE* f = std::fopen(ee_path, "wb")) {
        std::fwrite(m.rdram(), 1, kRdramSize, f);
        std::fclose(f);
    }
    if (std::FILE* f = std::fopen(iop_path, "wb")) {
        std::fwrite(m.Translate(kIopRamBase, kIopRamSize), 1, kIopRamSize, f);
        std::fclose(f);
    }
}

void SetHangReportExtra(void (*extra)()) { g_extra = extra; }

void HangReport(const std::string& heading, const std::string& ee_path, const std::string& iop_path) {
    std::fprintf(stderr, "%s\n%s%s", heading.c_str(), ThreadReport().c_str(), iop::Report().c_str());
    if (g_extra) g_extra();
    std::fflush(stderr);
    SaveMemories(ee_path.c_str(), iop_path.c_str());
}

void CaptureHangReport(const std::string& dir) {
    HangReport("[capture] the threads, the game's report and both memories too", dir + "/ee.bin",
               dir + "/iop.bin");
}

void EnableCaptureHangReport() { SetCaptureHook(CaptureHangReport); }

void StuckLoad(u32 load, u32 caller) {
    char heading[256];
    std::snprintf(heading, sizeof heading,
                  "[load] 0x%08X (from 0x%08X) not done after 20 s, frame %llu: the threads, the game's report "
                  "and stuck_ee.bin / stuck_iop.bin beside the log",
                  load, caller, static_cast<unsigned long long>(FrameNumber()));
    HangReport(heading, "stuck_ee.bin", "stuck_iop.bin");
}

}  // namespace ps2host
