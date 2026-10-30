/* What a port says about a hang, and loads stepped faster.
 *
 * A hang a player meets (an endless loading screen, music that stopped) is
 * looked into from what the port saves at the moment: every thread's state
 * (ThreadReport), the IOP's (iop::Report), what the game's port adds (its
 * sound engine's last words, say: SetHangReportExtra) and both memories
 * (SaveMemories), which the engine's tools read back.
 *
 *   - F12 (a capture) saves it all into the capture's folder
 *     (EnableCaptureHangReport);
 *   - a load that has not finished after 20 s saves it once beside the log
 *     (FastLoading's watch).
 *
 * FastLoading<Load, address>: a game's load step, called once a frame until
 * it returns non-zero, taken again within the frame - the IOP running a
 * slice between steps - for up to half a frame, while the vertical blank is
 * not due: the loading screen keeps its pace, the load is a few times
 * faster. PS2HOST_LOAD_SPEED=1 keeps a step a frame, as the console. The
 * step must be one that can be called again with its arguments (Burnout 3's
 * race and boot loads: 6.3 s -> 2.6 s and 5.9 s -> 2.4 s).
 */

#ifndef PS2HOST_HANG_REPORT_H
#define PS2HOST_HANG_REPORT_H

#include <cstdlib>
#include <cstring>
#include <string>

#include "ps2host/host.h"
#include "ps2host/run.h"

namespace ps2host {

/* EE and IOP memory, whole, to two files. */
void SaveMemories(const char* ee_path, const char* iop_path);
/* What the game's port adds to a hang report, written to stderr after the
   threads and the IOP (null: nothing). */
void SetHangReportExtra(void (*extra)());
/* A hang report to the log: `heading`, the threads, the IOP, the port's
   extra; then both memories to ee_path and iop_path. */
void HangReport(const std::string& heading, const std::string& ee_path, const std::string& iop_path);
/* F12 (a capture) saves a hang report too: the log's, and ee.bin and iop.bin
   in the capture's folder (SetCaptureHook's; a port with its own capture
   hook calls CaptureHangReport from it instead). */
void EnableCaptureHangReport();
void CaptureHangReport(const std::string& dir);
/* FastLoading's: a load at `load` (called from `caller`) unfinished after 20 s. */
void StuckLoad(u32 load, u32 caller);

template <void (*Load)(lifted::Context&), u32 kAddress>
void FastLoading(lifted::Context& c) {
    static const bool console = [] {
        const char* speed = std::getenv("PS2HOST_LOAD_SPEED");
        return speed && std::atoi(speed) <= 1;
    }();
    static u64 waiting_since = 0;
    static bool said = false;
    constexpr u64 kStepsNs = 8'000'000;         // half a frame
    const u32 caller = u32(c.r[31]);
    u64 args[32];
    std::memcpy(args, c.r, sizeof args);
    const u64 from = ClockNs();
    Load(c);
    if (!console)
        while (!u32(c.r[2]) && !BlankDue() && ClockNs() - from < kStepsNs && RunIopBetweenSteps()) {
            std::memcpy(c.r, args, sizeof args);
            Load(c);
        }
    if (u32(c.r[2])) {
        waiting_since = 0;
        said = false;
    } else if (!waiting_since) {
        waiting_since = from;
    } else if (!said && from - waiting_since > 20'000'000'000ull) {
        said = true;
        StuckLoad(kAddress, caller);
    }
}

}  // namespace ps2host

#endif  /* PS2HOST_HANG_REPORT_H */
