/* Crash reports for players (Windows): what a player sends when the game
 * stops, in one zip.
 *
 * A player's run (started without arguments, main.cpp) arms them
 * (ArmCrashReports): its folder, its log, the game's title, version and
 * GitHub repository. Then:
 *
 *   - a crash (the unhandled-exception filter, profile.cpp: the code, the
 *     address and the stack written to the log first) starts this program
 *     again as `--crash-report <pid> <log>` (CrashReportHelper): it waits
 *     for the crashed process to go, makes the report and tells the player;
 *   - the game stopping with an error (ps2host::Main's code not 0): the
 *     report made by the process itself (ReportStop).
 *
 * The report: "crash reports/<target>-<version>-<date>-<time>.zip" beside
 * the game - the run's log, the run before's (<target>.prev.log), the PC
 * settings, the newest capture's pictures (if made in the last hour) and
 * report.txt (the game, its version, the executable, why, Windows' build,
 * the processor, the memory). The player's own folder in them is written
 * %USERPROFILE% (a log names the disc's path). The files are stored, not
 * compressed (the engine has no deflate). The message offers the folder
 * and a GitHub issue.
 *
 * A freeze too: a thread of its own watches the window's (run.h's
 * WindowAnsweredNs - read every frame, or told to wait in a long load);
 * no answer for 10 s, the stuck threads' stacks go to the log ([freeze]),
 * the report is made ("it stopped responding") and the player asked: the
 * folder, a GitHub issue, wait on, or close the game. Once a freeze; its end
 * logged. Not with a debugger attached; a PC asleep (the watch's own half
 * second long gone) is no freeze. freeze_report=0 (the PC settings) turns
 * it off.
 *
 * PS2HOST_TEST_CRASH=SECONDS (a test's): a player's run crashes that long
 * after it starts (a write to address 0 on a thread of its own); the report
 * made, no message shown.
 */

#ifndef PS2HOST_CRASH_REPORT_H
#define PS2HOST_CRASH_REPORT_H

#include <filesystem>
#include <string>

namespace ps2host {

struct ReportGame {
    std::string title;          // "Resident Evil Outbreak File 2"
    std::string version;        // "0.6.2-alpha"
    std::string target;         // "re_outbreak2": its executable's and log's name
    std::string github;         // "owner/repository" ("" none: no issue offered)
};

/* A player's run: crashes and errors stops reported (the folder beside the
   game, the run's log there). */
void ArmCrashReports(const std::filesystem::path& folder, const std::string& log, const ReportGame& game);
/* Whether `argv` is the crash helper's (`--crash-report <pid> <log>`). */
bool IsCrashReportHelper(int argc, char** argv);
/* The helper: waits for the crashed process, makes the report, tells the
   player. Its exit code. */
int CrashReportHelper(int argc, char** argv, const ReportGame& game);
/* The game stopped with an error (`code` not 0): the report made and the
   player told. False when not armed (the caller says what it can). */
bool ReportStop(int code);

}  // namespace ps2host

#endif
