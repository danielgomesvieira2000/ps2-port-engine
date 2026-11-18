/* The update check for players (Windows): each start of a player's run
 * (main.cpp, started without arguments) asks GitHub for the game's newest
 * release; one newer than this build asks the player, in a small window:
 *
 *   - UPDATE NOW: its zip downloaded (a window with the progress), checked
 *     (its size, and its SHA-256 when GitHub gives one), unpacked with
 *     Windows' own tar, and put in place - each file of the release over
 *     the one beside the game, an executable (this one runs) renamed
 *     <name>.old first (removed at the next start), the player's files
 *     (memory card, settings, captures) untouched; then the new version
 *     started and this process ends;
 *   - OPEN THE RELEASE PAGE: the release's page in the browser, the game
 *     starting as it is;
 *   - CONTINUE WITHOUT UPDATING.
 *
 * The releases: https://api.github.com/repos/<github>/releases (pre-releases
 * too), the newest by version whose assets have this game's zip -
 * "<release name>-<version>-windows-x64.zip" (engine/tools/package.py's
 * name). Versions are compared by their numbers ("0.6.2-alpha" > "0.6.1-
 * alpha"). Nothing in reach (offline, GitHub slow: the start waits 4 s at
 * most - the check on a thread of its own), or anything failing: the game
 * starts as it is, no message, the reason in the log.
 *
 * The setting update_check=0 (pc_settings.ini) turns it off.
 * PS2HOST_TEST_VERSION=V (a test's): this build taken as version V.
 * PS2HOST_TEST_UPDATE=update|page|continue (a test's): the player's answer.
 * PS2HOST_TEST_UPDATE_DELAY=MS (a test's): GitHub asked that much later (a
 * slow network: past 4000, the game starts without it).
 * PS2HOST_GITHUB_TOKEN=T (a tester's, a private repository's): GitHub asked
 * with that token (sent to api.github.com alone).
 */

#ifndef PS2HOST_UPDATER_H
#define PS2HOST_UPDATER_H

#include <filesystem>
#include <string>

namespace ps2host {

struct UpdateGame {
    std::string title;          // "Resident Evil Outbreak File 2"
    std::string version;        // this build's: "0.6.1-alpha"
    std::string target;         // its executable: "re_outbreak2"
    std::string github;         // "owner/repository"
    std::string release_name;   // its zip's: "ResidentEvilOutbreakFile2-PC"
};

/* At a player's start: the check, the question, the update. True: the new
   version was put in place and started - this process should end now. */
bool CheckForUpdate(const std::filesystem::path& folder, const UpdateGame& game);

/* "0.6.2-alpha" against "0.6.1-alpha": -1, 0 or 1 by their numbers. */
int CompareVersions(const std::string& a, const std::string& b);

}  // namespace ps2host

#endif
