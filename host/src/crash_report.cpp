/* Crash reports for players (ps2host/crash_report.h). */

#include "ps2host/crash_report.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#include "ps2host/profile.h"
#include "ps2host/run.h"
#include "ps2host/settings.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#include <shellapi.h>

#include <SDL3/SDL.h>
#endif

namespace ps2host {

#ifdef _WIN32

namespace fs = std::filesystem;

namespace {

bool g_armed = false;
fs::path g_folder;
std::string g_log;
ReportGame g_game;

std::string Narrow(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(std::size_t(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

std::wstring Widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(std::size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
    return w;
}

/* The message titled `title`, once it is up, put in front of every window
   and given the focus: a process started in the background (the crash's
   helper) opens its windows behind the one the player is at. */
void Raise(const std::string& title) {
    std::thread([wide = Widen(title)] {
        for (int i = 0; i < 60; ++i) {
            Sleep(50);
            if (HWND w = FindWindowW(nullptr, wide.c_str())) {
                SetWindowPos(w, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
                SetForegroundWindow(w);
                return;
            }
        }
    }).detach();
}

// ------------------------------------------------------------ a zip, stored

std::uint32_t Crc32(const std::string& data) {
    static std::uint32_t table[256];
    static const bool made = [] {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        return true;
    }();
    (void)made;
    std::uint32_t c = 0xFFFFFFFFu;
    for (unsigned char b : data) c = table[(c ^ b) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

struct Entry {
    std::string name;           // in the zip, '/' between folders
    std::string data;
};

void Put16(std::string& s, unsigned v) { s += char(v & 0xFF), s += char((v >> 8) & 0xFF); }
void Put32(std::string& s, std::uint32_t v) { Put16(s, v & 0xFFFF), Put16(s, v >> 16); }

/* The entries stored (method 0), their names UTF-8 (flag bit 11). */
bool WriteZip(const fs::path& path, const std::vector<Entry>& entries) {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &now);
    const unsigned dos_time = unsigned(tm.tm_hour << 11 | tm.tm_min << 5 | tm.tm_sec / 2);
    const unsigned dos_date = unsigned((tm.tm_year - 80) << 9 | (tm.tm_mon + 1) << 5 | tm.tm_mday);
    std::FILE* f = _wfopen(path.wstring().c_str(), L"wb");
    if (!f) return false;
    std::string central;
    std::uint32_t at = 0;
    bool ok = true;
    for (const Entry& e : entries) {
        const std::uint32_t crc = Crc32(e.data), size = std::uint32_t(e.data.size());
        std::string local;
        Put32(local, 0x04034B50u);
        Put16(local, 20), Put16(local, 0x0800), Put16(local, 0), Put16(local, dos_time), Put16(local, dos_date);
        Put32(local, crc), Put32(local, size), Put32(local, size);
        Put16(local, unsigned(e.name.size())), Put16(local, 0);
        local += e.name;
        ok = ok && std::fwrite(local.data(), 1, local.size(), f) == local.size();
        ok = ok && std::fwrite(e.data.data(), 1, e.data.size(), f) == e.data.size();
        Put32(central, 0x02014B50u);
        Put16(central, 20), Put16(central, 20), Put16(central, 0x0800), Put16(central, 0);
        Put16(central, dos_time), Put16(central, dos_date);
        Put32(central, crc), Put32(central, size), Put32(central, size);
        Put16(central, unsigned(e.name.size())), Put16(central, 0), Put16(central, 0);
        Put16(central, 0), Put16(central, 0), Put32(central, 0), Put32(central, at);
        central += e.name;
        at += std::uint32_t(local.size()) + size;
    }
    std::string end;
    Put32(end, 0x06054B50u);
    Put16(end, 0), Put16(end, 0), Put16(end, unsigned(entries.size())), Put16(end, unsigned(entries.size()));
    Put32(end, std::uint32_t(central.size())), Put32(end, at), Put16(end, 0);
    ok = ok && std::fwrite(central.data(), 1, central.size(), f) == central.size();
    ok = ok && std::fwrite(end.data(), 1, end.size(), f) == end.size();
    ok = std::fclose(f) == 0 && ok;
    return ok;
}

// ------------------------------------------------------------ the report

constexpr std::size_t kMostOfAFile = 48u << 20;     // a longer log: its first 8 MB and last 40
constexpr std::uintmax_t kMostOfACapture = 16u << 20;

bool ReadFile(const fs::path& path, std::string* out) {
    std::FILE* f = _wfopen(path.wstring().c_str(), L"rb");
    if (!f) return false;
    out->clear();
    char chunk[1 << 16];
    for (std::size_t n; (n = std::fread(chunk, 1, sizeof chunk, f)) > 0;) out->append(chunk, n);
    std::fclose(f);
    if (out->size() > kMostOfAFile)
        *out = out->substr(0, 8u << 20) + "\n\n[report] ... (the middle of the file left out) ...\n\n" +
               out->substr(out->size() - (40u << 20));
    return true;
}

/* The player's own folder (C:\Users\<name>) written %USERPROFILE% - either
   slash, any case. */
std::string Private(std::string text) {
    wchar_t profile[MAX_PATH] = {};
    if (!GetEnvironmentVariableW(L"USERPROFILE", profile, MAX_PATH)) return text;
    const std::string mine = Narrow(profile);
    if (mine.size() < 4) return text;
    for (const bool forward : {false, true}) {
        std::string from = mine;
        if (forward)
            for (char& c : from) c = c == '\\' ? '/' : c;
        std::string lower_text = text, lower_from = from;
        for (char& c : lower_text) c = char(tolower(static_cast<unsigned char>(c)));
        for (char& c : lower_from) c = char(tolower(static_cast<unsigned char>(c)));
        std::string out;
        std::size_t at = 0;
        for (std::size_t hit; (hit = lower_text.find(lower_from, at)) != std::string::npos; at = hit + from.size())
            out += text.substr(at, hit - at) + "%USERPROFILE%";
        out += text.substr(at);
        text = out;
    }
    return text;
}

std::string Processor() {
    int info[4];
    __cpuid(info, 0x80000000);
    if (unsigned(info[0]) < 0x80000004u) return "?";
    char brand[49] = {};
    for (int i = 0; i < 3; ++i) {
        __cpuid(info, 0x80000002 + i);
        std::memcpy(brand + 16 * i, info, 16);
    }
    std::string s = brand;
    while (!s.empty() && s.front() == ' ') s.erase(s.begin());
    return s;
}

std::string WindowsBuild() {
    using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
    const auto get = reinterpret_cast<RtlGetVersionFn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion")));
    OSVERSIONINFOW v{};
    v.dwOSVersionInfoSize = sizeof v;
    if (!get || get(&v) != 0) return "?";
    char text[64];
    std::snprintf(text, sizeof text, "%lu.%lu.%lu", v.dwMajorVersion, v.dwMinorVersion, v.dwBuildNumber);
    return text;
}

std::string Stamp(const char* form) {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &now);
    char text[64];
    std::strftime(text, sizeof text, form, &tm);
    return text;
}

/* The report made: its path ("" none). `why`: what happened, in words. */
fs::path MakeReport(const fs::path& folder, const fs::path& log, const ReportGame& game, const std::string& why) {
    std::error_code error;
    const fs::path dir = folder / "crash reports";
    fs::create_directories(dir, error);
    std::vector<Entry> entries;
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof memory;
    GlobalMemoryStatusEx(&memory);
    char ram[32];
    std::snprintf(ram, sizeof ram, "%.1f GB", double(memory.ullTotalPhys) / double(1ull << 30));
    std::string text = game.title + " " + game.version + "\r\n" +
                       "Executable: " + Narrow(fs::path(exe).filename().wstring()) + "\r\n" +
                       "When: " + Stamp("%Y-%m-%d %H:%M:%S") + "\r\n" +
                       "What happened: " + why + "\r\n" +
                       "Windows: " + WindowsBuild() + "\r\n" +
                       "Processor: " + Processor() + "\r\n" +
                       "Memory: " + ram + "\r\n";
    entries.push_back({"report.txt", text});
    std::string data;
    if (ReadFile(log, &data)) entries.push_back({Narrow(log.filename().wstring()), Private(data)});
    fs::path prev = log;
    prev.replace_extension(".prev.log");
    if (ReadFile(prev, &data)) entries.push_back({Narrow(prev.filename().wstring()), Private(data)});
    if (ReadFile(folder / "pc_settings.ini", &data)) entries.push_back({"pc_settings.ini", Private(data)});
    // The newest capture (F12) of the last hour: its pictures and small files
    // (not its GS stream or trace, tens of MB).
    fs::path newest;
    fs::file_time_type newest_time{};
    for (const auto& e : fs::directory_iterator(folder / "captures", error))
        if (e.is_directory(error) && (newest.empty() || e.last_write_time(error) > newest_time))
            newest = e.path(), newest_time = e.last_write_time(error);
    if (!newest.empty() && fs::file_time_type::clock::now() - newest_time < std::chrono::hours(1))
        for (const auto& e : fs::directory_iterator(newest, error)) {
            const std::string name = Narrow(e.path().filename().wstring());
            if (!e.is_regular_file(error) || e.file_size(error) > kMostOfACapture || name == "replay.bin" ||
                name == "trace.txt" || !ReadFile(e.path(), &data))
                continue;
            entries.push_back({"capture/" + Narrow(newest.filename().wstring()) + "/" + name, data});
        }
    const fs::path zip = dir / Widen(game.target + "-" + game.version + "-" + Stamp("%Y%m%d-%H%M%S") + ".zip");
    if (!WriteZip(zip, entries)) {
        std::fprintf(stderr, "[report] could not write %s\n", Narrow(zip.wstring()).c_str());
        return {};
    }
    std::fprintf(stderr, "[report] %s: %zu files\n", Narrow(zip.wstring()).c_str(), entries.size());
    return zip;
}

/* The player told where the report is: the folder, a GitHub issue, or close. */
void Tell(const fs::path& zip, const ReportGame& game, bool crashed) {
    if (std::getenv("PS2HOST_TEST_CRASH")) return;      // (a test's: no one to answer)
    const std::string title = game.title.empty() ? "PS2 port" : game.title;
    std::string text = title + (crashed ? " crashed." : " stopped because of an error.");
    if (zip.empty()) {
        text += "\n\nNo crash report could be saved. What happened is in " + game.target + ".log, beside the game.";
        Raise(title);
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, title.c_str(), text.c_str(), nullptr);
        return;
    }
    text += "\n\nA crash report was saved in the \"crash reports\" folder beside the game:\n" +
            Narrow(zip.filename().wstring()) +
            "\n\nPlease attach it to a report of the problem, with what you were doing when it happened.";
    std::vector<SDL_MessageBoxButtonData> buttons = {{SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT, 0, "Open the folder"}};
    if (!game.github.empty()) buttons.push_back({0, 1, "Report on GitHub"});
    buttons.push_back({SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT, 2, "Close"});
    const SDL_MessageBoxData box{SDL_MESSAGEBOX_ERROR | SDL_MESSAGEBOX_BUTTONS_LEFT_TO_RIGHT, nullptr, title.c_str(),
                                 text.c_str(), int(buttons.size()), buttons.data(), nullptr};
    int chosen = 2;
    Raise(title);
    if (!SDL_ShowMessageBox(&box, &chosen)) return;
    if (chosen == 0) {
        const std::wstring select = L"/select,\"" + zip.wstring() + L"\"";
        ShellExecuteW(nullptr, L"open", L"explorer.exe", select.c_str(), nullptr, SW_SHOWNORMAL);
    } else if (chosen == 1) {
        const std::wstring url = Widen("https://github.com/" + game.github + "/issues/new");
        ShellExecuteW(nullptr, L"open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    }
}

/* A freeze: the player asked what now (the report saved or not). */
void TellFreeze(const fs::path& zip, const ReportGame& game, double seconds) {
    if (std::getenv("PS2HOST_TEST_HANG")) return;       // (a test's: no one to answer)
    const std::string title = game.title.empty() ? "PS2 port" : game.title;
    char waited[64];
    std::snprintf(waited, sizeof waited, "%.0f seconds", seconds);
    std::string text = title + " has not responded for " + waited + ".";
    text += zip.empty() ? "\n\nNo report could be saved; what happened is in " + game.target + ".log, beside the game."
                        : "\n\nA report was saved in the \"crash reports\" folder beside the game:\n" +
                              Narrow(zip.filename().wstring()) +
                              "\n\nPlease attach it to a report of the problem, with what you were doing when it happened.";
    text += "\n\nYou can wait to see whether it comes back, or close it.";
    std::vector<SDL_MessageBoxButtonData> buttons;
    if (!zip.empty()) buttons.push_back({0, 0, "Open the folder"});
    if (!zip.empty() && !game.github.empty()) buttons.push_back({0, 1, "Report on GitHub"});
    buttons.push_back({SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT | SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT, 2, "Keep waiting"});
    buttons.push_back({0, 3, "Close the game"});
    const SDL_MessageBoxData box{SDL_MESSAGEBOX_WARNING | SDL_MESSAGEBOX_BUTTONS_LEFT_TO_RIGHT, nullptr, title.c_str(),
                                 text.c_str(), int(buttons.size()), buttons.data(), nullptr};
    int chosen = 2;
    Raise(title);
    if (!SDL_ShowMessageBox(&box, &chosen)) return;
    if (chosen == 0) {
        const std::wstring select = L"/select,\"" + zip.wstring() + L"\"";
        ShellExecuteW(nullptr, L"open", L"explorer.exe", select.c_str(), nullptr, SW_SHOWNORMAL);
    } else if (chosen == 1) {
        ShellExecuteW(nullptr, L"open", Widen("https://github.com/" + game.github + "/issues/new").c_str(), nullptr,
                      nullptr, SW_SHOWNORMAL);
    } else if (chosen == 3) {
        std::fprintf(stderr, "[freeze] the player closed the game\n");
        std::fflush(nullptr);
        TerminateProcess(GetCurrentProcess(), 3);
    }
}

constexpr std::int64_t kFrozenNs = 10'000'000'000;      // no answer this long: a freeze
constexpr std::int64_t kAsleepNs = 5'000'000'000;       // the watch's half second this long: the PC slept

std::int64_t SteadyNow() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/* The freeze watch (crash_report.h): every half second, the window's answer. */
void WatchForFreezes() {
    std::int64_t last_look = SteadyNow(), grace_from = 0;
    bool reported = false;
    std::int64_t frozen_since = 0;
    for (;;) {
        Sleep(500);
        const std::int64_t now = SteadyNow();
        if (now - last_look > kAsleepNs) grace_from = now;      // (asleep, or the whole process stopped: counted from now)
        last_look = now;
        const std::int64_t answered = WindowAnsweredNs();
        if (!answered || IsDebuggerPresent()) continue;
        const std::int64_t since = now - std::max(answered, grace_from);
        if (since < kFrozenNs) {
            if (reported)
                std::fprintf(stderr, "[freeze] the game answers again, after %.1f s\n",
                             double(now - frozen_since) / 1e9);
            reported = false;
            continue;
        }
        if (reported) continue;
        reported = true;
        frozen_since = std::max(answered, grace_from);
        const double seconds = double(since) / 1e9;
        std::fprintf(stderr, "[freeze] the game has not answered for %.1f s\n", seconds);
        const std::string window = ThreadStack(WindowThreadId());
        std::fprintf(stderr, "[freeze]   the window's thread: %s\n", window.empty() ? "?" : window.c_str());
        const std::string renderer = ThreadStack(RendererThreadId());
        if (!renderer.empty()) std::fprintf(stderr, "[freeze]   the renderer's thread: %s\n", renderer.c_str());
        std::fflush(stderr);
        const fs::path zip = MakeReport(g_folder, g_folder / Widen(g_log), g_game,
                                        "it stopped responding (no answer for 10 s: the [freeze] lines in the log)");
        TellFreeze(zip, g_game, seconds);
    }
}

}  // namespace

void ArmCrashReports(const fs::path& folder, const std::string& log, const ReportGame& game) {
    g_armed = true;
    g_folder = folder;
    g_log = log;
    g_game = game;
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    const std::wstring command = L"\"" + std::wstring(exe) + L"\" --crash-report " +
                                 std::to_wstring(GetCurrentProcessId()) + L" \"" + (folder / Widen(log)).wstring() + L"\"";
    SetCrashCommand(command.c_str());
    if (SettingOn("freeze_report", true)) std::thread(WatchForFreezes).detach();
    if (const char* v = std::getenv("PS2HOST_TEST_CRASH")) {
        const int seconds = std::atoi(v);
        std::fprintf(stderr, "[report] a test's crash in %d s\n", seconds);
        std::thread([seconds] {
            Sleep(DWORD(seconds) * 1000);
            volatile int* nowhere = nullptr;
            *nowhere = 1;
        }).detach();
    }
}

bool IsCrashReportHelper(int argc, char** argv) { return argc == 4 && std::string(argv[1]) == "--crash-report"; }

int CrashReportHelper(int, char**, const ReportGame& game) {
    // (the arguments again as Windows has them: a log's path in any language)
    int count = 0;
    wchar_t** args = CommandLineToArgvW(GetCommandLineW(), &count);
    if (!args || count != 4) return 2;
    const DWORD pid = DWORD(std::wcstoul(args[2], nullptr, 10));
    const fs::path log = args[3];
    LocalFree(args);
    // The crashed process gone first (Windows' own report of it may hold it a while).
    if (HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, pid)) {
        WaitForSingleObject(process, 20000);
        CloseHandle(process);
    }
    const fs::path zip = MakeReport(log.parent_path(), log, game,
                                    "it crashed (the [crash] lines at the end of the log say where)");
    Tell(zip, game, true);
    return 0;
}

bool ReportStop(int code) {
    if (!g_armed) return false;
    std::fflush(nullptr);
    const fs::path zip = MakeReport(g_folder, g_folder / Widen(g_log), g_game,
                                    "it stopped with an error (code " + std::to_string(code) + ")");
    Tell(zip, g_game, false);
    return true;
}

#else

void ArmCrashReports(const std::filesystem::path&, const std::string&, const ReportGame&) {}
bool IsCrashReportHelper(int, char**) { return false; }
int CrashReportHelper(int, char**, const ReportGame&) { return 0; }
bool ReportStop(int) { return false; }

#endif

}  // namespace ps2host
