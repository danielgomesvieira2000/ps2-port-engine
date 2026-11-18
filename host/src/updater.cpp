/* The update check for players (ps2host/updater.h). */

#include "ps2host/updater.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ps2host/settings.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <shellapi.h>
#include <winhttp.h>

#include <SDL3/SDL.h>
#endif

namespace ps2host {

int CompareVersions(const std::string& a, const std::string& b) {
    // The numbers before the first '-' ("0.6.2" of "0.6.2-alpha"), one by one.
    auto numbers = [](const std::string& v) {
        std::vector<long> out;
        std::size_t at = v.size() && (v[0] == 'v' || v[0] == 'V') ? 1 : 0;
        while (at < v.size() && std::isdigit(static_cast<unsigned char>(v[at]))) {
            long n = 0;
            while (at < v.size() && std::isdigit(static_cast<unsigned char>(v[at]))) n = n * 10 + (v[at++] - '0');
            out.push_back(n);
            if (at < v.size() && v[at] == '.') ++at;
            else break;
        }
        return out;
    };
    const std::vector<long> x = numbers(a), y = numbers(b);
    for (std::size_t i = 0; i < x.size() || i < y.size(); ++i) {
        const long p = i < x.size() ? x[i] : 0, q = i < y.size() ? y[i] : 0;
        if (p != q) return p < q ? -1 : 1;
    }
    return 0;
}

#ifdef _WIN32

namespace fs = std::filesystem;

namespace {

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

// ------------------------------------------------------------ JSON, read

struct Json {
    enum Kind { kNull, kBool, kNumber, kString, kArray, kObject } kind = kNull;
    bool flag = false;
    double number = 0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> fields;

    const Json& operator[](const char* key) const {
        static const Json none;
        for (const auto& [name, value] : fields)
            if (name == key) return value;
        return none;
    }
};

struct JsonReader {
    const std::string& s;
    std::size_t at = 0;
    bool ok = true;

    void Space() {
        while (at < s.size() && (s[at] == ' ' || s[at] == '\t' || s[at] == '\n' || s[at] == '\r')) ++at;
    }
    bool Eat(char c) {
        Space();
        if (at < s.size() && s[at] == c) return ++at, true;
        return false;
    }
    static void PutUtf8(std::string& out, std::uint32_t c) {
        if (c < 0x80) out += char(c);
        else if (c < 0x800) out += char(0xC0 | (c >> 6)), out += char(0x80 | (c & 0x3F));
        else if (c < 0x10000) out += char(0xE0 | (c >> 12)), out += char(0x80 | ((c >> 6) & 0x3F)), out += char(0x80 | (c & 0x3F));
        else out += char(0xF0 | (c >> 18)), out += char(0x80 | ((c >> 12) & 0x3F)), out += char(0x80 | ((c >> 6) & 0x3F)),
             out += char(0x80 | (c & 0x3F));
    }
    std::uint32_t Hex4() {
        if (at + 4 > s.size()) return ok = false, 0;
        const std::uint32_t v = std::uint32_t(std::strtoul(s.substr(at, 4).c_str(), nullptr, 16));
        at += 4;
        return v;
    }
    std::string String() {
        std::string out;
        if (!Eat('"')) return ok = false, out;
        while (at < s.size() && s[at] != '"') {
            char c = s[at++];
            if (c != '\\') {
                out += c;
                continue;
            }
            if (at >= s.size()) break;
            c = s[at++];
            switch (c) {
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    std::uint32_t u = Hex4();
                    if (u >= 0xD800 && u < 0xDC00 && at + 6 <= s.size() && s[at] == '\\' && s[at + 1] == 'u') {
                        at += 2;
                        const std::uint32_t low = Hex4();
                        u = 0x10000 + ((u - 0xD800) << 10) + (low - 0xDC00);
                    }
                    PutUtf8(out, u);
                    break;
                }
                default: out += c;
            }
        }
        if (at >= s.size()) return ok = false, out;
        ++at;
        return out;
    }
    Json Value(int depth = 0) {
        Json v;
        Space();
        if (at >= s.size() || depth > 32) return ok = false, v;
        const char c = s[at];
        if (c == '{') {
            ++at;
            v.kind = Json::kObject;
            if (Eat('}')) return v;
            do {
                std::string key = String();
                if (!Eat(':')) return ok = false, v;
                v.fields.emplace_back(std::move(key), Value(depth + 1));
            } while (ok && Eat(','));
            if (!Eat('}')) ok = false;
        } else if (c == '[') {
            ++at;
            v.kind = Json::kArray;
            if (Eat(']')) return v;
            do v.items.push_back(Value(depth + 1));
            while (ok && Eat(','));
            if (!Eat(']')) ok = false;
        } else if (c == '"') {
            v.kind = Json::kString;
            v.text = String();
        } else if (s.compare(at, 4, "true") == 0) {
            v.kind = Json::kBool, v.flag = true, at += 4;
        } else if (s.compare(at, 5, "false") == 0) {
            v.kind = Json::kBool, at += 5;
        } else if (s.compare(at, 4, "null") == 0) {
            at += 4;
        } else {
            char* end = nullptr;
            v.kind = Json::kNumber;
            v.number = std::strtod(s.c_str() + at, &end);
            if (end == s.c_str() + at) return ok = false, v;
            at = std::size_t(end - s.c_str());
        }
        return v;
    }
};

// ------------------------------------------------------------ HTTPS

/* A GET of `url`: its body to `sink` as it comes (false from it: stop), its
   length (0 not said) to `length` first. False with `why` on any failure.
   `token` (PS2HOST_GITHUB_TOKEN: a private repository's tester's): sent to
   api.github.com alone - its redirects followed here, the token left out
   for another host (a release asset's storage). */
bool Get(const std::string& url, const std::string& user_agent, DWORD timeout_ms, bool api,
         const std::function<void(std::uint64_t)>& length, const std::function<bool(const char*, std::size_t)>& sink,
         std::string* why, const std::string& token = "", int hops = 0) {
    const std::wstring wide = Widen(url);
    URL_COMPONENTS parts{};
    parts.dwStructSize = sizeof parts;
    wchar_t host[256] = {}, path[2048] = {};
    parts.lpszHostName = host, parts.dwHostNameLength = 256;
    parts.lpszUrlPath = path, parts.dwUrlPathLength = 2048;
    wchar_t extra[2048] = {};
    parts.lpszExtraInfo = extra, parts.dwExtraInfoLength = 2048;
    if (!WinHttpCrackUrl(wide.c_str(), 0, 0, &parts) || parts.nScheme != INTERNET_SCHEME_HTTPS)
        return *why = "not an https address: " + url, false;
    HINTERNET session = WinHttpOpen(Widen(user_agent).c_str(), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                    WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) return *why = "no WinHTTP session (" + std::to_string(GetLastError()) + ")", false;
    WinHttpSetTimeouts(session, int(timeout_ms), int(timeout_ms), int(timeout_ms), int(timeout_ms));
    bool done = false;
    HINTERNET connection = WinHttpConnect(session, host, parts.nPort, 0);
    HINTERNET request = connection ? WinHttpOpenRequest(connection, L"GET", (std::wstring(path) + extra).c_str(), nullptr,
                                                        WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                                        WINHTTP_FLAG_SECURE)
                                   : nullptr;
    std::wstring headers = api ? L"Accept: application/vnd.github+json\r\nX-GitHub-Api-Version: 2022-11-28\r\n"
                               : L"Accept: application/octet-stream\r\n";
    if (!token.empty() && std::wstring(host) == L"api.github.com") headers += L"Authorization: Bearer " + Widen(token) + L"\r\n";
    if (request && !token.empty()) {
        DWORD never = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
        WinHttpSetOption(request, WINHTTP_OPTION_REDIRECT_POLICY, &never, sizeof never);
    }
    std::string redirect;
    if (!request || !WinHttpSendRequest(request, headers.c_str(), DWORD(-1), WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(request, nullptr)) {
        *why = "no answer from " + Narrow(host) + " (" + std::to_string(GetLastError()) + ")";
    } else {
        DWORD status = 0, size = sizeof status;
        WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                            &status, &size, WINHTTP_NO_HEADER_INDEX);
        wchar_t location[4096] = {};
        DWORD location_size = sizeof location;
        if (!token.empty() && status >= 301 && status <= 308 &&
            WinHttpQueryHeaders(request, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX, location, &location_size,
                                WINHTTP_NO_HEADER_INDEX)) {
            redirect = Narrow(location);
        } else if (status != 200) {
            *why = Narrow(host) + " answered " + std::to_string(status);
        } else {
            wchar_t text[32] = {};
            DWORD text_size = sizeof text;
            const std::uint64_t total =
                WinHttpQueryHeaders(request, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX, text,
                                    &text_size, WINHTTP_NO_HEADER_INDEX)
                    ? std::wcstoull(text, nullptr, 10)
                    : 0;
            if (length) length(total);
            std::vector<char> buffer(1 << 16);
            done = true;
            for (;;) {
                DWORD read = 0;
                if (!WinHttpReadData(request, buffer.data(), DWORD(buffer.size()), &read)) {
                    *why = "the download stopped (" + std::to_string(GetLastError()) + ")";
                    done = false;
                    break;
                }
                if (read == 0) break;
                if (!sink(buffer.data(), read)) {
                    *why = "stopped";
                    done = false;
                    break;
                }
            }
        }
    }
    if (request) WinHttpCloseHandle(request);
    if (connection) WinHttpCloseHandle(connection);
    WinHttpCloseHandle(session);
    if (!redirect.empty()) {
        if (hops >= 5) return *why = "too many redirects", false;
        return Get(redirect, user_agent, timeout_ms, api, length, sink, why, token, hops + 1);
    }
    return done;
}

/* A private repository's tester's token (PS2HOST_GITHUB_TOKEN), "" none - a player's run needs none. */
std::string Token() {
    const char* v = std::getenv("PS2HOST_GITHUB_TOKEN");
    return v ? v : "";
}

// ------------------------------------------------------------ SHA-256

struct Sha256 {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    Sha256() {
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) == 0)
            BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0);
    }
    ~Sha256() {
        if (hash) BCryptDestroyHash(hash);
        if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    }
    void Add(const char* data, std::size_t size) {
        if (hash) BCryptHashData(hash, reinterpret_cast<PUCHAR>(const_cast<char*>(data)), ULONG(size), 0);
    }
    std::string Hex() {
        unsigned char out[32] = {};
        if (!hash || BCryptFinishHash(hash, out, sizeof out, 0) != 0) return "";
        std::string s;
        char two[3];
        for (unsigned char b : out) std::snprintf(two, sizeof two, "%02x", b), s += two;
        return s;
    }
};

// ------------------------------------------------------------ the progress window

struct ProgressWindow {
    HWND window = nullptr, label = nullptr;
    static inline bool closed = false;

    static LRESULT CALLBACK Proc(HWND w, UINT message, WPARAM wp, LPARAM lp) {
        if (message == WM_CLOSE) {
            closed = true;          // (the download given up: the game starts as it is)
            return 0;
        }
        return DefWindowProcW(w, message, wp, lp);
    }

    ProgressWindow(const std::string& title) {
        closed = false;
        WNDCLASSW kind{};
        kind.lpfnWndProc = Proc;
        kind.hInstance = GetModuleHandleW(nullptr);
        kind.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));    // (IDC_ARROW)
        kind.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        kind.lpszClassName = L"Ps2PortUpdate";
        kind.hIcon = LoadIconW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(1));
        RegisterClassW(&kind);
        const int width = 440, height = 130;
        const int x = (GetSystemMetrics(SM_CXSCREEN) - width) / 2, y = (GetSystemMetrics(SM_CYSCREEN) - height) / 2;
        window = CreateWindowExW(WS_EX_TOPMOST, L"Ps2PortUpdate", Widen(title).c_str(),
                                 WS_POPUP | WS_CAPTION | WS_SYSMENU, x, y, width, height, nullptr, nullptr,
                                 kind.hInstance, nullptr);
        if (!window) return;
        RECT inside{};
        GetClientRect(window, &inside);
        label = CreateWindowExW(0, L"STATIC", L"", WS_CHILD | WS_VISIBLE | SS_CENTER, 12, 24, inside.right - 24, 48,
                                window, nullptr, kind.hInstance, nullptr);
        SendMessageW(label, WM_SETFONT, WPARAM(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
        ShowWindow(window, SW_SHOWNORMAL);
        Pump();
    }
    ~ProgressWindow() {
        if (window) DestroyWindow(window);
        Pump();
    }
    void Say(const std::string& text) {
        if (label) SetWindowTextW(label, Widen(text).c_str());
        Pump();
    }
    static void Pump() {
        MSG message;
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
};

// ------------------------------------------------------------ the update

struct Release {
    std::string version, page, notes, zip_url, sha256;
    std::string asset_api;      // the asset's API address (a token's download: a private repository's)
    std::uint64_t size = 0;
};

std::string UserAgent(const UpdateGame& game) { return game.target + "/" + game.version + " (PS2 port updater)"; }

/* The newest release with this game's zip, newer than `current` (false: none, or no answer). */
bool NewestRelease(const UpdateGame& game, const std::string& current, Release* best, std::string* why) {
    std::string body;
    const bool got = Get("https://api.github.com/repos/" + game.github + "/releases?per_page=20", UserAgent(game), 4000,
                         true, nullptr, [&](const char* d, std::size_t n) { return body.append(d, n), body.size() < (8u << 20); },
                         why, Token());
    if (!got) return false;
    JsonReader reader{body};
    const Json releases = reader.Value();
    if (!reader.ok || releases.kind != Json::kArray) return *why = "GitHub's answer was not a list of releases", false;
    bool found = false;
    for (const Json& r : releases.items) {
        if (r["draft"].flag || r["tag_name"].kind != Json::kString) continue;
        std::string version = r["tag_name"].text;
        if (!version.empty() && (version[0] == 'v' || version[0] == 'V')) version.erase(0, 1);
        if (CompareVersions(version, current) <= 0 || (found && CompareVersions(version, best->version) <= 0)) continue;
        const std::string want = game.release_name + "-" + version + "-windows-x64.zip";
        for (const Json& a : r["assets"].items) {
            if (a["name"].text != want || a["browser_download_url"].kind != Json::kString) continue;
            best->version = version;
            best->page = r["html_url"].text;
            best->notes = r["body"].text;
            best->zip_url = a["browser_download_url"].text;
            best->asset_api = a["url"].text;
            best->size = std::uint64_t(a["size"].number);
            best->sha256 = a["digest"].text.rfind("sha256:", 0) == 0 ? a["digest"].text.substr(7) : "";
            found = true;
        }
    }
    if (!found) *why = "none newer than " + current;
    return found;
}

/* What is new, for the question: the release notes' "### " headings. */
std::string Highlights(const std::string& notes) {
    std::string out;
    int count = 0;
    std::size_t at = 0;
    while (at < notes.size() && count < 6) {
        std::size_t end = notes.find('\n', at);
        if (end == std::string::npos) end = notes.size();
        std::string line = notes.substr(at, end - at);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("### ", 0) == 0) out += "  - " + line.substr(4) + "\n", ++count;
        at = end + 1;
    }
    return out;
}

enum Choice { kUpdate, kPage, kContinue };

Choice Ask(const UpdateGame& game, const std::string& current, const Release& r) {
    if (const char* v = std::getenv("PS2HOST_TEST_UPDATE")) {
        const std::string s = v;
        std::fprintf(stderr, "[update] a test's answer: %s\n", v);
        return s == "update" ? kUpdate : s == "page" ? kPage : kContinue;
    }
    const std::string title = game.title + " - update";
    std::string text = game.title + " " + r.version + " is available. You have " + current + ".\n\n";
    const std::string news = Highlights(r.notes);
    if (!news.empty()) text += "New in it:\n" + news + "\n";
    text += "Updating keeps your memory card, settings and captures; the game starts again when it is done.";
    const SDL_MessageBoxButtonData buttons[] = {
        {SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT, kUpdate, "Update now"},
        {0, kPage, "Open the release page"},
        {SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT, kContinue, "Continue without updating"},
    };
    const SDL_MessageBoxData box{SDL_MESSAGEBOX_INFORMATION | SDL_MESSAGEBOX_BUTTONS_LEFT_TO_RIGHT, nullptr,
                                 title.c_str(), text.c_str(), 3, buttons, nullptr};
    int chosen = kContinue;
    Raise(title);
    if (!SDL_ShowMessageBox(&box, &chosen)) return kContinue;
    return Choice(chosen);
}

void OpenPage(const std::string& url) {
    ShellExecuteW(nullptr, L"open", Widen(url).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

/* A program run to its end, its window hidden: its exit code (-1 not started). */
int RunHidden(const std::wstring& command, DWORD timeout_ms) {
    std::wstring line = command;
    STARTUPINFOW startup{};
    startup.cb = sizeof startup;
    PROCESS_INFORMATION started{};
    if (!CreateProcessW(nullptr, line.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                        &started))
        return -1;
    CloseHandle(started.hThread);
    DWORD code = DWORD(-1);
    if (WaitForSingleObject(started.hProcess, timeout_ms) == WAIT_OBJECT_0) GetExitCodeProcess(started.hProcess, &code);
    else TerminateProcess(started.hProcess, 1);
    CloseHandle(started.hProcess);
    return int(code);
}

/* The release's files over the game's: an executable or library renamed
   .old first (one running can be renamed, not written). Undone on failure. */
bool Install(const fs::path& from, const fs::path& folder, std::string* why) {
    std::error_code error;
    std::vector<std::pair<fs::path, fs::path>> moved;      // (in place, .old)
    std::vector<fs::path> written;
    bool ok = true;
    for (const auto& e : fs::recursive_directory_iterator(from, error)) {
        if (!e.is_regular_file(error)) continue;
        const fs::path dest = folder / fs::relative(e.path(), from, error);
        fs::create_directories(dest.parent_path(), error);
        std::wstring ext = dest.extension().wstring();
        for (wchar_t& c : ext) c = wchar_t(towlower(c));
        if ((ext == L".exe" || ext == L".dll") && fs::exists(dest, error)) {
            fs::path old = dest;
            old += L".old";
            DeleteFileW(old.wstring().c_str());
            if (!MoveFileExW(dest.wstring().c_str(), old.wstring().c_str(), MOVEFILE_REPLACE_EXISTING)) {
                *why = "could not move " + Narrow(dest.filename().wstring()) + " aside (" + std::to_string(GetLastError()) + ")";
                ok = false;
                break;
            }
            moved.emplace_back(dest, old);
        }
        if (!CopyFileW(e.path().wstring().c_str(), dest.wstring().c_str(), FALSE)) {
            *why = "could not write " + Narrow(dest.filename().wstring()) + " (" + std::to_string(GetLastError()) + ")";
            ok = false;
            break;
        }
        written.push_back(dest);
    }
    if (error && ok) *why = "could not read the update (" + error.message() + ")", ok = false;
    if (!ok)
        for (auto it = moved.rbegin(); it != moved.rend(); ++it)
            MoveFileExW(it->second.wstring().c_str(), it->first.wstring().c_str(), MOVEFILE_REPLACE_EXISTING);
    return ok;
}

/* Download, check, unpack, put in place, start. False with `why`. */
bool Update(const fs::path& folder, const UpdateGame& game, const Release& r, std::string* why) {
    std::error_code error;
    const fs::path work = folder / "update";
    fs::remove_all(work, error);
    if (!fs::create_directories(work, error)) return *why = "the game's folder cannot be written (" + error.message() + ")", false;
    const fs::path zip = work / Widen(game.release_name + "-" + r.version + "-windows-x64.zip");
    {
        ProgressWindow window(game.title + " - update");
        window.Say("Downloading " + r.version + "...");
        std::FILE* f = _wfopen(zip.wstring().c_str(), L"wb");
        if (!f) return *why = "could not write the download", false;
        Sha256 hash;
        std::uint64_t total = r.size, got = 0;
        DWORD last_said = 0;
        const std::string token = Token();
        const bool ok = Get(!token.empty() && !r.asset_api.empty() ? r.asset_api : r.zip_url, UserAgent(game), 20000, false,
                            [&](std::uint64_t length) { if (length) total = length; },
                            [&](const char* d, std::size_t n) {
                                if (std::fwrite(d, 1, n, f) != n) return false;
                                hash.Add(d, n);
                                got += n;
                                if (GetTickCount() - last_said > 100) {
                                    last_said = GetTickCount();
                                    char text[128];
                                    std::snprintf(text, sizeof text, "Downloading %s...\n%.1f of %.1f MB",
                                                  r.version.c_str(), double(got) / 1e6, double(total) / 1e6);
                                    window.Say(text);
                                }
                                return !ProgressWindow::closed;
                            },
                            why, token);
        std::fclose(f);
        if (ProgressWindow::closed) return *why = "the download was stopped", false;
        if (!ok) return false;
        if (r.size && got != r.size) return *why = "the download is " + std::to_string(got) + " bytes, not " + std::to_string(r.size), false;
        const std::string sum = hash.Hex();
        if (!r.sha256.empty() && sum != r.sha256) return *why = "the download's SHA-256 is not the release's", false;
        std::fprintf(stderr, "[update] downloaded %llu bytes, SHA-256 %s%s\n", (unsigned long long)got, sum.c_str(),
                     r.sha256.empty() ? " (GitHub gave none to check)" : " (as GitHub's)");
        window.Say("Unpacking " + r.version + "...");
        wchar_t system[MAX_PATH] = {};
        GetSystemDirectoryW(system, MAX_PATH);
        const fs::path tar = fs::path(system) / L"tar.exe";
        if (!fs::exists(tar, error)) return *why = "Windows' tar.exe is missing (Windows 10 1803 or later has it)", false;
        const fs::path unpacked = work / L"new";
        fs::create_directories(unpacked, error);
        const int code = RunHidden(L"\"" + tar.wstring() + L"\" -xf \"" + zip.wstring() + L"\" -C \"" + unpacked.wstring() + L"\"", 120000);
        if (code != 0) return *why = "could not unpack the download (tar: " + std::to_string(code) + ")", false;
        // The zip's one folder (<release name>-<version>/), or its files.
        fs::path top = unpacked;
        int dirs = 0, files = 0;
        for (const auto& e : fs::directory_iterator(unpacked, error)) (e.is_directory(error) ? dirs : files)++, top = e.path();
        if (!(dirs == 1 && files == 0)) top = unpacked;
        if (!fs::exists(top / Widen(game.target + ".exe"), error)) return *why = "the download has no " + game.target + ".exe", false;
        window.Say("Installing " + r.version + "...");
        if (!Install(top, folder, why)) return false;
    }
    fs::remove_all(work, error);
    // The new version started (its own processor check picks its build).
    std::wstring line = L"\"" + (folder / Widen(game.target + ".exe")).wstring() + L"\"";
    STARTUPINFOW startup{};
    startup.cb = sizeof startup;
    PROCESS_INFORMATION started{};
    if (!CreateProcessW(nullptr, line.data(), nullptr, nullptr, FALSE, 0, nullptr, folder.wstring().c_str(), &startup,
                        &started))
        return *why = "installed, but could not start it (" + std::to_string(GetLastError()) + "): start the game again", false;
    CloseHandle(started.hThread);
    CloseHandle(started.hProcess);
    return true;
}

/* A last update's leftovers: the executables it moved aside (beside the
   game: the release's are there), its folder. */
void Tidy(const fs::path& folder) {
    std::error_code error;
    for (const auto& e : fs::directory_iterator(folder, error)) {
        if (!e.is_regular_file(error) || e.path().extension() != L".old") continue;
        std::error_code gone;
        if (fs::remove(e.path(), gone)) std::fprintf(stderr, "[update] removed %s\n", Narrow(e.path().filename().wstring()).c_str());
    }
    fs::remove_all(folder / "update", error);
}

}  // namespace

constexpr std::chrono::seconds kCheckS{4};       // the start held for the check at most

bool CheckForUpdate(const fs::path& folder, const UpdateGame& game) {
    if (game.github.empty() || game.release_name.empty()) return false;
    Tidy(folder);
    if (!SettingOn("update_check", true)) {
        std::fprintf(stderr, "[update] the check is off (update_check=0)\n");
        return false;
    }
    const char* test = std::getenv("PS2HOST_TEST_VERSION");
    const std::string current = test ? test : game.version;
    // GitHub asked on a thread of its own, the start waiting kCheckS at most
    // (WinHTTP's timeouts are each phase's: a network that drops packets
    // would add them up). Late, the thread is left to end by itself.
    struct Check {
        std::mutex lock;
        std::condition_variable said;
        bool done = false, found = false;
        Release release;
        std::string why;
    };
    const auto check = std::make_shared<Check>();
    std::thread([check, game, current] {
        if (const char* v = std::getenv("PS2HOST_TEST_UPDATE_DELAY")) Sleep(DWORD(std::atoi(v)));
        Release found;
        std::string why;
        const bool ok = NewestRelease(game, current, &found, &why);
        std::lock_guard<std::mutex> hold(check->lock);
        check->done = true, check->found = ok, check->release = found, check->why = why;
        check->said.notify_all();
    }).detach();
    Release r;
    std::string why;
    {
        std::unique_lock<std::mutex> hold(check->lock);
        if (!check->said.wait_for(hold, kCheckS, [&] { return check->done; })) {
            std::fprintf(stderr, "[update] no update: GitHub did not answer within %lld s\n", (long long)kCheckS.count());
            return false;
        }
        if (!check->found) {
            std::fprintf(stderr, "[update] no update: %s\n", check->why.c_str());
            return false;
        }
        r = check->release;
    }
    std::fprintf(stderr, "[update] %s is available (this is %s): %s\n", r.version.c_str(), current.c_str(), r.page.c_str());
    switch (Ask(game, current, r)) {
        case kPage:
            OpenPage(r.page);
            return false;
        case kContinue:
            std::fprintf(stderr, "[update] the player goes on with %s\n", current.c_str());
            return false;
        case kUpdate:
            break;
    }
    if (Update(folder, game, r, &why)) {
        std::fprintf(stderr, "[update] %s installed and started\n", r.version.c_str());
        return true;
    }
    std::fprintf(stderr, "[update] the update failed: %s\n", why.c_str());
    std::error_code error;
    fs::remove_all(folder / "update", error);
    if (std::getenv("PS2HOST_TEST_UPDATE")) return false;
    const std::string title = game.title + " - update";
    const std::string text = "The update could not be installed:\n" + why +
                             "\n\nThe game starts as it is. The release page has the new version to download.";
    const SDL_MessageBoxButtonData buttons[] = {{0, kPage, "Open the release page"},
                                                {SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT | SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT,
                                                 kContinue, "Continue"}};
    const SDL_MessageBoxData box{SDL_MESSAGEBOX_WARNING | SDL_MESSAGEBOX_BUTTONS_LEFT_TO_RIGHT, nullptr, title.c_str(),
                                 text.c_str(), 2, buttons, nullptr};
    int chosen = kContinue;
    Raise(title);
    if (SDL_ShowMessageBox(&box, &chosen) && chosen == kPage) OpenPage(r.page);
    return false;
}

#else

bool CheckForUpdate(const std::filesystem::path&, const UpdateGame&) { return false; }

#endif

}  // namespace ps2host
