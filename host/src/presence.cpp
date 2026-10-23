/*
 * Discord's Rich Presence (ps2host/presence.h), through the Discord app's
 * own local pipe - its RPC protocol, as Discord's SDKs speak it:
 *
 *   - a frame is an op code (u32), the JSON's length (u32), the JSON;
 *   - op 0 the handshake ({"v":1,"client_id":...}), answered by op 1 with
 *     "evt":"READY" (or op 2, closed: a wrong Application ID);
 *   - op 1 a command: SET_ACTIVITY with this process's id and the activity
 *     (null: none), answered with the same nonce;
 *   - op 3 a ping (answered by op 4 with its JSON), op 2 Discord closing.
 *
 * Everything on one thread: connecting (each pipe -0 to -9 tried, again
 * every few seconds while none answers), the latest activity sent once it
 * differs from the last sent and a few seconds after it (Discord takes
 * five updates in twenty seconds), the answers read as they come.
 */

#include "ps2host/presence.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace ps2host {
namespace {

using Clock = std::chrono::steady_clock;

constexpr auto kRetry = std::chrono::seconds(5);        // between attempts to connect
constexpr auto kRefused = std::chrono::seconds(60);     // after Discord refused the handshake
constexpr auto kBetweenUpdates = std::chrono::seconds(4);
constexpr auto kAnswerWait = std::chrono::seconds(3);   // the handshake's answer
constexpr std::size_t kMaxChars = 127;                  // Discord's limit for a line

enum Op : std::uint32_t { kHandshake = 0, kFrame = 1, kClose = 2, kPing = 3, kPong = 4 };

/* Up to `limit` characters (whole UTF-8 sequences), JSON-escaped. */
std::string Json(const std::string& s, std::size_t limit = kMaxChars) {
    std::string out = "\"";
    std::size_t chars = 0;
    for (std::size_t i = 0; i < s.size() && chars < limit; ++chars) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        const std::size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
        if (c == '"' || c == '\\') {
            out += '\\';
            out += char(c);
        } else if (c < 0x20) {
            char esc[8];
            std::snprintf(esc, sizeof esc, "\\u%04x", c);
            out += esc;
        } else {
            out.append(s, i, len);
        }
        i += len;
    }
    return out + "\"";
}

std::string ActivityJson(const PresenceActivity& a) {
    std::string j = "{";
    auto field = [&j](const char* key, const std::string& value) {
        if (value.empty()) return;
        if (j.size() > 1) j += ",";
        j += "\"";
        j += key;
        j += "\":" + value;
    };
    // Discord wants two characters at least in a line.
    auto line = [](const std::string& s) { return s.size() >= 2 ? Json(s) : s.empty() ? "" : Json(s + " "); };
    field("details", line(a.details));
    field("state", line(a.state));
    if (a.start > 0) field("timestamps", "{\"start\":" + std::to_string(a.start) + "}");
    std::string assets = "{";
    auto asset = [&assets](const char* key, const std::string& value) {
        if (value.empty()) return;
        if (assets.size() > 1) assets += ",";
        assets += "\"" + std::string(key) + "\":" + Json(value, 255);
    };
    asset("large_image", a.large_image);
    asset("large_text", a.large_text);
    asset("small_image", a.small_image);
    asset("small_text", a.small_text);
    if (assets.size() > 1) field("assets", assets + "}");
    if (!a.button_label.empty() && !a.button_url.empty())
        field("buttons", "[{\"label\":" + Json(a.button_label, 31) + ",\"url\":" + Json(a.button_url, 512) + "}]");
    return j + "}";
}

class Discord {
public:
    ~Discord() { Stop(); }

    void Start(const std::string& id) {
        Stop();
        std::lock_guard<std::mutex> lock(mutex_);
        id_ = id;
        quit_ = false;
        thread_ = std::thread([this] { Run(); });
    }
    void Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!thread_.joinable()) return;
            quit_ = true;
        }
        wake_.notify_all();
        thread_.join();
    }
    bool Running() {
        std::lock_guard<std::mutex> lock(mutex_);
        return thread_.joinable();
    }
    void Set(const PresenceActivity* a) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const bool shown = a != nullptr;
            if (shown == shown_ && (!shown || *a == wanted_)) return;
            shown_ = shown;
            if (a) wanted_ = *a;
            ++generation_;
        }
        wake_.notify_all();
    }
    bool Connected() const { return connected_; }

private:
    void Run() {
        Clock::time_point next_try{};
        Clock::time_point last_sent{};
        unsigned sent = 0;
        std::uint64_t nonce = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (quit_) break;
            const std::string id = id_;
            const unsigned generation = generation_;
            const bool shown = shown_;
            const PresenceActivity wanted = wanted_;
            lock.unlock();

            const Clock::time_point now = Clock::now();
            if (pipe_ == INVALID_HANDLE_VALUE && now >= next_try) {
                const int answer = Connect(id);
                next_try = now + (answer < 0 ? kRefused : kRetry);
                if (answer > 0) sent = generation - 1;      // what was wanted, sent again
            }
            if (pipe_ != INVALID_HANDLE_VALUE) {
                if (!Drain()) Disconnect("closed");
            }
            if (pipe_ != INVALID_HANDLE_VALUE && sent != generation && now - last_sent >= kBetweenUpdates) {
                const std::string args = "{\"pid\":" + std::to_string(GetCurrentProcessId()) +
                                         ",\"activity\":" + (shown ? ActivityJson(wanted) : "null") + "}";
                const std::string json = "{\"cmd\":\"SET_ACTIVITY\",\"args\":" + args + ",\"nonce\":\"" +
                                         std::to_string(++nonce) + "\"}";
                if (Send(kFrame, json)) {
                    sent = generation;
                    last_sent = now;
                } else {
                    Disconnect("could not be written to");
                }
            }

            lock.lock();
            if (quit_) break;
            wake_.wait_for(lock, std::chrono::milliseconds(500));
        }
        if (pipe_ != INVALID_HANDLE_VALUE) Disconnect(nullptr);
    }

    /* 1: connected; 0: no Discord (or no answer); -1: Discord refused us. */
    int Connect(const std::string& id) {
        for (int n = 0; n < 10; ++n) {
            const std::string name = "\\\\.\\pipe\\discord-ipc-" + std::to_string(n);
            HANDLE pipe = CreateFileA(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
            if (pipe == INVALID_HANDLE_VALUE) continue;
            pipe_ = pipe;
            buffer_.clear();
            if (!Send(kHandshake, "{\"v\":1,\"client_id\":" + Json(id, 64) + "}")) {
                Disconnect(nullptr);
                continue;
            }
            // Its answer: READY, or closed (the reason said).
            const Clock::time_point until = Clock::now() + kAnswerWait;
            std::uint32_t op = 0;
            std::string json;
            while (Clock::now() < until) {
                if (!Fill()) break;
                if (Take(&op, &json)) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (op == kFrame && json.find("\"READY\"") != std::string::npos) {
                connected_ = true;
                std::fprintf(stderr, "[presence] Discord connected (discord-ipc-%d)\n", n);
                return 1;
            }
            std::fprintf(stderr, "[presence] Discord did not take the handshake: %s\n",
                         json.empty() ? "no answer" : json.substr(0, 200).c_str());
            Disconnect(nullptr);
            return op == kClose ? -1 : 0;
        }
        return 0;
    }

    void Disconnect(const char* why) {
        if (why) std::fprintf(stderr, "[presence] Discord's pipe %s: trying again\n", why);
        CloseHandle(pipe_);
        pipe_ = INVALID_HANDLE_VALUE;
        connected_ = false;
        buffer_.clear();
    }

    bool Send(std::uint32_t op, const std::string& json) {
        std::vector<char> frame(8 + json.size());
        const std::uint32_t len = std::uint32_t(json.size());
        std::memcpy(frame.data(), &op, 4);
        std::memcpy(frame.data() + 4, &len, 4);
        std::memcpy(frame.data() + 8, json.data(), json.size());
        DWORD written = 0;
        return WriteFile(pipe_, frame.data(), DWORD(frame.size()), &written, nullptr) && written == frame.size();
    }

    /* What has come in, added to the buffer (without waiting); false when
       the pipe is gone. */
    bool Fill() {
        DWORD available = 0;
        if (!PeekNamedPipe(pipe_, nullptr, 0, nullptr, &available, nullptr)) return false;
        if (!available) return true;
        const std::size_t at = buffer_.size();
        buffer_.resize(at + available);
        DWORD read = 0;
        if (!ReadFile(pipe_, buffer_.data() + at, available, &read, nullptr)) return false;
        buffer_.resize(at + read);
        return true;
    }

    /* A whole frame off the buffer's front. */
    bool Take(std::uint32_t* op, std::string* json) {
        if (buffer_.size() < 8) return false;
        std::uint32_t len = 0;
        std::memcpy(op, buffer_.data(), 4);
        std::memcpy(&len, buffer_.data() + 4, 4);
        if (buffer_.size() < 8 + std::size_t(len)) return false;
        json->assign(buffer_.data() + 8, len);
        buffer_.erase(buffer_.begin(), buffer_.begin() + 8 + len);
        return true;
    }

    /* The answers read: pings answered, errors said; false once Discord has
       closed. */
    bool Drain() {
        if (!Fill()) return false;
        std::uint32_t op = 0;
        std::string json;
        while (Take(&op, &json)) {
            if (op == kClose) return false;
            if (op == kPing && !Send(kPong, json)) return false;
            if (op == kFrame && json.find("\"evt\":\"ERROR\"") != std::string::npos)
                std::fprintf(stderr, "[presence] Discord: %s\n", json.substr(0, 300).c_str());
        }
        return true;
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread thread_;
    bool quit_ = false;
    std::string id_;
    PresenceActivity wanted_;
    bool shown_ = false;
    unsigned generation_ = 0;

    // The thread's own.
    HANDLE pipe_ = INVALID_HANDLE_VALUE;
    std::vector<char> buffer_;
    std::atomic<bool> connected_{false};
};

Discord& TheDiscord() {
    static Discord discord;
    return discord;
}

}  // namespace

void PresenceStart(const std::string& application_id) {
    if (application_id.empty()) return;
    TheDiscord().Start(application_id);
}
void PresenceStop() { TheDiscord().Stop(); }
bool PresenceRunning() { return TheDiscord().Running(); }
void PresenceSet(const PresenceActivity& activity) { TheDiscord().Set(&activity); }
void PresenceClear() { TheDiscord().Set(nullptr); }
bool PresenceConnected() { return TheDiscord().Connected(); }

long long PresenceNow() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace ps2host
