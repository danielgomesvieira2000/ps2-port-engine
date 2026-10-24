/*
 * The PC settings panel: F9 (or L3 + R3 on a controller) over any game.
 *
 * A port can put its PC settings in the game's own menus (Burnout 3's main
 * menu has PC SETTINGS); this panel is the engine's, for every port: the
 * settings the host itself applies, changed while the game runs and saved
 * with the PC settings (settings.h). While it is open the game gets no input.
 *
 * A port's own settings (AddPortOptions) follow as pages, each row opening
 * one: its options, then Back. A port without controls of its own (no
 * SetKeyboardMapper) has the Keyboard page: each of the DualShock 2's
 * inputs and the key that holds it (bindings.h's GameKeys) - Enter, then the
 * key to bind (Esc leaves it as it was).
 *
 *   Up / Down          choose
 *   Left / Right       change (Enter or Cross too; on a page's row, open it)
 *   Esc, Circle        back from a page; close
 *   F9                 close
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "ps2host/bindings.h"
#include "ps2host/high_fps.h"
#include "ps2host/run.h"
#include "ps2host/settings.h"
#include "ps2host/texture_pack.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace ps2host {

namespace {

bool g_open = false;
bool g_releasing = false;                // closed, the game's input held back until the keys are let go
int g_row = 0;
int g_page = -1;                          // a port's page open (Pages()' index), -1 the first, kKeyboard
constexpr int kKeyboard = -2;             // the Keyboard page (GameKeys)
int g_binding = -1;                       // the input a key is awaited for (the Keyboard page), or -1
bool g_binding_armed = false;             // ...every key let go since it began

/* The Keyboard page's rows, in the order a player thinks of them. */
constexpr int kKeyRows[] = {bindings::kUp,     bindings::kDown, bindings::kLeft,   bindings::kRight,
                            bindings::kCross,  bindings::kCircle, bindings::kSquare, bindings::kTriangle,
                            bindings::kL1,     bindings::kR1,   bindings::kL2,     bindings::kR2,
                            bindings::kL3,     bindings::kR3,   bindings::kStart,  bindings::kSelect};

std::vector<PortOption>& PortOptions() {
    static std::vector<PortOption> options;
    return options;
}

/* The ports' pages, in the order first named. */
std::vector<std::string> Pages() {
    std::vector<std::string> pages;
    for (const PortOption& o : PortOptions())
        if (std::find(pages.begin(), pages.end(), o.page) == pages.end()) pages.push_back(o.page);
    return pages;
}

/* A row shown: the engine's setting, a page's link, a port's option, or the
   last (Close; Back on a page). */
struct Row {
    enum Kind { kChoice, kPage, kOption, kKeyLink, kKey, kLast } kind;
    int index;
};

struct Choice {
    const char* label;
    std::vector<std::string> values;        // as the setting keeps them
    std::vector<std::string> shown;         // as the panel says them
    std::string (*get)();
    void (*set)(const std::string&);
    const char* game_label;                 // short and in capitals, for a game's own menu (PcSettings)
    std::vector<std::string> game_values;
    const char* help;
};

std::string OnOff(bool on) { return on ? "on" : "off"; }

const std::vector<Choice>& Choices() {
    static const std::vector<Choice> choices = {
        {"Resolution", {"original", "2x", "3x", "4x", "display"},
         {"Original (640x448)", "2x", "3x", "4x", "The screen's"},
         [] { return Setting("resolution", "original"); },
         [](const std::string& v) { SetSetting("resolution", v); },
         "RESOLUTION", {"1X", "2X", "3X", "4X", "AUTO"}, "The resolution the game is drawn at."},
        {"Aspect ratio", {"4:3", "16:9", "auto"}, {"4:3", "16:9", "Fill the window"},
         [] { return Setting("aspect", "4:3"); },
         [](const std::string& v) { SetWindowAspect(v); },
         "ASPECT", {"4:3", "16:9", "FILL"}, "The picture's shape: 4:3, 16:9 or the window's."},
        {"Fullscreen", {"off", "on"}, {"Off", "On"},
         [] { return OnOff(WindowFullscreen()); },
         [](const std::string& v) { SetWindowFullscreen(v == "on"); },
         "FULLSCREEN", {"OFF", "ON"}, "Play in a window or on the whole screen."},
        {"V-sync", {"off", "on"}, {"Off", "On"},
         [] { return OnOff(WindowVsync()); },
         [](const std::string& v) { SetWindowVsync(v == "on"); },
         "V-SYNC", {"OFF", "ON"}, "Wait for the screen's refresh: no tearing."},
        {"Frame pacing", {"on", "off"}, {"On", "Off"},
         [] { return OnOff(SettingOn("frame_pacing", FramePacingDefault())); },
         [](const std::string& v) { SetSettingOn("frame_pacing", v == "on"); },
         "PACING", {"ON", "OFF"}, "Frame pacing: pictures shown at an even pace."},
        // FRAME RATE (high_fps.h): a game at 30 pictures a second has 30
        // among its rates; one at 60 does not (the row its own, below).
        {"Frame rate", {"30", "60", "120", "display"}, {"30 (the game's)", "60", "120", "The screen's"},
         [] { return Setting("frame_rate", "30"); },
         [](const std::string& v) { SetSetting("frame_rate", v); },
         "FRAME RATE", {"30", "60", "120", "DISPLAY"}, "Pictures a second; the game still steps at its own pace."},
        {"Frame rate ", {"60", "120", "display"}, {"60 (the game's)", "120", "The screen's"},
         [] { return Setting("frame_rate", "60"); },
         [](const std::string& v) { SetSetting("frame_rate", v); },
         "FRAME RATE", {"60", "120", "DISPLAY"}, "Pictures a second; the game still steps at its own pace."},
        {"Texture pack", {"on", "off"}, {"On", "Off"},
         [] { return OnOff(texture_pack::Enabled()); },
         [](const std::string& v) {
             texture_pack::SetEnabled(v == "on");
             SetSettingOn("texture_pack", v == "on");
         },
         "TEXTURES", {"ON", "OFF"}, "The texture pack's textures, or the game's own."},
    };
    return choices;
}

// The engine's rows shown: a texture pack's only when one is loaded.
std::vector<int> Rows() {
    std::vector<int> rows;
    const auto& c = Choices();
    for (int i = 0; i < int(c.size()); ++i)
    {
        const std::string label = c[i].label;
        // FRAME RATE: the row for the game's own rate (30 or 60), when it has one.
        if (label == "Frame rate" && HighFpsOwnRate() != 30.0) continue;
        if (label == "Frame rate " && HighFpsOwnRate() != 60.0) continue;
        if (label != "Texture pack" || texture_pack::Count()) rows.push_back(i);
    }
    return rows;
}

/* What the panel shows now: the first page's rows and the ports' pages, or
   an open page's options. */
std::vector<Row> Shown() {
    std::vector<Row> rows;
    if (g_page < 0) {
        for (int i : Rows()) rows.push_back({Row::kChoice, i});
        const int pages = int(Pages().size());
        for (int p = 0; p < pages; ++p) rows.push_back({Row::kPage, p});
        if (!bindings::PortMapsKeyboard()) rows.push_back({Row::kKeyLink, 0});
    } else if (g_page == kKeyboard) {
        for (int input : kKeyRows) rows.push_back({Row::kKey, input});
    } else {
        const std::string page = Pages()[std::size_t(g_page)];
        const auto& options = PortOptions();
        for (int i = 0; i < int(options.size()); ++i)
            if (options[std::size_t(i)].page == page) rows.push_back({Row::kOption, i});
    }
    rows.push_back({Row::kLast, -1});
    return rows;
}

std::string Padded(const std::string& label) {
    return label + std::string(std::size_t(20) - std::min<std::size_t>(19, label.size()), ' ');
}

void Draw(bool force = false) {
    std::vector<std::string> lines = {g_page == kKeyboard ? "PC SETTINGS > Keyboard   (F9 closes)"
                                      : g_page < 0           ? "PC SETTINGS   (F9 closes)"
                                      : "PC SETTINGS > " + Pages()[std::size_t(g_page)] + "   (F9 closes)"};
    const std::vector<Row> rows = Shown();
    std::string help;
    for (int r = 0; r < int(rows.size()); ++r) {
        std::string text;
        const Row& row = rows[std::size_t(r)];
        if (row.kind == Row::kLast) {
            text = g_page < 0 ? "Close" : "Back";
        } else if (row.kind == Row::kPage) {
            text = Pages()[std::size_t(row.index)] + "  >";
        } else if (row.kind == Row::kKeyLink) {
            text = "Keyboard  >";
        } else if (row.kind == Row::kKey) {
            const int vk = bindings::GameKeys().physical[row.index];
            const char* key = vk ? bindings::KeyName(vk) : nullptr;
            text = Padded(bindings::ButtonName(row.index, bindings::Style::kPlayStation)) +
                   (g_binding == row.index ? std::string("< press a key, Esc leaves it >")
                                           : "< " + std::string(key ? key : "NONE") + " >");
            if (r == g_row) help = "Enter, then the key for it";
        } else if (row.kind == Row::kOption) {
            const PortOption& o = PortOptions()[std::size_t(row.index)];
            const int now = o.now ? o.now() : 0;
            const std::string shown = now >= 0 && now < int(o.values.size()) ? o.values[std::size_t(now)] : "?";
            text = Padded(o.name) + "< " + shown + " >";
            if (r == g_row) help = o.help;
        } else {
            const Choice& c = Choices()[std::size_t(row.index)];
            const std::string now = c.get();
            std::string shown = now;
            for (std::size_t i = 0; i < c.values.size(); ++i)
                if (c.values[i] == now) shown = c.shown[i];
            text = std::string(c.label) + std::string(16 - std::min<std::size_t>(15, std::string(c.label).size()), ' ') +
                   "< " + shown + " >";
        }
        lines.push_back((r == g_row ? "> " : "  ") + text);
    }
    lines.push_back("");
    lines.push_back(help.empty() ? "Up/Down choose, Left/Right change" : help);
    static std::vector<std::string> shown;
    if (force || lines != shown) SetWindowOverlay(lines);     // drawn again only when it changes
    shown = lines;
}

/* Back from a page to the first (its row chosen), or the panel closed. */
void Back() {
    if (g_page >= 0 || g_page == kKeyboard) {
        const int page = g_page;
        g_page = -1;
        g_row = page == kKeyboard ? int(Rows().size() + Pages().size()) : int(Rows().size()) + page;
        return;
    }
    g_open = false;
    SetWindowOverlay({});
}

void Change(int step) {
    const std::vector<Row> rows = Shown();
    const Row row = rows[std::size_t((std::min)(g_row, int(rows.size()) - 1))];
    if (row.kind == Row::kLast) {
        Back();
        return;
    }
    if (row.kind == Row::kPage) {
        g_page = row.index;
        g_row = 0;
        std::fprintf(stderr, "[pc menu] page %s\n", Pages()[std::size_t(g_page)].c_str());
        return;
    }
    if (row.kind == Row::kKeyLink) {
        g_page = kKeyboard;
        g_row = 0;
        std::fprintf(stderr, "[pc menu] page Keyboard\n");
        return;
    }
    if (row.kind == Row::kKey) {
        g_binding = row.index;
        g_binding_armed = false;
        std::fprintf(stderr, "[pc menu] keyboard: the key for %s?\n",
                     bindings::ButtonName(row.index, bindings::Style::kPlayStation));
        return;
    }
    if (row.kind == Row::kOption) {
        const PortOption& o = PortOptions()[std::size_t(row.index)];
        const int count = int(o.values.size());
        if (!o.now || !o.set || count == 0) return;
        o.set(((o.now() + step) % count + count) % count);
        const int now = o.now();
        std::fprintf(stderr, "[pc menu] %s: %s %s\n", o.page.c_str(), o.name.c_str(),
                     now >= 0 && now < count ? o.values[std::size_t(now)].c_str() : "?");
        return;
    }
    const Choice& c = Choices()[std::size_t(row.index)];
    const std::string now = c.get();
    int at = 0;
    for (int i = 0; i < int(c.values.size()); ++i)
        if (c.values[i] == now) at = i;
    const int count = int(c.values.size());
    const std::string& to = c.values[std::size_t(((at + step) % count + count) % count)];
    c.set(to);
    std::fprintf(stderr, "[pc menu] %s: %s\n", c.label, to.c_str());
}

// Keys and buttons, each acting when it goes down.
enum Input { kToggle, kUp, kDown, kLeft, kRight, kAccept, kBack, kInputs };

std::array<bool, kInputs> Held() {
    std::array<bool, kInputs> held{};
    const bool focused = WindowFocused() || KeysScripted();     // a scripted run's keys (--key) in the background too
#ifdef _WIN32
    auto key = [&](int vk) { return focused && HostKeyDown(vk); };
    held[kToggle] = key(VK_F9);
    held[kUp] = key(VK_UP);
    held[kDown] = key(VK_DOWN);
    held[kLeft] = key(VK_LEFT);
    held[kRight] = key(VK_RIGHT);
    held[kAccept] = key(VK_RETURN);
    held[kBack] = key(VK_ESCAPE);
#endif
    HostPadState pad;
    if (focused && HostPad(&pad)) {
        // DualShock 2 order: 1 L3, 2 R3, 4 up, 5 right, 6 down, 7 left, 13 circle, 14 cross.
        auto button = [&](int bit) { return (pad.buttons >> bit) & 1; };
        held[kToggle] = held[kToggle] || (button(1) && button(2));
        held[kUp] = held[kUp] || button(4);
        held[kDown] = held[kDown] || button(6);
        held[kLeft] = held[kLeft] || button(7);
        held[kRight] = held[kRight] || button(5);
        held[kAccept] = held[kAccept] || button(14);
        held[kBack] = held[kBack] || button(13);
    }
    return held;
}

}  // namespace

void AddPortOptions(const std::vector<PortOption>& options) {
    PortOptions().insert(PortOptions().end(), options.begin(), options.end());
}

std::vector<PcSetting> PcSettings() {
    std::vector<PcSetting> settings;
    for (int i : Rows()) {
        const Choice& c = Choices()[std::size_t(i)];
        const std::string now = c.get();
        std::string value = now;
        for (std::size_t v = 0; v < c.values.size(); ++v)
            if (c.values[v] == now) value = c.game_values[v];
        settings.push_back({c.game_label, value, c.help});
    }
    return settings;
}

void StepPcSetting(int index, int step) {
    int n = 0;
    for (int i : Rows()) {
        if (n++ != index) continue;
        const Choice& c = Choices()[std::size_t(i)];
        const std::string now = c.get();
        int at = 0;
        for (int v = 0; v < int(c.values.size()); ++v)
            if (c.values[std::size_t(v)] == now) at = v;
        const int count = int(c.values.size());
        c.set(c.values[std::size_t(((at + step) % count + count) % count)]);
        return;
    }
}

bool PcMenuOpen() { return g_open || g_releasing; }

void OpenPcMenu() {
    if (g_open) return;
    g_open = true;
    g_row = 0;
    g_page = -1;
    Draw(true);
}

/* A key awaited for the Keyboard page's input: the first one down once all
   were let go, or Esc leaving it as it was. */
void AwaitKey() {
    if (!(WindowFocused() || KeysScripted())) return;
    if (!g_binding_armed) {
        g_binding_armed = !bindings::AnyKeyDown();
        return;
    }
#ifdef _WIN32
    if (HostKeyDown(VK_ESCAPE)) {
        std::fprintf(stderr, "[pc menu] keyboard: %s left as it was\n",
                     bindings::ButtonName(g_binding, bindings::Style::kPlayStation));
        g_binding = -1;
        return;
    }
#endif
    for (int vk : bindings::BindableKeys())
        if (HostKeyDown(vk)) {
            bindings::SetGameKey(g_binding, vk);
            std::fprintf(stderr, "[pc menu] keyboard: %s on %s\n",
                         bindings::ButtonName(g_binding, bindings::Style::kPlayStation), bindings::KeyName(vk));
            g_binding = -1;
            return;
        }
}

void UpdatePcMenu() {
    static std::array<bool, kInputs> before{};
    const std::array<bool, kInputs> now = Held();
    const bool was_open = g_open;
    auto went_down = [&](Input i) { return now[i] && !before[i]; };
    if (g_open && g_binding >= 0 && !went_down(kToggle)) {
        // The keys are the binding's until it is made (Esc too: not Back).
        AwaitKey();
        before = now;
        Draw();
        return;
    }
    g_binding = -1;
    if (went_down(kToggle)) {
        g_open = !g_open;
        g_row = 0;
        g_page = -1;
        if (!g_open) SetWindowOverlay({});
        else Draw(true);
    } else if (g_open && went_down(kBack)) {
        Back();
    } else if (g_open) {
        const int rows = int(Shown().size());
        if (went_down(kUp)) g_row = (g_row + rows - 1) % rows;
        if (went_down(kDown)) g_row = (g_row + 1) % rows;
        if (went_down(kLeft)) Change(-1);
        if (went_down(kRight) || went_down(kAccept)) Change(+1);
    }
    before = now;
    if (g_open != was_open) std::fprintf(stderr, "[pc menu] %s\n", g_open ? "open" : "closed");
    if (!g_open && was_open) g_releasing = true;
    if (g_releasing) {
        bool any = false;
        for (bool held : now) any = any || held;
        g_releasing = any;
    }
    if (g_open) Draw();
}

}  // namespace ps2host
