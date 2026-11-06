/*
 * The PC settings panel: F9 (or L3 + R3 on a controller) over any game.
 *
 * A port can put its PC settings in the game's own menus (Burnout 3's main
 * menu has PC SETTINGS); this panel is the engine's, for every port: the
 * settings the host itself applies, changed while the game runs and saved
 * with the PC settings (settings.h). While it is open the game gets no input.
 *
 * Its tabs: DISPLAY (the engine's settings), each of a port's pages
 * (AddPortOptions), each of its binding pages (bindings::AddPanelPage: a
 * context's keys or buttons, rebound - Enter, then the key; DEFAULTS), and,
 * for a port without controls of its own (no SetKeyboardMapper), KEYBOARD
 * (the DualShock 2's inputs and their keys, bindings.h's GameKeys).
 *
 *   Up / Down             choose a row
 *   Left / Right          change it (Enter or Cross: change, open, bind)
 *   Page Up / Page Down,  the tab before / after (L1 / R1 on a controller)
 *   Q / E
 *   Esc, Circle           close (a binding waiting: leave it as it was)
 *   F9                    close
 *   the mouse             a tab or a row chosen by pointing, a click
 *                         changes (its value's arrows: back or on), the
 *                         wheel scrolls; a right click closes
 *
 * Drawn on Windows with its own font (Segoe UI) as a panel in the window's
 * middle; elsewhere as the overlay's lines.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ps2host/bindings.h"
#include "ps2host/high_fps.h"
#include "ps2host/run.h"
#include "ps2host/settings.h"
#include "ps2host/texture_pack.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ps2host {

namespace {

bool g_open = false;
bool g_releasing = false;                // closed, the game's input held back until the keys are let go
int g_tab = 0;                           // the tab shown (Tabs()' index)
int g_row = 0;                           // its row chosen
int g_scroll = 0;                        // its first row shown
int g_binding = -1;                      // the row a key (or button) is awaited for, or -1
bool g_binding_armed = false;            // ...every key let go since it began
u64 g_binding_until = 0;                 // ...the panel's tick it gives up at
u64 g_tick = 0;

/* The KEYBOARD tab's rows, in the order a player thinks of them. */
constexpr int kKeyRows[] = {bindings::kUp,     bindings::kDown, bindings::kLeft,   bindings::kRight,
                            bindings::kCross,  bindings::kCircle, bindings::kSquare, bindings::kTriangle,
                            bindings::kL1,     bindings::kR1,   bindings::kL2,     bindings::kR2,
                            bindings::kL3,     bindings::kR3,   bindings::kStart,  bindings::kSelect};

std::vector<PortOption>& PortOptions() {
    static std::vector<PortOption> options;
    return options;
}

std::vector<bindings::PanelPage>& PanelPages() {
    static std::vector<bindings::PanelPage> pages;
    return pages;
}

/* The ports' pages, in the order first named. */
std::vector<std::string> Pages() {
    std::vector<std::string> pages;
    for (const PortOption& o : PortOptions())
        if (std::find(pages.begin(), pages.end(), o.page) == pages.end()) pages.push_back(o.page);
    return pages;
}

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
        {"Fullscreen mode", {"borderless", "exclusive"}, {"Borderless", "Exclusive (60/120 Hz)"},
         [] { return Setting("fullscreen_mode", "borderless"); },
         [](const std::string& v) { SetWindowFullscreenMode(v); },
         "FULLSCREEN MODE", {"BORDERLESS", "EXCLUSIVE"},
         "Borderless: no flicker, other screens usable. Exclusive: 60 or 120 Hz, no judder."},
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
        // VRR: last, so a game's own menu that shows the first few rows
        // (Mister Mosquito's) keeps the ones it had.
        {"VRR", {"off", "on"}, {"Off", "On"},
         [] { return OnOff(WindowVrr()); },
         [](const std::string& v) { SetWindowVrr(v == "on"); },
         "VRR", {"OFF", "ON"}, "A G-Sync or FreeSync screen: its refresh follows the pictures."},
        {"GPU boost", {"on", "off"}, {"On", "Off"},
         [] { return OnOff(WindowGpuBoost()); },
         [](const std::string& v) { SetWindowGpuBoost(v == "on"); },
         "GPU BOOST", {"ON", "OFF"}, "The GPU's clocks kept up: no hitch at a new scene. Off saves battery."},
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
        // A phone's screen: always the whole of it, synced, no VRR or GPU boost to ask for.
        if (OnPhone() && (label == "Fullscreen" || label == "Fullscreen mode" || label == "V-sync" || label == "VRR" || label == "GPU boost"))
            continue;
        if (label != "Texture pack" || texture_pack::Count()) rows.push_back(i);
    }
    return rows;
}


// ------------------------------------------------------------------ tabs and rows

struct Tab {
    enum Kind { kDisplay, kPort, kBindings, kKeyboard } kind;
    int index;                          // the port's page, or its binding page
    std::string title;
};

std::vector<Tab> Tabs() {
    std::vector<Tab> tabs = {{Tab::kDisplay, 0, "DISPLAY"}};
    const std::vector<std::string> pages = Pages();
    for (int i = 0; i < int(pages.size()); ++i) tabs.push_back({Tab::kPort, i, pages[std::size_t(i)]});
    for (int i = 0; i < int(PanelPages().size()); ++i)
        tabs.push_back({Tab::kBindings, i, PanelPages()[std::size_t(i)].title});
    if (!bindings::PortMapsKeyboard() && KeyboardPresent()) tabs.push_back({Tab::kKeyboard, 0, "KEYBOARD"});
    return tabs;
}

Tab CurrentTab() {
    const std::vector<Tab> tabs = Tabs();
    g_tab = std::clamp(g_tab, 0, int(tabs.size()) - 1);
    return tabs[std::size_t(g_tab)];
}

/* A row shown: the engine's setting, a port's option, a binding, a KEYBOARD key, DEFAULTS. */
struct Row {
    enum Kind { kChoice, kOption, kBinding, kKey, kDefaults } kind;
    int index;                          // the choice, the option, the binding page's row, the input
};

std::vector<Row> Shown() {
    std::vector<Row> rows;
    const Tab tab = CurrentTab();
    if (tab.kind == Tab::kDisplay) {
        for (int i : Rows()) rows.push_back({Row::kChoice, i});
    } else if (tab.kind == Tab::kPort) {
        const std::string page = Pages()[std::size_t(tab.index)];
        const auto& options = PortOptions();
        for (int i = 0; i < int(options.size()); ++i)
            if (options[std::size_t(i)].page == page) rows.push_back({Row::kOption, i});
    } else if (tab.kind == Tab::kBindings) {
        const bindings::PanelPage& page = PanelPages()[std::size_t(tab.index)];
        for (int i = 0; i < int(page.rows.size()); ++i) rows.push_back({Row::kBinding, i});
        rows.push_back({Row::kDefaults, 0});
    } else {
        for (int input : kKeyRows) rows.push_back({Row::kKey, input});
    }
    return rows;
}

bindings::Table PageTable(const bindings::PanelPage& page) {
    return bindings::Load(page.device, page.context, page.defaults);
}

/* A key's name as the player's keyboard labels it: a letter or a digit is a
   place (a QWERTY's - the host's keys are scancodes), named by what this
   layout has there (an AZERTY's W where a QWERTY has Z); the rest by the
   engine's names. */
std::string KeyLabel(int vk) {
    const char* name = vk ? bindings::KeyName(vk) : nullptr;
    if (!name) return "NONE";
#ifdef _WIN32
    static const unsigned char kLetters[26] = {0x1E, 0x30, 0x2E, 0x20, 0x12, 0x21, 0x22, 0x23, 0x17,
                                               0x24, 0x25, 0x26, 0x32, 0x31, 0x18, 0x19, 0x10, 0x13,
                                               0x1F, 0x14, 0x16, 0x2F, 0x11, 0x2D, 0x15, 0x2C};
    unsigned scan = 0;
    if (vk >= 'A' && vk <= 'Z') scan = kLetters[vk - 'A'];
    else if (vk >= '1' && vk <= '9') scan = 0x02 + unsigned(vk - '1');
    else if (vk == '0') scan = 0x0B;
    wchar_t text[32] = {};
    if (scan && GetKeyNameTextW(LONG(scan << 16), text, 32) > 0) {
        std::string out;
        for (const wchar_t* p = text; *p; ++p) out += *p < 128 ? char(towupper(*p)) : '?';
        if (!out.empty() && out.find('?') == std::string::npos) return out;
    }
#endif
    return name;
}

std::string BindingName(const bindings::PanelPage& page, int physical) {
    if (page.device == bindings::Device::kKeyboard) return KeyLabel(physical);
    return physical >= 0 ? bindings::ButtonName(physical, bindings::Style::kPlayStation) : "NONE";
}

/* A row's words: its label, its value, its help. */
struct RowText {
    std::string label, value, help;
    bool steps = false;                 // Left / Right change it (a value with arrows)
};

RowText TextOf(const Row& row) {
    RowText t;
    if (row.kind == Row::kChoice) {
        const Choice& c = Choices()[std::size_t(row.index)];
        const std::string now = c.get();
        t.value = now;
        for (std::size_t i = 0; i < c.values.size(); ++i)
            if (c.values[i] == now) t.value = c.shown[i];
        t.label = c.label;
        t.help = c.help ? c.help : "";
        t.steps = true;
    } else if (row.kind == Row::kOption) {
        const PortOption& o = PortOptions()[std::size_t(row.index)];
        const int now = o.now ? o.now() : 0;
        t.label = o.name;
        t.value = now >= 0 && now < int(o.values.size()) ? o.values[std::size_t(now)] : "?";
        t.help = o.help;
        t.steps = true;
    } else if (row.kind == Row::kBinding) {
        const bindings::PanelPage& page = PanelPages()[std::size_t(CurrentTab().index)];
        const auto& [input, what] = page.rows[std::size_t(row.index)];
        t.label = what;
        t.value = g_binding == row.index ? "PRESS A " + std::string(page.device == bindings::Device::kKeyboard ? "KEY" : "BUTTON")
                                         : BindingName(page, PageTable(page).physical[input]);
        t.help = page.device == bindings::Device::kKeyboard
                     ? "Enter (or a click), then the key for it. Esc leaves it as it was; a key another action had swaps them."
                     : "Cross (or a click), then the button for it.";
    } else if (row.kind == Row::kKey) {
        const int vk = bindings::GameKeys().physical[row.index];
        t.label = bindings::ButtonName(row.index, bindings::Style::kPlayStation);
        t.value = g_binding == row.index ? "PRESS A KEY" : KeyLabel(vk);
        t.help = "Enter (or a click), then the key for it. Esc leaves it as it was.";
    } else {
        t.label = "Defaults";
        t.value = "RESET";
        t.help = "Every binding on this tab as it was at first.";
    }
    return t;
}

// ------------------------------------------------------------------ the picture

/* The panel's layout in its own pixels, for the drawing and the mouse. */
struct Layout {
    int w = 0, h = 0;
    float s = 1.0f;
    int header = 0, tabs_w = 0, row_h = 0, help_h = 0, pad = 0;
    int rows_y = 0, visible = 0;
    int pill_w = 0;
};

Layout MakeLayout() {
    Layout l;
    u32 ww = 1920, wh = 1080;
    WindowPixelSize(&ww, &wh);
    l.s = std::clamp(float(wh) / 1080.0f, 0.7f, 1.4f);
    l.w = std::min({1280, int(ww) - 32, int(1120 * l.s)});
    l.h = std::min({720, int(wh) - 32, int(660 * l.s)});
    l.header = int(72 * l.s);
    l.tabs_w = int(250 * l.s);
    l.row_h = int(48 * l.s);
    l.help_h = int(92 * l.s);
    l.pad = int(24 * l.s);
    l.rows_y = l.header + int(12 * l.s);
    l.visible = std::max(1, (l.h - l.rows_y - l.help_h) / l.row_h);
    l.pill_w = int(300 * l.s);
    return l;
}

void KeepRowShown(const Layout& l, int rows) {
    if (g_row < g_scroll) g_scroll = g_row;
    if (g_row >= g_scroll + l.visible) g_scroll = g_row - l.visible + 1;
    g_scroll = std::clamp(g_scroll, 0, std::max(0, rows - l.visible));
}

#ifdef _WIN32
COLORREF Rgb(int r, int g, int b) { return RGB(r, g, b); }

void Fill(HDC dc, int x, int y, int w, int h, COLORREF colour, int round = 0) {
    HBRUSH brush = CreateSolidBrush(colour);
    if (round <= 0) {
        const RECT r{x, y, x + w, y + h};
        FillRect(dc, &r, brush);
    } else {
        HGDIOBJ old_brush = SelectObject(dc, brush);
        HGDIOBJ old_pen = SelectObject(dc, GetStockObject(NULL_PEN));
        RoundRect(dc, x, y, x + w + 1, y + h + 1, round, round);
        SelectObject(dc, old_pen);
        SelectObject(dc, old_brush);
    }
    DeleteObject(brush);
}

HFONT Font(int px, int weight) {
    return CreateFontW(-px, 0, 0, 0, weight, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                       CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
}

std::wstring Wide(const std::string& s) {
    std::wstring w;
    for (unsigned char c : s) w += wchar_t(c);
    return w;
}

void Text(HDC dc, HFONT font, COLORREF colour, const std::string& text, int x, int y, int w, int h, UINT format) {
    HGDIOBJ old = SelectObject(dc, font);
    SetTextColor(dc, colour);
    RECT r{x, y, x + w, y + h};
    const std::wstring t = Wide(text);
    DrawTextW(dc, t.c_str(), int(t.size()), &r, format | DT_NOPREFIX);
    SelectObject(dc, old);
}

/* The panel's picture. */
bool DrawPanel(const Layout& l, std::vector<u32>* pixels) {
    HDC dc = CreateCompatibleDC(nullptr);
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof info.bmiHeader;
    info.bmiHeader.biWidth = l.w;
    info.bmiHeader.biHeight = -l.h;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!bitmap || !bits) {
        DeleteDC(dc);
        return false;
    }
    HGDIOBJ old_bitmap = SelectObject(dc, bitmap);
    SetBkMode(dc, TRANSPARENT);
    const float s = l.s;
    const auto S = [&](float v) { return int(v * s); };
    const COLORREF kBack = Rgb(22, 24, 29), kHead = Rgb(15, 16, 20), kTabs = Rgb(18, 20, 24),
                   kTabOn = Rgb(40, 44, 54), kAccent = Rgb(206, 44, 48), kRowOn = Rgb(37, 41, 50),
                   kPill = Rgb(48, 53, 64), kPillOn = Rgb(62, 68, 82), kText = Rgb(232, 233, 237),
                   kDim = Rgb(150, 156, 168), kFaint = Rgb(108, 114, 126), kWait = Rgb(255, 196, 70);
    HFONT title = Font(S(28), FW_SEMIBOLD), tab_font = Font(S(18), FW_SEMIBOLD), row_font = Font(S(19), FW_NORMAL),
          value_font = Font(S(18), FW_SEMIBOLD), small_font = Font(S(15), FW_NORMAL);
    // the card, its header
    Fill(dc, 0, 0, l.w, l.h, kBack);
    Fill(dc, 0, 0, l.w, l.header, kHead);
    Fill(dc, 0, l.header - S(3), l.w, S(3), kAccent);
    Text(dc, title, kText, OnPhone() ? "PORT SETTINGS" : "PC SETTINGS", l.pad, 0, l.w / 2, l.header,
         DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    Text(dc, small_font, kFaint, OnPhone() ? "L3 + R3 closes" : "F9 or Esc closes", l.w / 2, 0, l.w / 2 - l.pad, l.header,
         DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
    // the tabs
    Fill(dc, 0, l.header, l.tabs_w, l.h - l.header, kTabs);
    const std::vector<Tab> tabs = Tabs();
    for (int i = 0; i < int(tabs.size()); ++i) {
        const int y = l.header + S(12) + i * S(44);
        if (y + S(44) > l.h - S(40)) break;
        if (i == g_tab) {
            Fill(dc, S(10), y, l.tabs_w - S(20), S(38), kTabOn, S(10));
            Fill(dc, S(10), y + S(8), S(4), S(22), kAccent);
        }
        Text(dc, tab_font, i == g_tab ? kText : kDim, tabs[std::size_t(i)].title, S(26), y, l.tabs_w - S(36), S(38),
             DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
    }
    Text(dc, small_font, kFaint, "PgUp / PgDn: tabs", S(18), l.h - S(36), l.tabs_w - S(24), S(28),
         DT_LEFT | DT_VCENTER | DT_SINGLELINE);
    // the rows
    const std::vector<Row> rows = Shown();
    const int x0 = l.tabs_w + l.pad, x1 = l.w - l.pad;
    std::string help;
    for (int r = g_scroll; r < int(rows.size()) && r < g_scroll + l.visible; ++r) {
        const int y = l.rows_y + (r - g_scroll) * l.row_h;
        const RowText t = TextOf(rows[std::size_t(r)]);
        const bool on = r == g_row;
        if (on) {
            Fill(dc, x0 - S(10), y + S(3), x1 - x0 + S(20), l.row_h - S(6), kRowOn, S(12));
            Fill(dc, x0 - S(10), y + S(12), S(4), l.row_h - S(24), kAccent);
            help = t.help;
        }
        Text(dc, row_font, on ? kText : Rgb(205, 207, 214), t.label, x0 + S(8), y, x1 - x0 - l.pill_w - S(24), l.row_h,
             DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
        const int px = x1 - l.pill_w, py = y + S(8), ph = l.row_h - S(16);
        const bool waiting = g_binding == r;
        Fill(dc, px, py, l.pill_w, ph, waiting ? Rgb(84, 64, 24) : on ? kPillOn : kPill, S(16));
        if (t.steps) {
            Text(dc, value_font, on ? kAccent : kFaint, "<", px + S(8), py, S(28), ph, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
            Text(dc, value_font, on ? kAccent : kFaint, ">", px + l.pill_w - S(36), py, S(28), ph,
                 DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        }
        Text(dc, value_font, waiting ? kWait : kText, t.value, px + S(36), py, l.pill_w - S(72), ph,
             DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
    }
    // a scroll bar, when not all show
    if (int(rows.size()) > l.visible) {
        const int track_y = l.rows_y, track_h = l.visible * l.row_h;
        const int bar_h = std::max(S(24), track_h * l.visible / int(rows.size()));
        const int bar_y = track_y + (track_h - bar_h) * g_scroll / std::max(1, int(rows.size()) - l.visible);
        Fill(dc, l.w - S(10), track_y, S(4), track_h, kPill, S(4));
        Fill(dc, l.w - S(10), bar_y, S(4), bar_h, kDim, S(4));
    }
    // the help, the keys
    const int hy = l.h - l.help_h;
    Fill(dc, l.tabs_w, hy, l.w - l.tabs_w, S(1), kPill);
    Text(dc, row_font, kDim, help, x0 + S(8), hy + S(10), x1 - x0 - S(8), S(46), DT_LEFT | DT_WORDBREAK);
    Text(dc, small_font, kFaint, "Up/Down choose    Left/Right change    Enter select    Esc close    the mouse: point, click, scroll",
         x0 + S(8), l.h - S(34), x1 - x0, S(28), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
    GdiFlush();
    pixels->assign(static_cast<const u32*>(bits), static_cast<const u32*>(bits) + std::size_t(l.w) * l.h);
    for (u32& p : *pixels) p |= 0xFF000000u;
    for (HFONT f : {title, tab_font, row_font, value_font, small_font}) DeleteObject(f);
    SelectObject(dc, old_bitmap);
    DeleteObject(bitmap);
    DeleteDC(dc);
    return true;
}
#endif

/* The panel shown again (its picture, or the overlay's lines elsewhere). */
void Draw(bool force = false) {
    const Layout l = MakeLayout();
    const std::vector<Row> rows = Shown();
    g_row = std::clamp(g_row, 0, std::max(0, int(rows.size()) - 1));
    KeepRowShown(l, int(rows.size()));
#ifdef _WIN32
    // (drawn again only when it changes: its words, the tab, the row)
    std::string key = std::to_string(g_tab) + "/" + std::to_string(g_row) + "/" + std::to_string(g_scroll) + "/" +
                      std::to_string(g_binding) + "/" + std::to_string(l.w) + "x" + std::to_string(l.h);
    for (const Row& r : rows) {
        const RowText t = TextOf(r);
        key += "|" + t.value;
    }
    static std::string shown;
    if (!force && key == shown) return;
    shown = key;
    std::vector<u32> pixels;
    if (DrawPanel(l, &pixels)) SetWindowPanel(std::move(pixels), u32(l.w), u32(l.h));
#else
    std::vector<std::string> lines = {std::string(OnPhone() ? "PORT SETTINGS" : "PC SETTINGS") + " > " + CurrentTab().title};
    for (int r = 0; r < int(rows.size()); ++r) {
        const RowText t = TextOf(rows[std::size_t(r)]);
        lines.push_back((r == g_row ? "> " : "  ") + t.label + "   < " + t.value + " >");
    }
    static std::vector<std::string> shown;
    if (force || lines != shown) SetWindowOverlay(lines);
    shown = lines;
#endif
}

void Close() {
    g_open = false;
    g_binding = -1;
#ifdef _WIN32
    SetWindowPanel({}, 0, 0);
#else
    SetWindowOverlay({});
#endif
}

void SwitchTab(int step) {
    const int count = int(Tabs().size());
    g_tab = ((g_tab + step) % count + count) % count;
    g_row = 0;
    g_scroll = 0;
    g_binding = -1;
    std::fprintf(stderr, "[pc menu] tab %s\n", CurrentTab().title.c_str());
}

void Change(int step) {
    const std::vector<Row> rows = Shown();
    if (rows.empty()) return;
    const Row row = rows[std::size_t(std::min(g_row, int(rows.size()) - 1))];
    if (row.kind == Row::kBinding || row.kind == Row::kKey) {
        g_binding = g_row;
        g_binding_armed = false;
        g_binding_until = g_tick + 5 * 60;
        std::fprintf(stderr, "[pc menu] %s: waiting for its key\n", TextOf(row).label.c_str());
        return;
    }
    if (row.kind == Row::kDefaults) {
        const bindings::PanelPage& page = PanelPages()[std::size_t(CurrentTab().index)];
        bindings::Save(page.device, page.context, page.defaults);
        if (page.changed) page.changed();
        std::fprintf(stderr, "[pc menu] %s: the defaults again\n", page.title.c_str());
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
enum Input { kToggle, kUp, kDown, kLeft, kRight, kAccept, kBack, kTabBefore, kTabAfter, kInputs };

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
    held[kTabBefore] = key(VK_PRIOR) || key('Q');
    held[kTabAfter] = key(VK_NEXT) || key('E');
#endif
    HostPadState pad;
    if (focused && HostPad(&pad)) {
        // DualShock 2 order: 1 L3, 2 R3, 4 up, 5 right, 6 down, 7 left, 10 L1, 11 R1, 13 circle, 14 cross.
        auto button = [&](int bit) { return (pad.buttons >> bit) & 1; };
        held[kToggle] = held[kToggle] || (button(1) && button(2));
        held[kUp] = held[kUp] || button(4);
        held[kDown] = held[kDown] || button(6);
        held[kLeft] = held[kLeft] || button(7);
        held[kRight] = held[kRight] || button(5);
        held[kAccept] = held[kAccept] || button(14);
        held[kBack] = held[kBack] || button(13);
        held[kTabBefore] = held[kTabBefore] || button(10);
        held[kTabAfter] = held[kTabAfter] || button(11);
    }
    return held;
}

/* A key (or button) awaited for a binding row: the first one down once all
   were let go; Esc, five seconds or a right click leave it as it was. */
void AwaitBinding(bool cancel) {
    const std::vector<Row> rows = Shown();
    if (g_binding < 0 || g_binding >= int(rows.size())) {
        g_binding = -1;
        return;
    }
    const Row row = rows[std::size_t(g_binding)];
    if (cancel || g_tick > g_binding_until) {
        std::fprintf(stderr, "[pc menu] %s: left as it was\n", TextOf(row).label.c_str());
        g_binding = -1;
        return;
    }
    const bool keyboard =
        row.kind == Row::kKey || PanelPages()[std::size_t(CurrentTab().index)].device == bindings::Device::kKeyboard;
    if (keyboard) {
        if (!(WindowFocused() || KeysScripted())) return;
        if (!g_binding_armed) {
            g_binding_armed = !bindings::AnyKeyDown();
            return;
        }
#ifdef _WIN32
        if (HostKeyDown(VK_ESCAPE)) {
            std::fprintf(stderr, "[pc menu] %s: left as it was\n", TextOf(row).label.c_str());
            g_binding = -1;
            return;
        }
#endif
        for (int vk : bindings::BindableKeys())
            if (HostKeyDown(vk)) {
                if (row.kind == Row::kKey) {
                    bindings::SetGameKey(row.index, vk);
                } else {
                    const bindings::PanelPage& page = PanelPages()[std::size_t(CurrentTab().index)];
                    bindings::Table t = PageTable(page);
                    bindings::Bind(t, page.rows[std::size_t(row.index)].first, vk);
                    bindings::Save(page.device, page.context, t);
                    if (page.changed) page.changed();
                }
                std::fprintf(stderr, "[pc menu] %s: %s\n", TextOf(row).label.c_str(), KeyLabel(vk).c_str());
                g_binding = -1;
                return;
            }
        return;
    }
    // a controller's button (its DualShock 2 place)
    HostPadState pad;
    if (!HostPad(&pad)) return;
    if (!g_binding_armed) {
        g_binding_armed = pad.buttons == 0;
        return;
    }
    for (int b = 0; b < bindings::kInputs; ++b)
        if (pad.buttons & (1u << b)) {
            const bindings::PanelPage& page = PanelPages()[std::size_t(CurrentTab().index)];
            bindings::Table t = PageTable(page);
            bindings::Bind(t, page.rows[std::size_t(row.index)].first, b);
            bindings::Save(page.device, page.context, t);
            if (page.changed) page.changed();
            std::fprintf(stderr, "[pc menu] %s: %s\n", TextOf(row).label.c_str(),
                         bindings::ButtonName(b, bindings::Style::kPlayStation));
            g_binding = -1;
            return;
        }
}

/* The mouse over the panel: a tab or a row pointed at, a click, the wheel. */
void PanelMouse() {
    PanelMouseState m;
    if (!TakePanelMouse(&m) || !m.over) return;
    u32 ww = 0, wh = 0;
    if (!WindowPixelSize(&ww, &wh)) return;
    const Layout l = MakeLayout();
    const float x = m.px - float((int(ww) - l.w) / 2), y = m.py - float((int(wh) - l.h) / 2);
    if (m.pressed & 4) {            // a right click: as Esc
        if (g_binding >= 0) AwaitBinding(true);
        else Close();
        return;
    }
    if (g_binding >= 0) return;
    const std::vector<Row> rows = Shown();
    if (m.wheel != 0.0f) {
        g_scroll = std::clamp(g_scroll - int(m.wheel), 0, std::max(0, int(rows.size()) - l.visible));
        g_row = std::clamp(g_row, g_scroll, std::min(int(rows.size()) - 1, g_scroll + l.visible - 1));
    }
    if (x < 0 || y < 0 || x >= float(l.w) || y >= float(l.h)) return;
    if (x < float(l.tabs_w)) {
        const int tab = int((y - float(l.header) - 12.0f * l.s) / (44.0f * l.s));
        if ((m.pressed & 1) && y > float(l.header) && tab >= 0 && tab < int(Tabs().size()) && tab != g_tab) {
            g_tab = tab;
            g_row = g_scroll = 0;
        }
        return;
    }
    const int r = g_scroll + int((y - float(l.rows_y)) / float(l.row_h));
    if (y < float(l.rows_y) || r < 0 || r >= int(rows.size()) || r >= g_scroll + l.visible) return;
    if (m.moved) g_row = r;
    if (m.pressed & 1) {
        g_row = r;
        const RowText t = TextOf(rows[std::size_t(r)]);
        const float pill_x = float(l.w - l.pad - l.pill_w);
        // the value's arrows: back on the left one, on elsewhere
        Change(t.steps && x >= pill_x && x < pill_x + 44.0f * l.s ? -1 : +1);
    }
}

}  // namespace

void AddPortOptions(const std::vector<PortOption>& options) {
    PortOptions().insert(PortOptions().end(), options.begin(), options.end());
}

namespace bindings {
void AddPanelPage(const PanelPage& page) { PanelPages().push_back(page); }
}  // namespace bindings

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
    g_tab = g_row = g_scroll = 0;
    g_binding = -1;
    std::fprintf(stderr, "[pc menu] open\n");
    Draw(true);
}

void UpdatePcMenu() {
    ++g_tick;
    static std::array<bool, kInputs> before{};
    const std::array<bool, kInputs> now = Held();
    const bool was_open = g_open;
    auto went_down = [&](Input i) { return now[i] && !before[i]; };
    if (g_open && g_binding >= 0 && !went_down(kToggle)) {
        // The keys are the binding's until it is made (Esc too: not a close).
        PanelMouse();
        if (g_binding >= 0) AwaitBinding(false);
        before = now;
        if (g_open) Draw();
        return;
    }
    g_binding = -1;
    if (went_down(kToggle)) {
        if (g_open) {
            Close();
        } else {
            g_open = true;
            g_tab = g_row = g_scroll = 0;
            Draw(true);
        }
    } else if (g_open && went_down(kBack)) {
        Close();
    } else if (g_open) {
        const int rows = int(Shown().size());
        if (rows > 0 && went_down(kUp)) g_row = (g_row + rows - 1) % rows;
        if (rows > 0 && went_down(kDown)) g_row = (g_row + 1) % rows;
        if (went_down(kLeft)) Change(-1);
        if (went_down(kRight) || went_down(kAccept)) Change(+1);
        if (went_down(kTabBefore)) SwitchTab(-1);
        if (went_down(kTabAfter)) SwitchTab(+1);
        if (g_open) PanelMouse();
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
