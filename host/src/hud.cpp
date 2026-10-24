/*
 * The HUD's anchors and the HUD editor (ps2host/hud.h): what the ports'
 * widescreen code shares.
 */

#include "ps2host/hud.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "ps2host/run.h"
#include "ps2host/settings.h"

namespace ps2host {
namespace hud {
namespace {

Game g_game;
std::map<std::string, Anchoring> g_anchors;

std::size_t RowOf(char c) { return std::string("TMB").find(c); }
std::size_t ColumnOf(char c) { return std::string("LCR").find(c); }

struct Editor {
    bool on = false;
    ElementId chosen;
    std::map<ElementId, std::chrono::steady_clock::time_point> seen;   // drawn lately
    bool keys[8] = {};
    std::size_t shown = ~std::size_t(0);   // how many the panel lists
    bool paused = false;
};
Editor g_editor;

const char* CellName(const char cell[3]) {
    static const char* const kRows[] = {"top", "middle", "bottom"};
    static const char* const kColumns[] = {"left", "centre", "right"};
    static char name[24];
    const std::size_t row = RowOf(cell[0]), column = ColumnOf(cell[1]);
    std::snprintf(name, sizeof name, "%s %s", row < 3 ? kRows[row] : "?", column < 3 ? kColumns[column] : "?");
    return name;
}

/* The panel: what is on screen, the one chosen, the keys. */
void ShowStatus() {
    if (!g_editor.on) {
        SetWindowOverlay({});
        SetWindowStatus("");
        return;
    }
    std::vector<std::string> lines;
    if (!g_game.can_pause) lines.push_back("HUD EDITOR  (F10 closes)");
    else if (g_editor.paused) lines.push_back("HUD EDITOR  (F10 closes)   - PAUSED (P resumes)");
    else lines.push_back(std::string("HUD EDITOR  (F10 closes)   P pauses ") + g_game.paused_what);
    lines.push_back("PgUp/PgDn choose   Home left   Insert centre   End right   Delete stretch");
    lines.push_back("");
    std::size_t index = 0, count = 0;
    for (const auto& [id, when] : g_editor.seen) {
        ++count;
        const bool chosen = id == g_editor.chosen;
        if (chosen) index = count;
        const char* name = g_game.name ? g_game.name(id) : "";
        char text[160];
        if (g_game.name)
            std::snprintf(text, sizeof text, "%s %-13s %-10s %06X/%06X   %s", chosen ? ">" : " ", CellName(id.cell),
                          name, unsigned(id.object), unsigned(id.context), AnchorName(AnchorOf(id)));
        else
            std::snprintf(text, sizeof text, "%s %-13s %06X/%06X   %s", chosen ? ">" : " ", CellName(id.cell),
                          unsigned(id.object), unsigned(id.context), AnchorName(AnchorOf(id)));
        lines.push_back(text);
    }
    if (!count) lines.push_back("  (no HUD drawn)");
    g_editor.shown = count;
    char title[96];
    std::snprintf(title, sizeof title, "HUD EDIT %zu/%zu", index, count);
    SetWindowStatus(title);
    SetWindowOverlay(lines);
}

/* A key's press (down now, up before). A scripted run's keys (--key) count
   with the window in the background. */
bool Press(int slot, int vk, int vk2 = 0) {
    const bool down = (WindowFocused() || KeysScripted()) && (HostKeyDown(vk) || (vk2 && HostKeyDown(vk2)));
    const bool pressed = down && !g_editor.keys[slot];
    g_editor.keys[slot] = down;
    return pressed;
}

}  // namespace

bool ElementId::operator<(const ElementId& o) const {
    if (cell[0] != o.cell[0]) return RowOf(cell[0]) < RowOf(o.cell[0]);
    if (cell[1] != o.cell[1]) return ColumnOf(cell[1]) < ColumnOf(o.cell[1]);
    if (object != o.object) return object < o.object;
    return context < o.context;
}

std::string ElementId::Key() const {
    char key[48];
    std::snprintf(key, sizeof key, "hud.%06X.%06X.%s", unsigned(context), unsigned(object), cell);
    return key;
}

void SetGame(const Game& game) {
    g_game = game;
    g_anchors.clear();
}

const char* AnchorName(Anchor a) {
    switch (a) {
        case Anchor::kLeft: return "left";
        case Anchor::kRight: return "right";
        case Anchor::kStretch: return "stretch";
        default: return "centre";
    }
}

Anchoring AnchoringOf(const ElementId& id) {
    if (!id.known()) return {Anchor::kCentre, false};
    const std::string key = id.Key();
    auto found = g_anchors.find(key);
    if (found != g_anchors.end()) return found->second;
    Anchoring a{Anchor::kCentre, false};
    for (std::size_t i = 0; i < g_game.default_count; ++i) {
        const std::string name = g_game.defaults[i].first;
        if (key == name || (!name.empty() && name.back() == '.' && key.compare(0, name.size(), name) == 0))
            a = {g_game.defaults[i].second, true};
    }
    const std::string set = Setting(key, "");
    if (set == "left") a = {Anchor::kLeft, true};
    else if (set == "right") a = {Anchor::kRight, true};
    else if (set == "centre") a = {Anchor::kCentre, true};
    else if (set == "stretch") a = {Anchor::kStretch, true};
    g_anchors[key] = a;
    return a;
}

Anchor AnchorOf(const ElementId& id) { return AnchoringOf(id).anchor; }

float Wider() { return PictureAspect() / (4.0f / 3.0f); }
float Squeeze() { return std::min(1.0f, (4.0f / 3.0f) / PictureAspect()); }

float Place(float x, float centre, float half, float k, Anchor anchor) {
    if (anchor == Anchor::kStretch) return x;
    const float a = anchor == Anchor::kLeft ? centre - half : anchor == Anchor::kRight ? centre + half : centre;
    return a + (x - a) * k;
}

int PlaceGs(int x, int centre, int half, float k, Anchor anchor) {
    if (anchor == Anchor::kStretch) return x;
    const float a = anchor == Anchor::kLeft ? float(centre - half)
                  : anchor == Anchor::kRight ? float(centre + half) : float(centre);
    return int(std::lround(a + (float(x) - a) * k));
}

void CellOf(float x, float y, char cell[3]) {
    const float w = g_game.width, h = g_game.height;
    cell[0] = y < h / 3 ? 'T' : y < 2 * h / 3 ? 'M' : 'B';
    cell[1] = x < w / 3 ? 'L' : x < 2 * w / 3 ? 'C' : 'R';
    cell[2] = 0;
}

void EditorInput() {
    constexpr int kF10 = 0x79, kPageUp = 0x21, kPageDown = 0x22, kHome = 0x24, kEnd = 0x23, kInsert = 0x2D;
    constexpr int kPad4 = 0x64, kPad5 = 0x65, kPad6 = 0x66, kDelete = 0x2E, kPad8 = 0x68, kP = 'P';
    if (Press(0, kF10)) {
        g_editor.on = !g_editor.on && Squeeze() < 1.0f;
        g_editor.paused = false;
        g_editor.seen.clear();
        g_editor.chosen = ElementId{};
        g_editor.shown = ~std::size_t(0);
        std::fprintf(stderr, "[hud] editor %s\n", g_editor.on ? "on" : Squeeze() < 1.0f ? "off" : "off (a 4:3 picture)");
        ShowStatus();
    }
    if (!g_editor.on) return;
    // What has not been drawn for a while is off the screen; the panel
    // follows what is.
    const auto now = std::chrono::steady_clock::now();
    for (auto it = g_editor.seen.begin(); it != g_editor.seen.end();)
        it = now - it->second > std::chrono::milliseconds(500) ? g_editor.seen.erase(it) : std::next(it);
    if (g_editor.seen.size() != g_editor.shown) ShowStatus();
    std::vector<ElementId> list;
    for (const auto& [id, when] : g_editor.seen) list.push_back(id);
    const bool next = Press(1, kPageDown), previous = Press(2, kPageUp);
    if ((next || previous) && !list.empty()) {
        std::size_t at = list.size();
        for (std::size_t i = 0; i < list.size(); ++i)
            if (list[i] == g_editor.chosen) at = i;
        if (at == list.size()) at = next ? list.size() - 1 : 0;
        g_editor.chosen = list[(at + (next ? 1 : list.size() - 1)) % list.size()];
        ShowStatus();
    }
    Anchor set = Anchor::kCentre;
    bool setting = false;
    if (Press(3, kHome, kPad4)) setting = true, set = Anchor::kLeft;
    if (Press(4, kInsert, kPad5)) setting = true, set = Anchor::kCentre;
    if (Press(5, kEnd, kPad6)) setting = true, set = Anchor::kRight;
    if (Press(6, kDelete, kPad8)) setting = true, set = Anchor::kStretch;
    if (g_game.can_pause && Press(7, kP)) {
        g_editor.paused = !g_editor.paused;
        std::fprintf(stderr, "[hud] %s %s\n", g_game.paused_what, g_editor.paused ? "paused" : "resumed");
        ShowStatus();
    }
    if (setting && g_editor.chosen.known()) {
        const std::string key = g_editor.chosen.Key();
        SetSetting(key, AnchorName(set));
        g_anchors[key] = {set, true};
        std::fprintf(stderr, "[hud] %s anchored %s\n", key.c_str(), AnchorName(set));
        ShowStatus();
    }
}

bool EditorOn() { return g_editor.on; }

void Seen(const ElementId& id) {
    if (g_editor.on && id.known()) g_editor.seen[id] = std::chrono::steady_clock::now();
}

bool Hidden(const ElementId& id) {
    if (!g_editor.on || !(id == g_editor.chosen)) return false;
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
    return (ms / 250) % 2 == 1;
}

bool Paused() { return g_editor.on && g_editor.paused; }

}  // namespace hud
}  // namespace ps2host
