/*
 * A game's 2D in a wider picture: the HUD's anchors and the HUD editor (F10).
 *
 * A port widens a game's 3D at its camera and keeps its 2D - the HUD, the
 * menus, the words - in its 4:3 shape: each element brought in towards the
 * middle of the screen, or towards its left or right edge (an anchor), or
 * left stretched across the wider picture as the game drew it. Where a
 * game's 2D is drawn, and how its vertices are moved, is the port's
 * (Burnout 3's, Mister Mosquito's and RE Outbreak's widescreen.cpp); this is
 * what they share:
 *
 *   - an element: what one piece of the game's code draws in one part of the
 *     screen - the object (the function that drew it), what called that (its
 *     context), and the cell of a 3x3 grid over the 4:3 picture its middle
 *     lies in. The same in every run, so it can be named in a setting.
 *   - its anchor: the setting "hud.<context>.<object>.<cell>" (left, right,
 *     centre or stretch), else the port's default (Game::defaults), else the
 *     centre.
 *   - the HUD editor: F10 in a wider picture shows a panel over the picture
 *     listing the elements drawn lately and their anchors; Page Up and Page
 *     Down choose one (it blinks: Hidden), Home or numpad 4 anchors it left,
 *     Insert or numpad 5 in the middle, End or numpad 6 right, Delete or
 *     numpad 8 stretches it - saved at once. P (where the port can hold its
 *     game still: Game::can_pause) asks the port to pause (Paused).
 */

#ifndef PS2HOST_HUD_H
#define PS2HOST_HUD_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace ps2host {
namespace hud {

using u32 = std::uint32_t;

enum class Anchor { kCentre, kLeft, kRight, kStretch };

struct ElementId {
    u32 context = 0, object = 0;
    char cell[3] = {'?', '?', 0};        // row T M B, column L C R
    /* Rows top to bottom, each left to right (the editor's list). */
    bool operator<(const ElementId& o) const;
    bool operator==(const ElementId& o) const {
        return context == o.context && object == o.object && cell[0] == o.cell[0] && cell[1] == o.cell[1];
    }
    bool known() const { return object != 0; }
    /* Its setting's key: "hud.<context>.<object>.<cell>". */
    std::string Key() const;
};

/* What the port tells about its game (SetGame, once). */
struct Game {
    /* The port's own anchors, by key; one ending in '.' ("hud.<context>.<object>.")
       is the object's in every cell. */
    const std::pair<const char*, Anchor>* defaults = nullptr;
    std::size_t default_count = 0;
    float width = 640.0f, height = 448.0f;      // the 4:3 picture CellOf divides
    const char* (*name)(const ElementId&) = nullptr;    // an element's name in the panel ("" none)
    bool can_pause = false;                     // P holds the game still (Paused)
    const char* paused_what = "the game";       // what P holds still, in the panel and the log
};
void SetGame(const Game& game);

const char* AnchorName(Anchor anchor);

/* An element's anchor (`given`: a setting or the port's default named it).
   Looked up once, then remembered until the editor changes it. */
struct Anchoring {
    Anchor anchor;
    bool given;
};
Anchoring AnchoringOf(const ElementId& id);
Anchor AnchorOf(const ElementId& id);

/* The picture's width over 4:3's (1 for a 4:3 picture), and its inverse,
   what the 2D is brought in by (at most 1). */
float Wider();
float Squeeze();

/* x (about `centre`, `half` its distance to the 4:3 picture's edges) brought
   in by k towards the middle, or to the edge it is anchored to; stretched,
   as the game drew it. */
float Place(float x, float centre, float half, float k, Anchor anchor);
/* The same in GS units (sixteenths of a pixel), rounded. */
int PlaceGs(int x, int centre, int half, float k, Anchor anchor);

/* The cell of a point of the 4:3 picture, (0, 0) its top left (Game's
   width and height). */
void CellOf(float x, float y, char cell[3]);

// ------------------------------------------------------------ the editor

/* The editor's keys and panel, once a frame (as the 2D is drawn). */
void EditorInput();
bool EditorOn();
/* An element was drawn (the panel lists what is on screen). */
void Seen(const ElementId& id);
/* Whether to leave it out now: the chosen one blinks. */
bool Hidden(const ElementId& id);
/* P pressed in the editor: the port holds its game still till P again or
   the editor closes. */
bool Paused();

}  // namespace hud
}  // namespace ps2host

#endif  /* PS2HOST_HUD_H */
