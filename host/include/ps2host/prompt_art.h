/*
 * Button prompts' art, drawn rather than shipped: a port's own marks for a
 * game's prompt textures (texture_pack.h's overrides), from shapes and
 * strokes - no font, no image of anyone's. A key's cap (its arrow, mark or
 * short name), the DualShock's marks, an Xbox controller's letters and Menu
 * and View, a mouse with a button pressed; letters and digits as strokes.
 *
 * A Canvas is a texture's art, drawn `scale` times larger than its texels and
 * measured in texels throughout; Pixels() gives it as the texture pack takes
 * it (RGBA8, alpha as the GS's: 0x80 opaque). Where a game's textures keep
 * each mark (their boxes) is the game's: Burnout 3's prompts.cpp.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "ps2host/bindings.h"

namespace ps2host::prompt_art {

struct Point {
    float x, y;
};
using Line = std::vector<Point>;

struct Colour {
    float r, g, b;
};
constexpr Colour kWhite{255, 255, 255};
constexpr Colour kGrey{180, 180, 180};          // Burnout 3's START word
constexpr Colour kDark{68, 69, 69};             // its dark Start and Select
constexpr Colour kKeyEdge{28, 28, 28}, kKeyFace{150, 150, 150}, kKeyText{250, 250, 250};
// The DualShock's marks, as Burnout 3's art has them.
constexpr Colour kCrossBlue{169, 197, 255}, kTriangleTeal{107, 201, 204}, kCirclePink{255, 141, 166},
                 kSquareMagenta{250, 164, 251};
// An Xbox controller's letters, in its colours made as light as those marks.
constexpr Colour kXboxA{120, 210, 100}, kXboxB{255, 120, 120}, kXboxX{110, 170, 255}, kXboxY{245, 215, 90};

inline float Clamp01(float v) { return v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v; }
float SegmentDistance(float px, float py, Point a, Point b);
/* A rounded rectangle's signed distance (inside negative). */
float RoundRect(float px, float py, float x0, float y0, float x1, float y1, float radius);
/* A convex polygon's signed distance (near enough at its edges, where it counts). */
float Convex(float px, float py, const std::vector<Point>& points);

/* A texture's art, drawn `scale` times larger than its texels; in texels
   throughout. Colour premultiplied by coverage. */
class Canvas {
public:
    Canvas(int width, int height, int scale = 8)
        : scale_(scale), w_(width * scale), h_(height * scale), px_(std::size_t(w_) * h_ * 4, 0.0f) {}

    /* Paints `colour` where `distance` (texels, inside negative) says, within
       the box (x0, y0)-(x1, y1), anti-aliased over a pixel. */
    template <class Distance>
    void Paint(float x0, float y0, float x1, float y1, Colour colour, Distance distance) {
        const float k = float(scale_);
        const int a = std::max(0, int(std::floor(x0 * k)) - 1), b = std::min(w_, int(std::ceil(x1 * k)) + 1);
        const int c = std::max(0, int(std::floor(y0 * k)) - 1), d = std::min(h_, int(std::ceil(y1 * k)) + 1);
        for (int y = c; y < d; ++y)
            for (int x = a; x < b; ++x) {
                const float tx = (float(x) + 0.5f) / k, ty = (float(y) + 0.5f) / k;
                const float cover = Clamp01(0.5f - distance(tx, ty) * k);
                if (cover <= 0.0f) continue;
                float* p = &px_[(std::size_t(y) * w_ + x) * 4];
                p[0] = colour.r * cover + p[0] * (1.0f - cover);
                p[1] = colour.g * cover + p[1] * (1.0f - cover);
                p[2] = colour.b * cover + p[2] * (1.0f - cover);
                p[3] = cover + p[3] * (1.0f - cover);
            }
    }

    /* Lines `width` texels thick, round at their ends and joins: one shape
       (where they cross is no darker). */
    void Strokes(const std::vector<Line>& lines, float width, Colour colour);
    void Polygon(const std::vector<Point>& points, Colour colour);
    void RoundedBox(float x0, float y0, float x1, float y1, float radius, Colour colour);
    /* A rounded rectangle's outline, `width` thick inside its edge. */
    void RoundedOutline(float x0, float y0, float x1, float y1, float radius, float width, Colour colour);
    void Ring(float cx, float cy, float radius, float width, Colour colour);
    template <class Distance>
    void Shape(float x0, float y0, float x1, float y1, Colour colour, Distance distance) {
        Paint(x0, y0, x1, y1, colour, distance);
    }

    /* The art as the texture pack takes it: RGBA8, alpha as the GS's (0x80 opaque). */
    std::vector<std::uint32_t> Pixels() const;
    int Width() const { return w_; }
    int Height() const { return h_; }

private:
    int scale_, w_, h_;
    std::vector<float> px_;
};

/* Points along an ellipse's arc (angles in degrees, y down: 90 is the bottom). */
Line Arc(float cx, float cy, float rx, float ry, float a0, float a1, int steps = 24);
Line Join(Line a, const Line& b);

struct Glyph {
    std::vector<Line> lines;        // in a unit box, x right, y down
    float width = 1.0f;             // against its height
};
/* The strokes of a letter (capitals), digit or mark the keys' caps use. */
Glyph GlyphOf(char ch);
/* Words centred in the box (x0, y0)-(x1, y1), `part` of its height (smaller
   if they would not fit across it), strokes `stroke` thick (0: by their size). */
void Text(Canvas& c, const std::string& word, float x0, float y0, float x1, float y1, Colour colour, float part,
          float stroke = 0.0f);

struct Box {
    float x0, y0, x1, y1;
    float W() const { return x1 - x0; }
    float H() const { return y1 - y0; }
    float Cx() const { return (x0 + x1) / 2; }
    float Cy() const { return (y0 + y1) / 2; }
    /* The largest square in the middle, `part` of its side. */
    Box Square(float part = 1.0f) const {
        const float s = std::min(W(), H()) * part;
        return {Cx() - s / 2, Cy() - s / 2, Cx() + s / 2, Cy() + s / 2};
    }
};
Point At(const Box& b, float u, float v);
std::vector<Point> Placed(const Box& b, std::initializer_list<Point> unit);

// The marks, each in a cell (`width`: their strokes').
void EnterMark(Canvas& c, const Box& cell, Colour colour, float width);        // Enter's bent arrow
void BackspaceMark(Canvas& c, const Box& cell, Colour colour, float width);    // an arrow pointing left
/* An arrow key's arrow: a triangle pointing `way` (0 up, 1 right, 2 down, 3 left). */
void ArrowMark(Canvas& c, const Box& cell, int way, Colour colour);
void SpaceMark(Canvas& c, const Box& cell, Colour colour, float width);        // a wide bracket opening up
void ShiftMark(Canvas& c, const Box& cell, Colour colour, float width);        // a hollow arrow pointing up
/* A mouse, its left (0) or right (1) button filled. */
void MouseMark(Canvas& c, const Box& cell, int button, Colour colour);
void CrossMark(Canvas& c, const Box& cell, Colour colour, float width);
void TriangleMark(Canvas& c, const Box& cell, Colour colour, float width);
void CircleMark(Canvas& c, const Box& cell, Colour colour, float width);
void SquareMark(Canvas& c, const Box& cell, Colour colour, float width);
/* Start: a triangle pointing right; Select: a rectangle on its side (filled: the dark ones). */
void StartMark(Canvas& c, const Box& cell, Colour colour, float width, bool filled);
void SelectMark(Canvas& c, const Box& cell, Colour colour, float width, bool filled);
/* An Xbox controller's Menu (three lines) and View (two windows) buttons. */
void MenuMark(Canvas& c, const Box& cell, Colour colour, float width);
void ViewMark(Canvas& c, const Box& cell, Colour colour, float width);

// DualShock 2 inputs (libpad's order).
constexpr int kSelect = 0, kL3 = 1, kR3 = 2, kStart = 3, kUp = 4, kRight = 5, kDown = 6, kLeft = 7, kL2 = 8,
              kR2 = 9, kL1 = 10, kR1 = 11, kTriangle = 12, kCircle = 13, kCross = 14, kSquare = 15;

/* How a cell is drawn: its mark's own colours (face buttons), or all one. */
enum class Tint { kOwn, kWhite, kDark, kGrey, kKey };
Colour TintColour(Tint tint);

/* A key's cap (a Windows virtual-key code, platform.h's): a mark, or its short name. */
void KeyCap(Canvas& c, const Box& cell, int vk, Tint tint, float stroke);
/* A controller's button `button` (by its DualShock 2 place) in the style. */
void Button(Canvas& c, const Box& cell, int button, bindings::Style style, Tint tint, float stroke);

}  // namespace ps2host::prompt_art
