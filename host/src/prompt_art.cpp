/*
 * Button prompts' art (prompt_art.h): shapes, strokes, letters and marks.
 * From Burnout 3's prompts.cpp, where the textures it draws are laid out.
 */

#include "ps2host/prompt_art.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "ps2host/platform.h"      // the keys' codes

namespace ps2host::prompt_art {


float SegmentDistance(float px, float py, Point a, Point b) {
    const float dx = b.x - a.x, dy = b.y - a.y;
    const float length = dx * dx + dy * dy;
    const float t = length > 0.0f ? Clamp01(((px - a.x) * dx + (py - a.y) * dy) / length) : 0.0f;
    const float ex = px - (a.x + t * dx), ey = py - (a.y + t * dy);
    return std::sqrt(ex * ex + ey * ey);
}

/* A rounded rectangle's signed distance (inside negative). */
float RoundRect(float px, float py, float x0, float y0, float x1, float y1, float radius) {
    const float cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const float qx = std::abs(px - cx) - ((x1 - x0) / 2 - radius), qy = std::abs(py - cy) - ((y1 - y0) / 2 - radius);
    const float ox = std::max(qx, 0.0f), oy = std::max(qy, 0.0f);
    return std::sqrt(ox * ox + oy * oy) + std::min(std::max(qx, qy), 0.0f) - radius;
}

/* A convex polygon's signed distance (near enough at its edges, where it counts). */
float Convex(float px, float py, const std::vector<Point>& points) {
    float area = 0.0f;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Point a = points[i], b = points[(i + 1) % points.size()];
        area += a.x * b.y - b.x * a.y;
    }
    const float side = area > 0.0f ? 1.0f : -1.0f;
    float d = -1e9f;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Point a = points[i], b = points[(i + 1) % points.size()];
        const float ex = b.x - a.x, ey = b.y - a.y, length = std::sqrt(ex * ex + ey * ey);
        if (length <= 0.0f) continue;
        // Outward: to the right of an edge going round anticlockwise (y down).
        d = std::max(d, side * ((px - a.x) * ey - (py - a.y) * ex) / length);
    }
    return d;
}

void Canvas::Strokes(const std::vector<Line>& lines, float width, Colour colour) {
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (const Line& line : lines)
        for (const Point& p : line) {
            x0 = std::min(x0, p.x), y0 = std::min(y0, p.y);
            x1 = std::max(x1, p.x), y1 = std::max(y1, p.y);
        }
    const float r = width / 2;
    Paint(x0 - r, y0 - r, x1 + r, y1 + r, colour, [&](float x, float y) {
        float d = 1e9f;
        for (const Line& line : lines) {
            if (line.size() == 1) d = std::min(d, SegmentDistance(x, y, line[0], line[0]));
            for (std::size_t i = 1; i < line.size(); ++i) d = std::min(d, SegmentDistance(x, y, line[i - 1], line[i]));
        }
        return d - r;
    });
}

void Canvas::Polygon(const std::vector<Point>& points, Colour colour) {
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (const Point& p : points) {
        x0 = std::min(x0, p.x), y0 = std::min(y0, p.y);
        x1 = std::max(x1, p.x), y1 = std::max(y1, p.y);
    }
    Paint(x0, y0, x1, y1, colour, [&](float x, float y) { return Convex(x, y, points); });
}

void Canvas::RoundedBox(float x0, float y0, float x1, float y1, float radius, Colour colour) {
    Paint(x0, y0, x1, y1, colour, [&](float x, float y) { return RoundRect(x, y, x0, y0, x1, y1, radius); });
}

void Canvas::RoundedOutline(float x0, float y0, float x1, float y1, float radius, float width, Colour colour) {
    const float h = width / 2;
    Paint(x0, y0, x1, y1, colour, [&](float x, float y) {
        return std::abs(RoundRect(x, y, x0 + h, y0 + h, x1 - h, y1 - h, std::max(0.0f, radius - h))) - h;
    });
}

void Canvas::Ring(float cx, float cy, float radius, float width, Colour colour) {
    Paint(cx - radius - width, cy - radius - width, cx + radius + width, cy + radius + width, colour,
          [&](float x, float y) { return std::abs(std::hypot(x - cx, y - cy) - radius) - width / 2; });
}

std::vector<std::uint32_t> Canvas::Pixels() const {
    std::vector<std::uint32_t> out(std::size_t(w_) * h_);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float* p = &px_[i * 4];
        const float a = p[3];
        if (a <= 0.0f) continue;
        auto channel = [&](float v) { return std::uint32_t(std::lround(std::min(255.0f, v / a))); };
        const std::uint32_t alpha = std::uint32_t(std::lround(a * 128.0f));
        out[i] = channel(p[0]) | channel(p[1]) << 8 | channel(p[2]) << 16 | alpha << 24;
    }
    return out;
}

Line Arc(float cx, float cy, float rx, float ry, float a0, float a1, int steps) {
    Line out;
    for (int i = 0; i <= steps; ++i) {
        const float a = (a0 + (a1 - a0) * float(i) / float(steps)) * 3.14159265f / 180.0f;
        out.push_back({cx + rx * std::cos(a), cy + ry * std::sin(a)});
    }
    return out;
}

Line Join(Line a, const Line& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

Glyph GlyphOf(char ch) {
    switch (ch) {
        case 'A': return {{{{0, 1}, {0.5f, 0}, {1, 1}}, {{0.22f, 0.62f}, {0.78f, 0.62f}}}};
        case 'B':
            return {{Join(Join({{0, 1}, {0, 0}, {0.52f, 0}}, Arc(0.52f, 0.25f, 0.4f, 0.25f, -90, 90)), {{0, 0.5f}}),
                     Join(Join({{0, 0.5f}, {0.56f, 0.5f}}, Arc(0.56f, 0.75f, 0.44f, 0.25f, -90, 90)), {{0, 1}})},
                    0.8f};
        case 'C': return {{Arc(0.55f, 0.5f, 0.5f, 0.5f, -50, -310)}, 0.85f};
        case 'D': return {{Join(Join({{0, 0}, {0, 1}, {0.4f, 1}}, Arc(0.4f, 0.5f, 0.6f, 0.5f, 90, -90)), {{0, 0}})}, 0.9f};
        case 'E': return {{{{1, 0}, {0, 0}, {0, 1}, {1, 1}}, {{0, 0.5f}, {0.8f, 0.5f}}}, 0.75f};
        case 'F': return {{{{1, 0}, {0, 0}, {0, 1}}, {{0, 0.5f}, {0.8f, 0.5f}}}, 0.72f};
        case 'G': return {{Join(Arc(0.5f, 0.5f, 0.5f, 0.5f, -40, -360), {{0.55f, 0.5f}})}, 0.9f};
        case 'H': return {{{{0, 0}, {0, 1}}, {{1, 0}, {1, 1}}, {{0, 0.5f}, {1, 0.5f}}}, 0.85f};
        case 'I': return {{{{0.5f, 0}, {0.5f, 1}}, {{0.1f, 0}, {0.9f, 0}}, {{0.1f, 1}, {0.9f, 1}}}, 0.5f};
        case 'J': return {{Join({{0.85f, 0}, {0.85f, 0.68f}}, Arc(0.45f, 0.68f, 0.4f, 0.32f, 0, 165))}, 0.75f};
        case 'K': return {{{{0, 0}, {0, 1}}, {{0.95f, 0}, {0, 0.62f}}, {{0.32f, 0.4f}, {1, 1}}}, 0.82f};
        case 'L': return {{{{0, 0}, {0, 1}, {0.9f, 1}}}, 0.7f};
        case 'M': return {{{{0, 1}, {0, 0}, {0.5f, 0.65f}, {1, 0}, {1, 1}}}, 1.05f};
        case 'N': return {{{{0, 1}, {0, 0}, {1, 1}, {1, 0}}}, 0.88f};
        case 'O': return {{Arc(0.5f, 0.5f, 0.5f, 0.5f, 0, 360, 36)}, 0.95f};
        case 'P': return {{Join(Join({{0, 1}, {0, 0}, {0.56f, 0}}, Arc(0.56f, 0.27f, 0.44f, 0.27f, -90, 90)), {{0, 0.54f}})}, 0.8f};
        case 'Q': return {{Arc(0.5f, 0.48f, 0.5f, 0.48f, 0, 360, 36), {{0.6f, 0.65f}, {1.0f, 1.05f}}}, 0.95f};
        case 'R':
            return {{Join(Join({{0, 1}, {0, 0}, {0.6f, 0}}, Arc(0.6f, 0.25f, 0.38f, 0.25f, -90, 90)), {{0, 0.5f}}),
                     {{0.5f, 0.5f}, {1, 1}}},
                    0.82f};
        case 'S': return {{Join(Arc(0.5f, 0.25f, 0.45f, 0.25f, -20, -270), Arc(0.5f, 0.75f, 0.48f, 0.25f, -90, 160))}, 0.8f};
        case 'T': return {{{{0, 0}, {1, 0}}, {{0.5f, 0}, {0.5f, 1}}}, 0.85f};
        case 'U': return {{Join(Join({{0, 0}, {0, 0.62f}}, Arc(0.5f, 0.62f, 0.5f, 0.38f, 180, 0)), {{1, 0}})}, 0.85f};
        case 'V': return {{{{0, 0}, {0.5f, 1}, {1, 0}}}};
        case 'W': return {{{{0, 0}, {0.25f, 1}, {0.5f, 0.3f}, {0.75f, 1}, {1, 0}}}, 1.25f};
        case 'X': return {{{{0, 0}, {1, 1}}, {{1, 0}, {0, 1}}}};
        case 'Y': return {{{{0, 0}, {0.5f, 0.5f}, {1, 0}}, {{0.5f, 0.5f}, {0.5f, 1}}}};
        case 'Z': return {{{{0, 0}, {1, 0}, {0, 1}, {1, 1}}}};
        case '0': return {{Arc(0.5f, 0.5f, 0.5f, 0.5f, 0, 360, 36)}, 0.75f};
        case '1': return {{{{0.15f, 0.25f}, {0.6f, 0}, {0.6f, 1}}}, 0.55f};
        case '2': return {{Join(Arc(0.5f, 0.28f, 0.45f, 0.28f, -165, 25), {{0, 1}, {1, 1}})}, 0.8f};
        case '3': return {{Join(Arc(0.45f, 0.27f, 0.45f, 0.27f, -160, 90), Arc(0.45f, 0.75f, 0.47f, 0.25f, -90, 160))}, 0.8f};
        case '4': return {{{{0.72f, 1}, {0.72f, 0}, {0, 0.68f}, {1, 0.68f}}}, 0.85f};
        case '5': return {{Join({{0.92f, 0}, {0.15f, 0}, {0.07f, 0.5f}}, Arc(0.48f, 0.67f, 0.47f, 0.33f, -150, 150))}, 0.8f};
        case '6': return {{Arc(0.5f, 0.68f, 0.45f, 0.68f, -60, -180), Arc(0.5f, 0.68f, 0.45f, 0.32f, 0, 360, 30)}, 0.8f};
        case '7': return {{{{0, 0}, {1, 0}, {0.35f, 1}}}, 0.8f};
        case '8': return {{Arc(0.5f, 0.26f, 0.4f, 0.26f, 0, 360, 30), Arc(0.5f, 0.74f, 0.47f, 0.26f, 0, 360, 30)}, 0.8f};
        case '9': return {{Arc(0.5f, 0.32f, 0.45f, 0.32f, 0, 360, 30), Arc(0.5f, 0.32f, 0.45f, 0.68f, 0, 120)}, 0.8f};
        case ';': return {{{{0.5f, 0.3f}}, {{0.5f, 0.8f}, {0.35f, 1.05f}}}, 0.35f};
        case ',': return {{{{0.55f, 0.8f}, {0.35f, 1.05f}}}, 0.35f};
        case '.': return {{{{0.5f, 0.92f}}}, 0.3f};
        case '=': return {{{{0, 0.33f}, {1, 0.33f}}, {{0, 0.67f}, {1, 0.67f}}}, 0.8f};
        case '-': return {{{{0.1f, 0.5f}, {0.9f, 0.5f}}}, 0.7f};
        case '+': return {{{{0, 0.5f}, {1, 0.5f}}, {{0.5f, 0.12f}, {0.5f, 0.88f}}}, 0.8f};
        case '*':
            return {{{{0.5f, 0.15f}, {0.5f, 0.85f}}, {{0.2f, 0.32f}, {0.8f, 0.68f}}, {{0.2f, 0.68f}, {0.8f, 0.32f}}}, 0.8f};
        case '/': return {{{{0.9f, 0}, {0.1f, 1}}}, 0.6f};
        case '\\': return {{{{0.1f, 0}, {0.9f, 1}}}, 0.6f};
        case '[': return {{{{0.8f, 0}, {0.3f, 0}, {0.3f, 1}, {0.8f, 1}}}, 0.5f};
        case ']': return {{{{0.2f, 0}, {0.7f, 0}, {0.7f, 1}, {0.2f, 1}}}, 0.5f};
        case '`': return {{{{0.3f, 0}, {0.65f, 0.3f}}}, 0.5f};
        case '\'': return {{{{0.5f, 0}, {0.5f, 0.3f}}}, 0.3f};
        default: return {{}, 0.5f};     // a space
    }
}

void Text(Canvas& c, const std::string& word, float x0, float y0, float x1, float y1, Colour colour, float part,
          float stroke) {
    if (word.empty()) return;
    float h = (y1 - y0) * part, gap = h * 0.22f;
    std::vector<Glyph> glyphs;
    float total = 0.0f;
    for (char ch : word) {
        glyphs.push_back(GlyphOf(ch));
        total += h * 0.78f * glyphs.back().width;
    }
    total += gap * float(word.size() - 1);
    float scale = 1.0f;
    if (total > (x1 - x0) * 0.9f) scale = (x1 - x0) * 0.9f / total;
    h *= scale, gap *= scale, total *= scale;
    const float width = stroke > 0.0f ? stroke * std::min(1.0f, scale * 1.4f) : std::max(0.45f, h * 0.17f);
    float x = (x0 + x1) / 2 - total / 2;
    const float y = (y0 + y1) / 2 - h / 2;
    for (const Glyph& g : glyphs) {
        const float w = h * 0.78f * g.width;
        std::vector<Line> lines;
        for (const Line& line : g.lines) {
            Line placed;
            for (const Point& p : line)
                placed.push_back({x + width / 2 + p.x * (w - width), y + width / 2 + p.y * (h - width)});
            lines.push_back(placed);
        }
        if (!lines.empty()) c.Strokes(lines, width, colour);
        x += w + gap;
    }
}

Point At(const Box& b, float u, float v) { return {b.x0 + u * b.W(), b.y0 + v * b.H()}; }

std::vector<Point> Placed(const Box& b, std::initializer_list<Point> unit) {
    std::vector<Point> out;
    for (const Point& p : unit) out.push_back(At(b, p.x, p.y));
    return out;
}

/* The Enter key's bent arrow. */
void EnterMark(Canvas& c, const Box& cell, Colour colour, float width) {
    const Box b = cell.Square(0.8f);
    c.Strokes({Placed(b, {{0.85f, 0.1f}, {0.85f, 0.6f}, {0.3f, 0.6f}})}, width, colour);
    c.Polygon(Placed(b, {{0.0f, 0.6f}, {0.35f, 0.3f}, {0.35f, 0.9f}}), colour);
}

/* The Backspace key's arrow, pointing left. */
void BackspaceMark(Canvas& c, const Box& cell, Colour colour, float width) {
    const float s = std::min(cell.W(), cell.H()) * 0.85f;
    const Box b{cell.Cx() - s * 0.7f, cell.Cy() - s / 2, cell.Cx() + s * 0.7f, cell.Cy() + s / 2};
    c.Strokes({Placed(b, {{1.0f, 0.5f}, {0.3f, 0.5f}})}, width, colour);
    c.Polygon(Placed(b, {{0.0f, 0.5f}, {0.35f, 0.15f}, {0.35f, 0.85f}}), colour);
}

/* An arrow key's arrow: a triangle pointing `way` (0 up, 1 right, 2 down, 3 left). */
void ArrowMark(Canvas& c, const Box& cell, int way, Colour colour) {
    const Box b = cell.Square(0.62f);
    static const Point kUp[3] = {{0.5f, 0.1f}, {0.95f, 0.85f}, {0.05f, 0.85f}};
    std::vector<Point> points;
    for (const Point& p : kUp) {
        float u = p.x, v = p.y;
        for (int i = 0; i < way; ++i) {
            const float t = u;
            u = 1.0f - v;
            v = t;
        }
        points.push_back(At(b, u, v));
    }
    c.Polygon(points, colour);
}

/* The space bar's mark: a wide bracket opening up. */
void SpaceMark(Canvas& c, const Box& cell, Colour colour, float width) {
    const float w = std::min(cell.W() * 0.86f, cell.H() * 1.1f), h = w * 0.32f;
    const Box b{cell.Cx() - w / 2, cell.Cy() - h / 2, cell.Cx() + w / 2, cell.Cy() + h / 2};
    c.Strokes({Placed(b, {{0, 0}, {0, 1}, {1, 1}, {1, 0}})}, width, colour);
}

/* Shift's mark: a hollow arrow pointing up. */
void ShiftMark(Canvas& c, const Box& cell, Colour colour, float width) {
    const Box b = cell.Square(0.78f);
    c.Strokes({Placed(b, {{0.5f, 0.02f}, {0.98f, 0.5f}, {0.72f, 0.5f}, {0.72f, 0.98f}, {0.28f, 0.98f}, {0.28f, 0.5f},
                          {0.02f, 0.5f}, {0.5f, 0.02f}})},
              width, colour);
}

/* A mouse, its left (0) or right (1) button filled. */
void MouseMark(Canvas& c, const Box& cell, int button, Colour colour) {
    const float h = cell.H() * 0.86f, w = h * 0.66f;
    const float left = cell.Cx() - w / 2, top = cell.Cy() - h / 2;
    const float stroke = std::max(0.8f, h * 0.09f), split = top + h * 0.45f;
    const float x0 = left + stroke / 2, x1 = left + w - stroke / 2, y0 = top + stroke / 2;
    c.RoundedOutline(left, top, left + w, top + h, w / 2, stroke, colour);
    c.Strokes({{{x0, split}, {x1, split}}, {{cell.Cx(), y0}, {cell.Cx(), split}}}, stroke, colour);
    // The button pressed, filled: the body's top on its side, a little in from the lines.
    const float gap = stroke * 1.15f;
    const float sx0 = button == 0 ? left : cell.Cx() + gap - stroke / 2, sx1 = button == 0 ? cell.Cx() - gap + stroke / 2 : left + w;
    c.Shape(left, top, left + w, split, colour, [&](float x, float y) {
        const float body = RoundRect(x, y, left + gap, top + gap, left + w - gap, top + h - gap, w / 2 - gap);
        const float side = RoundRect(x, y, sx0, top, sx1, split - gap + stroke / 2, 0.0f);
        return std::max(body, side);
    });
}

// The DualShock's marks: outlines in a box (stroke `width`).
void CrossMark(Canvas& c, const Box& cell, Colour colour, float width) {
    const Box b = cell.Square(0.8f);
    c.Strokes({Placed(b, {{0, 0}, {1, 1}}), Placed(b, {{1, 0}, {0, 1}})}, width, colour);
}
void TriangleMark(Canvas& c, const Box& cell, Colour colour, float width) {
    const Box b = cell.Square(0.86f);
    c.Strokes({Placed(b, {{0.5f, 0.06f}, {0.98f, 0.9f}, {0.02f, 0.9f}, {0.5f, 0.06f}})}, width, colour);
}
void CircleMark(Canvas& c, const Box& cell, Colour colour, float width) {
    const Box b = cell.Square(0.84f);
    c.Ring(b.Cx(), b.Cy(), b.W() / 2 - width / 2, width, colour);
}
void SquareMark(Canvas& c, const Box& cell, Colour colour, float width) {
    const Box b = cell.Square(0.8f);
    c.RoundedOutline(b.x0, b.y0, b.x1, b.y1, 0.0f, width, colour);
}
/* Start: a triangle pointing right (filled, for the dark ones). */
void StartMark(Canvas& c, const Box& cell, Colour colour, float width, bool filled) {
    const Box b = cell.Square(0.86f);
    const float h = b.H() * 0.84f;
    const Box t{b.Cx() - h * 0.45f, b.Cy() - h / 2, b.Cx() + h * 0.55f, b.Cy() + h / 2};
    if (filled) c.Polygon(Placed(t, {{0, 0}, {1, 0.5f}, {0, 1}}), colour);
    else c.Strokes({Placed(t, {{0, 0}, {1, 0.5f}, {0, 1}, {0, 0}})}, width, colour);
}
/* Select: a rectangle on its side. */
void SelectMark(Canvas& c, const Box& cell, Colour colour, float width, bool filled) {
    const float w = std::min(cell.W() * 0.86f, cell.H() * 1.2f), h = w * 0.62f;
    const Box b{cell.Cx() - w / 2, cell.Cy() - h / 2, cell.Cx() + w / 2, cell.Cy() + h / 2};
    if (filled) c.RoundedBox(b.x0, b.y0, b.x1, b.y1, 0.0f, colour);
    else c.RoundedOutline(b.x0, b.y0, b.x1, b.y1, 0.0f, width, colour);
}
/* An Xbox controller's Menu (three lines) and View (two windows) buttons. */
void MenuMark(Canvas& c, const Box& cell, Colour colour, float width) {
    const Box b = cell.Square(0.66f);
    c.Strokes({Placed(b, {{0.1f, 0.2f}, {0.9f, 0.2f}}), Placed(b, {{0.1f, 0.5f}, {0.9f, 0.5f}}),
               Placed(b, {{0.1f, 0.8f}, {0.9f, 0.8f}})},
              width, colour);
}
void ViewMark(Canvas& c, const Box& cell, Colour colour, float width) {
    const Box b = cell.Square(0.74f);
    c.RoundedOutline(b.x0, b.y0, b.x0 + b.W() * 0.66f, b.y0 + b.H() * 0.66f, 0.0f, width, colour);
    c.Strokes({Placed(b, {{0.46f, 0.66f}, {0.46f, 0.98f}, {0.98f, 0.98f}, {0.98f, 0.46f}, {0.66f, 0.46f}})}, width, colour);
}

Colour TintColour(Tint tint) {
    switch (tint) {
        case Tint::kDark: return kDark;
        case Tint::kGrey: return kGrey;
        case Tint::kKey: return kKeyText;
        default: return kWhite;
    }
}

void KeyCap(Canvas& c, const Box& cell, int vk, Tint tint, float stroke) {
    const Colour colour = TintColour(tint);
    const float part = tint == Tint::kGrey ? 0.85f : 0.78f;
    switch (vk) {
        case VK_RETURN: EnterMark(c, cell, colour, stroke); return;
        case VK_BACK: BackspaceMark(c, cell, colour, stroke); return;
        case VK_SPACE: SpaceMark(c, cell, colour, stroke); return;
        case VK_LSHIFT:
        case VK_RSHIFT: ShiftMark(c, cell, colour, stroke * 0.8f); return;
        case VK_UP: ArrowMark(c, cell, 0, colour); return;
        case VK_RIGHT: ArrowMark(c, cell, 1, colour); return;
        case VK_DOWN: ArrowMark(c, cell, 2, colour); return;
        case VK_LEFT: ArrowMark(c, cell, 3, colour); return;
        default: break;
    }
    std::string text;
    if ((vk >= 'A' && vk <= 'Z') || (vk >= '0' && vk <= '9')) text = std::string(1, char(vk));
    else if (vk >= VK_F1 && vk <= VK_F12) text = "F" + std::to_string(vk - VK_F1 + 1);
    else if (vk >= VK_NUMPAD0 && vk <= VK_NUMPAD9) text = std::string(1, char('0' + vk - VK_NUMPAD0));
    else switch (vk) {
        case VK_LCONTROL: case VK_RCONTROL: text = "CTRL"; break;
        case VK_LMENU: case VK_RMENU: text = "ALT"; break;
        case VK_CAPITAL: text = "CAPS"; break;
        case VK_INSERT: text = "INS"; break;
        case VK_DELETE: text = "DEL"; break;
        case VK_HOME: text = "HOME"; break;
        case VK_END: text = "END"; break;
        case VK_PRIOR: text = "PGUP"; break;
        case VK_NEXT: text = "PGDN"; break;
        case VK_DECIMAL: text = "."; break;
        case VK_ADD: text = "+"; break;
        case VK_SUBTRACT: text = "-"; break;
        case VK_MULTIPLY: text = "*"; break;
        case VK_DIVIDE: text = "/"; break;
        case VK_OEM_1: text = ";"; break;
        case VK_OEM_PLUS: text = "="; break;
        case VK_OEM_COMMA: text = ","; break;
        case VK_OEM_MINUS: text = "-"; break;
        case VK_OEM_PERIOD: text = "."; break;
        case VK_OEM_2: text = "/"; break;
        case VK_OEM_3: text = "`"; break;
        case VK_OEM_4: text = "["; break;
        case VK_OEM_5: text = "\\"; break;
        case VK_OEM_6: text = "]"; break;
        case VK_OEM_7: text = "'"; break;
        default: text = "?"; break;
    }
    // The number pad's: NUM over the digit or sign.
    const bool numpad = (vk >= VK_NUMPAD0 && vk <= VK_NUMPAD9) || vk == VK_DECIMAL || vk == VK_ADD ||
                        vk == VK_SUBTRACT || vk == VK_MULTIPLY || vk == VK_DIVIDE;
    if (numpad) {
        const float split = cell.y0 + cell.H() * 0.36f;
        Text(c, "NUM", cell.x0, cell.y0, cell.x1, split, colour, 0.8f);
        Text(c, text, cell.x0, split, cell.x1, cell.y1, colour, 0.82f);
        return;
    }
    Text(c, text, cell.x0, cell.y0, cell.x1, cell.y1, colour, part, text.size() == 1 ? 0.0f : 0.0f);
}

void Button(Canvas& c, const Box& cell, int button, bindings::Style style, Tint tint, float stroke) {
    const bool own = tint == Tint::kOwn || tint == Tint::kKey;
    const Colour plain = TintColour(tint);
    if (style == bindings::Style::kXbox) {
        // As large as the game's marks: the game draws the button's disc
        // round the cell. The top cell (Triangle's) sits high in its box -
        // the triangle's point is at its top - and a letter centred in it
        // looks high: it goes a texel lower.
        auto letter = [&](const char* text, Colour colour, float lower = 0.0f) {
            Text(c, text, cell.x0, cell.y0 + lower, cell.x1, cell.y1 + lower, own ? colour : plain, 0.9f,
                 stroke * 1.2f);
        };
        switch (button) {
            case kCross: letter("A", kXboxA); return;
            case kCircle: letter("B", kXboxB); return;
            case kSquare: letter("X", kXboxX); return;
            case kTriangle: letter("Y", kXboxY, 1.0f); return;
            case kStart: MenuMark(c, cell, plain, stroke); return;
            case kSelect: ViewMark(c, cell, plain, stroke); return;
            default: break;
        }
    } else {
        switch (button) {
            case kCross: CrossMark(c, cell, own ? kCrossBlue : plain, stroke); return;
            case kCircle: CircleMark(c, cell, own ? kCirclePink : plain, stroke); return;
            case kSquare: SquareMark(c, cell, own ? kSquareMagenta : plain, stroke); return;
            case kTriangle: TriangleMark(c, cell, own ? kTriangleTeal : plain, stroke); return;
            case kStart: StartMark(c, cell, plain, stroke, tint == Tint::kDark); return;
            case kSelect: SelectMark(c, cell, plain, stroke, tint == Tint::kDark); return;
            default: break;
        }
    }
    switch (button) {
        case kUp: ArrowMark(c, cell, 0, plain); return;
        case kRight: ArrowMark(c, cell, 1, plain); return;
        case kDown: ArrowMark(c, cell, 2, plain); return;
        case kLeft: ArrowMark(c, cell, 3, plain); return;
        default: break;
    }
    static const char* const kPlayStation[16] = {"SEL", "L3", "R3", "ST", "", "", "", "", "L2", "R2", "L1", "R1"};
    static const char* const kXbox[16] = {"", "LS", "RS", "", "", "", "", "", "LT", "RT", "LB", "RB"};
    if (button >= 0 && button < 12)
        Text(c, style == bindings::Style::kXbox ? kXbox[button] : kPlayStation[button], cell.x0, cell.y0, cell.x1, cell.y1,
             plain, 0.7f);
}

}  // namespace ps2host::prompt_art
