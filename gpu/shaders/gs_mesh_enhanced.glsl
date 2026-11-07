// The enhanced native draws (PC SETTINGS' GRAPHICS: LIGHTING, GEOMETRY): gs_mesh.vert and gs_mesh_chars.vert
// built with ENHANCED. Not the GS's picture to the pixel - the game's own meshes drawn as a modern renderer
// would: each vertex's place homogeneous (the GPU clips a triangle across the guard band itself, as the
// microcode's clipper did on the CPU, and interpolates perspective-correctly), its X and Y the GS's sixteenths
// or (PRECISE) not rounded at all, and the lighting's inputs passed on for gs.frag (MESH_LIT) to light each
// pixel with the microprogram's own lights.
//
// The header's ext (12 quadwords, after the kept vertices): [0] x the kept vertices known (bits 0, 1: the
// game's), y the enhancements (bits: 1 per-pixel lighting, 2 precise places - the renderer's); then per kept
// vertex five: its guard place, its place before the divide (the view's x y z w), its lighting place, its
// normal, its colour (the room's).

layout(location = 4) out vec4 lit_place;     // the lighting's place; w the normal's length
layout(location = 5) out vec3 lit_normal;
layout(location = 6) out vec4 lit_colour;    // the room's vertex colour (its lighting's input)
layout(location = 7) out float lit_on;       // 1: lit per pixel (0 at a vertex whose inputs are not known)
layout(location = 8) flat out uint lit_program;     // 0 the room's lights, 1 the characters'
layout(location = 9) flat out vec3 lit_eye;         // the camera, where lit_place is (a highlight's view)
layout(location = 10) out vec3 lit_view;            // the place in the view's space (SHADOWS: gs_shadow.glsl)

// The camera where a place is, from the place's matrix to the screen's (`r0`-`r3`: the rows multiplying x, y, z
// and 1): the point whose x, y and w come out 0.
vec3 EyeOf(vec4 r0, vec4 r1, vec4 r2, vec4 r3) {
    const mat3 m = mat3(vec3(r0.x, r0.y, r0.w), vec3(r1.x, r1.y, r1.w), vec3(r2.x, r2.y, r2.w));
    return inverse(m) * -vec3(r3.x, r3.y, r3.w);
}

const uint kLitPixels = 1u, kPrecise = 2u;

// A place before the divide (`cp`: the view's x y z w) as the GPU's clip space: the same pixel and Z as the GS's
// for a point in front (the GS's rounded `position` and `z` when `snap`), the GPU's division and clipping for
// any. (Z is the GS's 3D range, under kDepthTop: DepthOfZ a straight line.) Rounded only inside the guard band
// (`inside`): past it the GS's X and Y wrap round their 16 bits - a triangle across it, the GPU's to clip, has
// such a vertex.
vec4 ClipPlace(vec4 cp, vec2 offset, vec2 size, vec2 position, uint z, bool snap, bool inside) {
    const float w = cp.w;
    if (snap && inside && w > 0.0) {
        const vec2 ndc = (position + 0.5) / size * 2.0 - 1.0;
        return vec4(ndc * w, DepthOfZ(z) * w, w);
    }
    const vec2 at = (cp.xy - (offset / 16.0 - 0.5) * w) / size * 2.0 - w;
    return vec4(at, cp.z * (1.0 / 8589934592.0), w);
}

// Inside the guard band (CLIP's test: no |x|, |y|, |z| past |w|).
bool InGuard(vec4 g) { return max(abs(g.x), max(abs(g.y), abs(g.z))) <= abs(g.w); }
