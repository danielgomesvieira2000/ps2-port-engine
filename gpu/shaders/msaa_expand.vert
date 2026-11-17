#version 450
// A target's multisampled copy filled again from its single-sample image
// (gs_renderer.cpp's ExpandTwins, ANTI-ALIASING's MSAA): one triangle over the
// whole of it.

void main() {
    const vec2 at = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(at * 2.0 - 1.0, 0.0, 1.0);
}
