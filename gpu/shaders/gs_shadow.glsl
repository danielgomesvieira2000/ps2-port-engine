// SHADOWS (gs_renderer.cpp's shadow pass): the characters' meshes drawn, each picture, into a depth image from
// their first point light (their header's vf11-vf13 y: the view's space, as their places are); the room's pixels
// (gs.frag's LitShade, the enhanced draws) back into the view's space - the characters' projection undone - and
// tested against it the next picture. Set 1 (bound for every draw): binding 1 the image, binding 2 these.
layout(set = 1, binding = 2, std140) uniform Shadow {
    mat4 light_clip;        // the view's space to the light's clip space
    mat4 unproject;         // a place's clip (the characters' projection's) back to the view's space
    vec4 light;             // xyz the light, the view's space; w 1 shadows on
    vec4 tuning;            // x the shade's strength, y the depth's bias, z the texel's size
} shadow;
