# SMAA

[SMAA](https://github.com/iryoku/smaa) (Jimenez et al., Enhanced Subpixel
Morphological Antialiasing), MIT licence (`LICENSE.txt`), from master
71c806a838bdd7d517df19192a20f0c61b3ca29d:

- `SMAA.glsl`: `SMAA.hlsl`, its three `discard;` written `SMAA_DISCARD;`
  (see its first lines) for compute shaders.
- `AreaTex.h`, `SearchTex.h`: the precomputed area and search textures
  (`Textures/`), unchanged.

Used by `gpu/shaders/smaa_*.comp` and `gpu/src/vulkan_display.cpp`
(ANTI-ALIASING: SMAA).
