# Native rendering for the engine's games

Branch `native-renderer`. 2026-11-06. The first game is Resident Evil
Outbreak (`games/re_outbreak2/docs/NATIVE_RENDERER.md` has its plan and its
measurements).

The goal is **performance and flexibility for every game the engine runs**,
not one game's renderer. This document decides how.

## What costs, measured

In Outbreak File #2's gameplay (RTX 4070 Laptop), the thread that runs
VIF1 and VU1 takes 22-24% of a core at 30 pictures a second, 46% at 60, 95%
at 120 and 97-99% at 144, where the game waits on it. Each picture runs
every thing's VU1 microprogram again. The GS thread follows (7% to 47%), and
the GPU is mostly idle (8-48%). The wall is the CPU's emulation of the PS2's
vector unit and GS, not the GPU.

## Three ways it could be done

| level | what is replaced | per game? | what it buys |
| --- | --- | --- | --- |
| A. The hardware (today) | nothing: VIF1, VU1, the GS emulated, the GS drawn on Vulkan | no: every game | works for all; the CPU wall above |
| B. **The microprograms** | each VU1 microprogram by a native version, fed by the game's own VIF1 data | per microprogram: shared by every game that uses it | the CPU wall gone; high frame rates nearly free; still the game's own data and draw order |
| C. The scene | the game's objects drawn from its own structures (its models, its camera), hooked at its draw functions | per game | everything of B, plus object-level changes: new lighting, mouse look at the screen's rate, LOD, asset swaps |

**The decision: B first, built as an engine framework; C as an optional
layer a game adds on top; A always underneath as the default and the
fallback.**

Why B:

- **The wall is VU1.** B replaces exactly what costs. C replaces it too, but
  needs a whole game's object model understood before it gains anything.
- **It's the most reusable unit there is.** On the PS2 the microprograms
  belong to the game's engine or middleware, not to the game: every
  RenderWare game (Burnout 3 among them) runs RenderWare's, and both
  Outbreaks run Capcom's. A native program is keyed by the microcode's hash.
  Write it once and every game that uploads the same code gets it
  automatically.
- **It's incremental and safe.** A thing whose program has no native version
  is emulated as today, in the same picture. Nothing has to be complete to
  be useful, and any program can be switched off at run time.
- **Its inputs are exact.** The game already put the matrices, vertices,
  colours and GIF tags in the list. A native program reads what the
  microprogram would have read, so there's no game structure to reverse
  engineer, and comparing against the emulated output is direct.
- **High frame rates get simpler.** Between two steps, only matrices change.
  The meshes stay on the GPU, and a picture between steps runs the native
  programs again with interpolated matrices, on the GPU. List blending's
  special cases (fades, a double buffer's leftovers) stay with the emulated
  remainder only.

Why still C, later and per game: what B can't see is *meaning*. It knows a
"thing with matrices and a mesh", not "the player's camera" or "a
character's head". Features like mouse look at the screen's rate, RE4-style
cameras without latency, replaced models or new lights need the game's
objects. C plugs into the same framework (it fills the same scene) for the
games where those are worth it.

## The framework (engine)

```
game's VIF1 list ──► segmenter ──► program registry ──► native program? ──yes──► native draws (GPU)
                        │                                     │no                     │
                        │                                     ▼                       ▼
                        └──────────────────────────► emulated VIF1/VU1/GS ──► one picture, one depth
```

1. **The segmenter.** Each VIF1 chain is cut into things, as the high-fps
   blender already does: each MSCAL/MSCNT with the UNPACKs before it, and the
   microprogram in VU1 memory at that moment, known by the hash of its code
   (from the MPGs seen). `display_list.cpp` has the parsing; this generalises
   it.
2. **The program registry.** Native programs are registered by microcode
   hash, from the engine (middleware: RenderWare, CRI) or from a game's port.
   Each one says which VU memory it reads (its inputs' layout), what it emits
   (primitives and their GIF tag), and its shader.
3. **Native programs.** The program's work, done on the GPU: transform,
   skinning, lighting, fog, clipping into primitives. A compute pass expands
   the thing's unpacked data into vertices, then one draw per GIF tag the
   program would have kicked, with the GS state (texture, CLUT, blend, test)
   the list's DIRECT packets set. The GS's state machine stays on the CPU:
   it's cheap, and it keeps every game's textures, render targets and effects
   exact.
4. **Composition.** A native thing is drawn at its place in the list's
   order, into the same GS render target and depth: the segment's words are
   replaced by a marker packet (as AMBIENT OCCLUSION's packet already is,
   `GsEffectId`) where the GS renderer draws the native batch. Blending
   order, render to texture and later passes are unchanged.
5. **Identity and interpolation.** Each thing keeps an identity across steps
   (the blender's matching), so a picture between steps interpolates its
   matrices on the GPU without touching the list.
6. **Checking.** For each program: `PS2HOST_NATIVE=off|on|check` and a
   per-hash switch. `check` draws both and records the difference (per-pixel
   maximum and SSIM per thing). The regression runs compare pictures with a
   threshold per program instead of a CRC.
7. **Measuring.** VU1 time per microprogram hash, and things and pixels per
   hash per frame (the census), so the next program to write is always the
   one that costs most.

## The game's layer (optional, per game: level C)

A port can also describe objects it understands (hooks at its draw
functions) into the same scene: a camera to draw at the screen's rate from
the newest input, objects to replace, lights to add. It can mark things as
its own so they aren't drawn twice. Outbreak's RE4 and RE7 cameras are the
first users.

## The rest of the emulation, the same way

The same rule applies to every emulated part: replace it at the level where
the code is shared, keyed by what identifies it, with the emulated version
underneath.

| part | shared level | native replacement |
| --- | --- | --- |
| VU1 microprograms | engine or middleware microcode (by hash) | native programs (above) |
| IPU (movies: MPEG-2) | Sofdec, the PS2's own MPEG library: most games | a native MPEG-2 decoder to a texture |
| SPU2 and the IOP's sound driver | sound middleware (per publisher's driver) | a native mixer per driver, keyed by the IOP module's hash; SPU2 emulated underneath |
| CDVD, memory card, pad | the SDK's libraries (already high level) | already native |
| The EE's threads, vblank, timers | the game's own logic | kept: it's the game's timing, not hardware |

## Order

1. **Measure** (engine): VU1 time per microprogram hash, the census per
   hash, the comparison tool. Outbreak's first.
2. **Build the framework** (engine) with one native program end to end:
   the segmenter, the registry, the compute pass and draws, composition,
   `check`. Exit check: one program native in Outbreak, its pictures within
   the threshold, its VU1 time gone.
3. **Outbreak's programs**, costliest first (the room and props, then
   characters) until VU1 is idle in gameplay. Exit check: the VIF1 worker
   under 10% at 144.
4. **A second engine's programs**, to prove the reuse: Burnout 3
   (RenderWare) or Mister Mosquito, whichever's census shows more shared
   microcode.
5. **Level C for Outbreak**: the camera at the screen's rate (mouse look
   without latency).
6. **Movies and sound** as in the table.
