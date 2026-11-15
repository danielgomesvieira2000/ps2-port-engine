# The VU1 translator: microprograms to GPU vertex functions

A native renderer (NATIVE_RENDERER.md) draws a VU1 microprogram's batches on the
GPU: the program's per-vertex arithmetic as a vertex shader, bit for bit, fed by
the batch's own vertices and a header of what it reads besides. Until November
2026 each program was specified by hand (Burnout 3's `build/native/spec_*.md`),
ported to C++, then to GLSL. `tools/vutrans` derives that per-vertex function
from the microcode itself, and writes the shader's arithmetic.

## How it works

1. **Run the program symbolically** (`vu1sym.py`). A `--vu-snapshot` holds
   VU1's state as each run began. vu1sym follows the interpreter's values and
   timing (vu.cpp: round toward zero, denormals flushed, the FMAC's and the
   divider's latencies, the MAC and CLIP flags landing late, delay slots), and
   every float lane, ACC lane, VI register, Q, P, I, flag and data-memory word
   carries an expression too: a shared DAG whose leaves are the run's starting
   memory and registers. Branches are recorded, each judged counter-like
   (loop counters, header counts) or data-dependent (a float or a flag in its
   condition: a clipper, a guard band). Its concrete values are checked
   against vu1emu and the snapshot's next run.
2. **Split the kicks into vertices** (`slice.py`): each XGKICKed GIF packet's
   tags (PACKED, REGLIST), each vertex's registers (ST/STQ, RGBAQ, XYZF2, UV,
   FOG) with the bits the GS reads.
3. **Find each vertex's function** (`generalize.py`): every vertex's words
   rewritten with their leaves as roles - a vertex input (TOP + stride x
   vertex + offset, the neighbours k-1 and k-2 a strip reads, groups of four,
   periodic lanes of an unrolled loop, an address the vertex itself holds: the
   bones' rows), or uniform (the same word for all: the header). Vertices with
   one canonical form share a function; `project` then takes what a shader
   reads (identities under data branches dropped, words masked to what the
   native path uses, assumptions the native path guarantees).
4. **Check it**: the emitted Python evaluator must reproduce every significant
   kicked word of every vertex of every tested run, bit for bit.
5. **Write the shader's arithmetic** (`emit.py`): a drop-in for the
   hand-written shader, reading its own buffers (the mesh's quadwords, the
   header), with gs_mesh_common.glsl's round-toward-zero helpers (fused
   multiply-adds, or Dekker products for Direct3D 11's NO_FMA). For a program
   whose header the port computes on the CPU (RenderWare's), `rwheader.py`
   binds each uniform to a header word by the expression's structure and
   `rwbind.py` merges what reads the same (exact rewrite rules; lane variants
   refitted; a mode's functions merged only where bit-equal on every vertex).

The generated GLSL lives in `gpu/shaders/generated/`, each file's header naming
the tool, the command and the snapshot. The hand-written shaders stay the
default; `PS2HOST_MESH_GENERATED` picks the generated ones (`1` all, `room`,
`chars`, `rw`).

## Where it stands (2026-11-15)

| Game | Program | Vertices checked | Generated | Pictures |
|---|---|---|---|---|
| Outbreak (both Files) | room (990887E7) | 44,710 | one function (104 ops) for every unclipped vertex | every scripted run as recorded |
| Outbreak | characters, rigid (6BD7E3E9) | 45,533 | one function (189 ops) | every scripted run as recorded |
| Burnout 3 | instanced strips (D4B61373, 34C3CF67) | 98,930 | 3 functions (modes A, B, lit C) | 8 runs as recorded, Vulkan and D3D11 |
| Burnout 3 | world paths A/B (2678EB26, 1332C99F, 01CE30D5, 0835A68B) | 41,959 | 5 functions, 7 modes | as above |
| Burnout 3 | cars (7E86891B) | 180,321 | analysed (111 -> 43 functions), not emitted | - |

Each was also shown to fail when wrong: one output shifted a pixel fails
zoo_street (41/49) and the race (57/67).

## Commands

    python engine/tools/vutrans/translate.py SNAP IMAGE[,IMAGE...] [--runs N] [--emit-py out.py] [--emit-glsl out.glsl]
    # Outbreak's (milestone 2), from a snapshot made with PS2HOST_VU_NATIVE=off --vu-snapshot zoo.snap 9000 3000:
    translate.py SNAP 990887E7 --runs 2000 --mask XYZF2.w=0xFF0 --assume V,4,0,3=0x7FFF --emit-glsl-mesh F
    translate.py SNAP 6BD7E3E9 --runs 2000 --mask XYZF2.w=0xFF0 --assume V,5,0,3=0x7FFF --binding gs_mesh_chars --emit-glsl-mesh F
    # Burnout 3's (milestone 3), race.snap:
    translate.py SNAP D4B61373,34C3CF67 --runs 5000 --mask XYZF2.w=0 --bind rw --emit-glsl-rw vu_rw_instanced.glsl --rw-name VuRwInstancedGenerated
    translate.py SNAP 2678EB26,1332C99F,01CE30D5,0835A68B --runs 5000 --no-data-branches --mask XYZF2.w=0 --bind rw ...

Run Python from a folder without scripts named after the standard library's
modules (the tool's own are not).

## What is left

- **Data-dependent work**: clippers (Outbreak's room at 0x0538, Burnout 3's
  world path C and sky F504D280: every sky vertex passes through its clipper,
  no two share a function) stay hand-written or on the CPU. The way on: lift
  the clipper's loop variables into inputs, or keep it a CPU pass.
- **The words the native path handles itself**: the guard band (ADC) and fog
  are masked, the shader keeping its own; the assumptions (`--assume`) are the
  native path's, taken on trust.
- **The CPU side**: what a run leaves for the next (registers, kept vertices,
  memory) and the payload are still the hand-written port's; generating them
  from the same DAG is the step that makes a new game's program native without
  hand work.
- **More snapshots**: each function so far came from one snapshot; they pass
  every scripted run, but a second scene's snapshot would widen the check.
- **Burnout 3's cars**: the env and lit passes share a header mode, and their
  remaining lane variants differ in arithmetic order (genuinely different
  functions); the skinning palette is not a role yet.
