# Renderer and runtime roadmap: ideas from other PS2 PC ports

Written 2026-10-29 after reading three other open-source PS2 PC projects and checking each idea against our own code and `games/burnout3/docs/PERFORMANCE_PLAN.md` (PP), so nothing already done or already tried comes back.

**Licences: take ideas, never code.** Tenkaichi3Decomp's port is AGPL-3.0 and rac-rs is GPL-3.0; neither can be copied into this MIT engine. ico-pc is MIT, but its code is written for its own RHI, so we borrow designs from it too, not files.

## The three projects in one line each

| Project | What it is | Why most of its speed doesn't carry over |
|---|---|---|
| [z3xox/Tenkaichi3Decomp](https://github.com/z3xox/Tenkaichi3Decomp) | Matching C decomp built natively. All 9 VU1 programs are recognised and replaced by GLSL vertex shaders. GS emulated, then replayed through SDL3 GPU. | It needs a finished decomp and hand-written shaders for a handful of VU programs. It isn't exact (0.06% of pixels differ). |
| [nathanialf/ico-pc](https://github.com/nathanialf/ico-pc) | Native decomp. The 5 VU1 programs are hand-ported to HLSL. No GS emulation: textures come from the game's own data. RHI over Vulkan, D3D12, D3D11 and GL. | There's no GS memory model to keep exact. Its CPU work is native C. |
| [AsobiCodes/rac-rs](https://github.com/AsobiCodes/rac-rs) (the launcher only downloads and installs it) | EE statically recompiled to Rust. The 3D world is re-rendered natively in wgpu from the game's own data, and only the 2D goes through a GS capture. | A per-game renderer. A generic GS-exact engine can't copy it. |

Our engine aims for exact output for any game, so the transferable parts are the **Vulkan techniques**, the **pacing rules** and a few **CPU tricks**. Those are below.

## What we already do (checked in code, so no work needed)

| Their technique | Ours |
|---|---|
| Fragment shader without `gl_FragDepth` so early-Z stays on (T3: iGPU 82 to 103 fps) | `gs.frag` never writes depth. |
| Feedback copy limited to the bounding box of what is read (T3 for DATE) | `gs_renderer.cpp:3468` copies only the texels read, plus 1 around. |
| Pipeline cache on disk, specialised pipelines built in the background (ico, rac) | `gpu_pipeline_cache.bin` plus 118 shipped hint keys, with the ubershader as stand-in (`gs_renderer.cpp:2111-2290`). |
| Pipeline lookup cache (T3: was 6% of CPU) | Last-key shortcut plus `std::map` (`gs_renderer.cpp:2242`). |
| Texture cache by content (T3: about 30% of their frame) | XXH3 per block with generations, LRU 4096, decode on the GPU. |
| Same-state draw merging and indexed strips (T3, ico) | `DrawTriangleRun` (renderer -21%). Strips sent as strips: tried, no gain (PP:1102). |
| Load/store `NONE`, passes kept open (ico) | Deferred passes, -41% passes (engine 56856b7). |
| Render thread with handed-over records (T3, rac) | GS thread, renderer thread and present thread. |
| Pipeline-cache file per build / adapter (ico, rac) | Done (`gpu_pipeline_cache.bin`). Naming it by build commit is in R9. |
| Interpolation for high refresh (all three) | `high_fps.h` |
| Frame skip when behind (T3) | **Not taken on purpose** (memory: "no fps step-down"). |

## The roadmap

The order follows our measured bottlenecks: own-copy feedback copies and passes on the GPU, the GS thread (OnDraw) and VU1 on the CPU. Every step is measured with `bench.py` and `PS2HOST_GPU_PROFILE` before and after, and run through `regress.py` on car_select, race_training and race (pixel-exact). Each new path gets an env switch to turn it off, plus a check mode where possible (rac-rs does this for every speedup and it pays off).

### Phase 1: GPU, small and contained (days)

**R1. DATE stencil setup limited to the run's bounding box**, from T3 `GsDraw_DateRegion`.
- `DateSetup` (`gs_renderer.cpp:2479`) clears stencil and draws over `renderArea = whole target`.
- Fix: limit `renderArea` and the setup quad to the bounding box of the draws in the DATE run, the same way the feedback copy already does.
- T3 measured iGPU at 2x going from 96 to 117 fps.
- First count how many DATE setups a race and a splitscreen frame have; if it's under ~5 per frame, drop this item.

**R2. Exact barrier stages instead of ALL_COMMANDS** (already on our own list, PP:1311).
- `Transition` (`gs_renderer.cpp:237`, `:247`) uses ALL_COMMANDS.
- There are about 200-410 barriers in a race frame and about 1,000 transitions in splitscreen. On tilers (Android) a wide barrier forces a full flush.
- Doubling barriers on the Intel iGPU cost nothing (PP:197), so measure this on Android and on the 4070 separately.

**R3. Frame store plan**, from ico-pc `rd_store_plan.c`.
- Scan the renderer's record list for a frame before replaying it. Find each depth/colour target's last use and set STORE_OP_NONE / DONT_CARE where the contents are never read again.
- Our deferred passes decide this per pass. This decides it per frame.
- Mostly an Android win: ico measured about 1 ms per full-frame touch on PowerVR.
- Needs a look-ahead over the records the renderer thread already has.

**R4. Pipeline map to a flat open-addressed table** (T3 4096-entry index).
- Small: the last-key shortcut catches most lookups.
- Do it only if `hotspots.py` shows `PipelineFor` or `std::map` above 1% on the renderer thread.

### Phase 2: own-copy feedback without copies (1-2 weeks; the biggest GPU lever)

**R5. Read the target inside the open pass.**
- Feedback copies today: about 73 per race frame and about 150 in splitscreen. Each one ends the pass, then does copy, barrier, re-begin (`gs_renderer.cpp:3455-3485`).
- Two paths, picked per device:
  - **`VK_KHR_dynamic_rendering_local_read`**: the target is bound as an input attachment, and a pipeline barrier inside the pass (`BY_REGION`) separates overlapping draws. This is ico-pc's route for fog depth and DATE alpha.
  - **`VK_EXT_rasterization_order_attachment_access`** or **`VK_EXT_fragment_shader_interlock`**: no barriers between draws. This lets the shader blend read the destination exactly, as the GS does per pixel. Use it for `shader_blend` (ico-pc's "exact blend" path).
  - Keep the copy path as the fallback and as the check mode (`PS2HOST_GPU_LOCAL_READ=0`).
- Traps from ico-pc:
  - Qualcomm's proprietary driver returns wrong colours from attachment reads (their issue 89). Use the copy there.
  - A sampled read with filtering or offsets, as opposed to the pixel's own texel, can't use local read. Keep the copy for those.
- Expected gain: fewer passes (each copy now costs about 2 passes), fewer transitions, and no copy bandwidth. Measure on the iGPU in splitscreen at 3x first, where draws are 73% of the GPU-bound frame.

**R6. Extended dynamic state 3** for blend enable, equation and colour-write mask, from ico-pc.
- Today blend, write mask and logic op are part of the pipeline key.
- Moving them to dynamic state cuts the number of pipelines. That means fewer first-run hitches (still an open item: "first-minutes pipeline hitch census") and a smaller hint list for each new game.
- **Trap:** RADV on RDNA3 decides dual-source blending from the blend factors alone, so a stale SRC1 equation striped the picture (ico-pc issues 146 and 154). Always set the draw's own full equation, never only what changed.
- Optional: `VK_EXT_graphics_pipeline_library` for fast linking when a pipeline is missing. Weigh this against simply having fewer pipelines after EDS3.

### Phase 3: submission and pacing (1 week)

**R7. Draw merge with a per-draw table**, from ico-pc `vuRunTake`.
- Today, runs of consecutive draws that differ only in a few parameters (scissor, constant colour, texture) are separate draws.
- Merge up to 16 of them: concatenate their vertex ranges, put each draw's parameters into a small buffer, and look them up in the vertex shader by a binary search on the vertex index.
- ico measured about 11 µs per draw on PowerVR. In splitscreen we have about 4,000 draws a frame.
- First measure how many of our draws break only on a push-constant change. If it's under ~20%, drop this item.

**R8. Timeline semaphores, `present_wait` and `FIFO_LATEST_READY`**, from ico-pc `vk_swapchain.c`.
- Fences and binary semaphores become one timeline per queue (simpler slot reuse, cheaper waits).
- `VK_KHR_present_wait` gives exact present times for the refresh lock (today it is inferred from the measured refresh).
- `VK_EXT_present_mode_fifo_latest_ready` gives no tearing with mailbox-like latency on VRR-less screens.
- ico-pc's slow-present rule: use the median of the last 8 presents, and skip presenting a frame that is already stale. That fits our refresh-lead and late-frame census without stepping the frame rate down.

**R9. Pipeline cache named by build and adapter**, from ico-pc and rac-rs.
- Delete caches from other builds instead of letting the driver reject them silently.
- Also from ico-pc: compile the whole known pipeline set in batches of 32 on cores−1 during the loading screen, with a careful policy (batches of 8, 1 thread) on Adreno.

### Phase 4: CPU side (GS thread and VU1 are still the frame in explosions and pile-ups)

**C1. Exact PS2 multiply/add in doubles**, from T3 `plat_fastvec.c`.
- A product of two 24-bit mantissas fits exactly in a double's 53 bits.
- Truncating the low 29 bits then gives the PS2's round-toward-zero result with no MXCSR switch and no software float. Special operands (denormals, overflow) take the slow route.
- Checked 2026-10-29: we already round in hardware. MXCSR is set to round-toward-zero (`runtime/include/ps2rt/vu_math.h:47-62`, `host/src/vu.cpp:927`, `runtime/src/eefpu.cpp`). That is cheaper than T3's software floats, which is what the trick replaces there.
- What's left: the cost of `_mm_setcsr` switches where host code and game code alternate. Count them with `hotspots.py` before doing anything. If they're under 1%, drop this item. If not, prove any replacement exact with an exhaustive or random-plus-edge test like T3's `fastvec_check.c`.

**C2. Race check on hand-over records**, from T3 `BT3_GS_RACE`.
- Hash each record block when it is handed to the renderer and again after it is drawn. A mismatch means the producer wrote into a record that was still in flight.
- This is a debug switch, not a speedup. It is the safety net needed before trying zero-copy hand-overs again (zero-copy PATH3 was reverted, fb3da96).

**C3. Hot-function natives with deopt**, rac-rs "speed hands".
- The hottest lifted functions are regenerated with registers held in locals and written back at exits. Memory access is one range test, and anything else falls back ("deopts") to the lifted code at that exact instruction.
- Measure first: the EE is rarely our bottleneck (GS thread and VU1 are). Use this only if `hotspots.py` shows one lifted function above ~5% in a scenario that drops frames.

**C4. Outlined memory helpers per access width**, rac-rs `--outline-mem`.
- About 2x less LLVM time with the same behaviour.
- This is a build-time win (PGO/ThinLTO builds are slow), not a run-time one. Try it on one game, check run time is unchanged, and keep it if builds get faster.

### Not taken, and why

| Idea | Source | Why not |
|---|---|---|
| VU1 programs replaced by hand-written vertex shaders | T3, ico | Not exact, and per game. Automatic VU1-to-compute translation is a research project; revisit only if VU1 in explosions stays the wall after the speculative VU1 threads (PP roadmap A). |
| Frame skip when behind | T3 | Hides hitches instead of fixing them (no fps step-down). |
| Native re-render of the 3D world | rac-rs | Per game; it throws away the GS model. |
| Prim memo for large repeated 2D prims | rac-rs | It fixed their CPU GS capture in menus. Our menus aren't a bottleneck, and our GS runs on the GPU. |
| Write-watch dirty-page mirror for a parser thread | rac-rs | Our GS thread already decodes off the EE thread. Revisit only if the EE thread waits on the GS hand-over. |
| Submit and present on a worker thread | rac-rs | We already present on our own thread. Theirs hung on AMD. |
| Second device or queue for decodes | rac-rs | Barriers and copies aren't the cost (PP:197). Revisit after R5 if decodes show up between passes. |
| Bindless texture array plus multi-draw-indirect | rac-rs | Part of roadmap C (GPU-side draw building). It belongs there, after B's remainder. |

## Order and exit checks

1. **Week 1:** R1, R2 and R4, each measured; drop any that's noise. Then C2, the safety net.
2. **Weeks 2-3:** R5 (local read and rasterization order), the biggest GPU lever. Exit check: exact on all three regressions, and the feedback count in a race (`g_feedback`) close to 0 on the 4070 and the iGPU.
3. **Week 4:** R6 (EDS3) plus R9 (pipeline set at load). Exit check: no pipeline builds after the loading screen on a fresh install.
4. **Week 5:** R8 (pacing), then R3 and R7 for Android.
5. **Any time:** C1 after the check, C4 when builds hurt.

Record each result in PERFORMANCE_PLAN.md, including the ones that fail, the same way B was recorded.
