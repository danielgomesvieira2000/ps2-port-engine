# Graphics backends roadmap: Vulkan 1.2, then DirectX 11 (and maybe 12)

Written 2026-10-30, after issue #24 (burnout3-pc): a player whose GPU has no Vulkan 1.3 driver. Today the engine draws on the GPU only with Vulkan 1.3. Without it, the window falls back to GDI and the GS is drawn on the CPU, which is too slow on a weak PC.

The goal is that every PC able to run the game's CPU side also draws on its GPU, in this order: **Vulkan 1.3 → Vulkan 1.2 → Direct3D 11 → software GS + GDI**.

## Status (2026-10-31)

**Done: Vulkan 1.2 (V1-V4) and Direct3D 11 (X0-X4).** The order of choice is as planned: `renderer=auto` takes Vulkan (1.3, else 1.2), then Direct3D 11, then GDI with the software GS.

**Vulkan 1.2** (`vk_device.cpp`): as planned. A 1.2 device needs `VK_EXT_extended_dynamic_state`; it draws with `VK_KHR_dynamic_rendering` where offered, else with render passes (`gs_renderer.cpp`'s `BeginRendering`). `PS2HOST_VK_LEGACY=1|2` takes either path on a 1.3 GPU. Every Burnout 3 GPU regression is the same picture for picture in all three paths.

**Older devices (V5, and Vulkan 1.1):**
- **Vulkan 1.1** draws too, with render passes. Every shader is SPIR-V 1.3 (`--target-env vulkan1.1`): none used anything of 1.4-1.5, and the renderer uses no 1.2 core feature (only 1.0 layouts and enums).
- **Without `VK_EXT_extended_dynamic_state`** (older phones' drivers) the GS draws' depth test, write and comparison, their stencil and the ranges' stride are made into the pipeline (`BakedPipelineFor`: the key plus 7 bits of states), not set per batch. The worker makes them as it makes the dynamic ones, with the whole shader's pipeline with the same states standing in, and the last runs' from the start (`gpu_pipelines_baked.txt` beside `gpu_pipelines.txt`; two real-time runs: the second had the first's 8 ready in 72 ms). `PS2HOST_VK_BAKED=1` takes this on any device, and backends_check's `baked` backend is render passes with it.
- **Without logic ops** (`logicOp`: Mali, an older phone's GPU) a channel FBMSK masks in part was written whole: two renderer tests failed, as they always had on such GPUs. Now such a draw blends in the shader (`MaskedInShader`), which takes the bits FBMSK keeps from its copy of the target: exact, whatever the blend. `PS2HOST_GPU_NO_LOGIC_OP=1` takes this on any GPU: car_select, race_training, race and crash are then as recorded.
- **Checked:** the renderer tests under the profiles layer posing as a 1.2 device without the extension, a 1.1 device with it, and a 1.1 phone (no extension, no logic ops, no BC), with the validation layer above: all pass, no messages. On the phone profile race_training and race are as recorded, again with no messages. backends_check's `baked` backend (render passes, `PS2HOST_VK_BAKED=1`) is as recorded on car_select, race_training, race, crash and race_modern.
- **race_modern is flaky** on every path: a run now and then has one to six frames off, and the next is as recorded.

**Direct3D 11: one renderer, not two.** X3 planned a second renderer, ported from `gs_renderer.cpp` part by part. What was built instead is a layer that implements the part of Vulkan the renderer uses with Direct3D 11 (`d3d11_vulkan.cpp`, ~1,900 lines: images, views, descriptor sets, pipelines, command buffers run at submit, copies, read-backs). The Vulkan renderer, its effects and `VulkanGpu` run on it unchanged.

Why the change:
- **Exactness by construction.** Both backends follow the same rules, so the Vulkan recordings are the D3D11 recordings. The one exception is the PC's modern effects: their float arithmetic is rounded differently by the two shader compilers (fxc and the Vulkan driver's), in about 0.6% of pixels and mostly by a few levels. race_modern therefore has a `.d3d11.crc` set of its own; regress.py uses it under `PS2HOST_RENDERER=d3d11`.
- **No second renderer to keep in step** with every fix, game rule and feature.
- **It was the shorter road**: about 100 distinct Vulkan calls against a 4,000-line renderer.

How it maps:
- **Shaders:** the same GLSL. Its SPIR-V goes through SPIRV-Cross (`tools/spirv_to_hlsl.cpp`: set `s` binding `b` is register `s*7+b`, push constants are `b0` with the same byte layout) and then fxc at shader model 5.0. The layer finds each shader's DXBC by its SPIR-V.
  Specialisation constants are left at their defaults (the whole shader).
  Under `-DD3D11` the GLSL differs where Direct3D 11 cannot follow:
  - RGBA8 storage images are read as R32_UINT words (feature level 11_0 has no RGBA8 UAV loads);
  - gs.frag's 4-texel words gather becomes four fetches (the same texels);
  - a depth view's stencil is in `.g`;
  - the effects' `float p[22]` is read through `vec4`s;
  - logic ops use a `UINT_OUTPUT` variant of gs.frag, drawn through the target's R8G8B8A8_UINT view with Direct3D 11.1's logic ops.
- **Memory:** device memory is only a size; images are textures made at `vkCreateImage`. Host-visible memory is the process's own: a host buffer's drawn range is copied to its GPU twin once per submission, and GPU-to-host copies are read back through staging resources when the fence is waited for.
- **Depth uploads** (Direct3D 11 updates no depth-stencil resource) are drawn with `SV_Depth`.
- **Buffer-to-image copies on the GPU** are a small compute shader.

**The display** (`d3d11_display.cpp`, on `SdlWindowDisplay`, `window_display.cpp`):
- flip-model swapchain, tearing where allowed (VRR);
- FXAA, SMAA and SSAA through the same compute shaders;
- the overlay.

**Checked:**
- renderer tests on Direct3D 11 on the RTX 4070 and the Intel UHD;
- Burnout 3 GPU regressions on Direct3D 11 (`PS2HOST_RENDERER=d3d11`, headless): car_select 44/44, race_training 55/55, race 67/67, crash 77/77, road_rage 69/69, the same CRCs as Vulkan's. splitscreen is 50/52, the same two frames Vulkan misses. race_modern (2x, 120 fps, MODERN) matches its own set: 99.4% of pixels as Vulkan's.
- The trap race_modern found: with dynamic vertex strides, binding a pipeline must keep the strides already bound. The renderer binds its sprite ranges at stride 0 and leaves them bound, so the layer's reset made 2x sprites read garbage: a dark sky, the paint unreflected.

**Fixed: the eliminator difference** (102/103 on Direct3D 11, frame 16200's blotch on the rock face above the tunnel; now 103/103).
- **Found:** a capture of frames 16000-16202 (`PS2HOST_CAPTURE_FRAME=15999 PS2HOST_CAPTURE_FRAMES=202`) replayed on both backends (`--gs-replay`, `PS2HOST_REPLAY_GPU=1`) and bisected with `PS2HOST_REPLAY_STOP_PRIM` to one primitive. The software GS and Vulkan drew it; Direct3D 11 drew nothing of its batch.
- **Cause:** the layer kept one set of push constants. The renderer pushes the target's size once for a run of draws; that batch's CLUT, loaded on the GPU each primitive, is a compute dispatch with push constants of its own, which overwrote the size. Vulkan keeps push constants by bind point, so the layer now keeps the graphics stages' and compute's apart.
- **Also:** after a secondary command buffer, or the depth upload's own draw, the layer forgets what it had bound (`Exec::ForgetBound`); the push-constant ring's blocks are rewritten after it wraps.

**Feature level 11_0** (Haswell's, Kepler's: `PS2HOST_D3D11_LEVEL=11_0` caps the device at it): on the RTX 4070, car_select, race_training, race and crash are as recorded. On WARP (Microsoft's software adapter, taken when named: `PS2HOST_GPU=basic`) at 11_0 the renderer tests pass; car_select's pictures differ from the NVIDIA ones by small amounts (no pixel more than 32 levels off, a few thousand pixels a picture: WARP's own filtering), nothing missing.

**Intel UHD, Vulkan against Direct3D 11** (its pictures differ from the NVIDIA recordings on both APIs alike, so compared with each other): car_select 42/44 the same, race 52/67. A frame that differs has 14-106 scattered pixels (at most 0.04%), on cars' coplanar parts and edges: the two Intel drivers' interpolation of depth and edges, not the layer (NVIDIA's pictures are the same on both).

**Not done:**
- **D3D12 (X6)**, decided against for now:
  - **No new GPUs.** Every D3D12 GPU here also has Vulkan 1.2 or Direct3D 11.
  - **Costly.** D3D12 has no dynamic depth test, write, compare or stencil ops. The renderer sets those per batch, so each pipeline would need baked variants for every combination drawn, made at draw time: hitches that Vulkan and D3D11 don't have.
  - **The one case for it** is the Intel D3D11 driver's CPU cost per draw (above), on weak CPUs whose GPU has no Vulkan (Haswell has D3D12 at 11_1).
  - **If wanted:** a `d3d12_vulkan.cpp` beside the D3D11 layer, plus a PSO cache keyed by the dynamic states;
- the NVIDIA low-latency boost.

**Speed** (the layer records closures and runs them at submit):
- **What makes it fast:**
  - host vertex ranges are uploaded once per submission, into DYNAMIC twins written with NO_OVERWRITE;
  - push constants go to a ring of 256-byte blocks bound by offset (Direct3D 11.1);
  - `vkCmdUpdateBuffer` and host-to-device copies go through an upload ring and a GPU copy;
  - fences are Direct3D 11.4 fences, waited for asleep, so the rings are safe;
  - a stage's resources are bound again only when its sets or pipeline change.
- **gs.frag specialisation:** each pipeline's feature set (Vulkan's specialisation constant) is compiled from the embedded HLSL with D3DCompile and cached on disk (`d3d11_shaders/` beside the pipeline cache, else `%TEMP%/ps2gpu_d3d11`). fxc unrolls the whole shader into 490 KB of DXBC, which halved an integrated GPU's speed; a variant is 10-170 KB and compiles in 0.1-6 s. `PS2HOST_D3D11_WHOLE_SHADER` uses the whole shader, to compare. Keeping its tap and level loops as loops (`[[dont_unroll]]`, which SPIRV-Cross passes on as `[loop]`) left fxc's output byte for byte the same, so the size isn't the loops.
- **Measured** on a GS capture's replay (3 pictures, 150 passes):

  | GPU | Direct3D 11 | Vulkan |
  |---|---|---|
  | Intel UHD | 10.2 ms a pass (26.2 before this work) | 10.2 |
  | RTX 4070 | 6.7 | 6.1 |

- **What is left:** the Intel D3D11 driver's own CPU per draw. The renderer's thread uses ~8.5 ms a pass against Vulkan's ~3.1.
- **GPU time:** timestamp queries, read with a disjoint query around each submission, give the overlay's GPU % and `PS2HOST_GPU_PROFILE` on Direct3D 11 too (car_select: GPU busy 8.3 s against Vulkan's 9.8; a profile window 260 ms against 264).
- **In play:** a real-time race on the Intel UHD holds 59.9 pictures a second on both. The display paces its presents like Vulkan's (`PresentImageAt` on a thread of its own); jitter 0.27-0.41 ms.

## Where we start (checked in code)

**What we ask for today:**
- `gpu.cpp` skips devices below 1.3.
- `vulkan_display.cpp` did not, until engine 5414705: it now skips them, names them in the log and falls back to GDI.
- Both create the device with `VkPhysicalDeviceVulkan13Features`: `dynamicRendering` and `synchronization2`.

**What we use from Vulkan 1.3:**

| Feature | Where | 1.2 equivalent |
|---|---|---|
| Dynamic rendering | `gs_renderer.cpp`: 4 `vkCmdBeginRendering` sites (the deferred pass, `DepthIn`, the DATE pass, the draw pass) and 3 pipeline kinds created with `VkPipelineRenderingCreateInfo` (`gs_renderer.cpp:1633`, `:2369`, `:2433`). Deferred passes are recorded into secondary command buffers. | The `VK_KHR_dynamic_rendering` extension, or classic `VkRenderPass` + `VkFramebuffer` |
| Extended dynamic state | Depth test/write/compare, stencil test/op/masks/reference, vertex stride (`gs_renderer.cpp:2360-2364`) | The `VK_EXT_extended_dynamic_state` extension (same functions, `EXT` names) |
| `synchronization2` | Enabled, but never called (no `*2` barrier or submit) | Drop it |
| SPIR-V 1.6 | Every shader is built with `--target-env vulkan1.3` (`gpu/CMakeLists.txt:70-127`) | `vulkan1.2` (SPIR-V 1.5) |

**Already optional:**
- dual-source blending
- logic op (FBMSK: there is a fallback path when it's missing)
- BC textures
- anisotropy
- `load_store_op_none`
- NVIDIA low latency

**Also noted under validation** (2026-10-30): the shaders use `DemoteToHelperInvocation` without the device enabling `shaderDemoteToHelperInvocation`. The SPIR-V 1.5 build of phase V2 removes it.

**The API-neutral seams:**
- `ps2gpu::GsRenderer` (`gs_renderer.h`) is what the host draws through.
- `ps2gpu::Display` (`display.h`) is the window, presentation, pacing and input.
- `ps2gpu::Gpu` is the shared device.
- `threaded_renderer.cpp` makes no Vulkan calls.
- The Vulkan parts are `gs_renderer.cpp` (4,000 lines), `vulkan_display.cpp` (2,600 lines: SDL window, input and swapchain together), `effects.cpp` and `gpu.cpp`.
- 17 GLSL shaders, about 2,100 lines.

## Which GPUs each step reaches (Windows)

| GPU | Best driver | Today | After Vulkan 1.2 | After D3D11 |
|---|---|---|---|---|
| NVIDIA Maxwell and later (GTX 900+), AMD Polaris and later (RX 400+), Intel HD 500+ (Skylake+) | Vulkan 1.3 | GPU | GPU | GPU |
| NVIDIA Kepler (GTX 600/700) | Vulkan 1.2 (R470), D3D11/12 | CPU | **GPU** (needs V3 if the driver lacks dynamic rendering) | GPU |
| AMD GCN 1-3 (HD 7000, R7/R9 200-300, Fury) | Vulkan 1.2 (21.5/22.6.1), D3D11/12 | CPU | **GPU** (as Kepler) | GPU |
| Intel HD 4000/4600, Iris (Ivy Bridge, Haswell, Broadwell) | **no Vulkan on Windows**, D3D11 | CPU | CPU | **GPU** |
| Older DX10-class GPUs | D3D10/11 at feature level 10 | CPU | CPU | out of scope (no compute shaders at level 10) |

Which driver versions carry `VK_KHR_dynamic_rendering` on Kepler and GCN 1-3 is **not verified**. V3 makes the answer not matter. A player's `burnout3.log` from 5414705 on names the GPU and its Vulkan version.

## Part 1: Vulkan 1.2

### V1. A 1.2 device (about 2 h)
- **One device-creation routine** for `gpu.cpp` and `vulkan_display.cpp`, which today duplicate it.
- **Which devices are accepted:** `apiVersion >= 1.2` with `VK_EXT_extended_dynamic_state`.
- **What is enabled:**
  - on 1.3: what we enable today, minus `synchronization2`;
  - on 1.2: `VK_KHR_dynamic_rendering` (when offered), `VK_EXT_extended_dynamic_state`, `VK_KHR_create_renderpass2` if V3 uses it, each with its feature struct.
- **On a 1.2 device, point the core function pointers volk declares at the extension versions:**
  - `vkCmdBeginRendering` and `vkCmdEndRendering` at their `KHR` versions;
  - `vkCmdSetStencilOp` and the other dynamic-state calls at their `EXT` versions.

  The structures are the same (the `KHR` ones are aliases), so the renderer's code stays as it is.
- **Logs:** the log says which path is in use: `Vulkan 1.3`, `Vulkan 1.2 + dynamic rendering` or `Vulkan 1.1+, render passes`, with `, states in pipelines` where they are baked.
- **Test switch:** `PS2HOST_VK_LEGACY=1|2` forces the 1.2 path (1 = with the extension, 2 = render passes) on a 1.3 GPU. The real driver then runs exactly what a 1.2 one would be asked for.

### V2. Shaders for SPIR-V 1.5 (about 1 h)
- Every shader built with `--target-env vulkan1.2`. One set for all devices: two sets would double the pipeline cache and the tests.
- `discard` becomes `OpKill` rather than 1.3's terminate or demote. The pixels should be the same, but the regressions decide.
- This also clears the `DemoteToHelperInvocation` validation error.

### V3. Render passes where dynamic rendering is missing (about 3-4 h)
- **`BeginPass` and `EndPass` helpers** replace the four `vkCmdBeginRendering` sites. They begin dynamic rendering, or a render pass with its framebuffer.
- **Render pass cache:** keyed by attachments (colour+depth, depth only), load ops (`LOAD` / `CLEAR` / `NONE`) and store ops (`STORE` / `NONE`).
  - Load and store ops don't affect compatibility. So one *compatible* pass per attachment set serves the pipelines and the deferred secondaries' inheritance.
  - The initial and final layouts are the attachment's own: the renderer already moves images between layouts itself.
- **Framebuffer cache:** keyed by image views and extent. A framebuffer is dropped when its image is (where `Image` is destroyed or resized).
- **Pipelines:** with `renderPass` set instead of `VkPipelineRenderingCreateInfo`. The pipeline cache key gets the path, so a cache made on one path isn't used on the other.
- **Deferred passes:** secondaries take `VkCommandBufferInheritanceInfo::renderPass` (with `subpass 0`), not `VkCommandBufferInheritanceRenderingInfo`.

### V4. Verification (about 1-2 h)
1. **GPU regressions in all three modes** (normal, `PS2HOST_VK_LEGACY=1` and `=2`): car_select, race_training, race, crash, road_rage. Every mode must match the recordings picture for picture. Nothing here changes what is drawn.
2. **The 1.3-only check:**
   - a run under the Khronos profiles layer, which makes this GPU look like a 1.2 one (`api-version 1.2.175`, `VK_KHR_dynamic_rendering` and `VK_KHR_synchronization2` hidden: `scratchpad/legacy` recipe, 2026-10-30);
   - the validation layer stacked above the profiles layer, so any leftover 1.3 call or feature is reported.
3. A hand run with AA off and on, a texture pack, the HUD overlay and splitscreen, in mode 2.
4. **Android**: the same build on the phone, since Android shares this code.
5. **A test build for the #24 reporter**, and any Kepler or GCN owner.

**Not in Part 1:** devices without `VK_EXT_extended_dynamic_state`, which means baking those states into pipelines (about 4× the pipeline variants). No Windows GPU in the table needs it. Older Android phones would, so it's a later Android item (V5) if wanted. (Done 2026-10-31: see Status.)

## Part 2: DirectX

### Which DirectX first: Direct3D 11
- **It reaches more GPUs.** Feature level 11_0 covers every GPU in the table, including Haswell's HD 4600, which has no Vulkan.
- **The drivers are mature everywhere.** Haswell's D3D12 support is weak.
- **Its two gaps are covered:**
  - logic ops need integer render targets in D3D11.1, so we use the engine's existing path for GPUs without logic ops;
  - D3D11 doesn't let us record passes in advance (secondary command buffers), but those passes are a speed-up for phone GPUs and not needed here.
- **It's the simplest of the three APIs to run right.**

**D3D12 brings no new GPUs on Windows:** every D3D12 GPU also has Vulkan 1.2+ or D3D11. It is a later, optional phase (X6), only if D3D11's per-draw CPU cost measures too high on weak CPUs. Its design is close to Vulkan's (command lists, pipeline state objects, explicit barriers), so it would follow `gs_renderer.cpp` almost one to one.

**Considered and not taken: SDL3's GPU API** (Vulkan, D3D12, Metal behind one interface). It has no D3D11, no dual-source blending and no logic ops. Both blending features carry exactness for us, and D3D11 is the backend #24 needs.

**Design reference:** ico-pc (MIT) has an RHI over Vulkan, D3D12, D3D11 and GL. We take ideas from it, not files (see RENDERER_ROADMAP.md).

### X0. The backend seam (about half a day)
- **Split `vulkan_display.cpp` in two:**
  - `window_display.cpp`, shared by every backend: the SDL window, input, controllers, mouse, rumble, triggers, pacing and the refresh timer (about 170 SDL, input and pad references today);
  - the Vulkan swapchain, `PresentImage`, overlay and AA posts.
- **The image handle `Display::PresentImage` takes** becomes the backend's own (`std::uint64_t` already: a `VkImage` or an `ID3D11Texture2D*`).
- **Backend choice:** `renderer=auto|vulkan|d3d11` in `pc_settings.ini`, plus a PC SETTINGS > VIDEO row. `auto` takes the first that works: Vulkan 1.3, Vulkan 1.2, D3D11, GDI. The log names the backend and why the ones before it were passed over.
- **GDI's software path through D3D11:** a CPU picture uploaded and shown through the D3D11 swapchain, so the CPU fallback gets proper vsync and scaling too.

### X1. Shaders for HLSL (about 1 day)
- **The build chain:** glslang (SPIR-V, already built) → SPIRV-Cross (vendored like glslang, MIT/Apache-2.0) → HLSL → FXC (`d3dcompiler_47.dll`, part of Windows) at shader model 5.0 → bytecode built into the exe like the SPIR-V today.
- **Mappings to settle:**
  - 12 push-constant blocks become constant buffers;
  - specialisation constants (`gs.frag`, `gs_address.glsl`, `swizzle.comp`) become `#define` variants compiled per value;
  - `imageLoad`/`imageStore` become `RWTexture2D` (a typed UAV load needs `R32_UINT` on feature level 11_0: check each format);
  - atomics (`swizzle.comp`) become `Interlocked*`;
  - the dual-source output becomes `SV_Target1`;
  - `texelFetch` becomes `Load`.
- **Exit:** all 17 shaders, in all their variants, compile at build time.

### X2. D3D11 presentation (about 1 day)
- **Swapchain:** a flip-model swapchain with its waitable object, for pacing like the Vulkan present thread. VRR through `ALLOW_TEARING`, fullscreen, vsync.
- **The window's extras:** FXAA, SMAA and SSAA as compute passes (the same shaders), the overlay, captures (F12).
- **Exit:** the menus shown through D3D11 with the software GS.

### X3. D3D11 GS renderer (about 3-5 days)
A `D3D11GsRenderer` behind `GsRenderer`, ported part by part from `gs_renderer.cpp`.

**The ported parts:**
1. **GS local memory:** a 4 MB raw buffer read and written by the compute shaders `swizzle`, `decode`, `shuffle` and `patch`.
2. **Targets and depth:**
   - colour is RGBA8;
   - depth is `D32_FLOAT_S8X24_UINT`, read through `R32_FLOAT_X8X24_TYPELESS` views;
   - feedback copies use `CopySubresourceRegion`.
3. **State:** blend, depth-stencil and raster state objects cached by the same keys as today's pipelines. D3D11 builds shaders and state separately, so there is no pipeline building and no first-start wait (#29 can't happen on this path).
4. **Draws:** dynamic vertex buffers, `DrawTriangleRun`'s batching kept as it is.
5. **Read-backs** (`MemoryRead`, `TargetToMemory`) through staging buffers, mapped once a GPU query says the copy is done.
6. **Texture packs:** BC formats are native, anisotropy too.
7. **`effects.cpp`'s passes** (AO, bloom, blur, grade) as compute.

The steps run in that order, each checked with `--gs-replay` on the recorded captures before the next.

### X4. D3D11 verification (about 1 day)
- **The same GPU regressions with `renderer=d3d11`.** Byte-for-byte equality with Vulkan is the aim. Where the rasterisers or the samplers' rounding legitimately differ, it is recorded as its own `.d3d11.crc` set, after looking at each difference (`framediff.py`).
- **Speed:** the race benchmark on the Intel UHD and the RTX 4070 against Vulkan, and real-time 60/120 with frame times.
- **A test build** for the #24 reporter and any owner of an old Intel chip.

### X5. Release
README and release notes: the backends, `renderer=` and what `auto` picks.

### X6 (optional). D3D12
Only if X4 shows D3D11's CPU cost hurting weak CPUs. It would be a port of the Vulkan renderer: root constants for push constants, PSOs with a disk cache, explicit barriers. About 3-5 days.

## Order and size

| Step | What | Size | Releasable alone |
|---|---|---|---|
| done | 5414705: old Vulkan GPUs skipped and named, GDI fallback | - | yes |
| V1-V4 | Vulkan 1.2 | ~1 day | yes (0.7.4) |
| X0-X1 | Backend seam, shaders to HLSL | ~1.5 days | no (no player change; X0 lands with X2) |
| X2-X4 | D3D11 presentation, renderer, verification | ~5-7 days | yes |
| X6 | D3D12 | ~3-5 days | optional |

**Risks:**
- **Exactness on D3D11:** the rasterisers' top-left rule is the same, but sampler and blend rounding may differ. X4 decides how the regressions handle it.
- **Typed UAV loads on feature level 11_0:** X1 may need `R32_UINT` views with packing done in the shader.
- **Kepler and GCN 1-3 drivers are old and could have bugs of their own.** That's the reason for test builds before releasing.
- **An abstraction tax on the Vulkan path:** X0 must keep the present thread and pacing exactly as they are. The Vulkan regressions plus the frame-time benchmark guard it.
