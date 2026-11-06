# macOS (Apple silicon)

A port builds and runs natively on an Apple silicon Mac: Apple's clang,
CMake and Ninja, the Vulkan display through MoltenVK, NEON through sse2neon.
`tools/macos.py` is the one script for it, as `tools/port.py` is for Windows.
Burnout 3: Takedown (NTSC-U) is the game it was brought up on: from a disc
image to a game that runs at about full speed, on an M1 Max.

Nothing of a game is in this repository or in what you build: everything
comes from a disc image you own, on your machine.

## What you need

* An Apple silicon Mac, macOS 14.4 or later (the fibers wait on addresses
  with `os_sync_wait_on_address`, 14.4's)
* Xcode's command line tools (`xcode-select --install`), Python 3.10+
* `brew install cmake ninja molten-vk vulkan-loader`
* Pillow (`pip3 install pillow`) for the icon of `app`
* A disc image of the game (an .iso)

The first build downloads SDL3 and the other libraries the host is built with
(CMake's FetchContent) and sse2neon (MIT, pinned by hash), so it needs a
network once.

## From a disc image to a running game

The first three steps are the same as on Windows (the game's README has its
own commands: `games/<name>/README.md`). Run from the repository root.

```sh
# 1. the game's program, from the disc (the game's README names the files)
python3 engine/tools/isotool.py extract "<disc>.iso" <executable> -o games/<name>/extracted/

# 2. the analysis, then the decompilation checked against the machine code
python3 engine/tools/analyze.py
python3 engine/tools/derive.py --emit --csv
python3 engine/tools/lift.py --game games/<name> --structure --emit --csv

# 3. build, run, and build again until a run finds no more code copied at run time
python3 engine/tools/macos.py --game games/<name> all -- --window --gpu --audio \
    --memcard games/<name>/build/memcard --timeout 86400
```

Steps 1 and 2 are done once per game and disc. `derive` and `lift` compile
and run their checks with clang: for Burnout 3 on an M1 Max `derive`'s
compiled check took about 22 minutes and `lift` about 90. Step 3's first
build took about 5 minutes.

`macos.py` actions:

| | |
|---|---|
| `build` | generate the dispatch table and build: `<game>/build/macos/<name>/<name>` |
| `run -- <options>` | run the built program (what follows `--` is the port's own options); sets what Vulkan needs: Homebrew's loader on the library path and MoltenVK's driver |
| `all -- <options>` | build, run and build again until a run finds no more code copied at run time (`port.py`'s default action) |
| `pgo` | the profile-guided build: a counting build, the game's scripted runs, `llvm-profdata`'s merge into `<game>/build/port/pgo.profdata`, and the build with it (below) |
| `app [--icon FILE_OR_URL]` | `<game>/build/macos/<Title>.app` (below) |

A build is ThinLTO when the game's `game.json` says `"lto": true` (`--lto` / `--no-lto` /
`PS2_LTO=off` override), and uses the profile `pgo` made when there is one (`--no-pgo`: without).

## Speed: the VU microprograms

As on every platform, record the microprograms the game runs and build again,
or the vector units run in the interpreter (about 1.5x slower, and a game
that leans on VU1 - a car panned around in a menu - drops frames):

```sh
for run in <the game's port/runs that reach new microcode>; do
  <the built program> <executable> --disc "<disc>.iso" $(cat games/<name>/port/runs/$run.args) \
      --vu-interpret --vu-record games/<name>/build/vu_programs.txt
done
python3 engine/tools/macos.py --game games/<name> build
```

The report's `VU: ... (N compiled)` line says they are compiled. The manifest
is the game's microcode: it stays in `build/`, never committed.

## Performance

Where the time goes (Burnout 3, a race): VU1 about 50-60% of the CPU, VU0 7-9%
(the recorded microprograms, compiled ahead of time with the VU's own cycle
accounting). What the build can do about it:

* **ThinLTO** (`game.json` `"lto"`): Apple's `ld` reads it. The Mac's build was
  not getting it before: the CMake branch for clang tested for `Clang`, and
  Apple's compiler calls itself `AppleClang`.
* **No stack canary or probe, hidden symbols** (`host/CMakeLists.txt`, the Apple
  branch): Apple's clang turns on `-fstack-protector` and `-fstack-check` by
  default, in every translated function's frame; Linux and Android already
  built without them.
* **A profile-guided build** (`macos.py pgo`, about an hour: the counting build's
  link is slow, then the six scripted runs): the binary is also smaller (50 MB
  against 119 MB with LTO alone).

Measure with `tools/bench.py` (it finds the Mac's program) - and expect noise:
the same binary took 467 s and 494 s on a busy Mac. Interleave the builds you
compare, or time CPU rather than wall, and quit what else is running. In one
interleaved pair of the `race` run, ThinLTO took about 9% less wall time and 3%
less CPU than a build with neither; play-testing the profile-guided build felt
smoother still.

## An app: `macos.py app`

`<game>/build/macos/<Title>.app`, double-clickable, 80 MB or so: the program,
the Vulkan loader and MoltenVK, a launcher that asks for the disc image the
first time (and remembers it) and keeps saves and settings in
`~/Library/Application Support/<Title>`. Nothing of the game is inside it: it
plays from the player's own disc image. It is signed ad hoc, so it runs on the
Mac that made it; on another Mac, Gatekeeper wants a right-click and Open (a
release for others needs a Developer ID and notarisation, which this does not
do).

The icon is the disc's own title logo (a game's `port/tools/make_icon.py`
reads it from the disc), tilted on a macOS rounded square. `--icon` takes a
picture of your own instead - a file, or an https URL - fetched to your
machine once and kept in `<game>/build/port/` with your other build files,
never committed: a game's artwork is its publisher's. Delete
`build/port/icon_mac.png` to go back to the disc's logo.

## What is different from the other platforms

* `host/src/fiber.cpp`: fibers are Mach-O assembly (leading underscores, no
  `.type`/`.size`); waits and wakes on an address are `os_sync_*`.
* `gpu/src/gpu.cpp`, `vulkan_display.cpp`: MoltenVK is a portability
  implementation, so the instance enables `VK_KHR_portability_enumeration`
  with `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR`. SDL has to make the
  window's surface with the same loader the instance came from
  (`SDL_VULKAN_LIBRARY`; the app's launcher sets it).
* `gpu/src/gs_renderer.cpp`: `MemoryPatch` (a compute pass with push constants of
  its own) now resets the draw-state cache as the other compute passes do:
  MoltenVK keeps one block of push constants, so a draw after the patch could
  keep a stale target size and land in the wrong place (the renderer test
  `ComputeBetweenDraws` failed on MoltenVK because of it; it passes now).
* `ps2decomp/cppcheck.py`, `liftcheck.py`, `equivalence.py`: the verification
  builds its check libraries with clang++ outside Windows. `derive` first runs
  clang's `-fsyntax-only` over the emitted source (seconds, where the
  optimising compile is minutes), so a function that does not compile is
  dropped without a round of the full build.
* Undefined behaviour: clang compiles a constant shift of a 32-bit value by
  32 or more to a trap (`brk #1`) where MSVC tolerates it. The PS2's own
  shifts are 64-bit (`dsll`, `dsra32`), so the emitter widens a compound
  operand to `u64` (`symexec.py`), and the syntax pre-pass makes
  `-Wshift-count-overflow` an error.

## Known issues

* The software renderer's pictures match the recorded ones (`car_select`, 44 of
  44) but the GPU renderer's differ in some: the car paint's reflection
  highlights in the garage are weaker on the GPU (a few percent of a frame's
  pixels, at most 112 levels out of 255) than the software GS draws them. It is
  not the logic-op or the dynamic-state paths (`PS2HOST_GPU_NO_LOGIC_OP`,
  `PS2HOST_VK_BAKED` change nothing); not looked into further, and not known
  whether the Windows GPU path shows it too.
* In a loop with several exits `derive` can emit the last exit's case with a
  loop variable the emitter never declared; clang rejects it and `derive`
  drops the function from its output (a few dozen in Burnout 3). The bug is
  the emitter's, not macOS's; MSVC may be accepting what clang rejects.
* A debugging run that stops with a crash report: `~/Library/Logs/DiagnosticReports`.
