---
name: ps2-high-fps
description: Add high frame rates (120, the screen's own rate) to a ported PS2 game with the engine's ps2host/high_fps.h - the game kept at its own 60 steps a second, the pictures between its steps drawn with its transforms and camera placed between them (or, for a game at 30 whose drawing is inside its updates, its display lists blended). How to find a game's step loop, its once-a-frame logic, its transforms and its camera's inputs, the hooks to call, the traps, and the checks that prove nothing runs faster and nothing jumps. Use when adding or debugging high fps in any game under games/<name>/port.
---

# High frame rates for a PS2 port

The engine (`host/include/ps2host/high_fps.h`, `host/src/high_fps.cpp`) does
the general part: the rate (the FRAME RATE setting: 60, 120, DISPLAY), the
blanks (SetFastForward / SetBlankRate), how many steps each picture asks for,
where the picture is between the game's last two steps, taking down and
placing RwMatrix transforms and camera values, putting them back, cuts,
stepping down where the PC cannot keep up, a watchdog back to 60, and a
trace. A game supplies **where** (a `HighFpsGame`) and calls five hooks from
its overrides. Burnout 3 is the worked example:
`games/burnout3/port/high_fps.cpp` (about 100 lines) and
`games/burnout3/docs/HIGH_FPS_PLAN.md` (every measurement behind it).

## The rule that matters most

**Gate the game's step count, never one function inside a step.** A PS2
game runs fixed steps (its physics, timers, AI, menus: a sixtieth each) from
a loop in its main loop, as many as the blanks since the last picture
(at least one). Held back alone, the world update still left the race
clock, the crash logic and the menus running once a picture: twice as fast
at 120. Find the count and give the loop the engine's count instead
(`HighFpsSteps`): then everything in a step runs exactly as at 60, by
construction, menus included.

And: **never assume speed - measure it** (the checks below). Anything that
runs once a picture rather than once a step is now twice as fast.

## Finding the game's places (in this order)

Work in virtual time (`--virtual-time`, scripted presses) - deterministic,
pictures dumped with `--frames DIR --frames-every 1 --frames-from N`. Dumps
stop at 200 pictures a run.

1. **The main loop and its states.** Find the function the game loops in
   each frame (from the `[call]` log, the disassembly in `asm/`). Log its
   state variable each frame to see which state is which (Burnout: 5 was
   both the race *and* the frontend; 4 and 7 loading). The engine's hooks
   go in the state(s) that run the game's steps.
2. **The step loop and its count.** In that state, a loop calling the
   step's functions a counted number of times. The count comes from a
   function returning "steps since last frame, at least N" (Burnout:
   0x00376570, called at two call sites; the count stored and used by the
   *next* frame's loop). Override it: call the original (its state stays
   right), then, only at those call sites (check `$ra`), replace `$v0` with
   `HighFpsSteps(blanks, most)` when it returns >= 0.
3. **The game's own step counter.** A global incremented once per step
   inside the loop (Burnout: gp-0x5C5C at 0x00132DC8). The engine needs it to
   know how many steps really ran - **don't infer it** from when you granted
   them: the grant and the run were a frame apart in Burnout, and only the
   measured count gave the right phase.
4. **Once-a-frame logic that must go once a step.** Functions called each
   frame *outside* the step loop that move the game on (Burnout: 0x00131F90 -
   the crash camera's moves, Impact Time's clock). Test: skip a candidate on
   frames that ran no step; if the game's timeline changes at 120 vs 60
   (`frame_motion.py match`), it moves the game on and must be gated.
   Override it: `if (!HighFpsFrameSteps(counter, blanks)) return; original;
   HighFpsStepped();` - that is also where the steps are counted and the
   transforms taken down.
5. **The frame's start** - before the steps run (Burnout: 0x002225C0, the
   main loop's preamble, one call site). `HighFpsFrameStart()` there puts
   back what the last picture placed. Make sure it runs **every** frame:
   Burnout's first guess (0x0019A950) ran only while loading.
6. **The picture's hook** - after the steps, before the draw:
   `HighFpsPicture()` (Burnout: the display's frame step, 0x001F43B0,
   mods.cpp's Frame). It may run several times a picture.
7. **The vertical blank** - the game's blank handler, which runs in every
   mode: `HighFpsBlank()` (drops to 60 when the game's frames stop: loading
   screens, movies outside the loop - without it the menus after a race ran
   at 120). `blanks` for the hooks: the blanks the last picture took
   (Burnout: counted in sixty_fps.cpp's BlankHandler; a host-side counter
   works as well).

## What to place between steps

8. **Transforms.** RAM dumps a step apart (60, virtual time: `--dump-ram`
   at frames N, N+1, N+2), then `python engine/tools/find_transforms.py
   a.bin b.bin c.bin`: every RwMatrix that moves smoothly, grouped by stride.
   Give the engine the regions holding the game's objects
   (`HighFpsGame::transforms`). **Leave out the stack (0x01FF0000 on) and any
   region shared with buffers** - Burnout's crash debris sits beside the
   movie decoder's buffers, and blending there broke a movie. Test a new
   region in a crash *and* across a movie.
9. **The camera's inputs.** The view is usually **made in the draw** from
   the camera's own values, so placing the camera's matrices does nothing.
   Find the values the draw reads: search the dumps for floats near the
   camera's position (a small script over three dumps), then **probe**:
   `PS2HOST_HIGH_FPS_POKE=<address>` adds 0.5 to that float before each
   draw - the picture changes only if the draw reads it (Burnout's race:
   0x00517030's block, +0x30 position, +0x20 direction, +0x14 FOV; player
   2's 0xBB0 on). Different screens can differ: Burnout's garage moves the
   camera's RwFrame in the steps instead (its modelling matrix, found the
   same way, added as a transform region). `HighFpsCamera{address, floats,
   position index, direction index}`; the engine blends only plausible
   float values and detects the camera's own cuts (a jump, a reversal, a
   sharp turn against the step before).
10. **Cuts.** `HighFpsGame::cut`: a transform moving further than this in a
    step is shown, not blended (a car's step is well under it; a respawn
    over it).

## Traps (each cost hours once)

- Gating one function in a step instead of the count (above).
- Inferring the picture's time from your own grants - use the game's counter.
- A camera matrix "placed" that the draw rewrites: no effect, no error.
- Blending a value that is not a transform (a buffer, a timer the game reads
  for logic): only memory and values proven by probing.
- A put-back that overwrites what the draw wrote: the engine restores only
  memory still as it placed it - keep it that way.
- The menus may be inside the same loop (Burnout's were): then they get
  high fps too, for free and at the right speed; screens' 2D animation and
  movies stay at their own step rate (not a bug).
- A frontend frame-time (dt) variable is tempting (tell the game 120: native
  120 pictures) - it made Burnout's car select genuinely 120 but its main
  menu twice as fast. Only per screen, with `match` proving each.
- Scripted runs above 60 desync from their presses (the menus take more
  pictures): switch at a frame with `PS2HOST_HIGH_FPS_FROM`.
- Real time frame dumps stall: judge real time with the benchmark only.
- A false cut is a jitter: the camera not blended for one step stands
  still for its pictures, then goes on. Burnout's +0x20 values (its "way of
  looking") are negated for a step now and then at speed, and shrink to a
  few hundredths: both read as sharp turns. The engine now takes a negation
  with the position moving on as no cut, and judges turns only for values
  of a length. Find them with a real-time race at the screen's rate and
  `PS2HOST_HIGH_FPS_TRACE`: `high_fps_trace.py`'s "jump" where the placed
  position equals the step's for 2-3 pictures.
- Anything that waits on the renderer's thread once a draw ruins high
  rates: the texture pack's "is the replacement still on the GPU" drained
  the threaded renderer's queue each draw (half the GS thread, the race
  stepped down to 60). `PS2HOST_GPU_THREAD_LOG=1` lists the waits; a GS
  thread profile (`--sample-gs`, `PS2HOST_SAMPLE_RAW`, `hotspots.py`) shows
  `_Cnd_wait` under the culprit.

## Games at 30 pictures a second (Resident Evil Outbreak)

First measure the game's own rate: dump consecutive pictures in virtual time
and `frame_motion.py steps` - "x 0 x 0" is 30 a second (Outbreak: its frame
end waits for blanks past an interval - 0x001A90B0, 0x0029FA04, reloaded
each blank from 0x0029FA08 by its own blank handler).

1. **Try the interval first** (a picture every blank): if the game's steps
   follow time (dt), that is native 60 - check with `match` against the
   30-picture run at equal frames. Outbreak's steps follow pictures: twice
   as fast. Burnout's car select followed time; its main menu did not.
2. **Then the draw**: an in-between picture needs the game's drawing without
   its step. Look for a separate draw: in Burnout the world update and the
   draw were apart (one could run alone). Test a candidate split by running
   only the later part on in-between pictures: `match` shows whether the
   game keeps its speed (Outbreak: its gameplay frame 0x005FED00's calls
   from the 25th on - speed exact), and the pictures whether they are drawn
   (Outbreak: black - its scene is drawn *inside* the update functions, calls
   3-24, not after them; skipping the whole game task left the last picture
   on screen).
3. Where drawing is inside the updates: **blend the display lists**
   (`ps2host/display_list.h`; Outbreak's port/high_fps.cpp is the example).
   `PS2HOST_TRACE_KICKS=<frame>` shows where the lists are sent (Outbreak:
   0x0018D470, at the next frame's start: 3 VIF1 chains a step, ~1.4M words).
   Keep each step's chains as sent (`SetVif1ChainFilter`, within that
   function only), and send every picture's as `BlendVif1Chains(before, now,
   HighFpsPictureTime())` - a frame that did not step sends the last step's
   so in place of its own. Before relying on it, dump two steps' chains and
   classify the words that differ (UNPACK floats blended, the rest): Outbreak's
   lists match code for code to their end and only ~9,000 floats differ
   (matrices; vertices are model space) - and 228 of them, a projection's
   translations up to 2e7, were left out by a float bound of 1e6: the back
   wall and the mirror jumped back each in-between picture (now 1e9).
   The cut must be large (100000: screen matrices with 2048-scale terms).
   **Never blend the lists in lockstep**: the blender cuts them into the
   things drawn (from each UNPACK to the things' matrix address - not one with
   FLG, the double buffer's vertex batches) and matches each with the older
   step's thing of the same codes and nearest values. Outbreak drew things in
   another order from step to step (two light billboards traded places: a
   beam across the room; 2D sprites with one screen matrix: a textured disc
   over the picture), and a thing whose values jump (a flickering light's w)
   is left as the game steps it. Glitches are found with
   `engine/tools/blend_glitches.py DIR` over a long gameplay run's every
   picture (Outbreak's `roam`): pixels unlike both neighbours where they
   agree - then the list bisected (a switch that blends only a word range,
   eight ranges at once) and the two steps' segments dumped and compared.
   Watch also for lists sent after the frame's function returns (from the DMA
   interrupt: Outbreak's 2D, once in a while): a frame's lists are all those
   sent until the next frame's start, its step and picture time taken at its
   start.
   **Judge the camera by its view's rows**, each by its own length
   (`cut_segment`, `view_first`/`view_end`), not by lanes: a projection's
   translations (2e7) make every turn tiny by the lane, so a real cut was
   blended through the walls, while a translation passing through 0 looked
   like a jump - the camera still for a step, then on (jitter). Leave its
   lights' rows out (a shot's flash), and its place in the list (a HUD
   segment before it comes and goes).
   **Watch a double buffer**: a list made for one half may name its half
   where it sets FRAME itself (Outbreak's light composited after its shadow
   pass). A picture between steps drawn into the other half sends those
   draws to the half not shown - characters and lights flicker every other
   picture. Find the writes (`GsRegisterWrites`, FRAME_1/2) and turn them to
   the frame's half (Outbreak's ToHalf; the half from the game's own flip).
   Test a scene of every kind the game has - the bar's lists had none, the
   tunnels' did.
   **Not every changing float is motion**: a texture offset that wraps (1/24
   a step, 23/24 back to 0 - blended, half a turn away: a flash), a value a
   program reads as an index (a decal's CLUT from its batch's own data:
   squares on the floor). The engine treats a 0-1 value moving more than half
   as a jump and never blends the double buffer's (FLG) data. Hunt the rest
   with long wanders (the game's health cheat on, so the player survives),
   every picture scanned, the worst bisected word by word.
   **Profile the motion, not only the pictures**: a judder is a thing that
   moves on a step's picture and stands on the next. `ps2host::ListMotion`
   (Outbreak's `RE_MOTION_TRACE=<file.csv>`) writes each picture's things'
   moves as sent; `engine/tools/list_motion.py` ranks the things stepped or
   uneven and the steps where all stepped (cuts). Pictures' phase
   correlation is no measure in a dark scene - it locks on the player.
   `PS2HOST_BLEND_JUMPS=1` logs why each thing jumped (its word, the values,
   its lane's scale, its change the step before).
   **A camera that turns** (Outbreak's RE4 style, behind the shoulder) broke
   three rules made with fixed cameras: judge the view's axis rows together
   (`view_axes_end` - a turn hands length from one row to another, a short
   row's step read as a cut every 60 degrees); a screen matrix's
   translations move 1e6 a step - past `cut`, blended only while steady
   (`cut_steady`), or every prop shakes; and a cut by less than a cut's
   length is a sudden move (`view_sudden`). A camera that eases changes its
   move by ~20% a step: the steady test's quarter (`steady`) stepped its
   things as it came to rest - a half. A snap the game knows of (a camera
   pulled in at a wall) is best made a cut by the port, with
   `ListBlendPlan::carry` so its moves are not the next step's history.
   Test with the camera turning on the spot, not only walking, at 60 and
   120 (list_motion.py judges a step's pictures however many).
4. Configure `steps_per_blank = 0.5`, `default_rate = "30"`: the engine's
   rate, steps and picture time then work for 60, 120 and the screen's (the
   F9 panel's FRAME RATE row shows 30/60/120/screen for such a game). Set the
   interval to 0 above 30 and run the step (Outbreak: the task system
   0x001B7CB0) only when `HighFpsSteps` says - never by parity: the game
   sometimes waits two frames itself.
5. Gate it to gameplay (`HighFpsFrameSteps` only in the gameplay mode): menus
   and movies stay at the console's rate - check: the menu regressions with
   `PS2HOST_FRAME_RATE=120` give the recorded CRCs.

Outbreak's measures (the walk): a step's picture at 120 matches the 30 run's
(0.01); every picture moves by an even amount; real time 0.58 -> ~1 core at
60 (the blend of 1.4M words a picture, the VIF1 worker 15 -> 30%).

## Checks (all of them, every change)

- **Regressions at 60 unchanged** (the setting off): `regress_all.py
  --gpu-only`. Everything above 60 is off at 60.
- **Speed**: `frame_motion.py match DIR60 DIR120 FROM` for each kind of
  screen (race, crash, menu, a movie, a map, splitscreen) - offsets steady.
- **Smoothness**: `frame_motion.py steps DIR120` - even, no "x 0" pairs where
  things move.
- **Jumps**: `PS2HOST_HIGH_FPS_TRACE=trace.csv` on a run, then
  `engine/tools/high_fps_trace.py trace.csv` - no blended cuts.
- **Real time**: the benchmark at 60, 120 and 144 (and splitscreen): late
  and dropped pictures, cores.
- A refactor of the mechanism: record 120 pictures before (a few stretches,
  switched with PS2HOST_HIGH_FPS_FROM) and compare their CRCs after -
  identical (Burnout's move into the engine: 324 of 324).

## The hooks, in a frame's order

```cpp
ps2host::HighFpsConfigure(game);          // once
// frame start (before the steps):        ps2host::HighFpsFrameStart();
// the step-count function (its call sites): int n = HighFpsSteps(blanks, most); if (n >= 0) v0 = n;
// after the steps, once a frame:         if (HighFpsFrameSteps(counter, blanks)) { once-a-frame logic; HighFpsStepped(); }
// before the draw, each picture:         ps2host::HighFpsPicture();
// each vertical blank:                   ps2host::HighFpsBlank();
```
