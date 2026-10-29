# VU1 compiled code: the next steps

What the compiled VU programs (tools/vurecomp.py, host/include/ps2host/vu_core.h)
still spend, and the plan to take it down, every step exact.

## Why

A high frame rate draws every picture with VU1: at 144 a second Burnout 3's
crash-breaker explosion takes VU1 8-16 ms for a picture's worth of work
against the 6.94 ms a picture has, and VIF1's worker (VU1 is ~75-80% of it)
is the bound there, with the GS thread close behind (games/burnout3/docs/
PERFORMANCE_PLAN.md, "High frame rates with the modern graphics and a texture
pack").

## Where a pair's time goes

Sampled (`--vu-fast --sample --vu-bench`, the explosion's 20,000 runs,
`vu_crash.bin` = `--vu-snapshot 20060 20000` of the crash run): ~1.73 ns a
pair. Of the compiled code's samples, by VuCore step:

| | share |
| --- | --- |
| operand clamps (ClampLanes) | ~9% (before step 8) |
| stall checks (Wait and its max) | ~11% |
| MAC flags worked out (MacFlagsOf) | ~8% |
| flags landed and queued (LandFlags, QueueFlags, FinishUpper's queue) | ~8% |
| a pair's end: landing times, pending writes (Finish, FinishUpper) | ~8% |
| the block's own lines (count, cycle, exits) | ~7% |
| the arithmetic itself (Fmac, LanesSse, Result) | ~10% |

In a hot block of a program that reads MAC (program 88's 0x108, a 10-pair
loop body), MacFlagsOf 18%, LandFlags 17%, stall checks 17%, clamps 10%,
arithmetic 13%: the flags handed from block to block through the queue, and
the stall checks at a block's start, cost more than the work.

## The steps

Each is checked by VuCompiledIsTheInterpreter (every register, flag,
pipeline and cycle; the corpus also tried at 600 programs, seed 7), the VU
bench's two hashes (exact and fast), and the GPU regressions. Speed: two
`--no-pgo` builds side by side on `--vu-bench vu_crash.bin` (an edit to
vu_core.h changes every block, which would lose its PGO profile and skew a
comparison with a profiled build).

1. **Landing times only where they can be read** (done). A write's time
   (vf_ready_, four u64 stores) is read only by a stall check within the
   three pairs after it, or after the block leaves within two pairs of it.
   The rest are left out (`untimed_writes`; `PS2_VU_ALL_TIMES=1` keeps them
   all). A held lower write other than a load (MOVE, MR32, MFIR, COP2) lands
   only where the value changes, so it does not count as overwriting an
   earlier time. Vu::SameState compares times as far as they can still
   stall (max with the cycle). Bench: fast 60.8 -> 57.6 ms (-5%), exact
   82.6 -> 80.8 ms (-2%), both hashes the same.
2. **Flags across a known edge.** A block that goes on into a known block
   (PS2HOST_VU_NEXT) queues its in-flight flags and the next lands them in a
   loop. A version of the next block for entry from that edge can take them
   as they are - their cycles known relative to the exit - and land them
   straight-line where it reads MAC.
3. **Stall checks across a known edge** (done, before 2: the larger share).
   An exit that goes on into a known block, three pairs or more into its
   own, goes into an *entry version* of it (vurecomp's Chain, up to
   MAX_VERSIONS a block; `PS2_VU_NO_VERSIONS=1` for none): one compiled
   knowing which fields were written in the last three pairs before it and
   how many pairs before the exit (`recent`). A field written d pairs before
   lands by the exit pair's issue + 4 - d; the version's pair i issues at
   that issue + 1 + i at the earliest - no stall where d + i >= 3; anything
   else landed before it. In the explosion's hottest block all fifteen
   checks were in its first three pairs. Burnout 3's generated files 58 ->
   131 (the build ~2 -> ~4 minutes). Bench: fast 58.7 -> 54.2 ms (-8%),
   exact 81.1 -> 74.2 ms (-8.5%), hashes the same; with step 1, the
   explosion's VU1 time -11%.
   Versions are told only the fields the block's first three pairs read
   (Chain.used): the same code, 115 generated files instead of 131.
4. **Clean operands across an edge** (tried, dropped). Versions that also
   knew the fields an FMAC left clean before the exit, their clamps left
   out: more versions (221-256 files), and slower - fast bench 53.8 ->
   56.6-58.7 ms at 6 or 16 versions a block. The code's size costs more
   than the clamps.
6. **MAC flags only where a reader can see them, timed** (done). Fast
   mode's walk (mac_live) kept an FMAC's flags for any MAC reader met
   before a later FMAC's had landed - counting from the newest later FMAC.
   Now it counts from the first (once its flags land MAC is never the
   older one's: every FMAC that sets MAC replaces it whole), and keeps a
   bound on each pair's issue along the path (a reader that issues before
   the flags land cannot see them). Burnout 3's flag-computing FMACs 6,073
   -> 2,243; fast bench 52.7 -> 46.0 ms (-13%), its flag-free hash the same
   (the full hash differs: what fast mode gives up, the flags a run ends
   with). `PS2_VU_OLD_MAC_LIVE=1` for the walk as before. With steps 1 and
   3: 60.8 -> 46.0 ms, -24%.
7. **ITOF four lanes at once** (done). It was a double division a lane and
   Result's checks - a third of a hot block (program 20's 0x560). cvtdq2ps
   rounds as the run's control word says (toward zero) as the narrowing
   did, the scale a power of two (exact, commuting with the rounding), and
   an integer is never a NaN, an infinity or a denormal. Fast bench 46.6 ->
   43.5 ms (-6.6%), exact 74.5 -> 70.6, both hashes the same. ABS likewise
   (CleanLanes, rare here). With the steps before: 60.8 -> 43.5 ms, -28%.
   Tried, no gain: entry versions carried through blocks of fewer than three
   pairs (the generic blocks left are entered from the driver - a run's
   start, MSCNT - 7% of the samples against 28% in versions).
8. **Clean on entry, from every way in** (done). Where the clamps left
   come from (each pair weighted by how often the explosion ran it, the
   interpreter counting): 79% are of fields from before the block, 14%
   loaded in it (LQ), 4% Q. vurecomp's entry_clean works out what a block
   can start knowing clean from all its known ways in (going on, a branch,
   a conditional branch's side exit) - met over them, no versions, the same
   code. A run's start, the pair after an E bit and a return's place start
   knowing nothing. A jump to a register's address says so (s.jumped); the
   driver hands the run to the interpreter where it lands on a block that
   knows something (VuCompiled::knows; under 1,024 of the bench's 20,000
   runs did). The first try had
   the guard in every block (5,536 switches, +9% code) and was 1% slower.
   Proven clean: 22% of the clamps run; the ceiling even with every loaded
   value clean would be 47% - the rest are registers a run starts with,
   left by the runs before. Fast bench 43.5 -> 42.4 ms (-2.6%), exact
   71.4 -> 70.8, both hashes the same; the corpus at 600 programs, seeds 7
   and 3. `PS2_VU_NO_ENTRY_CLEAN=1` for blocks knowing nothing on entry.
5. **Superblocks for loops.** A loop body chained into itself (most of the
   hot code) compiled as one function with the loop kept, so the steps above
   hold from one iteration to the next.

Steps 2-4 need an entry version per known edge: code grows; only edges a
recorded run took (PS2HOST_VU_BY_BLOCK counts) are worth one.

## 2026-10-29, night: CLIP, and where a pair's time is now

Measured with the same benches (`--vu-fast --vu-bench`, the crash, the
pile-up and the race snapshots; two `--no-pgo` builds side by side, twice).

9. **CLIP without branches** (done, engine da8aa95 and a2929f7). ClipJudge
   was a scalar loop with a data-dependent branch a lane (a vertex's
   judgement), ~8% of the compiled code's samples in the pile-up. Now the
   three lanes' magnitudes are compared at once, and each lane's two bits
   are put in place in the vector and ORed across (no table lookups after
   a movemask); ft is read whole, not its w alone just after a pair stored
   the register. Fast bench: crash 42.5 -> 40.3 ms, pile-up 42.2 -> 39.2,
   race 37.4 -> 34.9 (-5 to -7%); exact pile-up 71.4 -> 66.8; every hash
   the same.

Tried, dropped (all exact, hashes the same):
- *Stalls settled by the generator* within a block - a check on a field
  the block itself wrote after its last pair with a run-time stall, the
  stall's cycles known: only 851 of ~17,000 checks could be (most read
  fields written before the block); pile-up -3%, race +1-4%: no clear gain.
- *The run's count, cycle, pending branch and E bit as the blocks'
  arguments* (registers from block to block, stored only where one returns
  to the driver), against the ~5% of samples in the blocks' entry and exit
  lines: pile-up and race +2.5% (register pressure in the blocks).
- *DIV's and SQRT's results not cleaned* (they are clean under a run's
  control word): no measurable change.

**Where a pair's time goes now.** A 48-pair fast block of the pile-up's
hottest program (Program70 0CA8), compiled: 807 instructions, of which the
arithmetic is 46 (25 vmulps, 21 vaddps); the operand clamps ~138 (vpand,
vpminud, vpor: three times the arithmetic); vector loads and stores of the
VF registers 132; ~300 scalar (VI registers, addresses, the flags and the
timing). At ~1.15 ns a pair the code runs near four instructions a cycle:
it is bound by how many it has, not by one part. The registers' traffic,
counted: of the block's 76 VF loads, 58 are a register's first read in the
block and 18 read one again (7 after a store of it) - so VF held in host
registers across a block's pairs (loaded at the first read, stored at the
exits; a rewrite of the emitter and VuCore's operations) would save the 18
and some overwritten stores, ~4% of the instructions: not worth it. What
is left is across blocks (the first reads and last writes, superblocks)
and the clamps of operands from before the block (step 8's ceiling).
