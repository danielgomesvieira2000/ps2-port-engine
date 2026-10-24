#!/usr/bin/env python3
"""Build a game's native port and run it.

    python engine/tools/port.py --game games/<name>            # generate, build, run
    python engine/tools/port.py --game games/<name> build      # generate and build only
    python engine/tools/port.py --game games/<name> run -- --trace
    python engine/tools/port.py --game games/<name> pgo        # a profile-guided build (below)
    python engine/tools/port.py init <name> --iso "<disc image>" --title "..."  # a new game

The port is the game's translated functions (src/lifted, from lift.py --emit) on
the native host (engine/host): guest memory, a dispatch table from guest address
to translated function, the EE kernel's services. This generates the dispatch
table, builds games/<name>/port/ with CMake and Ninja - with LLVM's clang-cl
when it is installed (its code is faster: the VU programs about 17%, the GS
about 20%; the same pictures), else MSVC's own compiler; PS2_COMPILER=msvc
picks MSVC's - and runs the result on the executable from the user's own disc,
printing how far it got.

Some code only exists at run time: kept as data and copied out - a kernel
extension, an overlay - and run where it was copied to. The host records each
such call it meets with the code's source in the executable (--discover); this
translates those at their run addresses and builds again, until a run finds
nothing new. They are translated, not verified: nothing tests them against the
interpreter, which never sees them run.

Nothing from the disc is built into the port: the executable is loaded at run
time, from the game's extracted/ directory.

`pgo` (clang-cl): a build that counts what its code does, run through the
game's scripted runs (game.json's "pgo_runs", else all of port/runs), the
counts merged into build/port/pgo.profdata - which every build after it is
compiled with (--no-pgo: without), the branches laid out and the inlining
chosen as the game runs: Burnout 3's EE and GS threads 10-12% less CPU in a
race, the pictures the same. game.json's "pgo_env" gives a run environment
variables of its own ({"run": {"NAME": "value"}}, a value's "{game}" the
game's folder): a texture pack's folder, say, where one is on this PC - a
path that does not exist leaves the variable out, the run as without it.
What no run reaches is laid out as cold: train the paths players take
(modern graphics, high frame rates, a pack). A profile from older code still serves: what
changed since is compiled as without one. A game whose game.json says
"lto": true is linked with ThinLTO too (clang-cl; PS2_LTO=on/off overrides):
the host's functions the game's code calls inlined across files - Burnout
3's race in virtual time 12% faster again, its link a minute (RE Outbreak's
took half an hour: a game's choice).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys

# The host stops itself at its --timeout (30 s by default); this long past it,
# it is stuck.
HOST_TIMEOUT_SECONDS = 30
HOST_GRACE_SECONDS = 90
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp import cppcheck, gaps, lift, naming, overlay  # noqa: E402
from ps2decomp import game as game_config  # noqa: E402
import vurecomp  # noqa: E402

_FUNCTION = re.compile(r"^void F_([0-9A-F]{8})\(lifted::Context& c\) \{$", re.M)


def translated_functions(directory: str) -> list:
    """Addresses of every translated function in the game's src/lifted."""
    found = set()
    for name in sorted(os.listdir(directory)):
        if name.endswith(".cpp"):
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                found.update(int(m.group(1), 16) for m in _FUNCTION.finditer(handle.read()))
    return sorted(found)


def dispatch_source(addresses, names: dict, untranslated=(), sizes=None) -> str:
    """The dispatch table: every translated function, by address, with its name
    where one is known; functions with no translation are listed with no code, so
    a call to one is reported by name."""
    entries = sorted(set(addresses) | set(untranslated))
    lines = ['#include "ps2host/host.h"', ""]
    lines += ["void F_%08X(lifted::Context& c);" % a for a in addresses]
    lines += ["", "extern const ps2host::FunctionEntry kGuestFunctions[] = {"]
    present = set(addresses)
    for address in entries:
        name = names.get(address)
        label = '"%s"' % name.replace('"', "") if name else "nullptr"
        code = "F_%08X" % address if address in present else "nullptr"
        lines.append("    {0x%08Xu, %s, %s, %du}," % (address, code, label,
                                                   (sizes or {}).get(address, 0)))
    lines += ["};", "extern const std::size_t kGuestFunctionCount = %d;" % len(entries), ""]
    return "\n".join(lines)


def overlays_source(game, out: str, pairs=()) -> str:
    """The overlays' dispatch tables (ps2host::OverlayEntry), and in `out` a source
    for each of their translated files - which stay where lift.py wrote them, in
    the overlay's own src/lifted. Overlays loaded at the same address have
    functions of the same names, so each overlay's are compiled in a namespace of
    its own: its files are included inside it, after the header they include
    (guarded, so the inner include is empty).

    Code a run found in an overlay's range (`pairs`, from discovered.txt) is
    translated from each overlay whose code covers it - the run cannot say
    which was loaded - and added to its table; the check bytes pick the one in
    memory. NOT VERIFIED, as the executable's found code."""
    lines = ['#include "ps2host/host.h"', ""]
    entries, wrappers = [], set()
    for relative in game.overlays:
        part = game_config.load(game.path(relative))
        space = overlay.namespace(part.directory)
        translated = part.path(os.path.join(os.path.dirname(part.generated), "lifted"))
        if not os.path.isdir(translated):
            print("  overlay %s: no translated sources in %s (lift.py --emit)" % (relative, translated))
            continue
        lifted_here = set(translated_functions(translated))
        mine = {(run, source) for run, source in pairs if source == run and run not in lifted_here
                and overlay.in_code(part.code_ranges, run)}
        found = discovered_functions(part, "", mine) if mine else {}
        if found:
            wrapper = "%s_discovered.cpp" % space
            wrappers.add(wrapper)
            body = ['#include "ps2rt/lifted.h"', "", "using namespace generated;",
                    "using namespace lifted;", "", "namespace %s {" % space, "",
                    "/* Reached at run time; no function in the overlay's analysis starts "
                    "there. NOT VERIFIED. */", ""]
            body += [text for _source, text in (found[a] for a in sorted(found))]
            _write_if_changed(os.path.join(out, wrapper), "\n".join(body + ["}", ""]))
        for name in sorted(os.listdir(translated)):
            if name.endswith(".cpp"):
                wrapper = "%s_%s" % (space, name)
                wrappers.add(wrapper)
                path = os.path.abspath(os.path.join(translated, name)).replace("\\", "/")
                _write_if_changed(os.path.join(out, wrapper),
                                  '#include "ps2rt/lifted.h"\n\nnamespace %s {\n#include "%s"\n}\n'
                                  % (space, path))
        addresses = sorted(lifted_here | set(found))
        elf = part.load_elf()
        section = elf.exec_sections[0]
        image = elf.section_data(section)
        sizes = {f["addr"]: f["size"] for f in part.load_database()["functions"]}
        names = {a: v[0] for a, v in naming.load(
            part.path(os.path.join(os.path.dirname(part.status), "names.csv"))).items()}
        label = os.path.basename(os.path.normpath(part.directory))
        lines.append("namespace %s {" % space)
        lines += ["void F_%08X(lifted::Context& c);" % a for a in addresses]
        lines += ["}", "", "static const ps2host::FunctionEntry k_%s[] = {" % space]
        for a in addresses:
            text = ("%s: %s" % (label, names[a]) if a in names else label).replace('"', "")
            lines.append('    {0x%08Xu, %s::F_%08X, "%s", %du},' % (a, space, a, text, sizes.get(a, 0)))
        lines += ["};", "static const u32 k_%s_checks[] = {" % space]
        for a in addresses:
            words = struct.unpack_from("<4I", image.ljust(a - section.addr + 16, b"\0"), a - section.addr)
            lines.append("    " + ", ".join("0x%08Xu" % w for w in words) + ",")
        lines += ["};", ""]
        low = min(r[0] for r in part.code_ranges) if part.code_ranges else section.addr
        high = max(r[1] for r in part.code_ranges) if part.code_ranges else section.addr + len(image)
        entries.append('    {"%s", 0x%08Xu, 0x%08Xu, k_%s, %d, k_%s_checks},'
                       % (label, low, high, space, len(addresses), space))
        print("overlay %s: %d translated functions, %d found at run time"
              % (label, len(addresses) - len(found), len(found)))
    for stale in os.listdir(out):
        if stale.startswith("overlay_") and stale.endswith(".cpp") and stale not in wrappers:
            os.remove(os.path.join(out, stale))
    lines.append("extern const ps2host::OverlayEntry kGuestOverlays[] = {")
    lines += entries or ["    {nullptr, 0, 0, nullptr, 0, nullptr},"]
    lines += ["};", "extern const std::size_t kGuestOverlayCount = %d;" % len(entries), ""]
    return "\n".join(lines)


def discovered_pairs(path: str) -> set:
    """(run address, source address) for each piece of code a run found."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        return {(int(run, 16), int(source, 16)) for run, source in
                (line.split() for line in handle if line.strip())}


def discovered_functions(game, path: str, pairs=None) -> dict:
    """{run address: C++} for the code runs found copied at run time, translated
    at their run addresses from the executable's bytes - or, for an overlay
    (`game` is one), from the overlay's, given its `pairs`."""
    if pairs is None:
        pairs = discovered_pairs(path)
    if not pairs:
        return {}
    elf = game.load_elf()
    section = elf.exec_sections[0]
    image = elf.section_data(section)
    # A run is lifted from a slice of the image; its jump tables lie elsewhere
    # in it (Burnout 3's 0x00242AA0 reaches 0x00242BD0 through one).
    lift.use_sections([(image, section.addr)] + [(elf.section_data(s), s.addr) for s in elf.sections
                                                  if s.is_alloc and s.size and s is not section and s.type != 8])
    found = {}
    for run, source in sorted(pairs):
        offset = source - section.addr
        code = image[offset:offset + gaps.MAX_FUNCTION]
        try:
            length = gaps.function_extent(code, run, run)
            found[run] = (source, lift.lift_function(code, run, run, run + length))
        except (ValueError, lift.Refused) as exc:
            print("  cannot translate the code at 0x%08X (from 0x%08X): %s" % (run, source, exc))
    return found


def generate(game) -> str:
    """Write the dispatch table; returns its directory."""
    lifted = game.path(os.path.join(os.path.dirname(game.generated), "lifted"))
    if not os.path.isdir(lifted):
        raise SystemExit("no translated sources in %s: run lift.py --emit first" % lifted)
    addresses = translated_functions(lifted)
    database = game.load_database()
    status_dir = os.path.dirname(game.status)
    names = {a: v[0] for a, v in naming.load(game.path(os.path.join(status_dir, "names.csv"))).items()}
    untranslated = [f["addr"] for f in database["functions"]
                    if f["size"] and f["addr"] not in set(addresses)]
    out = game.path(os.path.join(game.build, "port", "generated"))
    os.makedirs(out, exist_ok=True)
    pairs = discovered_pairs(game.path(os.path.join(game.build, "port", "discovered.txt")))
    # Code found in an overlay's range is the overlay's to translate.
    ranges = [r for relative in game.overlays
              for r in game_config.load(game.path(relative)).code_ranges]
    in_overlay = {p for p in pairs if p[0] == p[1] and any(lo <= p[0] < hi for lo, hi in ranges)}
    discovered = discovered_functions(game, "", pairs - in_overlay)
    # What the analysis has since found is translated - and verified - with the rest.
    discovered = {a: v for a, v in discovered.items() if a not in set(addresses)}
    copies = ['#include "ps2rt/lifted.h"', "", "using namespace generated;",
              "using namespace lifted;", "",
              "/* Code found copied at run time, translated at its run address. NOT VERIFIED. */",
              ""]
    for run, (source, text) in sorted(discovered.items()):
        if source == run:
            copies += ["/* 0x%08X: reached at run time; no function in the analysis starts "
                       "here */" % run, text]
            names.setdefault(run, "found at run time")
        else:
            copies += ["/* 0x%08X: a copy of the executable's code at 0x%08X */" % (run, source),
                       text]
            names.setdefault(run, "copy of 0x%08X" % source)
    _write_if_changed(os.path.join(out, "discovered.cpp"), "\n".join(copies) + "\n")
    path = os.path.join(out, "dispatch.cpp")
    sizes = {f["addr"]: f["size"] for f in database["functions"]}
    text = dispatch_source(sorted(set(addresses) | set(discovered)), names, untranslated, sizes)
    _write_if_changed(path, text)
    # Named so that the overlays' own sources (overlay_*.cpp) never take it for one.
    _write_if_changed(os.path.join(out, "dispatch_overlays.cpp"),
                      overlays_source(game, out, in_overlay))
    print("dispatch table: %d translated functions, %d found at run time, %d without a "
          "translation" % (len(addresses), len(discovered), len(untranslated)))
    # The microprograms runs recorded (--vu-record), compiled ahead of time.
    vurecomp.main(["--manifest", vu_manifest_path(game), "--out-dir", out])
    return out


def vu_manifest_path(game) -> str:
    """Where runs record the microprograms they start (--vu-record): game-derived, in build/."""
    return game.path(os.path.join(game.build, "vu_programs.txt"))


def _write_if_changed(path: str, text: str) -> None:
    """Leave an unchanged file untouched, so the build need not recompile it."""
    old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
    if text != old:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


def _visual_studio_tool(*parts) -> str:
    vcvars = cppcheck.find_vcvars()
    root = os.path.abspath(os.path.join(os.path.dirname(vcvars), "..", "..", ".."))
    return os.path.join(root, "Common7", "IDE", "CommonExtensions", "Microsoft", "CMake", *parts)


def _clang_cl() -> str | None:
    """LLVM's clang-cl, if installed and not turned away (PS2_COMPILER=msvc):
    on the path, in LLVM's own folder, or the one Visual Studio installs."""
    if os.environ.get("PS2_COMPILER", "").lower() == "msvc":
        return None
    candidates = [shutil.which("clang-cl"),
                  os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "LLVM", "bin", "clang-cl.exe")]
    try:
        vc = os.path.abspath(os.path.join(os.path.dirname(cppcheck.find_vcvars()), "..", ".."))
        candidates.append(os.path.join(vc, "Tools", "Llvm", "x64", "bin", "clang-cl.exe"))
    except Exception:
        pass
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def _forget_other_compiler(binary: str, compiler: str | None) -> None:
    """A build folder configured with the other compiler: CMake keeps the
    compiler it found first, so its cache goes (the objects are rebuilt)."""
    cache = os.path.join(binary, "CMakeCache.txt")
    if not os.path.exists(cache):
        return
    text = open(cache, encoding="utf-8", errors="replace").read()
    configured = re.search(r"^CMAKE_CXX_COMPILER:[A-Z]+=(.*)$", text, re.M)
    configured = configured.group(1).strip() if configured else ""
    clang = "clang-cl" in configured.lower()
    if clang == (compiler is not None) and (compiler is None or
                                             os.path.normcase(os.path.abspath(configured)) == os.path.normcase(compiler)):
        return
    print("build: %s now (the folder had %s) - configuring again"
          % (compiler or "MSVC", configured or "another compiler"))
    os.remove(cache)
    shutil.rmtree(os.path.join(binary, "CMakeFiles"), ignore_errors=True)


def _build_jobs() -> int:
    """Compilers at once: by the memory there is, not the processors - a
    file of compiled VU code takes the compiler 2 GB and more, and a build
    from nothing with Ninja's default (processors + 2) ran out of it on 16 GB."""
    jobs = os.cpu_count() or 4
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong), ("free_physical", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("free_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("free_virtual", ctypes.c_ulonglong),
                        ("free_extended", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            jobs = min(jobs, int(status.total_physical / (2.5 * 2 ** 30)))
    return max(1, jobs)


def _game_env(**extra) -> dict:
    """The environment a port runs in: this process's, less what verification
    sets for itself - ps2decomp/cppcheck.py, imported above, makes the runtime
    round to nearest as the interpreter does (PS2RT_EE_ROUND_NEAREST); the
    game itself must round as the console does (a run started here drew
    other pictures, and played so)."""
    env = dict(os.environ, **extra)
    env.pop("PS2RT_EE_ROUND_NEAREST", None)
    return env


def _game_json(game) -> dict:
    """game.json as written (its build options: lto, pgo_runs)."""
    with open(game.path("game.json"), encoding="utf-8") as handle:
        return json.load(handle)


def pgo_profile(game) -> str:
    """Where `pgo` leaves the merged profile every build after it uses (in
    full: the compiler runs in the build's folder)."""
    return os.path.abspath(game.path(os.path.join(game.build, "port", "pgo.profdata")))


def build(game, generated: str, sse4: bool = False, pgo: str = "") -> str:
    """Configure and build the port; returns the executable's path. `sse4`:
    the build for processors without AVX2 (PS2HOST_AVX2=OFF), in a folder of
    its own, copied beside the main build as <name>-sse4.exe - which the main
    build starts on such a processor (engine/host/src/cpu_check.cpp). The
    compiler: clang-cl when installed (_clang_cl), else MSVC's. `pgo`:
    "generate" for the build that counts (a folder of its own: pgo below),
    "off" for none; by default the profile `pgo` made, if there is one."""
    source = game.path("port")
    folder = "cmake-pgo-generate" if pgo == "generate" else "cmake-sse4" if sse4 else "cmake"
    binary = game.path(os.path.join(game.build, "port", folder))
    os.makedirs(binary, exist_ok=True)
    cmake = _visual_studio_tool("CMake", "bin", "cmake.exe")
    ninja = _visual_studio_tool("Ninja", "ninja.exe")
    clang = _clang_cl()
    _forget_other_compiler(binary, clang)
    compilers = (' -DCMAKE_C_COMPILER="%s" -DCMAKE_CXX_COMPILER="%s"' % ((clang.replace("\\", "/"),) * 2)
                 if clang else "")
    # Named every time (CMake keeps what it was last told).
    profile = ""
    if clang and pgo == "generate":
        profile = "generate"
    elif clang and pgo != "off" and os.path.exists(pgo_profile(game)):
        profile = pgo_profile(game).replace("\\", "/")
    # ThinLTO with clang-cl where the game asks for it (game.json's "lto":
    # its link can take long - RE Outbreak's half an hour, Burnout 3's a
    # minute), or PS2_LTO=on/off says; never for the build that only counts.
    wanted = os.environ.get("PS2_LTO", "").lower()
    lto = "ON" if clang and pgo != "generate" and (wanted == "on" or (wanted != "off" and _game_json(game).get("lto"))) \
        else "OFF"
    script = os.path.join(binary, "build.bat")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write('@echo off\ncall "%s" >nul\n' % cppcheck.find_vcvars())
        handle.write('"%s" -S "%s" -B "%s" -G Ninja -DCMAKE_BUILD_TYPE=Release '
                     '-DCMAKE_MAKE_PROGRAM="%s" -DPS2_GENERATED_DIR="%s" -DPS2HOST_AVX2=%s "-DPS2HOST_PGO=%s"'
                     ' -DPS2HOST_LTO=%s -DPS2HOST_LTO_JOBS=%d%s >nul || exit /b 1\n'
                     % (cmake, source, binary, ninja, generated.replace("\\", "/"), "OFF" if sse4 else "ON",
                        profile, lto, _build_jobs(), compilers))
        handle.write('"%s" --build "%s" -j %d || exit /b 1\n' % (cmake, binary, _build_jobs()))
    result = subprocess.run(["cmd", "/c", script], capture_output=True, text=True)
    if result.returncode:
        errors = [line for line in (result.stdout + result.stderr).splitlines()
                  if "error" in line.lower()]
        raise SystemExit("port build failed:\n" + "\n".join(errors[:40] or
                                                           (result.stdout + result.stderr)[-4000:].splitlines()))
    target = os.path.basename(os.path.normpath(game.directory))
    executable = os.path.join(binary, target + ".exe")
    if pgo == "generate":
        # (Its tests would count too: the main build's run them.)
        print("built %s (clang-cl, counting what runs)" % executable)
        return executable
    tests = os.path.join(binary, "ps2host", "ps2host_tests.exe")
    result = subprocess.run([tests], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode:
        raise SystemExit("the host's own tests failed")
    if sse4:
        beside = game.path(os.path.join(game.build, "port", "cmake", target + "-sse4.exe"))
        os.makedirs(os.path.dirname(beside), exist_ok=True)
        shutil.copyfile(executable, beside)
        executable = beside
    print("built %s (%s%s%s)" % (executable, "clang-cl" if clang else "MSVC",
                                 ", profile-guided" if profile else "", ", ThinLTO" if lto == "ON" else ""))
    return executable


def pgo_runs(game) -> list:
    """The scripted runs a profile is made from: game.json's "pgo_runs", else
    every one in port/runs."""
    named = _game_json(game).get("pgo_runs")
    if named:
        return list(named)
    runs = game.path(os.path.join("port", "runs"))
    return sorted(name[:-5] for name in os.listdir(runs) if name.endswith(".args")) if os.path.isdir(runs) else []


def pgo(game, generated: str) -> str:
    """The profile-guided build (the module's docstring): the counting build,
    its runs, their counts merged into pgo_profile, and the build with them.
    Returns the last's executable."""
    clang = _clang_cl()
    if not clang:
        raise SystemExit("pgo: a profile-guided build needs LLVM's clang-cl")
    merge = os.path.join(os.path.dirname(clang), "llvm-profdata.exe")
    if not os.path.isfile(merge):
        raise SystemExit("pgo: llvm-profdata.exe not found beside %s" % clang)
    runs = pgo_runs(game)
    if not runs:
        raise SystemExit("pgo: no scripted runs to make a profile from (port/runs/*.args)")
    counting = os.path.abspath(build(game, generated, pgo="generate"))
    raw = game.path(os.path.join(game.build, "port", "pgo"))
    shutil.rmtree(raw, ignore_errors=True)
    os.makedirs(raw)
    # Each run's counts in a file of its own (%p: the process).
    env = _game_env(LLVM_PROFILE_FILE=os.path.join(os.path.abspath(raw), "run-%p.profraw"))
    regress = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regress.py")
    for name in runs:
        gpu = os.path.exists(game.path(os.path.join("port", "runs", name + ".gpu.crc")))
        run_env = dict(env)
        for key, value in (_game_json(game).get("pgo_env") or {}).get(name, {}).items():
            value = value.replace("{game}", os.path.abspath(game.directory))
            if os.sep in value or "/" in value:
                if not os.path.exists(value):
                    print("pgo: %s: %s not here, left out" % (name, value), flush=True)
                    continue
            run_env[key] = value
        print("pgo: %s" % name, flush=True)
        # (Its pictures checked as ever: a difference is reported, the counts kept.)
        subprocess.run([sys.executable, regress, "--game", game.directory, name, "--exe", counting]
                       + (["--gpu"] if gpu else []), env=run_env)
    counts = [os.path.join(raw, f) for f in sorted(os.listdir(raw))
              if f.endswith(".profraw") and os.path.getsize(os.path.join(raw, f))]
    if not counts:
        raise SystemExit("pgo: the runs left no counts in %s" % raw)
    subprocess.run([merge, "merge", "-o", pgo_profile(game)] + counts, check=True)
    print("pgo: %d runs' counts in %s" % (len(counts), pgo_profile(game)))
    return build(game, generated)


def discoveries_path(game) -> str:
    return game.path(os.path.join(game.build, "port", "discovered.txt"))


def run(game, executable: str, extra) -> int:
    elf = game.path(os.path.join(game.extracted, game.executable))
    if not os.path.exists(elf):
        raise SystemExit("no executable at %s: extract it from your disc first (isotool.py)" % elf)
    report = game.path(os.path.join(game.build, "port", "report.txt"))
    with open(report, "w", encoding="utf-8") as out:
        disc = game.path(game.iso) if game.iso else ""
        options = ["--disc", disc] if disc and os.path.exists(disc) else []
        timeout = HOST_TIMEOUT_SECONDS
        extra = list(extra)
        if "--timeout" in extra and extra.index("--timeout") + 1 < len(extra):
            timeout = int(extra[extra.index("--timeout") + 1])
        try:
            result = subprocess.run([executable, elf, "--discover", discoveries_path(game)]
                                    + options + list(extra), stdout=out, stderr=subprocess.STDOUT,
                                    timeout=timeout + HOST_GRACE_SECONDS, env=_game_env())
            code = result.returncode
        except subprocess.TimeoutExpired:
            # The host stops itself at its own time limit; outliving it by this
            # much means the host itself is stuck, which is a host bug to report.
            out.write("\n=== host report ===\nkilled: the host outlived its own time limit by "
                      "%ds - a bug in the host, not the game\n" % HOST_GRACE_SECONDS)
            code = 3
    with open(report, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    print(text[text.find("=== host report ==="):] if "=== host report ===" in text else text[-3000:])
    print("full log: %s" % report)
    return code


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["init"]:
        # A new game: there is no game directory yet for --game to name.
        import init_game
        return init_game.main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("action", nargs="?", default="all", choices=("all", "build", "run", "pgo"))
    parser.add_argument("--sse4", action="store_true",
                        help="build: the build for processors without AVX2 (<name>-sse4.exe beside the main one)")
    parser.add_argument("--no-pgo", action="store_true",
                        help="build: without the profile `pgo` made (build/port/pgo.profdata)")
    parser.add_argument("--rounds", type=int, default=60,
                        help="build-and-run rounds to translate code found at run time")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="options for the port itself, after --")
    args = parser.parse_args(argv)
    game = game_config.from_args(args)
    extra = [a for a in args.extra if a != "--"]
    if args.action == "run":
        target = os.path.basename(os.path.normpath(game.directory))
        executable = game.path(os.path.join(game.build, "port", "cmake", target + ".exe"))
        return run(game, executable, extra)
    if args.action == "build":
        build(game, generate(game), sse4=args.sse4, pgo="off" if args.no_pgo else "")
        return 0
    if args.action == "pgo":
        pgo(game, generate(game))
        return 0
    # Build and run until a run finds no more code copied at run time.
    for _round in range(args.rounds):
        executable = build(game, generate(game))
        before = _count_lines(discoveries_path(game))
        code = run(game, executable, extra)
        if _count_lines(discoveries_path(game)) == before:
            return code
        print("found code copied at run time: translating it and building again\n")
    return code


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    # Distinct entries: code that cannot be translated is met - and recorded -
    # again on every run, and must not count as something new.
    with open(path, encoding="utf-8") as handle:
        return len({line.strip() for line in handle if line.strip()})


if __name__ == "__main__":
    raise SystemExit(main())
