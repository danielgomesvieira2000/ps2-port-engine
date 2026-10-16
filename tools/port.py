#!/usr/bin/env python3
"""Build a game's native port and run it.

    python engine/tools/port.py --game games/<name>            # generate, build, run
    python engine/tools/port.py --game games/<name> build      # generate and build only
    python engine/tools/port.py --game games/<name> run -- --trace
    python engine/tools/port.py init <name> --iso "<disc image>" --title "..."  # a new game

The port is the game's translated functions (src/lifted, from lift.py --emit) on
the native host (engine/host): guest memory, a dispatch table from guest address
to translated function, the EE kernel's services. This generates the dispatch
table, builds games/<name>/port/ with CMake and Ninja under MSVC, and runs the
result on the executable from the user's own disc, printing how far it got.

Some code only exists at run time: kept as data and copied out - a kernel
extension, an overlay - and run where it was copied to. The host records each
such call it meets with the code's source in the executable (--discover); this
translates those at their run addresses and builds again, until a run finds
nothing new. They are translated, not verified: nothing tests them against the
interpreter, which never sees them run.

Nothing from the disc is built into the port: the executable is loaded at run
time, from the game's extracted/ directory.
"""

from __future__ import annotations

import argparse
import os
import re
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
    """LLVM's clang-cl, when installed and not declined (PS2_COMPILER=msvc).
    It builds the same sources with MSVC's headers and libraries; Burnout 3's
    race measured 33 -> 36-40 pictures a second on an i7-10700F (the GS
    thread no longer the limit)."""
    if os.environ.get("PS2_COMPILER", "").lower() == "msvc":
        return None
    for root in (os.environ.get("LLVM_ROOT", ""), r"C:\Program Files\LLVM"):
        path = os.path.join(root, "bin", "clang-cl.exe") if root else ""
        if path and os.path.exists(path):
            return path
    return None


def _forget_other_compiler(binary: str, clang: bool) -> None:
    """A build directory configured for the other compiler (clang-cl or
    MSVC's cl) is configured again from scratch: CMake keeps the first one."""
    cache = os.path.join(binary, "CMakeCache.txt")
    if not os.path.exists(cache):
        return
    text = open(cache, encoding="utf-8", errors="replace").read()
    found = re.search(r"^CMAKE_CXX_COMPILER:[A-Z]+=(.*)$", text, re.M)
    if found and (os.path.basename(found.group(1)).lower() == "clang-cl.exe") != clang:
        os.remove(cache)
        import shutil
        shutil.rmtree(os.path.join(binary, "CMakeFiles"), ignore_errors=True)


def build(game, generated: str) -> str:
    """Configure and build the port; returns the executable's path."""
    source = game.path("port")
    binary = game.path(os.path.join(game.build, "port", "cmake"))
    os.makedirs(binary, exist_ok=True)
    cmake = _visual_studio_tool("CMake", "bin", "cmake.exe")
    ninja = _visual_studio_tool("Ninja", "ninja.exe")
    clang = _clang_cl()
    compilers = ""
    if clang:
        compilers = ' -DCMAKE_C_COMPILER="%s" -DCMAKE_CXX_COMPILER="%s"' % (clang, clang)
    _forget_other_compiler(binary, clang is not None)
    script = os.path.join(binary, "build.bat")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write('@echo off\ncall "%s" >nul\n' % cppcheck.find_vcvars())
        handle.write('"%s" -S "%s" -B "%s" -G Ninja -DCMAKE_BUILD_TYPE=Release%s '
                     '-DCMAKE_MAKE_PROGRAM="%s" -DPS2_GENERATED_DIR="%s" >nul || exit /b 1\n'
                     % (cmake, source, binary, compilers, ninja, generated.replace("\\", "/")))
        handle.write('"%s" --build "%s" || exit /b 1\n' % (cmake, binary))
    result = subprocess.run(["cmd", "/c", script], capture_output=True, text=True)
    if result.returncode:
        errors = [line for line in (result.stdout + result.stderr).splitlines()
                  if "error" in line.lower()]
        raise SystemExit("port build failed:\n" + "\n".join(errors[:40] or
                                                           (result.stdout + result.stderr)[-4000:].splitlines()))
    tests = os.path.join(binary, "ps2host", "ps2host_tests.exe")
    result = subprocess.run([tests], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode:
        raise SystemExit("the host's own tests failed")
    target = os.path.basename(os.path.normpath(game.directory))
    executable = os.path.join(binary, target + ".exe")
    print("built %s" % executable)
    return executable


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
                                    timeout=timeout + HOST_GRACE_SECONDS)
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
    parser.add_argument("action", nargs="?", default="all", choices=("all", "build", "run"))
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
        build(game, generate(game))
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
