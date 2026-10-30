"""A game's Windows release: the built port, its player README and the
licences of what it is built with, in one zip - checked to carry nothing of
the game's but its translated program, and nothing of the machine it was
built on.

    python engine/tools/package.py --game games/<name>

Build first (engine/tools/port.py --game games/<name> pgo for the
profile-guided build - or build - then --sse4 build). Written to
<game>/build/release/<release name>-<version>-windows-x64.zip (build/ is never
committed). Refuses:
  - a build carrying an icon made from the disc (build/port/icon.ico: the
    publisher's artwork, for a player's own build only) - delete it and
    build again;
  - an executable that needs the Visual C++ runtime installed (it must be
    linked in), or that names a folder of the machine it was built on.

What is the game's own comes from its game.json:
  "title"      the game's name (the notices' heading);
  "version"    the release's version ({version} in the README is replaced);
  "release"    optional: {"name": the zip's and its folder's name (default:
               the title's letters and digits, then "-PC"),
               "extras": {path in the zip: file in the game's directory}
               - more text files for the player (a texture pack folder's
               README, say), written with Windows line ends}.
The player README is the game's port/release/README.txt.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile

ENGINE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ENGINE)

# The libraries built into the port, and where their licence is in their source.
LICENCES = [
    ("SDL 3", "sdl3-src", ["LICENSE.txt"]),
    ("volk", "volk-src", ["LICENSE.md"]),
    ("glslang", "glslang-src", ["LICENSE.txt"]),
    ("Vulkan-Headers", "vulkan_headers-src", ["LICENSE.md", "LICENSE.txt"]),
]
# Single headers built in for texture packs and custom songs
# (engine/host/CMakeLists.txt): the licence in each, from where it starts to
# where its comment ends.
HEADER_LICENCES = [
    ("xxHash", "xxhash.h", "xxHash - Extremely Fast Hash algorithm"),
    ("bcdec", "bcdec.h", "This software is available under 2 licenses"),
    ("stb_image", "stb_image.h", "This software is available under 2 licenses"),
    ("minimp3", "minimp3.h", "To the extent possible under law"),
]
# Files of the engine's own third-party code: (name, path in the engine).
FILE_LICENCES = [
    ("SMAA", os.path.join("gpu", "third_party", "smaa", "LICENSE.txt")),
]


def fail(why):
    raise SystemExit("package: " + why)


def header_licence(headers, name, start):
    path = os.path.join(headers, name)
    if not os.path.exists(path):
        fail("no %s in %s: build first" % (name, headers))
    text = open(path, encoding="utf-8", errors="replace").read()
    at = text.find(start)
    end = text.find("*/", at)
    if at < 0 or end < 0:
        fail("no licence found in %s" % path)
    lines = [line.rstrip() for line in text[at:end].splitlines()]
    return "\n".join(line[3:] if line.startswith(" * ") else line.lstrip(" *") for line in lines).strip()


def imports(build, exe):
    """The DLLs the executable imports (dumpbin, with the compiler's environment)."""
    from ps2decomp import cppcheck
    script = os.path.join(build, "release_dumpbin.bat")
    with open(script, "w") as f:
        f.write('@echo off\ncall "%s" >nul\ndumpbin /nologo /dependents "%s"\n' % (cppcheck.find_vcvars(), exe))
    out = subprocess.run(["cmd", "/c", script], capture_output=True, text=True).stdout
    return sorted(set(m.lower() for m in re.findall(r"^\s+(\S+\.dll)\s*$", out, re.M)))


def check_executable(build, exe, roots):
    dlls = imports(build, exe)
    runtime = [d for d in dlls if d.startswith(("msvcp", "vcruntime", "ucrtbase", "api-ms-win-crt"))]
    if runtime:
        fail("%s needs the Visual C++ runtime installed (%s): it must be linked in"
             % (os.path.basename(exe), ", ".join(runtime)))
    data = open(exe, "rb").read()
    for folder in roots:
        for form in (folder, folder.replace("\\", "/")):
            if form.encode() in data or form.lower().encode() in data:
                fail("%s names a folder of this machine: %s" % (os.path.basename(exe), form))
    if re.search(rb"[A-Za-z]:\\Users\\[^\\\x00]{1,64}\\", data):
        fail("%s names a user's folder" % os.path.basename(exe))
    return dlls


def notices(title, deps, headers):
    heading = "Third-party software in the %s PC port" % title
    lines = [heading, "=" * len(heading), ""]
    for name, folder, files in LICENCES:
        path = next((os.path.join(deps, folder, f) for f in files if os.path.exists(os.path.join(deps, folder, f))),
                    None)
        if not path:
            fail("no licence for %s in %s" % (name, os.path.join(deps, folder)))
        with open(path, encoding="utf-8", errors="replace") as f:
            lines += ["-" * 72, name, "-" * 72, f.read().strip(), ""]
    for name, header, start in HEADER_LICENCES:
        lines += ["-" * 72, name, "-" * 72, header_licence(headers, header, start), ""]
    for name, relative in FILE_LICENCES:
        path = os.path.join(ENGINE, relative)
        if not os.path.exists(path):
            fail("no licence for %s at %s" % (name, path))
        with open(path, encoding="utf-8", errors="replace") as f:
            lines += ["-" * 72, name, "-" * 72, f.read().strip(), ""]
    return "\r\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True, help="the game's directory (with game.json)")
    args = parser.parse_args()
    game = os.path.abspath(args.game)
    root = os.path.normpath(os.path.join(ENGINE, ".."))
    target = os.path.basename(os.path.normpath(game))
    with open(os.path.join(game, "game.json"), encoding="utf-8") as f:
        spec = json.load(f)
    title = spec.get("title") or spec.get("name") or target
    version = spec.get("version", "0.0.0")
    release = spec.get("release") or {}
    build = os.path.join(game, "build")
    cmake = os.path.join(build, "port", "cmake")
    exe = os.path.join(cmake, target + ".exe")
    # The build for processors without AVX2 (port.py build --sse4), which the
    # main one starts on one (engine/host/src/cpu_check.cpp).
    exe_sse4 = os.path.join(cmake, target + "-sse4.exe")
    if not os.path.exists(exe):
        fail("no build at %s: build first" % exe)
    if not os.path.exists(exe_sse4):
        fail("no build at %s: build it (python engine/tools/port.py --game %s build --sse4)" % (exe_sse4, args.game))
    if os.path.getmtime(exe_sse4) < os.path.getmtime(exe) - 3600:
        fail("%s is older than %s.exe: build it again (port.py build --sse4)" % (exe_sse4, target))
    if os.path.exists(os.path.join(build, "port", "icon.ico")):
        fail("build/port/icon.ico is the disc-made logo (the publisher's artwork): delete it, build again")
    roots = {os.path.expanduser("~"), root}
    dlls = []
    for path in (exe, exe_sse4):
        dlls = check_executable(build, path, roots)
    text = notices(title, os.path.join(cmake, "_deps"), os.path.join(cmake, "ps2host", "third_party"))
    with open(os.path.join(game, "port", "release", "README.txt"), encoding="utf-8") as f:
        readme = f.read().replace("{version}", version)
    name = "%s-%s" % (release.get("name") or re.sub(r"[^A-Za-z0-9]", "", title) + "-PC", version)
    out_dir = os.path.join(build, "release")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "%s-windows-x64.zip" % name)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(exe, "%s/%s.exe" % (name, target))
        z.write(exe_sse4, "%s/%s-sse4.exe" % (name, target))
        z.writestr("%s/README.txt" % name, readme.replace("\n", "\r\n"))
        z.writestr("%s/THIRD-PARTY-NOTICES.txt" % name, text)
        for inside, source in (release.get("extras") or {}).items():
            with open(os.path.join(game, source), encoding="utf-8") as f:
                z.writestr("%s/%s" % (name, inside), f.read().replace("\n", "\r\n"))
    print("package: %s (%.1f MB; imports %s)" % (out, os.path.getsize(out) / 1e6, ", ".join(dlls)))


if __name__ == "__main__":
    main()
