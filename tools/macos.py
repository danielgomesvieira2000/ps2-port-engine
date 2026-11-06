#!/usr/bin/env python3
"""macOS (Apple silicon): a port built and run on the Mac itself, with Apple's
clang, CMake and Ninja - the Vulkan display through MoltenVK.

    python engine/tools/macos.py --game games/<name> build
    python engine/tools/macos.py --game games/<name> run [-- <the port's options>]
    python engine/tools/macos.py --game games/<name> app     # a double-clickable .app
    python engine/tools/macos.py --game games/<name> app --icon <picture file or URL>

The dispatch table is generated first (port.py's, as for every build). The
build is kept in <game>/build/macos/<target>, the program beside it.

Needs: Xcode's command line tools, CMake, Ninja, Python 3, and for the display
MoltenVK and the Vulkan loader (`brew install molten-vk vulkan-loader`).
`run` puts Homebrew's lib folder on the loader's path and names MoltenVK's
driver; `app` bundles both.

`app` builds first when the program is not built yet, then writes
<game>/build/macos/<Title>.app: the program, the Vulkan loader and MoltenVK,
a launcher that asks for the disc image the first time (remembering it) and
keeps saves and settings in ~/Library/Application Support/<Title>. Nothing of
the game's is inside: it plays from the player's own disc image. Signed ad hoc,
so it runs on the Mac that made it; another Mac's Gatekeeper needs a right-click
Open (a release for others wants a Developer ID and notarisation).

The icon is the disc's own title logo (port/tools/make_icon.py reads it from the
disc), tilted on a rounded tile. `--icon` takes a picture of your own instead - a
file, or an https URL: fetched to your machine once and kept in <game>/build/port/
with your other build files, never committed (the picture is the publisher's).
Delete build/port/icon_mac.png to go back to the disc's logo.
"""

import argparse
import os
import re
import urllib.request
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from ps2decomp import game as game_config  # noqa: E402
import port  # noqa: E402

# The oldest macOS the build runs on: os_sync_wait_on_address (fiber.cpp) is 14.4's.
DEPLOYMENT_TARGET = "14.4"
GB = 1 << 30


def tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise SystemExit("macos: %s not found (brew install %s)" % (name, name))
    return found


def jobs() -> int:
    """Compilers at once: by the memory there is - a file of compiled VU code
    takes the compiler 2 GB and more (port.py's _build_jobs)."""
    if os.environ.get("PS2_BUILD_JOBS", "").isdigit():
        return max(1, int(os.environ["PS2_BUILD_JOBS"]))
    memory = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout or 0)
    return max(1, min(os.cpu_count() or 4, int(memory / (2.5 * GB))))


def target_of(game) -> str:
    return os.path.basename(os.path.normpath(game.directory))


def folder_of(game) -> str:
    return game.path(os.path.join(game.build, "macos", target_of(game)))


def build(game, lto: bool) -> str:
    generated = port.generate(game)
    binary = folder_of(game)
    os.makedirs(binary, exist_ok=True)
    target = target_of(game)
    configure = [tool("cmake"), "-S", game.path("port"), "-B", binary, "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_MAKE_PROGRAM=" + tool("ninja"),
                 "-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_OSX_DEPLOYMENT_TARGET=" + DEPLOYMENT_TARGET,
                 "-DPS2_GENERATED_DIR=" + generated, "-DPS2HOST_LTO=" + ("ON" if lto else "OFF")]
    print("+ " + " ".join(configure[:4]) + " ...", flush=True)
    subprocess.run(configure, check=True)
    subprocess.run([tool("cmake"), "--build", binary, "-j", str(jobs())], check=True)
    tests = os.path.join(binary, "ps2host", "ps2host_tests")
    if os.path.exists(tests):
        result = subprocess.run([tests], capture_output=True, text=True, env=run_env())
        print(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "(no test output)")
        if result.returncode:
            raise SystemExit("the host's own tests failed")
    executable = os.path.join(binary, target)
    print("built " + executable)
    return executable


LAUNCHER = """#!/bin/bash
# Starts {title}: the disc image from the first run's choice, saves and settings in
# the user's Library, MoltenVK and the Vulkan loader from this app.
here="$(cd "$(dirname "$0")/.." && pwd)"
data="$HOME/Library/Application Support/{folder}"
mkdir -p "$data"
cd "$data" || exit 1
disc="$(cat disc.txt 2>/dev/null)"
until [ -n "$disc" ] && [ -f "$disc" ]; do
    disc="$(osascript -e 'POSIX path of (choose file with prompt "Choose your {title} disc image (.iso)" of type {{"iso", "public.iso-image", "public.data"}})' 2>/dev/null)" || exit 0
    [ -n "$disc" ] && printf '%s' "$disc" > disc.txt
done
export DYLD_FALLBACK_LIBRARY_PATH="$here/Frameworks"
export VK_ICD_FILENAMES="$here/Resources/vulkan/icd.d/MoltenVK_icd.json"
# SDL makes the window's surface: with the loader the instance came from, not MoltenVK itself.
export SDL_VULKAN_LIBRARY="$here/Frameworks/libvulkan.1.dylib"
exec "$here/MacOS/{target}" --disc "$disc" --memcard "$data/memcard" --window --gpu --audio \\
    --timeout 315360000 >> "$data/{target}.log" 2>&1
"""

PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>{title}</string>
<key>CFBundleDisplayName</key><string>{title}</string>
<key>CFBundleIdentifier</key><string>pc.port.{target}</string>
<key>CFBundleExecutable</key><string>{target}-launcher</string>
<key>CFBundleIconFile</key><string>AppIcon</string>
<key>CFBundleVersion</key><string>{version}</string>
<key>CFBundleShortVersionString</key><string>{version}</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>LSMinimumSystemVersion</key><string>{deployment}</string>
<key>LSApplicationCategoryType</key><string>public.app-category.games</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
"""


def icon_tile(logo):
    """The 1024 px icon: the logo tilted and scaled to run corner to corner on a white macOS rounded square, a
    soft shadow under both (Pillow image in, Pillow image out)."""
    import math
    from PIL import Image, ImageDraw, ImageFilter
    size, body, angle = 1024, 824, 17
    offset = (size - body) // 2
    logo = logo.convert("RGBA")
    logo = logo.crop(logo.getbbox())
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((offset, offset + 12, offset + body, offset + body + 12), 185, fill=(0, 0, 0, 90))
    tile.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)))
    face = Image.new("RGBA", (body, body))
    lines = ImageDraw.Draw(face)
    for y in range(body):
        grey = int(255 - 18 * y / body)
        lines.line((0, y, body, y), fill=(grey, grey, grey, 255))
    tilt = math.radians(angle)
    width = int(body * 1.02 / (math.cos(tilt) + logo.height / logo.width * math.sin(tilt)))
    big = logo.resize((width, round(logo.height * width / logo.width)), Image.LANCZOS)
    big = big.rotate(angle, resample=Image.BICUBIC, expand=True)
    under = Image.new("RGBA", big.size, (0, 0, 0, 0))
    under.putalpha(big.getchannel("A").point(lambda a: a * 0.35))
    under = under.filter(ImageFilter.GaussianBlur(10))
    where = ((body - big.width) // 2, (body - big.height) // 2)
    face.alpha_composite(under, (where[0] + 6, where[1] + 14))
    face.alpha_composite(big, where)
    mask = Image.new("L", (body, body), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, body - 1, body - 1), 185, fill=255)
    face.putalpha(mask)
    tile.alpha_composite(face, (offset, offset))
    return tile


def picture_tile(picture):
    """The 1024 px icon from a square-ish picture: cropped square from the middle, on a macOS rounded square with
    the same soft shadow as icon_tile's (Pillow image in, Pillow image out)."""
    from PIL import Image, ImageDraw, ImageFilter
    size, body = 1024, 824
    offset = (size - body) // 2
    picture = picture.convert("RGBA")
    side = min(picture.size)
    left, top = (picture.width - side) // 2, (picture.height - side) // 2
    face = picture.crop((left, top, left + side, top + side)).resize((body, body), Image.LANCZOS)
    mask = Image.new("L", (body, body), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, body - 1, body - 1), 185, fill=255)
    face.putalpha(mask)
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((offset, offset + 12, offset + body, offset + body + 12), 185, fill=(0, 0, 0, 90))
    tile.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)))
    tile.alpha_composite(face, (offset, offset))
    return tile


def fetch_icon(game, source: str) -> None:
    """--icon: the picture (a file or an https URL) made into build/port/icon_mac.png, the icon the app then uses."""
    from PIL import Image
    folder = game.path(os.path.join(game.build, "port"))
    os.makedirs(folder, exist_ok=True)
    local = os.path.join(folder, "icon_source")
    if re.match(r"^https?://", source):
        if not source.startswith("https://"):
            raise SystemExit("macos: --icon: https only")
        request = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0 (macos.py icon fetch)"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response, open(local, "wb") as out:
                out.write(response.read())
        except OSError as exc:
            raise SystemExit("macos: --icon: cannot fetch %s: %s" % (source, exc))
    else:
        path = os.path.expanduser(source)
        if not os.path.exists(path):
            raise SystemExit("macos: --icon: no such file: %s" % path)
        shutil.copyfile(path, local)
    try:
        with Image.open(local) as picture:
            tile = picture_tile(picture)
            print("icon: %dx%d picture%s" % (picture.width, picture.height,
                                              " (small: the icon will be soft at large sizes)" if min(picture.size) < 600 else ""))
    except OSError:
        raise SystemExit("macos: --icon: %s is not a picture Pillow can read" % source)
    tile.save(os.path.join(folder, "icon_mac.png"))


def make_icns(game, destination: str) -> bool:
    """The app's icon, by iconutil. From the player's own disc: its title logo (port/tools/make_icon.py, run here
    when build/port/icon.ico is not made yet) on a tile - or a picture of theirs, build/port/icon_mac.png, as it is;
    else the port's own port/icon.ico. False (no icon) when none is found or Pillow is missing. Nothing of this is
    ever committed: build/ is not."""
    try:
        from PIL import Image
    except ImportError:
        print("macos: no icon (pip install pillow)")
        return False
    own = game.path(os.path.join(game.build, "port", "icon_mac.png"))
    disc = game.path(os.path.join(game.build, "port", "icon.ico"))
    original = game.path(os.path.join("port", "icon.ico"))
    maker = game.path(os.path.join("port", "tools", "make_icon.py"))
    if not os.path.exists(own) and not os.path.exists(disc) and os.path.exists(maker):
        print("macos: the icon, from the disc's title screen", flush=True)
        subprocess.run([sys.executable, maker], check=False)
    def ico_picture(path):
        picture = Image.open(path)
        picture.size = max(picture.ico.sizes())
        return picture.convert("RGBA")
    if os.path.exists(own):
        biggest = Image.open(own).convert("RGBA")
    elif os.path.exists(disc):
        biggest = icon_tile(ico_picture(disc))
    elif os.path.exists(original):
        biggest = ico_picture(original)
    else:
        return False
    work = destination + ".iconset"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    for points in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            pixels = points * scale
            if pixels > biggest.width:
                continue
            name = "icon_%dx%d%s.png" % (points, points, "@2x" if scale == 2 else "")
            biggest.resize((pixels, pixels), Image.LANCZOS).save(os.path.join(work, name))
    subprocess.run(["iconutil", "-c", "icns", work, "-o", destination], check=True)
    shutil.rmtree(work, ignore_errors=True)
    return True


def app(game, lto: bool = False, icon: str = "") -> str:
    """<game>/build/macos/<Title>.app, made from the built program (building it first when it is not)."""
    if icon:
        fetch_icon(game, icon)
    config = port._game_json(game)
    title = config.get("title") or target_of(game)
    version = str(config.get("version", "0"))
    target = target_of(game)
    folder = re.sub(r"[^A-Za-z0-9 ._-]", "", title).strip() or target
    executable = os.path.join(folder_of(game), target)
    if not os.path.exists(executable):
        executable = build(game, lto)
    prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
    libraries = [os.path.join(prefix, "lib", name) for name in ("libvulkan.1.dylib", "libMoltenVK.dylib")]
    for library in libraries:
        if not os.path.exists(library):
            raise SystemExit("macos: %s not found (brew install molten-vk vulkan-loader)" % library)
    bundle = os.path.join(game.path(os.path.join(game.build, "macos")), folder + ".app")
    shutil.rmtree(bundle, ignore_errors=True)
    contents = os.path.join(bundle, "Contents")
    for sub in ("MacOS", "Frameworks", os.path.join("Resources", "vulkan", "icd.d")):
        os.makedirs(os.path.join(contents, sub))
    shutil.copy2(executable, os.path.join(contents, "MacOS", target))
    for library in libraries:    # the real file, not Homebrew's link to it
        shutil.copy2(os.path.realpath(library), os.path.join(contents, "Frameworks", os.path.basename(library)))
    with open(os.path.join(contents, "Resources", "vulkan", "icd.d", "MoltenVK_icd.json"), "w") as out:
        out.write('{"file_format_version": "1.0.0", "ICD": {"library_path": "../../../Frameworks/libMoltenVK.dylib", '
                  '"api_version": "1.4.0", "is_portability_driver": true}}\n')
    launcher = os.path.join(contents, "MacOS", target + "-launcher")
    with open(launcher, "w") as out:
        out.write(LAUNCHER.format(title=title, folder=folder, target=target))
    os.chmod(launcher, 0o755)
    make_icns(game, os.path.join(contents, "Resources", "AppIcon.icns"))
    with open(os.path.join(contents, "Info.plist"), "w") as out:
        out.write(PLIST.format(title=title, target=target, version=version, deployment=DEPLOYMENT_TARGET))
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", bundle], check=True)
    print("built " + bundle)
    return bundle


def run_env() -> dict:
    """The environment a Mac run needs for Vulkan: Homebrew's loader and MoltenVK's driver."""
    env = port._game_env()
    prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
    lib = os.path.join(prefix, "lib")
    if os.path.isdir(lib):
        env["DYLD_FALLBACK_LIBRARY_PATH"] = lib + (":" + env["DYLD_FALLBACK_LIBRARY_PATH"]
                                                   if env.get("DYLD_FALLBACK_LIBRARY_PATH") else "")
    icd = os.path.join(prefix, "etc", "vulkan", "icd.d", "MoltenVK_icd.json")
    if os.path.exists(icd):
        env.setdefault("VK_ICD_FILENAMES", icd)
    return env


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("action", nargs="?", default="all", choices=("all", "build", "run", "app"))
    parser.add_argument("--rounds", type=int, default=60,
                        help="all: build-and-run rounds to translate code found at run time")
    parser.add_argument("--lto", action="store_true", help="build: ThinLTO (a long link)")
    parser.add_argument("--icon", default="", metavar="FILE_OR_URL",
                        help="app: a picture of your own for the icon, a file or an https URL (default: the disc's "
                             "title logo)")
    # What follows "--" is the port's own options (run, all); this script's come before it.
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    args.extra = argv[split + 1:]
    game = game_config.from_args(args)
    if args.action == "build":
        build(game, args.lto)
        return 0
    if args.action == "app":
        app(game, args.lto, args.icon)
        return 0
    executable = os.path.join(folder_of(game), target_of(game))
    if args.action == "all":
        # As port.py's: build and run until a run finds no more code copied at run time.
        os.environ.update(run_env())
        extra = [a for a in args.extra if a != "--"]
        for _round in range(args.rounds):
            executable = build(game, args.lto)
            before = port._count_lines(port.discoveries_path(game))
            code = port.run(game, executable, extra)
            if port._count_lines(port.discoveries_path(game)) == before:
                return code
            print("found code copied at run time: translating it and building again\n")
        return code
    if not os.path.exists(executable):
        raise SystemExit("macos: not built yet: %s" % executable)
    os.environ.update(run_env())
    return port.run(game, executable, [a for a in args.extra if a != "--"])


if __name__ == "__main__":
    sys.exit(main())
