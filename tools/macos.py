#!/usr/bin/env python3
"""macOS (Apple silicon): a port built and run on the Mac itself, with Apple's
clang, CMake and Ninja - the Vulkan display through MoltenVK.

    python engine/tools/macos.py --game games/<name> build
    python engine/tools/macos.py --game games/<name> run [-- <the port's options>]

The dispatch table is generated first (port.py's, as for every build). The
build is kept in <game>/build/macos/<target>, the program beside it.

Needs: Xcode's command line tools, CMake, Ninja, Python 3, and for the display
MoltenVK and the Vulkan loader (`brew install molten-vk vulkan-loader`).
`run` puts Homebrew's lib folder on the loader's path and names MoltenVK's
driver; a release bundles both.
"""

import argparse
import os
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
    parser.add_argument("action", nargs="?", default="build", choices=("build", "run"))
    parser.add_argument("--lto", action="store_true", help="build: ThinLTO (a long link)")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="run: options for the port itself, after --")
    args = parser.parse_args(argv)
    game = game_config.from_args(args)
    if args.action == "build":
        build(game, args.lto)
        return 0
    executable = os.path.join(folder_of(game), target_of(game))
    if not os.path.exists(executable):
        raise SystemExit("macos: not built yet: %s" % executable)
    os.environ.update(run_env())
    return port.run(game, executable, [a for a in args.extra if a != "--"])


if __name__ == "__main__":
    sys.exit(main())
