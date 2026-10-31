#!/usr/bin/env python3
"""Linux: a port built and tested for Linux (x86-64) in a container, from
Windows or Linux alike - Docker with the image in tools/linux/Dockerfile
(clang 19, CMake, Ninja, SDL's system headers, Mesa's Vulkan: lavapipe draws on
the processor, so the GPU renderer runs its tests without a GPU).

    python engine/tools/linux.py --game games/<name> build          # <game>/build/linux/<target>
    python engine/tools/linux.py --game games/<name> test [--runs car_select,race] [--gpu]
    python engine/tools/linux.py --game games/<name> shell          # a shell in the container

The repository (the folder holding engine/ and games/) is mounted at /src;
the build is kept in a Docker volume (ps2port_linux_<target>: building on the
mounted Windows folder is slow) and the program copied out. The dispatch
table is generated first, here (port.py's), as for every build. `test` runs
the host's tests, the renderer's (on lavapipe), then regress.py's runs with
the software GS (--gpu: with lavapipe's Vulkan - its pictures are its own,
as an Intel GPU's are, so a count, not a pass, against NVIDIA's recordings).
"""

import argparse
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from ps2decomp import game as game_config  # noqa: E402
import port  # noqa: E402

IMAGE = "ps2port-linux"


def run(cmd, **kw):
    print("+ " + " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""), flush=True)
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        raise SystemExit("linux: %s failed (%d)" % (cmd[0], result.returncode))
    return result


def root_of(game) -> str:
    """The repository: the nearest folder above the game holding engine/."""
    here = os.path.abspath(game.directory)
    while True:
        if os.path.isdir(os.path.join(here, "engine", "host")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise SystemExit("linux: no engine/ above %s" % game.directory)
        here = parent


def inside(root: str, path: str) -> str:
    """`path` as the container sees it (under /src)."""
    rel = os.path.relpath(os.path.abspath(path), root)
    if rel.startswith(".."):
        raise SystemExit("linux: %s is not inside %s" % (path, root))
    return "/src/" + rel.replace(os.sep, "/")


def _answers() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def docker_up(wait_s: int = 30):
    """Docker's engine answering. Docker Desktop's engine stops answering now
    and then (idle, its resource saver; or the machine short of memory) and
    does not come back by itself: after `wait_s`, `docker desktop restart`
    once (Docker Desktop's own command), then told how to start it."""
    for attempt in range(max(1, wait_s // 5)):
        if _answers():
            return
        if attempt == 0:
            print("linux: waiting for Docker's engine...", flush=True)
        time.sleep(5)
    print("linux: restarting Docker Desktop (docker desktop restart)", flush=True)
    subprocess.run(["docker", "desktop", "restart"], capture_output=True, timeout=300)
    for _ in range(24):
        if _answers():
            return
        time.sleep(5)
    raise SystemExit("linux: Docker's engine does not answer: start Docker Desktop (or `docker desktop restart`)")


def image():
    """The image, made (or brought up to date: Docker's cache) from tools/linux."""
    docker_up()
    run(["docker", "build", "-q", "-t", IMAGE, os.path.join(HERE, "linux")], stdout=subprocess.DEVNULL)


def container(game, script: str, interactive: bool = False):
    root = root_of(game)
    target = port_target(game)
    cmd = ["docker", "run", "--rm", "-v", root + ":/src", "-v", "ps2port_linux_%s:/build" % target, "-w", "/src"]
    if interactive:
        cmd += ["-it", IMAGE, "bash"]
    else:
        cmd += [IMAGE, "bash", "-c", script]
    env = dict(os.environ)
    env["MSYS_NO_PATHCONV"] = "1"       # (Git Bash's path rewriting off: these are the container's)
    return run(cmd, env=env)


def port_target(game) -> str:
    return os.path.basename(os.path.normpath(game.directory))


def build(game, jobs: int):
    generated = port.generate(game)
    root = root_of(game)
    target = port_target(game)
    out = game.path(os.path.join(game.build, "linux"))
    os.makedirs(out, exist_ok=True)
    # ThinLTO where the game asks for it (game.json's "lto", as port.py's;
    # PS2_LTO=on/off overrides): its link compiles the files again, a VU
    # file's 2 GB and more - fewer at once than the compiler's jobs.
    wanted = os.environ.get("PS2_LTO", "").lower()
    lto = wanted == "on" or (wanted != "off" and bool(port._game_json(game).get("lto")))
    script = " && ".join([
        "cmake -S %s -B /build/%s -G Ninja -DCMAKE_C_COMPILER=clang-19 -DCMAKE_CXX_COMPILER=clang++-19 "
        "-DCMAKE_BUILD_TYPE=Release -DPS2_GENERATED_DIR=%s -DPS2HOST_LTO=%s -DPS2HOST_LTO_JOBS=%d "
        "'-DCMAKE_EXE_LINKER_FLAGS=-static-libstdc++ -static-libgcc' > /build/configure.log"
        % (inside(root, game.path("port")), target, inside(root, generated), "ON" if lto else "OFF",
           max(1, jobs // 2)),
        "cmake --build /build/%s -j %d" % (target, jobs),
        "cp /build/%s/%s %s/" % (target, target, inside(root, out)),
    ])
    container(game, script)
    print("built %s%s" % (os.path.join(out, target), " (ThinLTO)" if lto else ""))


def test(game, runs: str, gpu: bool, renderer: str = ""):
    root = root_of(game)
    target = port_target(game)
    steps = ["/build/%s/ps2host/ps2host_tests | tail -1" % target,
             "/build/%s/ps2gpu/ps2gpu_tests | tail -1" % target]
    # A run with --window gets an X server of its own (Xvfb); PS2HOST_RENDERER
    # picks its display (sdl: SDL's 2D renderer, as on a PC without Vulkan).
    chosen = "PS2HOST_RENDERER=%s " % shlex.quote(renderer) if renderer else ""
    for r in [r for r in runs.split(",") if r]:
        steps.append("%sxvfb-run -a python3 engine/tools/regress.py --game %s %s --exe /build/%s/%s%s | grep -E '^  [^ ]|recorded'"
                     % (chosen, shlex.quote(inside(root, game.directory)), r, target, target,
                        " --gpu" if gpu else ""))
    container(game, "set -o pipefail; " + " && ".join(steps))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("action", nargs="?", default="build", choices=("build", "test", "shell"))
    parser.add_argument("--runs", default="", help="test: regress.py runs, by commas")
    parser.add_argument("--gpu", action="store_true", help="test: the runs with the GPU renderer (lavapipe)")
    parser.add_argument("--renderer", default="", help="test: PS2HOST_RENDERER for the runs (sdl, vulkan)")
    parser.add_argument("--jobs", type=int, default=6, help="build: compile jobs (the container's memory: ~1 GB each)")
    args = parser.parse_args(argv)
    game = game_config.from_args(args)
    image()
    if args.action == "build":
        build(game, args.jobs)
    elif args.action == "test":
        test(game, args.runs, args.gpu, args.renderer)
    else:
        container(game, "", interactive=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
