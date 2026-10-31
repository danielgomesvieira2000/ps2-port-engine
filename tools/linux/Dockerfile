# A Linux build and test machine for PS2 ports. Ubuntu 22.04 for the C
# library: a program built here needs glibc 2.35 (Ubuntu 22.04, Debian 12's
# 2.36, SteamOS, Fedora 36 and later) - the C++ runtime is linked into it
# (linux.py). clang 19 from LLVM's own packages (the VU code's preserve_none
# calls: 18 ignores them), GCC 12's libstdc++ for C++20; CMake, Ninja, SDL's
# system headers (X11, Wayland, ALSA, PulseAudio), and Mesa's Vulkan
# (lavapipe: a GPU on the processor, for the renderer's tests).
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget gnupg ca-certificates lsb-release software-properties-common \
    && wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh && bash /tmp/llvm.sh 19 \
    && apt-get install -y --no-install-recommends \
        g++-12 libstdc++-12-dev cmake ninja-build python3 python3-pip git pkg-config \
        libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxss-dev libxfixes-dev libxtst-dev libxinerama-dev \
        libxkbcommon-dev libwayland-dev wayland-protocols libdecor-0-dev libegl-dev libgl-dev \
        libasound2-dev libpulse-dev libudev-dev libdbus-1-dev \
        libvulkan-dev mesa-vulkan-drivers vulkan-tools xvfb binutils \
    && pip3 install --no-cache-dir pillow \
    && rm -rf /var/lib/apt/lists/* /tmp/llvm.sh
