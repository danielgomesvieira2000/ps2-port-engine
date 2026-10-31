# A Linux build and test machine for PS2 ports: clang 19 (the VU code's
# preserve_none calls: 18 ignores them), CMake, Ninja, SDL's
# system headers, and Mesa's Vulkan (lavapipe: a GPU on the processor).
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        clang-19 lld-19 llvm-19 cmake ninja-build python3 python3-pil git ca-certificates pkg-config \
        libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxss-dev libxfixes-dev libxtst-dev libxinerama-dev \
        libxkbcommon-dev libwayland-dev wayland-protocols libdecor-0-dev libegl-dev libgl-dev \
        libasound2-dev libpulse-dev libudev-dev libdbus-1-dev \
        libvulkan-dev mesa-vulkan-drivers vulkan-tools xvfb \
    && rm -rf /var/lib/apt/lists/*
