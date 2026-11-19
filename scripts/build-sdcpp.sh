#!/bin/bash
# Fetches stable-diffusion.cpp at the commit WAI Gen is tested with, applies WAI Gen's patch
# (patches/sdcpp-vae-hook.patch: lets the app run the VAE in MLX) and builds the static libraries
# that Package.swift links (Metal backend).
set -euo pipefail
cd "$(dirname "$0")/.."

SDCPP_COMMIT=532d2c15c430a785a712b2d94281f04e733008a4
DIR=vendor/stable-diffusion.cpp

if [ ! -d "$DIR/.git" ]; then
    git clone https://github.com/leejet/stable-diffusion.cpp.git "$DIR"
fi
git -C "$DIR" fetch --quiet origin "$SDCPP_COMMIT" || true
git -C "$DIR" checkout --quiet "$SDCPP_COMMIT"
git -C "$DIR" submodule update --init --recursive

if git -C "$DIR" apply --check ../../patches/sdcpp-vae-hook.patch 2>/dev/null; then
    git -C "$DIR" apply ../../patches/sdcpp-vae-hook.patch
    echo "applied patches/sdcpp-vae-hook.patch"
else
    echo "patch already applied (or doesn't apply cleanly): leaving $DIR as it is"
fi

cmake -S "$DIR" -B "$DIR/build" -DSD_METAL=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cmake --build "$DIR/build" --config Release -j "$(sysctl -n hw.ncpu)"
echo "stable-diffusion.cpp built in $DIR/build"
