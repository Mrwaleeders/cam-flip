#!/bin/bash
# Builds tests/test_plugin.m against the libobs inside an installed OBS.app and runs it.
set -euo pipefail
cd "$(dirname "$0")/.."
OBS_APP="${OBS_APP:-/Applications/OBS.app}"
OBS_VERSION="${OBS_VERSION:-33.0.1}"
OBS_SRC="build/obs-studio-$OBS_VERSION"
FW="$OBS_APP/Contents/Frameworks"
ls "$FW" | head -50
clang -arch "$(uname -m)" -mmacosx-version-min=13.0 -O1 -g -fobjc-arc \
  -I"$OBS_SRC/libobs" -Isrc \
  tests/test_plugin.m src/gaze.c src/warp.c \
  -F"$FW" -framework libobs -Wl,-rpath,"$FW" \
  -framework Foundation -framework Vision -framework CoreVideo -framework CoreGraphics -framework ImageIO \
  -o build/test_plugin
./build/test_plugin
