#!/bin/bash
# Cuts the eye patches out of a capture session for training (see extract_patches.m).
#   tools/model/extract.sh <capture-folder> [<capture-folder> ...]
# Needs OBS in /Applications (for libobs, as tools/render.sh) and clang. Nothing leaves this Mac.
set -euo pipefail
cd "$(dirname "$0")/../.."
OBS_APP="${OBS_APP:-/Applications/OBS.app}"
FW="$OBS_APP/Contents/Frameworks"
OBS_VERSION="${OBS_VERSION:-31.0.0}"
[ -d "build/obs-studio-$OBS_VERSION/libobs" ] || ./build.sh >/dev/null
clang -arch "$(uname -m)" -mmacosx-version-min=13.0 -O2 -fobjc-arc \
  -I"build/obs-studio-$OBS_VERSION/libobs" -Ibuild/simde -Ibuild/include -Isrc \
  tools/model/extract_patches.m src/gaze.c src/warp.c src/pose.c src/ml_eye.c \
  -F"$FW" -framework libobs -Wl,-rpath,"$FW" \
  -framework Foundation -framework Vision -framework CoreVideo -framework CoreGraphics -framework ImageIO \
  -framework AVFoundation -framework CoreMedia \
  -o build/extract_patches
for d in "$@"; do
  ./build/extract_patches "$d"
done
