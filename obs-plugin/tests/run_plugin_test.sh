#!/bin/bash
# Builds tests/test_plugin.m against the libobs inside an installed OBS.app and runs it.
set -euo pipefail
cd "$(dirname "$0")/.."
OBS_APP="${OBS_APP:-/Applications/OBS.app}"
OBS_VERSION="${OBS_VERSION:-31.0.0}"  # oldest OBS supported; newer OBS loads it too
OBS_SRC="build/obs-studio-$OBS_VERSION"
FW="$OBS_APP/Contents/Frameworks"
clang -arch "$(uname -m)" -mmacosx-version-min=13.0 -O1 -g -Wall -fobjc-arc \
  -I"$OBS_SRC/libobs" -Ibuild/simde -Ibuild/include -Isrc \
  tests/test_plugin.m src/gaze.c src/warp.c src/pose.c \
  -F"$FW" -framework libobs -Wl,-rpath,"$FW" \
  -framework Foundation -framework Vision -framework CoreVideo -framework CoreGraphics -framework ImageIO \
  -o build/test_plugin
# The filter runs face tracking on its own thread: fail (instead of hanging CI) if anything deadlocks.
perl -e 'alarm shift; exec @ARGV' 180 ./build/test_plugin

# Real module load, exactly like OBS at startup.
clang -arch "$(uname -m)" -mmacosx-version-min=13.0 \
  -I"$OBS_SRC/libobs" -Ibuild/simde -Ibuild/include \
  tests/test_load.c -F"$FW" -framework libobs -Wl,-rpath,"$FW" -o build/test_load
./build/test_load "$PWD/build/cam-flip.plugin/Contents/MacOS/cam-flip"

# The diagnostics tools (tools/*.m) must keep compiling against the plugin internals.
# (render_video also carries the personal eye model engine.)
for tool in diag_video render_video; do
  extra=()
  [ "$tool" = render_video ] && extra=(src/ml_eye.c src/ml_model.m -framework CoreML)
  clang -arch "$(uname -m)" -mmacosx-version-min=13.0 -O1 -fobjc-arc -Wall -Werror \
    -I"$OBS_SRC/libobs" -Ibuild/simde -Ibuild/include -Isrc \
    "tools/$tool.m" src/gaze.c src/warp.c src/pose.c "${extra[@]+"${extra[@]}"}" \
    -F"$FW" -framework libobs -Wl,-rpath,"$FW" \
    -framework Foundation -framework Vision -framework CoreVideo -framework CoreGraphics -framework ImageIO \
    -framework AVFoundation -framework CoreMedia \
    -o "build/$tool"
done
echo "ok: diagnostics tools build"
