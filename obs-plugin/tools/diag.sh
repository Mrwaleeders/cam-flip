#!/bin/bash
# Runs Cam Flip's face tracking over a recording and writes the per-frame numbers to a CSV.
#   tools/diag.sh <recording.mp4|.mov|.mkv> [out.csv]
# Needs OBS in /Applications and the Command Line Tools (clang). Nothing leaves this Mac.
set -euo pipefail
cd "$(dirname "$0")/.."
IN="$1"
OUT="${2:-$HOME/Downloads/camflip-diag/landmarks.csv}"
mkdir -p "$(dirname "$OUT")"
OBS_APP="${OBS_APP:-/Applications/OBS.app}"
FW="$OBS_APP/Contents/Frameworks"
OBS_VERSION="${OBS_VERSION:-31.0.0}"

[ -d "build/obs-studio-$OBS_VERSION/libobs" ] || ./build.sh >/dev/null

case "$IN" in
  *.mkv)
    if command -v ffmpeg >/dev/null; then
      ffmpeg -loglevel error -y -i "$IN" -c copy "${OUT%.csv}-remux.mp4"
      IN="${OUT%.csv}-remux.mp4"
    else
      echo "This is an .mkv: remux it in OBS (File > Remux Recordings) and pass the .mp4"; exit 1
    fi;;
esac

clang -arch "$(uname -m)" -mmacosx-version-min=13.0 -O2 -fobjc-arc \
  -I"build/obs-studio-$OBS_VERSION/libobs" -Ibuild/simde -Ibuild/include -Isrc \
  tools/diag_video.m src/gaze.c src/warp.c src/pose.c \
  -F"$FW" -framework libobs -Wl,-rpath,"$FW" \
  -framework Foundation -framework Vision -framework CoreVideo -framework CoreGraphics -framework ImageIO \
  -framework AVFoundation -framework CoreMedia \
  -o build/diag_video
./build/diag_video "$IN" "$OUT"
echo "Wrote $OUT"
