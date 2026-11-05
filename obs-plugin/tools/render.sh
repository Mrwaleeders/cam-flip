#!/bin/bash
# Renders a recording through Cam Flip (tracking + gaze model + warp, as live) for review.
#   tools/render.sh <recording> <out-dir> [render_video options]
# With no calibration options, uses the calibration saved in OBS (the first Cam Flip filter found
# in your scene collections). Needs OBS in /Applications and clang. Nothing leaves this Mac.
set -euo pipefail
cd "$(dirname "$0")/.."
IN="$1"; OUT="$2"; shift 2
OBS_APP="${OBS_APP:-/Applications/OBS.app}"
FW="$OBS_APP/Contents/Frameworks"
OBS_VERSION="${OBS_VERSION:-31.0.0}"
[ -d "build/obs-studio-$OBS_VERSION/libobs" ] || ./build.sh >/dev/null
mkdir -p "$OUT"

case "$IN" in
  *.mkv)
    ffmpeg -loglevel error -y -i "$IN" -c copy "$OUT/remux.mp4"; IN="$OUT/remux.mp4";;
esac

clang -arch "$(uname -m)" -mmacosx-version-min=13.0 -O2 -fobjc-arc \
  -I"build/obs-studio-$OBS_VERSION/libobs" -Ibuild/simde -Ibuild/include -Isrc \
  tools/render_video.m src/gaze.c src/warp.c src/pose.c src/ml_eye.c src/ml_model.m \
  -F"$FW" -framework libobs -Wl,-rpath,"$FW" \
  -framework Foundation -framework Vision -framework CoreVideo -framework CoreGraphics -framework ImageIO \
  -framework AVFoundation -framework CoreMedia -framework CoreML \
  -o build/render_video

CAL=()
if ! printf '%s\n' "$@" | grep -q -- '--baseline\|--lens'; then
  CAL=($(python3 - <<'PY'
import glob, json, os
for p in glob.glob(os.path.expanduser("~/Library/Application Support/obs-studio/basic/scenes/*.json")):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    for s in d.get("sources", []):
        for f in s.get("filters", []) or []:
            if f.get("id") == "camflip_filter":
                st = f.get("settings", {})
                out = ["--baseline", st.get("baseline", 0), "--baseline-open", st.get("baseline_open", 0),
                       "--strength", st.get("strength", 100) / 100.0,
                       "--scan-damp", st.get("scan_damp", 0) / 100.0]
                if st.get("has_notes"):
                    out += ["--notes-offset", st.get("notes_offset", 0), "--notes-side", st.get("notes_side", 0),
                            "--notes-open", st.get("notes_open", 0)]
                if st.get("head_ref"):  # the head looking into the camera (step 1); else a typical face
                    out += ["--head-side", st.get("head_side", 0), "--head-vert", st.get("head_vert", 0),
                            "--head-eyes", st.get("head_eyes", 0), "--head-tilt", st.get("head_tilt", 0)]
                if st.get("baseline_upper", 0) > 0 and st.get("baseline_lower", 0) > 0:  # lids at the lens (step 1)
                    out += ["--baseline-upper", st["baseline_upper"], "--baseline-lower", st["baseline_lower"]]
                if st.get("has_notes") and (int(st.get("vis_cal", 0)) & 3) == 3:  # Vision's point at both steps
                    out += ["--vis-baseline", st.get("vis_baseline", 0), "--vis-notes", st.get("vis_notes", 0)]
                print(" ".join(str(x) for x in out))
                raise SystemExit
PY
))
  echo "Calibration from OBS: ${CAL[*]:-none found}"
fi
./build/render_video "$IN" "$OUT" ${CAL[@]+"${CAL[@]}"} "$@"
