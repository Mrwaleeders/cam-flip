#!/bin/bash
# One eye-model round in one go: cut patches (only if missing), train, render the A/B clip.
#   tools/model/round.sh <name> [train.py options, e.g. --steps 6000 --res-weight 0.1]
# Options from the environment:
#   SESSIONS  capture folders to train on (default: every folder in ~/Downloads/camflip-diag/captures)
#   CLIP      recording to render (default: clip 3)
#   RENDER    extra render_video options (default: clip 3's calibration, 12-44 s is the part to watch)
# Writes ~/Downloads/camflip-diag/<name>/ (model, training log) and
# ~/Downloads/camflip-diag/renders/model-ab/<name>/ (corrected.mov, compare.mov, frames.csv, render.log).
# The installed model in ~/Library/Application Support/cam-flip is left alone. Nothing leaves this Mac.
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME="${1:?usage: tools/model/round.sh <name> [train options]}"; shift
DIAG="$HOME/Downloads/camflip-diag"
OUT="$DIAG/$NAME"
AB="$DIAG/renders/model-ab/$NAME"
CLIP="${CLIP:-$HOME/Movies/2026-11-03 21-57-22.mov}"
RENDER="${RENDER:---lens 2.4-4.8 --notes 5.2-8.0 --strength 1.0 --lag 1 --scan-damp 0}"
if [ -z "${SESSIONS:-}" ]; then
  SESSIONS=$(find "$DIAG/captures" -mindepth 1 -maxdepth 1 -type d | sort | tr '\n' ' ')
fi
[ -n "$SESSIONS" ] || { echo "no capture folders in $DIAG/captures"; exit 1; }
mkdir -p "$OUT" "$AB"
t0=$(date +%s)

need=()
for s in $SESSIONS; do [ -f "$s/patches.u8" ] || need+=("$s"); done
if [ ${#need[@]} -gt 0 ]; then
  echo "== cutting eye patches for ${#need[@]} session(s)"
  tools/model/extract.sh "${need[@]}"
fi

echo "== training $NAME ($*)"
tools/model/train.sh $SESSIONS --out "$OUT" "$@" 2>&1 | tee "$OUT/train.log"
t1=$(date +%s)

echo "== rendering the A/B clip"
# shellcheck disable=SC2086
tools/render.sh "$CLIP" "$AB" $RENDER --engine model --model "$OUT/eye-model.mlpackage" 2>&1 | tee "$AB/render.log"
t2=$(date +%s)

echo "== $NAME done: training $(( (t1 - t0) / 60 )) min, render $(( (t2 - t1) / 60 )) min"
echo "   model:  $OUT/eye-model.mlpackage"
echo "   render: $AB/compare.mov (watch 12-44 s), numbers in $AB/frames.csv"
