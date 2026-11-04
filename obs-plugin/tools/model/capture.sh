#!/bin/bash
# Records the personal eye model's training session (see capture_pairs.m).
#   tools/model/capture.sh --distance 60 [--camera "MacBook"] [--out DIR]
# Do it once at your usual call distance and once at your desk distance, ideally in your usual light.
# CAMFLIP_QUICK=1 runs a short version (to try it out). Everything stays on this Mac.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p build
clang -arch "$(uname -m)" -mmacosx-version-min=13.0 -O2 -Wall -fobjc-arc tools/model/capture_pairs.m \
  -framework AppKit -framework AVFoundation -framework CoreMedia -framework QuartzCore \
  -o build/capture_pairs
./build/capture_pairs "$@"
