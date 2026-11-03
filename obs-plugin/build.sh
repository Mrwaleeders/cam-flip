#!/bin/bash
# Builds cam-flip.plugin for OBS on macOS (Apple Silicon + Intel).
# Needs only the Command Line Tools (clang), not Xcode. GitHub runs this for you.
set -euo pipefail
cd "$(dirname "$0")"

OBS_VERSION="${OBS_VERSION:-31.0.0}"  # oldest OBS supported; newer OBS loads it too
VERSION="${CAMFLIP_VERSION:-$(git describe --tags --always 2>/dev/null || echo dev)}"
OBS_SRC="build/obs-studio-$OBS_VERSION"

mkdir -p build
if [ ! -d "$OBS_SRC/libobs" ]; then
  echo "Fetching OBS $OBS_VERSION headers…"
  git clone -q --depth 1 --branch "$OBS_VERSION" --filter=blob:none --sparse \
    https://github.com/obsproject/obs-studio.git "$OBS_SRC"
  git -C "$OBS_SRC" sparse-checkout set libobs
fi
# OBS's headers use SIMDe (header-only) for SSE intrinsics on Apple Silicon.
SIMDE_SRC="build/simde"
if [ ! -d "$SIMDE_SRC/simde" ]; then
  git clone -q --depth 1 --branch v0.8.2 https://github.com/simd-everywhere/simde.git "$SIMDE_SRC"
fi

# libobs/obs-config.h includes a header that OBS generates at build time; plugins only need stubs.
mkdir -p build/include
cat > build/include/obsconfig.h <<'CFG'
#pragma once
#define OBS_DATA_PATH "../../data"
#define OBS_PLUGIN_PATH "../../obs-plugins"
#define OBS_PLUGIN_DESTINATION "obs-plugins"
#define OBS_RELEASE_CANDIDATE 0
#define OBS_BETA 0
CFG

OUT="build/cam-flip.plugin"
rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS"

# Warnings are reported but not fatal (no -Werror) until a real macOS build has been checked clean.
clang -bundle -arch arm64 -arch x86_64 -mmacosx-version-min=13.0 -O2 -Wall -Wextra \
  -fobjc-arc -fvisibility=hidden \
  -I"$OBS_SRC/libobs" -I"$SIMDE_SRC" -Ibuild/include -Isrc \
  -DCAMFLIP_VERSION="\"$VERSION\"" \
  src/plugin.m src/gaze.c src/warp.c src/pose.c \
  -framework Foundation -framework Vision -framework CoreVideo -framework CoreGraphics -framework ImageIO \
  -undefined dynamic_lookup \
  -o "$OUT/Contents/MacOS/cam-flip"

cat > "$OUT/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key><string>cam-flip</string>
  <key>CFBundleIdentifier</key><string>com.mrwaleeders.cam-flip.obs</string>
  <key>CFBundleName</key><string>cam-flip</string>
  <key>CFBundlePackageType</key><string>BNDL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
</dict>
</plist>
PLIST

codesign --force --sign - "$OUT"
(cd build && rm -f cam-flip-obs-macos.zip && ditto -c -k --keepParent cam-flip.plugin cam-flip-obs-macos.zip)
echo "Built $OUT and build/cam-flip-obs-macos.zip (version $VERSION)"
