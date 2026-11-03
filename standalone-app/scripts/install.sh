#!/bin/bash
# Builds Cam Flip with your Apple Developer team and installs it into /Applications.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v xcodegen >/dev/null; then
  echo "Installing XcodeGen with Homebrew…"
  brew install xcodegen
fi

if [ ! -f Config/Local.xcconfig ]; then
  read -r -p "Your Apple Developer Team ID (10 characters): " TEAM
  echo "DEVELOPMENT_TEAM = $TEAM" > Config/Local.xcconfig
fi

xcodegen generate
xcodebuild -project CamFlip.xcodeproj -scheme CamFlip -configuration Release \
  -derivedDataPath build -allowProvisioningUpdates build

APP="build/Build/Products/Release/CamFlip.app"
if [ -d /Applications/CamFlip.app ]; then
  osascript -e 'quit app "CamFlip"' 2>/dev/null || true
  mv /Applications/CamFlip.app "/tmp/CamFlip.previous.$(date +%s).app"
fi
ditto "$APP" /Applications/CamFlip.app
open /Applications/CamFlip.app
echo "Installed. Look for the camera icon in your menu bar."
