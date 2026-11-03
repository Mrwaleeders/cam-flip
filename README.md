# Cam Flip

A macOS menu bar app that adds a virtual webcam, **Cam Flip Camera**, for Teams, Zoom, Google Meet and any other video app. It does two things:

1. **Mirrors your video for everyone.** The people in your call see the flipped image, not just you.
2. **Keeps eye contact** when you read notes placed slightly below the camera, by moving your pupils back up on the GPU in real time.

Both are separate switches in the menu bar, and eye contact has a strength slider.

## Why the other person never sees your mirror today

Teams, Zoom and Meet mirror only your **self-view**. They always send the original, unflipped image to everyone else, so writing behind you reads the right way round for them. No setting in those apps changes what others receive.

Cam Flip flips the image *before* the call app gets it, so the flipped image is what gets sent.

> **Tip:** once you use Cam Flip Camera, turn **off** "Mirror my video" in Zoom/Teams. Otherwise the app flips your self-view back, and your preview won't match what others see. The Cam Flip preview window always shows exactly what others see.

> Text behind you (a poster, a shirt logo) will look backwards to others while mirroring is on. That's what mirroring means.

## Image quality

* The physical camera is captured at the size the call app asks for (1080p by default, up to 4K), at 30 fps.
* Frames move between the app and the virtual camera as IOSurfaces (shared GPU memory). They are never copied, re-encoded or compressed.
* Mirroring is an exact pixel flip with no resampling, so it costs zero sharpness.
* Eye contact only touches pixels inside the two eye regions. Everything else is bit-for-bit unchanged (the CI self-test checks this).
* With both switches off, the camera frames pass straight through untouched.

The call services still compress video on their side, as they always do. Cam Flip doesn't add to that.

## How eye contact works

Apple Vision finds your face, eyes and pupils on every frame. Cam Flip compares where your pupils sit with where they sat when you calibrated (looking into the lens), and a small Core Image Metal kernel shifts the eye region up by that amount, with a smooth falloff so the edges blend. Head tilt is handled. It's tuned for **small** downward glances: when you look far down (at the keyboard), the correction fades out on purpose, because a big correction looks fake. Blinks are detected and left alone.

Click **Calibrate eye contact** once while looking straight into the camera. Re-calibrate if you change your seat or screen position a lot.

## Install on your Mac

You need:
* Xcode (latest, from the App Store) and [Homebrew](https://brew.sh)
* A **paid Apple Developer Program** membership. macOS only lets signed system extensions install, and the free personal team can't sign them. Your Team ID is under *Membership details* at [developer.apple.com/account](https://developer.apple.com/account).

Steps:

1. Open Xcode once and sign in with your Apple ID (Xcode › Settings › Accounts).
2. In Terminal:
   ```bash
   git clone https://github.com/Mrwaleeders/cam-flip.git
   cd cam-flip
   ./scripts/install.sh
   ```
   It asks for your Team ID the first time, builds, and copies the app to `/Applications` (camera extensions only install from there).
3. Cam Flip asks to install its camera. Open **System Settings › General › Login Items & Extensions › Camera Extensions** and switch **Cam Flip** on. Allow camera access when asked.
4. In Teams, Zoom or Meet, pick **Cam Flip Camera** as your camera. (Quit and reopen the call app if it doesn't show up yet.)
5. Click the camera icon in the menu bar to switch mirroring and eye contact, set the strength, and calibrate.

To make it start at login: System Settings › General › Login Items › add CamFlip.

### Building in Xcode instead
```bash
brew install xcodegen
echo "DEVELOPMENT_TEAM = YOURTEAMID" > Config/Local.xcconfig
xcodegen generate
open CamFlip.xcodeproj
```
Build the **CamFlip** scheme, then copy the built `CamFlip.app` into `/Applications` and run it from there.

## Troubleshooting

* **Call app shows "Open the Cam Flip app to start your camera":** the Cam Flip app isn't running or doesn't have camera access.
* **Cam Flip Camera missing in the call app:** check it's switched on under Camera Extensions, then restart the call app.
* **Uninstall:** menu › *Remove camera*, then delete `/Applications/CamFlip.app`.
* Logs: `log stream --predicate 'subsystem BEGINSWITH "com.mrwaleeders.CamFlip"'`

## Project layout

| Path | What |
| --- | --- |
| `CamFlip/` | Menu bar app: capture, Vision gaze tracking, GPU processing, menu UI |
| `CamFlip/GazeWarp.ci.metal` | Eye-contact warp kernel (Core Image, Metal) |
| `CamFlipExtension/` | CoreMediaIO camera extension (the virtual camera) |
| `Shared/` | Constants used by both |
| `project.yml` | XcodeGen project definition |
| `Tests/KernelSelfTest.swift` | Headless GPU checks run in CI |
