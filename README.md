# Cam Flip

Mirror your camera **for everyone in your call**, and keep **eye contact** while you read notes placed just below the camera. It works in Teams, Zoom, Google Meet and any app that can use a webcam.

Cam Flip is a free plugin for [OBS Studio](https://obsproject.com). OBS's built-in virtual camera, which OBS has already signed with Apple, carries the video into your call app. You don't need an Apple Developer account or Xcode.

## Why other people don't see your mirror today

Teams, Zoom and Meet mirror only your **own preview**. They always send the unflipped image to everyone else, and no setting changes that. Cam Flip flips the video before the call app receives it, so the flipped video is what gets sent.

> Once you use Cam Flip, turn **off** "Mirror my video" in Zoom and Teams. Otherwise they flip your preview back, and it won't match what others see.

## What it does

* **Mirror:** an exact pixel flip with no resampling, so there's no softening.
* **Eye contact:** Apple's Vision framework finds your eyes and pupils on every frame. When you glance slightly down at notes, Cam Flip moves the pupils back up toward the lens.
* **Looks natural when you move:**
  * The correction tracks your eyes frame by frame and follows your head as it moves, so it never lags or slips.
  * When you turn your head, look far down (for example at the keyboard) or to the side, or blink, the correction fades out smoothly instead of forcing a stare. When you look back, it fades in again over about a third of a second. It never switches suddenly.
  * Automated tests check these behaviours on every build. See `obs-plugin/tests/test_gaze.c`.
* **Image quality:**
  * Cam Flip works on the camera's frames in their native format (NV12, UYVY, BGRA and others). It never converts, scales or compresses them.
  * Pixels outside your eyes are not touched at all. The tests check this byte for byte.
* **Delay:** Cam Flip itself adds no frame of delay. It processes each frame as it arrives, and eye tracking takes a few milliseconds. OBS's virtual camera adds roughly 15–20 ms with the settings below.

## Setup (one time, about 10 minutes)

### 1. Install OBS
Download OBS Studio for macOS (Apple Silicon) from [obsproject.com](https://obsproject.com) and open it once. Allow camera access when asked.

### 2. Install the Cam Flip plugin
Download **cam-flip-obs-macos.zip** from this repo's [Releases › latest](https://github.com/Mrwaleeders/cam-flip/releases/tag/latest), then paste this into Terminal:

```bash
cd ~/Downloads && { [ -f cam-flip-obs-macos.zip ] && unzip -oq cam-flip-obs-macos.zip; true; } \
 && P="$HOME/Library/Application Support/obs-studio/plugins" && mkdir -p "$P" \
 && rm -rf "$P/cam-flip.plugin" && mv cam-flip.plugin "$P/" \
 && xattr -dr com.apple.quarantine "$P/cam-flip.plugin" && echo "Cam Flip installed"
```

The last step tells macOS that you trust the plugin. The plugin isn't signed by Apple, but it is built by this repo's own GitHub Actions, and you can read the source here.

Quit and reopen OBS.

### 3. Set up OBS for low delay and full quality
In **OBS › Settings › Video**:
* **Base (Canvas) Resolution** and **Output (Scaled) Resolution**: both `1920x1080`, or your camera's resolution. Matching sizes mean no scaling.
* **Common FPS Values**: `60`. Your camera still runs at 30. OBS just outputs on a faster beat, which halves the waiting time.

### 4. Add your camera with the filter
1. In **Sources**, click **+**, choose **Video Capture Device**, and select your camera (for example "MacBook Pro Camera").
   * In its properties, turn off **Use Buffering**. That removes another frame of delay.
   * Right-click the source, choose **Transform › Fit to screen**.
2. Right-click the camera source, choose **Filters**, click **+** under *Audio/Video Filters*, and pick **Cam Flip (mirror + eye contact)**.
3. In the filter's settings:
   * **Mirror video**: on.
   * **Eye contact**: on. Strength starts at 70%.
   * Look straight into the camera and click **Calibrate**. It takes about a second.

### 5. Use it in calls
* Click **Start Virtual Camera** in OBS. It's in the Controls panel at the bottom right.
* In Teams, Zoom or Meet, choose **OBS Virtual Camera** as your camera. If it isn't listed, quit and reopen the call app.
* Turn off "Mirror my video" in Zoom and Teams, as explained above.

To have it ready every time, add OBS to **System Settings › General › Login Items**. OBS remembers your scene and filter. Click Start Virtual Camera before a call.

## Settings explained

| Setting | What it does |
| --- | --- |
| Mirror video | Flips the video that is actually sent, so everyone sees it mirrored. |
| Eye contact | Turns gaze correction on or off. |
| Eye contact strength | 100% fully cancels a small glance down. Lower values keep a little of the natural movement. |
| Correct glances up to | How far down you can look and still be corrected. Beyond this, your real eyes show again, gradually. |
| Calibrate | Learns where your pupils sit when you look into the lens. Redo it if you move your screen or chair. |

## Troubleshooting

* **The filter isn't in the list:** reopen OBS after installing, and check the plugin is in `~/Library/Application Support/obs-studio/plugins/cam-flip.plugin`. OBS's log (Help › Log Files) shows `[cam-flip] loaded`.
* **Eye contact does nothing:** make sure your face is well lit and fills a fair part of the frame, then calibrate. The log shows `[cam-flip]` messages if your camera's pixel format isn't supported. If so, set the camera's **Video Format** to NV12 or UYVY in its properties.
* **Others see you un-mirrored:** check the filter is on the camera source, and the call app uses **OBS Virtual Camera**.

## Repo layout

| Path | What |
| --- | --- |
| `obs-plugin/src/plugin.m` | The OBS filter, plus Apple Vision face tracking |
| `obs-plugin/src/gaze.c` | Eye-contact logic: smoothing, fading, calibration |
| `obs-plugin/src/warp.c` | Mirror and eye warp, working directly on the camera's own pixel formats |
| `obs-plugin/tests/` | Behaviour, quality and OBS integration tests, which run on every push |
| `obs-plugin/build.sh` | Builds the plugin. It only needs clang, and GitHub runs it for you |
| `standalone-app/` | An earlier design: a standalone virtual camera app. It needs a paid Apple Developer account, so it's parked |
