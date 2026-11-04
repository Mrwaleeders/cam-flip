# Cam Flip

Mirror your camera **for everyone in your call**, and keep **eye contact** while you read notes placed just below the camera. It works in Teams, Zoom, Google Meet and any app that can use a webcam.

Cam Flip is a free plugin for [OBS Studio](https://obsproject.com). OBS's built-in virtual camera, which OBS has already signed with Apple, carries the video into your call app. You don't need an Apple Developer account or Xcode.

## Why other people don't see your mirror today

Teams, Zoom and Meet mirror only your **own preview**. They always send the unflipped image to everyone else, and no setting changes that. Cam Flip flips the video before the call app receives it, so the flipped video is what gets sent.

> Once you use Cam Flip, turn **off** "Mirror my video" in Zoom and Teams. Otherwise they flip your preview back, and it won't match what others see.

## What it does

* **Mirror:** an exact pixel flip with no resampling, so there's no softening.
* **Eye contact:** Apple's Vision framework finds your eyes and pupils. When you glance down at notes placed just under the camera, Cam Flip moves your irises back up toward the lens, so the other person shouldn't be able to tell you're reading.
  * Most people tilt their head down a little while reading as well, and their eyes then move only a pixel or two. Cam Flip measures the tilt (against step 1) and lifts your irises by that much more, so your eyes meet the lens again. A head tilted down while your eyes stay on the lens gets no correction, and tilts past about 18° are left alone. Half of a sideways head turn is made up for in the same way.
  * Reading a line moves your eyes sideways, then back for the next line. **Hide reading along a line** (off by default) takes out that much of the scanning (95% leaves about a third visible). Tracking runs a frame behind the video, so your eyes then twitch for one frame at each jump back to the next line. A look held to one side is never shifted.
  * Reading also lowers your upper lid and raises your lower lid a little, which looks like looking down even when the irises are lifted. While correcting, Cam Flip opens your lids back to how they were when you looked into the camera in step 1.
* **Looks natural when you move:**
  * Eye tracking runs in the background, a little behind the video, and Cam Flip predicts where your eyes are in each frame it corrects. So the correction stays on your eyes during normal head movement. Very fast, sudden movements can briefly show it a few pixels off before tracking catches up.
  * When you turn your head, look far down (for example at the keyboard) or to the side, the correction fades out smoothly instead of forcing a stare. When you glance back, it takes over again during the eye's own quick jump, where nobody can see it. When you turn or drift back slowly, it fades in over about half a second. It never switches suddenly.
  * Head turns and tilts are measured on every frame from your eye corners and nose, against how your head sits when you look into the camera (step 1). Turning or tilting your head more than about 15° either way starts the fade, and by about 25 to 30° the correction is gone. The exact angles vary a little from face to face.
  * The corrected eyes stay still while you read. Tracking noise, and stray readings such as a lid catching the pupil or one eye misread for a frame, don't show as shaking. A real glance between the camera and your notes still lands within a frame or two, at 15 fps as well as 30 fps.
  * When you blink, the correction is held steady through the blink, so your eyes don't flicker. If you glance during a blink, the correction follows as your lids open. If your eyes stay closed for longer, it fades out, and comes back about as quickly as your lids open.
  * Automated tests check these behaviours on every build. See `obs-plugin/tests/test_gaze.c`.
* **Image quality:**
  * Cam Flip works on the camera's frames in their native format (NV12, UYVY, BGRA and others). It never converts, scales or compresses them. (For cameras 2560 pixels wide or more, face tracking looks at a reduced-size copy to stay fast. The pupils are still located on the full-size frame, and the video itself is never scaled.)
  * Pixels outside your eyes are not touched at all. The tests check this byte for byte.
* **Delay:** Cam Flip adds no frame of delay. Eye tracking runs in the background on its own thread, always on the newest frame. Each frame is corrected as it passes through, using where your eyes are predicted to be at that moment, which takes well under a millisecond. So OBS never waits for face tracking. OBS's virtual camera adds roughly 15–20 ms with the settings below.
  * OBS's log (Help › Log Files) shows the real numbers every 10 seconds, for example `[cam-flip] tracking 6.1 ms avg (30 fps analysed, 10 full face detections), warp 0.20 ms`.

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
   * **Eye contact**: on. Strength starts at 100%.
4. Calibrate, in two steps. Sit as you will in calls, with your notes where you'll read them (just under the camera works best).
   1. Click **1. Click, then look into the camera**. Then look straight into the camera lens until the status says it's done (a second or two).
   2. Click **2. Click, then look at your notes**. Then look at the **middle of the line you usually read** in your notes, with your head still, until the status says it's done.

   The status text in the filter's settings shows when each step is done. With both steps done, Cam Flip knows exactly where your notes are, and how open your eyes are while you read them, so it fully cancels that glance, tells reading apart from blinking, and fades out for anything further away (like your keyboard). It also learns how much of a glance Vision's own pupil points show for your eyes, which keeps the correction steadiest, and, in step 1, how open your lids are and how your head sits when you look into the lens, which the head-tilt and lid corrections work from. If you calibrated with an older version, do both steps again once to get this.
   If your eyes or lids moved while you calibrated, or your notes read barely below the camera, the status shows a warning after the step. The calibration is still used, but redo the step it names if the effect looks off.
   Redo both steps if you move your screen, notes or chair. If you redo only step 1 and your notes no longer sit below the new camera position, the status says **Notes calibration no longer matches: please redo step 2**. Until you do, Cam Flip uses the "Correct glances up to" slider instead.

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
| Eye contact strength | Starts at 100%, which fully cancels the glance at your notes as it is measured. If you still look like you're reading, go higher (up to 200%) to lift your eyes further. The lids are never opened past how they were in step 1, and the eyes never move further than the warp allows without visible distortion. Lower values keep a little of the natural movement. |
| Correct glances up to | How far down you can look and still be corrected. Beyond this, your real eyes show again, gradually. Not used once step 2 is calibrated, because Cam Flip then knows where your notes are. |
| 1. Click, then look into the camera | Learns where your pupils sit, how open your eyes and each lid are, and how your head is turned and tilted, when you look into the lens. |
| 2. Click, then look at your notes | Learns exactly how far down (and sideways) your notes are, and how open your eyes are while you read, so that glance is fully cancelled. Your notes must be below the camera. If they aren't, or you looked elsewhere, the status says **Notes calibration failed** and nothing is changed. |
| Forget notes calibration | Goes back to using the "Correct glances up to" slider. |
| Status | Shows calibration progress and problems, whether your face is found, and how long tracking takes. |
| Show tracked eyes (for tuning) | While a glance is being corrected, draws thin outlines exactly where the correction is applied in that frame: a white ellipse for the area that may change, and black for the eye opening your iris is moved inside. Nothing is drawn while you look into the camera. Turn it off before a call. |
| Defaults | Resets everything, including calibration. |

**Hotkey:** to switch eye contact on and off during a call, set a key in **OBS › Settings › Hotkeys**. Look for **Cam Flip: toggle eye contact** under your camera's name. If a camera has more than one Cam Flip filter, each one has its own hotkey, named after the filter, for example **Cam Flip: toggle eye contact (Cam Flip (mirror + eye contact) 2)** for a second filter OBS named “Cam Flip (mirror + eye contact) 2”. Renaming a filter keeps its key.

## Troubleshooting

* **The filter isn't in the list:** reopen OBS after installing, and check the plugin is in `~/Library/Application Support/obs-studio/plugins/cam-flip.plugin`. OBS's log (Help › Log Files) shows `[cam-flip] loaded`.
* **Eye contact does nothing:** check the status text in the filter's settings. If it says no face is found, light your face so it fills a fair part of the frame, then calibrate. Turn on **Show tracked eyes** to see what Cam Flip sees. The log shows `[cam-flip]` messages if your camera's pixel format isn't supported. If so, set the camera's **Video Format** to NV12 or UYVY in its properties.
* **You still look like you're reading:** redo step 1 once if it was done by an older version: until then your lids (and, with older calibrations still, your head tilt) aren't corrected. Then raise **Eye contact strength** above 100% until your eyes meet the lens.
* **Status says "Please calibrate again":** this version measures your eyes differently, so calibrations from older versions were reset. Do both calibration steps again.
* **Status says "Calibration didn't finish: no camera frames":** the camera sent no video for 12 seconds after you clicked. Check that the camera is on and not used by another app, and that you can see it in OBS, then click the step again.
* **Status says "Calibration gave up":** Cam Flip didn't see a steady, front-facing face for about 10 seconds. Light your face, face the camera, keep your head still and try again.
* **Status says "Notes calibration failed":** step 2 measured a gaze that wasn't below the camera. Look at the middle of your notes under the camera (not at the screen or the camera) and try again. If it keeps failing, redo step 1 first.
* **Status says "Notes calibration no longer matches: please redo step 2":** a newer step 1 put the camera level with or below your saved notes position. Do step 2 again.
* **Status says "Redo step 1 so head turns and tilts are measured for your face":** step 1 was done by an older version (or your nose wasn't in view). Until you redo it, Cam Flip assumes a typical face: turning your head still fades the correction, a little later than it should, but tilting your head up or down doesn't.
* **Status says "your eyes moved while calibrating":** the step was taken, but your gaze or lids moved during it, so it may be a little off. If the effect looks wrong, redo that step, keeping your gaze still on the lens (step 1) or on the middle of your notes (step 2).
* **Status says "they read barely below the camera":** step 2 measured only a very small glance. If your notes are further down than that, step 1 was probably taken while looking slightly below the lens (at the screen). Redo step 1 looking straight into the lens, then step 2.
* **Others see you un-mirrored:** check the filter is on the camera source, and the call app uses **OBS Virtual Camera**.

## Personal eye model (experimental, not in the plugin yet)

A small learned model, trained only on **your own face**, that may replace the hand-built eye warp. It moves the iris and both lids together, so the lowered lower lid and the side-to-side scan while reading no longer show. For now it runs in the render tool for side-by-side comparisons, not in OBS. Everything it uses or makes is your face, so it stays on your Mac: the repo only holds the code, and `.gitignore` blocks captures and models.

1. **Record** (about 4 minutes, head still, eyes follow a dot): `obs-plugin/tools/model/capture.sh --distance 60`. Use your real distance in cm. Do one session at your call distance and one at your desk distance.
2. **Cut out the eyes:** `obs-plugin/tools/model/extract.sh ~/Downloads/camflip-diag/captures/<session>` (for each session).
3. **Train** (about 30 to 60 minutes on Apple Silicon): `obs-plugin/tools/model/train.sh <session> <session>`. This writes `~/Library/Application Support/cam-flip/eye-model.mlpackage`, plus a preview of held-out frames showing your eye, the model's correction, and the real look into the lens.
4. **Compare:** `obs-plugin/tools/render.sh <clip.mov> <out-dir> --engine model`, against the same command without `--engine model`. The tool prints the model's time per frame. `--ml-compute cpu|gpu|ane` picks where it runs.

The model covers anything within about 20 cm around your notes: the whole screen width and down to the bottom of the screen. It's trained for 0.5 to 1.5 m, and anything further out fades out like the classic warp. The research and plan are in the [design doc](https://claude.ai/artifact/CTFrxG6jB1H3EasJs5kSW1).

## Repo layout

| Path | What |
| --- | --- |
| `obs-plugin/src/plugin.m` | The OBS filter, plus Apple Vision face tracking |
| `obs-plugin/src/gaze.c` | Eye-contact logic: smoothing, fading, calibration |
| `obs-plugin/src/pose.c` | Head turn and tilt from the face landmarks, for the fade |
| `obs-plugin/src/warp.c` | Mirror and eye warp, working directly on the camera's own pixel formats |
| `obs-plugin/src/ml_eye.c`, `ml_model.m` | Personal eye model: eye patches, full-resolution compositing, Core ML |
| `obs-plugin/tools/model/` | Record, extract and train the personal eye model (on your Mac only) |
| `obs-plugin/tests/` | Behaviour, quality and OBS integration tests, which run on every push |
| `obs-plugin/build.sh` | Builds the plugin. It only needs clang, and GitHub runs it for you |
| `standalone-app/` | An earlier design: a standalone virtual camera app. It needs a paid Apple Developer account, so it's parked |
