# Cam Flip

Mirror your camera **for everyone in your call**, and keep **eye contact** while you read notes placed just below the camera. It works in Teams, Zoom, Google Meet and any app that can use a webcam.

Cam Flip is a free plugin for [OBS Studio](https://obsproject.com). OBS's built-in virtual camera, which OBS has already signed with Apple, carries the video into your call app. You don't need an Apple Developer account or Xcode.

## Why other people don't see your mirror today

Teams, Zoom and Meet mirror only your **own preview**. They always send the unflipped image to everyone else, and no setting changes that. Cam Flip flips the video before the call app receives it, so the flipped video is what gets sent.

> Once you use Cam Flip, turn **off** "Mirror my video" in Zoom and Teams. Otherwise they flip your preview back, and it won't match what others see. (Cam Flip's own setting of the same name stays on.)

## What it does

* **Mirror:** an exact pixel flip with no resampling, so there's no softening.
* **Eye contact:** Apple's Vision framework finds your eyes and pupils. When you glance down at notes placed just under the camera, Cam Flip moves your irises back up toward the lens, so the other person shouldn't be able to tell you're reading.
  * It sets itself up: you look into the camera once, and Cam Flip learns where your notes are from your own reading.
  * Most people tilt their head down a little while reading as well, and their eyes then move only a pixel or two. Cam Flip measures the tilt (against how your head sat when you looked into the camera) and lifts your irises by that much more, so your eyes meet the lens again. A head tilted down while your eyes stay on the lens gets no correction, and tilts past about 18° are left alone. Half of a sideways head turn is made up for in the same way.
  * Reading a line moves your eyes sideways, then back for the next line. **Hide reading along a line** (an advanced setting, off by default) takes out that much of the scanning (95% leaves about a third visible). Tracking runs a frame behind the video, so your eyes then twitch for one frame at each jump back to the next line. A look held to one side is never shifted.
  * Reading also lowers your upper lid and raises your lower lid a little, which looks like looking down even when the irises are lifted. While correcting, Cam Flip opens your lids back to how they were when you looked into the camera.
* **Looks natural when you move:**
  * Eye tracking runs in the background, a little behind the video, and Cam Flip predicts where your eyes are in each frame it corrects. So the correction stays on your eyes during normal head movement. Very fast, sudden movements can briefly show it a few pixels off before tracking catches up.
  * When you turn your head, look far down (for example at the keyboard) or to the side, the correction fades out smoothly instead of forcing a stare. When you glance back, it takes over again during the eye's own quick jump, where nobody can see it. When you turn or drift back slowly, it fades in over about half a second. It never switches suddenly.
  * Head turns and tilts are measured on every frame from your eye corners and nose, against how your head sits when you look into the camera. Turning or tilting your head more than about 15° either way starts the fade, and by about 25 to 30° the correction is gone. The exact angles vary a little from face to face.
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
3. In the filter's settings, **Mirror my video** and **Eye contact** are already on.
4. Sit as you will in calls, with your notes where you'll read them (just under the camera works best). Click **Look at the camera**, then look straight into the camera lens until you hear a sound (about 2 seconds).
   * You hear macOS's *Glass* sound when it worked, or *Basso* when it didn't. Either way, the status under the button says what to do next.
   * Nothing is corrected until this is done. Until then the status says **Almost ready**.
5. Read your notes as usual. Cam Flip learns where they are from your reading: the first time, after roughly 10 to 20 seconds of reading. The status then says **Ready. It has learnt where your notes are.** It remembers them after a restart.

Do **Look at the camera** once, and again only if you move the camera or your seat a lot. If you move your notes somewhere else, click **Forget my notes** (under **Show advanced settings**), and Cam Flip learns the new place as you read.

### 5. Use it in calls
* Click **Start Virtual Camera** in OBS. It's in the Controls panel at the bottom right.
* In Teams, Zoom or Meet, choose **OBS Virtual Camera** as your camera. If it isn't listed, quit and reopen the call app.
* Turn off "Mirror my video" in Zoom and Teams, as explained above. Keep Cam Flip's on.

To have it ready every time, add OBS to **System Settings › General › Login Items**. OBS remembers your scene and filter. Click Start Virtual Camera before a call.

## Settings explained

| Setting | What it does |
| --- | --- |
| Mirror my video | Flips the video that is actually sent, so everyone sees it mirrored. Zoom and Teams have a setting with the same name that only flips your own preview: turn theirs off. |
| Eye contact | Turns gaze correction on or off. |
| Look at the camera | Click, then look into the camera lens until you hear a sound (about 2 seconds). Cam Flip learns where your pupils sit, how open your eyes and lids are, and how your head sits when you look into the lens. Every correction is measured from this, so nothing is corrected until it's done. If your eyes moved, or it looked like a look at the screen rather than the lens, it isn't used: your previous one stays, and the status says what to do. |
| Status | Two short lines: what to do next (or **Ready**), and whether eye contact is working right now. |
| Show advanced settings | Shows the settings below. You shouldn't need them. |
| Defaults | Resets everything, including your camera look and the notes Cam Flip learnt. |

**Advanced settings** (tick **Show advanced settings**):

| Setting | What it does |
| --- | --- |
| Strength | Starts at 100%, which lifts your eyes by exactly how far they drop, measured from your camera look and your reading. If you still look like you're reading, go higher (up to 200%). The lids are never opened past how they were when you looked into the camera, and the eyes never move further than the warp allows without visible distortion. Lower values keep a little of the natural movement. |
| Hide reading along a line | Off by default. Takes out this much of your eyes' sideways movement as you read along a line. See *What it does* above. |
| Correct glances up to | How far down you can look and still be corrected. Beyond this, your real eyes show again, gradually. The notes Cam Flip learnt are always covered too. Not used while a notes step is in use. |
| Look at your notes (optional) | Not needed: Cam Flip learns where your notes are as you read. To set them yourself, click, then look at the middle of the line you usually read until you hear a sound. A notes step comes before what Cam Flip learnt. Your notes must be below the camera: if they aren't, or you looked elsewhere, it isn't used and the status says so. |
| Forget my notes | Forgets your notes step and what Cam Flip learnt from your reading. It learns them again as you read. Use it after moving your notes. |
| Show tracked eyes (for tuning) | While a glance is being corrected, draws thin outlines exactly where the correction is applied in that frame: a white ellipse for the area that may change, and black for the eye opening your iris is moved inside. Nothing is drawn while you look into the camera. Turn it off before a call. Hiding the advanced settings turns it off too. |
| Details | What Cam Flip knows about your notes (for example "Notes learnt from 40 s of reading"), and how long tracking takes. |

**Hotkey:** to switch eye contact on and off during a call, set a key in **OBS › Settings › Hotkeys**. Look for **Cam Flip: toggle eye contact** under your camera's name. If a camera has more than one Cam Flip filter, each one has its own hotkey, named after the filter, for example **Cam Flip: toggle eye contact (Cam Flip (mirror + eye contact) 2)** for a second filter OBS named “Cam Flip (mirror + eye contact) 2”. Renaming a filter keeps its key.

## Troubleshooting

* **The filter isn't in the list:** reopen OBS after installing, and check the plugin is in `~/Library/Application Support/obs-studio/plugins/cam-flip.plugin`. OBS's log (Help › Log Files) shows `[cam-flip] loaded`.
* **Eye contact does nothing:** check the status text in the filter's settings. **Almost ready** means **Look at the camera** isn't done yet. **Can't see your face** means Cam Flip can't find it: light your face so it fills a fair part of the frame. To see what Cam Flip sees, turn on **Show tracked eyes** in the advanced settings. The log shows `[cam-flip]` messages if your camera's pixel format isn't supported. If so, set the camera's **Video Format** to NV12 or UYVY in its properties.
* **You still look like you're reading:** give it 10 to 20 seconds of reading to learn where your notes are (the status then says it has). If the status says **For the best result, click "Look at the camera" once more**, do that: your camera look is from an older version (or your nose wasn't in view), so your lids and head tilt aren't corrected yet. Then raise **Strength** (in the advanced settings) above 100% until your eyes meet the lens.
* **No sound:** check your Mac's volume. The status says the same as the sound.
* **Status says "Your eyes moved":** your gaze moved during the camera look, so it wasn't used. Click **Look at the camera** again and keep your gaze on the lens until the sound.
* **Status says "That looked like a look at the screen, not the lens":** the camera look was level with where you read, so it can't have been the lens. Look right into the lens itself, not at the screen just below it.
* **Status says "Your camera look seems off":** the notes Cam Flip learnt from your reading aren't below your camera look, so the camera look was most likely at the screen. Click **Look at the camera** and look right into the lens.
* **Status says "Couldn't see your face well":** Cam Flip didn't see a steady, front-facing face for about 10 seconds. Light your face, face the camera, keep your head still and try again.
* **Status says "That didn't work: no video from the camera":** the camera sent no video for 12 seconds after you clicked. Check that the camera is on and not used by another app, and that you can see it in OBS, then click the button again.
* **Status says the camera's video comes upside down:** eye contact can't use this camera yet. Mirroring still works.
* **Status says "That didn't look like your notes below the camera"** (the optional notes step): it measured a gaze that wasn't below the camera. Look at the middle of your notes under the camera (not at the screen or the camera) and try again, or leave it: Cam Flip learns your notes as you read.
* **Status says "Your notes step no longer fits your camera look":** a newer camera look put the camera level with or below your notes step. Do **Look at your notes** again, or click **Forget my notes** to use the notes Cam Flip learns instead.
* **Your settings changed after updating:** this version sets itself up. The first time it starts, **Strength**, **Hide reading along a line** and **Correct glances up to** go back to their defaults, and a saved notes step is dropped, because your notes are now learnt as you read. Your camera look is kept. (A camera look from a much older version measured your eyes differently: that one is reset, and the status says **Almost ready**.)
* **Others see you un-mirrored:** check the filter is on the camera source, and the call app uses **OBS Virtual Camera**.

## Personal eye model (experimental, not in the plugin yet)

A small learned model, trained only on **your own face**, that may replace the hand-built eye warp. It moves the iris and both lids together, so the lowered lower lid and the side-to-side scan while reading no longer show. For now it runs in the render tool for side-by-side comparisons, not in OBS. Everything it uses or makes is your face, so it stays on your Mac: the repo only holds the code, and `.gitignore` blocks captures and models.

1. **Record** (about 4 minutes, head still, eyes follow a dot): `obs-plugin/tools/model/capture.sh --distance 60`. Use your real distance in cm. Do one session at your call distance and one at your desk distance.
2. **Cut out the eyes:** `obs-plugin/tools/model/extract.sh ~/Downloads/camflip-diag/captures/<session>` (for each session).
3. **Train** (about 30 to 60 minutes on Apple Silicon): `obs-plugin/tools/model/train.sh <session> <session>`. This writes `~/Library/Application Support/cam-flip/eye-model.mlpackage`, plus a preview of held-out frames showing your eye, the model's correction, and the real look into the lens.
4. **Compare:** `obs-plugin/tools/render.sh <clip.mov> <out-dir> --engine model`, against the same command without `--engine model`. The tool prints the model's time per frame. `--ml-compute cpu|gpu|ane` picks where it runs. By default the model holds your iris where it sits when you look into the lens (learnt from the recording), so the sideways sweep along a line and the jump back to the next line don't show; `--ml-target follow` instead follows the tracked pupil plus the classic shift.
5. **Try a training change in one go:** `obs-plugin/tools/model/round.sh <name> [train options]` cuts any missing patches, trains into `~/Downloads/camflip-diag/<name>/` (the installed model is left alone) and renders a test clip (`CLIP=<clip.mov>`, calibration in `RENDER="..."`) into `~/Downloads/camflip-diag/renders/model-ab/<name>/`. For quick trials use `--steps 6000` (about a third of the time); train the winner with the full 20000. Training lines up every frame once per session and keeps it in `align-corners.npy` next to the patches, so later rounds skip that step.

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
