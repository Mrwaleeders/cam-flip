/*
 * Runs the real OBS filter code against OBS's own libobs, on macOS, without the OBS app UI.
 * Built and run in CI by tests/run_plugin_test.sh.
 *
 * Checks: the filter creates (Apple Vision objects and tracking thread included), its settings
 * panel builds in the agreed order with the advanced settings hidden until "Show advanced settings"
 * is on, Apple Vision accepts our grey frames on the tracking thread, every camera format comes out
 * exactly mirrored when no face is present, the eye warp never touches pixels outside the eye
 * ellipses and nothing is warped before the camera look, calibration requests reach the tracking
 * thread, a calibration without camera frames gives up, a camera look that is unsteady or at the
 * screen is turned down (keeping the previous one, with the failure sound), notes learnt from the
 * reading are used, saved, read back and forgotten, a newer camera calibration that invalidates the
 * notes is reported, a real face photo is found and then tracked (also at 2560x1440, where Vision
 * sees a shrunk copy), each filter gets its own hotkey name, old calibrations are reset, settings
 * from before the automatic setup are simplified once, and destroying the filter stops the tracking
 * thread promptly. All calibration numbers here are made up.
 *
 * tests/face.jpg: the astronaut Eileen Collins, an official NASA portrait (a NASA photograph, so
 * public domain), cropped to 384x216. Taken from scikit-image's sample data (skimage/data/
 * astronaut.png), whose documentation says: "This image was downloaded from the NASA Great Images
 * database <https://flic.kr/p/r9qvLn>. No known copyright restrictions, released into the public
 * domain."
 */
#define CAMFLIP_TEST 1
#include "../src/plugin.m"

#import <CoreGraphics/CoreGraphics.h>

#include <stdio.h>
#include <unistd.h>

static int failures = 0;

static void check(bool ok, const char *what)
{
	printf("%s: %s\n", ok ? "ok" : "FAIL", what);
	if (!ok)
		failures++;
}

static void fill(uint8_t *p, size_t n)
{
	for (size_t i = 0; i < n; i++)
		p[i] = (uint8_t)(rand() & 0xff);
}

static double ms_since(uint64_t t0)
{
	return (double)(now_ns() - t0) / 1e6;
}

static unsigned long runs(struct camflip *f)
{
	pthread_mutex_lock(&f->lock);
	unsigned long n = f->vision_runs;
	pthread_mutex_unlock(&f->lock);
	return n;
}

/* Worker has nothing waiting and nothing in progress. */
static bool idle(struct camflip *f)
{
	pthread_mutex_lock(&f->lock);
	bool r = f->busy < 0 && f->pending < 0;
	pthread_mutex_unlock(&f->lock);
	return r;
}

static bool wait_idle(struct camflip *f, int timeout_ms)
{
	uint64_t t0 = now_ns();
	while (!idle(f)) {
		if (ms_since(t0) > timeout_ms)
			return false;
		usleep(2000);
	}
	return true;
}

static bool wait_runs(struct camflip *f, unsigned long at_least, int timeout_ms)
{
	uint64_t t0 = now_ns();
	while (runs(f) < at_least) {
		if (ms_since(t0) > timeout_ms)
			return false;
		usleep(2000);
	}
	return true;
}

/* The sound play_cue last recorded (CUE_DONE / CUE_FAILED), or NULL. */
static const char *last_cue(struct camflip *f)
{
	pthread_mutex_lock(&f->lock);
	const char *cue = f->last_cue;
	pthread_mutex_unlock(&f->lock);
	return cue;
}

static bool cue_is(struct camflip *f, const char *want)
{
	const char *cue = last_cue(f);
	return cue && strcmp(cue, want) == 0;
}

/* The tracking thread plays its cue a moment after it sets the state the test waits for: wait for it. */
static bool wait_cue(struct camflip *f, const char *want, int timeout_ms)
{
	const uint64_t t0 = now_ns();
	while (!cue_is(f, want)) {
		if (ms_since(t0) > timeout_ms)
			return false;
		usleep(2000);
	}
	return true;
}

/* No step running, nothing failed: the panel then shows the setup state itself. */
static void clear_step_state(struct camflip *f)
{
	pthread_mutex_lock(&f->lock);
	f->calib_request = 0;
	f->calib_running = CALIB_IDLE;
	f->calib_failed = CALIB_IDLE;
	f->calib_notes_bad = false;
	f->look_bad = CF_LOOK_OK;
	f->calib_no_frames = false;
	f->last_cue = NULL;
	pthread_mutex_unlock(&f->lock);
}

/* A step running, as the tracking thread has it once it took the request (no frames needed). */
static void begin_step(struct camflip *f, int step)
{
	clear_step_state(f);
	pthread_mutex_lock(&f->lock);
	f->calib_running = step;
	f->calib_progress_ns = now_ns(); /* (else the tracking thread gives it up for lack of frames) */
	pthread_mutex_unlock(&f->lock);
}

typedef struct {
	struct obs_source_frame frame;
	uint8_t *orig[3];
	size_t sizes[3];
} test_frame;

static void make_frame(test_frame *tf, enum video_format vf, int w, int h)
{
	memset(tf, 0, sizeof(*tf));
	tf->frame.width = w;
	tf->frame.height = h;
	tf->frame.format = vf;
	switch (vf) {
	case VIDEO_FORMAT_NV12:
		tf->frame.linesize[0] = w;
		tf->frame.linesize[1] = w;
		tf->sizes[0] = (size_t)w * h;
		tf->sizes[1] = (size_t)w * h / 2;
		break;
	case VIDEO_FORMAT_UYVY:
		tf->frame.linesize[0] = w * 2;
		tf->sizes[0] = (size_t)w * h * 2;
		break;
	default: /* BGRA */
		tf->frame.linesize[0] = w * 4;
		tf->sizes[0] = (size_t)w * h * 4;
		break;
	}
	for (int i = 0; i < 3; i++)
		if (tf->sizes[i]) {
			tf->frame.data[i] = malloc(tf->sizes[i]);
			fill(tf->frame.data[i], tf->sizes[i]);
			tf->orig[i] = malloc(tf->sizes[i]);
			memcpy(tf->orig[i], tf->frame.data[i], tf->sizes[i]);
		}
}

static void free_frame(test_frame *tf)
{
	for (int i = 0; i < 3; i++) {
		free(tf->frame.data[i]);
		free(tf->orig[i]);
	}
}

static void run_format(struct camflip *f, enum video_format vf, const char *name)
{
	const int w = 1280, h = 720;
	test_frame tf;
	make_frame(&tf, vf, w, h);
	struct obs_source_frame *frame = &tf.frame;

	const unsigned long before = runs(f);
	for (int k = 0; k < 5; k++) {
		frame->timestamp = (uint64_t)k * 33333333ull;
		uint64_t t0 = now_ns();
		camflip_filter_video(f, frame);
		const double ms = ms_since(t0);
		if (ms > 50) /* Vision alone takes longer than this on CI; the video thread must not wait for it */
			printf("info: %s filter_video took %.1f ms (slow runner?)\n", name, ms);
		/* Give the tracking thread time to pick the frame up, like a real 30 fps camera would. */
		wait_idle(f, 3000);
		/* Undo the mirror on all but the last pass, so the final check is a single flip. */
		if (k < 4) {
			cf_channel ch[8];
			int n = cf_channels(to_cf_format(vf), frame->data, frame->linesize, w, h, ch);
			cf_mirror_frame(ch, n);
		}
	}
	char msg[160];
	snprintf(msg, sizeof(msg), "%s frames reach the tracking thread", name);
	check(wait_runs(f, before + 1, 5000), msg);

	/* Expect exactly one horizontal flip of the original. */
	cf_channel a[8], b[8];
	int n = cf_channels(to_cf_format(vf), tf.orig, frame->linesize, w, h, a);
	cf_channels(to_cf_format(vf), frame->data, frame->linesize, w, h, b);
	int bad = 0;
	for (int c = 0; c < n && !bad; c++)
		for (int y = 0; y < a[c].height && !bad; y++)
			for (int x = 0; x < a[c].width; x++)
				if (a[c].base[(size_t)y * a[c].linesize + (size_t)x * a[c].step] !=
				    b[c].base[(size_t)y * b[c].linesize + (size_t)(a[c].width - 1 - x) * b[c].step]) {
					bad = 1;
					break;
				}
	snprintf(msg, sizeof(msg), "%s frames pass through the filter as an exact mirror", name);
	check(!bad, msg);
	free_frame(&tf);
}

/* A face the gaze model can't miss: two eyes glancing slightly down. */
static void synthetic_face(cf_face *face, int w, int h)
{
	memset(face, 0, sizeof(*face));
	face->found = 1;
	face->image_width = w;
	face->image_height = h;
	face->face_width = 420;
	face->has_pose = 1;
	const float cx[2] = {560, 720}, cy = 330, rx = 30, ry = 11;
	cf_pt *eyes[2] = {face->left_eye, face->right_eye};
	for (int e = 0; e < 2; e++) {
		for (int i = 0; i < 16; i++) {
			const float a = 6.2831853f * (float)i / 16.0f;
			eyes[e][i] = (cf_pt){cx[e] + rx * cosf(a), cy + ry * sinf(a)};
		}
	}
	face->left_count = face->right_count = 16;
	face->left_pupil = (cf_pt){cx[0], cy + 3};
	face->right_pupil = (cf_pt){cx[1], cy + 3};
}

/*
 * With the tracking thread paused, feeds the gaze model a known face glancing down and runs one
 * unmirrored NV12 frame through the filter, with the camera look done or not. Returns how many eyes
 * the model would warp for that frame (in `eyes`). End with resume_tracking().
 */
static int filter_synthetic_glance(struct camflip *f, bool lens_done, test_frame *tf, cf_eye eyes[2],
				   bool *lens_before)
{
	const int w = 1280, h = 720;
	pthread_mutex_lock(&f->lock);
	f->test_pause = true;
	f->mirror = false;
	f->eye_contact = true;
	*lens_before = f->lens_done;
	f->lens_done = lens_done; /* (the warp's own gate; the gaze params don't depend on it) */
	const cf_gaze_params params = params_locked(f);
	pthread_mutex_unlock(&f->lock);
	check(wait_idle(f, 5000), "tracking thread pauses");

	cf_face face;
	synthetic_face(&face, w, h);
	cf_eye tmp[2];
	pthread_mutex_lock(&f->gaze_lock);
	cf_gaze_init(&f->gaze);
	for (int k = 0; k < 45; k++)
		cf_gaze_update(&f->gaze, &face, &params, 10.0 + k / 30.0, tmp);
	pthread_mutex_unlock(&f->gaze_lock);
	f->vt_was_on = true; /* don't let filter_video reset the model we just fed */

	make_frame(tf, VIDEO_FORMAT_NV12, w, h);
	const double t = 10.0 + 45 / 30.0;
	tf->frame.timestamp = (uint64_t)(t * 1e9);
	camflip_filter_video(f, &tf->frame);

	pthread_mutex_lock(&f->gaze_lock);
	const int n = cf_gaze_predict(&f->gaze, &params, t, eyes);
	pthread_mutex_unlock(&f->gaze_lock);
	return n;
}

static void resume_tracking(struct camflip *f, bool lens_before)
{
	pthread_mutex_lock(&f->lock);
	f->test_pause = false;
	f->mirror = true;
	f->lens_done = lens_before;
	pthread_cond_broadcast(&f->wake);
	pthread_mutex_unlock(&f->lock);
	check(wait_idle(f, 5000), "tracking thread resumes");
}

/* The warp only changes pixels inside the eye ellipses the gaze model reports. */
static void run_warp_bounds(struct camflip *f)
{
	const int w = 1280, h = 720;
	test_frame tf;
	cf_eye eyes[2];
	bool lens_before = false;
	const int n = filter_synthetic_glance(f, true, &tf, eyes, &lens_before);

	cf_channel a[8], b[8];
	const int count = cf_channels(CF_FMT_NV12, tf.orig, tf.frame.linesize, w, h, a);
	cf_channels(CF_FMT_NV12, tf.frame.data, tf.frame.linesize, w, h, b);
	int outside = 0, inside = 0;
	for (int c = 0; c < count; c++)
		for (int y = 0; y < a[c].height; y++)
			for (int x = 0; x < a[c].width; x++) {
				const size_t o = (size_t)y * a[c].linesize + (size_t)x * a[c].step;
				if (a[c].base[o] == b[c].base[o])
					continue;
				/* centre of this sample in image pixels */
				const float X = ((float)x + 0.5f) / a[c].scale_x, Y = ((float)y + 0.5f) / a[c].scale_y;
				const float pad = 1.5f / a[c].scale_x;
				bool in = false;
				for (int e = 0; e < n; e++) {
					const float dx = (X - eyes[e].cx) / (eyes[e].rx + pad);
					const float dy = (Y - eyes[e].cy) / (eyes[e].ry + pad);
					if (dx * dx + dy * dy <= 1.0f)
						in = true;
				}
				if (in)
					inside++;
				else
					outside++;
			}
	printf("info: synthetic glance: %d eyes warped, %d samples changed inside the eyes\n", n, inside);
	check(n == 2 && inside > 0, "with the camera look done, a glance is warped");
	check(outside == 0, "eye warp leaves every pixel outside the eye ellipses unchanged");
	free_frame(&tf);
	resume_tracking(f, lens_before);
}

/* No correction before the camera look: there is no "looking into the lens" to correct towards. The
 * eyes are still tracked (the model would warp them), but the frame comes out untouched. */
static void run_no_warp_before_lens(struct camflip *f)
{
	test_frame tf;
	cf_eye eyes[2];
	bool lens_before = false;
	const int n = filter_synthetic_glance(f, false, &tf, eyes, &lens_before);
	bool same = true;
	for (int i = 0; i < 3; i++)
		if (tf.sizes[i] && memcmp(tf.frame.data[i], tf.orig[i], tf.sizes[i]) != 0)
			same = false;
	printf("info: before the camera look: the model tracks %d eyes, frame %s\n", n, same ? "untouched" : "changed");
	check(n == 2 && same, "nothing is warped before the camera look, though the eyes are tracked");
	free_frame(&tf);
	resume_tracking(f, lens_before);
}

static void run_calibration_request(struct camflip *f)
{
	camflip_calib_lens_clicked(NULL, NULL, f);
	test_frame tf;
	make_frame(&tf, VIDEO_FORMAT_NV12, 640, 360);
	tf.frame.timestamp = 20000000000ull;
	camflip_filter_video(f, &tf.frame);
	wait_idle(f, 5000);
	pthread_mutex_lock(&f->lock);
	const bool taken = f->calib_request == 0 && f->calib_running == CALIB_LENS;
	pthread_mutex_unlock(&f->lock);
	check(taken, "calibration request is started by the tracking thread");

	char text[700];
	bool warn = false;
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(strstr(text, "Look into the camera lens…") != NULL && !warn, "status says to look into the lens");
	free_frame(&tf);
}

/* A calibration that gets no camera frames gives up after CALIB_STALL_NS instead of asking the user
 * to look forever, both on the tracking thread (which wakes up for it) and in the panel, with the
 * failure sound. */
static void run_calibration_stall(struct camflip *f)
{
	camflip_calib_notes_clicked(NULL, NULL, f);
	pthread_mutex_lock(&f->lock);
	f->last_cue = NULL;
	f->calib_progress_ns = now_ns() - CALIB_STALL_NS - 1000000ull; /* pretend 12 s passed without frames */
	pthread_mutex_unlock(&f->lock);
	bool gave_up = false;
	const uint64_t t0 = now_ns();
	while (!gave_up && ms_since(t0) < 3000) {
		pthread_mutex_lock(&f->lock);
		gave_up = f->calib_request == 0 && f->calib_running == CALIB_IDLE && f->calib_no_frames;
		pthread_mutex_unlock(&f->lock);
		if (!gave_up)
			usleep(10000);
	}
	check(gave_up, "calibration without camera frames gives up (tracking thread)");
	check(wait_cue(f, CUE_FAILED, 3000), "giving up plays the failure sound");

	char text[700];
	bool warn = false;
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(warn && strstr(text, "no video from the camera") != NULL && strstr(text, "Look at the middle") == NULL,
	      "status says the step didn't work for lack of camera frames");

	/* The panel finds it on its own too (whichever thread gets there first plays the sound). */
	camflip_calib_lens_clicked(NULL, NULL, f);
	pthread_mutex_lock(&f->lock);
	f->last_cue = NULL;
	f->calib_progress_ns = now_ns() - CALIB_STALL_NS - 1000000ull;
	pthread_mutex_unlock(&f->lock);
	status_text(f, text, sizeof(text), &warn);
	check(warn && strstr(text, "no video from the camera") != NULL && wait_cue(f, CUE_FAILED, 3000),
	      "settings panel gives up a calibration without frames");

	/* A frame arriving later doesn't bring the abandoned calibration back. */
	test_frame tf;
	make_frame(&tf, VIDEO_FORMAT_NV12, 640, 360);
	tf.frame.timestamp = 25000000000ull;
	camflip_filter_video(f, &tf.frame);
	wait_idle(f, 5000);
	pthread_mutex_lock(&f->lock);
	const bool still_idle = f->calib_running == CALIB_IDLE && f->calib_request == 0 && !f->calib_cancel;
	pthread_mutex_unlock(&f->lock);
	check(still_idle, "an abandoned calibration stays abandoned when frames come back");
	free_frame(&tf);
}

/* Upside-down frames (frame->flip): not tracked, logged once, and a calibration requested on them
 * is cancelled at once (PR #1) with the panel's upside-down message and the failure sound, instead
 * of waiting for the no-frames timeout. A normal frame afterwards doesn't bring it back. */
static void run_flipped_frames(struct camflip *f)
{
	wait_idle(f, 5000);
	const unsigned long before = runs(f);
	camflip_calib_lens_clicked(NULL, NULL, f);
	pthread_mutex_lock(&f->lock);
	f->last_cue = NULL;
	pthread_mutex_unlock(&f->lock);
	test_frame tf;
	make_frame(&tf, VIDEO_FORMAT_NV12, 640, 360);
	tf.frame.timestamp = 26000000000ull;
	tf.frame.flip = true;
	camflip_filter_video(f, &tf.frame);
	wait_idle(f, 5000);
	pthread_mutex_lock(&f->lock);
	const bool eye_contact = f->eye_contact;
	const bool cancelled = f->calib_request == 0 && f->calib_running == CALIB_IDLE && f->calib_no_frames &&
			       f->frames_flipped;
	pthread_mutex_unlock(&f->lock);
	check(cancelled && runs(f) == before && (!eye_contact || f->warned_flip) && cue_is(f, CUE_FAILED),
	      "an upside-down frame skips tracking and cancels the calibration at once, with the failure sound");

	char text[700];
	bool warn = false;
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(warn && strstr(text, "upside down") != NULL && strstr(text, "Look into the camera lens") == NULL,
	      "status says the camera look can't run on upside-down frames");

	tf.frame.flip = false;
	tf.frame.timestamp = 26100000000ull;
	camflip_filter_video(f, &tf.frame);
	wait_idle(f, 5000);
	pthread_mutex_lock(&f->lock);
	const bool still_idle = f->calib_running == CALIB_IDLE && f->calib_request == 0 && !f->calib_cancel &&
				!f->frames_flipped;
	pthread_mutex_unlock(&f->lock);
	check(still_idle, "the cancelled calibration stays cancelled when upright frames come");
	free_frame(&tf);
}

/* The calibration lives in the settings (written through obs_source_update on the UI thread), so a
 * saved value is read back and the Defaults button (user value removed) resets it. (PR #1's check;
 * the default baseline is DEFAULT_BASELINE here.) */
static void run_baseline_defaults(struct camflip *f, obs_data_t *settings)
{
	obs_data_set_double(settings, S_BASELINE, 0.125);
	obs_data_set_double(settings, S_BASELINE_OPEN, 0.3);
	camflip_update(f, settings);
	pthread_mutex_lock(&f->lock);
	const float saved = f->baseline;
	const bool lens_saved = f->lens_done;
	pthread_mutex_unlock(&f->lock);
	obs_data_unset_user_value(settings, S_BASELINE);
	obs_data_unset_user_value(settings, S_BASELINE_OPEN);
	camflip_update(f, settings);
	pthread_mutex_lock(&f->lock);
	const float reset = f->baseline;
	const bool lens_reset = f->lens_done;
	pthread_mutex_unlock(&f->lock);
	check(fabsf(saved - 0.125f) < 1e-6f && lens_saved && fabsf(reset - (float)DEFAULT_BASELINE) < 1e-6f && !lens_reset,
	      "baseline comes from the settings and Defaults resets it");
}

/* Redoing the camera calibration so the saved notes step is no longer below it: the panel says so
 * and the notes step is not used until it is redone. */
static void run_notes_stale(struct camflip *f, obs_data_t *restore)
{
	clear_step_state(f);
	pthread_mutex_lock(&f->lock);
	f->baseline = 0.0f;
	f->baseline_open = 0.3f;
	f->lens_done = true;
	f->has_notes = true;
	f->notes_offset = -0.10f; /* 0.10 eye widths below the lens */
	f->notes_side = 0.0f;
	f->notes_open = 0.25f;
	check_notes_locked(f);
	const bool used_before = !f->notes_stale && params_locked(f).has_notes == 1;
	const cf_calib_result lens = {.target = CF_CALIB_LENS, .offset = -0.09f, .openness = 0.3f, .side = 0.0f};
	obs_data_t *changes = apply_calibration_locked(f, &lens);
	const bool stale = changes && f->notes_stale && f->has_notes && params_locked(f).has_notes == 0;
	pthread_mutex_unlock(&f->lock);
	obs_data_release(changes);
	check(used_before && stale, "a camera calibration level with the notes stops using the notes");

	char text[700];
	bool warn = false;
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(warn && strstr(text, "no longer fits") != NULL && strstr(text, "“Look at your notes”") != NULL &&
		      strstr(text, "Ready") == NULL,
	      "status asks to redo the notes step");

	obs_properties_t *props = camflip_properties(f);
	obs_property_t *slider = props ? obs_properties_get(props, S_MAX_GLANCE) : NULL;
	obs_property_t *forget = props ? obs_properties_get(props, "forget_notes") : NULL;
	check(slider && obs_property_enabled(slider) && forget && obs_property_enabled(forget),
	      "glance slider is back in use while the notes are invalid, and they can be forgotten");
	obs_properties_destroy(props);

	/* Same after a restart: worked out from the saved settings. */
	obs_data_t *saved = obs_data_create();
	camflip_defaults(saved);
	obs_data_set_double(saved, S_BASELINE, -0.09);
	obs_data_set_double(saved, S_BASELINE_OPEN, 0.3);
	obs_data_set_double(saved, S_NOTES_OFFSET, -0.10);
	obs_data_set_bool(saved, S_HAS_NOTES, true);
	obs_data_set_int(saved, S_CALIB_VERSION, CALIB_VERSION);
	camflip_update(f, saved);
	pthread_mutex_lock(&f->lock);
	const bool stale_loaded = f->notes_stale;
	pthread_mutex_unlock(&f->lock);
	obs_data_set_double(saved, S_BASELINE, 0.0);
	camflip_update(f, saved);
	pthread_mutex_lock(&f->lock);
	const bool valid_again = !f->notes_stale && params_locked(f).has_notes == 1;
	pthread_mutex_unlock(&f->lock);
	check(stale_loaded && valid_again, "saved notes are checked against the saved camera calibration");
	obs_data_release(saved);

	camflip_update(f, restore);
}

/* The camera look also learns the head looking into the camera (pose.c): saved with the calibration
 * and read back after a restart. A camera look saved by an older version has none: a typical face is
 * assumed and the panel asks for the camera look once more. */
static void run_head_reference(struct camflip *f, obs_data_t *restore)
{
	clear_step_state(f);
	pthread_mutex_lock(&f->lock);
	cf_calib_result lens;
	memset(&lens, 0, sizeof(lens));
	lens.target = CF_CALIB_LENS;
	lens.offset = 0.01f;
	lens.openness = 0.3f;
	lens.head = (cf_head_ref){.ok = 1, .side = 0.03f, .vert = 0.62f, .eyes = -0.02f, .tilt = 0.11f};
	lens.upper = 0.21f;
	lens.lower = 0.11f;
	obs_data_t *changes = apply_calibration_locked(f, &lens);
	const bool learnt = changes && f->head_ref.ok && params_locked(f).head_ref.ok &&
			    fabsf(params_locked(f).head_ref.vert - 0.62f) < 1e-6f &&
			    fabsf(params_locked(f).baseline_upper - 0.21f) < 1e-6f &&
			    fabsf(params_locked(f).baseline_lower - 0.11f) < 1e-6f;
	pthread_mutex_unlock(&f->lock);
	const bool saved = changes && obs_data_get_bool(changes, S_HEAD_REF) &&
			   fabs(obs_data_get_double(changes, S_HEAD_TILT) - 0.11) < 1e-6 &&
			   fabs(obs_data_get_double(changes, S_BASELINE_UPPER) - 0.21) < 1e-6 &&
			   fabs(obs_data_get_double(changes, S_BASELINE_LOWER) - 0.11) < 1e-6;
	obs_data_release(changes);
	check(learnt && saved, "the camera calibration learns and saves the head and the lids looking into the camera");

	char text[700];
	bool warn = false;
	status_text(f, text, sizeof(text), &warn);
	check(!warn && strstr(text, "Ready.") != NULL && strstr(text, "For the best result") == NULL,
	      "status: ready with the head measured");

	/* Saved by the previous version: the camera calibrated, no head. */
	obs_data_t *saved_old = obs_data_create();
	camflip_defaults(saved_old);
	obs_data_set_double(saved_old, S_BASELINE, 0.01);
	obs_data_set_double(saved_old, S_BASELINE_OPEN, 0.3);
	obs_data_set_int(saved_old, S_CALIB_VERSION, CALIB_VERSION);
	camflip_update(f, saved_old);
	pthread_mutex_lock(&f->lock);
	const bool typical = f->lens_done && !f->head_ref.ok && !params_locked(f).head_ref.ok &&
			     params_locked(f).baseline_upper == 0.0f && params_locked(f).baseline_lower == 0.0f;
	pthread_mutex_unlock(&f->lock);
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(typical && !warn && strstr(text, "Ready. For the best result, click “Look at the camera” once more.") != NULL,
	      "a camera look without the head assumes a typical face and asks for the look once more");

	/* Read back with it. */
	obs_data_set_bool(saved_old, S_HEAD_REF, true);
	obs_data_set_double(saved_old, S_HEAD_SIDE, 0.03);
	obs_data_set_double(saved_old, S_HEAD_VERT, 0.62);
	obs_data_set_double(saved_old, S_HEAD_EYES, -0.02);
	obs_data_set_double(saved_old, S_HEAD_TILT, 0.11);
	obs_data_set_double(saved_old, S_BASELINE_UPPER, 0.21);
	obs_data_set_double(saved_old, S_BASELINE_LOWER, 0.11);
	camflip_update(f, saved_old);
	pthread_mutex_lock(&f->lock);
	const bool loaded = f->head_ref.ok && fabsf(f->head_ref.vert - 0.62f) < 1e-6f &&
			    fabsf(f->head_ref.tilt - 0.11f) < 1e-6f && fabsf(f->head_ref.eyes + 0.02f) < 1e-6f &&
			    fabsf(params_locked(f).baseline_upper - 0.21f) < 1e-6f &&
			    fabsf(params_locked(f).baseline_lower - 0.11f) < 1e-6f;
	pthread_mutex_unlock(&f->lock);
	check(loaded, "the head and the lids looking into the camera are read back from the settings");
	obs_data_release(saved_old);

	camflip_update(f, restore);
}

/* The panel, top to bottom, as agreed: the simple part, then the advanced settings, then the tip. */
static void run_panel_layout(struct camflip *f)
{
	static const char *const want[] = {S_MIRROR,     S_EYE_CONTACT,     "calibrate_lens", "status",
					   S_ADVANCED,   S_STRENGTH,        S_SCAN_DAMP,      S_MAX_GLANCE,
					   "calibrate_notes", "forget_notes", S_SHOW_EYES,    "details",
					   "info"};
	const size_t n_want = sizeof(want) / sizeof(want[0]);
	obs_properties_t *props = camflip_properties(f);
	size_t n = 0;
	bool order = props != NULL;
	for (obs_property_t *p = props ? obs_properties_first(props) : NULL; p; obs_property_next(&p)) {
		const char *name = obs_property_name(p);
		printf("info: panel: %s \"%s\"%s\n", name, obs_property_description(p),
		       obs_property_visible(p) ? "" : " (hidden)");
		if (n >= n_want || strcmp(name, want[n]) != 0)
			order = false;
		n++;
	}
	check(order && n == n_want, "settings panel builds, in the agreed order");
	obs_properties_destroy(props);
	props = camflip_properties(NULL); /* OBS may ask without a filter */
	check(props != NULL, "settings panel builds without a filter");
	obs_properties_destroy(props);
}

static bool advanced_visible(obs_properties_t *props, bool visible)
{
	for (size_t i = 0; i < sizeof(advanced_ids) / sizeof(advanced_ids[0]); i++) {
		obs_property_t *p = obs_properties_get(props, advanced_ids[i]);
		if (!p || obs_property_visible(p) != visible)
			return false;
	}
	return true;
}

static bool simple_visible(obs_properties_t *props)
{
	static const char *const ids[] = {S_MIRROR, S_EYE_CONTACT, "calibrate_lens", "status", S_ADVANCED, "info"};
	for (size_t i = 0; i < sizeof(ids) / sizeof(ids[0]); i++) {
		obs_property_t *p = obs_properties_get(props, ids[i]);
		if (!p || !obs_property_visible(p))
			return false;
	}
	return true;
}

/* "Show advanced settings": the advanced settings are hidden until it is on, both when the panel is
 * built (from the saved setting) and when it is ticked (its modified callback). The tracked-eyes
 * overlay only works while they are shown, so it can't be left on unseen. */
static void run_advanced_visibility(struct camflip *f, obs_data_t *restore)
{
	obs_data_t *s = obs_data_create();
	camflip_defaults(s);
	camflip_update(f, s);
	obs_properties_t *props = camflip_properties(f);
	check(props && simple_visible(props) && advanced_visible(props, false), "advanced settings are hidden by default");

	obs_property_t *adv = obs_properties_get(props, S_ADVANCED);
	obs_data_set_bool(s, S_ADVANCED, true);
	const bool relayout = obs_property_modified(adv, s);
	check(relayout && advanced_visible(props, true) && simple_visible(props),
	      "ticking “Show advanced settings” shows them");
	obs_data_set_bool(s, S_ADVANCED, false);
	obs_property_modified(adv, s);
	check(advanced_visible(props, false) && simple_visible(props), "unticking it hides them again");
	obs_properties_destroy(props);

	/* Saved on: shown from the start, also once OBS applies the settings to the new panel. */
	obs_data_set_bool(s, S_ADVANCED, true);
	camflip_update(f, s);
	props = camflip_properties(f);
	const bool shown = advanced_visible(props, true);
	obs_properties_apply_settings(props, s);
	check(shown && advanced_visible(props, true), "with it saved on, the panel opens with them shown");
	obs_properties_destroy(props);

	obs_data_set_bool(s, S_SHOW_EYES, true);
	camflip_update(f, s);
	pthread_mutex_lock(&f->lock);
	const bool overlay_on = f->show_eyes;
	pthread_mutex_unlock(&f->lock);
	obs_data_set_bool(s, S_ADVANCED, false);
	camflip_update(f, s);
	pthread_mutex_lock(&f->lock);
	const bool overlay_off = !f->show_eyes;
	pthread_mutex_unlock(&f->lock);
	check(overlay_on && overlay_off, "the tracked-eyes overlay works only while the advanced settings are shown");

	obs_data_release(s);
	camflip_update(f, restore);
}

/* Settings with a camera look (synthetic) and no notes step, advanced settings shown. */
static obs_data_t *set_up_settings(void)
{
	obs_data_t *s = obs_data_create();
	camflip_defaults(s);
	obs_data_set_double(s, S_BASELINE, 0.0);
	obs_data_set_double(s, S_BASELINE_OPEN, 0.3);
	obs_data_set_bool(s, S_HEAD_REF, true);
	obs_data_set_double(s, S_HEAD_VERT, 0.5);
	obs_data_set_double(s, S_BASELINE_UPPER, 0.2);
	obs_data_set_double(s, S_BASELINE_LOWER, 0.1);
	obs_data_set_int(s, S_VIS_CAL, 1);
	obs_data_set_double(s, S_VIS_BASELINE, 0.01);
	obs_data_set_bool(s, S_ADVANCED, true);
	obs_data_set_int(s, S_CALIB_VERSION, CALIB_VERSION);
	return s;
}

/*
 * Notes learnt from the reading (a synthetic cf_learnt_notes, taken the way the tracking thread takes
 * them): passed to the gaze model, saved without rebuilding an open panel (except the first time,
 * which changes what it shows), read back after a restart, second to a notes step, and forgotten with
 * "Forget my notes", in the gaze model too.
 */
static void run_learnt_notes(struct camflip *f, obs_data_t *restore)
{
	obs_data_t *s = set_up_settings();
	camflip_update(f, s);
	clear_step_state(f);
	char text[700], details[320];
	bool warn = false;
	status_text(f, text, sizeof(text), &warn);
	details_text(f, details, sizeof(details));
	printf("info: status panel says:\n%s\ninfo: details say:\n%s\n", text, details);
	check(!warn && strstr(text, "Ready. Read your notes as usual: it learns where they are as you read.") != NULL &&
		      strstr(details, "Notes not learnt yet") != NULL,
	      "set up, nothing learnt yet: the panel says it learns the notes as you read");

	cf_learnt_notes l;
	memset(&l, 0, sizeof(l));
	l.has = 1;
	l.offset = -0.07f;
	l.side = 0.04f;
	l.open = 0.26f;
	l.has_vis = 1;
	l.vis_offset = -0.02f;
	l.seconds = 40.0f;
	l.serial = 1;
	pthread_mutex_lock(&f->lock);
	f->last_sig = status_sig_locked(f);
	f->last_refresh = 0;
	pthread_mutex_unlock(&f->lock);
	obs_data_t *first = take_learnt_notes(f, &l);
	pthread_mutex_lock(&f->lock);
	const cf_gaze_params p = params_locked(f);
	const bool first_refresh = refresh_due_locked(f, false, now_ns());
	pthread_mutex_unlock(&f->lock);
	check(first && obs_data_get_bool(first, S_LEARNT) &&
		      fabs(obs_data_get_double(first, S_LEARNT_OFFSET) + 0.07) < 1e-6 &&
		      fabs(obs_data_get_double(first, S_LEARNT_SIDE) - 0.04) < 1e-6 &&
		      fabs(obs_data_get_double(first, S_LEARNT_OPEN) - 0.26) < 1e-6 &&
		      obs_data_get_bool(first, S_LEARNT_HAS_VIS) &&
		      fabs(obs_data_get_double(first, S_LEARNT_VIS) + 0.02) < 1e-6 &&
		      fabs(obs_data_get_double(first, S_LEARNT_SECONDS) - 40.0) < 1e-6,
	      "learnt notes are saved");
	check(p.has_notes == 1 && p.notes_learnt == 1 && fabsf(p.notes_offset + 0.07f) < 1e-6f &&
		      fabsf(p.notes_side - 0.04f) < 1e-6f && fabsf(p.notes_open - 0.26f) < 1e-6f &&
		      fabsf(p.vis_notes + 0.02f) < 1e-6f && p.has_vis_calib == 1,
	      "learnt notes are passed to the gaze model");
	check(first_refresh, "learning the notes the first time refreshes an open panel");
	pthread_mutex_lock(&f->lock);
	f->vis_cal = 0;
	const bool no_vis_gain = params_locked(f).has_vis_calib == 0 && params_locked(f).notes_learnt == 1;
	f->vis_cal = 1;
	pthread_mutex_unlock(&f->lock);
	check(no_vis_gain, "Vision's gain from learnt notes needs Vision's point at the camera look too");

	/* Refined as the user reads: saved, without rebuilding an open panel for it. */
	l.offset = -0.075f;
	l.seconds = 55.0f;
	l.serial = 2;
	obs_data_t *again = take_learnt_notes(f, &l);
	pthread_mutex_lock(&f->lock);
	const bool no_rebuild = !refresh_due_locked(f, false, now_ns() + 2 * REFRESH_NS);
	pthread_mutex_unlock(&f->lock);
	check(again && no_rebuild, "refined notes are saved without rebuilding an open panel");
	status_text(f, text, sizeof(text), &warn);
	details_text(f, details, sizeof(details));
	printf("info: status panel says:\n%s\ninfo: details say:\n%s\n", text, details);
	check(!warn && strstr(text, "Ready. It has learnt where your notes are.") != NULL &&
		      strstr(details, "Notes learnt from 55 s of reading.") != NULL,
	      "the panel says the notes are learnt, and from how much reading");

	/* After a restart: read back from the saved settings. */
	obs_data_apply(s, first);
	obs_data_apply(s, again);
	obs_data_release(first);
	obs_data_release(again);
	pthread_mutex_lock(&f->lock);
	memset(&f->learnt, 0, sizeof(f->learnt));
	pthread_mutex_unlock(&f->lock);
	camflip_update(f, s);
	pthread_mutex_lock(&f->lock);
	const cf_gaze_params pl = params_locked(f);
	const bool loaded = f->learnt.has && fabsf(f->learnt.offset + 0.075f) < 1e-6f &&
			    fabsf(f->learnt.seconds - 55.0f) < 1e-6f && f->learnt.has_vis && pl.notes_learnt == 1 &&
			    fabsf(pl.notes_offset + 0.075f) < 1e-6f;
	pthread_mutex_unlock(&f->lock);
	check(loaded, "learnt notes are read back from the settings");

	/* A notes step comes first ... */
	pthread_mutex_lock(&f->lock);
	f->has_notes = true;
	f->notes_offset = -0.12f;
	f->notes_side = 0.0f;
	f->notes_open = 0.24f;
	f->vis_cal |= 2;
	f->vis_notes = -0.04f;
	check_notes_locked(f);
	const cf_gaze_params pm = params_locked(f);
	pthread_mutex_unlock(&f->lock);
	status_text(f, text, sizeof(text), &warn);
	check(pm.has_notes == 1 && pm.notes_learnt == 0 && fabsf(pm.notes_offset + 0.12f) < 1e-6f &&
		      fabsf(pm.vis_notes + 0.04f) < 1e-6f && pm.has_vis_calib == 1 &&
		      strstr(text, "Ready (using your notes step).") != NULL,
	      "a notes step comes before the learnt notes");
	/* ... until it no longer fits the camera look: the learnt notes are used again. */
	pthread_mutex_lock(&f->lock);
	f->notes_offset = 0.005f;
	check_notes_locked(f);
	const cf_gaze_params ps = params_locked(f);
	pthread_mutex_unlock(&f->lock);
	check(ps.has_notes == 1 && ps.notes_learnt == 1 && fabsf(ps.notes_offset + 0.075f) < 1e-6f,
	      "a notes step that no longer fits falls back to the learnt notes");

	/* "Forget my notes": both go, in the gaze model too. The model's own estimate is set by hand here
	 * (as if it had learnt them) so taking it would hand it out. */
	pthread_mutex_lock(&f->gaze_lock);
	f->gaze.learnt = l;
	f->gaze.learnt_taken = 0;
	pthread_mutex_unlock(&f->gaze_lock);
	camflip_forget_notes_clicked(NULL, NULL, f);
	pthread_mutex_lock(&f->lock);
	const bool gone = !f->has_notes && !f->learnt.has && params_locked(f).has_notes == 0 && (f->vis_cal & 2) == 0 &&
			  f->forget_learnt;
	pthread_mutex_unlock(&f->lock);
	cf_learnt_notes out;
	pthread_mutex_lock(&f->gaze_lock);
	const bool model_forgot = cf_gaze_take_learnt_notes(&f->gaze, &out) == 0;
	pthread_mutex_unlock(&f->gaze_lock);
	check(gone && model_forgot, "“Forget my notes” forgets the notes step and the learnt notes, in the model too");
	/* Notes the model handed out before the click (taken while the forget is still pending) don't
	 * come back. */
	l.serial = 3;
	obs_data_t *late = take_learnt_notes(f, &l);
	pthread_mutex_lock(&f->lock);
	const bool still_gone = !f->learnt.has;
	f->forget_learnt = false; /* (what the tracking thread does on its next frame) */
	pthread_mutex_unlock(&f->lock);
	check(!late && still_gone, "notes learnt before “Forget my notes” don't come back");
	if (late)
		obs_data_release(late);

	obs_data_release(s);
	camflip_update(f, restore);
}

/* "Look at the camera" is checked before it is used (cf_gaze_check_lens_look): a look with the eyes
 * moving, or one that is not above where the user reads (a look at the screen), is turned down with
 * the failure sound and the previous camera look is kept; a good one is saved with the success
 * sound. The notes step and a step that gave up have their sounds too. */
static void run_lens_look(struct camflip *f, obs_data_t *restore)
{
	obs_data_t *s = set_up_settings();
	obs_data_set_double(s, S_BASELINE, 0.01); /* the previous camera look */
	camflip_update(f, s);

	cf_calib_result look;
	memset(&look, 0, sizeof(look));
	look.target = CF_CALIB_LENS;
	look.offset = 0.02f;
	look.openness = 0.31f;
	look.head = (cf_head_ref){.ok = 1, .side = 0.01f, .vert = 0.5f, .eyes = 0.0f, .tilt = 0.1f};
	look.upper = 0.2f;
	look.lower = 0.1f;
	char text[700];
	bool warn = false;
	obs_data_t *changes = NULL;

	/* The eyes wandered off and back. */
	cf_calib_result moved = look;
	moved.spread = 0.5f;
	moved.drift = 0.5f;
	begin_step(f, CALIB_LENS);
	bool taken = finish_step(f, 1, &moved, &changes);
	pthread_mutex_lock(&f->lock);
	bool kept = !changes && f->lens_done && fabsf(f->baseline - 0.01f) < 1e-6f && f->look_bad == CF_LOOK_UNSTEADY &&
		    f->calib_running == CALIB_IDLE;
	pthread_mutex_unlock(&f->lock);
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(taken && kept, "a camera look with the eyes moving is turned down, keeping the previous one");
	check(warn && strstr(text, "Your eyes moved: click “Look at the camera” and hold your gaze on the lens.") &&
		      cue_is(f, CUE_FAILED),
	      "the panel says the eyes moved, with the failure sound");

	/* Level with where the user reads: a look at the screen, not into the lens. */
	cf_learnt_notes reading;
	memset(&reading, 0, sizeof(reading));
	reading.has = 1;
	reading.offset = -0.05f;
	reading.open = 0.25f;
	reading.seconds = 30.0f;
	reading.serial = 1;
	obs_data_t *learnt = take_learnt_notes(f, &reading);
	check(learnt != NULL, "(notes learnt for the next check)");
	if (learnt)
		obs_data_release(learnt);
	cf_calib_result low = look;
	low.offset = -0.045f;
	begin_step(f, CALIB_LENS);
	taken = finish_step(f, 1, &low, &changes);
	pthread_mutex_lock(&f->lock);
	kept = !changes && fabsf(f->baseline - 0.01f) < 1e-6f && f->look_bad == CF_LOOK_LOW;
	pthread_mutex_unlock(&f->lock);
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(taken && kept && warn &&
		      strstr(text, "That looked like a look at the screen, not the lens: click “Look at the camera” and "
				   "look right into the lens.") &&
		      cue_is(f, CUE_FAILED),
	      "a camera look that is not above the reading is turned down as a look at the screen");

	/* A good look: saved, with the success sound. */
	begin_step(f, CALIB_LENS);
	taken = finish_step(f, 1, &look, &changes);
	pthread_mutex_lock(&f->lock);
	const bool used = fabsf(f->baseline - 0.02f) < 1e-6f && f->look_bad == CF_LOOK_OK && f->lens_done;
	pthread_mutex_unlock(&f->lock);
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(taken && used && changes && fabs(obs_data_get_double(changes, S_BASELINE) - 0.02) < 1e-6 &&
		      obs_data_get_int(changes, S_SETUP_VERSION) == SETUP_VERSION && cue_is(f, CUE_DONE) && !warn &&
		      strstr(text, "Ready. It has learnt where your notes are.") != NULL,
	      "a good camera look is saved, with the success sound");
	if (changes)
		obs_data_release(changes);

	/* The notes step: turned down when not below the camera look, else used, each with its sound. */
	cf_calib_result notes;
	memset(&notes, 0, sizeof(notes));
	notes.target = CF_CALIB_NOTES;
	notes.offset = 0.03f;
	notes.openness = 0.27f;
	begin_step(f, CALIB_NOTES);
	finish_step(f, 1, &notes, &changes);
	status_text(f, text, sizeof(text), &warn);
	check(!changes && warn && strstr(text, "That didn't look like your notes below the camera") != NULL &&
		      cue_is(f, CUE_FAILED),
	      "a notes step that is not below the camera is turned down, with the failure sound");
	notes.offset = -0.08f;
	begin_step(f, CALIB_NOTES);
	finish_step(f, 1, &notes, &changes);
	status_text(f, text, sizeof(text), &warn);
	check(changes && !warn && strstr(text, "Ready (using your notes step).") != NULL && cue_is(f, CUE_DONE),
	      "a notes step below the camera is used, with the success sound");
	if (changes)
		obs_data_release(changes);

	/* A step that gave up (no steady face for about 10 s). */
	begin_step(f, CALIB_LENS);
	taken = finish_step(f, -1, &look, &changes);
	pthread_mutex_lock(&f->lock);
	const bool failed = f->calib_failed == CALIB_LENS && fabsf(f->baseline - 0.02f) < 1e-6f;
	pthread_mutex_unlock(&f->lock);
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(taken && !changes && failed && warn && strstr(text, "Couldn't see your face well") != NULL &&
		      cue_is(f, CUE_FAILED),
	      "a camera look that gave up says so, with the failure sound");

	/* A step abandoned meanwhile (no frames): its late result is dropped, silently. */
	clear_step_state(f);
	taken = finish_step(f, 1, &look, &changes);
	check(!taken && !changes && last_cue(f) == NULL, "a late result of an abandoned step is dropped");

	clear_step_state(f);
	obs_data_release(s);
	camflip_update(f, restore);
}

/* Two camera looks running that are not above the learnt reading: it is the learnt notes that are off (from
 * before a new seat or camera), so the second look is taken and the notes are forgotten, to be learnt again. */
static void run_lens_look_second_try(struct camflip *f, obs_data_t *restore)
{
	obs_data_t *s = set_up_settings();
	obs_data_set_double(s, S_BASELINE, 0.01); /* the previous camera look */
	camflip_update(f, s);
	pthread_mutex_lock(&f->lock);
	f->look_low_prev = false;
	pthread_mutex_unlock(&f->lock);

	cf_learnt_notes reading;
	memset(&reading, 0, sizeof(reading));
	reading.has = 1;
	reading.offset = -0.05f;
	reading.open = 0.25f;
	reading.seconds = 30.0f;
	reading.serial = 1;
	obs_data_t *learnt = take_learnt_notes(f, &reading);
	check(learnt != NULL, "(notes learnt for the second-try check)");
	if (learnt)
		obs_data_release(learnt);

	cf_calib_result low;
	memset(&low, 0, sizeof(low));
	low.target = CF_CALIB_LENS;
	low.offset = -0.045f; /* level with the reading */
	low.openness = 0.31f;
	low.head = (cf_head_ref){.ok = 1, .side = 0.01f, .vert = 0.5f, .eyes = 0.0f, .tilt = 0.1f};
	low.upper = 0.2f;
	low.lower = 0.1f;
	obs_data_t *changes = NULL;
	begin_step(f, CALIB_LENS);
	finish_step(f, 1, &low, &changes);
	check(!changes && cue_is(f, CUE_FAILED), "(the first look level with the reading is turned down)");
	if (changes)
		obs_data_release(changes);

	begin_step(f, CALIB_LENS);
	const bool taken = finish_step(f, 1, &low, &changes);
	pthread_mutex_lock(&f->lock);
	const bool used = fabsf(f->baseline + 0.045f) < 1e-6f && f->look_bad == CF_LOOK_OK && f->lens_done && !f->learnt.has;
	pthread_mutex_unlock(&f->lock);
	check(taken && used && changes && fabs(obs_data_get_double(changes, S_BASELINE) + 0.045) < 1e-6 &&
		      obs_data_has_user_value(changes, S_LEARNT) && !obs_data_get_bool(changes, S_LEARNT) &&
		      cue_is(f, CUE_DONE),
	      "a second camera look running level with the reading is taken, and the learnt notes are forgotten");
	if (changes)
		obs_data_release(changes);

	/* A look turned down for another reason in between starts the count again. (The forget is passed on to
	 * the gaze model as the tracking thread does before its next frame, and notes are learnt again.) */
	pthread_mutex_lock(&f->lock);
	const bool forget = f->forget_learnt;
	f->forget_learnt = false;
	pthread_mutex_unlock(&f->lock);
	pthread_mutex_lock(&f->gaze_lock);
	if (forget)
		cf_gaze_forget_learnt_notes(&f->gaze);
	pthread_mutex_unlock(&f->gaze_lock);
	learnt = take_learnt_notes(f, &reading);
	if (learnt)
		obs_data_release(learnt);
	begin_step(f, CALIB_LENS);
	finish_step(f, 1, &low, &changes);
	if (changes)
		obs_data_release(changes);
	cf_calib_result moved = low;
	moved.spread = 0.5f;
	moved.drift = 0.5f;
	begin_step(f, CALIB_LENS);
	finish_step(f, 1, &moved, &changes);
	if (changes)
		obs_data_release(changes);
	begin_step(f, CALIB_LENS);
	finish_step(f, 1, &low, &changes);
	pthread_mutex_lock(&f->lock);
	const bool again = f->look_bad == CF_LOOK_LOW && f->learnt.has;
	pthread_mutex_unlock(&f->lock);
	check(!changes && again && cue_is(f, CUE_FAILED),
	      "only two looks running level with the reading count: an unsteady one in between starts it again");
	if (changes)
		obs_data_release(changes);

	clear_step_state(f);
	obs_data_release(s);
	camflip_update(f, restore);
}

/* OBS's Defaults button (or undo) clearing learnt notes from the settings: the gaze model is told to forget them
 * too (else it takes them as handed out already and never hands the same notes out again). And the setup stamp
 * is put back, so the next start doesn't simplify the settings once more. */
static void run_learnt_defaults(struct camflip *f, obs_data_t *restore)
{
	obs_data_t *s = set_up_settings();
	obs_data_set_bool(s, S_LEARNT, true);
	obs_data_set_double(s, S_LEARNT_OFFSET, -0.05);
	obs_data_set_double(s, S_LEARNT_SIDE, 0.0);
	obs_data_set_double(s, S_LEARNT_OPEN, 0.25);
	obs_data_set_double(s, S_LEARNT_SECONDS, 30.0);
	camflip_update(f, s);
	pthread_mutex_lock(&f->lock);
	f->forget_learnt = false;
	const bool had = f->learnt.has != 0;
	pthread_mutex_unlock(&f->lock);

	obs_data_t *d = obs_data_create(); /* what Defaults leaves: the defaults only */
	camflip_defaults(d);
	camflip_update(f, d);
	pthread_mutex_lock(&f->lock);
	const bool forgot = !f->learnt.has && f->forget_learnt;
	f->forget_learnt = false;
	pthread_mutex_unlock(&f->lock);
	check(had && forgot, "Defaults clearing the learnt notes makes the gaze model forget them too");
	check(obs_data_has_user_value(d, S_SETUP_VERSION) && obs_data_get_int(d, S_SETUP_VERSION) == SETUP_VERSION,
	      "settings this version has read keep its setup stamp, also after Defaults");

	/* Settings that never had them (notes the tracking thread took, their save not landed yet) are no forget. */
	camflip_update(f, d);
	pthread_mutex_lock(&f->lock);
	const bool calm = !f->forget_learnt;
	pthread_mutex_unlock(&f->lock);
	check(calm, "settings without learnt notes, after settings without them, don't make the model forget");

	obs_data_release(d);
	obs_data_release(s);
	camflip_update(f, restore);
	pthread_mutex_lock(&f->lock);
	f->forget_learnt = false;
	pthread_mutex_unlock(&f->lock);
}

/* Notes learnt from the reading that are not below the camera look: the look was off (most likely at
 * the screen), so the panel asks for it again. */
static void run_look_seems_off(struct camflip *f, obs_data_t *restore)
{
	obs_data_t *s = set_up_settings();
	obs_data_set_double(s, S_BASELINE, -0.02);
	camflip_update(f, s);
	clear_step_state(f);

	cf_learnt_notes l;
	memset(&l, 0, sizeof(l));
	l.has = 1;
	l.offset = -0.015f; /* above the camera look */
	l.open = 0.25f;
	l.seconds = 20.0f;
	l.serial = 1;
	obs_data_t *changes = take_learnt_notes(f, &l);
	char text[700];
	bool warn = false;
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(changes && warn &&
		      strstr(text, "Your camera look seems off: click “Look at the camera” and look right into the lens.") !=
			      NULL,
	      "notes learnt above the camera look: the panel says the camera look seems off");
	if (changes)
		obs_data_release(changes);

	l.offset = -0.09f; /* below it */
	l.serial = 2;
	changes = take_learnt_notes(f, &l);
	status_text(f, text, sizeof(text), &warn);
	check(changes && !warn && strstr(text, "seems off") == NULL && strstr(text, "It has learnt") != NULL,
	      "notes learnt below the camera look: no warning");
	if (changes)
		obs_data_release(changes);

	obs_data_release(s);
	camflip_update(f, restore);
}

static void run_hotkey_names(void)
{
	char n[4][512], d[4][512], again[512], again_d[512];
	hotkey_names(FILTER_NAME, n[0], sizeof(n[0]), d[0], sizeof(d[0]));
	hotkey_names(FILTER_NAME " 2", n[1], sizeof(n[1]), d[1], sizeof(d[1]));
	hotkey_names(NULL, n[2], sizeof(n[2]), d[2], sizeof(d[2]));
	hotkey_names("", n[3], sizeof(n[3]), d[3], sizeof(d[3]));
	hotkey_names(FILTER_NAME " 2", again, sizeof(again), again_d, sizeof(again_d));
	bool distinct = true;
	for (int i = 0; i < 4; i++)
		for (int j = i + 1; j < 4; j++)
			if (strcmp(n[i], n[j]) == 0)
				distinct = false;
	printf("info: hotkey names: \"%s\", \"%s\", \"%s\", \"%s\"\n", n[0], n[1], n[2], n[3]);
	check(strcmp(n[0], HOTKEY_NAME) == 0 && strcmp(d[0], HOTKEY_DESC) == 0,
	      "a filter with the default name keeps the hotkey name earlier versions saved bindings under");
	check(distinct, "each filter on a camera gets its own hotkey name");
	check(strcmp(again, n[1]) == 0 && strstr(d[1], FILTER_NAME " 2") != NULL,
	      "hotkey name is stable and the description names the filter");
}

/* tests/face.jpg drawn to fill a w x h BGRA frame (grey bars if the shapes differ), moved right
 * by shift_x pixels. NULL if it can't be loaded. */
static uint8_t *face_bgra(int w, int h, double shift_x)
{
	const char *path = getenv("CAMFLIP_FACE");
	if (!path)
		path = "tests/face.jpg"; /* run_plugin_test.sh runs from obs-plugin/ */
	CFURLRef url = CFURLCreateFromFileSystemRepresentation(kCFAllocatorDefault, (const UInt8 *)path,
							       (CFIndex)strlen(path), false);
	if (!url)
		return NULL;
	CGImageSourceRef src = CGImageSourceCreateWithURL(url, NULL);
	CFRelease(url);
	if (!src)
		return NULL;
	CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
	CFRelease(src);
	if (!img)
		return NULL;
	uint8_t *buf = calloc((size_t)w * h * 4, 1);
	CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
	/* Premultiplied-first alpha in little-endian 32-bit words: bytes B, G, R, A in memory. */
	CGContextRef ctx = buf ? CGBitmapContextCreate(buf, (size_t)w, (size_t)h, 8, (size_t)w * 4, cs,
						       (CGBitmapInfo)kCGImageAlphaPremultipliedFirst |
							       kCGBitmapByteOrder32Little)
			       : NULL;
	CGColorSpaceRelease(cs);
	if (ctx) {
		CGContextSetRGBFillColor(ctx, 0.5, 0.5, 0.5, 1.0);
		CGContextFillRect(ctx, CGRectMake(0, 0, w, h));
		CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
		const double dh = h, dw = (double)CGImageGetWidth(img) * h / (double)CGImageGetHeight(img);
		CGContextDrawImage(ctx, CGRectMake((w - dw) / 2 + shift_x, 0, dw, dh), img);
		CGContextRelease(ctx);
	} else {
		free(buf);
		buf = NULL;
	}
	CGImageRelease(img);
	return buf;
}

/* Feeds `n` copies of a BGRA picture through the filter like a 30 fps camera, letting the tracking
 * thread analyse each one. */
static void feed_picture(struct camflip *f, const uint8_t *bgra, int w, int h, int n, double t0)
{
	test_frame tf;
	memset(&tf, 0, sizeof(tf));
	tf.frame.width = w;
	tf.frame.height = h;
	tf.frame.format = VIDEO_FORMAT_BGRA;
	tf.frame.linesize[0] = w * 4;
	const size_t size = (size_t)w * h * 4;
	tf.frame.data[0] = malloc(size);
	for (int k = 0; k < n; k++) {
		memcpy(tf.frame.data[0], bgra, size); /* the filter mirrors (and may warp) in place */
		tf.frame.timestamp = (uint64_t)((t0 + k / 30.0) * 1e9);
		camflip_filter_video(f, &tf.frame);
		wait_idle(f, 5000);
	}
	free(tf.frame.data[0]);
}

static cf_pt eye_centre(const cf_face *face, int e)
{
	const cf_pt *p = e ? face->right_eye : face->left_eye;
	const int n = e ? face->right_count : face->left_count;
	cf_pt c = {0, 0};
	for (int i = 0; i < n; i++) {
		c.x += p[i].x / (float)n;
		c.y += p[i].y / (float)n;
	}
	return c;
}

/* A real face photo: found by full detection, then followed by fast tracking (also when it moves),
 * and the gaze model sees it. At 2560x1440 Vision gets a half-size copy; the landmarks must still
 * come back in full-resolution pixels. */
static void run_face_tracking(struct camflip *f)
{
	const int w = 1280, h = 720;
	uint8_t *pic = face_bgra(w, h, 0), *moved = face_bgra(w, h, 8), *big = face_bgra(2 * w, 2 * h, 0);
	check(pic && moved && big, "tests/face.jpg loads");
	if (!pic || !moved || !big) {
		free(pic);
		free(moved);
		free(big);
		return;
	}
	pthread_mutex_lock(&f->lock);
	f->eye_contact = true;
	const unsigned long full0 = f->full_detections, tracked0 = f->tracked_runs;
	pthread_mutex_unlock(&f->lock);

	feed_picture(f, pic, w, h, 20, 40.0);
	pthread_mutex_lock(&f->lock);
	const cf_face seen = f->seen_face;
	const bool face_still = f->st_face;
	pthread_mutex_unlock(&f->lock);

	feed_picture(f, moved, w, h, 10, 40.0 + 20 / 30.0);
	pthread_mutex_lock(&f->lock);
	const cf_face seen_moved = f->seen_face;
	const bool face_moved = f->st_face;
	const unsigned long full = f->full_detections - full0, tracked = f->tracked_runs - tracked0;
	const bool too_far = f->st_too_far, turned = f->st_turned, lens = f->lens_done;
	pthread_mutex_unlock(&f->lock);

	const cf_pt l = eye_centre(&seen, 0), r = eye_centre(&seen, 1);
	const cf_pt lm = eye_centre(&seen_moved, 0);
	printf("info: face photo: %lu full detections, %lu tracked; eyes at (%.1f, %.1f) and (%.1f, %.1f), "
	       "left eye moved %.1f px for an 8 px shift\n",
	       full, tracked, l.x, l.y, r.x, r.y, lm.x - l.x);
	check(full >= 1, "a real face is found by full face detection");
	check(tracked >= 1, "the face is then followed by fast tracking");
	check(seen.found && seen_moved.found && face_still && face_moved, "the gaze model sees the face, still and moving");
	float head_yaw = 0, head_pitch = 0;
	cf_head_angles(&seen.head, NULL, &head_yaw, &head_pitch);
	printf("info: head measures (pose.c): side %.3f vert %.3f eyes %.3f tilt %.3f (moved: side %.3f vert %.3f); "
	       "a typical face would be turned %.1f deg\n",
	       seen.head.side, seen.head.vert, seen.head.eyes, seen.head.tilt, seen_moved.head.side, seen_moved.head.vert,
	       head_yaw);
	check(seen.has_head_geom && seen_moved.has_head_geom && seen.head.vert > 0.2f && seen.head.vert < 1.5f &&
		      fabsf(seen.head.side) < 0.5f,
	      "the head pose is measured from the face's landmarks (eyes and nose), tracked frames included");

	char text[700];
	bool warn = false;
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	const char *live = too_far ? "You're too far from the camera to correct your eyes."
			   : turned ? "Turn your face towards the camera."
			   : lens   ? "Eye contact is working."
				    : "It can see your face.";
	check(strstr(text, live) != NULL && strstr(text, "Can't see your face") == NULL,
	      "status says the face is found (in plain words)");

	feed_picture(f, big, 2 * w, 2 * h, 8, 50.0);
	pthread_mutex_lock(&f->lock);
	const cf_face seen_big = f->seen_face;
	pthread_mutex_unlock(&f->lock);
	const cf_pt bl = eye_centre(&seen_big, 0), br = eye_centre(&seen_big, 1);
	const float iod = hypotf(br.x - bl.x, br.y - bl.y);
	const float el = hypotf(bl.x - 2 * l.x, bl.y - 2 * l.y), er = hypotf(br.x - 2 * r.x, br.y - 2 * r.y);
	const float pl = hypotf(seen_big.left_pupil.x - 2 * seen.left_pupil.x, seen_big.left_pupil.y - 2 * seen.left_pupil.y);
	printf("info: 2560x1440: eyes at (%.1f, %.1f) and (%.1f, %.1f), %.1f and %.1f px from twice the 1280x720 "
	       "positions, left pupil %.1f px (eye distance %.1f px)\n",
	       bl.x, bl.y, br.x, br.y, el, er, pl, iod);
	check(seen_big.found && seen_big.image_width == 2 * w && iod > 0 && el < 0.1f * iod && er < 0.1f * iod &&
		      pl < 0.15f * iod,
	      "frames 2560 px and wider are tracked at reduced size with full-resolution coordinates");
	printf("info: 2560x1440 head measures: side %.3f vert %.3f eyes %.3f tilt %.3f\n", seen_big.head.side,
	       seen_big.head.vert, seen_big.head.eyes, seen_big.head.tilt);
	check(seen_big.has_head_geom && fabsf(seen_big.head.side - seen.head.side) < 0.05f &&
		      fabsf(seen_big.head.vert - seen.head.vert) < 0.05f,
	      "the head measures do not depend on the frame size");

	free(pic);
	free(moved);
	free(big);
}

static void run_migration(void)
{
	obs_data_t *old = obs_data_create();
	camflip_defaults(old);
	obs_data_set_double(old, S_BASELINE, 0.3); /* saved by an older version, other units */
	obs_data_set_double(old, S_MAX_GLANCE, 60.0);
	obs_data_set_bool(old, S_HEAD_REF, true);
	obs_data_set_double(old, S_HEAD_VERT, 0.6);
	struct camflip *f = camflip_create(old, NULL);
	char text[700];
	bool warn = false;
	status_text(f, text, sizeof(text), &warn);
	check(!f->lens_done && fabsf(f->baseline - (float)DEFAULT_BASELINE) < 1e-6f &&
		      fabsf(f->max_glance - 0.15f) < 1e-6f && obs_data_get_int(old, S_CALIB_VERSION) == CALIB_VERSION &&
		      !f->head_ref.ok && !obs_data_has_user_value(old, S_HEAD_REF) &&
		      !obs_data_has_user_value(old, S_HEAD_VERT) && strstr(text, "Almost ready") != NULL,
	      "calibration saved by an older version is reset and the panel asks for the camera look");
	camflip_destroy(f);
	obs_data_release(old);

	obs_data_t *cur = obs_data_create();
	camflip_defaults(cur);
	obs_data_set_double(cur, S_BASELINE, 0.02);
	obs_data_set_double(cur, S_BASELINE_OPEN, 0.3);
	obs_data_set_int(cur, S_CALIB_VERSION, CALIB_VERSION);
	f = camflip_create(cur, NULL);
	check(fabsf(f->baseline - 0.02f) < 1e-6f && f->lens_done, "current calibration is kept");
	camflip_destroy(f);
	obs_data_release(cur);
}

/* Settings saved before the panel set itself up: once, the hand-tuned strength, reading damping and
 * glance limit go back to their defaults and a saved notes step is dropped; the camera look stays.
 * Settings changed after that are left alone. */
static void run_setup_migration(void)
{
	obs_data_t *s = obs_data_create();
	camflip_defaults(s);
	obs_data_set_int(s, S_CALIB_VERSION, CALIB_VERSION);
	obs_data_set_double(s, S_STRENGTH, 160.0);
	obs_data_set_double(s, S_SCAN_DAMP, 55.0);
	obs_data_set_double(s, S_MAX_GLANCE, 22.0);
	obs_data_set_double(s, S_BASELINE, 0.02);
	obs_data_set_double(s, S_BASELINE_OPEN, 0.3);
	obs_data_set_bool(s, S_HEAD_REF, true);
	obs_data_set_double(s, S_HEAD_VERT, 0.5);
	obs_data_set_bool(s, S_HAS_NOTES, true);
	obs_data_set_double(s, S_NOTES_OFFSET, -0.08);
	obs_data_set_double(s, S_NOTES_SIDE, 0.05);
	obs_data_set_double(s, S_NOTES_OPEN, 0.25);
	obs_data_set_int(s, S_VIS_CAL, 3);
	obs_data_set_double(s, S_VIS_BASELINE, 0.01);
	obs_data_set_double(s, S_VIS_NOTES, -0.03);
	struct camflip *f = camflip_create(s, NULL);
	pthread_mutex_lock(&f->lock);
	const bool defaults = fabsf(f->strength - 1.0f) < 1e-6f && f->scan_damp == 0.0f &&
			      fabsf(f->max_glance - 0.15f) < 1e-6f;
	const bool notes_gone = !f->has_notes && f->vis_cal == 1 && params_locked(f).has_notes == 0;
	const bool lens_kept = f->lens_done && fabsf(f->baseline - 0.02f) < 1e-6f && f->head_ref.ok &&
			       fabsf(f->vis_baseline - 0.01f) < 1e-6f;
	pthread_mutex_unlock(&f->lock);
	const bool saved = !obs_data_has_user_value(s, S_STRENGTH) && !obs_data_has_user_value(s, S_SCAN_DAMP) &&
			   !obs_data_has_user_value(s, S_MAX_GLANCE) && !obs_data_has_user_value(s, S_HAS_NOTES) &&
			   !obs_data_has_user_value(s, S_NOTES_OFFSET) && !obs_data_has_user_value(s, S_VIS_NOTES) &&
			   obs_data_get_int(s, S_VIS_CAL) == 1 && obs_data_get_int(s, S_SETUP_VERSION) == SETUP_VERSION;
	check(defaults, "older settings: strength, reading damping and glance limit go back to their defaults");
	check(notes_gone && saved, "older settings: a saved notes step is dropped, in the saved settings too");
	check(lens_kept, "older settings: the camera look is kept");
	camflip_destroy(f);

	/* Only once: a setting changed since stays as it is at the next start. */
	obs_data_set_double(s, S_STRENGTH, 160.0);
	f = camflip_create(s, NULL);
	pthread_mutex_lock(&f->lock);
	const bool kept = fabsf(f->strength - 1.6f) < 1e-6f;
	pthread_mutex_unlock(&f->lock);
	check(kept && obs_data_has_user_value(s, S_STRENGTH), "the settings are simplified only once");
	camflip_destroy(f);
	obs_data_release(s);
}

int main(void)
{
	srand(1);
	obs_data_t *settings = obs_data_create();
	camflip_defaults(settings);
	struct camflip *f = camflip_create(settings, NULL);
	if (!f || !f->request || !f->handler || !f->worker_started) {
		printf("FAIL: filter did not create\n");
		return 1;
	}
	printf("ok: filter created with Apple Vision and a tracking thread\n");

	run_panel_layout(f);

	run_format(f, VIDEO_FORMAT_NV12, "NV12");
	run_format(f, VIDEO_FORMAT_UYVY, "UYVY");
	run_format(f, VIDEO_FORMAT_BGRA, "BGRA");

	pthread_mutex_lock(&f->lock);
	const unsigned long vr = f->vision_runs, vf = f->vision_failures, fd = f->full_detections,
			    tr = f->tracked_runs;
	pthread_mutex_unlock(&f->lock);
	printf("info: Vision analysed %lu frames (%lu full detections, %lu tracked), %lu failures, grey input %s\n", vr,
	       fd, tr, vf, f->vision_needs_bgra ? "rejected (using BGRA fallback)" : "accepted");
	check(vr > 0 && !(f->vision_needs_bgra && vf > 1), "Apple Vision processes the frames");

	/* Overlay on: still no change outside eyes when there is no face (mirror check again). */
	pthread_mutex_lock(&f->lock);
	f->show_eyes = true;
	pthread_mutex_unlock(&f->lock);
	run_format(f, VIDEO_FORMAT_NV12, "NV12 with tracked-eyes overlay on (no face)");
	pthread_mutex_lock(&f->lock);
	f->show_eyes = false;
	pthread_mutex_unlock(&f->lock);

	run_warp_bounds(f);
	run_no_warp_before_lens(f);

	camflip_hotkey(f, 0, NULL, true);
	camflip_hotkey(f, 0, NULL, false);
	pthread_mutex_lock(&f->lock);
	const bool off = !f->eye_contact;
	pthread_mutex_unlock(&f->lock);
	camflip_hotkey(f, 0, NULL, true);
	pthread_mutex_lock(&f->lock);
	const bool on = f->eye_contact;
	pthread_mutex_unlock(&f->lock);
	check(off && on, "hotkey toggles eye contact");

	run_calibration_request(f);
	run_calibration_stall(f);
	run_flipped_frames(f);
	run_baseline_defaults(f, settings);
	run_face_tracking(f);
	run_notes_stale(f, settings);
	run_head_reference(f, settings);
	run_advanced_visibility(f, settings);
	run_learnt_notes(f, settings);
	run_lens_look(f, settings);
	run_lens_look_second_try(f, settings);
	run_learnt_defaults(f, settings);
	run_look_seems_off(f, settings);
	run_hotkey_names();

	/* Destroy while a frame is waiting: must stop the tracking thread promptly. */
	test_frame tf;
	make_frame(&tf, VIDEO_FORMAT_NV12, 1280, 720);
	tf.frame.timestamp = 30000000000ull;
	camflip_filter_video(f, &tf.frame);
	uint64_t t0 = now_ns();
	camflip_destroy(f);
	const double dms = ms_since(t0);
	printf("info: destroy took %.1f ms\n", dms);
	check(dms < 2000, "destroy stops the tracking thread cleanly");
	free_frame(&tf);
	obs_data_release(settings);

	run_migration();
	run_setup_migration();

	if (failures) {
		printf("%d FAILED\n", failures);
		return 1;
	}
	printf("ALL CHECKS PASSED\n");
	return 0;
}
