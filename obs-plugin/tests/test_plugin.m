/*
 * Runs the real OBS filter code against OBS's own libobs, on macOS, without the OBS app UI.
 * Built and run in CI by tests/run_plugin_test.sh.
 *
 * Checks: the filter creates (Apple Vision objects and tracking thread included), its settings
 * panel builds, Apple Vision accepts our grey frames on the tracking thread, every camera format
 * comes out exactly mirrored when no face is present, the eye warp never touches pixels outside
 * the eye ellipses, calibration requests reach the tracking thread, a calibration without camera
 * frames gives up, a newer camera calibration that invalidates the notes is reported, a real face
 * photo is found and then tracked (also at 2560x1440, where Vision sees a shrunk copy), each
 * filter gets its own hotkey name, old calibrations are reset, and destroying the filter stops
 * the tracking thread promptly.
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

/* With the tracking thread paused, feed the gaze model a known face and check the warp only
 * changes pixels inside the eye ellipses it reports. */
static void run_warp_bounds(struct camflip *f)
{
	const int w = 1280, h = 720;
	pthread_mutex_lock(&f->lock);
	f->test_pause = true;
	f->mirror = false;
	f->eye_contact = true;
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

	test_frame tf;
	make_frame(&tf, VIDEO_FORMAT_NV12, w, h);
	const double t = 10.0 + 45 / 30.0;
	tf.frame.timestamp = (uint64_t)(t * 1e9);
	camflip_filter_video(f, &tf.frame);

	cf_eye eyes[2];
	pthread_mutex_lock(&f->gaze_lock);
	const int n = cf_gaze_predict(&f->gaze, &params, t, eyes);
	pthread_mutex_unlock(&f->gaze_lock);

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
	check(outside == 0, "eye warp leaves every pixel outside the eye ellipses unchanged");
	free_frame(&tf);

	pthread_mutex_lock(&f->lock);
	f->test_pause = false;
	f->mirror = true;
	pthread_cond_broadcast(&f->wake);
	pthread_mutex_unlock(&f->lock);
	check(wait_idle(f, 5000), "tracking thread resumes");
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
	check(strstr(text, "Calibrating") != NULL, "status shows calibration in progress");
	free_frame(&tf);
}

/* A calibration that gets no camera frames gives up after CALIB_STALL_NS instead of showing
 * "Calibrating…" forever, both on the tracking thread (which wakes up for it) and in the panel. */
static void run_calibration_stall(struct camflip *f)
{
	camflip_calib_notes_clicked(NULL, NULL, f);
	pthread_mutex_lock(&f->lock);
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

	char text[700];
	bool warn = false;
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(warn && strstr(text, "no camera frames") != NULL && strstr(text, "Calibrating") == NULL,
	      "status says calibration didn't finish for lack of camera frames");

	/* The panel finds it on its own too (whichever thread gets there first). */
	camflip_calib_lens_clicked(NULL, NULL, f);
	pthread_mutex_lock(&f->lock);
	f->calib_progress_ns = now_ns() - CALIB_STALL_NS - 1000000ull;
	pthread_mutex_unlock(&f->lock);
	status_text(f, text, sizeof(text), &warn);
	check(warn && strstr(text, "no camera frames") != NULL, "settings panel gives up a calibration without frames");

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
 * is cancelled at once (PR #1) with the panel's flipped-frames message, instead of waiting for the
 * no-frames timeout. A normal frame afterwards doesn't bring it back. */
static void run_flipped_frames(struct camflip *f)
{
	wait_idle(f, 5000);
	const unsigned long before = runs(f);
	camflip_calib_lens_clicked(NULL, NULL, f);
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
	check(cancelled && runs(f) == before && (!eye_contact || f->warned_flip),
	      "an upside-down frame skips tracking and cancels the calibration at once");

	char text[700];
	bool warn = false;
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(warn && strstr(text, "upside-down") != NULL && strstr(text, "Calibrating") == NULL,
	      "status says calibration can't run on upside-down frames");

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

/* Redoing the camera calibration so the saved notes are no longer below it: the panel says so and
 * the notes are not used until step 2 is redone. */
static void run_notes_stale(struct camflip *f, obs_data_t *restore)
{
	pthread_mutex_lock(&f->lock);
	f->baseline = 0.0f;
	f->baseline_open = 0.3f;
	f->lens_done = true;
	f->has_notes = true;
	f->notes_offset = -0.10f; /* 0.10 eye widths below the lens */
	f->notes_side = 0.0f;
	f->notes_open = 0.25f;
	f->needs_recalib = false;
	f->calib_running = CALIB_IDLE;
	f->calib_failed = CALIB_IDLE;
	f->calib_notes_bad = false;
	f->calib_no_frames = false;
	f->calib_done_msg = NULL;
	check_notes_locked(f);
	const bool used_before = !f->notes_stale && params_locked(f).has_notes == 1;
	const cf_calib_result lens = {.target = CF_CALIB_LENS, .offset = -0.09f, .openness = 0.3f, .side = 0.0f};
	bool bad = false;
	obs_data_t *changes = apply_calibration_locked(f, &lens, &bad);
	const bool stale = changes && !bad && f->notes_stale && f->has_notes && params_locked(f).has_notes == 0;
	pthread_mutex_unlock(&f->lock);
	obs_data_release(changes);
	check(used_before && stale, "a camera calibration level with the notes stops using the notes");

	char text[700];
	bool warn = false;
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(warn && strstr(text, "no longer matches") != NULL && strstr(text, "redo step 2") != NULL &&
		      strstr(text, "notes done") == NULL,
	      "status asks to redo the notes calibration");

	obs_properties_t *props = camflip_properties(f);
	obs_property_t *slider = props ? obs_properties_get(props, S_MAX_GLANCE) : NULL;
	check(slider && obs_property_enabled(slider) && obs_properties_get(props, "forget_notes"),
	      "glance slider is back in use while the notes are invalid");
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
	pthread_mutex_lock(&f->lock);
	f->calib_done_msg = NULL;
	pthread_mutex_unlock(&f->lock);
}

/* Step 1 also learns the head looking into the camera (pose.c): saved with the calibration and read
 * back after a restart. A camera calibration saved by an older version has none: a typical face is
 * assumed and the panel asks to redo step 1. */
static void run_head_reference(struct camflip *f, obs_data_t *restore)
{
	pthread_mutex_lock(&f->lock);
	f->needs_recalib = false;
	f->calib_running = CALIB_IDLE;
	f->calib_failed = CALIB_IDLE;
	f->calib_notes_bad = false;
	f->calib_no_frames = false;
	f->calib_done_msg = NULL;
	cf_calib_result lens;
	memset(&lens, 0, sizeof(lens));
	lens.target = CF_CALIB_LENS;
	lens.offset = 0.01f;
	lens.openness = 0.3f;
	lens.head = (cf_head_ref){.ok = 1, .side = 0.03f, .vert = 0.62f, .eyes = -0.02f, .tilt = 0.11f};
	lens.upper = 0.21f;
	lens.lower = 0.11f;
	bool bad = false;
	obs_data_t *changes = apply_calibration_locked(f, &lens, &bad);
	const bool learnt = changes && !bad && f->head_ref.ok && params_locked(f).head_ref.ok &&
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
	check(strstr(text, "camera done") != NULL && strstr(text, "Redo step 1") == NULL,
	      "status: camera calibrated with the head");

	/* Saved by the previous version: the camera calibrated, no head. */
	obs_data_t *saved_old = obs_data_create();
	camflip_defaults(saved_old);
	obs_data_set_double(saved_old, S_BASELINE, 0.01);
	obs_data_set_double(saved_old, S_BASELINE_OPEN, 0.3);
	obs_data_set_int(saved_old, S_CALIB_VERSION, CALIB_VERSION);
	camflip_update(f, saved_old);
	pthread_mutex_lock(&f->lock);
	f->calib_done_msg = NULL;
	const bool typical = f->lens_done && !f->head_ref.ok && !params_locked(f).head_ref.ok &&
			     params_locked(f).baseline_upper == 0.0f && params_locked(f).baseline_lower == 0.0f;
	pthread_mutex_unlock(&f->lock);
	status_text(f, text, sizeof(text), &warn);
	printf("info: status panel says:\n%s\n", text);
	check(typical && !warn && strstr(text, "camera done") != NULL && strstr(text, "Redo step 1") != NULL,
	      "a camera calibration without the head assumes a typical face and asks to redo step 1");

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
	pthread_mutex_lock(&f->lock);
	f->calib_done_msg = NULL;
	pthread_mutex_unlock(&f->lock);
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
	check(strstr(text, "Face found.") != NULL, "status says the face is found");

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
	check(f->needs_recalib && fabsf(f->baseline - (float)DEFAULT_BASELINE) < 1e-6f &&
		      fabsf(f->max_glance - 0.15f) < 1e-6f && obs_data_get_int(old, S_CALIB_VERSION) == CALIB_VERSION &&
		      !f->head_ref.ok && !obs_data_has_user_value(old, S_HEAD_REF) &&
		      !obs_data_has_user_value(old, S_HEAD_VERT),
	      "calibration saved by an older version is reset and asks to calibrate again");
	camflip_destroy(f);
	obs_data_release(old);

	obs_data_t *cur = obs_data_create();
	camflip_defaults(cur);
	obs_data_set_double(cur, S_BASELINE, 0.02);
	obs_data_set_double(cur, S_BASELINE_OPEN, 0.3);
	obs_data_set_int(cur, S_CALIB_VERSION, CALIB_VERSION);
	f = camflip_create(cur, NULL);
	check(!f->needs_recalib && fabsf(f->baseline - 0.02f) < 1e-6f && f->lens_done,
	      "current calibration is kept");
	camflip_destroy(f);
	obs_data_release(cur);
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

	obs_properties_t *props = camflip_properties(f);
	check(props && obs_properties_get(props, S_STRENGTH) && obs_properties_get(props, "calibrate_lens") &&
		      obs_properties_get(props, "calibrate_notes") && obs_properties_get(props, "status") &&
		      obs_properties_get(props, S_SHOW_EYES),
	      "settings panel builds");
	obs_properties_destroy(props);
	props = camflip_properties(NULL); /* OBS may ask without a filter */
	check(props != NULL, "settings panel builds without a filter");
	obs_properties_destroy(props);

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

	if (failures) {
		printf("%d FAILED\n", failures);
		return 1;
	}
	printf("ALL CHECKS PASSED\n");
	return 0;
}
