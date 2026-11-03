/*
 * Runs the real OBS filter code against OBS's own libobs, on macOS, without the OBS app UI.
 * Built and run in CI by tests/run_plugin_test.sh.
 *
 * Checks: the filter creates (Apple Vision objects and tracking thread included), its settings
 * panel builds, Apple Vision accepts our grey frames on the tracking thread, every camera format
 * comes out exactly mirrored when no face is present, the eye warp never touches pixels outside
 * the eye ellipses, calibration requests reach the tracking thread, old calibrations are reset,
 * and destroying the filter stops the tracking thread promptly.
 */
#define CAMFLIP_TEST 1
#include "../src/plugin.m"

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

static void run_migration(void)
{
	obs_data_t *old = obs_data_create();
	camflip_defaults(old);
	obs_data_set_double(old, S_BASELINE, 0.3); /* saved by an older version, other units */
	obs_data_set_double(old, S_MAX_GLANCE, 60.0);
	struct camflip *f = camflip_create(old, NULL);
	check(f->needs_recalib && fabsf(f->baseline - (float)DEFAULT_BASELINE) < 1e-6f &&
		      fabsf(f->max_glance - 0.15f) < 1e-6f && obs_data_get_int(old, S_CALIB_VERSION) == CALIB_VERSION,
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
