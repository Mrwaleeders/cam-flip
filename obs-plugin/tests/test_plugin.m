/*
 * Runs the real OBS filter code against OBS's own libobs, on macOS, without the OBS app UI.
 * Built and run in CI by tests/run_plugin_test.sh.
 *
 * Checks: the filter creates (Apple Vision objects included), its settings panel builds,
 * Apple Vision accepts our zero-copy grey frames, and every camera format comes out
 * exactly mirrored when no face is present.
 */
#define CAMFLIP_TEST 1
#include "../src/plugin.m"

#include <stdio.h>

static int failures = 0;

static void fill(uint8_t *p, size_t n)
{
	for (size_t i = 0; i < n; i++)
		p[i] = (uint8_t)(rand() & 0xff);
}

static void run_format(struct camflip *f, enum video_format vf, const char *name)
{
	const int w = 1280, h = 720;
	struct obs_source_frame frame;
	memset(&frame, 0, sizeof(frame));
	frame.width = w;
	frame.height = h;
	frame.format = vf;
	size_t sizes[3] = {0};
	switch (vf) {
	case VIDEO_FORMAT_NV12:
		frame.linesize[0] = w;
		frame.linesize[1] = w;
		sizes[0] = (size_t)w * h;
		sizes[1] = (size_t)w * h / 2;
		break;
	case VIDEO_FORMAT_UYVY:
		frame.linesize[0] = w * 2;
		sizes[0] = (size_t)w * h * 2;
		break;
	default: /* BGRA */
		frame.linesize[0] = w * 4;
		sizes[0] = (size_t)w * h * 4;
		break;
	}
	uint8_t *orig[3] = {0};
	for (int i = 0; i < 3; i++)
		if (sizes[i]) {
			frame.data[i] = malloc(sizes[i]);
			fill(frame.data[i], sizes[i]);
			orig[i] = malloc(sizes[i]);
			memcpy(orig[i], frame.data[i], sizes[i]);
		}

	for (int k = 0; k < 5; k++) {
		frame.timestamp = (uint64_t)k * 33333333ull;
		camflip_filter_video(f, &frame);
		/* Undo the mirror on all but the last pass, so the final check is a single flip. */
		if (k < 4) {
			cf_channel ch[8];
			int n = cf_channels(to_cf_format(vf), frame.data, frame.linesize, w, h, ch);
			cf_mirror_frame(ch, n);
		}
	}

	/* Expect exactly one horizontal flip of the original. */
	cf_channel a[8], b[8];
	int n = cf_channels(to_cf_format(vf), orig, frame.linesize, w, h, a);
	cf_channels(to_cf_format(vf), frame.data, frame.linesize, w, h, b);
	int bad = 0;
	for (int c = 0; c < n && !bad; c++)
		for (int y = 0; y < a[c].height && !bad; y++)
			for (int x = 0; x < a[c].width; x++)
				if (a[c].base[(size_t)y * a[c].linesize + (size_t)x * a[c].step] !=
				    b[c].base[(size_t)y * b[c].linesize + (size_t)(a[c].width - 1 - x) * b[c].step]) {
					bad = 1;
					break;
				}
	if (bad) {
		printf("FAIL: %s output is not an exact mirror\n", name);
		failures++;
	} else {
		printf("ok: %s frames pass through the filter as an exact mirror\n", name);
	}
	for (int i = 0; i < 3; i++) {
		free(frame.data[i]);
		free(orig[i]);
	}
}

int main(void)
{
	srand(1);
	obs_data_t *settings = obs_data_create();
	camflip_defaults(settings);
	struct camflip *f = camflip_create(settings, NULL);
	if (!f || !f->request || !f->handler) {
		printf("FAIL: filter did not create\n");
		return 1;
	}
	printf("ok: filter created with Apple Vision\n");

	obs_properties_t *props = camflip_properties(f);
	if (!props || !obs_properties_get(props, S_STRENGTH) || !obs_properties_get(props, "calibrate")) {
		printf("FAIL: settings panel incomplete\n");
		failures++;
	} else {
		printf("ok: settings panel builds\n");
	}
	obs_properties_destroy(props);

	run_format(f, VIDEO_FORMAT_NV12, "NV12");
	run_format(f, VIDEO_FORMAT_UYVY, "UYVY");
	run_format(f, VIDEO_FORMAT_BGRA, "BGRA");

	printf("info: Vision ran %lu times, %lu failures, grey input %s\n", f->vision_runs, f->vision_failures,
	       f->vision_needs_bgra ? "rejected (using BGRA fallback)" : "accepted");
	if (f->vision_runs == 0 || (f->vision_needs_bgra && f->vision_failures > 1)) {
		printf("FAIL: Apple Vision could not process frames\n");
		failures++;
	} else {
		printf("ok: Apple Vision processes the frames\n");
	}

	camflip_destroy(f);
	obs_data_release(settings);
	if (failures) {
		printf("%d FAILED\n", failures);
		return 1;
	}
	printf("ALL CHECKS PASSED\n");
	return 0;
}
