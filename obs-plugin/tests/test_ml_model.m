/*
 * Checks the personal-model engine end to end on macOS with a model trained on synthetic eyes (no face
 * data): train.py --selftest --selftest-export <path> makes it, then
 *   clang -fobjc-arc -Isrc tests/test_ml_model.m src/ml_model.m src/ml_eye.c src/warp.c -framework CoreML \
 *     -framework Foundation -o test_ml_model && ./test_ml_model <path>
 *  - the model loads and keeps the input / output contract ml_model.h expects, on every compute unit,
 *  - a synthetic eye with the iris low, asked to put it 6 patch px higher, gets it there (roughly) on
 *    a 1080p frame, and nothing outside the eye changes,
 *  - and how long it takes per frame (printed; the live budget is a few ms).
 */
#import <Foundation/Foundation.h>

#include "ml_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void draw_eye(uint8_t *img, int w, float cx, float cy, float ew, float iris_dx, float iris_dy)
{
	/* matches train.py's synthetic_eye roughly, in image px: eye ew wide */
	const float s = ew / 80.0f;
	for (int y = (int)(cy - 40 * s); y < (int)(cy + 40 * s); y++)
		for (int x = (int)(cx - 64 * s); x < (int)(cx + 64 * s); x++) {
			const float u = (x - cx) / (40 * s), py = (y - cy) / s;
			const float k = fmaxf(0, 1 - u * u);
			float v = 158;
			if (py > -13 * k && py < 9 * k) {
				v = 224;
				const float r = hypotf((x - cx - iris_dx * s) / s, (y - cy - iris_dy * s) / s);
				if (r < 11)
					v = 82;
				if (r < 4.5f)
					v = 20;
			}
			img[(size_t)y * w + x] = (uint8_t)v;
		}
}

static float dark_y(const uint8_t *img, int w, float cx, float cy, float ew)
{
	double sw = 0, sy = 0;
	const float s = ew / 80.0f;
	for (int y = (int)(cy - 20 * s); y < (int)(cy + 15 * s); y++)
		for (int x = (int)(cx - 30 * s); x < (int)(cx + 30 * s); x++) {
			const double d = pow((255 - img[(size_t)y * w + x]) / 255.0, 6);
			sw += d;
			sy += d * y;
		}
	return (float)(sy / sw);
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: test_ml_model <model.mlpackage>\n");
		return 2;
	}
	int fails = 0;
	const char *names[] = {"all", "cpu", "cpu+gpu", "cpu+ane"};
	for (int cu = 0; cu < 4; cu++) {
		char err[512] = "";
		cf_ml_model *m = cf_ml_load(argv[1], (enum cf_ml_compute)cu, err, sizeof(err));
		if (!m) {
			printf("FAIL: load (%s): %s\n", names[cu], err);
			fails++;
			continue;
		}
		const int w = 1920, h = 1080;
		uint8_t *img = malloc((size_t)w * h), *orig = malloc((size_t)w * h);
		memset(img, 158, (size_t)w * h);
		const float ew = 70, ecx = 900, ecy = 520;
		draw_eye(img, w, ecx, ecy, ew, 0, 7); /* iris low: reading */
		memcpy(orig, img, (size_t)w * h);
		cf_channel ch[8];
		uint8_t *pl[1] = {img};
		const uint32_t ls[1] = {(uint32_t)w};
		cf_channels(CF_FMT_Y800, pl, ls, w, h, ch);
		cf_eye e;
		memset(&e, 0, sizeof(e));
		e.cx = ecx;
		e.cy = ecy;
		e.rx = ew * 0.8f;
		e.ry = ew * 0.45f;
		e.aperture_count = 2;
		e.aperture[0][0] = ecx - ew / 2;
		e.aperture[0][1] = ecy;
		e.aperture[1][0] = ecx + ew / 2;
		e.aperture[1][1] = ecy;
		const float s = ew / 80.0f;
		const float tgt[1][2] = {{ecx, ecy + 1 * s}}; /* iris up to 1 patch px below the corner line */
		const float alpha = 1;
		double ms = 0, total = 0, worst = 0;
		int k = 0;
		const int reps = 30;
		for (int r = 0; r < reps; r++) {
			memcpy(img, orig, (size_t)w * h);
			k = cf_ml_correct(m, &ch[0], ch, 1, 1u, &e, 1, tgt, &alpha, &ms);
			if (r > 2) { /* the first runs warm up */
				total += ms;
				worst = fmax(worst, ms);
			}
		}
		if (k != 1) {
			printf("FAIL: run (%s)\n", names[cu]);
			fails++;
		} else {
			const float before = dark_y(orig, w, ecx, ecy, ew), after = dark_y(img, w, ecx, ecy, ew);
			const float moved = (before - after) / s; /* patch px up */
			int outside = 0;
			for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++) {
					const float dx = (x + 0.5f - 0.5f - ecx) / e.rx, dy = (y - ecy) / e.ry;
					if (dx * dx + dy * dy >= 1 && img[(size_t)y * w + x] != orig[(size_t)y * w + x])
						outside++;
				}
			printf("%s: iris moved up %.1f patch px (asked 6), model %.2f ms avg, %.2f ms max\n", names[cu], moved,
			       total / (reps - 3), worst);
			if (moved < 2.5f || moved > 10) {
				printf("FAIL: the iris didn't move as asked (%s)\n", names[cu]);
				fails++;
			}
			if (outside) {
				printf("FAIL: %d pixels outside the eye changed (%s)\n", outside, names[cu]);
				fails++;
			}
		}
		free(img);
		free(orig);
		cf_ml_free(m);
	}
	if (fails) {
		printf("%d FAILED\n", fails);
		return 1;
	}
	printf("all ml_model tests passed\n");
	return 0;
}
