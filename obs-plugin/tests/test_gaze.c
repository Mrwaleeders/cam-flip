/*
 * Behaviour checks for the eye-contact logic ("does it feel natural?"):
 *   cc -O2 -Isrc tests/test_gaze.c src/gaze.c src/warp.c -lm -o test_gaze && ./test_gaze
 */
#include "gaze.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, ...)                         \
	do {                                     \
		if (!(cond)) {                   \
			printf("FAIL: " __VA_ARGS__); \
			printf("\n");            \
			failures++;              \
		}                                \
	} while (0)

/* Builds a synthetic face. Eyes are 40x16 px ellipses; pupil_down is in eye heights. */
static cf_face face_at(float x, float y, float pupil_down, float eye_h, float yaw)
{
	cf_face f = {0};
	f.found = 1;
	f.image_width = 1920;
	f.image_height = 1080;
	f.face_width = 300;
	f.has_pose = 1;
	f.yaw_deg = yaw;
	const float cx[2] = {x - 60, x + 60};
	for (int e = 0; e < 2; e++) {
		cf_pt *pts = e == 0 ? f.left_eye : f.right_eye;
		for (int i = 0; i < 8; i++) {
			const float a = (float)i / 8.0f * 6.2831853f;
			pts[i].x = cx[e] + 20 * cosf(a);
			pts[i].y = y + eye_h / 2 * sinf(a);
		}
		cf_pt pupil = {cx[e], y + pupil_down * eye_h};
		if (e == 0) {
			f.left_count = 8;
			f.left_pupil = pupil;
		} else {
			f.right_count = 8;
			f.right_pupil = pupil;
		}
	}
	return f;
}

static float shift_px(const cf_eye *e)
{
	return sqrtf(e->sx * e->sx + e->sy * e->sy);
}

int main(void)
{
	const cf_gaze_params p = {.strength = 1.0f, .max_glance = 0.45f, .baseline = 0.0f};
	cf_gaze g;
	cf_eye out[2];
	double t = 0;
	const double dt = 1.0 / 30.0;

	/* 1. Looking into the lens: no correction. */
	cf_gaze_init(&g);
	int n = 0;
	for (int i = 0; i < 30; i++, t += dt)
		n = cf_gaze_update(&g, (cf_face[]){face_at(960, 500, 0, 16, 0)}, &p, t, out);
	CHECK(n == 0 || shift_px(&out[0]) < 0.2f, "looking at lens should give no correction");
	printf("ok: no correction when looking into the lens\n");

	/* 2. Small glance down at notes (0.25 eye heights = 4 px): corrected, upward, without a jump. */
	float prev = 0, max_step = 0;
	for (int i = 0; i < 45; i++, t += dt) {
		n = cf_gaze_update(&g, (cf_face[]){face_at(960, 500, 0.25f, 16, 0)}, &p, t, out);
		float s = n ? shift_px(&out[0]) : 0;
		max_step = fmaxf(max_step, fabsf(s - prev));
		prev = s;
	}
	CHECK(n == 2, "small glance should be corrected");
	CHECK(fabsf(prev - 4.0f) < 0.6f, "expected ~4 px correction, got %.2f", prev);
	CHECK(out[0].sy < 0, "correction should move the eyes up");
	CHECK(max_step < 1.5f, "correction jumped %.2f px in one frame", max_step);
	printf("ok: small glance corrected by %.2f px, max step %.2f px/frame\n", prev, max_step);

	/* 3. Head moving sideways at 300 px/s while glancing: the warp stays on the eye. */
	float worst = 0;
	for (int i = 0; i < 30; i++, t += dt) {
		const float x = 960 + 300.0f * (float)(i * dt);
		n = cf_gaze_update(&g, (cf_face[]){face_at(x, 500, 0.25f, 16, 0)}, &p, t, out);
		if (i > 5)
			worst = fmaxf(worst, fabsf(out[0].cx - (x - 60)));
	}
	CHECK(worst < 4.0f, "warp lagged %.1f px behind a moving head", worst);
	printf("ok: warp follows a moving head (worst lag %.1f px)\n", worst);

	/* 4. Turning the head away (yaw 40°): correction fades out, smoothly. */
	prev = shift_px(&out[0]);
	max_step = 0;
	for (int i = 0; i < 20; i++, t += dt) {
		n = cf_gaze_update(&g, (cf_face[]){face_at(1050, 500, 0.25f, 16, 40)}, &p, t, out);
		float s = n ? shift_px(&out[0]) : 0;
		max_step = fmaxf(max_step, fabsf(s - prev));
		prev = s;
	}
	CHECK(prev < 0.1f, "turned head should not be corrected (%.2f px left)", prev);
	printf("ok: head turn fades the effect out (largest step %.2f px)\n", max_step);

	/* 5. Coming back: fades back in over a few hundred ms, not instantly. */
	n = cf_gaze_update(&g, (cf_face[]){face_at(1050, 500, 0.25f, 16, 0)}, &p, t, out);
	t += dt;
	float first = n ? shift_px(&out[0]) : 0;
	for (int i = 0; i < 30; i++, t += dt)
		n = cf_gaze_update(&g, (cf_face[]){face_at(1050, 500, 0.25f, 16, 0)}, &p, t, out);
	CHECK(first < 1.0f, "effect snapped back in (%.2f px on the first frame)", first);
	CHECK(n == 2 && shift_px(&out[0]) > 3.0f, "effect should return after looking back");
	printf("ok: effect eases back in (first frame %.2f px)\n", first);

	/* 6. Looking far down (keyboard): left alone. */
	for (int i = 0; i < 30; i++, t += dt)
		n = cf_gaze_update(&g, (cf_face[]){face_at(1050, 500, 0.9f, 12, 0)}, &p, t, out);
	CHECK(n == 0 || shift_px(&out[0]) < 0.2f, "far glance should not be corrected (%.2f)", n ? shift_px(&out[0]) : 0);
	printf("ok: far-down glance is left natural\n");

	/* 7. Blink: correction drops out quickly. */
	for (int i = 0; i < 30; i++, t += dt)
		n = cf_gaze_update(&g, (cf_face[]){face_at(1050, 500, 0.25f, 16, 0)}, &p, t, out);
	for (int i = 0; i < 4; i++, t += dt)
		n = cf_gaze_update(&g, (cf_face[]){face_at(1050, 500, 0.0f, 2, 0)}, &p, t, out);
	CHECK(n == 0 || shift_px(&out[0]) < 1.0f, "blink should release the correction (%.2f)", n ? shift_px(&out[0]) : 0);
	printf("ok: blinks are left alone\n");

	/* 8. Face lost: fades out, then stops warping. */
	for (int i = 0; i < 30; i++, t += dt)
		n = cf_gaze_update(&g, &(cf_face){0}, &p, t, out);
	CHECK(n == 0, "no face should mean no warp");
	printf("ok: no face, no warp\n");

	/* 9. Calibration picks up the user's own resting pupil position. */
	cf_gaze_init(&g);
	cf_gaze_start_calibration(&g);
	float baseline = 0;
	int got = 0;
	for (int i = 0; i < 40 && !got; i++, t += dt) {
		cf_gaze_update(&g, (cf_face[]){face_at(960, 500, -0.1f, 16, 0)}, &p, t, out);
		got = cf_gaze_take_calibration(&g, &baseline);
	}
	CHECK(got && fabsf(baseline - 0.1f) < 0.01f, "calibration baseline %.3f", baseline);
	printf("ok: calibration baseline %.3f\n", baseline);

	/* 10. Noisy landmarks while sitting still (±1.5 px, like real detection): the warp holds steady. */
	cf_gaze_init(&g);
	srand(7);
	float sum = 0, sum2 = 0, ssum = 0, ssum2 = 0;
	int count = 0;
	for (int i = 0; i < 120; i++, t += dt) {
		cf_face f = face_at(960, 500, 0.25f, 16, 0);
		for (int k = 0; k < 8; k++) {
			f.left_eye[k].x += ((float)rand() / RAND_MAX - 0.5f) * 3;
			f.left_eye[k].y += ((float)rand() / RAND_MAX - 0.5f) * 3;
		}
		f.left_pupil.y += ((float)rand() / RAND_MAX - 0.5f) * 3;
		n = cf_gaze_update(&g, &f, &p, t, out);
		if (i >= 60 && n) {
			sum += out[0].cx;
			sum2 += out[0].cx * out[0].cx;
			float s = shift_px(&out[0]);
			ssum += s;
			ssum2 += s * s;
			count++;
		}
	}
	float sd = sqrtf(fmaxf(sum2 / count - (sum / count) * (sum / count), 0));
	float ssd = sqrtf(fmaxf(ssum2 / count - (ssum / count) * (ssum / count), 0));
	CHECK(sd < 0.5f && ssd < 0.3f, "jitter too high: position %.2f px, shift %.2f px", sd, ssd);
	printf("ok: steady when still (position jitter %.2f px, shift jitter %.2f px)\n", sd, ssd);

	if (failures) {
		printf("%d FAILED\n", failures);
		return 1;
	}
	printf("ALL CHECKS PASSED\n");
	return 0;
}
