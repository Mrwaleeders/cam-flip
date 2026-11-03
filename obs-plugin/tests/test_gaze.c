/*
 * Behaviour checks for the eye-contact logic ("does it feel natural?"):
 *   cc -O2 -Isrc tests/test_gaze.c src/gaze.c src/warp.c -lm -o test_gaze && ./test_gaze
 *
 * Gaze offsets are in eye widths (corner to corner). The synthetic eyes are 40 px wide, so
 * 0.1 eye widths = 4 px.
 */
#include "gaze.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int failures = 0;
#define CHECK(cond, ...)                         \
	do {                                     \
		if (!(cond)) {                   \
			printf("FAIL: " __VA_ARGS__); \
			printf("\n");            \
			failures++;              \
		}                                \
	} while (0)

/*
 * Synthetic face. Eyes are 40 px wide ellipses of height eye_h, corners on the line y.
 * The pupil sits (pdx, pdy) px from the eye centre (positive pdy = down). upper_drop lowers only
 * the upper lid (as when looking down) without moving the corners.
 */
static cf_face face_px(float x, float y, float pdx, float pdy, float eye_h, float yaw, float upper_drop)
{
	cf_face f;
	memset(&f, 0, sizeof(f));
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
			const float s = sinf(a);
			pts[i].x = cx[e] + 20 * cosf(a);
			pts[i].y = y + eye_h / 2 * s;
			if (s < -1e-3f) /* upper lid */
				pts[i].y += upper_drop * -s;
		}
		cf_pt pupil = {cx[e] + pdx, y + pdy};
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

/* pupil_down in eye heights, as the original tests were written. */
static cf_face face_at(float x, float y, float pupil_down, float eye_h, float yaw)
{
	return face_px(x, y, 0, pupil_down * eye_h, eye_h, yaw, 0);
}

static float shift_px(const cf_eye *e)
{
	return sqrtf(e->sx * e->sx + e->sy * e->sy);
}

/* Uniform noise: +-lid px on every outline point, +-pupil px on the pupils. */
static void noisy(cf_face *f, float lid, float pupil)
{
	for (int k = 0; k < f->left_count; k++) {
		f->left_eye[k].x += ((float)rand() / (float)RAND_MAX - 0.5f) * 2 * lid;
		f->left_eye[k].y += ((float)rand() / (float)RAND_MAX - 0.5f) * 2 * lid;
	}
	for (int k = 0; k < f->right_count; k++) {
		f->right_eye[k].x += ((float)rand() / (float)RAND_MAX - 0.5f) * 2 * lid;
		f->right_eye[k].y += ((float)rand() / (float)RAND_MAX - 0.5f) * 2 * lid;
	}
	f->left_pupil.y += ((float)rand() / (float)RAND_MAX - 0.5f) * 2 * pupil;
	f->right_pupil.y += ((float)rand() / (float)RAND_MAX - 0.5f) * 2 * pupil;
}

/* Is the model holding (blink, half blink, closing lids or settling)? Only what the status shows is
 * public; this peeks at the state for the squint test. */
static int holding_now(const cf_gaze *g, double t)
{
	return g->blink || g->closing || g->half || t < g->settle_until;
}

static int update(cf_gaze *g, cf_face f, const cf_gaze_params *p, double t, cf_eye out[2])
{
	return cf_gaze_update(g, &f, p, t, out);
}

/* Settles on a glance and returns the shift. */
static float settle(cf_gaze *g, cf_face f, const cf_gaze_params *p, double *t, int frames, cf_eye out[2])
{
	int n = 0;
	for (int i = 0; i < frames; i++, *t += 1.0 / 30.0)
		n = cf_gaze_update(g, &f, p, *t, out);
	return n ? shift_px(&out[0]) : 0;
}

/* Grey eye image for cf_refine_pupil: skin, white sclera inside the outline, dark iris. */
static void draw_eye(uint8_t *img, int w, int h, const cf_pt *eye, int count, float ix, float iy, float ir,
		     int sclera, int iris, unsigned seed)
{
	srand(seed);
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++) {
			const float px = (float)x + 0.5f, py = (float)y + 0.5f;
			int inside = 0;
			for (int k = 0, j = count - 1; k < count; j = k++)
				if ((eye[k].y > py) != (eye[j].y > py) &&
				    px < (eye[j].x - eye[k].x) * (py - eye[k].y) / (eye[j].y - eye[k].y) + eye[k].x)
					inside = !inside;
			int v = 150; /* skin */
			if (inside) {
				const float d = sqrtf((px - ix) * (px - ix) + (py - iy) * (py - iy));
				v = d < ir * 0.35f ? iris - 40 : (d < ir ? iris : sclera);
			}
			v += rand() % 11 - 5;
			img[y * w + x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
		}
}

int main(void)
{
	const cf_gaze_params p = {.strength = 1.0f, .max_glance = 0.15f, .baseline = 0.0f};
	cf_gaze g;
	cf_eye out[2];
	double t = 0;
	const double dt = 1.0 / 30.0;

	/* 1. Looking into the lens: no correction. */
	cf_gaze_init(&g);
	int n = 0;
	for (int i = 0; i < 30; i++, t += dt)
		n = update(&g, face_at(960, 500, 0, 16, 0), &p, t, out);
	CHECK(n == 0 || shift_px(&out[0]) < 0.2f, "looking at lens should give no correction");
	printf("ok: no correction when looking into the lens\n");

	/* 2. Small glance down at notes (4 px = 0.1 eye widths): corrected, upward, without a jump.
	 * What the viewer sees is the pupil plus the shift: the correction must cancel the glance as it
	 * happens (no half-second dip and climb back), and never overshoot above the lens. */
	float prev = 0, max_step = 0, first_app = -1, worst_up = 0;
	for (int i = 0; i < 45; i++, t += dt) {
		n = update(&g, face_at(960, 500, 0.25f, 16, 0), &p, t, out);
		float s = n ? shift_px(&out[0]) : 0;
		max_step = fmaxf(max_step, fabsf(s - prev));
		prev = s;
		const float app = 4.0f + (n ? out[0].sy : 0); /* apparent pupil below the lens position, px */
		if (first_app < 0)
			first_app = app;
		worst_up = fminf(worst_up, app);
		if (i == 6)
			CHECK(app < 0.5f, "glance still showing %.2f px after 0.2 s (correction eases in too slowly)", app);
	}
	CHECK(n == 2, "small glance should be corrected");
	CHECK(fabsf(prev - 4.0f) < 0.6f, "expected ~4 px correction, got %.2f", prev);
	CHECK(out[0].sy < 0, "correction should move the eyes up");
	CHECK(first_app < 0.75f * 4.0f, "glance showed %.2f px of 4 on its first frame", first_app);
	CHECK(worst_up > -0.2f, "correction overshot above the lens by %.2f px", -worst_up);
	CHECK(max_step <= 4.0f, "correction jumped %.2f px in one frame", max_step);
	CHECK(fabsf(out[0].lid_sy - 0.45f * out[0].sy) < 0.05f, "upper lid should follow by ~0.45");
	CHECK(out[0].aperture_count == 8, "eye outline should be passed to the warp");
	{
		float ext_x = 0, ext_y = 0;
		for (int k = 0; k < out[0].aperture_count; k++) {
			ext_x = fmaxf(ext_x, fabsf(out[0].aperture[k][0] - out[0].cx));
			ext_y = fmaxf(ext_y, fabsf(out[0].aperture[k][1] - out[0].cy));
		}
		CHECK(out[0].rx >= 0.74f * 40 && out[0].rx > ext_x + 5, "rx %.1f too small", out[0].rx);
		CHECK(out[0].ry >= 1.55f * 16 && out[0].ry > ext_y + 8, "ry %.1f too small", out[0].ry);
	}
	printf("ok: small glance corrected by %.2f px, first frame shows %.2f of 4 px, max step %.2f px/frame\n", prev,
	       first_app, max_step);

	/* 3. Head moving sideways at 300 px/s while glancing: the warp stays on the eye. */
	float worst = 0;
	for (int i = 0; i < 30; i++, t += dt) {
		const float x = 960 + 300.0f * (float)(i * dt);
		n = update(&g, face_at(x, 500, 0.25f, 16, 0), &p, t, out);
		if (i > 5)
			worst = fmaxf(worst, fabsf(out[0].cx - (x - 60)));
	}
	CHECK(worst < 4.0f, "warp lagged %.1f px behind a moving head", worst);
	printf("ok: warp follows a moving head (worst lag %.1f px)\n", worst);

	/* 4. Turning the head away (yaw 40°): correction fades out, smoothly. */
	prev = shift_px(&out[0]);
	max_step = 0;
	for (int i = 0; i < 20; i++, t += dt) {
		n = update(&g, face_at(1050, 500, 0.25f, 16, 40), &p, t, out);
		float s = n ? shift_px(&out[0]) : 0;
		max_step = fmaxf(max_step, fabsf(s - prev));
		prev = s;
	}
	CHECK(prev < 0.1f, "turned head should not be corrected (%.2f px left)", prev);
	printf("ok: head turn fades the effect out (largest step %.2f px)\n", max_step);

	/* 5. Coming back: fades back in over a few hundred ms, not instantly. */
	n = update(&g, face_at(1050, 500, 0.25f, 16, 0), &p, t, out);
	t += dt;
	float first = n ? shift_px(&out[0]) : 0;
	for (int i = 0; i < 30; i++, t += dt)
		n = update(&g, face_at(1050, 500, 0.25f, 16, 0), &p, t, out);
	CHECK(first < 1.0f, "effect snapped back in (%.2f px on the first frame)", first);
	CHECK(n == 2 && shift_px(&out[0]) > 3.0f, "effect should return after looking back");
	printf("ok: effect eases back in (first frame %.2f px)\n", first);

	/* 6. Looking far down (keyboard, 0.27 eye widths): left alone. */
	for (int i = 0; i < 30; i++, t += dt)
		n = update(&g, face_at(1050, 500, 0.9f, 12, 0), &p, t, out);
	CHECK(n == 0 || shift_px(&out[0]) < 0.2f, "far glance should not be corrected (%.2f)", n ? shift_px(&out[0]) : 0);
	printf("ok: far-down glance is left natural\n");

	/* 7. Blink (changed on purpose: it used to release the correction, which made the eyes sink
	 * after every blink). A ~150 ms blink now holds the correction steady: no dip, no jump. */
	const float steady = settle(&g, face_at(1050, 500, 0.25f, 16, 0), &p, &t, 45, out);
	{
		/* Lid heights through the blink; Vision's pupil point is unreliable then (put it centred). */
		const float lids[] = {9, 2, 2, 9, 16};
		float lo = 1e9f, hi = 0;
		int saw_blink = 0;
		for (int i = 0; i < 5; i++, t += dt) {
			const float pd = lids[i] < 16 ? 0.0f : 4.0f;
			n = update(&g, face_px(1050, 500, 0, pd, lids[i], 0, 0), &p, t, out);
			cf_gaze_status st;
			cf_gaze_get_status(&g, &st);
			saw_blink |= st.blinking;
			const float s = n ? shift_px(&out[0]) : 0;
			lo = fminf(lo, s);
			hi = fmaxf(hi, s);
		}
		for (int i = 0; i < 30; i++, t += dt) {
			n = update(&g, face_at(1050, 500, 0.25f, 16, 0), &p, t, out);
			const float s = n ? shift_px(&out[0]) : 0;
			lo = fminf(lo, s);
			hi = fmaxf(hi, s);
		}
		CHECK(saw_blink, "blink should be detected");
		CHECK(lo > 0.95f * steady && hi < 1.05f * steady, "correction moved across a blink: %.2f..%.2f (steady %.2f)",
		      lo, hi, steady);
		printf("ok: 150 ms blink holds the correction (%.2f..%.2f px, steady %.2f)\n", lo, hi, steady);
	}

	/* 7b. Eyes closed for long (0.7 s): that's not a blink any more, correction fades out. */
	for (int i = 0; i < 21; i++, t += dt)
		n = update(&g, face_px(1050, 500, 0, 0, 2, 0, 0), &p, t, out);
	CHECK(n == 0 || shift_px(&out[0]) < 0.5f, "closed eyes should fade out (%.2f)", n ? shift_px(&out[0]) : 0);
	printf("ok: long eye closure fades out\n");

	/* 7c. Downward glance with lowered lids is not a blink, even for narrow eyes (calibrated open
	 * eye 0.25; the glance lowers the lids to 0.135, under the old fixed 0.14 threshold). */
	{
		cf_gaze_params pn = p;
		pn.baseline_open = 0.25f;
		cf_gaze_init(&g);
		settle(&g, face_px(960, 500, 0, 0, 10, 0, 0), &pn, &t, 30, out);
		update(&g, face_px(960, 500, 0, 2, 10, 0, 2.0f), &pn, t, out);
		t += dt;
		int blinked = 0;
		for (int i = 0; i < 45; i++, t += dt) {
			n = update(&g, face_px(960, 500, 0, 4, 10, 0, 4.6f), &pn, t, out);
			cf_gaze_status st;
			cf_gaze_get_status(&g, &st);
			blinked |= st.blinking;
		}
		const float s = n ? shift_px(&out[0]) : 0;
		CHECK(!blinked, "glance with lowered lids was read as a blink");
		CHECK(fabsf(s - 4.0f) < 0.6f, "glance with lowered lids should be corrected by ~4 px, got %.2f", s);
		printf("ok: downward glance with lowered lids is corrected, not a blink (%.2f px)\n", s);
	}

	/* 7d. The glance is measured against the eye corners: moving the upper lid does not change it. */
	{
		cf_gaze_init(&g);
		const float a = settle(&g, face_px(960, 500, 0, 4, 16, 0, 0), &p, &t, 45, out);
		float b = 0;
		for (int i = 0; i <= 30; i++, t += dt) { /* lower the upper lid by 4 px over a second */
			n = update(&g, face_px(960, 500, 0, 4, 16, 0, 4.0f * (float)i / 30.0f), &p, t, out);
			b = n ? shift_px(&out[0]) : 0;
		}
		b = settle(&g, face_px(960, 500, 0, 4, 16, 0, 4.0f), &p, &t, 20, out);
		CHECK(fabsf(a - b) < 0.15f, "lid movement changed the correction: %.2f -> %.2f px", a, b);
		printf("ok: corner-based glance ignores lid movement (%.2f vs %.2f px)\n", a, b);
	}

	/* 8. Face lost: fades out, then stops warping. */
	for (int i = 0; i < 30; i++, t += dt)
		n = cf_gaze_update(&g, &(cf_face){0}, &p, t, out);
	CHECK(n == 0, "no face should mean no warp");
	printf("ok: no face, no warp\n");

	/* 9. Calibration picks up the user's own resting pupil position and openness. */
	cf_gaze_init(&g);
	cf_gaze_start_calibration(&g, CF_CALIB_LENS);
	cf_calib_result cal = {0};
	int got = 0;
	for (int i = 0; i < 60 && !got; i++, t += dt) {
		/* For the first 0.5 s the eyes are still on the Calibrate button (far down, to the side). */
		const double since = (double)i * dt;
		if (since < 0.5)
			update(&g, face_px(960, 500, 8.0f, 6.0f, 12, 0, 3.0f), &p, t, out);
		else
			update(&g, face_px(960, 500, 0, -1.6f, 16, 0, 0), &p, t, out);
		got = cf_gaze_take_calibration(&g, &cal);
	}
	CHECK(got == 1 && cal.target == CF_CALIB_LENS && fabsf(cal.offset - 0.04f) < 0.005f &&
		      fabsf(cal.openness - 0.4f) < 0.01f,
	      "calibration offset %.3f openness %.3f", cal.offset, cal.openness);
	CHECK(fabsf(cal.side) < 0.005f, "calibration used samples from the button glance (side %.3f)", cal.side);
	printf("ok: calibration skips the first %.1f s, baseline %.3f, openness %.3f\n", CF_CALIB_WARMUP_S, cal.offset,
	       cal.openness);

	/* 9b. Calibrating with no usable face gives up instead of running (and keeping Vision busy) forever. */
	cf_gaze_init(&g);
	cf_gaze_start_calibration(&g, CF_CALIB_LENS);
	got = 0;
	for (int i = 0; i < CF_CALIB_TIMEOUT_FRAMES + 5 && !got; i++, t += dt) {
		update(&g, face_at(960, 500, -0.1f, 16, 40), &p, t, out); /* head turned */
		got = cf_gaze_take_calibration(&g, &cal);
	}
	CHECK(got == -1 && !g.calibrating, "calibration should time out (got %d, calibrating %d)", got, g.calibrating);
	printf("ok: calibration times out without a usable face\n");

	/* 9c. Two-point calibration: lens, then notes. Looking at the notes is then cancelled exactly
	 * (the corrected pupil lands where it was when looking into the lens), even with a manual
	 * limit far too small; glances well past the notes are left alone. */
	{
		cf_gaze_params pc = {.strength = 1.0f, .max_glance = 0.02f};
		cf_calib_result lens = {0}, notes = {0};
		got = 0;
		cf_gaze_init(&g);
		cf_gaze_start_calibration(&g, CF_CALIB_LENS);
		for (int i = 0; i < 60 && !got; i++, t += dt) {
			update(&g, face_px(960, 500, 0, -1.0f, 16, 0, 0), &pc, t, out);
			got = cf_gaze_take_calibration(&g, &lens);
		}
		got = 0;
		cf_gaze_start_calibration(&g, CF_CALIB_NOTES);
		for (int i = 0; i < 60 && !got; i++, t += dt) {
			update(&g, face_px(960, 500, 1.2f, 5.0f, 16, 0, 2.5f), &pc, t, out);
			got = cf_gaze_take_calibration(&g, &notes);
		}
		CHECK(got == 1 && notes.target == CF_CALIB_NOTES, "notes calibration failed");
		pc.baseline = lens.offset;
		pc.baseline_open = lens.openness;
		pc.has_notes = 1;
		pc.notes_offset = notes.offset;
		pc.notes_side = notes.side;
		CHECK(fabsf(notes.side - 0.03f) < 0.005f, "notes side %.3f, expected 0.03", notes.side);
		const float glance_px = (pc.baseline - pc.notes_offset) * 40.0f; /* 6 px */
		settle(&g, face_px(960, 500, 1.2f, 5.0f, 16, 0, 2.5f), &pc, &t, 60, out);
		const float notes_shift = -out[0].sy;
		const float corrected_y = 500 + 5.0f + out[0].sy;
		CHECK(fabsf(-out[0].sy - glance_px) < 0.25f && fabsf(corrected_y - (500 - 1.0f)) < 0.25f,
		      "notes glance %.2f px, corrected by %.2f px (pupil at %.2f, lens %.2f)", glance_px, -out[0].sy,
		      corrected_y, 500 - 1.0f);
		const float far = settle(&g, face_px(960, 500, 1.2f, 1.0f + 2.2f * glance_px, 16, 0, 4), &pc, &t, 45, out);
		CHECK(far < 0.2f, "glance at 2.2x the notes should be left alone (%.2f px)", far);
		printf("ok: two-point calibration cancels the notes glance (%.2f px of %.2f), ignores far glances\n",
		       notes_shift, glance_px);
	}

	/* 10. Noisy landmarks while sitting still (±1.5 px, like real detection): the warp holds steady. */
	cf_gaze_init(&g);
	srand(7);
	double sum = 0, sum2 = 0, ssum = 0, ssum2 = 0;
	int count = 0;
	for (int i = 0; i < 120; i++, t += dt) {
		cf_face f = face_at(960, 500, 0.25f, 16, 0);
		for (int k = 0; k < 8; k++) {
			f.left_eye[k].x += ((float)rand() / (float)RAND_MAX - 0.5f) * 3;
			f.left_eye[k].y += ((float)rand() / (float)RAND_MAX - 0.5f) * 3;
		}
		f.left_pupil.y += ((float)rand() / (float)RAND_MAX - 0.5f) * 3;
		n = cf_gaze_update(&g, &f, &p, t, out);
		if (i >= 60 && n) {
			sum += out[0].cx - 900.0;
			sum2 += (out[0].cx - 900.0) * (out[0].cx - 900.0);
			float s = shift_px(&out[0]);
			ssum += s;
			ssum2 += s * s;
			count++;
		}
	}
	float sd = (float)sqrt(fmax(sum2 / count - (sum / count) * (sum / count), 0));
	float ssd = (float)sqrt(fmax(ssum2 / count - (ssum / count) * (ssum / count), 0));
	CHECK(sd < 0.5f && ssd < 0.3f, "jitter too high: position %.2f px, shift %.2f px", sd, ssd);
	printf("ok: steady when still (position jitter %.2f px, shift jitter %.2f px)\n", sd, ssd);

	/* 11. Background tracking: analysis runs 2 frames behind the shown frame while the head moves.
	 * Prediction keeps the warp closer to the eye than using the analysed position as is. */
	{
		cf_gaze_init(&g);
		const double lag = 2 * dt;
		double e_pred = 0, e_plain = 0;
		int samples = 0;
		for (int i = 0; i < 60; i++, t += dt) {
			const float v = i < 15 ? 0.0f : 250.0f; /* px/s */
			const float x = 960 + v * (float)((i - 15) * dt > 0 ? (i - 15) * dt : 0);
			n = update(&g, face_at(x, 500, 0.25f, 16, 0), &p, t, out);
			cf_eye pred[2];
			const int np = cf_gaze_predict(&g, &p, t + lag, pred);
			const float x_now = x + v * (float)lag - 60; /* where the left eye is in the shown frame */
			if (i >= 30 && n && np) {
				e_pred += fabsf(pred[0].cx - x_now);
				e_plain += fabsf(out[0].cx - x_now);
				samples++;
			}
			if (i == 59) {
				CHECK(fabsf(pred[0].sy - out[0].sy) < 1e-4f, "prediction should keep the correction");
				CHECK(fabsf((pred[0].aperture[0][0] - pred[0].cx) - (out[0].aperture[0][0] - out[0].cx)) < 1e-3f,
				      "outline should move with the eye");
				cf_eye far_pred[2], max_pred[2];
				cf_gaze_predict(&g, &p, t + 5.0, far_pred);
				cf_gaze_predict(&g, &p, t + 0.1, max_pred);
				CHECK(fabsf(far_pred[0].cx - max_pred[0].cx) < 1e-3f && far_pred[0].cx - out[0].cx < 0.75f * 40,
				      "extrapolation must be clamped to 0.1 s (moved %.1f px)", far_pred[0].cx - out[0].cx);
			}
		}
		e_pred /= samples;
		e_plain /= samples;
		CHECK(e_pred < 0.5 * e_plain, "prediction error %.2f px vs %.2f px without", e_pred, e_plain);
		printf("ok: prediction during head motion (error %.2f px vs %.2f px without)\n", e_pred, e_plain);

		/* The head stops dead: the prediction must stop with it within a few frames instead of
		 * carrying the warp past the eye (it used the 1 Hz One Euro derivative, ~0.4 s of overshoot). */
		{
			const float x_stop = 960 + 250.0f * (float)(45 * dt);
			float late = 0;
			for (int i = 0; i < 20; i++, t += dt) {
				update(&g, face_at(x_stop, 500, 0.25f, 16, 0), &p, t, out);
				cf_eye pred[2];
				if (cf_gaze_predict(&g, &p, t + lag, pred) && i >= 5)
					late = fmaxf(late, fabsf(pred[0].cx - (x_stop - 60)));
			}
			CHECK(late < 2.0f, "prediction still %.1f px past the eye 5 frames after the head stopped", late);
			printf("ok: prediction stops with the head (%.2f px after 5 frames)\n", late);
		}

		/* Still head with noisy landmarks: prediction adds no visible jitter. */
		srand(11);
		double s1 = 0, s2 = 0;
		int c = 0;
		for (int i = 0; i < 90; i++, t += dt) {
			cf_face f = face_at(960, 500, 0.25f, 16, 0);
			for (int k = 0; k < 8; k++)
				f.left_eye[k].x += ((float)rand() / (float)RAND_MAX - 0.5f) * 3;
			cf_gaze_update(&g, &f, &p, t, out);
			cf_eye pred[2];
			if (i >= 45 && cf_gaze_predict(&g, &p, t + lag, pred)) {
				s1 += pred[0].cx - 900.0;
				s2 += (pred[0].cx - 900.0) * (pred[0].cx - 900.0);
				c++;
			}
		}
		const float psd = (float)sqrt(fmax(s2 / c - (s1 / c) * (s1 / c), 0));
		CHECK(psd < 0.5f, "prediction jitter %.2f px when still", psd);
		printf("ok: prediction steady when still (jitter %.2f px)\n", psd);
	}

	/* 12. Reading notes spread along the top of the screen: sideways sweeps (±8 px over 1.5 s, then
	 * a quick return) are damped so the eyes barely move, the gaze stays at the lens height, and
	 * nothing jumps. */
	{
		cf_gaze_params pr = {.strength = 1.0f, .max_glance = 0.15f, .baseline = 0.0f, .baseline_open = 0.4f,
				     .has_notes = 1, .notes_offset = -0.08f, .notes_side = 0.0f};
		cf_gaze_init(&g);
		float raw_lo = 1e9f, raw_hi = -1e9f, app_lo = 1e9f, app_hi = -1e9f, max_dy = 0;
		float raw_step = 0, app_step = 0, prev_raw = 0, prev_app = 0;
		int frame = 0;
		for (int line = 0; line < 4; line++) {
			for (int i = 0; i < 47; i++, frame++, t += dt) {
				float pdx;
				if (i < 45)
					pdx = -8.0f + 16.0f * (float)i / 44.0f; /* sweep along the line */
				else
					pdx = 8.0f - 8.0f * (float)(i - 44); /* return saccade over 2 frames */
				const float pdy = 3.2f + 0.4f * sinf((float)frame * 0.3f);
				n = update(&g, face_px(960, 500, pdx, pdy, 16, 0, 1.0f), &pr, t, out);
				const float raw_x = 900 + pdx;
				const float app_x = raw_x + (n ? out[0].sx : 0);
				const float app_y = 500 + pdy + (n ? out[0].sy : 0);
				if (line >= 1) {
					raw_lo = fminf(raw_lo, raw_x);
					raw_hi = fmaxf(raw_hi, raw_x);
					app_lo = fminf(app_lo, app_x);
					app_hi = fmaxf(app_hi, app_x);
					max_dy = fmaxf(max_dy, fabsf(app_y - 500));
					raw_step = fmaxf(raw_step, fabsf(raw_x - prev_raw));
					app_step = fmaxf(app_step, fabsf(app_x - prev_app));
				}
				prev_raw = raw_x;
				prev_app = app_x;
				CHECK(fabsf(out[0].sx) <= 0.3f * out[0].rx + 1e-3f, "sideways shift not clamped");
			}
		}
		const float reduction = 1.0f - (app_hi - app_lo) / (raw_hi - raw_lo);
		CHECK(reduction > 0.5f, "reading sweep only reduced by %.0f%%", reduction * 100);
		CHECK(reduction < 0.9f, "reading sweep frozen (%.0f%% reduced): some movement should stay", reduction * 100);
		CHECK(max_dy < 1.0f, "gaze should stay at lens height while reading (off by %.2f px)", max_dy);
		CHECK(app_step < 0.5f * raw_step && app_step < 3.5f, "visible jump %.2f px/frame (raw %.2f)", app_step,
		      raw_step);
		printf("ok: reading sweeps damped by %.0f%% (%.1f -> %.1f px), height off by %.2f px, max step %.2f px "
		       "(raw %.2f)\n",
		       reduction * 100, raw_hi - raw_lo, app_hi - app_lo, max_dy, app_step, raw_step);

		/* A big sideways look (half an eye width) is not reading: fades out. */
		const float s = settle(&g, face_px(960, 500, 22.0f, 3.2f, 16, 0, 1.0f), &pr, &t, 30, out);
		CHECK(s < 0.2f, "large side glance should fade out (%.2f px left)", s);
		printf("ok: large side glance fades out\n");
	}

	/* 12b. Looking sideways out of the notes band: the damping lets go with the saccade itself, so
	 * the eye lands where it really is instead of gliding there afterwards. */
	{
		cf_gaze_params pr = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
				     .notes_offset = -0.08f};
		cf_gaze_init(&g);
		settle(&g, face_px(960, 500, 4, 3.2f, 16, 0, 1.0f), &pr, &t, 60, out);
		float lo = 1e9f, hi = -1e9f;
		for (int i = 0; i < 10; i++, t += dt) {
			n = update(&g, face_px(960, 500, 20, 3.2f, 16, 0, 1.0f), &pr, t, out);
			const float app = 20 + (n ? out[0].sx : 0);
			lo = fminf(lo, app);
			hi = fmaxf(hi, app);
		}
		CHECK(hi - lo < 0.5f, "eye glided %.2f px after a sideways saccade", hi - lo);
		printf("ok: sideways look out of the notes band lands without a glide (%.2f px)\n", hi - lo);
	}

	/* 12c. Robustness while reading: a 2-frame detector dropout holds the correction (no dip), and a
	 * NaN pose or pupil from the detector does not poison the filters. */
	{
		cf_gaze_init(&g);
		const float steady2 = settle(&g, face_at(960, 500, 0.25f, 16, 0), &p, &t, 45, out);
		float lo = 1e9f;
		for (int i = 0; i < 2; i++, t += dt) {
			n = cf_gaze_update(&g, &(cf_face){0}, &p, t, out);
			lo = fminf(lo, n ? shift_px(&out[0]) : 0);
		}
		for (int i = 0; i < 20; i++, t += dt) {
			n = update(&g, face_at(960, 500, 0.25f, 16, 0), &p, t, out);
			lo = fminf(lo, n ? shift_px(&out[0]) : 0);
		}
		CHECK(lo > 0.95f * steady2, "2-frame dropout dipped the correction to %.2f (steady %.2f)", lo, steady2);
		cf_face bad = face_at(960, 500, 0.25f, 16, 0);
		bad.yaw_deg = NAN;
		update(&g, bad, &p, t, out);
		t += dt;
		bad = face_at(960, 500, 0.25f, 16, 0);
		bad.right_pupil.y = NAN;
		update(&g, bad, &p, t, out);
		t += dt;
		const float after = settle(&g, face_at(960, 500, 0.25f, 16, 0), &p, &t, 20, out);
		CHECK(isfinite(g.factor) && isfinite(out[0].sy) && fabsf(after - steady2) < 0.2f,
		      "NaN input poisoned the state (shift %.2f, factor %.2f)", after, g.factor);
		printf("ok: dropout held (min %.2f of %.2f px), NaN inputs ignored\n", lo, steady2);
	}

	/* 14. Step glances lens -> notes -> lens (calibrated notes 0.08 eye widths = 3.2 px under the lens,
	 * the upper lid following the eye down). What the other person sees is the measured pupil plus the
	 * shift: it must stay at the lens on the way down (no dip) and on the way back (no overshoot above
	 * the camera), from the first frame on. Also with the glance spread over two frames. */
	{
		const float gl = 0.08f, gpx = gl * 40.0f;
		const cf_gaze_params pg = {.strength = 1.0f, .max_glance = 0.15f, .baseline = 0.0f, .baseline_open = 0.4f,
					   .has_notes = 1, .notes_offset = -gl, .notes_open = 0.35f};
		for (int ramp = 0; ramp < 2; ramp++)
			for (int lid = 0; lid < 2; lid++) {
				cf_gaze_init(&g);
				const float drop = lid ? 3.0f : 0.0f; /* 3 px: fast enough to look like a closing blink */
				settle(&g, face_px(960, 500, 0, 0, 16, 0, 0), &pg, &t, 45, out);
				float first_dn = 0, first_up = 0, worst_dn = 0, worst_back = 0;
				for (int cyc = 0; cyc < 3; cyc++) {
					for (int i = 0; i < 20; i++, t += dt) {
						const float k = (ramp && i == 0) ? 0.5f : 1.0f;
						n = update(&g, face_px(960, 500, 0, k * gpx, 16, 0, k * drop), &pg, t, out);
						const float vis = k * gpx + (n ? out[0].sy : 0); /* below the lens, px */
						if (i == 0)
							first_dn = fmaxf(first_dn, fabsf(vis));
						else
							worst_dn = fmaxf(worst_dn, fabsf(vis));
					}
					for (int i = 0; i < 20; i++, t += dt) {
						const float k = (ramp && i == 0) ? 0.5f : 0.0f;
						n = update(&g, face_px(960, 500, 0, k * gpx, 16, 0, k * drop), &pg, t, out);
						const float vis = k * gpx + (n ? out[0].sy : 0);
						if (i == 0)
							first_up = fmaxf(first_up, fabsf(vis));
						else
							worst_back = fmaxf(worst_back, fabsf(vis));
					}
				}
				CHECK(worst_dn < 0.5f && worst_back < 0.5f,
				      "step glance (lid %d, ramp %d): eyes off the lens by %.2f px going down, %.2f px coming back",
				      lid, ramp, worst_dn, worst_back);
				CHECK(first_dn < 0.5f && first_up < 0.5f,
				      "step glance (lid %d, ramp %d): first frame off by %.2f px down, %.2f px back", lid, ramp,
				      first_dn, first_up);
				printf("ok: step glance (lid drop %d, 2-frame %d): first frame %.2f / %.2f px, then within %.2f / "
				       "%.2f px of the lens\n",
				       lid, ramp, first_dn, first_up, worst_dn, worst_back);
			}
		/* A real blink right after landing on the notes still holds the correction. */
		cf_gaze_init(&g);
		const float steady3 = settle(&g, face_px(960, 500, 0, gpx, 16, 0, 2.0f), &pg, &t, 45, out);
		const float lids[] = {9, 2, 2, 9, 14};
		float lo = 1e9f, hi = 0;
		for (int i = 0; i < 5; i++, t += dt) {
			n = update(&g, face_px(960, 500, 0, lids[i] < 14 ? 0.0f : gpx, lids[i], 0, 0), &pg, t, out);
			lo = fminf(lo, n ? shift_px(&out[0]) : 0);
			hi = fmaxf(hi, n ? shift_px(&out[0]) : 0);
		}
		CHECK(lo > 0.95f * steady3 && hi < 1.05f * steady3, "blink at the notes moved the correction %.2f..%.2f (%.2f)",
		      lo, hi, steady3);
		printf("ok: blink at the notes still holds (%.2f..%.2f px, steady %.2f)\n", lo, hi, steady3);
	}

	/* 15. Smiling or squinting while reading lowers the lids a lot: with the notes openness known
	 * (and also without it) the correction stays on. A far glance whose pupil under-reads but whose lids
	 * drop well past the notes is still released. */
	{
		const float gl = 0.08f, gpx = gl * 40.0f;
		for (int known = 0; known < 2; known++) {
			const cf_gaze_params pq = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
						   .notes_offset = -gl, .notes_open = known ? 0.35f : 0.0f};
			cf_gaze_init(&g);
			settle(&g, face_px(960, 500, 0, gpx, 16, 0, 2.0f), &pq, &t, 45, out);
			float lo = 1e9f;
			for (int i = 0; i <= 15; i++, t += dt) { /* lids from 14 px to 9 px over half a second */
				n = update(&g, face_px(960, 500, 0, gpx, 16, 0, 2.0f + 5.0f * (float)i / 15.0f), &pq, t, out);
				lo = fminf(lo, n ? shift_px(&out[0]) : 0);
			}
			for (int i = 0; i < 30; i++, t += dt) {
				n = update(&g, face_px(960, 500, 0, gpx, 16, 0, 7.0f), &pq, t, out);
				lo = fminf(lo, n ? shift_px(&out[0]) : 0);
			}
			CHECK(lo > 0.95f * gpx, "squint while reading (notes openness %s) dropped the correction to %.2f px",
			      known ? "known" : "unknown", lo);
			printf("ok: squint while reading keeps the correction (notes openness %s, min %.2f of %.2f px)\n",
			       known ? "known" : "unknown", lo, gpx);
		}
		const cf_gaze_params pq = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
					   .notes_offset = -gl, .notes_open = 0.35f};
		cf_gaze_init(&g);
		settle(&g, face_px(960, 500, 0, gpx, 16, 0, 2.0f), &pq, &t, 45, out);
		/* Pupil reads 0.12 eye widths, just past the 0.112 limit (alone that would keep ~94% of the
		 * correction), but the lids are down to 8 px: a keyboard glance the pupil under-reads.
		 * (Changed from 0.10: the lid cue now only starts past 1.05 x the limit, see 15b.)
		 * (Settles over 2 s instead of 1 s: lids dropping fast to r 0.5 with the pupil barely moving
		 * look exactly like a half blink, so they are held for 0.5 s and then eased off, see 19/20.) */
		const float s = settle(&g, face_px(960, 500, 0, 4.8f, 16, 0, 8.0f), &pq, &t, 60, out);
		CHECK(s < 0.2f, "far glance with the lids well down should be left alone (%.2f px)", s);
		printf("ok: far glance read from the lids is released (%.2f px)\n", s);
	}

	/* 15b. Smiling while reading a line a quarter below the notes point (1.25 x the notes glance,
	 * 0.9 x the limit): the lids drop a lot, the correction must stay. */
	{
		const float gl = 0.08f, lpx = 1.25f * gl * 40.0f;
		for (int known = 0; known < 2; known++) {
			const cf_gaze_params pq = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
						   .notes_offset = -gl, .notes_open = known ? 0.35f : 0.0f};
			cf_gaze_init(&g);
			settle(&g, face_px(960, 500, 0, lpx, 16, 0, 2.5f), &pq, &t, 45, out);
			float lo = 1e9f;
			for (int i = 0; i <= 10; i++, t += dt) { /* smile: lids from 13.5 px to 8 px in a third of a second */
				n = update(&g, face_px(960, 500, 0, lpx, 16, 0, 2.5f + 5.5f * (float)i / 10.0f), &pq, t, out);
				lo = fminf(lo, n ? shift_px(&out[0]) : 0);
			}
			for (int i = 0; i < 30; i++, t += dt) {
				n = update(&g, face_px(960, 500, 0, lpx, 16, 0, 8.0f), &pq, t, out);
				lo = fminf(lo, n ? shift_px(&out[0]) : 0);
			}
			CHECK(lo > 0.95f * lpx, "smile 1.25x below the notes (openness %s) kept only %.0f%% of the correction",
			      known ? "known" : "unknown", 100.0f * lo / lpx);
			printf("ok: smile 1.25x below the notes keeps %.0f%% of the correction (notes openness %s)\n",
			       100.0f * lo / lpx, known ? "known" : "unknown");
		}
	}

	/* 16. Keyboard and back: the far fade follows the gaze directly. Landing on the keyboard, the eye
	 * stays where it lands (no slide down afterwards); back on the notes, it is at the lens at once
	 * (no slide up over the next ~0.7 s). The visible position never moves against the gaze. */
	{
		const float gl = 0.08f, gpx = gl * 40.0f, kpx = 0.3f * 40.0f;
		const cf_gaze_params pk = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
					   .notes_offset = -gl, .notes_open = 0.35f};
		cf_gaze_init(&g);
		settle(&g, face_px(960, 500, 0, gpx, 16, 0, 2.0f), &pk, &t, 45, out);
		float k_lo = 1e9f, k_hi = -1e9f, back = 0;
		for (int i = 0; i < 30; i++, t += dt) {
			n = update(&g, face_px(960, 500, 0, kpx, 16, 0, 6.0f), &pk, t, out);
			const float vis = kpx + (n ? out[0].sy : 0);
			if (i >= 1) {
				k_lo = fminf(k_lo, vis);
				k_hi = fmaxf(k_hi, vis);
			}
		}
		for (int i = 0; i < 30; i++, t += dt) {
			n = update(&g, face_px(960, 500, 0, gpx, 16, 0, 2.0f), &pk, t, out);
			back = fmaxf(back, fabsf(gpx + (n ? out[0].sy : 0)));
		}
		CHECK(k_hi - k_lo < 0.3f, "eye slid %.2f px after landing on the keyboard", k_hi - k_lo);
		CHECK(back < 0.5f, "back on the notes, eye off the lens by up to %.2f px", back);
		printf("ok: keyboard and back: %.2f px slide at the keyboard, back on the notes within %.2f px\n",
		       k_hi - k_lo, back);

		/* Sweeping slowly from the notes past the limit: visible position only moves down. */
		cf_gaze_init(&g);
		settle(&g, face_px(960, 500, 0, gpx, 16, 0, 2.0f), &pk, &t, 45, out);
		float prev_vis = -1e9f, against = 0;
		for (int i = 0; i <= 60; i++, t += dt) {
			const float pd = gpx + (kpx - gpx) * (float)i / 60.0f;
			n = update(&g, face_px(960, 500, 0, pd, 16, 0, 2.0f), &pk, t, out);
			const float vis = pd + (n ? out[0].sy : 0);
			if (i > 0)
				against = fmaxf(against, prev_vis - vis);
			prev_vis = vis;
		}
		CHECK(against < 0.05f, "visible eye moved up %.2f px while the gaze moved down", against);
		printf("ok: fading out past the notes is monotonic (%.3f px against the gaze)\n", against);
	}

	/* 17. Broken settings (NaN/inf) never poison the state, and a notes calibration that isn't below
	 * the lens is ignored (falls back to the manual limit) and reported as invalid. */
	{
		cf_gaze_init(&g);
		const float good = settle(&g, face_at(960, 500, 0.25f, 16, 0), &p, &t, 45, out);
		const float bad_vals[] = {NAN, INFINITY, -INFINITY};
		for (int k = 0; k < 3; k++) {
			cf_gaze_params pb = p;
			pb.baseline = bad_vals[k];
			pb.strength = bad_vals[k];
			pb.baseline_open = bad_vals[k];
			pb.max_glance = bad_vals[k];
			pb.has_notes = 1;
			pb.notes_offset = bad_vals[k];
			pb.notes_side = bad_vals[k];
			pb.notes_open = bad_vals[k];
			for (int i = 0; i < 3; i++, t += dt) {
				n = update(&g, face_at(960, 500, 0.25f, 16, 0), &pb, t, out);
				cf_eye pr[2];
				const int np = cf_gaze_predict(&g, &pb, t + dt, pr);
				CHECK(!n || (isfinite(out[0].sx) && isfinite(out[0].sy)), "NaN setting gave a NaN shift");
				CHECK(!np || (isfinite(pr[0].sy) && isfinite(pr[0].cx)), "NaN setting gave a NaN prediction");
			}
		}
		const float after = settle(&g, face_at(960, 500, 0.25f, 16, 0), &p, &t, 30, out);
		CHECK(isfinite((float)g.down.value) && isfinite(g.factor) && fabsf(after - good) < 0.2f,
		      "NaN settings poisoned the state (shift %.2f, was %.2f)", after, good);
		printf("ok: NaN/inf settings ignored (shift %.2f px after, %.2f before)\n", after, good);

		/* Notes very close under the camera (0.015 eye widths, ~0.6 px on a 40 px eye) are accepted. */
		CHECK(cf_gaze_notes_valid(0.0f, -0.08f) && cf_gaze_notes_valid(0.0f, -0.015f) &&
			      !cf_gaze_notes_valid(0.0f, 0.02f) && !cf_gaze_notes_valid(0.0f, -0.005f) &&
			      !cf_gaze_notes_valid(NAN, -0.08f),
		      "notes validity check");
		/* Reversed notes (above the lens) behave exactly like no notes. */
		cf_gaze g2;
		cf_gaze_params prv = p;
		prv.has_notes = 1;
		prv.notes_offset = 0.03f;
		prv.notes_side = 0.2f;
		cf_gaze_init(&g);
		cf_gaze_init(&g2);
		float diff = 0;
		cf_eye out2[2];
		for (int i = 0; i < 40; i++, t += dt) {
			const float pd = i < 20 ? 0.25f : 0.9f;
			const int n1 = update(&g, face_at(960, 500, pd, 16, 0), &prv, t, out);
			const int n2 = update(&g2, face_at(960, 500, pd, 16, 0), &p, t, out2);
			diff = fmaxf(diff, fabsf((n1 ? out[0].sy : 0) - (n2 ? out2[0].sy : 0)));
		}
		CHECK(diff < 1e-4f, "reversed notes calibration changed the output by %.3f px", diff);
		printf("ok: reversed notes calibration is ignored and reported invalid\n");
	}

	/* 18. Blink while reading where Vision's pupil point slides ~1 px down with the closing lids (it
	 * follows the upper lid). It must stay a blink: the correction is held all the way through, with
	 * no frames of zero correction and no snap back afterwards. First lid frame 11 px (still fairly
	 * open: the slide may pass as a glance for that frame) or 9 px (already closing: held). */
	for (int sc = 0; sc < 4; sc++) {
		const float gl = sc & 1 ? 0.08f : 0.04f, gpx = gl * 40.0f, h1 = sc & 2 ? 9.0f : 11.0f;
		const cf_gaze_params pb = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
					   .notes_offset = -gl, .notes_open = 0.35f};
		cf_gaze_init(&g);
		const float st = settle(&g, face_px(960, 500, 0, gpx, 16, 0, 2.0f), &pb, &t, 45, out);
		const float lids[] = {h1, 5, 2, 2, 8, 14};
		float lo = 1e9f, hi = 0, after_lo = 1e9f, after_hi = 0;
		int zero = 0;
		for (int i = 0; i < 6 + 20; i++, t += dt) {
			const float h = i < 6 ? lids[i] : 14.0f;
			n = update(&g, face_px(960, 500, 0, h < 14 ? gpx + 1.0f : gpx, h, 0, 0), &pb, t, out);
			const float s = n ? shift_px(&out[0]) : 0;
			zero += s < 0.5f * st;
			lo = fminf(lo, s);
			hi = fmaxf(hi, s);
			if (i >= 5) {
				after_lo = fminf(after_lo, s);
				after_hi = fmaxf(after_hi, s);
			}
		}
		CHECK(zero == 0 && lo > 0.95f * st, "blink with sliding pupil (notes %.2f, lids %.0f px): %d frames lost the "
						     "correction, min %.2f of %.2f px",
		      gl, h1, zero, lo, st);
		CHECK(hi < 1.1f * st, "blink with sliding pupil (notes %.2f, lids %.0f px): over-corrected to %.2f px (%.2f)",
		      gl, h1, hi, st);
		CHECK(after_lo > st - 0.1f && after_hi < st + 0.1f,
		      "after the blink the correction moved %.2f..%.2f px (steady %.2f): a snap", after_lo, after_hi, st);
		printf("ok: blink with sliding pupil (notes %.2f, first lid frame %.0f px): correction %.2f..%.2f px of %.2f, "
		       "%d frames lost\n",
		       gl, h1, lo, hi, st, zero);
	}

	/* 19. Half blinks while reading: the lids close only to r 0.42 or 0.5 (just above the blink
	 * threshold) and reopen, while Vision's pupil point slides 0, 1 or 2 px down with the lid. Reading
	 * normally (lids 14 px) or smiling (lids 11 px). The correction must not drop (no frames of lost
	 * correction, no snap back) and must not overshoot. */
	{
		const float gl = 0.04f, gpx = gl * 40.0f;
		const cf_gaze_params ph = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
					   .notes_offset = -gl, .notes_open = 0.35f};
		float worst_lo = 1e9f, worst_hi = 0;
		for (int smile = 0; smile < 2; smile++)
			for (int depth = 0; depth < 2; depth++)
				for (int slide = 0; slide <= 2; slide++) {
					const float base = smile ? 11.0f : 14.0f, dh = (depth ? 0.5f : 0.42f) * 16.0f;
					const float lids[] = {0.5f * (base + dh), dh, dh, dh, dh, 0.5f * (base + dh)};
					cf_gaze_init(&g);
					const float st = settle(&g, face_px(960, 500, 0, gpx, base, 0, 0), &ph, &t, 60, out);
					float lo = 1e9f, hi = 0;
					for (int i = 0; i < 6 + 30; i++, t += dt) {
						const float h = i < 6 ? lids[i] : base;
						const float pd = gpx + (float)slide * (base - h) / (base - dh);
						n = update(&g, face_px(960, 500, 0, pd, h, 0, 0), &ph, t, out);
						const float s = n ? shift_px(&out[0]) : 0;
						lo = fminf(lo, s);
						hi = fmaxf(hi, s);
					}
					CHECK(lo > 0.9f * st && hi < 1.1f * st,
					      "half blink to r %.2f (%s, %d px slide): correction %.2f..%.2f px (steady %.2f)",
					      depth ? 0.5 : 0.42, smile ? "smiling" : "reading", slide, lo, hi, st);
					worst_lo = fminf(worst_lo, lo / st);
					worst_hi = fmaxf(worst_hi, hi / st);
				}
		printf("ok: half blinks (r 0.42/0.5, 0-2 px slide, smiling or not): correction %.0f%%..%.0f%% of steady\n",
		       100 * worst_lo, 100 * worst_hi);

		/* Sustained squint: lids drop quickly to r 0.55 and stay there for 2 s while reading. Held at
		 * first (could be a half blink), then measured normally: still corrected, not stuck. */
		cf_gaze_init(&g);
		const float st = settle(&g, face_px(960, 500, 0, gpx, 14, 0, 0), &ph, &t, 60, out);
		float lo = 1e9f, hi = 0;
		int held = 0;
		for (int i = 0; i < 60; i++, t += dt) {
			n = update(&g, face_px(960, 500, 0, gpx, i == 0 ? 11.4f : 8.8f, 0, 0), &ph, t, out);
			const float s = n ? shift_px(&out[0]) : 0;
			lo = fminf(lo, s);
			hi = fmaxf(hi, s);
			if (i == 59)
				held = holding_now(&g, t);
		}
		CHECK(lo > 0.9f * st && hi < 1.1f * st && !held, "sustained squint: correction %.2f..%.2f (steady %.2f), held %d",
		      lo, hi, st, held);
		/* And once it has settled, a glance at the lens is followed (not frozen). */
		float lens = 9;
		for (int i = 0; i < 5; i++, t += dt) {
			n = update(&g, face_px(960, 500, 0, 0, 8.8f, 0, 0), &ph, t, out);
			lens = n ? shift_px(&out[0]) : 0;
		}
		CHECK(lens < 0.3f, "after a long squint the gaze is frozen (%.2f px left at the lens)", lens);
		printf("ok: sustained squint at r 0.55 for 2 s: correction %.2f..%.2f px of %.2f, then follows the gaze\n", lo,
		       hi, st);

		/* A keyboard glance that drops the lids deep (to r 0.44, like a half blink) is not held: the
		 * pupil moves far more than a lid could drag it, so the correction lets go at once. */
		cf_gaze_init(&g);
		settle(&g, face_px(960, 500, 0, gpx, 14, 0, 0), &ph, &t, 60, out);
		float kb = 0;
		for (int i = 0; i < 15; i++, t += dt) {
			n = update(&g, face_px(960, 500, 0, 12.0f, 7.0f, 0, 0), &ph, t, out);
			if (i >= 1)
				kb = fmaxf(kb, n ? shift_px(&out[0]) : 0);
		}
		CHECK(kb < 0.2f, "keyboard glance with deep lids was held (%.2f px corrected after the first frame)", kb);
		printf("ok: keyboard glance with the lids at r 0.44 is not held (%.2f px after the first frame)\n", kb);
	}

	/* 20. A half-blink hold that times out (lids stay at r 0.5, the pupil under-reads a glance by
	 * reading only 3 px further down) hands over gently: the eyes are still, so the output may not
	 * jump when the hold lets go and the measurement takes over. */
	{
		const float gl = 0.04f, gpx = gl * 40.0f;
		const cf_gaze_params pt = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
					   .notes_offset = -gl, .notes_open = 0.35f};
		cf_gaze_init(&g);
		settle(&g, face_px(960, 500, 0, gpx, 14, 0, 0), &pt, &t, 60, out);
		float prev_s = -1, worst_step = 0, last = 0;
		int was_half = 0, released_at = -1;
		for (int i = 0; i < 60; i++, t += dt) {
			n = update(&g, face_px(960, 500, 0, gpx + 3.0f, 8.0f, 0, 0), &pt, t, out);
			const float s = n ? shift_px(&out[0]) : 0;
			was_half |= g.half;
			if (released_at < 0 && was_half && !g.half)
				released_at = i;
			if (i >= 2 && prev_s >= 0) /* from the third frame on the lids and pupil are still */
				worst_step = fmaxf(worst_step, fabsf(s - prev_s));
			prev_s = s;
			last = s;
		}
		CHECK(was_half && released_at > 0, "the lid drop should be held as a half blink, then released");
		CHECK(worst_step < 0.35f, "hold timeout changed the output by %.2f px in one frame (eyes still)", worst_step);
		printf("ok: half-blink timeout hands over gently (largest step %.2f px/frame, released at frame %d, ends "
		       "at %.2f px)\n",
		       worst_step, released_at, last);
	}

	/* 21. Half blinks with noisy lid landmarks (+-1 px on every outline point, fixed seed): still
	 * caught (held), so the correction neither drops nor overshoots; and smiling steadily with the
	 * same noise rarely starts a hold. */
	{
		const float gl = 0.04f, gpx = gl * 40.0f;
		const cf_gaze_params pn = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
					   .notes_offset = -gl, .notes_open = 0.35f};
		srand(21);
		int blinks = 0, caught = 0, frames = 0, zero = 0, over = 0;
		for (int smile = 0; smile < 2; smile++)
			for (int depth = 0; depth < 2; depth++)
				for (int slide = 0; slide <= 2; slide += 2) {
					const float base = smile ? 11.0f : 14.0f, dh = (depth ? 0.5f : 0.42f) * 16.0f;
					const float lids[] = {0.5f * (base + dh), dh, dh, dh, dh, 0.5f * (base + dh)};
					cf_gaze_init(&g);
					for (int rep = 0; rep < 25; rep++) {
						int seen = 0;
						for (int i = 0; i < 36 + 6; i++, t += dt) {
							const int kb = i - 36; /* blink frames 0..5 */
							const float h = kb >= 0 ? lids[kb] : base;
							const float pd = gpx + (float)slide * (base - h) / (base - dh);
							cf_face f = face_px(960, 500, 0, pd, h, 0, 0);
							noisy(&f, 1.0f, 0.3f);
							n = cf_gaze_update(&g, &f, &pn, t, out);
							const float sv = n ? shift_px(&out[0]) : 0;
							if (kb == 1)
								blinks++;
							if (kb >= 1 && kb <= 4 && !seen && holding_now(&g, t)) {
								caught++;
								seen = 1;
							}
							if (rep > 0 && (kb >= 0 || i >= 12)) {
								frames++;
								zero += sv < 0.5f * gpx;
								over += sv > 1.3f * gpx;
							}
						}
					}
				}
		/* Smiling steadily, same noise. */
		cf_gaze_init(&g);
		int held = 0, total = 0;
		for (int i = 0; i < 3000; i++, t += dt) {
			cf_face f = face_px(960, 500, 0, gpx, 10.0f, 0, 0);
			noisy(&f, 1.0f, 0.3f);
			cf_gaze_update(&g, &f, &pn, t, out);
			if (i >= 60) {
				held += holding_now(&g, t);
				total++;
			}
		}
		const float det = (float)caught / (float)blinks, fh = (float)held / (float)total;
		CHECK(det > 0.95f, "noisy half blinks: only %.0f%% caught", 100 * det);
		CHECK(zero <= frames / 500 && over <= frames / 150, "noisy half blinks: %d of %d frames lost the correction, %d "
								    "over-corrected by 30%%+",
		      zero, frames, over);
		CHECK(fh < 0.05f, "smiling with noisy lids holds %.1f%% of the time", 100 * fh);
		printf("ok: noisy half blinks: %.0f%% caught, %d of %d frames lost the correction, %d over-corrected; smiling "
		       "holds %.1f%% of frames\n",
		       100 * det, zero, frames, over, 100 * fh);
	}

	/* 13. Iris refinement: finds a dark iris offset from Vision's guess, partly under the lower lid. */
	{
		enum { W = 200, H = 160 };
		static uint8_t img[W * H];
		cf_pt eye[16];
		for (int i = 0; i < 16; i++) {
			const float a = (float)i / 16.0f * 6.2831853f;
			eye[i].x = 100 + 30 * cosf(a);
			eye[i].y = 80 + 12.5f * sinf(a);
		}
		const cf_pt truth = {106, 83};
		draw_eye(img, W, H, eye, 16, truth.x, truth.y, 12, 215, 70, 3);
		const cf_pt guess = {100, 80};
		cf_pt r = cf_refine_pupil(img, W, W, H, eye, 16, guess);
		const float err = hypotf(r.x - truth.x, r.y - truth.y);
		CHECK(err < 1.0f, "refined pupil (%.2f, %.2f), expected (%.1f, %.1f)", r.x, r.y, truth.x, truth.y);

		/* Iris higher up, under the upper lid, guess on the other side. */
		const cf_pt truth2 = {94, 77};
		draw_eye(img, W, H, eye, 16, truth2.x, truth2.y, 12, 215, 70, 4);
		cf_pt r2 = cf_refine_pupil(img, W, W, H, eye, 16, (cf_pt){99, 81});
		const float err2 = hypotf(r2.x - truth2.x, r2.y - truth2.y);
		CHECK(err2 < 1.0f, "refined pupil (%.2f, %.2f), expected (%.1f, %.1f)", r2.x, r2.y, truth2.x, truth2.y);

		const clock_t c0 = clock();
		for (int i = 0; i < 2000; i++)
			r = cf_refine_pupil(img, W, W, H, eye, 16, guess);
		const double ms = (double)(clock() - c0) / CLOCKS_PER_SEC * 1000.0 / 2000.0;
		CHECK(ms < 0.3, "refine_pupil took %.3f ms", ms);

		/* Unsure: no contrast, or eye nearly closed -> the guess comes back unchanged. */
		draw_eye(img, W, H, eye, 16, truth.x, truth.y, 12, 150, 145, 5);
		cf_pt r3 = cf_refine_pupil(img, W, W, H, eye, 16, guess);
		CHECK(r3.x == guess.x && r3.y == guess.y, "low contrast should return the guess");
		cf_pt closed[16];
		for (int i = 0; i < 16; i++) {
			closed[i] = eye[i];
			closed[i].y = 80 + (eye[i].y - 80) * 0.15f;
		}
		cf_pt r4 = cf_refine_pupil(img, W, W, H, closed, 16, guess);
		CHECK(r4.x == guess.x && r4.y == guess.y, "closed eye should return the guess");
		printf("ok: iris refinement error %.2f / %.2f px, %.3f ms per eye\n", err, err2, ms);
	}

	if (failures) {
		printf("%d FAILED\n", failures);
		return 1;
	}
	printf("ALL CHECKS PASSED\n");
	return 0;
}
