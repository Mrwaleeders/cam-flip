#include "gaze.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- One Euro filter (Casiez et al.): adaptive low-pass, little jitter and little lag. */

static double oe_alpha(double cutoff, double dt)
{
	const double tau = 1.0 / (2.0 * M_PI * cutoff);
	return 1.0 / (1.0 + tau / dt);
}

static double oe_filter(cf_one_euro *f, double x, double t, double min_cutoff, double beta)
{
	if (!f->ready) {
		f->value = x;
		f->deriv = 0;
		f->t = t;
		f->ready = 1;
		return x;
	}
	double dt = t - f->t;
	if (dt <= 1e-4)
		dt = 1.0 / 30.0;
	if (dt > 0.5) { /* long gap: restart instead of dragging an old value along */
		f->value = x;
		f->deriv = 0;
		f->t = t;
		return x;
	}
	const double dx = (x - f->value) / dt;
	const double ad = oe_alpha(1.0, dt);
	f->deriv = f->deriv + ad * (dx - f->deriv);
	const double cutoff = min_cutoff + beta * fabs(f->deriv);
	const double a = oe_alpha(cutoff, dt);
	f->value = f->value + a * (x - f->value);
	f->t = t;
	return f->value;
}

static float smoothstepf(float e0, float e1, float x)
{
	float t = (x - e0) / (e1 - e0);
	t = t < 0 ? 0 : (t > 1 ? 1 : t);
	return t * t * (3 - 2 * t);
}

/* ---- Measurements */

typedef struct {
	cf_pt center;
	float width, height;
	cf_pt pupil;
} eye_measure;

static int measure(const cf_pt *pts, int count, cf_pt pupil, eye_measure *m)
{
	if (count < 4)
		return 0;
	float minx = pts[0].x, maxx = pts[0].x, miny = pts[0].y, maxy = pts[0].y;
	for (int i = 1; i < count; i++) {
		minx = fminf(minx, pts[i].x);
		maxx = fmaxf(maxx, pts[i].x);
		miny = fminf(miny, pts[i].y);
		maxy = fmaxf(maxy, pts[i].y);
	}
	/* Centre = average of the contour points: steadier than the bounding box, because noise
	 * on individual landmarks averages out. */
	float sx = 0, sy = 0;
	for (int i = 0; i < count; i++) {
		sx += pts[i].x;
		sy += pts[i].y;
	}
	m->center.x = sx / (float)count;
	m->center.y = sy / (float)count;
	m->width = maxx - minx;
	m->height = maxy - miny;
	m->pupil = pupil;
	return m->width > 2;
}

void cf_gaze_init(cf_gaze *g)
{
	memset(g, 0, sizeof(*g));
	g->up.x = 0;
	g->up.y = -1;
}

void cf_gaze_start_calibration(cf_gaze *g)
{
	g->calibrating = 1;
	g->calib_count = 0;
	g->calib_frames = 0;
	g->calib_done = 0;
	g->calib_failed = 0;
}

static int cmpf(const void *a, const void *b);

static void finish_calibration(cf_gaze *g)
{
	qsort(g->calib_samples, (size_t)g->calib_count, sizeof(float), cmpf);
	g->calib_result = g->calib_samples[g->calib_count / 2];
	g->calibrating = 0;
	g->calib_done = 1;
}

/* Called once per frame while calibrating, whether or not a face was found, so a calibration
 * started while the user looks away (or is out of frame) can't run forever. */
static void calibration_tick(cf_gaze *g)
{
	if (!g->calibrating || ++g->calib_frames < CF_CALIB_TIMEOUT_FRAMES)
		return;
	if (g->calib_count >= CF_CALIB_MIN_SAMPLES) {
		finish_calibration(g);
	} else {
		g->calibrating = 0;
		g->calib_failed = 1;
	}
}

int cf_gaze_take_calibration(cf_gaze *g, float *baseline_out)
{
	if (g->calib_failed) {
		g->calib_failed = 0;
		return -1;
	}
	if (!g->calib_done)
		return 0;
	g->calib_done = 0;
	*baseline_out = g->calib_result;
	return 1;
}

static int cmpf(const void *a, const void *b)
{
	float x = *(const float *)a, y = *(const float *)b;
	return (x > y) - (x < y);
}

/* Moves `factor` towards `target`: fades in gently, fades out quickly (blinks, head turns). */
static float ease_factor(float factor, float target, double dt)
{
	const double tau = target > factor ? 0.30 : 0.08;
	const double a = 1.0 - exp(-dt / tau);
	return factor + (float)a * (target - factor);
}

int cf_gaze_update(cf_gaze *g, const cf_face *face, const cf_gaze_params *p, double t, cf_eye out[2])
{
	double dt = g->last_t > 0 ? t - g->last_t : 1.0 / 30.0;
	if (dt <= 0 || dt > 0.5)
		dt = 1.0 / 30.0;
	g->last_t = t;

	eye_measure m[2];
	int ok = face->found && measure(face->left_eye, face->left_count, face->left_pupil, &m[0]) &&
		 measure(face->right_eye, face->right_count, face->right_pupil, &m[1]);

	float target = 0;

	calibration_tick(g);

	if (ok) {
		g->lost_time = 0;

		/* Face "up": perpendicular to the line between the eyes, pointing to the top of the image.
		 * The raw eye centres jitter by a pixel or so every frame, which would make the direction
		 * of the warp wobble, so the roll angle of that line is smoothed (One Euro again: steady
		 * when the head is still, quick to follow a real tilt). The line is taken left to right,
		 * so the angle stays well inside -90..90 degrees and never wraps around. */
		float lx = m[1].center.x - m[0].center.x, ly = m[1].center.y - m[0].center.y;
		if (lx < 0) {
			lx = -lx;
			ly = -ly;
		}
		const double roll = oe_filter(&g->roll, atan2(ly, fmaxf(lx, 1e-3f)), t, 1.0, 3.0);
		const float ax = (float)cos(roll), ay = (float)sin(roll);
		cf_pt up = {-ay, ax};
		if (up.y > 0) {
			up.x = -up.x;
			up.y = -up.y;
		}
		g->up = up;

		/* Pupil position along "up" (positive = higher), in eye heights; and sideways, in eye widths. */
		float offset = 0, side = 0, openness = 1e9f;
		for (int i = 0; i < 2; i++) {
			const float dx = m[i].pupil.x - m[i].center.x, dy = m[i].pupil.y - m[i].center.y;
			offset += (dx * up.x + dy * up.y) / fmaxf(m[i].height, 1.0f);
			side += (dx * ax + dy * ay) / fmaxf(m[i].width, 1.0f);
			openness = fminf(openness, m[i].height / fmaxf(m[i].width, 1.0f));
		}
		offset /= 2;
		side /= 2;

		const int blinking = openness < 0.14f;
		const float abs_yaw = face->has_pose ? fabsf(face->yaw_deg) : 0;
		const float abs_pitch = face->has_pose ? fabsf(face->pitch_deg) : 0;

		if (g->calibrating && !blinking && abs_yaw < 15) {
			g->calib_samples[g->calib_count++] = offset;
			if (g->calib_count >= CF_CALIB_SAMPLES)
				finish_calibration(g);
		}

		/* Eye geometry: follows the head closely (high beta), so the warp stays on the eyes. */
		const float face_scale = fmaxf(face->face_width, 50.0f);
		for (int i = 0; i < 2; i++) {
			const double beta = 0.25 / face_scale * 100.0;
			oe_filter(&g->cx[i], m[i].center.x, t, 1.5, beta);
			oe_filter(&g->cy[i], m[i].center.y, t, 1.5, beta);
			oe_filter(&g->rx[i], m[i].width * 0.6f, t, 1.5, 0.0);
			oe_filter(&g->ry[i], fmaxf(m[i].height, m[i].width * 0.3f) * 1.1f, t, 1.5, 0.0);
			if (!blinking)
				oe_filter(&g->eye_h[i], m[i].height, t, 1.5, 0.0);
		}
		g->have_geometry = 1;

		/* How far below the "looking at the lens" position the pupils are. Smoothed so pupil
		 * jitter disappears, but quick enough to follow a real glance. */
		double down = p->baseline - offset;
		if (!blinking)
			down = oe_filter(&g->down, down, t, 1.2, 1.5);
		else
			down = g->down.ready ? g->down.value : 0;

		/* Everything that should gently switch the effect off. Each term is 1 when it's fine. */
		const float f_yaw = 1.0f - smoothstepf(15.0f, 28.0f, abs_yaw);
		const float f_pitch = 1.0f - smoothstepf(20.0f, 35.0f, abs_pitch);
		const float f_side = 1.0f - smoothstepf(0.16f, 0.28f, fabsf(side));
		const float f_far = 1.0f - smoothstepf(p->max_glance, p->max_glance + 0.30f, (float)down);
		const float f_blink = blinking ? 0.0f : 1.0f;
		const float f_size = smoothstepf(0.05f, 0.09f, face->face_width / fmaxf((float)face->image_width, 1.0f));
		target = f_yaw * f_pitch * f_side * f_far * f_blink * f_size;
		if (down <= 0)
			target = 0; /* already looking at (or above) the lens */

		g->factor = ease_factor(g->factor, target, dt);
		g->last_target = target;

		const float d = (float)(down > 0 ? down : 0);
		for (int i = 0; i < 2; i++) {
			const float ry = (float)g->ry[i].value;
			float amount = d * (float)g->eye_h[i].value * p->strength * g->factor;
			if (amount > ry * 0.38f)
				amount = ry * 0.38f; /* well inside the limit where the warp could fold */
			out[i].cx = (float)g->cx[i].value;
			out[i].cy = (float)g->cy[i].value;
			out[i].rx = (float)g->rx[i].value;
			out[i].ry = ry;
			out[i].sx = up.x * amount;
			out[i].sy = up.y * amount;
		}
	} else {
		/* Face lost. A frame or two without a face is usually just the detector missing one,
		 * so hold the warp exactly as it was. If the face stays away longer, the person may well
		 * have moved and the spot we'd be warping may no longer be their eyes, so fade out fast. */
		g->lost_time += dt;
		g->last_target = 0;
		if (g->lost_time >= CF_LOST_GRACE_S) {
			const double a = 1.0 - exp(-dt / 0.02);
			g->factor -= (float)a * g->factor;
			if (g->factor < 0.05f)
				g->factor = 0; /* the last few percent are invisible: just stop */
		}
		if (!g->have_geometry || g->factor < 0.01f) {
			g->have_geometry = g->factor >= 0.01f && g->have_geometry;
			return 0;
		}
		const float d = g->down.ready ? (float)fmax(g->down.value, 0) : 0;
		for (int i = 0; i < 2; i++) {
			const float ry = (float)g->ry[i].value;
			float amount = fminf(d * (float)g->eye_h[i].value * p->strength * g->factor, ry * 0.38f);
			out[i].cx = (float)g->cx[i].value;
			out[i].cy = (float)g->cy[i].value;
			out[i].rx = (float)g->rx[i].value;
			out[i].ry = ry;
			out[i].sx = g->up.x * amount;
			out[i].sy = g->up.y * amount;
		}
	}

	if (g->factor < 0.005f)
		return 0;
	return 2;
}
