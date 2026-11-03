#include "gaze.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Units: every gaze offset (baseline, notes_offset, max_glance, notes_side, the calibration
 * results) is measured in eye widths, where the eye width is the distance between the inner and
 * outer eye corners. Corners are fixed to the skull, so neither the measure nor its scale moves
 * when the lids do (looking down lowers the upper lid; blinking closes both).
 */

/* ---- Tuning */

#define DEFAULT_OPEN 0.30f    /* typical eye height / width, until calibrated or learned */
#define BLINK_ENTER 0.40f     /* relative openness (vs. open-eye) below which the lids count as closed */
#define BLINK_EXIT 0.50f      /* ... and above which the blink is over (hysteresis) */
#define CLOSING_RATE 5.0f     /* relative openness per second: lids dropping this fast may be a blink */
#define CLOSING_MAX 0.15      /* s: lids that stop dropping above BLINK_ENTER are a glance or a squint */
#define BLINK_HOLD_MAX 0.40   /* s: a "blink" longer than this is eyes closed: fade out */
#define SETTLE_TIME 0.07      /* s: ignore the pupil this long after the lids reopen */
#define LID_GLANCE_K 0.30f    /* eye widths of downward gaze per unit of lid drop (far-glance cue only) */
#define LID_SHIFT 0.35f       /* upper lid follows the iris by this fraction */
#define AMOUNT_MAX_RY 0.38f   /* vertical iris shift limit, fraction of ry (well inside the fold limit) */
#define HSHIFT_MAX_RX 0.30f   /* sideways iris shift limit, fraction of rx */
#define READ_DAMP 0.65f       /* fraction of the sideways reading excursion cancelled at full strength */
#define SIDE_FADE_LO 0.30f    /* sideways distance from the notes centre (eye widths) where fading starts */
#define SIDE_FADE_HI 0.45f
#define PREDICT_MAX_DT 0.10   /* s: never extrapolate further than this */
#define VEL_TAU 0.05         /* s: smoothing of the velocity used for prediction */
#define LOST_HOLD 0.20        /* s: a face missing for less than this (detector dropout) is held, not faded */
#define RESYNC_JUMP 0.03f     /* eye widths: after a blink, a gaze change bigger than this is taken at once */

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

/* Low-pass for eye positions that jump (saccades): noise-sized changes are smoothed with a
 * 3 Hz low-pass, changes over ~0.06 eye widths are taken immediately. No lag on jumps. */
static double saccade_filter(cf_one_euro *f, double x, double t, double dt)
{
	if (!f->ready || t - f->t > 0.5) {
		f->value = x;
		f->deriv = 0;
		f->t = t;
		f->ready = 1;
		return x;
	}
	const double e = x - f->value;
	const double a_min = oe_alpha(3.0, dt);
	const double a = a_min + (1.0 - a_min) * smoothstepf(0.02f, 0.06f, (float)fabs(e));
	f->value += a * e;
	f->t = t;
	return f->value;
}

static float clampf(float x, float lo, float hi)
{
	return x < lo ? lo : (x > hi ? hi : x);
}

/* ---- Measurements */

typedef struct {
	cf_pt center;    /* corner midpoint, moved along the corner line to the middle of the outline */
	float width;     /* corner to corner */
	float height;    /* extent of the outline across the corner line */
	float offset;    /* pupil above the corner line, eye widths (positive = up) */
	float side;      /* pupil along the corner line from the corner midpoint, eye widths */
	float openness;  /* height / width */
} eye_measure;

/* axis: unit vector from this face's left eye towards its right eye (image space).
 * up: unit vector perpendicular to it, pointing to the top of the head. */
static int measure(const cf_pt *pts, int count, cf_pt pupil, cf_pt axis, cf_pt up, eye_measure *m)
{
	if (count < 4 || count > CF_MAX_EYE_POINTS)
		return 0;
	/* Corners: the outline's extreme points along the inter-eye axis. */
	int ia = 0, ib = 0;
	float pa = pts[0].x * axis.x + pts[0].y * axis.y, pb = pa;
	for (int i = 1; i < count; i++) {
		const float d = pts[i].x * axis.x + pts[i].y * axis.y;
		if (d < pa) {
			pa = d;
			ia = i;
		}
		if (d > pb) {
			pb = d;
			ib = i;
		}
	}
	float ux = pts[ib].x - pts[ia].x, uy = pts[ib].y - pts[ia].y;
	const float w = sqrtf(ux * ux + uy * uy);
	if (!(w > 2))
		return 0;
	ux /= w;
	uy /= w;
	float nx = uy, ny = -ux; /* perpendicular to the corner line */
	if (nx * up.x + ny * up.y < 0) {
		nx = -nx;
		ny = -ny;
	}
	const cf_pt mid = {(pts[ia].x + pts[ib].x) * 0.5f, (pts[ia].y + pts[ib].y) * 0.5f};
	float lo = 0, hi = 0, along = 0;
	for (int i = 0; i < count; i++) {
		const float dx = pts[i].x - mid.x, dy = pts[i].y - mid.y;
		const float v = dx * nx + dy * ny;
		lo = fminf(lo, v);
		hi = fmaxf(hi, v);
		along += dx * ux + dy * uy;
	}
	along /= (float)count;
	m->center.x = mid.x + ux * along;
	m->center.y = mid.y + uy * along;
	m->width = w;
	m->height = hi - lo;
	m->openness = m->height / w;
	const float px = pupil.x - mid.x, py = pupil.y - mid.y;
	m->offset = (px * nx + py * ny) / w;
	m->side = (px * ux + py * uy) / w;
	/* A broken landmark (NaN/inf) would poison every filter for good: treat it as no face. */
	return isfinite(m->offset) && isfinite(m->side) && isfinite(m->openness) && isfinite(m->center.x) &&
	       isfinite(m->center.y);
}

/* Inter-eye axis and face "up" from the two outlines. */
static int face_axes(const cf_face *face, cf_pt *axis, cf_pt *up)
{
	if (face->left_count < 1 || face->right_count < 1 || face->left_count > CF_MAX_EYE_POINTS ||
	    face->right_count > CF_MAX_EYE_POINTS)
		return 0;
	float lx = 0, ly = 0, rx = 0, ry = 0;
	for (int i = 0; i < face->left_count; i++) {
		lx += face->left_eye[i].x;
		ly += face->left_eye[i].y;
	}
	for (int i = 0; i < face->right_count; i++) {
		rx += face->right_eye[i].x;
		ry += face->right_eye[i].y;
	}
	lx /= (float)face->left_count;
	ly /= (float)face->left_count;
	rx /= (float)face->right_count;
	ry /= (float)face->right_count;
	float ax = rx - lx, ay = ry - ly;
	const float alen = sqrtf(ax * ax + ay * ay);
	if (!(alen > 1e-3f))
		return 0;
	ax /= alen;
	ay /= alen;
	cf_pt u = {-ay, ax};
	if (u.y > 0) {
		u.x = -u.x;
		u.y = -u.y;
	}
	axis->x = ax;
	axis->y = ay;
	*up = u;
	return 1;
}

void cf_gaze_init(cf_gaze *g)
{
	memset(g, 0, sizeof(*g));
	g->up.x = 0;
	g->up.y = -1;
	g->axis.x = 1;
	g->axis.y = 0;
	g->open_ref = DEFAULT_OPEN;
}

/* ---- Calibration */

static int cmpf(const void *a, const void *b)
{
	float x = *(const float *)a, y = *(const float *)b;
	return (x > y) - (x < y);
}

static float median(float *v, int n)
{
	qsort(v, (size_t)n, sizeof(float), cmpf);
	return (n & 1) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

void cf_gaze_start_calibration(cf_gaze *g, enum cf_calib_target target)
{
	g->calibrating = 1;
	g->calib_target = target;
	g->calib_count = 0;
	g->calib_frames = 0;
	g->calib_have_start = 0;
	g->calib_done = 0;
	g->calib_failed = 0;
}

static void finish_calibration(cf_gaze *g)
{
	g->calib_result = median(g->calib_samples, g->calib_count);
	g->calib_open_result = median(g->calib_open, g->calib_count);
	g->calib_side_result = median(g->calib_side, g->calib_count);
	g->calibrating = 0;
	g->calib_done = 1;
}

/* Called once per frame while calibrating, whether or not a face was found, so a calibration
 * started while the user looks away (or is out of frame) can't run forever. */
static void calibration_tick(cf_gaze *g, double t)
{
	if (g->calibrating && !g->calib_have_start) {
		g->calib_start = t;
		g->calib_have_start = 1;
	}
	if (!g->calibrating || ++g->calib_frames < CF_CALIB_TIMEOUT_FRAMES)
		return;
	if (g->calib_count >= CF_CALIB_MIN_SAMPLES) {
		finish_calibration(g);
	} else {
		g->calibrating = 0;
		g->calib_failed = 1;
	}
}

int cf_gaze_take_calibration(cf_gaze *g, cf_calib_result *out)
{
	if (g->calib_failed) {
		g->calib_failed = 0;
		return -1;
	}
	if (!g->calib_done)
		return 0;
	g->calib_done = 0;
	out->target = g->calib_target;
	out->offset = g->calib_result;
	out->openness = g->calib_open_result;
	out->side = g->calib_side_result;
	return 1;
}

/* ---- Blinks */

/* Moves `factor` towards `target`: fades in gently, fades out quickly. */
static float ease_factor(float factor, float target, double dt)
{
	const double tau = target > factor ? 0.30 : 0.08;
	const double a = 1.0 - exp(-dt / tau);
	return factor + (float)a * (target - factor);
}

/*
 * r: openness relative to the user's open eye (1 = normal). rate: how fast r falls, per second.
 * A blink closes the lids to a slit within ~2 frames; looking down lowers them part way and
 * then they stay. So: below BLINK_ENTER = blink; dropping fast = hold until it's clear which
 * one it is; a drop that stops above BLINK_ENTER is released as a glance (or squint).
 */
static void update_blink(cf_gaze *g, float r, float rate, double t)
{
	if (g->blink) {
		const double len = t - g->hold_start;
		if (r > BLINK_EXIT || (r > BLINK_ENTER + 0.05f && fabsf(rate) < 1.0f && len > 0.1)) {
			g->blink = 0;
			g->settle_until = t + SETTLE_TIME;
		}
		return;
	}
	if (r < BLINK_ENTER) {
		g->blink = 1;
		if (!g->closing)
			g->hold_start = t;
		g->closing = 0;
		return;
	}
	if (g->closing) {
		if (rate < 1.5f || t - g->hold_start > CLOSING_MAX)
			g->closing = 0;
		return;
	}
	if (rate > CLOSING_RATE && r < 0.9f) {
		g->closing = 1;
		g->hold_start = t;
	}
}

static int holding(const cf_gaze *g, double t)
{
	return g->blink || g->closing || t < g->settle_until;
}

/* ---- Output */

/* Fills out[] from the filtered state. The shifts before strength are in g->amount_unit
 * (vertical, along face up) and g->hshift_unit (sideways, along the eye axis). */
static void emit(const cf_gaze *g, const cf_gaze_params *p, cf_eye out[2])
{
	const float strength = clampf(p->strength, 0.0f, 1.0f);
	for (int i = 0; i < 2; i++) {
		cf_eye *e = &out[i];
		memset(e, 0, sizeof(*e));
		const float w = (float)g->rx[i].value, h = (float)g->ry[i].value;
		e->cx = (float)g->cx[i].value;
		e->cy = (float)g->cy[i].value;

		/* Outline, and how far it reaches from the centre. */
		float ext_x = 0.5f * w, ext_y = 0.5f * h;
		const int n = g->ap_count[i] < CF_MAX_APERTURE ? g->ap_count[i] : CF_MAX_APERTURE;
		for (int k = 0; k < n; k++) {
			e->aperture[k][0] = e->cx + g->ap[i][k][0];
			e->aperture[k][1] = e->cy + g->ap[i][k][1];
			ext_x = fmaxf(ext_x, fabsf(g->ap[i][k][0]));
			ext_y = fmaxf(ext_y, fabsf(g->ap[i][k][1]));
		}
		e->aperture_count = n;

		/* Outer ellipse: comfortably contains the outline plus a gentle falloff margin. */
		e->rx = fmaxf(0.75f * w, ext_x + 0.25f * w);
		e->ry = fmaxf(fmaxf(1.6f * h, 0.5f * w), ext_y + 1.1f * h);

		float amount = g->amount_unit[i] * strength;
		amount = fminf(amount, fminf(AMOUNT_MAX_RY * e->ry, 0.5f * (e->ry - ext_y)));
		amount = fmaxf(amount, 0.0f);
		float hs = g->hshift_unit[i] * strength;
		hs = clampf(hs, -HSHIFT_MAX_RX * e->rx, HSHIFT_MAX_RX * e->rx);

		e->sx = g->up.x * amount + g->axis.x * hs;
		e->sy = g->up.y * amount + g->axis.y * hs;
		/* The upper lid follows vertical eye movement, not sideways movement. */
		e->lid_sx = g->up.x * amount * LID_SHIFT;
		e->lid_sy = g->up.y * amount * LID_SHIFT;
	}
}

static int finish_output(cf_gaze *g, const cf_gaze_params *p, cf_eye out[2])
{
	emit(g, p, out);
	int n = g->factor < 0.005f ? 0 : 2;
	if (n && fabsf(g->amount_unit[0]) + fabsf(g->amount_unit[1]) + fabsf(g->hshift_unit[0]) +
			     fabsf(g->hshift_unit[1]) < 0.01f)
		n = 0; /* nothing to move (looking into the lens): skip the warp */
	g->last_n = n;
	g->last_out[0] = out[0];
	g->last_out[1] = out[1];
	return n;
}

int cf_gaze_update(cf_gaze *g, const cf_face *face, const cf_gaze_params *p, double t, cf_eye out[2])
{
	double dt = g->last_t > 0 ? t - g->last_t : 1.0 / 30.0;
	if (dt <= 0 || dt > 0.5)
		dt = 1.0 / 30.0;
	g->last_t = t;

	calibration_tick(g, t);

	cf_pt axis = {1, 0}, up = {0, -1};
	eye_measure m[2];
	const int ok = face->found && face_axes(face, &axis, &up) &&
		       measure(face->left_eye, face->left_count, face->left_pupil, axis, up, &m[0]) &&
		       measure(face->right_eye, face->right_count, face->right_pupil, axis, up, &m[1]);
	g->face_found = ok;

	if (!ok) {
		/* Face lost: fade out where the eyes last were. */
		g->tracking = 0;
		g->have_rel_open = 0;
		g->blink = g->closing = 0;
		/* The detector often misses a frame or two: hold the correction through that instead of
		 * letting the eyes dip and climb back. Lost for longer: fade out. */
		const int brief = g->have_geometry && g->cx[0].ready && t - g->cx[0].t < LOST_HOLD;
		const float old = g->factor;
		if (!brief)
			g->factor = ease_factor(g->factor, 0, dt);
		g->last_target = 0;
		if (!g->have_geometry || g->factor < 0.01f) {
			g->have_geometry = g->factor >= 0.01f && g->have_geometry;
			g->last_n = 0;
			return 0;
		}
		/* Fade the last shifts (vertical and sideways) by the same ratio as the factor. */
		const float ratio = old > 1e-6f ? g->factor / old : 0.0f;
		for (int i = 0; i < 2; i++) {
			g->amount_unit[i] *= ratio;
			g->hshift_unit[i] *= ratio;
		}
		return finish_output(g, p, out);
	}
	g->tracking = 1;
	g->up = up;
	g->axis = axis;

	const float offset = 0.5f * (m[0].offset + m[1].offset);
	const float side = 0.5f * (m[0].side + m[1].side);
	const float openness = fminf(m[0].openness, m[1].openness);

	/* Lids: relative to the calibrated (or learned) open eye. */
	const float ref = p->baseline_open > 0 ? p->baseline_open : g->open_ref;
	const float r = openness / fmaxf(ref, 0.05f);
	const float rate = g->have_rel_open ? (g->last_rel_open - r) / (float)dt : 0.0f;
	g->last_rel_open = r;
	g->have_rel_open = 1;
	update_blink(g, r, rate, t);
	const int hold = holding(g, t);

	const float down_raw = p->baseline - offset;

	/* Learn the open-eye openness when uncalibrated: follows wide-open quickly, closing slowly. */
	if (p->baseline_open <= 0 && !hold && fabsf(down_raw) < 0.08f && r > 0.6f) {
		const double tau = openness > g->open_ref ? 1.0 : 5.0;
		g->open_ref += (float)(1.0 - exp(-dt / tau)) * (openness - g->open_ref);
		g->open_ref = clampf(g->open_ref, 0.15f, 0.6f);
	}

	float abs_yaw = face->has_pose ? fabsf(face->yaw_deg) : 0;
	float abs_pitch = face->has_pose ? fabsf(face->pitch_deg) : 0;
	abs_yaw = isfinite(abs_yaw) ? abs_yaw : 0; /* broken pose: ignore it rather than poison the factor */
	abs_pitch = isfinite(abs_pitch) ? abs_pitch : 0;

	/* The first moments after starting are skipped: the eyes are still on the Calibrate button. */
	const int calib_warm = g->calibrating && t - g->calib_start >= CF_CALIB_WARMUP_S;
	if (calib_warm && !hold && abs_yaw < 15 && g->calib_count < 64) {
		g->calib_samples[g->calib_count] = offset;
		g->calib_open[g->calib_count] = openness;
		g->calib_side[g->calib_count] = side;
		g->calib_count++;
		if (g->calib_count >= CF_CALIB_SAMPLES)
			finish_calibration(g);
	}

	/* Eye geometry. Position follows the head closely (high beta) and comes from the corners,
	 * which don't move when blinking, so it keeps tracking through blinks. Size and outline are
	 * held while the lids are closed or moving fast. */
	const float face_scale = fmaxf(face->face_width, 50.0f);
	const double beta = 0.25 / face_scale * 100.0;
	for (int i = 0; i < 2; i++) {
		const int had = g->cx[i].ready && t - g->cx[i].t <= 0.5 && t - g->cx[i].t > 1e-3;
		const double px = g->cx[i].value, py = g->cy[i].value, pt = g->cx[i].t;
		oe_filter(&g->cx[i], m[i].center.x, t, 1.5, beta);
		oe_filter(&g->cy[i], m[i].center.y, t, 1.5, beta);
		if (had) {
			/* Velocity of the filtered centre, lightly smoothed: the One Euro derivative is
			 * low-passed at 1 Hz and keeps "moving" for ~0.3 s after the head stops, which
			 * made the prediction overshoot the eye. */
			const double vdt = t - pt;
			const float a = (float)(1.0 - exp(-vdt / VEL_TAU));
			g->vel[i].x += a * ((float)((g->cx[i].value - px) / vdt) - g->vel[i].x);
			g->vel[i].y += a * ((float)((g->cy[i].value - py) / vdt) - g->vel[i].y);
		} else {
			g->vel[i].x = g->vel[i].y = 0;
		}
		const cf_pt *pts = i == 0 ? face->left_eye : face->right_eye;
		const int count = i == 0 ? face->left_count : face->right_count;
		if (!hold || !g->rx[i].ready) {
			oe_filter(&g->rx[i], m[i].width, t, 1.5, 0.0);
			oe_filter(&g->ry[i], m[i].height, t, 1.5, 0.0);
		}
		const int n = count < CF_MAX_APERTURE ? count : CF_MAX_APERTURE;
		if (g->ap_count[i] != n) {
			for (int k = 0; k < n; k++) {
				g->ap[i][k][0] = pts[k].x - m[i].center.x;
				g->ap[i][k][1] = pts[k].y - m[i].center.y;
			}
			g->ap_count[i] = n;
		} else if (!hold) {
			const float a = (float)(1.0 - exp(-dt / 0.05));
			for (int k = 0; k < n; k++) {
				g->ap[i][k][0] += a * (pts[k].x - m[i].center.x - g->ap[i][k][0]);
				g->ap[i][k][1] += a * (pts[k].y - m[i].center.y - g->ap[i][k][1]);
			}
		}
	}
	g->have_geometry = 1;

	/* How far below the "looking at the lens" position the pupils are: smoothed so pupil
	 * jitter disappears, quick enough to follow a real glance, frozen through blinks. */
	double down;
	if (!hold) {
		/* First look after a blink: eyes often move during a blink (blinks go with gaze shifts).
		 * A clear change is taken at once instead of sliding there from the held value. */
		if (g->was_holding && g->down.ready && fabs(down_raw - g->down.value) > RESYNC_JUMP)
			g->down.ready = 0;
		g->was_holding = 0;
		down = oe_filter(&g->down, down_raw, t, 1.2, 4.0);
	} else {
		down = g->down.ready ? g->down.value : 0;
		/* Only a real blink (lids shut), not a noise-triggered "closing" hold, may resync. */
		g->was_holding |= g->blink;
	}

	/* Sideways position for reading damping. Eyes move in saccades: a return sweep jumps a
	 * fifth of an eye width in one or two frames. A lagging filter would let that jump show at
	 * full size and then drift back (a visible snap), so: jumps bigger than landmark noise are
	 * followed at once, and only noise-sized changes are smoothed. */
	const float side_c = p->has_notes ? p->notes_side : 0.0f;
	double side_f;
	if (!hold)
		side_f = saccade_filter(&g->side, side, t, dt);
	else
		side_f = g->side.ready ? g->side.value : side_c;
	const float side_dev = (float)side_f - side_c;

	/* Glance limit: just past the notes when calibrated, else the manual setting. */
	const float notes_glance = p->baseline - p->notes_offset;
	const float limit = (p->has_notes && notes_glance > 0.01f) ? 1.4f * notes_glance : p->max_glance;
	const float band = fmaxf(0.5f * limit, 0.04f);
	/* Second cue: lids drop when looking down. When the pupil also says "down", a big lid drop
	 * means a far glance (keyboard) even if the pupil landmark under-reads it. It can only
	 * switch the effect off, never add correction. */
	float far = (float)down;
	if (!hold && far > 0.25f * limit)
		far = fmaxf(far, LID_GLANCE_K * (1.0f - r));

	/* Everything that should gently switch the effect off. Each term is 1 when it's fine. */
	const float f_yaw = 1.0f - smoothstepf(15.0f, 28.0f, abs_yaw);
	const float f_pitch = 1.0f - smoothstepf(20.0f, 35.0f, abs_pitch);
	const float f_side = 1.0f - smoothstepf(SIDE_FADE_LO, SIDE_FADE_HI, fabsf(side_dev));
	const float f_far = 1.0f - smoothstepf(limit, limit + band, far);
	float f_size = smoothstepf(0.05f, 0.09f, face->face_width / fmaxf((float)face->image_width, 1.0f));
	f_size = isfinite(f_size) ? f_size : 0;
	/* No "down <= 0 -> off" here: the shift is already 0 at the lens (it is proportional to the
	 * glance), and switching the factor off there made every glance at the notes start from 0 and
	 * ease in over half a second, so the eyes visibly dipped and climbed back each time. */
	float target = f_yaw * f_pitch * f_far * f_size;

	if (hold) {
		/* Blink: hold the correction steady. Only head-based fades (reliable while the eyes are
		 * closed) may lower it; eyes closed for long fade out. */
		const float head = f_yaw * f_pitch * f_size;
		if (g->blink && t - g->hold_start > BLINK_HOLD_MAX)
			g->factor = ease_factor(g->factor, 0, dt);
		else if (g->factor > head)
			g->factor = ease_factor(g->factor, head, dt);
	} else {
		g->factor = ease_factor(g->factor, target, dt);
	}
	g->last_target = target;

	const float d = (float)(down > 0 ? down : 0);
	/* Sideways fade is applied directly, not through the eased factor: the side position moves in
	 * saccades, so the damping lets go during the jump itself instead of gliding after it lands.
	 * Sideways damping only engages as the gaze drops towards the notes, so looking around at lens
	 * height (at the other person) stays untouched. */
	const float engage = smoothstepf(0.0f, fmaxf(0.35f * limit, 0.01f), d);
	for (int i = 0; i < 2; i++) {
		const float w = (float)g->rx[i].value;
		g->amount_unit[i] = d * w * g->factor * f_side;
		g->hshift_unit[i] = -READ_DAMP * side_dev * w * g->factor * f_side * engage;
	}
	return finish_output(g, p, out);
}

int cf_gaze_predict(const cf_gaze *g, const cf_gaze_params *p, double t_now, cf_eye out[2])
{
	if (!g->last_n || !g->have_geometry)
		return 0;
	/* Re-emit so a strength change shows immediately, then move to t_now. */
	emit(g, p, out);
	if (!g->tracking)
		return g->last_n;
	for (int i = 0; i < 2; i++) {
		double dtp = t_now - g->cx[i].t;
		dtp = dtp < -PREDICT_MAX_DT ? -PREDICT_MAX_DT : (dtp > PREDICT_MAX_DT ? PREDICT_MAX_DT : dtp);
		const float w = (float)g->rx[i].value;
		float vx = g->vel[i].x, vy = g->vel[i].y;
		/* Ignore tiny velocities (landmark noise while sitting still). */
		const float speed = sqrtf(vx * vx + vy * vy);
		const float k = smoothstepf(0.1f * w, 0.5f * w, speed);
		float dx = vx * k * (float)dtp, dy = vy * k * (float)dtp;
		const float dist = sqrtf(dx * dx + dy * dy), max_d = 0.75f * w;
		if (dist > max_d) {
			dx *= max_d / dist;
			dy *= max_d / dist;
		}
		out[i].cx += dx;
		out[i].cy += dy;
		for (int j = 0; j < out[i].aperture_count; j++) {
			out[i].aperture[j][0] += dx;
			out[i].aperture[j][1] += dy;
		}
	}
	return g->last_n;
}

void cf_gaze_get_status(const cf_gaze *g, cf_gaze_status *out)
{
	out->face_found = g->face_found;
	out->blinking = g->blink;
	out->glance = g->down.ready ? (float)g->down.value : 0.0f;
	/* What is actually being corrected: nothing while looking into the lens. */
	out->factor = (g->down.ready && g->down.value > 0) ? g->factor : 0.0f;
}

/* ---- Iris centre refinement */

#define RP_MAXW 80
#define RP_MAXH 48

/*
 * Two steps on a small grid over the eye's bounding box, masked by the eye outline:
 *  1. darkest square (side ~1.4 R, inscribed in the iris) near the guess, via integral images;
 *  2. means of gradients (Timm & Barth) in a few pixels around it: the point most image
 *     gradients point away from, i.e. the centre of the dark disc's edge. This also works when
 *     the lids cover part of the iris, where a darkness centroid would be pulled off centre.
 */
cf_pt cf_refine_pupil(const uint8_t *grey, int stride, int width, int height, const cf_pt *eye, int count,
		      cf_pt guess)
{
	if (!grey || !eye || count < 4 || count > CF_MAX_EYE_POINTS || width <= 0 || height <= 0)
		return guess;

	/* Corners: the farthest pair of outline points. */
	int ia = 0, ib = 1;
	float best = -1;
	for (int i = 0; i < count; i++)
		for (int j = i + 1; j < count; j++) {
			const float dx = eye[j].x - eye[i].x, dy = eye[j].y - eye[i].y;
			const float d = dx * dx + dy * dy;
			if (d > best) {
				best = d;
				ia = i;
				ib = j;
			}
		}
	const float w = sqrtf(best);
	if (!(w >= 8))
		return guess;
	const float ux = (eye[ib].x - eye[ia].x) / w, uy = (eye[ib].y - eye[ia].y) / w;
	float lo = 0, hi = 0;
	float minx = eye[0].x, maxx = eye[0].x, miny = eye[0].y, maxy = eye[0].y;
	for (int i = 0; i < count; i++) {
		const float v = (eye[i].x - eye[ia].x) * uy - (eye[i].y - eye[ia].y) * ux;
		lo = fminf(lo, v);
		hi = fmaxf(hi, v);
		minx = fminf(minx, eye[i].x);
		maxx = fmaxf(maxx, eye[i].x);
		miny = fminf(miny, eye[i].y);
		maxy = fmaxf(maxy, eye[i].y);
	}
	if ((hi - lo) / w < 0.12f)
		return guess; /* nearly closed */

	/* Grid: bounding box, subsampled if the eye is large. */
	int x0 = (int)floorf(minx), y0 = (int)floorf(miny);
	int x1 = (int)ceilf(maxx), y1 = (int)ceilf(maxy);
	x0 = x0 < 0 ? 0 : x0;
	y0 = y0 < 0 ? 0 : y0;
	x1 = x1 > width ? width : x1;
	y1 = y1 > height ? height : y1;
	if (x1 - x0 < 6 || y1 - y0 < 3)
		return guess;
	int s = 1;
	while ((x1 - x0 + s - 1) / s > RP_MAXW || (y1 - y0 + s - 1) / s > RP_MAXH)
		s++;
	const int gw = (x1 - x0 + s - 1) / s, gh = (y1 - y0 + s - 1) / s;

	uint8_t img[RP_MAXH][RP_MAXW];
	uint8_t mask[RP_MAXH][RP_MAXW];
	uint32_t isum[RP_MAXH + 1][RP_MAXW + 1];
	uint16_t icnt[RP_MAXH + 1][RP_MAXW + 1];

	/* Mask: scanline fill of the outline polygon at cell centres. */
	uint32_t total = 0, n_in = 0;
	for (int j = 0; j < gh; j++) {
		const float y = (float)(y0 + j * s) + 0.5f;
		float xs[CF_MAX_EYE_POINTS];
		int nx = 0;
		for (int k = 0; k < count; k++) {
			const cf_pt a = eye[k], b = eye[(k + 1) % count];
			if ((a.y <= y && b.y > y) || (b.y <= y && a.y > y)) {
				const float x = a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x);
				int q = nx++;
				while (q > 0 && xs[q - 1] > x) {
					xs[q] = xs[q - 1];
					q--;
				}
				xs[q] = x;
			}
		}
		const uint8_t *row = grey + (size_t)(y0 + j * s) * (size_t)stride;
		for (int i = 0; i < gw; i++) {
			const float x = (float)(x0 + i * s) + 0.5f;
			int inside = 0;
			for (int q = 0; q + 1 < nx; q += 2)
				if (x >= xs[q] && x < xs[q + 1])
					inside = 1;
			img[j][i] = row[x0 + i * s];
			mask[j][i] = (uint8_t)inside;
			if (inside) {
				total += img[j][i];
				n_in++;
			}
		}
	}
	if (n_in < 20)
		return guess;
	const float mean_all = (float)total / (float)n_in;
	float bright = 0;
	int nb = 0;
	for (int j = 0; j < gh; j++)
		for (int i = 0; i < gw; i++)
			if (mask[j][i] && img[j][i] >= mean_all) {
				bright += img[j][i];
				nb++;
			}
	bright = nb ? bright / (float)nb : mean_all;

	for (int i = 0; i <= gw; i++) {
		isum[0][i] = 0;
		icnt[0][i] = 0;
	}
	for (int j = 0; j < gh; j++) {
		uint32_t rs = 0;
		uint16_t rc = 0;
		isum[j + 1][0] = 0;
		icnt[j + 1][0] = 0;
		for (int i = 0; i < gw; i++) {
			if (mask[j][i]) {
				rs += img[j][i];
				rc++;
			}
			isum[j + 1][i + 1] = isum[j][i + 1] + rs;
			icnt[j + 1][i + 1] = (uint16_t)(icnt[j][i + 1] + rc);
		}
	}

	/* 1. Darkest square near the guess (grid units from here on). */
	const float R = 0.2f * w / (float)s; /* iris radius */
	int hb = (int)lrintf(0.7f * R);
	hb = hb < 1 ? 1 : hb;
	const float gx = (guess.x - (float)x0 - 0.5f) / (float)s, gy = (guess.y - (float)y0 - 0.5f) / (float)s;
	const float wg = w / (float)s;
	const int rxs = (int)ceilf(0.3f * wg), rys = (int)ceilf(0.2f * wg);
	const int area = (2 * hb + 1) * (2 * hb + 1);
	float best_score = 1e9f, best_mean = 0;
	int bi = -1, bj = -1;
	for (int j = (int)lrintf(gy) - rys; j <= (int)lrintf(gy) + rys; j++) {
		if (j < 0 || j >= gh)
			continue;
		for (int i = (int)lrintf(gx) - rxs; i <= (int)lrintf(gx) + rxs; i++) {
			if (i < 0 || i >= gw || !mask[j][i])
				continue;
			const int ax = i - hb < 0 ? 0 : i - hb, bx = i + hb + 1 > gw ? gw : i + hb + 1;
			const int ay = j - hb < 0 ? 0 : j - hb, by = j + hb + 1 > gh ? gh : j + hb + 1;
			const int c = icnt[by][bx] - icnt[ay][bx] - icnt[by][ax] + icnt[ay][ax];
			if (c < 6 || c * 10 < area * 3)
				continue;
			const uint32_t sum = isum[by][bx] - isum[ay][bx] - isum[by][ax] + isum[ay][ax];
			const float mean = (float)sum / (float)c;
			const float dx = (float)i - gx, dy = (float)j - gy;
			const float dn = (dx * dx + dy * dy) / (0.09f * wg * wg);
			const float score = mean + 20.0f * dn;
			if (score < best_score) {
				best_score = score;
				best_mean = mean;
				bi = i;
				bj = j;
			}
		}
	}
	if (bi < 0 || bright - best_mean < 15.0f)
		return guess; /* nothing dark enough: low contrast */

	/* 2. Means of gradients around the darkest square. */
	enum { MAXG = 512 };
	float gpx[MAXG], gpy[MAXG], gux[MAXG], guy[MAXG];
	float mags[MAXG];
	int ng = 0;
	double msum = 0, msum2 = 0;
	const float rr = 2.0f * R;
	for (int j = 1; j < gh - 1 && ng < MAXG; j++)
		for (int i = 1; i < gw - 1 && ng < MAXG; i++) {
			if (!(mask[j][i] && mask[j - 1][i] && mask[j + 1][i] && mask[j][i - 1] && mask[j][i + 1]))
				continue;
			const float dx = (float)(i - bi), dy = (float)(j - bj);
			if (dx * dx + dy * dy > rr * rr)
				continue;
			const float ggx = 0.5f * ((float)img[j][i + 1] - (float)img[j][i - 1]);
			const float ggy = 0.5f * ((float)img[j + 1][i] - (float)img[j - 1][i]);
			const float mg = sqrtf(ggx * ggx + ggy * ggy);
			if (mg < 1.0f)
				continue;
			gpx[ng] = (float)i;
			gpy[ng] = (float)j;
			gux[ng] = ggx / mg;
			guy[ng] = ggy / mg;
			mags[ng] = mg;
			msum += mg;
			msum2 += (double)mg * mg;
			ng++;
		}
	float cx = (float)bi, cy = (float)bj;
	if (ng >= 8) {
		const double mm = msum / ng;
		const float thr = (float)(mm + 0.3 * sqrt(fmax(msum2 / ng - mm * mm, 0)));
		int k = 0;
		for (int q = 0; q < ng; q++)
			if (mags[q] >= thr) {
				gpx[k] = gpx[q];
				gpy[k] = gpy[q];
				gux[k] = gux[q];
				guy[k] = guy[q];
				k++;
			}
		ng = k;
	}
	if (ng >= 8) {
		const float span = fmaxf(2.0f, 0.4f * R);
		float bestv = -1;
		for (float oy = -span; oy <= span + 1e-3f; oy += 0.5f)
			for (float ox = -span; ox <= span + 1e-3f; ox += 0.5f) {
				const float qx = (float)bi + ox, qy = (float)bj + oy;
				float v = 0;
				for (int q = 0; q < ng; q++) {
					const float dx = gpx[q] - qx, dy = gpy[q] - qy;
					const float dd = dx * dx + dy * dy;
					if (dd < 0.25f)
						continue;
					const float dot = (dx * gux[q] + dy * guy[q]) / sqrtf(dd);
					if (dot > 0)
						v += dot * dot;
				}
				if (v > bestv) {
					bestv = v;
					cx = qx;
					cy = qy;
				}
			}
	}

	const cf_pt res = {(float)x0 + cx * (float)s + 0.5f, (float)y0 + cy * (float)s + 0.5f};
	/* Sanity: must be inside the outline and not far from the guess. */
	const int ci = (int)lrintf(cx), cj = (int)lrintf(cy);
	if (ci < 0 || cj < 0 || ci >= gw || cj >= gh || !mask[cj][ci])
		return guess;
	const float mdx = res.x - guess.x, mdy = res.y - guess.y;
	if (mdx * mdx + mdy * mdy > 0.35f * 0.35f * w * w)
		return guess;
	return res;
}
