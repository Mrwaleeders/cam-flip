#include "pose.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * ---- The model
 *
 * Head coordinates, in inter-eye distances (between the two eye centres, each eye's centre being the
 * midpoint of its corners): x towards the image's right, y down, z towards the camera, origin between
 * the eyes, with the head as it was during the lens calibration (looking into the camera). There:
 *
 *   eye centres        (-0.5, 0, 0) and (0.5, 0, 0)
 *   nose centre        (s0, v0, NOSE_DEPTH)                  s0, v0: the calibrated side and vert
 *   each eye's corners its centre -+ (w/2, k0 w/2, CORNER_DEPTH w/2): the inner corner nearer the nose,
 *                      lower by the calibrated tilt k0 and nearer the camera, the outer one the opposite;
 *                      widths w in the ratio (1 + e0) : (1 - e0), e0 the calibrated eye asymmetry.
 *
 * The x and y of every point are the person's own (measured looking into the camera); only the two
 * depths are typical values. The head tilts by `pitch` about the eyes' axis (down is positive), then
 * turns by `yaw` about the vertical (the nose towards the image's right is positive), and is seen
 * straight on (orthographic: at 50-70 cm a face is 8-11 inter-eye distances away, so perspective only
 * scales the angles: the near eye looks a little bigger, and a head turning about the neck moves off
 * the camera's axis, so at 60 cm turns read ~1.2x and tilts ~1.1x; tests/test_pose.c measures it):
 *
 *   x' = x cos(yaw) + (z cos(pitch) - y sin(pitch)) sin(yaw)
 *   y' = y cos(pitch) + z sin(pitch)
 *
 * The eye centres land at (-+0.5 cos(yaw), 0), so every measure (in the eyes' frame) comes out exactly
 * as (t = tan(yaw)):
 *
 *   side = s0 + gn t                               gn = NOSE_DEPTH cos(pitch) - v0 sin(pitch)
 *   eyes = (e0 + ge t) / (1 + e0 ge t)             ge = CORNER_DEPTH cos(pitch) - k0 sin(pitch)
 *   vert cos(yaw) = v0 cos(pitch) + NOSE_DEPTH sin(pitch)
 *   tilt (cos(yaw)^2 - ge^2 sin(yaw)^2) / cos(yaw) = k0 cos(pitch) + CORNER_DEPTH sin(pitch)
 *
 * Yaw, given pitch: both sideways cues are linear in tan(yaw), combined by least squares with each
 * weighted by (gain / noise)^2. Pitch, given yaw: each vertical cue inverts in closed form. The nose
 * alone is not enough for pitch: tilting the head down moves it down at first, but foreshortening pulls
 * it back up past atan(NOSE_DEPTH / v0) (20-30 degrees), so it has a fold there and a far tilt would
 * read like a small one. Its inverse is taken on the rising branch and trusted by its gain, counted for
 * a nose 30% flatter than typical (and less the more the head is turned, as its yaw correction is only
 * as good as the yaw), which is zero well before any real fold: past it, pitch comes from
 * the corners alone, whose tilt keeps growing (sin(pitch)) up to 90 degrees. Looking up there is no
 * fold, and the nose, measured more precisely, dominates. The two are alternated twice (each needs the
 * other's angle for the couplings: a turned head's eyes look closer together, which alone would read as
 * a tilt, and a head tilted down has its nose less far in front of the eyes, so a turn moves it less).
 * Every step is a continuous function of the measures, so the pose never jumps: no search, no branches.
 *
 * The depths only scale the angles (a flatter nose turns "less"), they never move the zero: that comes
 * from the calibration. Typical faces: the centre of Vision's nose landmarks (outline and crest) is
 * ~16 mm in front of the eye corners' midpoints (0.26 of a ~63 mm inter-eye distance); the outer eye
 * corner is ~9 mm behind the inner one (0.30 of a ~29 mm eye width).
 */
#define NOSE_DEPTH 0.26
#define CORNER_DEPTH 0.30
#define NOSE_DEPTH_MIN (0.7 * NOSE_DEPTH) /* a flat nose: where the nose's pitch stops being trusted */
/* Typical frame-to-frame noise of each measure (Vision, 1080p, eyes ~85 px wide): the weights. */
#define SD_SIDE 0.004
#define SD_VERT 0.004
#define SD_EYES 0.012
#define SD_TILT 0.012
#define SIDE_PITCH_SD 0.08 /* ... and the nose's sideways one, per sin(pitch): see yaw_given() */
#define YAW_REL_SD 0.20    /* the yaw is known to this fraction of itself (typical depths): see pitch_given() */
#define MAX_ANGLE (80.0 * M_PI / 180.0) /* results are kept inside this (rad) */

static int finite_head(const cf_head *m)
{
	return isfinite(m->side) && isfinite(m->vert) && isfinite(m->eyes) && isfinite(m->tilt);
}

/* ---- Measuring */

static int centroid(const cf_pt *p, int n, cf_pt *c)
{
	double sx = 0, sy = 0;
	for (int i = 0; i < n; i++) {
		sx += p[i].x;
		sy += p[i].y;
	}
	c->x = (float)(sx / n);
	c->y = (float)(sy / n);
	return isfinite(c->x) && isfinite(c->y);
}

/* The outline's extreme points along `axis` (its corners). */
static void corners(const cf_pt *p, int n, cf_pt axis, cf_pt *lo, cf_pt *hi)
{
	float plo = p[0].x * axis.x + p[0].y * axis.y, phi = plo;
	*lo = *hi = p[0];
	for (int i = 1; i < n; i++) {
		const float d = p[i].x * axis.x + p[i].y * axis.y;
		if (d < plo) {
			plo = d;
			*lo = p[i];
		}
		if (d > phi) {
			phi = d;
			*hi = p[i];
		}
	}
}

int cf_head_measure(const cf_pt *eye_a, int na, const cf_pt *eye_b, int nb, const cf_pt *nose, int nn,
		    cf_head *out)
{
	memset(out, 0, sizeof(*out));
	if (!eye_a || !eye_b || !nose || na < 3 || nb < 3 || nn < 1)
		return 0;
	na = na < CF_HEAD_MAX_POINTS ? na : CF_HEAD_MAX_POINTS;
	nb = nb < CF_HEAD_MAX_POINTS ? nb : CF_HEAD_MAX_POINTS;
	nn = nn < CF_HEAD_MAX_POINTS ? nn : CF_HEAD_MAX_POINTS;
	cf_pt ca, cb, cn;
	if (!centroid(eye_a, na, &ca) || !centroid(eye_b, nb, &cb) || !centroid(nose, nn, &cn))
		return 0;
	if (cb.x < ca.x) { /* a = the eye on the image's left */
		const cf_pt *tp = eye_a;
		eye_a = eye_b;
		eye_b = tp;
		const int tn = na;
		na = nb;
		nb = tn;
		const cf_pt tc = ca;
		ca = cb;
		cb = tc;
	}
	/* Corners along the line through the outlines' centres, then the eye centres as the corners'
	 * midpoints: the corners are fixed to the skull, the outlines move with the lids. */
	float ax = cb.x - ca.x, ay = cb.y - ca.y, len = sqrtf(ax * ax + ay * ay);
	if (!(len > 1.0f))
		return 0;
	const cf_pt axis0 = {ax / len, ay / len};
	cf_pt a_out, a_in, b_in, b_out; /* left eye: outer corner on the left; right eye: inner on the left */
	corners(eye_a, na, axis0, &a_out, &a_in);
	corners(eye_b, nb, axis0, &b_in, &b_out);
	const cf_pt ea = {0.5f * (a_out.x + a_in.x), 0.5f * (a_out.y + a_in.y)};
	const cf_pt eb = {0.5f * (b_in.x + b_out.x), 0.5f * (b_in.y + b_out.y)};
	ax = eb.x - ea.x;
	ay = eb.y - ea.y;
	const float iod = sqrtf(ax * ax + ay * ay);
	if (!(iod > 1.0f))
		return 0;
	const cf_pt axis = {ax / iod, ay / iod}, down = {-axis.y, axis.x};
	const cf_pt o = {0.5f * (ea.x + eb.x), 0.5f * (ea.y + eb.y)};
	const float wa = hypotf(a_in.x - a_out.x, a_in.y - a_out.y), wb = hypotf(b_out.x - b_in.x, b_out.y - b_in.y);
	if (!(wa > 0.05f * iod) || !(wb > 0.05f * iod))
		return 0;
	const float nx = cn.x - o.x, ny = cn.y - o.y;
	out->side = (nx * axis.x + ny * axis.y) / iod;
	out->vert = (nx * down.x + ny * down.y) / iod;
	out->eyes = (wa - wb) / (wa + wb);
	/* inner minus outer corner, along "down": positive when the outer corner is the higher one */
	const float rise_a = ((a_in.x - a_out.x) * down.x + (a_in.y - a_out.y) * down.y) / wa;
	const float rise_b = ((b_in.x - b_out.x) * down.x + (b_in.y - b_out.y) * down.y) / wb;
	out->tilt = 0.5f * (rise_a + rise_b);
	if (!finite_head(out)) {
		memset(out, 0, sizeof(*out));
		return 0;
	}
	return 1;
}

/* ---- Estimating */

typedef struct {
	double s0, v0, e0, k0; /* the frontal face (calibrated measures) */
	double side, vert, eyes, tilt; /* seen now */
} pose_in;

static double clampd(double v, double lo, double hi)
{
	return v < lo ? lo : (v > hi ? hi : v);
}

/* tan(yaw) given pitch: least squares over the two sideways cues (see the model above). */
static double yaw_given(const pose_in *in, double pitch)
{
	const double cp = cos(pitch), sp = sin(pitch);
	const double gn = NOSE_DEPTH * cp - in->v0 * sp, ge = CORNER_DEPTH * cp - in->k0 * sp;
	const double dn = in->side - in->s0;
	const double de = (in->eyes - in->e0) / (1.0 - in->e0 * in->eyes);
	/* The calibrated s0 is the face's own asymmetry plus whatever turn the head had while calibrating,
	 * and only the second part shrinks with gn as the head tilts: the nose is trusted less for yaw the
	 * further the head is tilted (the eyes' cue hardly depends on pitch). */
	const double sn = SIDE_PITCH_SD * sin(fabs(pitch));
	const double wn = 1.0 / (SD_SIDE * SD_SIDE + sn * sn), we = 1.0 / (SD_EYES * SD_EYES);
	const double den = wn * gn * gn + we * ge * ge;
	return den > 1e-9 ? (wn * gn * dn + we * ge * de) / den : 0.0;
}

/* Pitch given yaw (and the previous pitch, for the corners' small yaw coupling). */
static double pitch_given(const pose_in *in, double yaw, double pitch_prev)
{
	const double cy = cos(yaw), sy = sin(yaw);
	/* Nose, on the rising branch: v0 cos + NOSE_DEPTH sin = R sin... = R cos(pitch - a), pitch <= a. */
	const double rn = hypot(in->v0, NOSE_DEPTH), an = atan2(NOSE_DEPTH, in->v0);
	const double pn = an - acos(clampd(in->vert * cy / rn, -1.0, 1.0));
	/* Corners: k0 cos + CORNER_DEPTH sin = R sin(pitch + b), monotonic over the whole range. */
	const double ge = CORNER_DEPTH * cos(pitch_prev) - in->k0 * sin(pitch_prev);
	const double y = in->tilt * (cy * cy - ge * ge * sy * sy) / cy;
	const double rt = hypot(in->k0, CORNER_DEPTH), bt = atan2(in->k0, CORNER_DEPTH);
	const double pt = asin(clampd(y / rt, -1.0, 1.0)) - bt;
	/* Weights: (gain / noise)^2 of each. The nose's gain is counted for a flat nose and at the larger of
	 * the two answers, so it hands over to the corners well before its fold, and is not trusted when it
	 * reads close to the fold itself (there its inverse is steep: a little error in the yaw correction
	 * above reads as a big tilt). The same up and down: the nose alone would do for a tilt up, but then
	 * up and down would read with different cues, and a face whose depths are not the typical ones would
	 * let go sooner one way than the other. */
	const double ap = fmax(fabs(pt), fabs(pn));
	/* The nose's vert also carries the yaw's error through the cos(yaw) above (a turned head's eyes look
	 * closer together, which the nose's height is measured against): with the yaw known to YAW_REL_SD
	 * of itself, that is vert sin(yaw) YAW_REL_SD yaw. Small turns leave the nose in charge; a large
	 * one hands pitch to the corners, whose tilt hardly depends on the yaw. */
	const double sv = in->vert * fabs(sy) * YAW_REL_SD * fabs(yaw);
	const double gn = fmax(NOSE_DEPTH_MIN * cos(ap) - in->v0 * sin(ap), 0.0) / sqrt(SD_VERT * SD_VERT + sv * sv);
	const double gt = fmax(CORNER_DEPTH * cos(fabs(pt)) - fabs(in->k0) * sin(fabs(pt)), 0.0) / SD_TILT;
	const double wn = gn * gn, wt = gt * gt;
	return wn + wt > 1e-9 ? (wn * pn + wt * pt) / (wn + wt) : pt;
}

static int ref_ok(const cf_head_ref *ref)
{
	return ref && ref->ok && isfinite(ref->side) && isfinite(ref->vert) && isfinite(ref->eyes) &&
	       isfinite(ref->tilt) && fabsf(ref->side) < 1.0f && ref->vert > 0.05f && ref->vert < 2.0f &&
	       fabsf(ref->eyes) < 0.5f && fabsf(ref->tilt) < 1.0f;
}

void cf_head_angles(const cf_head *m, const cf_head_ref *ref, float *yaw_deg, float *pitch_deg)
{
	*yaw_deg = *pitch_deg = 0;
	if (!m || !finite_head(m) || fabsf(m->eyes) >= 0.99f)
		return;
	const int cal = ref_ok(ref);
	/* Without a calibration: a typical face, symmetric and level. Pitch is then not estimated (it
	 * would need the nose height, which differs between faces and camera heights far too much), so the
	 * couplings use pitch 0. */
	const pose_in in = {
		.s0 = cal ? ref->side : 0.0,
		.v0 = cal ? ref->vert : 0.5,
		.e0 = cal ? ref->eyes : 0.0,
		.k0 = cal ? ref->tilt : 0.0,
		.side = m->side,
		.vert = m->vert,
		.eyes = m->eyes,
		.tilt = m->tilt,
	};
	double yaw = atan(yaw_given(&in, 0.0)), pitch = 0.0;
	if (cal) {
		for (int it = 0; it < 2; it++) {
			pitch = pitch_given(&in, yaw, pitch);
			yaw = atan(yaw_given(&in, pitch));
		}
	}
	yaw = clampd(yaw, -MAX_ANGLE, MAX_ANGLE) * (180.0 / M_PI);
	pitch = clampd(pitch, -MAX_ANGLE, MAX_ANGLE) * (180.0 / M_PI);
	if (!cal) /* known only to a few degrees: report it that much smaller (still continuous) */
		yaw = yaw > CF_HEAD_DEFAULT_YAW_MARGIN ? yaw - CF_HEAD_DEFAULT_YAW_MARGIN
		      : yaw < -CF_HEAD_DEFAULT_YAW_MARGIN ? yaw + CF_HEAD_DEFAULT_YAW_MARGIN : 0.0;
	*yaw_deg = isfinite(yaw) ? (float)yaw : 0.0f;
	*pitch_deg = isfinite(pitch) ? (float)pitch : 0.0f;
}

/* ---- Calibration */

static int cmpf(const void *a, const void *b)
{
	const float x = *(const float *)a, y = *(const float *)b;
	return (x > y) - (x < y);
}

static float median_of(float *v, int n)
{
	qsort(v, (size_t)n, sizeof(float), cmpf);
	return (n & 1) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

cf_head_ref cf_head_reference(cf_head *samples, int n)
{
	cf_head_ref ref;
	memset(&ref, 0, sizeof(ref));
	if (!samples || n < CF_HEAD_MIN_SAMPLES)
		return ref;
	float *v = malloc(sizeof(float) * (size_t)n);
	if (!v)
		return ref;
	int k = 0;
	for (int i = 0; i < n; i++)
		if (finite_head(&samples[i]))
			samples[k++] = samples[i];
	if (k >= CF_HEAD_MIN_SAMPLES) {
		for (int i = 0; i < k; i++)
			v[i] = samples[i].side;
		ref.side = median_of(v, k);
		for (int i = 0; i < k; i++)
			v[i] = samples[i].vert;
		ref.vert = median_of(v, k);
		for (int i = 0; i < k; i++)
			v[i] = samples[i].eyes;
		ref.eyes = median_of(v, k);
		for (int i = 0; i < k; i++)
			v[i] = samples[i].tilt;
		ref.tilt = median_of(v, k);
		ref.ok = 1;
	}
	free(v);
	return ref;
}

/* ---- Smoothing */

static float med3(float a, float b, float c)
{
	return fmaxf(fminf(a, b), fminf(fmaxf(a, b), c));
}

int cf_pose_update(cf_pose *p, const cf_head *m, const cf_head_ref *ref, double t, float *yaw_deg,
		   float *pitch_deg)
{
	*yaw_deg = *pitch_deg = 0;
	const int fresh = m && finite_head(m);
	const int stale = !p->ready || t - p->t > CF_POSE_HOLD || t < p->t;
	if (fresh && stale) { /* first measure, or after a long gap: start afresh */
		p->hist[0] = *m;
		p->hist_n = 1;
		p->smooth = *m;
		p->t = t;
		p->ready = 1;
	} else if (fresh) {
		if (p->hist_n == 3)
			memmove(&p->hist[0], &p->hist[1], 2 * sizeof(p->hist[0]));
		else
			p->hist_n++;
		p->hist[p->hist_n - 1] = *m;
		cf_head md = *m;
		if (p->hist_n == 3) {
			const cf_head *h = p->hist;
			md.side = med3(h[0].side, h[1].side, h[2].side);
			md.vert = med3(h[0].vert, h[1].vert, h[2].vert);
			md.eyes = med3(h[0].eyes, h[1].eyes, h[2].eyes);
			md.tilt = med3(h[0].tilt, h[1].tilt, h[2].tilt);
		}
		double dt = t - p->t;
		if (!(dt > 1e-4))
			dt = 1.0 / 30.0;
		const float a = (float)(1.0 - exp(-dt / CF_POSE_TAU));
		p->smooth.side += a * (md.side - p->smooth.side);
		p->smooth.vert += a * (md.vert - p->smooth.vert);
		p->smooth.eyes += a * (md.eyes - p->smooth.eyes);
		p->smooth.tilt += a * (md.tilt - p->smooth.tilt);
		p->t = t;
	} else if (stale) {
		p->ready = 0;
		p->hist_n = 0;
		p->yaw = p->pitch = 0;
		return 0;
	}
	cf_head_angles(&p->smooth, ref, yaw_deg, pitch_deg);
	p->yaw = *yaw_deg;
	p->pitch = *pitch_deg;
	return 1;
}
