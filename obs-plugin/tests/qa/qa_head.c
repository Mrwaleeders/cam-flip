/*
 * QA harness (synthetic only, not part of the suite): head-pitch / yaw compensation, lid restoration and
 * the far fade, driven by a 3D synthetic face seen through a pinhole camera.
 *
 *   cd obs-plugin && clang -O2 -Wall -Wno-unused-function -Isrc tests/qa/qa_head.c src/gaze.c src/warp.c src/pose.c \
 *     -lm -o /tmp/qa && /tmp/qa [scenarios] [verbose]
 *
 * Scenarios (letters, default abcdegw): a reading, b lens with the head down, c away and back, d noise /
 * nods / turns / blinks, e fps / strength / old calibrations, g pupil under-reading, w warp; only when
 * named: h and i side looks at lens height (i: 40 noise seeds each), k reading with a head-heavy posture.
 *
 * The face: test_pose.c's typical 3D face (8-point eye outlines, 13 nose points, neck pivot), with the eyes
 * turning in the head (iris R = 11.9 mm = EYE_K x 29 mm eye), the upper lid following LID_UP of the iris'
 * vertical travel in the head, the lower lid LID_LO of it, plus the reading narrowing; the head pitches
 * (down +) and turns (towards the image's right +) about the neck. Gaze targets are real points: the lens,
 * lines of notes 5-15 cm under it, a keyboard, something to the side.
 *
 * Two eye models (IRIS_Z): 0 = the iris moves against the eye corners only with the eyes' turn in the head
 * (what gaze.c's head term assumes); > 0 = the iris sits that far (mm) in front of the corners' plane, as
 * on a real eye, so a head tilt moves it against the corners too (perspective of a rotating eyeball).
 *
 * Everything is measured on eye 0 (image left), along the face's up (the face is level: roll 0):
 *   lift = -sy, lid = -lid_sy, low = low_sy (px, 0 when not warped), side = sx.
 *   G_true  how far below "looking into the lens" the real iris sits (px), at this head pose;
 *   G_seen  the same for the warped iris (G_true - lift): what the viewer sees. 0 = eye contact.
 */
#include "gaze.h"
#include "pose.h"
#include "warp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cf_warp_debug_grid(const cf_eye *e, float x0, float y0, float step, int nx, int ny, float *dx, float *dy);
int cf_warp_debug_field(const cf_eye *e, float x, float y, float *dx, float *dy, float *max_slope, float *scale);

#define D2R 0.017453292519943295
#define EYE_W 29.0  /* mm */
#define EYE_H 8.9   /* mm open (75 x 23 px eye at 0.6 m) */
#define IRIS_R 11.9 /* mm (EYE_K x EYE_W) */
#define IOD 63.0
#define MAXF 4000

/* ---- RNG */
static unsigned long long rs = 88172645463325252ULL;
static void seed(unsigned long long s) { rs = s * 2654435761ULL + 88172645463325252ULL; }
static double urand(void)
{
	rs ^= rs << 13;
	rs ^= rs >> 7;
	rs ^= rs << 17;
	return (double)(rs >> 11) / 9007199254740992.0;
}
static double gauss(double sd)
{
	if (sd <= 0)
		return 0;
	double u1 = urand(), u2 = urand();
	if (u1 < 1e-12)
		u1 = 1e-12;
	return sd * sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

/* ---- World */
typedef struct {
	double D;       /* mm, camera to eyes */
	double focal;   /* px */
	double iris_z;  /* mm: iris in front of the corners' plane (0 = gaze.c's model) */
	double lid_up, lid_lo; /* lids follow this share of the iris' vertical travel in the head */
	double bulge;          /* mm the lid landmarks sit in front of the corners' plane (mid-lid) */
	double sd_eye, sd_nose, sd_pupil, sd_vis;
	double vis_gain;
	double meas_gain; /* refined pupil reads this share of the true iris travel against the corners */
} world;

typedef struct {
	double pitch, yaw; /* head, deg */
	double tgt[3];     /* what the eyes look at, mm: x right, y down, z towards the camera (the lens is at
			    * (0, 0, D)); both eyes turn in the head to it (vergence included) */
	double up_drop, lo_rise; /* reading narrowing, eye widths */
	double shut;       /* blink 0..1 */
	double drag;       /* pupil point dragged down with a closing lid (eye widths) */
	int phase;         /* scenario label */
} st;

typedef struct {
	double x, y;
} v2;

static void rot(const world *w, const st *s, const double P[3], double out[3])
{
	const double cp = cos(s->pitch * D2R), sp = sin(s->pitch * D2R), cy = cos(s->yaw * D2R), sy = sin(s->yaw * D2R);
	const double pv[3] = {0, 70, -90};
	const double x = P[0] - pv[0], y = P[1] - pv[1], z = P[2] - pv[2];
	const double y1 = y * cp + z * sp, z1 = -y * sp + z * cp;
	const double x2 = x * cy + z1 * sy, z2 = -x * sy + z1 * cy;
	out[0] = x2 + pv[0];
	out[1] = y1 + pv[1];
	out[2] = z2 + pv[2];
	(void)w;
}

static v2 proj(const world *w, const st *s, const double P[3])
{
	double Q[3];
	rot(w, s, P, Q);
	const double k = w->focal / (w->D - Q[2]);
	const v2 r = {960 + Q[0] * k, 540 + Q[1] * k};
	return r;
}

/* Iris centre in head coordinates (eye e: 0 image left). */
static void iris3(const world *w, int e, double ev, double eh, double P[3])
{
	const double xc = (e == 0 ? -0.5 : 0.5) * IOD;
	P[0] = xc + IRIS_R * sin(eh * D2R) * cos(ev * D2R);
	P[1] = IRIS_R * sin(ev * D2R);
	if (w->iris_z <= 0)
		P[2] = 0;
	else
		P[2] = (w->iris_z - IRIS_R) + IRIS_R * cos(ev * D2R) * cos(eh * D2R);
}

/* Vector rotation by the head (no pivot) and its inverse. */
static void vrot(const st *s, const double v[3], double o[3], int inverse)
{
	const double cp = cos(s->pitch * D2R), sp = sin(s->pitch * D2R), cy = cos(s->yaw * D2R), sy = sin(s->yaw * D2R);
	if (!inverse) {
		const double y1 = v[1] * cp + v[2] * sp, z1 = -v[1] * sp + v[2] * cp;
		o[0] = v[0] * cy + z1 * sy;
		o[1] = y1;
		o[2] = -v[0] * sy + z1 * cy;
	} else {
		const double x = v[0] * cy - v[2] * sy, z1 = v[0] * sy + v[2] * cy;
		o[0] = x;
		o[1] = v[1] * cp - z1 * sp;
		o[2] = v[1] * sp + z1 * cp;
	}
}

/* Eye e's angles in the head (deg, down + / towards the image's right +) to look at `tgt`. */
static void eye_angles(const world *w, const st *s, int e, const double tgt[3], double *ev, double *eh)
{
	const double iz = w->iris_z > 0 ? w->iris_z : 6.0;
	const double C[3] = {(e == 0 ? -0.5 : 0.5) * IOD, 0, iz - IRIS_R};
	double Cw[3];
	rot(w, s, C, Cw);
	const double d[3] = {tgt[0] - Cw[0], tgt[1] - Cw[1], tgt[2] - Cw[2]};
	double h[3];
	vrot(s, d, h, 1);
	*ev = atan2(h[1], h[2]) / D2R;
	*eh = atan2(h[0], hypot(h[1], h[2])) / D2R;
}

/* Noise-free reference points of eye 0: corner midpoint, iris now, iris looking into the lens. */
typedef struct {
	v2 mid, iris, lens;
	double eye_px;
} truth;

static void face_make(const world *w, const st *s, cf_face *f, truth *tr, int noisy)
{
	memset(f, 0, sizeof(*f));
	f->found = 1;
	f->image_width = 1920;
	f->image_height = 1080;
	const double k0 = w->focal / w->D;
	f->face_width = (float)(150 * k0);
	for (int e = 0; e < 2; e++) {
		double ev, eh;
		eye_angles(w, s, e, s->tgt, &ev, &eh);
		const double rv = IRIS_R * sin(ev * D2R);
		cf_pt *pts = e == 0 ? f->left_eye : f->right_eye;
		const double side = e == 0 ? -1.0 : 1.0, xc = side * 0.5 * IOD;
		v2 c0 = {0, 0}, c1 = {0, 0};
		for (int i = 0; i < 8; i++) {
			const double a = 2 * M_PI * i / 8.0, u = cos(a), v = sin(a), out_u = side * u;
			double y = v * EYE_H / 2 - out_u * 1.0;
			if (v < -1e-6) /* upper lid: down with the iris, plus the reading drop */
				y += (w->lid_up * rv + s->up_drop * EYE_W) * -v;
			else if (v > 1e-6) /* lower lid: a little with the iris, up by the reading rise */
				y += (w->lid_lo * rv - s->lo_rise * EYE_W) * v;
			const double cl = -out_u * 1.0; /* corner line height here */
			y = cl + (y - cl) * (1 - s->shut);
			const double P[3] = {xc + u * EYE_W / 2, y, -out_u * 4.5 + w->bulge * (1 - u * u)};
			const v2 q = proj(w, s, P);
			if (i == 0)
				c0 = q;
			if (i == 4)
				c1 = q;
			pts[i].x = (float)(q.x + (noisy ? gauss(w->sd_eye) : 0));
			pts[i].y = (float)(q.y + (noisy ? gauss(w->sd_eye) : 0));
		}
		const v2 mid = {0.5 * (c0.x + c1.x), 0.5 * (c0.y + c1.y)};
		double P[3];
		iris3(w, e, ev, eh, P);
		const v2 ir = proj(w, s, P);
		/* refined pupil: meas_gain of the true travel against the corners, dragged by a closing lid */
		const double eye_px = hypot(c1.x - c0.x, c1.y - c0.y);
		v2 pr = {mid.x + w->meas_gain * (ir.x - mid.x), mid.y + w->meas_gain * (ir.y - mid.y)};
		pr.y += s->drag * eye_px;
		const cf_pt pupil = {(float)(pr.x + (noisy ? gauss(w->sd_pupil) : 0)),
				     (float)(pr.y + (noisy ? gauss(w->sd_pupil) : 0))};
		const cf_pt vp = {(float)(mid.x + w->vis_gain * (pr.x - mid.x) + (noisy ? gauss(w->sd_vis) : 0)),
				  (float)(mid.y + w->vis_gain * (pr.y - mid.y) + (noisy ? gauss(w->sd_vis) : 0))};
		if (e == 0) {
			f->left_count = 8;
			f->left_pupil = pupil;
			f->left_pupil_vision = vp;
			if (tr) {
				tr->mid = mid;
				tr->iris = ir;
				tr->eye_px = eye_px;
				/* the eyes on the lens with this head */
				const double lens[3] = {0, 0, w->D};
				double ev_l, eh_l, L[3];
				eye_angles(w, s, 0, lens, &ev_l, &eh_l);
				iris3(w, 0, ev_l, eh_l, L);
				tr->lens = proj(w, s, L);
			}
		} else {
			f->right_count = 8;
			f->right_pupil = pupil;
			f->right_pupil_vision = vp;
		}
	}
	f->has_vision_pupils = 1;
	/* Nose (test_pose.c's 13 points). */
	static const double nose[13][3] = {{0, 6, 13},  {0, 16, 18},  {0, 26, 23},  {0, 36, 28}, {-17, 37, 6},
					   {-14, 44, 10}, {-7, 47, 15}, {0, 47, 18}, {7, 47, 15}, {14, 44, 10},
					   {17, 37, 6},  {-8, 40, 20}, {8, 40, 20}};
	cf_pt np[13];
	for (int i = 0; i < 13; i++) {
		const v2 q = proj(w, s, nose[i]);
		np[i].x = (float)(q.x + (noisy ? gauss(w->sd_nose) : 0));
		np[i].y = (float)(q.y + (noisy ? gauss(w->sd_nose) : 0));
	}
	f->has_head_geom = cf_head_measure(f->left_eye, f->left_count, f->right_eye, f->right_count, np, 13, &f->head);
}

/* ---- Calibration through the API */
typedef struct {
	int old_lids; /* 1: calibrated before lid baselines (0) */
	int old_head; /* 1: no head reference */
	double notes_y; /* mm under the lens the notes calibration looks at */
	double kp;      /* head share of the gaze's pitch while reading */
	double strength;
	double fps;
} calib_opts;

static double total_v(const world *w, double y_mm) { return atan(y_mm / w->D) / D2R; }

static cf_gaze_params calibrate(const world *w, const calib_opts *o)
{
	cf_gaze_params p = {.strength = 1.0f, .max_glance = 0.15f};
	cf_gaze g;
	cf_gaze_init(&g);
	cf_eye out[2];
	double t = 10.0;
	const double dt = 1.0 / o->fps;
	cf_calib_result lens = {0}, notes = {0};
	int got = 0;
	cf_gaze_start_calibration(&g, CF_CALIB_LENS);
	for (int i = 0; i < 400 && got <= 0; i++, t += dt) {
		const st s = {.tgt = {0, 0, w->D}};
		cf_face f;
		face_make(w, &s, &f, NULL, 1);
		cf_gaze_update(&g, &f, &p, t, out);
		got = cf_gaze_take_calibration(&g, &lens);
	}
	if (got <= 0)
		printf("!! lens calibration failed\n");
	p.baseline = lens.offset;
	p.baseline_open = lens.openness;
	p.head_ref = lens.head;
	p.baseline_upper = lens.upper;
	p.baseline_lower = lens.lower;
	got = 0;
	cf_gaze_start_calibration(&g, CF_CALIB_NOTES);
	const double v = total_v(w, o->notes_y), hp = o->kp * v;
	for (int i = 0; i < 400 && got <= 0; i++, t += dt) {
		const st s = {.pitch = hp, .tgt = {0, o->notes_y, w->D}, .up_drop = 0.005, .lo_rise = 0.012};
		cf_face f;
		face_make(w, &s, &f, NULL, 1);
		cf_gaze_update(&g, &f, &p, t, out);
		got = cf_gaze_take_calibration(&g, &notes);
	}
	if (got <= 0)
		printf("!! notes calibration failed\n");
	p.has_notes = cf_gaze_notes_valid(p.baseline, notes.offset);
	p.notes_offset = notes.offset;
	p.notes_side = notes.side;
	p.notes_open = notes.openness;
	p.has_vis_calib = lens.has_vis && notes.has_vis;
	p.vis_baseline = lens.vis_offset;
	p.vis_notes = notes.vis_offset;
	if (o->old_lids)
		p.baseline_upper = p.baseline_lower = 0;
	if (o->old_head)
		p.head_ref.ok = 0;
	p.strength = (float)o->strength;
	return p;
}

/* ---- Runner */
typedef struct {
	int n;
	st s[MAXF];
	int sacc[MAXF]; /* the eyes jumped this frame */
} script;

typedef struct {
	float lift[MAXF], lid[MAXF], low[MAXF], side[MAXF];
	float gt[MAXF], gs[MAXF];     /* G_true, G_seen (px, + = below the lens) */
	float ht[MAXF], hs[MAXF];     /* sideways: true and seen (px, + = towards the image's right of the lens) */
	float eye_px;
	float notes_glance, limit_est;
	float pitch_meas[MAXF];
	float factor[MAXF], lockf[MAXF];
	float ev[MAXF], eh[MAXF];
	float dl[MAXF], ul[MAXF], hv[MAXF], sl[MAXF], hh[MAXF], down[MAXF], ff[MAXF], vo[MAXF];
	int nn[MAXF];
} trace;

static void run(const world *w, const cf_gaze_params *p, const script *sc, double fps, unsigned long long sd,
		trace *tr)
{
	cf_gaze g;
	cf_gaze_init(&g);
	cf_eye out[2];
	seed(sd);
	double t = 100.0;
	const double dt = 1.0 / fps;
	for (int i = 0; i < sc->n; i++, t += dt) {
		cf_face f;
		truth th;
		face_make(w, &sc->s[i], &f, &th, 1);
		const int n = cf_gaze_update(&g, &f, p, t, out);
		tr->nn[i] = n;
		tr->lift[i] = n ? -out[0].sy : 0;
		tr->lid[i] = n ? -out[0].lid_sy : 0;
		tr->low[i] = n ? out[0].low_sy : 0;
		tr->side[i] = n ? out[0].sx : 0;
		tr->gt[i] = (float)(th.iris.y - th.lens.y);
		tr->gs[i] = tr->gt[i] - tr->lift[i];
		tr->ht[i] = (float)(th.iris.x - th.lens.x);
		tr->hs[i] = tr->ht[i] + tr->side[i];
		tr->eye_px = (float)th.eye_px;
		tr->pitch_meas[i] = g.pose.pitch;
		tr->factor[i] = g.factor;
		tr->dl[i] = g.d_lock;
		tr->ul[i] = g.up_lock;
		tr->hv[i] = g.head_v_lock;
		tr->sl[i] = g.side_lock;
		tr->hh[i] = g.head_h_lock;
		tr->down[i] = (float)g.rb_down.shown;
		tr->ff[i] = g.far_fade;
		tr->vo[i] = g.vor_off;
		double ev, eh;
		eye_angles(w, &sc->s[i], 0, sc->s[i].tgt, &ev, &eh);
		tr->ev[i] = (float)ev;
		tr->eh[i] = (float)eh;
		tr->lockf[i] = g.lock_f;
	}
	tr->notes_glance = p->baseline - p->notes_offset;
}

/* Frames to leave out of step statistics: a saccade frame and the next `after` frames. */
static int near_sacc(const script *sc, int i, int after)
{
	for (int k = 0; k <= after && i - k >= 0; k++)
		if (sc->sacc[i - k])
			return 1;
	return 0;
}

typedef struct {
	float step_lift, step_lid, step_low, step_side; /* largest frame-to-frame change outside saccades */
	float rms_lift, rms_lid;                         /* RMS of it */
	float jerk;                                      /* largest change of the lift's change */
} steps;

static steps step_stats(const script *sc, const trace *tr, int i0, int i1, int after, double fps)
{
	steps r = {0};
	double s2 = 0, s2l = 0;
	int c = 0;
	const float k = (float)(fps / 30.0); /* per 1/30 s */
	float prev_d = 0;
	int have = 0;
	for (int i = i0 + 1; i < i1; i++) {
		if (near_sacc(sc, i, after)) {
			have = 0;
			continue;
		}
		const float dl = tr->lift[i] - tr->lift[i - 1];
		r.step_lift = fmaxf(r.step_lift, fabsf(dl) * k);
		r.step_lid = fmaxf(r.step_lid, fabsf(tr->lid[i] - tr->lid[i - 1]) * k);
		r.step_low = fmaxf(r.step_low, fabsf(tr->low[i] - tr->low[i - 1]) * k);
		r.step_side = fmaxf(r.step_side, fabsf(tr->side[i] - tr->side[i - 1]) * k);
		s2 += dl * dl;
		s2l += (tr->lid[i] - tr->lid[i - 1]) * (tr->lid[i] - tr->lid[i - 1]);
		if (have)
			r.jerk = fmaxf(r.jerk, fabsf(dl - prev_d));
		prev_d = dl;
		have = 1;
		c++;
	}
	r.rms_lift = c ? (float)sqrt(s2 / c) : 0;
	r.rms_lid = c ? (float)sqrt(s2l / c) : 0;
	return r;
}

/* ---- Scenario building */
static void add(script *sc, const st *s, int sacc)
{
	if (sc->n >= MAXF)
		return;
	sc->s[sc->n] = *s;
	sc->sacc[sc->n] = sacc;
	sc->n++;
}

/* Head dynamics: first-order lag of the head towards a share of the gaze. */
typedef struct {
	double pitch, yaw;
} head_st;

static void head_step(head_st *h, double tp, double ty, double tau, double dt)
{
	const double a = 1 - exp(-dt / tau);
	h->pitch += a * (tp - h->pitch);
	h->yaw += a * (ty - h->yaw);
}

/* Holds the gaze (total angles) for `secs`, head following (kp, ky) with lag tau; nod: amplitude (deg) and
 * frequency of a head nod the eyes compensate (VOR). Returns frames added. */
typedef struct {
	double fps, D;
	double kp, ky, tau;
	double up_drop, lo_rise; /* while reading */
	double nod_amp, nod_hz, turn_amp, turn_hz;
	double t;
} gen;

static void hold_tgt(script *sc, gen *G, head_st *h, const double tgt[3], double secs, int phase, int first_sacc,
		     double narrow)
{
	const double dt = 1.0 / G->fps;
	const int n = (int)lround(secs * G->fps);
	const double v = atan2(tgt[1], tgt[2]) / D2R, hz = atan2(tgt[0], tgt[2]) / D2R;
	for (int i = 0; i < n; i++) {
		head_step(h, G->kp * v, G->ky * hz, G->tau, dt);
		const double nod = G->nod_amp * sin(2 * M_PI * G->nod_hz * G->t);
		const double turn = G->turn_amp * sin(2 * M_PI * G->turn_hz * G->t);
		st s = {0};
		s.pitch = h->pitch + nod;
		s.yaw = h->yaw + turn;
		memcpy(s.tgt, tgt, sizeof(s.tgt));
		s.up_drop = narrow * G->up_drop;
		s.lo_rise = narrow * G->lo_rise;
		s.phase = phase;
		add(sc, &s, first_sacc && i == 0);
		G->t += dt;
	}
}

/* ---- Reporting helpers */
static float maxf_range(const float *a, int i0, int i1)
{
	float m = -1e9f;
	for (int i = i0; i < i1; i++)
		m = fmaxf(m, a[i]);
	return m;
}
static float minf_range(const float *a, int i0, int i1)
{
	float m = 1e9f;
	for (int i = i0; i < i1; i++)
		m = fminf(m, a[i]);
	return m;
}
static float mean_range(const float *a, int i0, int i1)
{
	double s = 0;
	for (int i = i0; i < i1; i++)
		s += a[i];
	return i1 > i0 ? (float)(s / (i1 - i0)) : 0;
}

static world W0(double D)
{
	world w = {.D = D, .focal = 1550, .iris_z = 0, .lid_up = 0.4, .lid_lo = 0.0, .bulge = 1.0, .sd_eye = 0.3, .sd_nose = 0.3,
		   .sd_pupil = 0.3, .sd_vis = 0.2, .vis_gain = 0.6, .meas_gain = 1.0};
	return w;
}

#include "qa_scen.c"
