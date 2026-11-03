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

/* The same face with the head tilted (rolled) by roll_deg, clockwise on screen, about (x, y). */
static cf_face face_rolled(float x, float y, float pupil_down, float eye_h, float roll_deg)
{
	cf_face f = face_at(x, y, pupil_down, eye_h, 0);
	const float c = cosf(roll_deg * 0.017453293f), s = sinf(roll_deg * 0.017453293f);
	cf_pt *all[2] = {f.left_eye, f.right_eye};
	cf_pt *pupils[2] = {&f.left_pupil, &f.right_pupil};
	for (int e = 0; e < 2; e++) {
		for (int i = 0; i <= 8; i++) {
			cf_pt *q = i < 8 ? &all[e][i] : pupils[e];
			const float dx = q->x - x, dy = q->y - y;
			q->x = x + c * dx - s * dy;
			q->y = y + s * dx + c * dy;
		}
	}
	return f;
}

static float noise(float amp)
{
	return ((float)rand() / (float)RAND_MAX - 0.5f) * 2 * amp;
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

/*
 * A more realistic eye for cf_refine_pupil, as Apple Vision sees it: curved lids, Vision's coarse
 * 6-point outline (both corners, two points on each lid: a hexagon that cuts the lids' corners), an
 * iris cut by both lids, a darker pupil and limbal ring, a catchlight, a dark lash band along the
 * upper lid (inside the hexagon), the lid's shadow on the sclera, optionally the screen's reflection on
 * the cornea, and per-pixel noise. Edges are anti-aliased (blurred over `blur` px), so a sub-pixel move
 * changes the image smoothly, like a camera.
 * Sizes for a 1080p MacBook camera: eye 86 px wide, opening ~25 px, iris radius 17 px; k scales them
 * (k = 2: a 4K-ish 172 px eye).
 */
typedef struct {
	float cx, cy;       /* corner midpoint */
	float hw;           /* half eye width */
	float up_h, lo_h;   /* lid heights above / below the corner line, at the middle */
	float ix, iy, ir;   /* iris centre and radius */
	float lash, blur;   /* lash band thickness under the upper lid, edge blur (px) */
	float noise;        /* uniform per-pixel noise, +-grey levels */
	uint32_t seed;
	float refl_w, refl_h, refl_x, refl_y, refl_v; /* the screen's reflection on the cornea: half sizes and
						       * centre from the iris centre (iris radii), grey; 0: none */
} real_eye;

static real_eye real_eye_at(float k, float cx, float cy)
{
	const real_eye r = {cx, cy, 43 * k, 14 * k, 13 * k, cx + 2 * k, cy - 0.5f * k, 17 * k, 3.5f * k, 1.2f * k, 6, 1,
			    0, 0, 0, 0, 0};
	return r;
}

static float real_up(const real_eye *s, float x)
{
	const float t = (x - s->cx) / s->hw;
	return fabsf(t) >= 1 ? s->cy : s->cy - s->up_h * powf(1 - t * t, 0.7f) * (1 + 0.08f * t);
}

static float real_lo(const real_eye *s, float x)
{
	const float t = (x - s->cx) / s->hw;
	return fabsf(t) >= 1 ? s->cy : s->cy + s->lo_h * powf(1 - t * t, 0.8f) * (1 - 0.05f * t);
}

static float cov01(float v)
{
	return v < 0 ? 0 : (v > 1 ? 1 : v);
}

/* Draws the eye into img (w x h) and fills eye[6] with the hexagon outline. */
static void draw_real_eye(uint8_t *img, int w, int h, const real_eye *s, cf_pt eye[6])
{
	uint32_t rng = s->seed * 2654435761u + 12345u; /* own generator: leaves rand() alone */
	const float b = s->blur;
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++) {
			const float px = (float)x + 0.5f, py = (float)y + 0.5f;
			const float yu = real_up(s, px), yl = real_lo(s, px);
			const float open = cov01((py - yu) / b + 0.5f) * cov01((yl - py) / b + 0.5f);
			float v = 150; /* skin */
			if (open > 0) {
				const float shade = 25 * expf(-fmaxf(py - yu, 0) / (0.07f * s->hw)); /* lid shadow */
				float vo = 200 - shade;                                             /* sclera */
				const float d = hypotf(px - s->ix, py - s->iy);
				const float ci = cov01((s->ir - d) / b + 0.5f);
				if (ci > 0) {
					float iv = 80 - 15 * cov01((d - 0.75f * s->ir) / (0.25f * s->ir)); /* limbal ring */
					iv += cov01((0.4f * s->ir - d) / b + 0.5f) * (35 - iv);            /* pupil */
					const float dc = hypotf(px - (s->ix + 0.25f * s->ir), py - (s->iy - 0.3f * s->ir));
					iv += cov01((0.12f * s->ir - dc) / b + 0.5f) * (235 - iv);         /* catchlight */
					if (s->refl_v > 0) {
						const float rx = fabsf(px - (s->ix + s->refl_x * s->ir)) - s->refl_w * s->ir;
						const float ry = fabsf(py - (s->iy + s->refl_y * s->ir)) - s->refl_h * s->ir;
						iv += cov01(-rx / b + 0.5f) * cov01(-ry / b + 0.5f) * (s->refl_v - iv);
					}
					vo += ci * (iv - 0.5f * shade - vo);
				}
				vo += cov01((yu + s->lash - py) / b + 0.5f) * (45 - vo); /* lashes */
				v += open * (vo - v);
			}
			if (fabsf((px - s->cx) / s->hw) < 0.95f) /* lashes above the lid line too */
				v += cov01((py - (yu - 0.5f * s->lash)) / b + 0.5f) * cov01((yu - py) / b + 0.5f) * (45 - v);
			rng = rng * 1664525u + 1013904223u;
			v += s->noise * ((float)(rng >> 8) / 8388608.0f - 1.0f);
			img[y * w + x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : lrintf(v)));
		}
	/* Vision's hexagon: corners, and each lid at a third of the way out from the middle. */
	const float xs[2] = {s->cx - 0.36f * s->hw, s->cx + 0.36f * s->hw};
	eye[0] = (cf_pt){s->cx - s->hw, s->cy};
	eye[1] = (cf_pt){xs[0], real_up(s, xs[0])};
	eye[2] = (cf_pt){xs[1], real_up(s, xs[1])};
	eye[3] = (cf_pt){s->cx + s->hw, s->cy};
	eye[4] = (cf_pt){xs[1], real_lo(s, xs[1])};
	eye[5] = (cf_pt){xs[0], real_lo(s, xs[0])};
}

/* Direction of the warp's push, in degrees (0 = straight up, positive = towards the right). */
static float shift_angle_deg(const cf_eye *e)
{
	return atan2f(e->sx, -e->sy) * 57.29578f;
}

/* Direction of the vertical (face "up") part of the push alone, in the same degrees. The full push
 * also has a sideways part (reading damping along the eye axis) that moves with the measured
 * sideways gaze on purpose; the upper lid follows only the vertical part, so its shift shows it. */
static float up_angle_deg(const cf_eye *e)
{
	return atan2f(e->lid_sx, -e->lid_sy) * 57.29578f;
}

/* ---- Real-camera measurement noise (checks 28-31) */

/* Seeded Gaussian noise on a private generator, so these checks don't depend on rand(). */
static unsigned long long rng_state = 1;
static void rng_seed(unsigned long long s)
{
	rng_state = s * 2862933555777941757ULL + 3037000493ULL;
}
static double rng_u(void)
{
	rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
	return ((double)(rng_state >> 11) + 0.5) / 9007199254740992.0;
}
static float gauss(float sd)
{
	const double u1 = rng_u(), u2 = rng_u();
	return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)) * sd;
}

/* A real-sized face as the user's 1080p MacBook camera sees it: eyes RW px wide and RH px high,
 * 16 outline points, corners on y = 500. The person looks `glance` eye widths below the lens and `side`
 * eye widths to the side (eyes only), with the head turned `yaw` degrees; the upper lid follows the eye
 * down by R_LID of the glance. The refined pupils are at the true position plus d[e] = {dx, dy} px;
 * Vision's own pupil points (if `vision`) are coarse: they move R_VIS of the true movement, plus vd[e]. */
#define RW 86.0f
#define RH 26.0f
#define R_LID 0.5f
#define R_VIS 0.6f
typedef struct {
	float glance, side, yaw;
} real_gaze;
static cf_face face_real(real_gaze gz, const float d[2][2], int vision, const float vd[2][2])
{
	cf_face f;
	memset(&f, 0, sizeof(f));
	f.found = 1;
	f.image_width = 1920;
	f.image_height = 1080;
	f.face_width = 420;
	f.has_pose = 1;
	f.yaw_deg = gz.yaw;
	const float gpx = gz.glance * RW, spx = gz.side * RW, drop = R_LID * gpx;
	const float cx[2] = {960 - 1.5f * RW, 960 + 1.5f * RW};
	for (int e = 0; e < 2; e++) {
		cf_pt *pts = e == 0 ? f.left_eye : f.right_eye;
		for (int i = 0; i < 16; i++) {
			const float a = (float)i / 16.0f * 6.2831853f;
			const float s = sinf(a);
			pts[i].x = cx[e] + 0.5f * RW * cosf(a);
			pts[i].y = 500 + 0.5f * RH * s;
			if (s < -1e-3f)
				pts[i].y += drop * -s;
		}
		const cf_pt pupil = {cx[e] + spx + d[e][0], 500 + gpx + d[e][1]};
		const cf_pt vp = {cx[e] + R_VIS * spx + vd[e][0], 500 + R_VIS * gpx + vd[e][1]};
		if (e == 0) {
			f.left_count = 16;
			f.left_pupil = pupil;
			f.left_pupil_vision = vp;
		} else {
			f.right_count = 16;
			f.right_pupil = pupil;
			f.right_pupil_vision = vp;
		}
	}
	f.has_vision_pupils = vision;
	return f;
}

/* Calibrated like the user: notes 0.10 eye widths (8.6 px) under the lens. */
static cf_gaze_params real_params(void)
{
	const cf_gaze_params p = {.strength = 1.0f, .max_glance = 0.15f, .baseline = 0.0f, .baseline_open = RH / RW,
				  .has_notes = 1, .notes_offset = -0.10f, .notes_side = 0.0f,
				  .notes_open = (RH - R_LID * 0.10f * RW) / RW};
	return p;
}

typedef struct {
	double fps;
	int vision;      /* Vision's pupil points given */
	float flip_p;    /* chance per frame that a refined pupil reads FLIP_PX too high (the bimodal refiner) */
	int flip_both;   /* 1: both eyes flip together, 0: each eye on its own */
	float noise_px;  /* Gaussian noise on the refined pupils (both directions) */
	float vnoise_px; /* Gaussian noise on Vision's pupils (both directions) */
	float spike_p;   /* chance per frame and eye of a single-frame SPIKE_PX..+1 px outlier (either way) on
			  * top of the noise, as the fixed refiner still gives now and then */
} real_noise;
#define FLIP_PX 10.0f
#define SPIKE_PX 2.0f

/* One frame of the noisy face looking at `gz`. */
static cf_face noisy_real(const real_noise *rn, real_gaze gz)
{
	float d[2][2], vd[2][2];
	const int both = rng_u() < rn->flip_p;
	for (int e = 0; e < 2; e++) {
		const int flip = rn->flip_both ? both : rng_u() < rn->flip_p;
		d[e][0] = gauss(rn->noise_px);
		d[e][1] = (flip ? -FLIP_PX : 0.0f) + gauss(rn->noise_px);
		if (rn->spike_p > 0 && rng_u() < rn->spike_p)
			d[e][1] += (float)(SPIKE_PX + rng_u()) * (rng_u() < 0.5 ? -1.0f : 1.0f);
		vd[e][0] = gauss(rn->vnoise_px);
		vd[e][1] = gauss(rn->vnoise_px);
	}
	return face_real(gz, d, rn->vision, vd);
}

static real_gaze at_notes(void)
{
	const real_gaze gz = {0.10f, 0.0f, 0.0f};
	return gz;
}

/* Reading steadily at the notes: 1 s with plain noise (as when calibrating), then the noise of `rn`; from
 * 3 s on, 10 s of the RMS frame-to-frame change of the applied vertical and sideways shift (px), the
 * largest single vertical change, the largest wander of the vertical shift within any 1 s (highest minus
 * lowest), and the mean correction (px). */
static void real_steady(const real_noise *rn, unsigned seed, float *rms, float *rms_x, float *worst, float *wobble,
			float *mean)
{
	const cf_gaze_params p = real_params();
	cf_gaze g;
	cf_gaze_init(&g);
	cf_eye out[2];
	rng_seed(seed);
	const double dt = 1.0 / rn->fps;
	const int start = (int)rn->fps, warm = (int)(3.0 * rn->fps), frames = warm + (int)(10.0 * rn->fps);
	const int win = (int)rn->fps;
	real_noise quiet = *rn;
	quiet.flip_p = 0;
	double s2 = 0, s2x = 0, sum = 0;
	float prev = 0, prev_x = 0, big = 0, ys[600];
	int c = 0;
	for (int i = 0; i < frames; i++) {
		cf_face f = noisy_real(i < start ? &quiet : rn, at_notes());
		const int n = cf_gaze_update(&g, &f, &p, 100.0 + i * dt, out);
		const float sy = n ? out[0].sy : 0.0f, sx = n ? out[0].sx : 0.0f;
		if (i > warm) {
			s2 += (double)(sy - prev) * (sy - prev);
			s2x += (double)(sx - prev_x) * (sx - prev_x);
			big = fmaxf(big, fabsf(sy - prev));
			sum += -sy;
			if (c < 600)
				ys[c] = sy;
			c++;
		}
		prev = sy;
		prev_x = sx;
	}
	*rms = (float)sqrt(s2 / c);
	*rms_x = (float)sqrt(s2x / c);
	*worst = big;
	*wobble = 0;
	for (int i = 0; i + win <= c && i + win <= 600; i++) {
		float lo = ys[i], hi = ys[i];
		for (int k = i; k < i + win; k++) {
			lo = fminf(lo, ys[k]);
			hi = fmaxf(hi, ys[k]);
		}
		*wobble = fmaxf(*wobble, hi - lo);
	}
	*mean = (float)(sum / c);
}

/* After the face first appears with the refiner flipping, the frame from which the correction stays
 * within 1 px of the notes correction (for 1 s). */
static int real_lock_on(const real_noise *rn, unsigned seed)
{
	const cf_gaze_params p = real_params();
	cf_gaze g;
	cf_gaze_init(&g);
	cf_eye out[2];
	rng_seed(seed);
	const double dt = 1.0 / rn->fps;
	const int need = (int)rn->fps;
	int run = 0;
	for (int i = 0; i < (int)(30.0 * rn->fps); i++) {
		cf_face f = noisy_real(rn, at_notes());
		const int n = cf_gaze_update(&g, &f, &p, 100.0 + i * dt, out);
		run = fabsf((n ? -out[0].sy : 0.0f) - 0.10f * RW) < 1.0f ? run + 1 : 0;
		if (run >= need)
			return i - need + 1;
	}
	return 9999;
}

/* Lens -> notes -> lens glances (1.5 s at each, CYC times). The mean correction over all glances on each
 * of the first K frames after the saccade (frame 0 = the first frame showing the new gaze): how a
 * glance looks with the measurement noise averaged out. mean_dn/up: the first frame on which that mean
 * reaches 90% of the notes correction going down, or falls to 10% of it coming back; over: how far the
 * mean goes past the target (a filter overshooting). For single glances, which carry their own frames'
 * measurement noise: quick, the share that reached 90% (or 10%) on the first or second frame, and late,
 * the latest frame on which one did. */
static void real_glances(const real_noise *rn, unsigned seed, int *mean_dn, int *mean_up, float *over, float *quick,
			 int *late)
{
	enum { CYC = 32, K = 10 };
	const cf_gaze_params p = real_params();
	const float target = 0.10f * RW;
	cf_gaze g;
	cf_gaze_init(&g);
	cf_eye out[2];
	rng_seed(seed);
	const double dt = 1.0 / rn->fps;
	const int hold = (int)(1.5 * rn->fps);
	double t = 200.0;
	double mean[2][K] = {{0}};
	*late = 0;
	int n_quick = 0;
	const real_gaze lens = {0, 0, 0};
	for (int i = 0; i < (int)(3.0 * rn->fps); i++, t += dt) {
		cf_face f = noisy_real(rn, lens);
		cf_gaze_update(&g, &f, &p, t, out);
	}
	for (int cyc = 0; cyc < CYC; cyc++)
		for (int dir = 0; dir < 2; dir++) {
			int land = 99;
			for (int i = 0; i < hold; i++, t += dt) {
				cf_face f = noisy_real(rn, dir == 0 ? at_notes() : lens);
				const int n = cf_gaze_update(&g, &f, &p, t, out);
				const float c = n ? -out[0].sy : 0.0f;
				if (land == 99 && (dir == 0 ? c >= 0.9f * target : c <= 0.1f * target))
					land = i;
				if (i < K)
					mean[dir][i] += c / CYC;
			}
			*late = *late > land ? *late : land;
			n_quick += land <= 1;
		}
	*quick = (float)n_quick / (2.0f * CYC);
	*mean_dn = *mean_up = 99;
	*over = 0;
	for (int k = K - 1; k >= 0; k--) {
		if (mean[0][k] >= 0.9f * target)
			*mean_dn = k;
		if (mean[1][k] <= 0.1f * target)
			*mean_up = k;
		*over = fmaxf(*over, fmaxf((float)mean[0][k] - target, (float)-mean[1][k]));
	}
	/* (the first frame from which on the mean stays reached is what counts) */
	for (int k = 0; k < K; k++) {
		if (mean[0][k] < 0.9f * target)
			*mean_dn = k + 1;
		if (mean[1][k] > 0.1f * target)
			*mean_up = k + 1;
	}
}

/* Steady at the notes with noisy landmarks; at frame 60 one frame carries an outlier of `px` (positive =
 * down) in one eye (eye 0, 1) or both (2). Compared with a twin run with exactly the same noise but no
 * outlier: `effect` is the largest difference of the applied shift on that frame and the next two; and
 * `step` is the largest frame-to-frame change of the applied shift in the outlier run from frame 60 on,
 * less the largest in the twin run (> 0 only if the outlier made a bigger step than the noise does). */
static void real_outlier(const real_noise *rn, int which, float px, unsigned seed, float *effect, float *step)
{
	const cf_gaze_params p = real_params();
	cf_gaze ga, gb;
	cf_gaze_init(&ga);
	cf_gaze_init(&gb);
	cf_eye oa[2], ob[2];
	rng_seed(seed);
	const double dt = 1.0 / rn->fps;
	float pa = 0, pb = 0, sa = 0, sb = 0;
	*effect = 0;
	for (int i = 0; i < 90; i++) {
		cf_face f = noisy_real(rn, at_notes());
		cf_face fo = f;
		if (i == 60) {
			if (which != 1)
				fo.left_pupil.y += px;
			if (which != 0)
				fo.right_pupil.y += px;
		}
		const int na = cf_gaze_update(&ga, &fo, &p, 300.0 + i * dt, oa);
		const int nb = cf_gaze_update(&gb, &f, &p, 300.0 + i * dt, ob);
		const float ya = na ? oa[0].sy : 0, yb = nb ? ob[0].sy : 0;
		if (i >= 60 && i <= 62)
			*effect = fmaxf(*effect, fabsf(ya - yb));
		if (i >= 60) {
			sa = fmaxf(sa, fabsf(ya - pa));
			sb = fmaxf(sb, fabsf(yb - pb));
		}
		pa = ya;
		pb = yb;
	}
	*step = sa - sb;
}

/* Looks away from the notes and back, with the noise of `rn`: 3 s at the notes, 1.5 s at `away`, 1.5 s
 * back at the notes. Over the hand-off both ways (the frame the gaze moves and everything after): the
 * largest frame-to-frame change of the applied shift (vertical or sideways, px). Over the fades (the
 * first 0.8 s of each): the largest step against the way the fade goes (the shift growing while
 * letting go, shrinking while taking over). And the shift left at the end of the look-away, and the
 * correction at the end of the look back (px). */
static void real_look_away(const real_noise *rn, real_gaze away, unsigned seed, float *step, float *against_out,
			   float *against_in, float *left, float *back)
{
	const cf_gaze_params p = real_params();
	cf_gaze g;
	cf_gaze_init(&g);
	cf_eye out[2];
	rng_seed(seed);
	const double dt = 1.0 / rn->fps;
	const int n0 = (int)(3.0 * rn->fps), n1 = (int)(1.5 * rn->fps), nf = (int)(0.8 * rn->fps);
	float py = 0, px = 0, pm = 0;
	*step = *against_out = *against_in = 0;
	for (int i = 0; i < n0 + 2 * n1; i++) {
		const int phase = i < n0 ? 0 : (i < n0 + n1 ? 1 : 2);
		cf_face f = noisy_real(rn, phase == 1 ? away : at_notes());
		const int n = cf_gaze_update(&g, &f, &p, 100.0 + i * dt, out);
		const float sy = n ? out[0].sy : 0, sx = n ? out[0].sx : 0, mag = sqrtf(sy * sy + sx * sx);
		if (phase > 0)
			*step = fmaxf(*step, fmaxf(fabsf(sy - py), fabsf(sx - px)));
		if (phase == 1 && i < n0 + nf)
			*against_out = fmaxf(*against_out, mag - pm);
		if (phase == 2 && i < n0 + n1 + nf)
			*against_in = fmaxf(*against_in, pm - mag);
		if (i == n0 + n1 - 1)
			*left = mag;
		*back = -sy;
		py = sy;
		px = sx;
		pm = mag;
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

	/* 7b. Eyes closed for long (0.8 s): that's not a blink any more, correction fades out. (Was 0.7 s:
	 * the fade-out is now an S-curve that starts gently, so it is gone ~0.1 s later.) */
	for (int i = 0; i < 24; i++, t += dt)
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

	/* 12b. Looking sideways out of the notes band lets go of the correction (changed on purpose: it used
	 * to let go within the saccade itself, so the eye landed where it really is at once; now letting go
	 * is a fade like every other hand-off, so it never shows as a snap). The applied shift (sideways
	 * damping and vertical correction together) changes by under 0.6 px a frame, only ever shrinks (it
	 * fades from its size before the look, not from the far-away gaze), and is gone within 0.6 s. */
	{
		cf_gaze_params pr = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
				     .notes_offset = -0.08f};
		cf_gaze_init(&g);
		const float before = settle(&g, face_px(960, 500, 4, 3.2f, 16, 0, 1.0f), &pr, &t, 60, out);
		float px = out[0].sx, py = out[0].sy, pm = before, step = 0, grow = 0, last = before;
		for (int i = 0; i < 18; i++, t += dt) {
			n = update(&g, face_px(960, 500, 20, 3.2f, 16, 0, 1.0f), &pr, t, out);
			const float sx = n ? out[0].sx : 0, sy = n ? out[0].sy : 0, m = sqrtf(sx * sx + sy * sy);
			step = fmaxf(step, sqrtf((sx - px) * (sx - px) + (sy - py) * (sy - py)));
			grow = fmaxf(grow, m - pm);
			px = sx;
			py = sy;
			pm = last = m;
		}
		CHECK(step < 0.6f && grow < 0.02f && last < 0.1f,
		      "sideways look out of the notes band: step %.2f px, grew %.2f px, %.2f px left after 0.6 s", step, grow,
		      last);
		printf("ok: sideways look out of the notes band fades out (%.2f px, largest step %.2f px, %.2f left)\n", before,
		       step, last);
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

	/* 16. Keyboard and back (changed on purpose: the far fade used to follow the gaze directly, so the
	 * correction switched off within the saccade to the keyboard and back on within the saccade back;
	 * now both hand-offs are a fade, so neither shows as a snap). Going to the keyboard, the correction
	 * lets go gradually: the applied shift changes by under 0.5 px a frame, the visible eye only moves
	 * down, and nothing is left after 1 s. Back on the notes it takes over the same way, the visible eye
	 * only moving up, and it is at the lens (within 0.5 px) after 1 s. */
	{
		const float gl = 0.08f, gpx = gl * 40.0f, kpx = 0.3f * 40.0f;
		const cf_gaze_params pk = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
					   .notes_offset = -gl, .notes_open = 0.35f};
		cf_gaze_init(&g);
		settle(&g, face_px(960, 500, 0, gpx, 16, 0, 2.0f), &pk, &t, 45, out);
		float ps = out[0].sy, pv = gpx + ps, step = 0, against = 0, left = 0, back = 0;
		for (int i = 0; i < 60; i++, t += dt) {
			const int away = i < 30;
			const float pd = away ? kpx : gpx;
			n = update(&g, face_px(960, 500, 0, pd, 16, 0, away ? 6.0f : 2.0f), &pk, t, out);
			const float sy = n ? out[0].sy : 0, vis = pd + sy; /* visible eye below the lens */
			step = fmaxf(step, fabsf(sy - ps));
			against = fmaxf(against, away ? pv - vis : vis - pv);
			if (i == 29)
				left = fabsf(sy);
			back = fabsf(vis);
			ps = sy;
			pv = vis;
		}
		CHECK(step < 0.5f && against < 0.05f && left < 0.1f && back < 0.5f,
		      "keyboard and back: shift step %.2f px, visible eye moved against the gaze %.2f px, %.2f px left at "
		      "the keyboard, back on the notes off the lens by %.2f px",
		      step, against, left, back);
		printf("ok: keyboard and back fade (largest step %.2f px, %.2f px against the gaze, %.2f px left at the "
		       "keyboard, %.2f px off the lens back on the notes)\n",
		       step, against, left, back);

		/* Sweeping slowly from the notes past the limit: visible position only moves down. */
		cf_gaze_init(&g);
		settle(&g, face_px(960, 500, 0, gpx, 16, 0, 2.0f), &pk, &t, 45, out);
		float prev_vis = -1e9f, sweep_against = 0;
		for (int i = 0; i <= 60; i++, t += dt) {
			const float pd = gpx + (kpx - gpx) * (float)i / 60.0f;
			n = update(&g, face_px(960, 500, 0, pd, 16, 0, 2.0f), &pk, t, out);
			const float vis = pd + (n ? out[0].sy : 0);
			if (i > 0)
				sweep_against = fmaxf(sweep_against, prev_vis - vis);
			prev_vis = vis;
		}
		CHECK(sweep_against < 0.05f, "visible eye moved up %.2f px while the gaze moved down", sweep_against);
		printf("ok: fading out past the notes is monotonic (%.3f px against the gaze)\n", sweep_against);
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
		 * pupil moves far more than a lid could drag it, so the correction starts letting go at once
		 * (changed on purpose: it used to be gone on the first frame; letting go is a fade now). It
		 * shrinks every frame from the first, is below 80% after 0.1 s and gone after 0.5 s. */
		cf_gaze_init(&g);
		const float kb0 = settle(&g, face_px(960, 500, 0, gpx, 14, 0, 0), &ph, &t, 60, out);
		float kb_prev = kb0, kb3 = 0, kb = 0;
		int kb_held = 0;
		for (int i = 0; i < 15; i++, t += dt) {
			n = update(&g, face_px(960, 500, 0, 12.0f, 7.0f, 0, 0), &ph, t, out);
			kb = n ? shift_px(&out[0]) : 0;
			/* the first frames must already shrink it (a hold would keep it); later it may not grow */
			kb_held += i < 3 ? kb > kb_prev - 0.01f : kb > kb_prev + 1e-4f;
			kb_prev = kb;
			if (i == 2)
				kb3 = kb;
		}
		CHECK(!kb_held && kb3 < 0.8f * kb0 && kb < 0.2f,
		      "keyboard glance with deep lids: held on %d frames, %.2f of %.2f px after 0.1 s, %.2f px after 0.5 s",
		      kb_held, kb3, kb0, kb);
		printf("ok: keyboard glance with the lids at r 0.44 is not held (%.2f -> %.2f px after 0.1 s, %.2f after "
		       "0.5 s)\n",
		       kb0, kb3, kb);
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

	/* 13b. Iris refinement on a realistic eye (draw_real_eye: Vision's 6-point outline, the iris cut by
	 * both lids, lashes inside the outline). On real footage the old refinement flipped between two
	 * heights ~10 px apart from one frame to the next while the iris moved smoothly (the vertical
	 * centre of a lid-clipped iris barely shows, and an argmax over a flat score jumped between the
	 * ends of its search range): the corrected eyes shook. The refined point must be steady, follow
	 * sub-pixel iris moves smoothly, and stay fast. 1080p (86 px eye) and 4K (172 px eye). */
	{
		enum { MW = 400, MH = 220 };
		static uint8_t img[MW * MH];
		for (int v = 0; v < 2; v++) {
			const float k = v ? 2.0f : 1.0f;
			const int W = (int)(200 * k), H = (int)(110 * k);
			const float cx = 0.5f * (float)W + 0.3f, cy = 0.5f * (float)H + 0.4f;
			const char *name = v ? "4K" : "1080p";
			cf_pt eye[6];

			/* 1. The same eye under different camera noise: one steady answer, not two clusters. */
			const int seeds = 24;
			double s1 = 0, s2 = 0;
			float lo = 1e9f, hi = -1e9f;
			for (int sd = 0; sd < seeds; sd++) {
				real_eye s = real_eye_at(k, cx, cy);
				s.seed = 100 + (uint32_t)sd;
				draw_real_eye(img, W, H, &s, eye);
				const cf_pt r = cf_refine_pupil(img, W, W, H, eye, 6, (cf_pt){s.ix - 3 * k, s.iy - 2 * k});
				s1 += r.y;
				s2 += (double)r.y * r.y;
				lo = fminf(lo, r.y);
				hi = fmaxf(hi, r.y);
			}
			const double mean = s1 / seeds, sdev = sqrt(fmax(s2 / seeds - mean * mean, 0));
			const float truth_y = real_eye_at(k, cx, cy).iy;
			CHECK(sdev < 0.2 && hi - lo < 0.6, "%s: refined y over %d noise seeds: sd %.3f px, range %.3f px",
			      name, seeds, sdev, hi - lo);
			CHECK(fabs(mean - truth_y) < 2 * k, "%s: refined y %.2f, iris centre at %.2f", name, mean, truth_y);

			/* 2. The iris moving down 10 px (20 at 4K) in 0.25 px steps, the lids still and cutting
			 * more and more of it: the refined point follows smoothly, never back up, no jumps. */
			const int steps = (int)(40 * k);
			float first = 0, prev = 0, max_step = 0, min_step = 1e9f;
			for (int q = 0; q <= steps; q++) {
				real_eye s = real_eye_at(k, cx, cy);
				s.iy += 0.25f * (float)q;
				s.seed = 7;
				draw_real_eye(img, W, H, &s, eye);
				/* Vision's point: coarse, follows only part of the move */
				const cf_pt g = {s.ix - 1 * k, truth_y + 0.125f * (float)q - 1 * k};
				const float y = cf_refine_pupil(img, W, W, H, eye, 6, g).y;
				if (q) {
					max_step = fmaxf(max_step, y - prev);
					min_step = fminf(min_step, y - prev);
				} else {
					first = y;
				}
				prev = y;
			}
			const float gain = (prev - first) / (0.25f * (float)steps);
			CHECK(min_step > 0 && max_step < 1.0f, "%s: iris moving down in 0.25 px steps: refined steps %.3f..%.3f px",
			      name, min_step, max_step);
			CHECK(gain > 0.4f && gain < 1.6f, "%s: iris moved %.1f px down, refined point %.2f px (%.0f%%)", name,
			      0.25f * (float)steps, prev - first, 100 * gain);

			/* 3. A glance down with the upper lid following it (iris 8 px down, upper lid 4 px; x k):
			 * reads as down, clearly. */
			float lid_move;
			{
				real_eye s = real_eye_at(k, cx, cy);
				s.seed = 9;
				draw_real_eye(img, W, H, &s, eye);
				const float y0 = cf_refine_pupil(img, W, W, H, eye, 6, (cf_pt){s.ix, s.iy}).y;
				s.iy += 8 * k;
				s.up_h -= 4 * k;
				draw_real_eye(img, W, H, &s, eye);
				lid_move = cf_refine_pupil(img, W, W, H, eye, 6, (cf_pt){s.ix, s.iy - 3 * k}).y - y0;
				CHECK(lid_move > 0.4f * 8 * k, "%s: glance with the lid following: refined point moved %.2f px of %.0f",
				      name, lid_move, 8 * k);
			}

			/* 4. A frame sequence: the iris jittering +-0.3 px, fresh camera noise and +-0.5 px outline
			 * noise every frame. Frame-to-frame steps stay about as small as the iris' own. */
			double step2 = 0, true2 = 0;
			{
				uint32_t rng = 5;
				float py = 0, ty = 0;
				const int frames = 120;
				for (int fr = 0; fr < frames; fr++) {
					real_eye s = real_eye_at(k, cx, cy);
					float u[16];
					for (int q = 0; q < 16; q++) {
						rng = rng * 1664525u + 1013904223u;
						u[q] = (float)(rng >> 8) / 8388608.0f - 1.0f; /* -1..1 */
					}
					s.ix += 0.3f * u[0];
					s.iy += 0.3f * u[1];
					s.seed = 1000 + (uint32_t)fr;
					draw_real_eye(img, W, H, &s, eye);
					for (int q = 0; q < 6; q++) {
						eye[q].x += 0.5f * u[2 + 2 * q];
						eye[q].y += 0.5f * u[3 + 2 * q];
					}
					const float y = cf_refine_pupil(img, W, W, H, eye, 6, (cf_pt){s.ix + 1, s.iy + 1}).y;
					if (fr) {
						step2 += (double)(y - py) * (y - py);
						true2 += (double)(s.iy - ty) * (s.iy - ty);
					}
					py = y;
					ty = s.iy;
				}
				step2 = sqrt(step2 / (frames - 1));
				true2 = sqrt(true2 / (frames - 1));
				CHECK(step2 < 0.6, "%s: jittering iris (own RMS step %.2f px): refined RMS step %.2f px", name, true2,
				      step2);
			}

			/* 5. Reading along the notes: the iris sweeping sideways 36 px (72 at 4K) at a steady
			 * height. The height stays put. */
			float side_range;
			{
				float ylo = 1e9f, yhi = -1e9f;
				for (int q = 0; q <= 144; q++) {
					real_eye s = real_eye_at(k, cx, cy);
					s.ix = cx + (-18 + 0.25f * (float)q) * k;
					s.seed = 7;
					draw_real_eye(img, W, H, &s, eye);
					const float y = cf_refine_pupil(img, W, W, H, eye, 6, (cf_pt){s.ix - 2 * k, s.iy - 1}).y;
					ylo = fminf(ylo, y);
					yhi = fmaxf(yhi, y);
				}
				side_range = yhi - ylo;
				CHECK(side_range < 0.5f * k, "%s: sideways sweep moved the refined height by %.2f px", name, side_range);
			}

			/* 6. The screen's reflection on the cornea (a bright patch in the iris, nearly as bright as
			 * the sclera): the down sweep of check 2 stays smooth. The patch moving with the iris below
			 * the pupil, staying put in the image while the iris moves under it, and off to the side
			 * reaching towards the iris edge. */
			float refl_lo = 1e9f, refl_hi = -1e9f;
			for (int rv = 0; rv < 3; rv++) {
				float rprev = 0;
				for (int q = 0; q <= steps; q++) {
					real_eye s = real_eye_at(k, cx, cy);
					const float rw[3] = {0.25f, 0.2f, 0.2f}, rh[3] = {0.18f, 0.13f, 0.15f};
					const float rx[3] = {0, 0, -0.5f}, rgrey[3] = {190, 170, 180};
					s.refl_w = rw[rv];
					s.refl_h = rh[rv];
					s.refl_x = rx[rv];
					s.refl_y = rv == 1 ? 0.2f - 0.25f * (float)q / s.ir : (rv ? 0 : 0.25f);
					s.refl_v = rgrey[rv];
					s.iy += 0.25f * (float)q;
					s.seed = 7;
					draw_real_eye(img, W, H, &s, eye);
					const cf_pt g = {s.ix - 1 * k, truth_y + 0.125f * (float)q - 1 * k};
					const float y = cf_refine_pupil(img, W, W, H, eye, 6, g).y;
					if (q) {
						refl_lo = fminf(refl_lo, y - rprev);
						refl_hi = fmaxf(refl_hi, y - rprev);
					}
					rprev = y;
				}
			}
			CHECK(refl_lo > 0 && refl_hi < 1.0f,
			      "%s: iris moving down in 0.25 px steps with a corneal reflection: refined steps %.3f..%.3f px", name,
			      refl_lo, refl_hi);

			/* Fast enough for every camera frame. */
			real_eye s = real_eye_at(k, cx, cy);
			draw_real_eye(img, W, H, &s, eye);
			float sink = 0;
			const clock_t c0 = clock();
			for (int i = 0; i < 1000; i++)
				sink += cf_refine_pupil(img, W, W, H, eye, 6, (cf_pt){s.ix + (float)(i & 1), s.iy}).y;
			const double ms = (double)(clock() - c0) / CLOCKS_PER_SEC * 1000.0 / 1000.0;
			CHECK(ms < 0.3 && sink > 0, "%s: refine_pupil took %.3f ms per eye", name, ms);
			printf("ok: realistic %s eye (6-point outline): noise sd %.3f px (range %.2f), %.0f px move followed "
			       "%.0f%% in steps %.2f..%.2f px, lid-follow %.1f of %.0f px, jitter RMS step %.2f px (iris %.2f), "
			       "sideways %.2f px, with a corneal reflection steps %.2f..%.2f px, %.3f ms\n",
			       name, sdev, hi - lo, 0.25f * (float)steps, 100 * gain, min_step, max_step, lid_move, 8 * k, step2,
			       true2, side_range, refl_lo, refl_hi, ms);
		}
	}

	/* 22. Noisy landmarks on both eyes while sitting still: the push keeps a steady direction
	 * (straight up), instead of wobbling with every frame's landmark noise. (PR #1 measured the whole
	 * push; here the sideways reading damping is part of it and follows the measured sideways gaze on
	 * purpose, so the face "up" direction it is pushed along is measured, from the lid shift.) */
	{
		cf_gaze_init(&g);
		srand(11);
		double asum = 0, asum2 = 0;
		float aworst = 0;
		int ac = 0;
		for (int i = 0; i < 150; i++, t += dt) {
			cf_face f = face_at(960, 500, 0.25f, 16, 0);
			for (int k = 0; k < 8; k++) {
				f.left_eye[k].x += noise(2);
				f.left_eye[k].y += noise(2);
				f.right_eye[k].x += noise(2);
				f.right_eye[k].y += noise(2);
			}
			n = cf_gaze_update(&g, &f, &p, t, out);
			if (i >= 60 && n) {
				const float a = up_angle_deg(&out[0]);
				asum += a;
				asum2 += a * a;
				aworst = fmaxf(aworst, fabsf(a));
				ac++;
			}
		}
		const float asd = (float)sqrt(fmax(asum2 / ac - (asum / ac) * (asum / ac), 0));
		CHECK(ac > 80 && asd < 0.25f && aworst < 0.8f, "push direction wobbles: sd %.2f deg, worst %.2f deg", asd,
		      aworst);
		printf("ok: steady push direction when still (sd %.2f deg, worst %.2f deg)\n", asd, aworst);

		/* 23. A real head tilt (15 degrees) is still followed within a few frames. */
		for (int i = 0; i < 6; i++, t += dt)
			n = update(&g, face_rolled(960, 500, 0.25f, 16, 15), &p, t, out);
		float roll_err = n ? fabsf(shift_angle_deg(&out[0]) - 15.0f) : 99;
		CHECK(roll_err < 3.0f, "push direction lags a head tilt by %.1f deg after 0.2 s", roll_err);
		for (int i = 0; i < 24; i++, t += dt)
			n = update(&g, face_rolled(960, 500, 0.25f, 16, 15), &p, t, out);
		roll_err = n ? fabsf(shift_angle_deg(&out[0]) - 15.0f) : 99;
		CHECK(roll_err < 0.5f, "push direction should settle on the tilt (%.2f deg off)", roll_err);
		CHECK(n == 2 && shift_px(&out[0]) > 3.0f, "a tilted head should still be corrected");
		printf("ok: head tilt followed (%.2f deg off after 1 s)\n", roll_err);

		/* 23b. Rolling the head continuously (30 deg/s, 15 -> -15 -> 15 degrees) while reading: the
		 * push direction trails only slightly, and the roll never reads as a glance (the correction's
		 * size stays put, since the measurement uses each frame's own eye axis). */
		const float roll_steady = shift_px(&out[0]);
		float worst_dir = 0, worst_size = 0;
		for (int i = 0; i < 60; i++, t += dt) {
			const float deg = i < 30 ? 15.0f - (float)i : -15.0f + (float)(i - 30);
			n = update(&g, face_rolled(960, 500, 0.25f, 16, deg), &p, t, out);
			if (n) {
				worst_dir = fmaxf(worst_dir, fabsf(shift_angle_deg(&out[0]) - deg));
				worst_size = fmaxf(worst_size, fabsf(shift_px(&out[0]) - roll_steady));
			} else {
				worst_size = 99;
			}
		}
		CHECK(worst_dir < 3.0f && worst_size < 0.2f, "head roll: direction off by %.2f deg, size off by %.2f px",
		      worst_dir, worst_size);
		printf("ok: rolling head (30 deg/s): push direction within %.2f deg, size within %.2f px\n", worst_dir,
		       worst_size);
	}

	/* 24. A one-frame detection dropout: the warp holds, no flicker. */
	{
		cf_gaze_init(&g);
		for (int i = 0; i < 60; i++, t += dt)
			n = update(&g, face_at(960, 500, 0.25f, 16, 0), &p, t, out);
		const float before = shift_px(&out[0]);
		n = cf_gaze_update(&g, &(cf_face){0}, &p, t, out);
		t += dt;
		const float during = n ? shift_px(&out[0]) : 0;
		n = update(&g, face_at(960, 500, 0.25f, 16, 0), &p, t, out);
		t += dt;
		const float after = n ? shift_px(&out[0]) : 0;
		CHECK(fabsf(during - before) < 0.1f && fabsf(after - before) < 0.2f,
		      "1-frame dropout changed the warp: %.2f -> %.2f -> %.2f px", before, during, after);
		printf("ok: 1-frame dropout holds the warp (%.2f -> %.2f -> %.2f px)\n", before, during, after);

		/* 25. The face really gone (say the person moved quickly). PR #1 asked for the warp to be over
		 * within 0.2 s; here a dropout under LOST_HOLD (0.2 s, it covers a missed analysis or two on
		 * the background tracking thread) is held unchanged on purpose, and then the warp lets go
		 * within ~0.1 s instead of lingering over a spot where the eyes no longer are. */
		for (int i = 0; i < 30; i++, t += dt)
			n = update(&g, face_at(960, 500, 0.25f, 16, 0), &p, t, out);
		const float held = shift_px(&out[0]);
		int frames_to_stop = -1;
		float held_lo = 1e9f, held_hi = 0;
		for (int i = 1; i <= 30 && frames_to_stop < 0; i++, t += dt) {
			n = cf_gaze_update(&g, &(cf_face){0}, &p, t, out);
			if (n == 0)
				frames_to_stop = i;
			else if (i * dt < 0.2 - 1e-6) {
				held_lo = fminf(held_lo, shift_px(&out[0]));
				held_hi = fmaxf(held_hi, shift_px(&out[0]));
			}
		}
		CHECK(held_lo > held - 0.05f && held_hi < held + 0.05f, "dropout under 0.2 s not held (%.2f..%.2f of %.2f)",
		      held_lo, held_hi, held);
		CHECK(frames_to_stop > 0 && frames_to_stop * dt <= 0.3 + 1e-6, "warp lingered %d frames after losing the face",
		      frames_to_stop);
		printf("ok: lost face held for 0.2 s, then the warp stops (after %.0f ms)\n", frames_to_stop * dt * 1000);
	}

	/* 26. Glance from the lens to the notes, then a blink 0, 1 or 2 frames later (full blink or half
	 * blink, pupil point sliding 0 or 1 px with the closing lids; lids steady during the glance, or
	 * following it down by 2 px). The glance itself was seen with steady lids: the blink must keep
	 * it, not undo it. No frame below 90% of the notes correction after the glance, no jump back. */
	{
		const float gl = 0.04f, gpx = gl * 40.0f;
		const cf_gaze_params pg = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
					   .notes_offset = -gl, .notes_open = 0.35f};
		float worst_lo = 1e9f, worst_hi = 0, worst_jump = 0;
		for (int follow = 0; follow < 2; follow++)
			for (int half = 0; half < 2; half++)
				for (int slide = 0; slide < 2; slide++)
					for (int gap = 0; gap <= 2; gap++) {
						const float base = follow ? 14.0f : 16.0f; /* lids at the notes */
						const float full[] = {9, 2, 2, 9, base}, part[] = {10, 6.7f, 6.7f, 6.7f, 10};
						const float *lids = half ? part : full;
						cf_gaze_init(&g);
						settle(&g, face_px(960, 500, 0, 0, 16, 0, 0), &pg, &t, 45, out);
						float lo = 1e9f, hi = 0, after_lo = 1e9f, after_hi = 0;
						for (int i = 0; i < 1 + gap + 5 + 25; i++, t += dt) {
							const int kb = i - 1 - gap; /* blink frames 0..4 */
							const float h = (kb >= 0 && kb < 5) ? lids[kb] : base;
							const float pd = gpx + (h < base ? (float)slide * (base - h) / (base - 2) : 0);
							n = update(&g, face_px(960, 500, 0, pd, h, 0, 0), &pg, t, out);
							const float sv = n ? shift_px(&out[0]) : 0;
							lo = fminf(lo, sv);
							hi = fmaxf(hi, sv);
							if (kb >= 5) {
								after_lo = fminf(after_lo, sv);
								after_hi = fmaxf(after_hi, sv);
							}
						}
						CHECK(lo > 0.9f * gpx && hi < 1.1f * gpx && after_hi - after_lo < 0.1f,
						      "glance then %s blink %d frames later (lids %s, slide %d): %.2f..%.2f px (notes "
						      "%.2f), after the blink %.2f..%.2f",
						      half ? "half" : "full", gap, follow ? "following" : "steady", slide, lo, hi, gpx,
						      after_lo, after_hi);
						worst_lo = fminf(worst_lo, lo / gpx);
						worst_hi = fmaxf(worst_hi, hi / gpx);
						worst_jump = fmaxf(worst_jump, after_hi - after_lo);
					}
		printf("ok: glance then blink 0-2 frames later (full/half, slide 0/1, lids steady/following): %.0f%%..%.0f%% "
		       "of the notes correction, %.2f px change after the blink\n",
		       100 * worst_lo, 100 * worst_hi, worst_jump);
	}

	/* 27. Keyboard glance over a few frames with the lids dropping to r 0.5: the correction lets go and
	 * stays off; it must not come back for a frame while the eye only moves down. */
	{
		const float gl = 0.04f, gpx = gl * 40.0f;
		const cf_gaze_params pk = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
					   .notes_offset = -gl, .notes_open = 0.35f};
		const float pupil[][3] = {{4.0f, 6.5f, 8.0f}, {5.87f, 6.54f, 8.0f}, {3.0f, 5.0f, 7.0f}};
		const float lid[][3] = {{12, 10, 8}, {11, 9.5f, 8}, {13, 11, 9}};
		float back = 0;
		for (int c = 0; c < 3; c++) {
			cf_gaze_init(&g);
			settle(&g, face_px(960, 500, 0, gpx, 14, 0, 0), &pk, &t, 60, out);
			float prev_vis = gpx - shift_px(&out[0]);
			for (int i = 0; i < 20; i++, t += dt) {
				const float pd = pupil[c][i < 3 ? i : 2], h = lid[c][i < 3 ? i : 2];
				n = update(&g, face_px(960, 500, 0, pd, h, 0, 0), &pk, t, out);
				const float vis = pd - (n ? shift_px(&out[0]) : 0); /* visible eye below the lens */
				back = fmaxf(back, prev_vis - vis);
				prev_vis = vis;
			}
		}
		CHECK(back < 0.3f, "keyboard glance: the visible eye moved back up %.2f px mid-glance", back);
		printf("ok: keyboard glance with dropping lids only moves down (%.2f px back up at most)\n", back);
	}

	/* 28. The user's camera: 86 px eyes, notes 0.10 eye widths (8.6 px) under the lens. Two noise models
	 * of the refined pupil, from real video: as it is now, 1.2 px Gaussian noise (both directions) plus, on
	 * 5% of frames in each eye, a 2-3 px single-frame outlier; and as the old refiner was, jumping 10 px up
	 * on 40% of frames (two clusters) plus 1 px noise. Vision's pupil is steady (0.3 px noise). Reading
	 * steadily, the applied shift must not shake and must still cancel the glance: realistic noise under
	 * 0.05 px RMS frame-to-frame (vertical and sideways), wandering under 0.6 px within any second, 95% of
	 * the clean correction (on the user's clip the refiner fix alone still left 0.46 px RMS); the bimodal
	 * refiner under 0.15 px RMS, 90% of it. At 30 and 15 fps; with and without Vision's points (the plugin
	 * always passes them; without, nothing steadier than the refined reading says where the eye is, so the
	 * low-pass follows the noise's own slow runs, a 1 px mean over most of a second now and then: under
	 * 0.1 px RMS, 1.5 px within a second); the bimodal refiner's eyes flipping on their own or together. Its
	 * flipping starts after a second of plain noise (as when the light changes while reading); when the face first
	 * appears with the refiner already flipping 60/40, which level is the eye takes a few seconds of
	 * readings to tell (reported below). 2 px Gaussian noise (no outliers) as a harder case. */
	{
		for (int fr = 0; fr < 2; fr++) {
			const double fps = fr ? 15.0 : 30.0;
			const real_noise clean = {fps, 1, 0, 0, 0, 0, 0.0f};
			float rms, rms_x, worst, wobble, clean_mean;
			real_steady(&clean, 1, &rms, &rms_x, &worst, &wobble, &clean_mean);
			CHECK(fabsf(clean_mean - 0.10f * RW) < 0.3f, "clean notes correction %.2f px, expected %.2f", clean_mean,
			      0.10f * RW);
			for (int vis = 1; vis >= 0; vis--) {
				const real_noise rn = {fps, vis, 0.0f, 0, 1.2f, 0.3f, 0.05f};
				float mean = 0, r_worst = 0, x_worst = 0, s_worst = 0, w_worst = 0, m_lo = 1e9f;
				for (unsigned seed = 1; seed <= 6; seed++) {
					real_steady(&rn, seed * 11, &rms, &rms_x, &worst, &wobble, &mean);
					r_worst = fmaxf(r_worst, rms);
					x_worst = fmaxf(x_worst, rms_x);
					s_worst = fmaxf(s_worst, worst);
					w_worst = fmaxf(w_worst, wobble);
					m_lo = fminf(m_lo, mean);
				}
				const float r_max = vis ? 0.05f : 0.1f, w_max = vis ? 0.6f : 1.5f;
				CHECK(r_worst < r_max && x_worst < r_max && w_worst < w_max && m_lo > 0.95f * clean_mean,
				      "realistic noise (%s, %.0f fps): shift shakes %.3f px RMS (largest step %.2f, sideways %.3f px "
				      "RMS), wanders %.2f px within a second, correction %.2f of %.2f px",
				      vis ? "Vision" : "no Vision", fps, r_worst, s_worst, x_worst, w_worst, m_lo, clean_mean);
				printf("ok: realistic noise (%s, %.0f fps): %.3f px RMS frame-to-frame (largest step %.2f px), "
				       "sideways %.3f px RMS, wanders %.2f px within a second at most, correction %.0f%% of clean\n",
				       vis ? "Vision" : "no Vision", fps, r_worst, s_worst, x_worst, w_worst, 100.0f * m_lo / clean_mean);
			}
			for (int vis = 1; vis >= 0; vis--)
				for (int both = 0; both < 2; both++) {
					const real_noise rn = {fps, vis, 0.4f, both, 1.0f, 0.3f, 0.0f};
					float mean = 0, r_worst = 0, x_worst = 0, s_worst = 0, m_lo = 1e9f;
					for (unsigned seed = 1; seed <= 4; seed++) {
						real_steady(&rn, seed * 7 + both, &rms, &rms_x, &worst, &wobble, &mean);
						r_worst = fmaxf(r_worst, rms);
						x_worst = fmaxf(x_worst, rms_x);
						s_worst = fmaxf(s_worst, worst);
						m_lo = fminf(m_lo, mean);
					}
					CHECK(r_worst < 0.15f && x_worst < 0.15f && m_lo > 0.9f * clean_mean,
					      "bimodal refiner (%s, eyes %s, %.0f fps): shift shakes %.2f px RMS (largest %.2f), "
					      "sideways %.2f px RMS, correction %.2f of %.2f px",
					      vis ? "Vision" : "no Vision", both ? "together" : "apart", fps, r_worst, s_worst, x_worst,
					      m_lo, clean_mean);
					printf("ok: bimodal refiner (%s, eyes flip %s, %.0f fps): %.3f px RMS frame-to-frame (largest "
					       "%.2f px), sideways %.3f px RMS, correction %.0f%% of clean\n",
					       vis ? "Vision" : "no Vision", both ? "together" : "apart", fps, r_worst, s_worst, x_worst,
					       100.0f * m_lo / clean_mean);
				}
			for (int vis = 1; vis >= 0; vis--) {
				const real_noise rn = {fps, vis, 0.0f, 0, 2.0f, 0.3f, 0.0f};
				float mean;
				real_steady(&rn, 5, &rms, &rms_x, &worst, &wobble, &mean);
				CHECK(rms < 0.15f && rms_x < 0.15f && fabsf(mean - clean_mean) < 0.05f * clean_mean,
				      "2 px refined noise (%s, %.0f fps): %.2f px RMS (sideways %.2f), correction %.2f of %.2f px",
				      vis ? "Vision" : "no Vision", fps, rms, rms_x, mean, clean_mean);
				printf("ok: 2 px refined noise (%s, %.0f fps): %.3f px RMS frame-to-frame (largest %.2f px), "
				       "sideways %.3f px RMS\n",
				       vis ? "Vision" : "no Vision", fps, rms, worst, rms_x);
			}
			/* Locking on when the face appears with the refiner already flipping 60/40 (each eye on its own,
			 * or both together): the wrong level can win the first readings, and only the readings since
			 * can show it. Bounded loosely so a regression shows. */
			int lock_worst = 0;
			for (int both = 0; both < 2; both++)
				for (unsigned seed = 1; seed <= 8; seed++) {
					const real_noise rn = {fps, 1, 0.4f, both, 1.0f, 0.3f, 0.0f};
					const int lk = real_lock_on(&rn, seed * 13 + both);
					lock_worst = lk > lock_worst ? lk : lock_worst;
				}
			CHECK(lock_worst < 300, "bimodal refiner from the start (%.0f fps): locked on only from frame %d", fps,
			      lock_worst);
			printf("ok: bimodal refiner from the start (%.0f fps): locked on by frame %d (%.1f s) at worst\n", fps,
			       lock_worst, lock_worst / fps);
		}
	}

	/* 29. Real glances stay crisp in the same noise: lens -> notes and back, both sensors moving (Vision
	 * by only 60% of the glance), 32 glances each way. Averaged over the glances (the measurement noise
	 * averaged out), the correction reaches 90% (or lets go to 10% coming back) on the first or second
	 * frame of the new gaze and never overshoots (5%). Single glances carry their own frames' measurement
	 * noise: at least 90% of them reach 90% on the first or second frame, every one within 4 frames on
	 * 1 px noise with Vision. With the old bimodal refiner a glance can only show once at least one eye
	 * reads it, which with 40% flips per eye can take a frame more (the average lands by the third frame,
	 * 85% of single glances by the second). Without Vision's points nothing better than the refined reading
	 * says where the gaze landed (85% by the second frame). 30 and 15 fps. */
	{
		const char *names[5] = {"1 px noise", "1 px noise", "realistic noise", "realistic noise", "bimodal refiner"};
		for (int fr = 0; fr < 2; fr++)
			for (int c = 0; c < 5; c++) {
				const double fps = fr ? 15.0 : 30.0;
				const int vis = c != 1 && c != 3, flips = c == 4, rnoise = c == 2 || c == 3;
				const real_noise rn = {fps, vis, flips ? 0.4f : 0.0f, 0, rnoise ? 1.2f : 1.0f, vis ? 0.3f : 0.0f,
						       rnoise ? 0.05f : 0.0f};
				int dn, up, late;
				float over, quick;
				real_glances(&rn, 11 + c, &dn, &up, &over, &quick, &late);
				const int mlate = flips ? 2 : 1;
				const float qmin = (flips || !vis) ? 0.85f : 0.9f;
				const int slate = c == 0 ? 4 : 12;
				CHECK(dn <= mlate && up <= mlate && over < 0.05f * 0.10f * RW && quick >= qmin && late <= slate,
				      "glances (%s, %s, %.0f fps): mean reaches on frame %d going down, %d coming back, mean "
				      "overshoot %.2f px; %.0f%% of single glances by the second frame, slowest frame %d",
				      vis ? "Vision" : "no Vision", names[c], fps, dn, up, over, 100.0f * quick, late);
				printf("ok: glances (%s, %s, %.0f fps): mean reaches 90%% on frame %d down / %d back, overshoot "
				       "%.2f px; %.0f%% of single glances by the second frame, slowest frame %d\n",
				       vis ? "Vision" : "no Vision", names[c], fps, dn, up, over, 100.0f * quick, late);
			}
	}

	/* 30. A single-frame outlier (10 px up or down, or 4 px) in one eye only, at the notes: on a clean
	 * signal the output moves under 0.2 px; in noise (1 px, or the realistic 1.2 px with its own 2-3 px
	 * outliers) it differs from the same run without the outlier by under 0.2 px, and never makes a bigger
	 * step than the noise does (0.05 px). With Vision's points, the same in both eyes at once is ignored
	 * too. Without them, a 4 px outlier in one eye in the realistic noise moves the two eyes' reading 2 px,
	 * which the noise itself does now and then: it is averaged like the noise (under 0.3 px). 30 and 15 fps. */
	{
		float worst1 = 0, worst2 = 0, worst_n[2] = {0, 0}, worst_s = 0;
		const float sizes[] = {-FLIP_PX, FLIP_PX, -4.0f, 4.0f};
		for (int fr = 0; fr < 2; fr++)
			for (int vis = 0; vis < 2; vis++)
				for (int k = 0; k < 4; k++)
					for (int which = 0; which < 3; which++) {
						if (which == 2 && !vis)
							continue;
						const double fps = fr ? 15.0 : 30.0;
						const real_noise quiet = {fps, vis, 0, 0, 0.0f, 0.0f, 0.0f};
						float d, st;
						real_outlier(&quiet, which, sizes[k], 21 + k, &d, &st);
						CHECK(d < 0.2f, "single-frame %.0f px outlier in %s (%s, %.0f fps, clean) moved the output %.2f px",
						      sizes[k], which == 2 ? "both eyes" : "one eye", vis ? "Vision" : "no Vision", fps, d);
						if (which == 2)
							worst2 = fmaxf(worst2, d);
						else
							worst1 = fmaxf(worst1, d);
						for (int nz = 0; nz < 2; nz++) {
							const real_noise rn = {fps, vis, 0, 0, nz ? 1.2f : 1.0f, 0.3f, nz ? 0.05f : 0.0f};
							real_outlier(&rn, which, sizes[k], 21 + k, &d, &st);
							const float d_max = !vis && nz && fabsf(sizes[k]) < FLIP_PX ? 0.3f : 0.2f;
							CHECK(d < d_max && st < 0.05f,
							      "single-frame %.0f px outlier in %s (%s, %.0f fps, %s): %.2f px off the run "
							      "without it, step %.2f px bigger than the noise makes",
							      sizes[k], which == 2 ? "both eyes" : "one eye", vis ? "Vision" : "no Vision", fps,
							      nz ? "realistic noise" : "1 px noise", d, st);
							worst_n[vis] = fmaxf(worst_n[vis], d);
							worst_s = fmaxf(worst_s, st);
						}
					}
		printf("ok: single-frame outliers: clean signal, one eye moves the output %.2f px at most, both eyes (with "
		       "Vision) %.2f px; in noise %.2f px off the run without it (%.2f without Vision), steps %.2f px over "
		       "the noise's\n",
		       worst1, worst2, worst_n[1], worst_n[0], worst_s);
	}

	/* 31. Looking away and back with real noise (Vision; the realistic refined noise, and the old refiner
	 * flipping 10 px on 40% of frames in each eye with 1 px noise): to the keyboard (0.3 eye widths down),
	 * sideways out of the notes (eyes only, half an eye width), and turning the head (40 degrees). Reading
	 * the notes the correction is locked on; looking away it lets go fully, and looking back it takes over
	 * fully, both as a gradual fade: the applied shift never changes by more than 1.25 px in a frame at
	 * 30 fps (2.5 px at 15: the same speed), so no snap; and it only ever shrinks while letting go (no
	 * wobble; the held pre-look-away shift fades) and only grows while taking over (up to the reading's own
	 * jitter, 0.3 px). */
	{
		const real_gaze aways[3] = {{0.30f, 0.0f, 0.0f}, {0.10f, 0.5f, 0.0f}, {0.10f, 0.0f, 40.0f}};
		const char *names[3] = {"keyboard", "sideways", "head turn"};
		for (int fr = 0; fr < 2; fr++)
			for (int nz = 0; nz < 2; nz++)
				for (int a = 0; a < 3; a++) {
					const double fps = fr ? 15.0 : 30.0;
					const real_noise rn = {fps, 1, nz ? 0.4f : 0.0f, 0, nz ? 1.0f : 1.2f, 0.3f, nz ? 0.0f : 0.05f};
					const float bound = fr ? 2.5f : 1.25f;
					float w_step = 0, w_out = 0, w_in = 0, w_left = 0, w_back = 1e9f;
					for (unsigned seed = 1; seed <= 4; seed++) {
						float step, ag_out, ag_in, left, back;
						real_look_away(&rn, aways[a], 40 + seed, &step, &ag_out, &ag_in, &left, &back);
						w_step = fmaxf(w_step, step);
						w_out = fmaxf(w_out, ag_out);
						w_in = fmaxf(w_in, ag_in);
						w_left = fmaxf(w_left, left);
						w_back = fminf(w_back, back);
					}
					CHECK(w_step <= bound && w_out < 0.05f && w_in < 0.3f && w_left < 0.2f &&
						      w_back > 0.9f * 0.10f * RW,
					      "look away (%s, %s, %.0f fps): largest step %.2f px, against the fade %.2f px out / %.2f "
					      "px in, %.2f px left away, %.2f px corrected back",
					      names[a], nz ? "bimodal refiner" : "realistic noise", fps, w_step, w_out, w_in, w_left,
					      w_back);
					printf("ok: look away and back (%s, %s, %.0f fps): largest step %.2f px, against the fade %.2f "
					       "out / %.2f in, %.2f px left away, %.2f of %.2f px corrected back\n",
					       names[a], nz ? "bimodal refiner" : "realistic noise", fps, w_step, w_out, w_in, w_left,
					       w_back, 0.10f * RW);
				}
	}

	if (failures) {
		printf("%d FAILED\n", failures);
		return 1;
	}
	printf("ALL CHECKS PASSED\n");
	return 0;
}
