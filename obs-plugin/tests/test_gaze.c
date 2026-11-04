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
	float clump;         /* lash band thickness varying along the lid by +-this fraction (clumps); 0: even */
	float sclera, iris;  /* grey levels; 0: 200 and 80 */
	float tex;           /* iris texture (radial furrows), +-grey levels; 0: none */
} real_eye;

static real_eye real_eye_at(float k, float cx, float cy)
{
	const real_eye r = {cx, cy, 43 * k, 14 * k, 13 * k, cx + 2 * k, cy - 0.5f * k, 17 * k, 3.5f * k, 1.2f * k, 6, 1,
			    0, 0, 0, 0, 0, 0, 0, 0, 0};
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
			const float lu = (px - s->cx) * (43 / s->hw);
			const float lash = s->lash * (1 + s->clump * sinf(0.9f * lu) * sinf(0.23f * lu));
			float v = 150; /* skin */
			if (open > 0) {
				const float shade = 25 * expf(-fmaxf(py - yu, 0) / (0.07f * s->hw)); /* lid shadow */
				float vo = (s->sclera > 0 ? s->sclera : 200) - shade;               /* sclera */
				const float d = hypotf(px - s->ix, py - s->iy);
				const float ci = cov01((s->ir - d) / b + 0.5f);
				if (ci > 0) {
					float iv = (s->iris > 0 ? s->iris : 80) -
						   15 * cov01((d - 0.75f * s->ir) / (0.25f * s->ir)); /* limbal ring */
					if (s->tex > 0) {
						const float an = atan2f(py - s->iy, px - s->ix), rr = d / s->ir;
						iv += s->tex * (0.6f * sinf(17 * an + 3 * rr) * sinf(5 * an - 2.1f) +
								0.4f * sinf(9 * an + 11 * rr + 1.3f));
					}
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
				vo += cov01((yu + lash - py) / b + 0.5f) * (45 - vo); /* lashes */
				v += open * (vo - v);
			}
			if (fabsf((px - s->cx) / s->hw) < 0.95f) /* lashes above the lid line too */
				v += cov01((py - (yu - 0.5f * lash)) / b + 0.5f) * cov01((yu - py) / b + 0.5f) * (45 - v);
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

/* JPEG-like compression of a grey image in place: 8x8 DCT blocks quantised with the standard luma
 * table at quality q (1..100), like the camera's or the encoder's. Blocks at the right and bottom
 * edges that don't fill 8x8 are left alone. */
static void jpeg_like(uint8_t *img, int w, int h, int q)
{
	static const int QT[64] = {16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
				   14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
				   18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
				   49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};
	const float scale = q < 50 ? 5000.0f / (float)q : 200.0f - 2.0f * (float)q;
	float Q[64], C[8][8];
	for (int i = 0; i < 64; i++)
		Q[i] = fmaxf(floorf(((float)QT[i] * scale + 50) / 100), 1);
	for (int u = 0; u < 8; u++)
		for (int x = 0; x < 8; x++)
			C[u][x] = (u ? sqrtf(2.0f / 8) : sqrtf(1.0f / 8)) * cosf((float)((2 * x + 1) * u) * 3.14159265f / 16);
	for (int by = 0; by + 8 <= h; by += 8)
		for (int bx = 0; bx + 8 <= w; bx += 8) {
			float B[8][8], T[8][8], D[8][8];
			for (int y = 0; y < 8; y++)
				for (int x = 0; x < 8; x++)
					B[y][x] = (float)img[(by + y) * w + bx + x] - 128;
			for (int u = 0; u < 8; u++)
				for (int x = 0; x < 8; x++) {
					float a = 0;
					for (int y = 0; y < 8; y++)
						a += C[u][y] * B[y][x];
					T[u][x] = a;
				}
			for (int u = 0; u < 8; u++)
				for (int v = 0; v < 8; v++) {
					float a = 0;
					for (int x = 0; x < 8; x++)
						a += T[u][x] * C[v][x];
					D[u][v] = roundf(a / Q[u * 8 + v]) * Q[u * 8 + v];
				}
			for (int y = 0; y < 8; y++)
				for (int v = 0; v < 8; v++) {
					float a = 0;
					for (int u = 0; u < 8; u++)
						a += C[u][y] * D[u][v];
					T[y][v] = a;
				}
			for (int y = 0; y < 8; y++)
				for (int x = 0; x < 8; x++) {
					float a = 128;
					for (int v = 0; v < 8; v++)
						a += T[y][v] * C[v][x];
					img[(by + y) * w + bx + x] = (uint8_t)(a < 0 ? 0 : (a > 255 ? 255 : lrintf(a)));
				}
		}
}

/* The realistic eye glancing down by dy px (check 13c): the upper lid follows fup of the move, the
 * lower lid drops a tenth of it; clumped lashes. */
static real_eye reading_eye(float k, float cx, float cy, float dy, float fup, float sclera, float iris, float tex,
			    uint32_t seed)
{
	real_eye s = real_eye_at(k, cx, cy);
	s.iy += dy;
	s.up_h -= fup * dy;
	s.lo_h += 0.1f * dy;
	s.clump = 0.35f;
	s.sclera = sclera;
	s.iris = iris;
	s.tex = tex;
	s.seed = seed;
	return s;
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
 * Vision's own pupil points (if `vision`) are coarse: they move R_VIS of the true movement, plus vd[e].
 * R_VIS is 0.6 unless a check sets r_vis; rx adds what a check needs on top (lids closing, the points
 * dragged by them, Vision's point off on its own), and is back at its defaults after each check. */
#define RW 86.0f
#define RH 26.0f
#define R_LID 0.5f
static float r_vis = 0.6f;
#define R_VIS r_vis
typedef struct {
	float open;     /* the lids' openness, as a fraction of where the glance puts them (1: open; under 0.5 both
			 * lids close on the corner line, and at 0 they meet: shut) */
	float drag[2];  /* each eye's refined point dragged down by this (eye widths; a lid, a refiner slip) */
	float vdrag;    /* Vision's points moved down by this (eye widths) */
} real_extra;
static const real_extra rx_none = {1.0f, {0.0f, 0.0f}, 0.0f};
static real_extra rx = {1.0f, {0.0f, 0.0f}, 0.0f};
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
	const float gpx = gz.glance * RW, spx = gz.side * RW;
	const float drop = R_LID * gpx + (1.0f - rx.open) * (RH - R_LID * gpx);
	const float cx[2] = {960 - 1.5f * RW, 960 + 1.5f * RW};
	for (int e = 0; e < 2; e++) {
		cf_pt *pts = e == 0 ? f.left_eye : f.right_eye;
		for (int i = 0; i < 16; i++) {
			const float a = (float)i / 16.0f * 6.2831853f;
			const float s = sinf(a);
			pts[i].x = cx[e] + 0.5f * RW * cosf(a);
			pts[i].y = 500 + 0.5f * RH * s;
			if (rx.open < 0.5f)
				pts[i].y = 500 + 0.5f * RH * s * fmaxf(rx.open, 0.0f);
			else if (s < -1e-3f)
				pts[i].y += drop * -s;
		}
		const cf_pt pupil = {cx[e] + spx + d[e][0], 500 + gpx + d[e][1] + rx.drag[e] * RW};
		const cf_pt vp = {cx[e] + R_VIS * spx + vd[e][0], 500 + R_VIS * gpx + vd[e][1] + rx.vdrag * RW};
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

/* Calibrated like the user: notes 0.10 eye widths (8.6 px) under the lens; with real_vis_cal, Vision's
 * points are known from the same two calibrations (as the plugin keeps them), which sets their gain. */
static int real_vis_cal = 0;
static cf_gaze_params real_params(void)
{
	cf_gaze_params p = {.strength = 1.0f, .max_glance = 0.15f, .baseline = 0.0f, .baseline_open = RH / RW,
			    .has_notes = 1, .notes_offset = -0.10f, .notes_side = 0.0f,
			    .notes_open = (RH - R_LID * 0.10f * RW) / RW};
	if (real_vis_cal) {
		p.has_vis_calib = 1;
		p.vis_baseline = 0.0f;
		p.vis_notes = -0.10f * R_VIS;
	}
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
 * the latest frame on which one did; near, the latest frame on which one got within 25% of the target
 * (2.2 px: twice the refined noise, so a glance that landed but carries its frames' noise counts). */
static void real_glances(const real_noise *rn, unsigned seed, int *mean_dn, int *mean_up, float *over, float *quick,
			 int *late, int *near)
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
	*late = *near = 0;
	int n_quick = 0;
	const real_gaze lens = {0, 0, 0};
	for (int i = 0; i < (int)(3.0 * rn->fps); i++, t += dt) {
		cf_face f = noisy_real(rn, lens);
		cf_gaze_update(&g, &f, &p, t, out);
	}
	for (int cyc = 0; cyc < CYC; cyc++)
		for (int dir = 0; dir < 2; dir++) {
			int land = 99, land25 = 99;
			for (int i = 0; i < hold; i++, t += dt) {
				cf_face f = noisy_real(rn, dir == 0 ? at_notes() : lens);
				const int n = cf_gaze_update(&g, &f, &p, t, out);
				const float c = n ? -out[0].sy : 0.0f;
				if (land == 99 && (dir == 0 ? c >= 0.9f * target : c <= 0.1f * target))
					land = i;
				if (land25 == 99 && (dir == 0 ? c >= 0.75f * target : c <= 0.25f * target))
					land25 = i;
				if (i < K)
					mean[dir][i] += c / CYC;
			}
			*late = *late > land ? *late : land;
			*near = *near > land25 ? *near : land25;
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
 * back at the notes. For each hand-off: land[], the frame (0 = the first with the new gaze) from which on
 * the applied shift stays where it ends up (letting go: under 0.5 px; taking over: within 1 px of the
 * correction at the end, the readings' own noise); the largest frame-to-frame change of the applied shift
 * (vertical or sideways, px) over both hand-offs, all of it (step) and after the landing frames
 * (step_after); over the first 0.8 s of each, the largest step against the way it goes (the shift growing
 * while letting go, shrinking while taking over); and the shift left at the end of the look-away, and the
 * correction at the end of the look back (px). */
typedef struct {
	int land[2];
	float step, step_after, against_out, against_in, left, back;
} look_away_result;
static void real_look_away(const real_noise *rn, real_gaze away, unsigned seed, look_away_result *r)
{
	const cf_gaze_params p = real_params();
	cf_gaze g;
	cf_gaze_init(&g);
	cf_eye out[2];
	rng_seed(seed);
	const double dt = 1.0 / rn->fps;
	const int n0 = (int)(3.0 * rn->fps), n1 = (int)(1.5 * rn->fps), nf = (int)(0.8 * rn->fps);
	float sys[2][128], sxs[2][128], mags[2][128];
	float py = 0, px = 0, pm = 0;
	memset(r, 0, sizeof(*r));
	for (int i = 0; i < n0 + 2 * n1; i++) {
		const int phase = i < n0 ? 0 : (i < n0 + n1 ? 1 : 2);
		cf_face f = noisy_real(rn, phase == 1 ? away : at_notes());
		const int n = cf_gaze_update(&g, &f, &p, 100.0 + i * dt, out);
		const float sy = n ? out[0].sy : 0, sx = n ? out[0].sx : 0, mag = sqrtf(sy * sy + sx * sx);
		if (phase > 0) {
			const int k = i - (phase == 1 ? n0 : n0 + n1);
			if (k < 128) {
				sys[phase - 1][k] = sy;
				sxs[phase - 1][k] = sx;
				mags[phase - 1][k] = mag;
			}
			r->step = fmaxf(r->step, fmaxf(fabsf(sy - py), fabsf(sx - px)));
		}
		if (i == n0 + n1 - 1)
			r->left = mag;
		r->back = -sy;
		py = sy;
		px = sx;
		pm = mag;
	}
	(void)pm;
	const int len = n1 < 128 ? n1 : 128;
	for (int ph = 0; ph < 2; ph++) {
		int land = 0;
		for (int k = 0; k < len; k++)
			if (ph == 0 ? mags[0][k] >= 0.5f : fabsf(sys[1][k] + r->back) >= 1.0f)
				land = k + 1;
		r->land[ph] = land;
		for (int k = 1; k < len; k++) {
			if (k > land)
				r->step_after = fmaxf(r->step_after, fmaxf(fabsf(sys[ph][k] - sys[ph][k - 1]),
									   fabsf(sxs[ph][k] - sxs[ph][k - 1])));
			if (k < nf && ph == 0)
				r->against_out = fmaxf(r->against_out, mags[0][k] - mags[0][k - 1]);
			if (k < nf && ph == 1)
				r->against_in = fmaxf(r->against_in, mags[1][k - 1] - mags[1][k]);
		}
	}
}

#include "test_pose.c"

/* ---- Head and lids (checks H1-H9)
 *
 * The user's case: a MacBook camera at 0.5-0.7 m, eyes ~75 px wide and ~23 px open, notes 5-15 cm under the
 * camera. Reading, the eyes move only ~1.5 px against their corners, the head tilts down ~6 deg (up to
 * 19), the upper lid drops ~0.01 eye widths and the lower one rises ~0.01. The faces below carry real head
 * landmarks (test_pose.c's typical 3D face at 60 cm, so the pose reads tilts ~1.1x like a real one). */
#define HW 75.0f
#define HH 23.0f
#define HL_EYE_K 0.41f    /* gaze.c's EYE_K */
#define HL_LID_SHIFT 0.45f /* gaze.c's LID_SHIFT */
#define HL_HEAD_SLEW_PX 0.4f /* gaze.c's HEAD_SLEW_PX */
typedef struct {
	float glance;           /* the eyes' drop in the head, eye widths (negative: up) */
	float side;             /* the eyes sideways in the head, eye widths */
	float pitch, yaw;       /* the head, degrees (down / towards the image's right positive) */
	float up_drop, lo_rise; /* the lids against looking into the lens, eye widths */
	float shut;             /* 0..1: the lids closing (a blink) */
} hl_state;

static float hl_head_noise = 0.5f; /* head landmarks' noise, x the eye landmarks' */

static cf_face face_hl(hl_state s, float noise_px)
{
	static const synth_face sf = {0, 9, 0, 0, 2, 6};
	cf_face f;
	memset(&f, 0, sizeof(f));
	f.found = 1;
	f.image_width = 1920;
	f.image_height = 1080;
	f.face_width = 380;
	const float cx[2] = {960 - 1.5f * HW, 960 + 1.5f * HW};
	const float gpx = s.glance * HW, spx = s.side * HW;
	for (int e = 0; e < 2; e++) {
		cf_pt *pts = e == 0 ? f.left_eye : f.right_eye;
		for (int i = 0; i < 16; i++) {
			const float a = (float)i / 16.0f * 6.2831853f, sn = sinf(a);
			float y = 0.5f * HH * sn;
			if (sn < -1e-3f) /* upper lid */
				y += s.up_drop * HW * -sn;
			else if (sn > 1e-3f) /* lower lid */
				y -= s.lo_rise * HW * sn;
			pts[i].x = cx[e] + 0.5f * HW * cosf(a) + (noise_px > 0 ? gauss(noise_px) : 0.0f);
			pts[i].y = 500 + y * (1.0f - s.shut) + (noise_px > 0 ? gauss(noise_px) : 0.0f);
		}
		const cf_pt pupil = {cx[e] + spx + (noise_px > 0 ? gauss(noise_px) : 0.0f),
				     500 + gpx + (noise_px > 0 ? gauss(noise_px) : 0.0f)};
		const cf_pt vp = {cx[e] + 0.6f * spx, 500 + 0.6f * gpx};
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
	f.has_vision_pupils = 1;
	synth_lm lm;
	synth_project(&sf, s.yaw, s.pitch, 0, SF_FOCAL, 960, 480, &lm);
	if (noise_px > 0)
		synth_noise(&lm, hl_head_noise * noise_px);
	f.has_head_geom = synth_measure(&lm, &f.head);
	return f;
}

/* Calibrates both steps through the API like the plugin does: into the lens (head straight, lids open),
 * then the notes 0.02 eye widths down with the head still (the user's finder reads ~1.5 px). */
static cf_gaze_params hl_params(cf_calib_result *lens_out)
{
	cf_gaze_params p = {.strength = 1.0f, .max_glance = 0.15f};
	cf_gaze g;
	cf_gaze_init(&g);
	cf_eye out[2];
	double t = 50.0;
	cf_calib_result lens = {0}, notes = {0};
	int got = 0;
	cf_gaze_start_calibration(&g, CF_CALIB_LENS);
	for (int i = 0; i < 120 && got <= 0; i++, t += 1.0 / 30.0) {
		const hl_state s = {0};
		cf_face f = face_hl(s, 0);
		cf_gaze_update(&g, &f, &p, t, out);
		got = cf_gaze_take_calibration(&g, &lens);
	}
	p.baseline = lens.offset;
	p.baseline_open = lens.openness;
	p.head_ref = lens.head;
	p.baseline_upper = lens.upper;
	p.baseline_lower = lens.lower;
	got = 0;
	cf_gaze_start_calibration(&g, CF_CALIB_NOTES);
	for (int i = 0; i < 120 && got <= 0; i++, t += 1.0 / 30.0) {
		const hl_state s = {.glance = 0.02f, .up_drop = 0.01f, .lo_rise = 0.01f};
		cf_face f = face_hl(s, 0);
		cf_gaze_update(&g, &f, &p, t, out);
		got = cf_gaze_take_calibration(&g, &notes);
	}
	p.has_notes = 1;
	p.notes_offset = notes.offset;
	p.notes_side = notes.side;
	p.notes_open = notes.openness;
	p.has_vis_calib = lens.has_vis && notes.has_vis; /* as the plugin saves them */
	p.vis_baseline = lens.vis_offset;
	p.vis_notes = notes.vis_offset;
	if (lens_out)
		*lens_out = lens;
	return p;
}

/* The applied shifts along the face's up direction (the synthetic face is level): iris lift, upper-lid
 * lift and lower-lid drop, px; and the iris' sideways shift. */
typedef struct {
	float iris, lid, low, side;
} hl_out;
static hl_out hl_read(int n, const cf_eye *e)
{
	const hl_out o = {n ? -e->sy : 0, n ? -e->lid_sy : 0, n ? e->low_sy : 0, n ? e->sx : 0};
	return o;
}

static float hl_lerp(float a, float b, float k)
{
	return a + (b - a) * k;
}

static void test_head_lids(void)
{
	cf_calib_result lens;
	const cf_gaze_params p = hl_params(&lens);
	cf_gaze g;
	cf_eye out[2];
	const double dt = 1.0 / 30.0;

	/* H1. Step 1 learns the lids: upper above and lower below the corner line, eye widths. */
	CHECK(fabsf(lens.upper - 0.5f * HH / HW) < 0.003f && fabsf(lens.lower - 0.5f * HH / HW) < 0.003f && lens.head.ok,
	      "lens calibration: lids %.4f / %.4f (expected %.4f), head %d", lens.upper, lens.lower, 0.5f * HH / HW,
	      lens.head.ok);
	CHECK(p.has_notes && fabsf((p.baseline - p.notes_offset) - 0.02f) < 0.003f, "notes calibration %.4f",
	      p.baseline - p.notes_offset);
	printf("ok: step 1 learns the lids (upper %.4f, lower %.4f eye widths), step 2 the notes at %.4f\n", lens.upper,
	       lens.lower, p.baseline - p.notes_offset);

	/* H2. Reading with the head tilted down 0-10 deg: the iris is lifted by the eyes' drop plus
	 * EYE_K sin(tilt) (the tilt as measured), the upper lid by LID_SHIFT of that plus what is left of its
	 * drop, the lower lid by its rise: the opening is restored to the lens one within 20%. */
	const float pitches[5] = {0, 3, 6, 10, 15};
	float lift6 = 0;
	for (int k = 0; k < 5; k++) {
		cf_gaze_init(&g);
		double t = 100.0;
		int n = 0;
		for (int i = 0; i < 45; i++, t += dt) {
			const hl_state s = {0};
			n = update(&g, face_hl(s, 0), &p, t, out);
		}
		hl_out o = {0, 0, 0, 0};
		double sum_i = 0, sum_l = 0, sum_b = 0;
		for (int i = 0; i < 90; i++, t += dt) {
			const hl_state s = {.glance = 0.02f, .pitch = pitches[k], .up_drop = 0.01f, .lo_rise = 0.01f};
			n = update(&g, face_hl(s, 0), &p, t, out);
			o = hl_read(n, &out[0]);
			if (i >= 75) {
				sum_i += o.iris / 15;
				sum_l += o.lid / 15;
				sum_b += o.low / 15;
			}
		}
		const float w = (float)g.rx[0].value, meas = g.pose.pitch;
		const float want = (HL_EYE_K * sinf(fminf(fmaxf(meas, 0), 18.0f) * 0.017453293f) + 0.02f) * w;
		const float head_px = HL_EYE_K * sinf(fmaxf(meas, 0) * 0.017453293f) * w;
		const float up_restore = (float)sum_l - HL_LID_SHIFT * head_px; /* beyond following the head's lift */
		const float opening = up_restore + (float)sum_b, narrowing = 0.02f * w;
		printf("    head %4.1f deg down (reads %4.1f): iris lifted %.2f px (want %.2f), upper lid %.2f (%.2f beyond the "
		       "head's share), lower lid %.2f: opening restored %.2f of %.2f px\n",
		       pitches[k], meas, sum_i, want, sum_l, up_restore, sum_b, opening, narrowing);
		CHECK(fabsf(meas - pitches[k]) <= 0.25f * pitches[k] + 0.3f, "head %.0f deg reads %.1f", pitches[k], meas);
		CHECK(fabsf((float)sum_i - want) < 0.06f * want + 0.1f, "head %.0f deg: iris lifted %.2f px, want %.2f",
		      pitches[k], sum_i, want);
		CHECK(fabsf(opening - narrowing) < 0.2f * narrowing && fabsf(up_restore - 0.01f * w) < 0.25f * 0.01f * w &&
			      fabsf((float)sum_b - 0.01f * w) < 0.25f * 0.01f * w,
		      "head %.0f deg: lids restore %.2f + %.2f px of %.2f + %.2f", pitches[k], up_restore, sum_b, 0.01f * w,
		      0.01f * w);
		if (k == 2)
			lift6 = (float)sum_i;
	}
	printf("ok: head tilt and lids: the iris is lifted by the eyes' drop plus 0.41 sin(tilt), the opening restored\n");

	/* H3. The head tilted down while the eyes look into the lens (the eyes then turn up in the head by just
	 * as much): nothing to correct. And without a lens calibration of the head (a typical face, whose
	 * tilt is not measured) the head adds nothing. */
	{
		float worst = 0;
		for (int k = 1; k < 4; k++) {
			cf_gaze_init(&g);
			double t = 200.0;
			int n = 0;
			for (int i = 0; i < 90; i++, t += dt) {
				const float pk = pitches[k] * fminf(1.0f, (float)i / 30.0f);
				const hl_state s = {.glance = -HL_EYE_K * sinf(1.1f * pk * 0.017453293f), .pitch = pk};
				n = update(&g, face_hl(s, 0), &p, t, out);
				if (i >= 60)
					worst = fmaxf(worst, fabsf(hl_read(n, &out[0]).iris));
			}
		}
		cf_gaze_params pu = p;
		pu.head_ref.ok = 0;
		cf_gaze_init(&g);
		double t = 300.0;
		hl_out o = {0, 0, 0, 0};
		for (int i = 0; i < 90; i++, t += dt) {
			const hl_state s = {.glance = 0.02f, .pitch = 8, .up_drop = 0.01f, .lo_rise = 0.01f};
			o = hl_read(update(&g, face_hl(s, 0), &pu, t, out), &out[0]);
		}
		printf("    head down 3-10 deg, eyes on the lens: %.2f px at most; uncalibrated head, reading: iris %.2f px\n",
		       worst, o.iris);
		CHECK(worst < 0.3f, "head down with the eyes on the lens corrected by %.2f px", worst);
		CHECK(fabsf(o.iris - 0.02f * HW) < 0.2f, "uncalibrated head added to the correction (%.2f px)", o.iris);
		printf("ok: a head tilted down with the eyes on the lens is left alone\n");
	}

	/* The same faces through a twin without head and lid calibration: what is left is the glance's own path
	 * (tested above), and the difference is what the head and lid terms add. */
	cf_gaze_params pu = p;
	pu.head_ref.ok = 0;
	pu.baseline_upper = pu.baseline_lower = 0;
	cf_gaze g2;
	cf_eye out2[2];

	/* H4. Smooth: lens -> notes (saccade) with the head following down to 8 deg and the lids narrowing over
	 * 0.3 s; reading lines (the eyes sweep sideways, then a return saccade) with the head nodding; back to
	 * the lens, the head following up while the eyes hold the lens (turning down in the head as it comes up,
	 * which the gate takes in steps: glide_catch_up()). What the head and lid terms add (against the twin)
	 * to any vertical shift (iris, either lid): with the head moving as while reading (down over 1 s,
	 * nodding 1.5 deg) it changes by no more than 0.3 px a frame outside the saccades; with a quick head
	 * (down in 0.5 s, 0.8 deg a frame) it follows the head as fast as it moves (up to HEAD_SLEW_PX), but
	 * never jumps: no frame's change differs from the last one's by 0.3 px. Back at the lens, the head still
	 * coming up, the eyes are lifted less than 1 px past it. (The whole output is printed too: the glance's
	 * own landing adds to it. With no head turn the sideways shift is the side gate's alone: only printed.) */
	for (int quick = 0; quick < 2; quick++) {
		const int down_n = quick ? 15 : 30, up_n = quick ? 12 : 24;
		const float nod = quick ? 2.0f : 1.5f;
		cf_gaze_init(&g);
		cf_gaze_init(&g2);
		double t = 400.0;
		hl_out prev = {0, 0, 0, 0}, prev_c = {0, 0, 0, 0};
		float prev_d = 0, worst = 0, worst_all = 0, worst_jerk = 0, worst_sacc = 0, peak = 0, worst_side = 0,
		      over = 0;
		int sacc_left = 0, have_d = 0;
		const int N = 300;
		for (int i = 0; i < N; i++, t += dt) {
			hl_state s = {0};
			float sacc = 0;
			if (i >= 30 && i < 240) { /* reading */
				const float k = fminf((float)(i - 30) / (float)down_n, 1.0f),
					    kl = fminf((float)(i - 30) / 9.0f, 1.0f);
				s.glance = 0.02f;
				s.pitch = 8.0f * k * k * (3 - 2 * k) + nod * sinf((float)(i - 30) * 0.105f) * k;
				s.up_drop = 0.01f * kl;
				s.lo_rise = 0.01f * kl;
				const int line = (i - 30) % 50; /* 1.5 s along a line, then back */
				s.side = -0.1f + 0.2f * (float)line / 49.0f;
				if (i == 30 || line == 0)
					sacc = 1;
			} else if (i >= 240) { /* back to the lens, the head following */
				const float k = fminf((float)(i - 240) / (float)up_n, 1.0f);
				s.pitch = 8.0f * (1 - k * k * (3 - 2 * k));
				s.glance = -HL_EYE_K * sinf(1.1f * s.pitch * 0.017453293f);
				if (i == 240)
					sacc = 1;
			}
			if (sacc)
				sacc_left = 5;
			const cf_face f = face_hl(s, 0);
			const hl_out o = hl_read(update(&g, f, &p, t, out), &out[0]);
			const hl_out o2 = hl_read(update(&g2, f, &pu, t, out2), &out2[0]);
			const hl_out c = {o.iris - o2.iris, o.lid - o2.lid, o.low - o2.low, o.side - o2.side};
			const float di = c.iris - prev_c.iris, dl = c.lid - prev_c.lid, db = c.low - prev_c.low;
			const float d = fmaxf(fmaxf(fabsf(di), fabsf(dl)), fabsf(db));
			const float d_all =
				fmaxf(fmaxf(fabsf(o.iris - prev.iris), fabsf(o.lid - prev.lid)), fabsf(o.low - prev.low));
			if (sacc_left > 0) {
				worst_sacc = fmaxf(worst_sacc, d_all);
				have_d = 0;
			} else {
				worst = fmaxf(worst, d);
				worst_all = fmaxf(worst_all, d_all);
				worst_side = fmaxf(worst_side, fabsf(o.side - prev.side));
				if (have_d)
					worst_jerk = fmaxf(worst_jerk, fabsf(di - prev_d));
				prev_d = di;
				have_d = 1;
				if (i > 240)
					over = fmaxf(over, o.iris);
			}
			sacc_left--;
			peak = fmaxf(peak, o.iris);
			prev = o;
			prev_c = c;
		}
		printf("    lens -> reading (head to 8 deg in %.1f s, nodding %.1f deg, scanning lines) -> lens: the head and "
		       "lid terms change by %.3f px a frame at most (change of change %.3f), the whole output by %.3f (%.2f "
		       "in the saccades); iris lifted up to %.2f px, %.2f px past the lens on the way back, %.2f left; "
		       "sideways (side gate) %.2f\n",
		       (float)down_n / 30.0f, nod, worst, worst_jerk, worst_all, worst_sacc, peak, over, prev.iris,
		       worst_side);
		if (quick)
			CHECK(worst < HL_HEAD_SLEW_PX + 0.05f && worst_jerk < 0.3f,
			      "quick head: head and lid terms changed by %.3f px a frame, jumped %.3f px", worst, worst_jerk);
		else
			CHECK(worst < 0.3f, "head and lids made a step of %.3f px outside the saccades", worst);
		CHECK(over < 1.0f && prev.iris < 0.3f && fabsf(prev.lid) < 0.3f && fabsf(prev.low) < 0.3f,
		      "back at the lens %.2f px past it, %.2f / %.2f / %.2f px left", over, prev.iris, prev.lid, prev.low);
	}
	printf("ok: head and lid terms ease in and out (no step over 0.3 px outside saccades)\n");

	/* H5. Letting go still works. To the keyboard with the head (eyes far down, head 25 deg, lids
	 * dropping): the correction fades out. The head alone tilted far down (35 deg): the pitch fade lets go.
	 * Either way the head's share of the eyes as the viewer sees them (its tilt, less what the head and lid
	 * terms add to the lift, against the twin) only ever moves down with the head, never back up against it.
	 * (The whole of it, glance included, is printed: the glance's own path is tested above.) Reading with
	 * the head tilted down 15 deg (well inside the measured range): kept. */
	{
		const struct {
			const char *name;
			hl_state away;
			int keep;
		} cases[3] = {
			{"keyboard with head and eyes", {0.15f, 0, 25, 0, 0.04f, 0.0f, 0}, 0},
			{"head tilted far down", {0.02f, 0, 35, 0, 0.01f, 0.01f, 0}, 0},
			{"reading, head 15 deg down", {0.02f, 0, 15, 0, 0.01f, 0.01f, 0}, 1},
		};
		for (int c = 0; c < 3; c++) {
			cf_gaze_init(&g);
			cf_gaze_init(&g2);
			double t = 500.0;
			hl_out o = {0, 0, 0, 0};
			for (int i = 0; i < 60; i++, t += dt) {
				const hl_state s = {.glance = 0.02f, .pitch = 6, .up_drop = 0.01f, .lo_rise = 0.01f};
				const cf_face f = face_hl(s, 0);
				o = hl_read(update(&g, f, &p, t, out), &out[0]);
				update(&g2, f, &pu, t, out2);
			}
			const hl_out before = o;
			float against = 0, against_all = 0, prev_seen = 0, prev_all = 0;
			for (int i = 0; i < 45; i++, t += dt) {
				const float k = fminf((float)i / 8.0f, 1.0f);
				const hl_state a = cases[c].away;
				const hl_state s = {hl_lerp(0.02f, a.glance, k), 0, hl_lerp(6, a.pitch, k), 0,
						    hl_lerp(0.01f, a.up_drop, k), hl_lerp(0.01f, a.lo_rise, k), 0};
				const cf_face f = face_hl(s, 0);
				o = hl_read(update(&g, f, &p, t, out), &out[0]);
				const hl_out o2 = hl_read(update(&g2, f, &pu, t, out2), &out2[0]);
				/* how far down the eyes look as the viewer sees them, px of iris: the head's share, and all */
				const float head_px = HL_EYE_K * sinf(s.pitch * 0.017453293f) * HW;
				const float seen = head_px - (o.iris - o2.iris), all = s.glance * HW + head_px - o.iris;
				if (i > 0) {
					against = fmaxf(against, prev_seen - seen);
					against_all = fmaxf(against_all, prev_all - all);
				}
				prev_seen = seen;
				prev_all = all;
			}
			printf("    %s: %.2f px -> %.2f px after 1.5 s (lids %.2f / %.2f); the head's share seen moving up "
			       "against it %.3f px (all of it %.3f)\n",
			       cases[c].name, before.iris, o.iris, o.lid, o.low, against, against_all);
			if (cases[c].keep)
				CHECK(o.iris > 0.9f * (HL_EYE_K * sinf(16.5f * 0.017453293f) + 0.02f) * HW,
				      "%s: correction dropped to %.2f px", cases[c].name, o.iris);
			else
				CHECK(o.iris < 0.2f && fabsf(o.lid) < 0.2f && fabsf(o.low) < 0.2f && against < 0.12f,
				      "%s: %.2f px left (lids %.2f / %.2f), %.3f px against the head", cases[c].name, o.iris,
				      o.lid, o.low, against);
		}
		printf("ok: keyboard and far head tilts still let go, reading with the head down is kept\n");
	}

	/* H6. A blink while reading with the head down holds the head and lid terms: what they add to the iris and
	 * to either lid (against the twin) stays within 0.15 px through the blink and after it. (The whole
	 * output, the glance's own hold included, is printed.) */
	{
		cf_gaze_init(&g);
		cf_gaze_init(&g2);
		double t = 600.0;
		hl_out o = {0, 0, 0, 0}, o2 = {0, 0, 0, 0};
		const hl_state rd = {.glance = 0.02f, .pitch = 6, .up_drop = 0.01f, .lo_rise = 0.01f};
		for (int i = 0; i < 75; i++, t += dt) {
			const cf_face f = face_hl(rd, 0);
			o = hl_read(update(&g, f, &p, t, out), &out[0]);
			o2 = hl_read(update(&g2, f, &pu, t, out2), &out2[0]);
		}
		const hl_out before = o, before_c = {o.iris - o2.iris, o.lid - o2.lid, o.low - o2.low, 0};
		const float shut[7] = {0.5f, 0.95f, 0.95f, 0.95f, 0.5f, 0.0f, 0.0f};
		float dev = 0, dev_all = 0;
		for (int i = 0; i < 7 + 15; i++, t += dt) {
			hl_state s = rd;
			if (i < 7) {
				s.shut = shut[i];
				s.glance = 0.0f; /* the pupil point is unreliable while the lids close */
			}
			const cf_face f = face_hl(s, 0);
			o = hl_read(update(&g, f, &p, t, out), &out[0]);
			o2 = hl_read(update(&g2, f, &pu, t, out2), &out2[0]);
			dev = fmaxf(dev, fmaxf(fmaxf(fabsf(o.iris - o2.iris - before_c.iris), fabsf(o.lid - o2.lid - before_c.lid)),
					       fabsf(o.low - o2.low - before_c.low)));
			dev_all = fmaxf(dev_all, fmaxf(fmaxf(fabsf(o.iris - before.iris), fabsf(o.lid - before.lid)),
						       fabsf(o.low - before.low)));
		}
		printf("    blink while reading: iris %.2f, lids %.2f / %.2f px; through it and after, the head and lid terms "
		       "%.3f px off at most (all of it %.3f)\n",
		       before.iris, before.lid, before.low, dev, dev_all);
		CHECK(dev < 0.15f, "a blink moved the held head and lid terms by %.2f px", dev);
		printf("ok: a blink holds the head and lid terms too\n");
	}

	/* H7. Strength over 100%: the eyes' share of the iris lift (and the upper lid that follows it) is lifted
	 * proportionally more, the head's share (measured from the head, not under-read like the pupil) at most
	 * once, the lids are restored no further than the lens opening, and the warp's cap still holds (eyes twice
	 * the notes glance with the head 14 deg down: further down, the head counts as looking past the notes). */
	{
		float iris[2], lid_x[2], low[2], head_px = 0;
		for (int k = 0; k < 2; k++) {
			cf_gaze_params ps = p;
			ps.strength = k ? 2.0f : 1.0f;
			cf_gaze_init(&g);
			double t = 700.0;
			hl_out o = {0, 0, 0, 0};
			for (int i = 0; i < 90; i++, t += dt) {
				const hl_state s = {.glance = 0.02f, .pitch = 6, .up_drop = 0.01f, .lo_rise = 0.01f};
				o = hl_read(update(&g, face_hl(s, 0), &ps, t, out), &out[0]);
			}
			iris[k] = o.iris;
			lid_x[k] = o.lid - HL_LID_SHIFT * o.iris;
			low[k] = o.low;
			if (!k)
				head_px = HL_EYE_K * sinf(fmaxf(g.pose.pitch, 0) * 0.017453293f) * (float)g.rx[0].value;
		}
		cf_gaze_params ps = p;
		ps.strength = 5.0f; /* clamped to 2 */
		cf_gaze_init(&g);
		double t = 800.0;
		int n = 0;
		for (int i = 0; i < 90; i++, t += dt) {
			const hl_state s = {.glance = 0.04f, .pitch = 14, .up_drop = 0.02f, .lo_rise = 0.01f};
			n = update(&g, face_hl(s, 0), &ps, t, out);
		}
		float ext_y = 0;
		for (int k = 0; k < out[0].aperture_count; k++)
			ext_y = fmaxf(ext_y, fabsf(out[0].aperture[k][1] - out[0].cy));
		const float cap = fminf(0.38f * out[0].ry, 0.5f * (out[0].ry - ext_y)), big = hl_read(n, &out[0]).iris;
		const float want2 = 2.0f * (iris[0] - head_px) + head_px;
		printf("    strength 100%% -> 200%%: iris %.2f -> %.2f px (want %.2f: the head's %.2f once), lid restore %.2f -> "
		       "%.2f, lower lid %.2f -> %.2f; 500%% far down: %.2f px (cap %.2f)\n",
		       iris[0], iris[1], want2, head_px, lid_x[0], lid_x[1], low[0], low[1], big, cap);
		CHECK(fabsf(iris[1] - want2) < 0.05f * iris[0] && fabsf(lid_x[1] - lid_x[0]) < 0.05f &&
			      fabsf(low[1] - low[0]) < 0.05f,
		      "strength 200%%: iris %.2f vs %.2f, lids %.2f vs %.2f, %.2f vs %.2f", iris[1], iris[0], lid_x[1], lid_x[0],
		      low[1], low[0]);
		CHECK(big <= cap + 1e-3f && big > 0.95f * cap, "strength 500%%: %.2f px (cap %.2f)", big, cap);
		printf("ok: strength up to 200%% overshoots the iris, never the lids, within the warp's cap\n");
		(void)lift6;
	}

	/* H8. The head turned while reading, the eyes turning back in the head to the notes: half of the turn is
	 * made up for sideways, by HEAD_DAMP (0.65). The eyes' own sideways offset is not: the scan damping is
	 * off by default, and when on (scan_damp 0.95) it takes out a move along a line at once but lets go of
	 * an offset held for seconds (centred on where the eyes have been: never a sideways offset the eyes
	 * don't have). */
	{
		const float yaw = 8.0f;
		cf_gaze_init(&g);
		double t = 900.0;
		hl_out o = {0, 0, 0, 0};
		float meas = 0;
		for (int i = 0; i < 90; i++, t += dt) {
			const float yk = yaw * fminf(1.0f, (float)i / 20.0f);
			const hl_state s = {.glance = 0.02f, .side = -HL_EYE_K * sinf(1.2f * yk * 0.017453293f), .yaw = yk,
					    .up_drop = 0.01f, .lo_rise = 0.01f};
			o = hl_read(update(&g, face_hl(s, 0), &p, t, out), &out[0]);
			meas = g.pose.yaw;
		}
		const float want = -0.65f * 0.5f * HL_EYE_K * sinf(meas * 0.017453293f) * HW;
		printf("    head turned %.0f deg (reads %.1f), eyes back on the notes: sideways shift %.2f px (%.2f expected)\n",
		       yaw, meas, o.side, want);
		CHECK(fabsf(o.side - want) < 0.15f + 0.05f * fabsf(want), "head turn: sideways %.2f px, want %.2f", o.side,
		      want);

		for (int on = 0; on < 2; on++) {
			cf_gaze_params pd = p;
			pd.scan_damp = on ? 0.95f : 0.0f;
			cf_gaze_init(&g);
			t = 950.0;
			float first = 0, last = 0;
			for (int i = 0; i < 300; i++, t += dt) {
				const hl_state s = {.glance = 0.02f, .side = i < 45 ? 0.0f : 0.08f, .up_drop = 0.01f, .lo_rise = 0.01f};
				o = hl_read(update(&g, face_hl(s, 0), &pd, t, out), &out[0]);
				if (i == 48)
					first = o.side;
				last = o.side;
			}
			printf("    eyes moving 0.08 sideways and staying, scan damping %s: sideways shift %.2f px after 0.1 s "
			       "(%.2f for 95%%), %.2f px 8 s on\n",
			       on ? "on" : "off", first, -0.95f * 0.08f * HW, last);
			if (on)
				CHECK(fabsf(first + 0.95f * 0.08f * HW) < 0.3f + 0.1f * 0.08f * HW && fabsf(last) < 0.2f * fabsf(first),
				      "scan damping: %.2f px at first, %.2f px 8 s on", first, last);
			else
				CHECK(fabsf(first) < 0.05f && fabsf(last) < 0.05f, "scan damping off: %.2f / %.2f px", first, last);
		}
		printf("ok: half the head's turn is made up for sideways; the eyes' own offset never held\n");
	}

	/* H9. Real-camera noise while reading with the head down (eye landmarks 0.4 px; head landmarks 0.2 px,
	 * and a noisier 0.6 px as test_pose.c's): what the head and lid terms add to the jitter of what the eyes
	 * show (RMS frame-to-frame change of the iris and of both lids, and the iris' spread, against a twin run
	 * without head and lid calibration). At 0.2 px it is a few hundredths of a pixel. The head term carries
	 * the pose's noise, so with noisy head landmarks it grows: only reported (and kept under 0.15 px a
	 * frame), it is what a quicker head term costs. */
	for (int hn = 0; hn < 2; hn++) {
		float rms[2], rms_l[2], sd[2];
		hl_head_noise = hn ? 1.5f : 0.5f;
		for (int k = 0; k < 2; k++) {
			cf_gaze_params pn = p;
			if (!k) {
				pn.head_ref.ok = 0;
				pn.baseline_upper = pn.baseline_lower = 0;
			}
			cf_gaze_init(&g);
			rng_seed(77);
			double t = 1000.0, s2 = 0, s2l = 0, s1 = 0, sq = 0;
			hl_out prev = {0, 0, 0, 0};
			int c = 0;
			for (int i = 0; i < 300; i++, t += dt) {
				const hl_state s = {.glance = 0.02f, .pitch = 6, .up_drop = 0.01f, .lo_rise = 0.01f};
				const hl_out o = hl_read(update(&g, face_hl(s, 0.4f), &pn, t, out), &out[0]);
				if (i > 90) {
					s2 += (double)(o.iris - prev.iris) * (o.iris - prev.iris);
					s2l += (double)(o.lid - prev.lid) * (o.lid - prev.lid) + (double)(o.low - prev.low) * (o.low - prev.low);
					s1 += o.iris;
					sq += (double)o.iris * o.iris;
					c++;
				}
				prev = o;
			}
			rms[k] = (float)sqrt(s2 / c);
			rms_l[k] = (float)sqrt(s2l / c);
			sd[k] = (float)sqrt(fmax(sq / c - (s1 / c) * (s1 / c), 0.0));
		}
		printf("    noisy reading, head landmarks %.1f px: RMS step iris %.3f px (%.3f without head and lids), lids %.3f "
		       "(%.3f); iris spread %.3f px (%.3f)\n",
		       0.4f * hl_head_noise, rms[1], rms[0], rms_l[1], rms_l[0], sd[1], sd[0]);
		if (!hn)
			CHECK(rms[1] < rms[0] + 0.03f && rms_l[1] < rms_l[0] + 0.03f,
			      "head and lids add jitter: %.3f vs %.3f, %.3f vs %.3f", rms[1], rms[0], rms_l[1], rms_l[0]);
		else
			CHECK(rms[1] < 0.15f && rms_l[1] < 0.15f, "noisy head landmarks: jitter %.3f / %.3f px a frame", rms[1],
			      rms_l[1]);
	}
	hl_head_noise = 0.5f;
	printf("ok: head and lid terms add no visible jitter\n");

	/* H10. To the keyboard (0.17 eye widths past the notes, ~34 deg down in all) with the head following
	 * part of the way. Eyes first: they jump down, then turn back up in the head as the head tilts down to 28
	 * deg, ending 0.04 down in the head, inside the reading band; the gate takes that slow turn in one step,
	 * like a look back up. Head first: the head tilts to 22 deg with the eyes still in the head, then the eyes
	 * jump down. Either way the correction lets go and stays let go: under 0.5 px over the last second, and
	 * it never grows by more than the head term's slew in a frame while looking away. */
	{
		const float s6 = sinf(6.0f * 0.017453293f);
		for (int c = 0; c < 2; c++) {
			const float head_to = c ? 22.0f : 28.0f;
			cf_gaze_init(&g);
			double t = 1300.0;
			const hl_state rd = {.glance = 0.02f, .pitch = 6, .up_drop = 0.01f, .lo_rise = 0.01f};
			hl_out o = {0, 0, 0, 0};
			for (int i = 0; i < 60; i++, t += dt)
				o = hl_read(update(&g, face_hl(rd, 0), &p, t, out), &out[0]);
			const float before = o.iris;
			float rise = 0, late = 0, prev = o.iris;
			for (int i = 0; i < 75; i++, t += dt) {
				/* the head: 6 deg -> head_to over 0.5 s (eyes first: from the 3rd frame; head first: at once) */
				const int h0 = c ? 0 : 2, e0 = c ? 21 : 0;
				const float kh = fminf(fmaxf((float)(i - h0) / 15.0f, 0.0f), 1.0f);
				const float ke = fminf(fmaxf((float)(i - e0) / 2.0f, 0.0f), 1.0f);
				const float pitch = hl_lerp(6, head_to, kh * kh * (3 - 2 * kh));
				/* the eyes in the head: from the notes to the keyboard, less what the head tilted further than
				 * reading (before they jump, head first, they ride with the head) */
				const float kbd = 0.19f - HL_EYE_K * (sinf(pitch * 0.017453293f) - s6);
				const hl_state s = {hl_lerp(0.02f, kbd, ke), 0, pitch, 0, hl_lerp(0.01f, 0.04f, ke), hl_lerp(0.01f, 0.0f, ke), 0};
				o = hl_read(update(&g, face_hl(s, 0), &p, t, out), &out[0]);
				rise = fmaxf(rise, o.iris - prev);
				prev = o.iris;
				if (i >= 45)
					late = fmaxf(late, o.iris);
			}
			printf("    keyboard, %s (head to %.0f deg): %.2f px before; while away it grew by %.2f px a frame at most, "
			       "%.2f px at most over the last second\n",
			       c ? "head first" : "eyes first", head_to, before, rise, late);
			CHECK(late < 0.5f && rise < HL_HEAD_SLEW_PX + 0.1f, "keyboard %s: %.2f px left, grew %.2f px in a frame",
			      c ? "head first" : "eyes first", late, rise);
		}
		printf("ok: a look past the notes carried by the head lets go and stays let go\n");
	}

	/* H11. From reading (head 6 deg down) to someone 25 deg to the side at lens height: the eyes turn up in
	 * the head by the head's tilt, the head stays down for 0.3 s, then comes up level over 0.5 s, the eyes
	 * turning back down in it. The pupil reads only 0.65 of that turn up (as a real one may), and is noisy.
	 * At the lens the head term and the eyes' turn up then cancel only to about 0: that must not engage the
	 * sideways damping (it would pull the whole side look back towards the middle for a few frames). Over 20
	 * noise seeds the sideways shift stays under 1.5 px. */
	{
		float worst = 0;
		for (int sd = 0; sd < 20; sd++) {
			cf_gaze_init(&g);
			rng_seed(500 + 31 * sd);
			double t = 1400.0;
			const hl_state rd = {.glance = 0.02f, .pitch = 6, .up_drop = 0.01f, .lo_rise = 0.01f};
			for (int i = 0; i < 60; i++, t += dt)
				update(&g, face_hl(rd, 0.3f), &p, t, out);
			for (int i = 0; i < 45; i++, t += dt) {
				const float k = fminf(fmaxf((float)(i - 9) / 15.0f, 0.0f), 1.0f), pitch = 6.0f * (1 - k * k * (3 - 2 * k));
				const hl_state s = {-0.65f * HL_EYE_K * sinf(pitch * 0.017453293f), 0.16f, pitch, 0, 0, 0, 0};
				const hl_out o = hl_read(update(&g, face_hl(s, 0.3f), &p, t, out), &out[0]);
				worst = fmaxf(worst, fabsf(o.side));
			}
		}
		printf("    side look at lens height, head coming up level: sideways shift %.2f px at most over 20 seeds\n", worst);
		CHECK(worst < 1.5f, "side look at the lens: sideways shift %.2f px", worst);
		printf("ok: a look to the side at lens height is not damped back\n");
	}
}

/* ---- Reading scan (check S1)
 *
 * Eyes reading lines on the notes: 8 fixations 0.3 s apart sweep the iris 6 px along a 75 px eye, then a
 * return saccade. The refined points take in 86% of a sideways move (as cf_refine_pupil does on rendered
 * eyes, 60-120 px wide), with 0.5 px noise; Vision's points follow `rvs` of it (0.3 px noise), while up
 * and down they follow 0.6 (the calibrations' gain). The eyes as seen (true position plus the sideways
 * shift; the warp's own shortfall not counted) over the last 6 of 14 lines: the middle 90% of their range
 * against the true one, their regression on it, and the largest frame-to-frame move. */
#define SC_W 75.0f
static cf_face sc_face(float glance, float side_px, float rvs)
{
	cf_face f;
	memset(&f, 0, sizeof(f));
	f.found = 1;
	f.image_width = 1920;
	f.image_height = 1080;
	f.face_width = 380;
	f.has_pose = 1;
	const float gpx = glance * SC_W, h = 23.0f;
	const float cx[2] = {960 - 1.5f * SC_W, 960 + 1.5f * SC_W};
	for (int e = 0; e < 2; e++) {
		cf_pt *pts = e == 0 ? f.left_eye : f.right_eye;
		for (int i = 0; i < 16; i++) {
			const float a = (float)i / 16.0f * 6.2831853f, sn = sinf(a);
			pts[i].x = cx[e] + 0.5f * SC_W * cosf(a);
			pts[i].y = 500 + 0.5f * h * sn + (sn < -1e-3f ? 0.5f * gpx * -sn : 0.0f);
		}
		const cf_pt pupil = {cx[e] + 0.86f * side_px + gauss(0.5f), 500 + gpx + gauss(0.5f)};
		const cf_pt vp = {cx[e] + rvs * side_px + gauss(0.3f), 500 + 0.6f * gpx + gauss(0.3f)};
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
	f.has_vision_pupils = 1;
	return f;
}

static int sc_cmp(const void *a, const void *b)
{
	const float x = *(const float *)a, y = *(const float *)b;
	return x < y ? -1 : x > y;
}

static float sc_range90(const float *v, int n)
{
	float c[600];
	memcpy(c, v, sizeof(float) * (size_t)n);
	qsort(c, (size_t)n, sizeof(float), sc_cmp);
	return c[(int)(0.95 * (n - 1))] - c[(int)(0.05 * (n - 1))];
}

static void test_side_scan(void)
{
	const float rvs[3] = {0.6f, 0.15f, 0.04f};
	for (int c = 0; c < 3; c++) {
		cf_gaze_params p = {.strength = 1.0f, .max_glance = 0.15f, .baseline = 0.0f, .baseline_open = 23.0f / SC_W,
				    .has_notes = 1, .notes_offset = -0.10f, .notes_side = 0.0f,
				    .notes_open = (23.0f - 0.05f * SC_W) / SC_W, .has_vis_calib = 1, .vis_baseline = 0,
				    .vis_notes = -0.06f, .scan_damp = 0.95f};
		cf_gaze g;
		cf_gaze_init(&g);
		cf_eye out[2];
		rng_seed(77 + (unsigned)c);
		const double dt = 1.0 / 30;
		const int line = 72, start = 180, lines = 14, from = start + 8 * line;
		float tru[600], app[600], prev = 0, worst = 0;
		int n = 0;
		for (int i = 0; i < start + lines * line; i++) {
			float side = -3.0f;
			if (i >= start)
				side = -3.0f + 6.0f * (float)(((i - start) % line) / 9) / 7.0f;
			const cf_face f = sc_face(i < 90 ? 0.0f : 0.10f, side, rvs[c]);
			const int k = cf_gaze_update(&g, &f, &p, 100.0 + i * dt, out);
			const float a = side + (k ? 0.5f * (out[0].sx + out[1].sx) : 0.0f);
			if (i >= from && n < 600) {
				if (n)
					worst = fmaxf(worst, fabsf(a - prev));
				tru[n] = side;
				app[n++] = a;
			}
			prev = a;
		}
		double mt = 0, ma = 0, sta = 0, stt = 0;
		for (int i = 0; i < n; i++) {
			mt += tru[i] / n;
			ma += app[i] / n;
		}
		for (int i = 0; i < n; i++) {
			sta += (tru[i] - mt) * (app[i] - ma);
			stt += (tru[i] - mt) * (tru[i] - mt);
		}
		const float ratio = sc_range90(app, n) / sc_range90(tru, n), slope = (float)(sta / stt);
		printf("    Vision following %.0f%% sideways: scan %.2f -> %.2f px (%.2f stays, regression %.2f), largest "
		       "step %.2f px (the return is 6), Vision scaled x%.2f%s\n",
		       100 * rvs[c], sc_range90(tru, n), sc_range90(app, n), ratio, slope, worst, g.svk,
		       g.side_novis ? ", not used sideways" : "");
		CHECK(ratio > 0.15f && ratio < 0.45f && slope > 0.1f && worst < 2.5f,
		      "reading scan, Vision following %.0f%% sideways: %.2f of the scan stays (regression %.2f), step %.2f px",
		      100 * rvs[c], ratio, slope, worst);
		if (c == 1)
			CHECK(fabsf(g.svk - 0.86f / 0.15f / (0.10f / 0.06f)) < 0.2f * 0.86f / 0.15f / (0.10f / 0.06f),
			      "Vision's sideways scale %.2f", g.svk);
	}
	printf("ok: S1 reading scan damped to a quarter or a third, whatever Vision's sideways point follows, the line "
	       "returns taken within the saccade\n");

	/* S2. The sideways reading's calibrated notes spot off by 0.073 eye widths (5.4 px; as on a user's clip,
	 * the refined reading's own offset): looking from the lens down to the notes and staying there, the eyes
	 * still in the middle, the scan damping (on) adds no sideways shift: none appears at the look, and none
	 * builds up. */
	{
		cf_gaze_params p = {.strength = 1.5f, .max_glance = 0.15f, .baseline = 0.0f, .baseline_open = 23.0f / SC_W,
				    .has_notes = 1, .notes_offset = -0.10f, .notes_side = -0.0726f,
				    .notes_open = (23.0f - 0.05f * SC_W) / SC_W, .has_vis_calib = 1, .vis_baseline = 0,
				    .vis_notes = -0.06f, .scan_damp = 0.95f};
		cf_gaze g;
		cf_gaze_init(&g);
		cf_eye out[2];
		rng_seed(91);
		float worst = 0, step = 0, prev = 0;
		for (int i = 0; i < 300; i++) {
			const cf_face f = sc_face(i < 90 ? 0.0f : 0.10f, 0.0f, 0.6f);
			const int k = cf_gaze_update(&g, &f, &p, 200.0 + i / 30.0, out);
			const float sx = k ? 0.5f * (out[0].sx + out[1].sx) : 0.0f;
			worst = fmaxf(worst, fabsf(sx));
			if (i)
				step = fmaxf(step, fabsf(sx - prev));
			prev = sx;
		}
		printf("    notes spot 5.4 px off, eyes in the middle: sideways shift %.2f px at most, %.2f px a frame\n", worst,
		       step);
		CHECK(worst < 0.5f && step < 0.3f, "miscalibrated notes spot: sideways shift %.2f px, step %.2f px", worst, step);
		printf("ok: S2 the scan damping adds no sideways offset of its own\n");
	}
}

/* 42. Vision's own frame-step tail, and jumps it can't back (patterns seen on the user's camera, generic
 * sizes; v_gain both low and high, the eyes' refined points 1.2 px noisy with 5% spikes, Vision's 0.25 px
 * with excursions of about 1 px held for 1-4 frames on 10% of frames, 30 and 15 fps):
 *  (a) steady reading for 20 s: Vision's excursions alone (with the refined noise) never back a jump: no
 *      output step past the slew limit;
 *  (b) one eye's refined point 0.06 high in the frame Vision's point steps 0.014 the same way: no step past
 *      the slew limit;
 *  (c) a small real move down (0.02), the refined points overstating it by 0.025 for 1/15 s: the
 *      correction lands no more than 1 px past where it settles, and overshoots it by no more than 1.2 px in
 *      the 0.3 s after, also when a Vision excursion overstates it in the same frames;
 *  (d) back from a keyboard look with the refined points (and Vision's half as much) overshooting for two
 *      frames: back within 1 px of where it settles within 0.5 s (0.3 s unless a Vision excursion falls in
 *      the landing);
 *  (e) 6 px saccades lens <-> notes with Vision's point and both eyes jumping together: 80% of the move
 *      shows by the frame after the eyes' jump;
 *  and in all: the lid terms never fall from a visible size to near 0 in a single frame. */
typedef struct {
	float amp, p, off;
	int left;
} vex_state;

static cf_face vex_face(vex_state *vx, float glance, float dl, float dr, float dv, float lo_rise)
{
	float d[2][2], vd[2][2];
	if (vx->left > 0) {
		if (--vx->left == 0)
			vx->off = 0;
	} else if (rng_u() < vx->p) {
		vx->off = gauss(vx->amp);
		vx->left = 1 + (int)(rng_u() * 4);
	}
	for (int e = 0; e < 2; e++) {
		d[e][0] = gauss(1.2f);
		d[e][1] = gauss(1.2f) + (e == 0 ? dl : dr) * RW;
		if (rng_u() < 0.05)
			d[e][1] += (float)(SPIKE_PX + rng_u()) * (rng_u() < 0.5 ? -1.0f : 1.0f);
		vd[e][0] = gauss(0.25f);
		vd[e][1] = gauss(0.25f) + dv * RW + vx->off;
	}
	const real_gaze gz = {glance, 0, 0};
	cf_face f = face_real(gz, d, 1, vd);
	for (int e = 0; e < 2; e++) {
		cf_pt *pts = e == 0 ? f.left_eye : f.right_eye;
		for (int i = 0; i < 16; i++) {
			const float s = (pts[i].y - 500) / (0.5f * RH);
			if (s > 1e-3f)
				pts[i].y -= lo_rise * RW * s;
		}
	}
	return f;
}

static void test_vision_tail(void)
{
	const float notes = 0.025f, gains[2] = {1.84f, 3.19f};
	const double fpss[2] = {30.0, 15.0};
	float w_a = 0, w_b = 0, w_c = 0, w_co = 0, w_d = 0, w_e = 1, w_lid = 0;
	int n_vis = 0, n_vis_all = 0, n_cb = 0;
	float w_cb = 0;
	for (int pat = 0; pat < 5; pat++)
		for (int fr = 0; fr < 2; fr++)
			for (int gi = 0; gi < 2; gi++)
				for (unsigned seed = 1; seed <= 4; seed++) {
					const double fps = fpss[fr], dts = 1.0 / fps;
					r_vis = 1.0f / gains[gi];
					cf_gaze_params p = {.strength = 1.0f, .max_glance = 0.15f, .baseline = 0.0f,
							    .baseline_open = RH / RW, .has_notes = 1, .notes_offset = -notes,
							    .notes_side = 0.0f, .notes_open = (RH - R_LID * notes * RW) / RW};
					p.has_vis_calib = 1;
					p.vis_baseline = 0.0f;
					p.vis_notes = -notes * R_VIS;
					p.baseline_upper = p.baseline_lower = 0.5f * RH / RW;
					cf_gaze g;
					cf_eye out[2];
					cf_gaze_init(&g);
					rng_seed(4200 + 97 * seed + 13 * pat + 5 * gi + fr);
					vex_state vx = {1.0f, 0.1f, 0, 0};
					double t = 100.0;
					const int n0 = (int)(4.0 * fps), kb = (int)(0.6 * fps), per = (int)fps;
					const int nf = n0 + (int)((pat == 0 ? 20.0 : pat == 4 ? 4.0 : 2.0) * fps);
					const float slew = 0.3f * (float)(30.0 / fps) + 0.05f;
					float prev = 0, prevl = 0, prevb = 0, big = 0, before = 0, land = 0, peak = 0, settle = 0;
					float v_before = 0, v_over = 0;
					float c_hist[400], v_hist[400];
					int n_settle = 0;
					float vprev = 0;
					for (int i = 0; i < nf; i++, t += dts) {
						const int k = i - n0;
						float gl = notes, dl = 0, dr = 0, dv = 0;
						if (pat == 1 && k == 0) {
							dl = -0.06f;
							dv = -0.014f;
						} else if (pat == 1 && k == 1) {
							dv = -0.007f;
						} else if (pat == 2 && k >= 0) {
							gl = notes + 0.02f;
							if (k * dts < 0.066)
								dl = dr = 0.025f;
						} else if (pat == 3) {
							if (k >= 0 && k < kb)
								gl = 0.20f;
							if (k == kb || k == kb + 1) {
								dl = dr = k == kb ? 0.07f : 0.04f;
								dv = (k == kb ? 0.5f : 0.2f) * (0.20f - notes) * R_VIS;
							}
						} else if (pat == 4 && k >= 0) {
							gl = (k / per) % 2 == 0 ? 0.07f : 0.0f;
						}
						cf_face f = vex_face(&vx, gl, dl, dr, dv, gl > 0.01f ? 0.012f : 0.0f);
						const float vr = 0.5f * ((f.left_pupil_vision.y - 500) + (f.right_pupil_vision.y - 500)) / RW;
						const int n = cf_gaze_update(&g, &f, &p, t, out);
						/* (the lid terms beyond following the iris: lid_unit, low_unit) */
						const float c = n ? -out[0].sy : 0.0f, lid = n ? g.lid_unit[0] : 0.0f,
							    low = n ? g.low_unit[0] : 0.0f;
						if (k >= 0 && k < 400) {
							c_hist[k] = c;
							v_hist[k] = vr;
						}
						if (i > 2 * (int)fps) {
							if (pat == 0) {
								n_vis_all++;
								n_vis += fabsf(vr - vprev) * RW >= 1.0f;
							}
							if (pat <= 1 && (pat == 0 || (k >= -2 && k < (int)fps)))
								big = fmaxf(big, fabsf(c - prev));
							/* (a lid term falling from a visible size to near 0 in one frame) */
							const float fl = fabsf(prevl) > 0.3f && fabsf(lid) < 0.25f * fabsf(prevl) ? fabsf(prevl) : 0.0f;
							const float fb = fabsf(prevb) > 0.3f && fabsf(low) < 0.25f * fabsf(prevb) ? fabsf(prevb) : 0.0f;
							w_lid = fmaxf(w_lid, fmaxf(fl, fb));
							CHECK(fl == 0 && fb == 0, "Vision tail pattern %d (%.0f fps, gain %.2f, seed %u): a lid term "
							      "fell from %.2f/%.2f to %.2f/%.2f px in one frame", pat, fps, gains[gi], seed,
							      prevl, prevb, lid, low);
						}
						if (k == -1) {
							before = c;
							v_before = vr;
						}
						if (pat == 2 && k >= 0 && k <= 1)
							v_over = fmaxf(v_over, gains[gi] * (vr - v_before) - 0.02f);
						if (pat == 2 && k >= 0 && k <= 1)
							land = fmaxf(land, c - before);
						if (pat == 2 && k >= 0 && k * dts <= 0.3 + 1e-6)
							peak = fmaxf(peak, c - before);
						if (pat == 2 && k >= nf - n0 - (int)(0.6 * fps)) {
							settle += c - before;
							n_settle++;
						}
						vprev = vr;
						prev = c;
						prevl = lid;
						prevb = low;
					}
					const char *gs = gains[gi] < 2.0f ? "low" : "high";
					if (pat <= 1) {
						CHECK(big < slew, "Vision tail pattern %c (%.0f fps, %s gain, seed %u): a step of %.2f px",
						      "ab"[pat], fps, gs, seed, big);
						if (pat == 0)
							w_a = fmaxf(w_a, big * (float)(fps / 30.0));
						else
							w_b = fmaxf(w_b, big * (float)(fps / 30.0));
					} else if (pat == 2) {
						settle /= (float)n_settle;
						/* (counted apart: Vision's point overstating it too, by half the move or more, an
						 * excursion in the same frames) */
						const int both = v_over > 0.01f;
						CHECK(land <= settle + 1.0f && peak <= settle + 1.2f,
						      "small move overstated (%.0f fps, %s gain, seed %u%s): lands %.2f px, peaks %.2f, settles %.2f",
						      fps, gs, seed, both ? ", Vision's point overstating it too" : "", land, peak, settle);
						if (both) {
							w_cb = fmaxf(w_cb, peak - settle);
							n_cb++;
						} else {
							w_c = fmaxf(w_c, land - settle);
							w_co = fmaxf(w_co, peak - settle);
						}
					} else if (pat == 3) {
						const int nk = nf - n0;
						float fin = 0;
						const int nl = (int)(0.5 * fps);
						for (int j = nk - nl; j < nk; j++)
							fin += c_hist[j] / (float)nl;
						double back = -1;
						for (int j = nk - 1; j > kb; j--)
							if (fabsf(c_hist[j] - fin) > 1.0f) {
								back = (j + 1 - kb) * dts;
								break;
							}
						if (back < 0)
							back = 0;
						CHECK(back <= 0.5 + 1e-6, "keyboard return overshooting (%.0f fps, %s gain, seed %u): "
						      "within 1 px of where it settles only %.3f s after", fps, gs, seed, back);
						w_d = fmaxf(w_d, (float)back);
					} else {
						/* each switch after the first period: the share of the move shown by frame 1 */
						for (int s = per; s + 1 < nf - n0 && s + per <= nf - n0; s += per) {
							const float from = c_hist[s - 1], to = c_hist[s + per - 1];
							/* (only where Vision's point jumps with the eyes, by half what its gain says at least) */
							const float vexp = (s / per) % 2 == 0 ? 0.07f * R_VIS : -0.07f * R_VIS;
							if (fabsf(to - from) < 3.0f || (v_hist[s] - v_hist[s - 1]) * vexp < 0.5f * vexp * vexp)
								continue;
							const float share = (c_hist[s + 1] - from) / (to - from);
							CHECK(share >= 0.8f, "6 px saccade (%.0f fps, %s gain, seed %u, at %d): %.0f%% of %.2f px "
							      "by the frame after the jump", fps, gs, seed, s, 100.0f * share, to - from);
							w_e = fminf(w_e, share);
						}
					}
				}
	r_vis = 0.6f;
	printf("ok: Vision's step tail (%.1f%% of frames step 1 px or more): steady reading steps at most %.2f px per "
	       "1/30 s; a one-eye outlier on a Vision step %.2f; a small move overstated lands at most %.2f px and peaks "
	       "%.2f px past where it settles (%d of 16 with Vision's point overstating it too: peaks %.2f); a "
	       "keyboard return is settled %.3f s after; 6 px saccades show %.0f%% by "
	       "frame 1; no lid term drops to 0 in one frame (largest such fall %.2f px)\n",
	       100.0 * n_vis / (n_vis_all ? n_vis_all : 1), w_a, w_b, w_c, w_co, n_cb, w_cb, w_d, 100.0f * w_e, w_lid);
}

/* ---- Notes learnt from reading (checks L1-L13)
 *
 * Synthetic reading, shaped on the summary statistics of the user's clips (~0.5 m from a MacBook camera, eyes 75-88
 * px wide; units eye widths): at rest the refined points' sideways reading changes ~0.005 a frame (robust SD over
 * 1 s ~0.006), up and down a little more. Reading: line returns of 0.04-0.062 in 1 or 2 frames every 2.5-6.5 s, the
 * eyes moving back along the line gradually in between, at one height (the reading's vertical SD ~0.017 with the
 * noise). Looking around: fixations of 0.3-2.5 s anywhere in 0.12 sideways, SD 0.036 up and down. Besides (generic
 * reading, not measured): fixations every ~0.25 s, now and then a small step back, and a quarter of the returns
 * landing short with a corrective saccade after. The upper lid follows the gaze down (half of it); Vision's points
 * move 0.6 of the eyes' movement. The noise is in pixels, so smaller eyes are noisier in eye widths. */
#define RD_SLOW_TAU 0.4 /* s: the slow part of the landmark noise wanders this slowly */
/* L13: the reading's height wandering slowly on top of the fixations' scatter (SD, eye widths; 0: none). On the
 * user's clips the height read while reading drifts over seconds (the head settling, the lids), not only from one
 * fixation to the next, so frames that read low come in runs. */
static float rd_wander_sd;
#define RD_WANDER_TAU 3.0
enum { RD_READ, RD_LOOK, RD_HOLD };

typedef struct {
	/* the camera and the face */
	float w;               /* eye width, px */
	float vis_k;           /* Vision's points move this share of the eyes' movement */
	float white, slow;     /* refined points' noise per eye, sideways (px): white, and a slow wander */
	float white_y, slow_y; /* ... and up and down */
	float vwhite, vslow;   /* the same for Vision's points */
	float lid_px;          /* the outline points' noise (px) */
	/* what the eyes do */
	int mode;
	float level;           /* the glance below the lens while reading or holding (eye widths) */
	float start;           /* where lines start, sideways (eye widths) */
	float len_lo, len_hi;  /* line length range (eye widths) */
	double dur_lo, dur_hi; /* line duration range (s) */
	int dir;               /* +1: lines run towards +side (returns go -), -1: the other way */
	float scatter;         /* fixations' vertical scatter while reading (SD, eye widths) */
	int blinks;            /* a blink every 2-5 s */
	int keyboard;          /* a look at the keyboard (0.30 below the lens, 0.7-1.5 s) every 6-10 s */
	/* state */
	double t;
	float side, glance;       /* where the eyes are this frame */
	float to_side, to_glance; /* where a 2-frame saccade lands next frame */
	int sac;
	double line_t0, line_t1, next_fix, correct_t;
	float line_from, line_len, correct_to;
	int correct;
	double next_blink;
	int blink_k;
	double next_kb, kb_until;
	float kb_side, kb_glance;
	float nz[2][4]; /* slow noise: eye x (refined x, y, Vision's x, y) */
	int returns;
	float wander; /* (see rd_wander_sd) */
} rd_sim;

static float rd_u(float a, float b)
{
	return a + (b - a) * (float)rng_u();
}

static void rd_init(rd_sim *s, float w, int mode, float level, double t)
{
	memset(s, 0, sizeof(*s));
	s->w = w;
	s->vis_k = 0.6f;
	/* per eye, px (both eyes together at 80 px: sideways ~0.004 white + 0.006 slow, up and down 0.006 + 0.0045) */
	s->white = 0.45f;
	s->slow = 0.65f;
	s->white_y = 0.7f;
	s->slow_y = 0.5f;
	s->vwhite = 0.3f;
	s->vslow = 0.3f;
	s->lid_px = 0.25f;
	s->mode = mode;
	s->level = level;
	s->start = -0.025f;
	s->len_lo = 0.040f;
	s->len_hi = 0.062f;
	s->dur_lo = 2.5;
	s->dur_hi = 6.5;
	s->dir = 1;
	s->scatter = 0.014f;
	s->t = t;
	s->glance = level;
	s->side = s->start;
	s->line_t0 = t;
	s->line_t1 = t + rd_u(0.5f, 4.0f); /* (reading starts mid-line) */
	s->line_from = s->start;
	s->line_len = 0.05f;
	s->next_blink = t + rd_u(2.0f, 5.0f);
	s->next_kb = t + rd_u(6.0f, 10.0f);
}

/* Right-to-left text: lines start on the other side and run the other way. */
static void rd_rtl(rd_sim *s)
{
	s->dir = -1;
	s->start = s->side = s->line_from = 0.025f;
}

/* Moves the eyes to (side, glance): this frame, or over this and the next (a 2-frame saccade). */
static void rd_saccade(rd_sim *s, float side, float glance)
{
	if (rng_u() < 0.5) {
		s->to_side = side;
		s->to_glance = glance;
		s->side = 0.5f * (s->side + side);
		s->glance = 0.5f * (s->glance + glance);
		s->sac = 1;
	} else {
		s->side = side;
		s->glance = glance;
	}
}

static void rd_step(rd_sim *s, double dt)
{
	s->t += dt;
	if (rd_wander_sd > 0) {
		const float a = (float)(dt / RD_WANDER_TAU);
		s->wander += -a * s->wander + rd_wander_sd * sqrtf(2.0f * a) * gauss(1.0f);
	}
	if (s->sac) {
		s->side = s->to_side;
		s->glance = s->to_glance;
		s->sac = 0;
		return;
	}
	if (s->mode == RD_READ && s->keyboard) {
		if (s->kb_until > 0 && s->t >= s->kb_until) { /* back to where the line was */
			s->kb_until = 0;
			rd_saccade(s, s->kb_side, s->kb_glance);
			s->next_kb = s->t + rd_u(6.0f, 10.0f);
			return;
		}
		if (s->kb_until > 0)
			return;
		if (s->t >= s->next_kb) {
			const double look = rd_u(0.7f, 1.5f);
			s->kb_until = s->t + look;
			s->kb_side = s->side;
			s->kb_glance = s->glance;
			s->line_t1 += look;
			s->line_t0 += look;
			s->next_fix += look;
			rd_saccade(s, s->side + gauss(0.02f), 0.30f);
			return;
		}
	}
	if (s->mode == RD_READ) {
		if (s->correct && s->t >= s->correct_t) {
			s->correct = 0;
			rd_saccade(s, s->correct_to, s->glance);
			return;
		}
		if (s->t >= s->line_t1) { /* a line return */
			s->line_from = s->start + gauss(0.003f);
			s->line_len = rd_u(s->len_lo, s->len_hi);
			s->line_t0 = s->t;
			s->line_t1 = s->t + rd_u((float)s->dur_lo, (float)s->dur_hi);
			s->next_fix = s->t + rd_u(0.18f, 0.32f);
			float land = s->line_from;
			if (rng_u() < 0.25) { /* short of the line's start, then a corrective saccade */
				land += (float)s->dir * rd_u(0.006f, 0.012f);
				s->correct = 1;
				s->correct_t = s->t + rd_u(0.12f, 0.2f);
				s->correct_to = s->line_from;
			}
			s->returns++;
			rd_saccade(s, land, s->level + s->wander + gauss(s->scatter));
			return;
		}
		if (s->t >= s->next_fix) {
			float prog = (float)((s->t - s->line_t0) / (s->line_t1 - s->line_t0)) + gauss(0.02f);
			prog = prog < 0 ? 0 : (prog > 1 ? 1 : prog);
			float to = s->line_from + (float)s->dir * s->line_len * prog;
			if (rng_u() < 0.08)
				to = s->side - (float)s->dir * rd_u(0.003f, 0.008f); /* a step back */
			s->next_fix = s->t + rd_u(0.18f, 0.32f);
			s->side = to; /* (word steps: small, one frame) */
			s->glance = s->level + s->wander + gauss(s->scatter);
		}
	} else if (s->mode == RD_LOOK) {
		if (s->t >= s->next_fix) {
			s->next_fix = s->t + rd_u(0.3f, 2.5f);
			float gl = s->level + gauss(0.036f);
			gl = gl < -0.02f ? -0.02f : (gl > 0.19f ? 0.19f : gl);
			rd_saccade(s, s->start + rd_u(-0.06f, 0.06f), gl);
		}
	} else if (s->t >= s->next_fix) { /* holding one spot: microsaccades */
		s->next_fix = s->t + rd_u(0.4f, 1.0f);
		s->side = s->start + gauss(0.003f);
		s->glance = s->level + gauss(0.004f);
	}
}

/* This frame's face, with the camera's noise. */
static cf_face rd_face(rd_sim *s, double dt)
{
	static const float shut_k[7] = {0.4f, 0.9f, 1.0f, 1.0f, 0.7f, 0.35f, 0.1f};
	float shut = 0;
	if (s->blinks) {
		if (s->blink_k == 0 && s->t >= s->next_blink)
			s->blink_k = 1;
		if (s->blink_k > 0) {
			shut = shut_k[s->blink_k - 1];
			if (++s->blink_k > 7) {
				s->blink_k = 0;
				s->next_blink = s->t + rd_u(2.0f, 5.0f);
			}
		}
	}
	cf_face f;
	memset(&f, 0, sizeof(f));
	f.found = 1;
	f.image_width = 1920;
	f.image_height = 1080;
	f.face_width = 5.0f * s->w;
	f.has_pose = 1;
	f.yaw_deg = gauss(0.5f);
	const float w = s->w, h = 0.30f * w, gpx = s->glance * w, spx = s->side * w;
	const float drop = 0.5f * fmaxf(gpx, 0.0f);
	const float rho = (float)exp(-dt / RD_SLOW_TAU), kr = sqrtf(1.0f - rho * rho);
	const float cx[2] = {960 - 1.5f * w, 960 + 1.5f * w};
	for (int e = 0; e < 2; e++) {
		for (int j = 0; j < 4; j++)
			s->nz[e][j] = rho * s->nz[e][j] + kr * gauss(j == 0 ? s->slow : j == 1 ? s->slow_y : s->vslow);
		cf_pt *pts = e == 0 ? f.left_eye : f.right_eye;
		for (int i = 0; i < 16; i++) {
			const float a = (float)i / 16.0f * 6.2831853f, sn = sinf(a);
			const float y = 0.5f * h * sn + (sn < -1e-3f ? drop * -sn : 0.0f);
			pts[i].x = cx[e] + 0.5f * w * cosf(a) + gauss(s->lid_px);
			pts[i].y = 500 + y * (1.0f - shut) + gauss(s->lid_px);
		}
		cf_pt pupil = {cx[e] + spx + gauss(s->white) + s->nz[e][0], 500 + gpx + gauss(s->white_y) + s->nz[e][1]};
		cf_pt vp = {cx[e] + s->vis_k * spx + gauss(s->vwhite) + s->nz[e][2],
			    500 + s->vis_k * gpx + gauss(s->vwhite) + s->nz[e][3]};
		if (shut > 0.5f) { /* the points are unreliable with the lids nearly shut */
			pupil.x += gauss(2.0f);
			pupil.y += gauss(2.0f);
			vp.y += 1.5f * shut;
		}
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
	f.has_vision_pupils = 1;
	return f;
}

/* What a run saw: the learnt notes taken (how many, the first and the last, when, the shortest time between two),
 * and the frames the status said reading (from when). */
typedef struct {
	int pubs;
	cf_learnt_notes first, last;
	double t_first, t_last, gap;
	int reading;
	double t_reading;
} rd_log;

/* Passes learnt notes back in the params, as the plugin does (the lens step here had Vision's point). */
static void rd_feed(cf_gaze_params *p, const cf_learnt_notes *ln)
{
	p->has_notes = 1;
	p->notes_offset = ln->offset;
	p->notes_side = ln->side;
	p->notes_open = ln->open;
	p->notes_learnt = 1;
	p->notes_seconds = ln->seconds;
	p->vis_notes = ln->vis_offset;
	p->has_vis_calib = ln->has_vis;
}

/* Runs the model on the synthetic eyes for secs, taking what it learns (fed back into the params with feed). */
static void rd_run(cf_gaze *g, cf_gaze_params *p, rd_sim *s, double secs, int feed, rd_log *lg)
{
	const double dt = 1.0 / 30.0, end = s->t + secs - 1e-6;
	while (s->t < end) {
		rd_step(s, dt);
		const cf_face f = rd_face(s, dt);
		cf_eye out[2];
		cf_gaze_update(g, &f, p, s->t, out);
		cf_learnt_notes ln;
		if (cf_gaze_take_learnt_notes(g, &ln)) {
			if (lg->pubs == 0) {
				lg->first = ln;
				lg->t_first = s->t;
				lg->gap = 1e9;
			} else {
				lg->gap = fmin(lg->gap, s->t - lg->t_last);
			}
			lg->last = ln;
			lg->t_last = s->t;
			lg->pubs++;
			if (feed)
				rd_feed(p, &ln);
		}
		cf_gaze_status st;
		cf_gaze_get_status(g, &st);
		if (st.reading && !lg->reading++)
			lg->t_reading = s->t;
	}
}

/* A calibration on the synthetic eyes held at `level` below the lens. */
static cf_calib_result rd_calibrate(cf_gaze *g, const cf_gaze_params *p, enum cf_calib_target target, float w,
				    float level, double *t)
{
	rd_sim s;
	rd_init(&s, w, RD_HOLD, level, *t);
	cf_calib_result res = {0};
	cf_gaze_start_calibration(g, target);
	int got = 0;
	for (int i = 0; i < 150 && got <= 0; i++) {
		rd_step(&s, 1.0 / 30.0);
		const cf_face f = rd_face(&s, 1.0 / 30.0);
		cf_eye out[2];
		cf_gaze_update(g, &f, p, s.t, out);
		got = cf_gaze_take_calibration(g, &res);
	}
	*t = s.t;
	return res;
}

/* The params after a lens calibration on the synthetic eyes (w px wide), as the plugin saves them; with
 * notes_cal, also what a notes calibration at `level` measures there (the reference for what is learnt). */
static cf_gaze_params rd_params(float w, float level, cf_calib_result *notes_cal)
{
	cf_gaze_params p = {.strength = 1.0f, .max_glance = 0.15f};
	cf_gaze g;
	cf_gaze_init(&g);
	double t = 20.0;
	const cf_calib_result lens = rd_calibrate(&g, &p, CF_CALIB_LENS, w, 0.0f, &t);
	p.baseline = lens.offset;
	p.baseline_open = lens.openness;
	p.head_ref = lens.head;
	p.baseline_upper = lens.upper;
	p.baseline_lower = lens.lower;
	p.vis_baseline = lens.vis_offset;
	if (notes_cal)
		*notes_cal = rd_calibrate(&g, &p, CF_CALIB_NOTES, w, level, &t);
	return p;
}

/* One reading session from a fresh model: mode, eye width, level; rtl, blinks, keyboard looks. */
typedef struct {
	int pubs, reading, spell;
	double t_first, t_reading, gap;
	cf_learnt_notes first, last;
} rd_result;

static rd_result rd_session(const cf_gaze_params *p0, int mode, float w, float level, unsigned seed, double secs,
			    int rtl, int blinks, int keyboard)
{
	cf_gaze_params p = *p0;
	cf_gaze g;
	cf_gaze_init(&g);
	rng_seed(seed);
	rd_sim s;
	rd_init(&s, w, mode, level, 100.0);
	if (rtl)
		rd_rtl(&s);
	s.blinks = blinks;
	s.keyboard = keyboard;
	rd_log lg = {0};
	rd_run(&g, &p, &s, secs, 1, &lg);
	const rd_result r = {lg.pubs, lg.reading, g.learn.spell, lg.pubs ? lg.t_first - 100.0 : -1.0,
			     lg.reading ? lg.t_reading - 100.0 : -1.0, lg.pubs > 1 ? lg.gap : 0.0, lg.first, lg.last};
	return r;
}

static int rd_cmpd(const void *a, const void *b)
{
	const double x = *(const double *)a, y = *(const double *)b;
	return (x > y) - (x < y);
}

/* How far learnt notes are from where the synthetic eyes read at `level` below the lens: the offset (0), the
 * openness (1: the upper lid follows the gaze down by half), Vision's point (2: it moves 0.6 of the eyes'
 * movement). */
static float rd_err(const cf_learnt_notes *ln, float level, int what)
{
	return what == 0 ? fabsf(ln->offset + level)
			 : what == 1 ? fabsf(ln->open - (0.30f - 0.5f * level)) : fabsf(ln->vis_offset + 0.6f * level);
}

static void test_learn_notes(void)
{
	const float W = 80.0f, LEVEL = 0.10f;
	cf_calib_result cal;
	const cf_gaze_params p0 = rd_params(W, LEVEL, &cal);
	printf("    synthetic eyes %.0f px: lens calibration at %.4f (open %.3f), a notes calibration at %.2f below reads "
	       "%.4f (open %.3f, side %.4f, Vision %.4f)\n",
	       W, p0.baseline, p0.baseline_open, LEVEL, cal.offset, cal.openness, cal.side, cal.vis_offset);

	/* L1. Reading at a level below the lens: learnt within ~20 s (half of 12 sessions within 20 s, all within
	 * 45), where the eyes are (offset and Vision's point within 0.008, the openness within 0.01, sideways within
	 * 0.01 of the middle of the lines), and handed out once only while nothing changes (90 s). The status says
	 * reading before the notes are handed out. */
	{
		double t1[12];
		float e_off = 0, e_open = 0, e_vis = 0, e_side = 0;
		int one = 1, status = 1;
		for (unsigned k = 0; k < 12; k++) {
			const rd_result r = rd_session(&p0, RD_READ, W, LEVEL, 1 + k, 90.0, 0, 0, 0);
			t1[k] = r.pubs ? r.t_first : 1e9;
			one &= r.pubs == 1;
			status &= r.reading > 0 && r.t_reading <= r.t_first;
			if (!r.pubs)
				continue;
			e_off = fmaxf(e_off, rd_err(&r.first, LEVEL, 0));
			e_open = fmaxf(e_open, rd_err(&r.first, LEVEL, 1));
			e_vis = fmaxf(e_vis, r.first.has_vis ? rd_err(&r.first, LEVEL, 2) : 1.0f);
			e_side = fmaxf(e_side, fabsf(r.first.side - 0.0f)); /* (lines from -0.025, 0.04-0.062 long) */
		}
		qsort(t1, 12, sizeof(double), rd_cmpd);
		printf("    reading at %.2f: handed out after %.1f-%.1f s (median %.1f); off by at most %.4f, open %.4f, "
		       "Vision %.4f, side %.4f from the lines' middle\n",
		       LEVEL, t1[0], t1[11], 0.5 * (t1[5] + t1[6]), e_off, e_open, e_vis, e_side);
		CHECK(t1[11] <= 45.0 && t1[5] <= 20.0, "reading learnt after %.1f s (half within %.1f)", t1[11], t1[5]);
		CHECK(e_off <= 0.008f && e_open <= 0.01f && e_vis <= 0.008f && e_side <= 0.01f,
		      "learnt notes off by %.4f (open %.4f, Vision %.4f, side %.4f)", e_off, e_open, e_vis, e_side);
		CHECK(one, "learnt notes handed out more than once while nothing changed");
		CHECK(status, "the status did not say reading before the notes were learnt");
		printf("ok: L1 reading below the lens is learnt (half within %.0f s, all within %.0f), where the eyes read, "
		       "and handed out once\n",
		       t1[5], t1[11]);
	}

	/* L2. Looking around below the lens for an hour (6 sessions; fixations anywhere in 0.12 sideways, 0.036 up
	 * and down, the jumps both ways), and holding still for 10 minutes at the lens and 10 below it: nothing is
	 * learnt, nothing handed out; holding still never even reads as reading. */
	{
		int pubs = 0, has = 0, look_reading = 0, hold_reading = 0;
		for (int c = 0; c < 8; c++) {
			const int mode = c < 6 ? RD_LOOK : RD_HOLD;
			const float level = c == 6 ? 0.0f : LEVEL;
			cf_gaze_params p = p0;
			cf_gaze g;
			cf_gaze_init(&g);
			rng_seed(200 + (unsigned)c);
			rd_sim s;
			rd_init(&s, W, mode, level, 100.0);
			rd_log lg = {0};
			rd_run(&g, &p, &s, 600.0, 1, &lg);
			pubs += lg.pubs;
			has += g.learnt.has;
			if (mode == RD_LOOK)
				look_reading += lg.reading;
			else
				hold_reading += lg.reading;
		}
		printf("    looking around for 60 min, holding still 10 min at the lens and 10 below: %d handed out, "
		       "reading for %d / %d frames\n",
		       pubs, look_reading, hold_reading);
		CHECK(pubs == 0 && has == 0, "learnt from looking around or holding still: %d handed out", pubs);
		CHECK(hold_reading == 0, "holding still read as reading for %d frames", hold_reading);
		printf("ok: L2 looking around and holding still teach nothing\n");
	}

	/* L3. Keyboard looks (0.30 below the lens, the lids dropped, 0.7-1.5 s every 6-10 s) in the reading: they
	 * don't pull the estimate (as L1). */
	{
		float e_off = 0, e_open = 0;
		int all = 1;
		double worst = 0;
		for (unsigned k = 0; k < 4; k++) {
			const rd_result r = rd_session(&p0, RD_READ, W, LEVEL, 11 + k, 90.0, 0, 0, 1);
			all &= r.pubs >= 1;
			if (!r.pubs)
				continue;
			worst = fmax(worst, r.t_first);
			e_off = fmaxf(fmaxf(e_off, rd_err(&r.first, LEVEL, 0)), rd_err(&r.last, LEVEL, 0));
			e_open = fmaxf(fmaxf(e_open, rd_err(&r.first, LEVEL, 1)), rd_err(&r.last, LEVEL, 1));
		}
		printf("    reading with keyboard looks: learnt within %.1f s, off by at most %.4f (open %.4f)\n", worst, e_off,
		       e_open);
		CHECK(all && e_off <= 0.008f && e_open <= 0.01f,
		      "reading with keyboard looks: learnt %d, off by %.4f (open %.4f)", all, e_off, e_open);
		printf("ok: L3 keyboard looks don't pull the learnt notes\n");
	}

	/* L4. Right-to-left text (the lines start on the other side, the returns go the other way): learnt as well,
	 * the spell's returns going +side. */
	{
		float e_off = 0;
		int all = 1, way = 1;
		double worst = 0;
		for (unsigned k = 0; k < 4; k++) {
			const rd_result r = rd_session(&p0, RD_READ, W, LEVEL, 21 + k, 90.0, 1, 0, 0);
			all &= r.pubs >= 1;
			way &= r.spell == 1;
			if (!r.pubs)
				continue;
			worst = fmax(worst, r.t_first);
			e_off = fmaxf(e_off, rd_err(&r.first, LEVEL, 0));
		}
		printf("    right-to-left: learnt within %.1f s, off by at most %.4f\n", worst, e_off);
		CHECK(all && way && worst <= 45.0 && e_off <= 0.008f,
		      "right-to-left reading: learnt %d (returns +side: %d) within %.1f s, off by %.4f", all, way, worst,
		      e_off);
		printf("ok: L4 right-to-left reading is learnt too\n");
	}

	/* L5. The notes move 0.04 lower after 75 s of reading (fed back as the plugin does): within about a minute
	 * the learnt notes are there (offset within 0.008), handed out again, at most every LEARN_REPUB (18 s), and
	 * no more than three times in all. */
	{
		double follow = 0, gap = 1e9;
		float e_new = 0;
		int pubs_max = 0, all = 1;
		for (unsigned k = 0; k < 4; k++) {
			cf_gaze_params p = p0;
			cf_gaze g;
			cf_gaze_init(&g);
			rng_seed(31 + k);
			rd_sim s;
			rd_init(&s, W, RD_READ, LEVEL, 100.0);
			rd_log lg = {0};
			rd_run(&g, &p, &s, 75.0, 1, &lg);
			const int before = lg.pubs;
			s.level = LEVEL + 0.04f;
			const double t_move = s.t;
			double t_there = -1;
			for (int sec = 0; sec < 90; sec++) {
				rd_run(&g, &p, &s, 1.0, 1, &lg);
				if (t_there < 0 && lg.pubs > before && rd_err(&lg.last, LEVEL + 0.04f, 0) <= 0.008f)
					t_there = s.t - t_move;
			}
			all &= before == 1 && t_there > 0;
			follow = fmax(follow, t_there > 0 ? t_there : 1e9);
			gap = fmin(gap, lg.pubs > 1 ? lg.gap : 1e9);
			pubs_max = lg.pubs > pubs_max ? lg.pubs : pubs_max;
			e_new = fmaxf(e_new, rd_err(&lg.last, LEVEL + 0.04f, 0));
		}
		printf("    notes moved 0.04 lower: learnt there %.1f s after (at worst), handed out %d times at most, at "
		       "least %.1f s apart; last off by %.4f\n",
		       follow, pubs_max, gap, e_new);
		CHECK(all && follow <= 60.0 && pubs_max <= 3 && gap >= 18.0 - 1e-6 && e_new <= 0.008f,
		      "notes moved: followed %d after %.1f s, %d handed out, %.1f s apart, off by %.4f", all, follow,
		      pubs_max, gap, e_new);
		printf("ok: L5 learnt notes follow a move within a minute, handed out again at most every 18 s\n");
	}

	/* L6. Learnt notes in the params (saved from an earlier run): a fresh model starts from them at once,
	 * counted as the reading they rest on (notes_seconds, 10-60 s; 25 when unknown), and hands nothing out:
	 * not the seed itself (taken already), nor anything while the reading stays where they are (90 s). Read
	 * somewhere else (0.04 lower), it moves there and hands that out, sooner when they rest on less reading.
	 * Forgotten, they are gone (the status no longer reading) and not taken up again from the params; the
	 * reading after it is learnt afresh. */
	{
		cf_gaze_params ps = p0;
		const cf_learnt_notes saved = {.has = 1, .offset = -LEVEL, .side = 0.0f, .open = 0.30f - 0.5f * LEVEL,
					       .has_vis = 1, .vis_offset = -0.6f * LEVEL, .seconds = 40.0f, .serial = 7};
		rd_feed(&ps, &saved);
		/* weights */
		const float in_s[4] = {40.0f, 0.0f, 3.0f, 500.0f}, want_s[4] = {40.0f, 25.0f, 10.0f, 60.0f};
		int weights = 1, at_once = 1, none_new = 1;
		for (int c = 0; c < 4; c++) {
			cf_gaze_params p = ps;
			p.notes_seconds = in_s[c];
			cf_gaze g;
			cf_gaze_init(&g);
			rng_seed(41);
			rd_sim s;
			rd_init(&s, W, RD_HOLD, LEVEL, 100.0);
			rd_log lg = {0};
			rd_run(&g, &p, &s, 1.0 / 30.0, 1, &lg);
			weights &= fabsf(g.learnt.seconds - want_s[c]) < 1e-3f;
			at_once &= g.learnt.has && fabsf(g.learnt.offset - saved.offset) < 1e-6f &&
				   fabsf(g.learnt.side - saved.side) < 1e-6f && fabsf(g.learnt.open - saved.open) < 1e-6f &&
				   g.learnt.has_vis && fabsf(g.learnt.vis_offset - saved.vis_offset) < 1e-6f;
			none_new &= lg.pubs == 0 && g.learnt_taken == g.learnt.serial;
		}
		CHECK(weights && at_once && none_new, "seeded from the params: weights %d, at once %d, nothing new %d",
		      weights, at_once, none_new);
		/* reading where they are: nothing new; the weight grows to the memory's minute */
		int quiet = 0;
		float sec_after = 0;
		for (unsigned k = 0; k < 4; k++) {
			cf_gaze_params p = ps;
			cf_gaze g;
			cf_gaze_init(&g);
			rng_seed(51 + k);
			rd_sim s;
			rd_init(&s, W, RD_READ, LEVEL, 100.0);
			rd_log lg = {0};
			rd_run(&g, &p, &s, 90.0, 1, &lg);
			quiet += lg.pubs;
			sec_after = fmaxf(sec_after, g.learnt.seconds);
		}
		printf("    seeded (40 s of reading): read where they are for 90 s: %d handed out, resting on %.0f s\n", quiet,
		       sec_after);
		CHECK(quiet == 0 && sec_after >= 59.0f, "seeded, reading where they are: %d handed out (resting on %.0f s)",
		      quiet, sec_after);
		/* reading 0.04 lower: handed out once moved, sooner with a lighter seed */
		double when[2] = {0, 0};
		float e_moved = 0;
		const float sec_in[2] = {10.0f, 60.0f};
		for (int c = 0; c < 2; c++) {
			for (unsigned k = 0; k < 4; k++) {
				cf_gaze_params p = ps;
				p.notes_seconds = sec_in[c];
				cf_gaze g;
				cf_gaze_init(&g);
				rng_seed(61 + k);
				rd_sim s;
				rd_init(&s, W, RD_READ, LEVEL + 0.04f, 100.0);
				rd_log lg = {0};
				rd_run(&g, &p, &s, 90.0, 1, &lg);
				when[c] += (lg.pubs ? lg.t_first - 100.0 : 1e3) / 4;
				if (lg.pubs)
					e_moved = fmaxf(e_moved, rd_err(&lg.first, LEVEL + 0.04f, 0));
			}
		}
		printf("    seeded, reading 0.04 lower: handed out after %.1f s on average (a seed of 10 s) / %.1f s "
		       "(60 s), off by %.4f\n",
		       when[0], when[1], e_moved);
		CHECK(when[0] >= 18.0 && when[0] < when[1] && when[1] <= 75.0 && e_moved <= 0.008f,
		      "seeded, reading elsewhere: handed out after %.1f / %.1f s, off by %.4f", when[0], when[1], e_moved);
		/* forget */
		{
			cf_gaze_params p = ps;
			cf_gaze g;
			cf_gaze_init(&g);
			rng_seed(71);
			rd_sim s;
			rd_init(&s, W, RD_READ, LEVEL, 100.0);
			rd_log lg = {0};
			rd_run(&g, &p, &s, 40.0, 0, &lg);
			const int was_reading = g.st_reading, had = g.learnt.has;
			cf_gaze_forget_learnt_notes(&g);
			cf_learnt_notes ln;
			cf_gaze_status st;
			cf_gaze_get_status(&g, &st);
			const int gone = had && !g.learnt.has && !cf_gaze_take_learnt_notes(&g, &ln) && !st.reading;
			/* the params still carry them (the plugin stops passing them then): not taken up again */
			rd_run(&g, &p, &s, 2.0, 0, &lg);
			const int not_again = !g.learnt.has;
			rd_log l2 = {0};
			rd_run(&g, &p, &s, 60.0, 0, &l2);
			printf("    forgotten after 40 s of reading (the status reading: %d): gone %d, not taken up again %d; "
			       "learnt afresh %.1f s after (%d handed out, off by %.4f)\n",
			       was_reading, gone, not_again, l2.pubs ? l2.t_first - 140.0 : -1.0, l2.pubs,
			       l2.pubs ? rd_err(&l2.first, LEVEL, 0) : 0.0f);
			CHECK(gone && not_again && l2.pubs >= 1 && l2.first.serial != 0 && rd_err(&l2.first, LEVEL, 0) <= 0.008f,
			      "forgetting learnt notes: gone %d, not again %d, learnt afresh %d", gone, not_again, l2.pubs);
		}
		printf("ok: L6 learnt notes in the params are started from at once (not handed out again), refined, moved "
		       "when the reading is elsewhere, and forgotten on request\n");
	}

	/* L7. With learnt notes 0.04 below the lens, a glance of 0.13 (past their own limit, within max_glance 0.15)
	 * stays corrected; with the same notes from a calibration (notes_learnt 0) it lets go, as before. A look at
	 * the keyboard (0.30) lets go with learnt notes too. */
	{
		float fac[3];
		for (int c = 0; c < 3; c++) {
			cf_gaze_params p = p0;
			p.has_notes = 1;
			p.notes_offset = -0.04f;
			p.notes_side = 0.0f;
			p.notes_open = 0.30f - 0.5f * 0.04f;
			p.notes_learnt = c != 1;
			cf_gaze g;
			cf_gaze_init(&g);
			rng_seed(81);
			rd_sim s;
			rd_init(&s, W, RD_HOLD, c == 2 ? 0.30f : 0.13f, 100.0);
			rd_log lg = {0};
			rd_run(&g, &p, &s, 3.0, 0, &lg);
			cf_gaze_status st;
			cf_gaze_get_status(&g, &st);
			fac[c] = st.factor;
		}
		printf("    glance 0.13, notes 0.04 below: correction %.2f learnt, %.2f calibrated; keyboard (learnt) %.2f\n",
		       fac[0], fac[1], fac[2]);
		CHECK(fac[0] > 0.8f && fac[1] < 0.2f && fac[2] < 0.2f,
		      "glance limit with learnt notes: %.2f learnt, %.2f calibrated, keyboard %.2f", fac[0], fac[1], fac[2]);
		printf("ok: L7 learnt notes only widen the glance limit (to max_glance); calibrated notes keep theirs\n");
	}

	/* L8. Vision's gain from learnt notes: a change of under 10% (VIS_CAL_LEARNT_TOL) is not re-applied (its
	 * levels go on), a bigger one is (they start afresh); from calibrated notes any change is, as before. */
	{
		float ratio[3] = {0, 0, 0};
		int restarted[3] = {0, 0, 0};
		for (int c = 0; c < 2; c++) {
			cf_gaze_params p = p0;
			p.has_notes = 1;
			p.notes_offset = -LEVEL;
			p.notes_open = 0.30f - 0.5f * LEVEL;
			p.notes_learnt = c == 0;
			p.has_vis_calib = 1;
			p.vis_notes = p.vis_baseline - 0.06f;
			cf_gaze g;
			cf_gaze_init(&g);
			rng_seed(91);
			rd_sim s;
			rd_init(&s, W, RD_HOLD, LEVEL, 100.0);
			rd_log lg = {0};
			rd_run(&g, &p, &s, 2.0, 0, &lg);
			const float g0 = g.vis_gain_cal;
			const float dv[2] = {0.057f, 0.05f}; /* the gain up 5%, then 20% */
			for (int k = 0; k < 2 - c; k++) {
				p.vis_notes = p.vis_baseline - dv[k];
				rd_run(&g, &p, &s, 1.0 / 30.0, 0, &lg);
				ratio[c ? 2 : k] = g.vis_gain_cal / g0;
				restarted[c ? 2 : k] = g.rb_down.v_off_n <= 2;
			}
		}
		printf("    Vision's gain from learnt notes: x%.3f after a 5%% change (levels restarted %d), x%.3f after 20%% "
		       "(%d); from calibrated notes x%.3f after 5%% (%d)\n",
		       ratio[0], restarted[0], ratio[1], restarted[1], ratio[2], restarted[2]);
		CHECK(fabsf(ratio[0] - 1.0f) < 1e-6f && !restarted[0] && fabsf(ratio[1] - 1.2f) < 0.01f && restarted[1] &&
			      fabsf(ratio[2] - 0.06f / 0.057f) < 0.01f && restarted[2],
		      "Vision's gain with learnt notes: x%.3f (5%%), x%.3f (20%%), calibrated x%.3f", ratio[0], ratio[1],
		      ratio[2]);
		printf("ok: L8 a small change of the learnt notes' Vision gain leaves its levels be; a big one restarts "
		       "them\n");
	}

	/* L9. Blinking every 2-5 s while reading: still learnt (as L1), the blinks (held frames) not pulling the
	 * openness down. */
	{
		float e_off = 0, e_open = 0;
		int all = 1;
		double worst = 0;
		for (unsigned k = 0; k < 4; k++) {
			const rd_result r = rd_session(&p0, RD_READ, W, LEVEL, 101 + k, 90.0, 0, 1, 0);
			all &= r.pubs >= 1;
			if (!r.pubs)
				continue;
			worst = fmax(worst, r.t_first);
			e_off = fmaxf(e_off, rd_err(&r.first, LEVEL, 0));
			e_open = fmaxf(e_open, rd_err(&r.first, LEVEL, 1));
		}
		printf("    reading with blinks: learnt within %.1f s, off by at most %.4f (open %.4f)\n", worst, e_off,
		       e_open);
		CHECK(all && worst <= 45.0 && e_off <= 0.008f && e_open <= 0.01f,
		      "reading with blinks: learnt %d within %.1f s, off by %.4f (open %.4f)", all, worst, e_off, e_open);
		printf("ok: L9 reading with blinks is learnt\n");
	}

	/* L10. Eyes 45 px wide (the same pixel noise: 1.8x as much in eye widths): reading is learnt more slowly but
	 * as well (most of 8 sessions within 90 s, off by 0.012 at most), and looking around or holding still below
	 * the lens for 10 minutes teaches nothing. */
	{
		const float w = 45.0f;
		const cf_gaze_params p = rd_params(w, LEVEL, NULL);
		int learnt = 0;
		float e_off = 0;
		double t1 = 0;
		for (unsigned k = 0; k < 8; k++) {
			const rd_result r = rd_session(&p, RD_READ, w, LEVEL, 111 + k, 90.0, 0, 0, 0);
			if (!r.pubs)
				continue;
			learnt++;
			t1 = fmax(t1, r.t_first);
			e_off = fmaxf(e_off, rd_err(&r.first, LEVEL, 0));
		}
		int wrong = 0;
		for (int c = 0; c < 4; c++)
			wrong += rd_session(&p, c < 2 ? RD_LOOK : RD_HOLD, w, LEVEL, 121 + (unsigned)c, 600.0, 0, 0, 0).pubs;
		printf("    eyes %.0f px: %d of 8 learnt within 90 s (the last after %.1f s), off by at most %.4f; looking "
		       "around and holding still for 40 min: %d handed out\n",
		       w, learnt, t1, e_off, wrong);
		CHECK(learnt >= 5 && e_off <= 0.012f && wrong == 0,
		      "eyes 45 px: %d of 8 learnt, off by %.4f; %d handed out from looking around or holding still", learnt,
		      e_off, wrong);
		printf("ok: L10 smaller eyes: reading learnt more slowly, nothing learnt from looking around\n");
	}

	/* L11. Without a lens calibration the reading is learnt as absolute offsets (no below-the-lens check). */
	{
		cf_gaze_params p = {.strength = 1.0f, .max_glance = 0.15f};
		const rd_result r = rd_session(&p, RD_READ, W, LEVEL, 131, 90.0, 0, 0, 0);
		printf("    no lens calibration: learnt after %.1f s, offset %.4f\n", r.t_first, r.first.offset);
		CHECK(r.pubs >= 1 && rd_err(&r.first, LEVEL, 0) <= 0.008f, "no lens calibration: learnt %d, offset %.4f",
		      r.pubs, r.first.offset);
		printf("ok: L11 without a lens calibration the reading is still learnt\n");
	}

	/* L12. The lens look against the learnt notes: "too low" (CF_LOOK_LOW) only when it is not CF_LOOK_ABOVE_READING
	 * above them and they rest on CF_LOOK_READING_MIN seconds of reading at least. */
	{
		cf_calib_result look = {.target = CF_CALIB_LENS, .offset = -0.097f};
		cf_learnt_notes rd = {.has = 1, .offset = -0.10f, .seconds = 30.0f};
		const int low = cf_gaze_check_lens_look(&look, &rd);
		rd.seconds = 5.0f;
		const int few = cf_gaze_check_lens_look(&look, &rd);
		rd.seconds = 30.0f;
		look.offset = -0.092f; /* (0.008 above: notes just under the camera, a look as noisy as they come) */
		const int above = cf_gaze_check_lens_look(&look, &rd);
		look.offset = -0.097f;
		const int none = cf_gaze_check_lens_look(&look, NULL);
		CHECK(low == CF_LOOK_LOW && few == CF_LOOK_OK && above == CF_LOOK_OK && none == CF_LOOK_OK,
		      "lens look against the reading: %d (0.003 above), %d (5 s of reading), %d (0.008 above), %d (none)", low,
		      few, above, none);
		printf("ok: L12 a lens look is too low only against learnt notes resting on enough reading\n");
	}

	/* L13. Notes just under the camera: reading only 0.014 below the lens look, its height wandering slowly as on
	 * the user's clips (so a quarter or more of its frames, in runs, read less than CF_NOTES_MIN_GLANCE below the
	 * lens, some of them above it), is learnt with a lens calibration too, where the eyes read. (Real reading at
	 * that height was never learnt while the learner left out frames less than CF_NOTES_MIN_GLANCE below the lens:
	 * its lines fell apart.) */
	{
		const float SMALL = 0.014f;
		rd_wander_sd = 0.006f;
		int learnt = 0;
		double tmax = 0;
		float e_off = 0;
		for (unsigned k = 0; k < 8; k++) {
			const rd_result r = rd_session(&p0, RD_READ, W, SMALL, 141 + k, 90.0, 0, 0, 0);
			if (!r.pubs)
				continue;
			learnt++;
			tmax = fmax(tmax, r.t_first);
			e_off = fmaxf(e_off, rd_err(&r.first, SMALL, 0));
		}
		rd_wander_sd = 0;
		printf("    reading %.3f below the lens: learnt in %d of 8 sessions (within %.1f s), off by at most %.4f\n",
		       SMALL, learnt, tmax, e_off);
		CHECK(learnt >= 7 && e_off <= 0.008f, "reading just under the camera: learnt in %d of 8, off by %.4f", learnt,
		      e_off);
		printf("ok: L13 reading just under the camera (0.014 below the lens look) is learnt\n");
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
				     .has_notes = 1, .notes_offset = -0.08f, .notes_side = 0.0f, .scan_damp = 0.95f};
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

	/* 12b. Looking sideways out of the notes band lets go of the correction within the saccade (changed
	 * back on purpose: letting go as a fade after landing showed as the iris rolling on for half a second
	 * after the eye had landed, which the user saw; the eye's own jump hides the hand-off). The applied
	 * shift (sideways damping and vertical correction together) is gone on the frame of the saccade, and
	 * stays gone. */
	{
		cf_gaze_params pr = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
				     .notes_offset = -0.08f};
		cf_gaze_init(&g);
		const float before = settle(&g, face_px(960, 500, 4, 3.2f, 16, 0, 1.0f), &pr, &t, 60, out);
		float first = 0, after = 0;
		for (int i = 0; i < 18; i++, t += dt) {
			n = update(&g, face_px(960, 500, 20, 3.2f, 16, 0, 1.0f), &pr, t, out);
			const float sx = n ? out[0].sx : 0, sy = n ? out[0].sy : 0, m = sqrtf(sx * sx + sy * sy);
			if (i == 0)
				first = m;
			else
				after = fmaxf(after, m);
		}
		CHECK(first < 0.1f && after < 0.1f,
		      "sideways look out of the notes band: %.2f px left on the saccade's frame, up to %.2f px after", first,
		      after);
		printf("ok: sideways look out of the notes band lets go within the saccade (%.2f px, %.2f left)\n", before,
		       first);
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
		/* Pupil reads 0.13 eye widths, just past the 0.12 limit (alone that would keep ~99% of the
		 * correction), but the lids are down to 8 px: a keyboard glance the pupil under-reads.
		 * (Changed from 0.10: the lid cue now only starts past 1.05 x the limit, see 15b. Then from 0.12:
		 * the limit is now at least FAR_MARGIN past the notes, 0.12 here, so reading lower down the notes,
		 * or a notes calibration that came out small, keeps the correction.)
		 * (Settles over 2 s instead of 1 s: lids dropping fast to r 0.5 with the pupil barely moving
		 * look exactly like a half blink, so they are held for 0.5 s and then eased off, see 19/20.) */
		const float s = settle(&g, face_px(960, 500, 0, 5.2f, 16, 0, 8.0f), &pq, &t, 60, out);
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

	/* 16. Keyboard and back (changed back on purpose: the correction lets go within the saccade to the
	 * keyboard and takes over again within the saccade back, hidden by the eye's own jump; as a fade after
	 * landing, the iris was seen rolling on for half a second after the eye had landed). On the frame of
	 * each saccade the visible eye lands where it then stays: at the keyboard with nothing left of the
	 * correction, back on the notes at the lens (within 0.5 px); after that the applied shift moves under
	 * 0.1 px a frame. */
	{
		const float gl = 0.08f, gpx = gl * 40.0f, kpx = 0.3f * 40.0f;
		const cf_gaze_params pk = {.strength = 1.0f, .max_glance = 0.15f, .baseline_open = 0.4f, .has_notes = 1,
					   .notes_offset = -gl, .notes_open = 0.35f};
		cf_gaze_init(&g);
		settle(&g, face_px(960, 500, 0, gpx, 16, 0, 2.0f), &pk, &t, 45, out);
		float ps = out[0].sy, step = 0, left = 0, back = 0, first_kb = 0, first_back = 0;
		for (int i = 0; i < 60; i++, t += dt) {
			const int away = i < 30;
			const float pd = away ? kpx : gpx;
			n = update(&g, face_px(960, 500, 0, pd, 16, 0, away ? 6.0f : 2.0f), &pk, t, out);
			const float sy = n ? out[0].sy : 0, vis = pd + sy; /* visible eye below the lens */
			if (i == 0)
				first_kb = fabsf(sy);
			else if (i == 30)
				first_back = fabsf(vis);
			else
				step = fmaxf(step, fabsf(sy - ps));
			if (i == 29)
				left = fabsf(sy);
			back = fabsf(vis);
			ps = sy;
		}
		CHECK(first_kb < 0.1f && first_back < 0.5f && step < 0.1f && left < 0.1f && back < 0.5f,
		      "keyboard and back: on the saccade's frame %.2f px of correction left at the keyboard, %.2f px off the "
		      "lens back on the notes; then steps of %.2f px, %.2f px left at the keyboard, %.2f px off the lens",
		      first_kb, first_back, step, left, back);
		printf("ok: keyboard and back within the saccades (%.2f px left at the keyboard, %.2f px off the lens on the "
		       "first frame back, steps %.2f px after)\n",
		       first_kb, first_back, step);

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
		 * pupil moves far more than a lid could drag it, so the correction lets go within the saccade
		 * (changed back on purpose, see 16): gone on the first frame, and it stays gone. */
		cf_gaze_init(&g);
		const float kb0 = settle(&g, face_px(960, 500, 0, gpx, 14, 0, 0), &ph, &t, 60, out);
		float kb1 = 0, kb = 0;
		for (int i = 0; i < 15; i++, t += dt) {
			n = update(&g, face_px(960, 500, 0, 12.0f, 7.0f, 0, 0), &ph, t, out);
			const float s = n ? shift_px(&out[0]) : 0;
			if (i == 0)
				kb1 = s;
			else
				kb = fmaxf(kb, s);
		}
		CHECK(kb1 < 0.1f && kb < 0.1f,
		      "keyboard glance with deep lids: %.2f of %.2f px left on the first frame, up to %.2f px after", kb1,
		      kb0, kb);
		printf("ok: keyboard glance with the lids at r 0.44 is not held (%.2f -> %.2f px on the first frame)\n", kb0,
		       kb1);
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

	/* 13c. Reading the notes: glances 0..15 degrees down on the realistic eye, 86 and 75 px wide.
	 * The iris moves down 0.41 sin(angle) eye widths, the upper lid follows 60-80% of it and covers
	 * more and more of the iris, the lower lid stays nearly put; clumped lashes, camera noise and
	 * JPEG-like compression; also a low-contrast eye (dull sclera, light textured iris). On real
	 * clips the refined point read reading glances far short (lens to notes 1-1.5 px instead of 3-5):
	 * with the opening narrowing it leaned on the darkness centroid, which stays near the opening's
	 * middle, and an iris body lighter than the dark threshold shrank the edge widths row by row. The
	 * refined move (means over 16 noise seeds, 64 for the lens frame, 3x that for the low-contrast eye,
	 * which is noisier) must be 85-115% of the true one
	 * at every angle, or within 0.1 px of it at 1 degree (a 0.6 px move, where 15% is finer than the
	 * frame's own sub-pixel error); rise with every degree; and stay steady under noise. */
	{
		enum { MW = 200, MH = 120 };
		static uint8_t img[MW * MH];
		static const struct {
			float wpx, fup, sclera, iris, tex;
		} cs[] = {{86, 0.6f, 0, 0, 0},       {86, 0.7f, 0, 0, 0},       {86, 0.8f, 0, 0, 0},
			  {75, 0.6f, 0, 0, 0},       {75, 0.7f, 0, 0, 0},       {75, 0.8f, 0, 0, 0},
			  {86, 0.7f, 150, 100, 12}, {75, 0.7f, 150, 100, 12}};
		for (size_t c = 0; c < sizeof(cs) / sizeof(cs[0]); c++) {
			const float wpx = cs[c].wpx, k = wpx / 86, fup = cs[c].fup;
			const int W = (int)(200 * k), H = (int)(120 * k);
			const float cx = 0.5f * (float)W + 0.3f, cy = 0.5f * (float)H + 0.4f;
			const float iy0 = real_eye_at(k, cx, cy).iy;
			cf_pt eye[6];
			float y[16], gain[16], gmin = 1e9f, gmax = -1e9f, smin = 1e9f, worst = 0;
			for (int a = 0; a <= 15; a++) {
				const float dy = 0.41f * sinf((float)a * 3.14159265f / 180) * wpx;
				/* every gain is measured from the lens frame; the low-contrast eye is noisier */
				const int seeds = (a ? 16 : 64) * (cs[c].sclera > 0 ? 3 : 1);
				double ym = 0;
				for (int sd = 0; sd < seeds; sd++) {
					const real_eye s = reading_eye(k, cx, cy, dy, fup, cs[c].sclera, cs[c].iris, cs[c].tex,
								       300 + (uint32_t)sd);
					draw_real_eye(img, W, H, &s, eye);
					jpeg_like(img, W, H, 75);
					/* Vision's point under-reads the glance: 30% of it */
					const cf_pt g = {s.ix - 1 * k, iy0 + 0.3f * dy - 1 * k};
					ym += cf_refine_pupil(img, W, W, H, eye, 6, g).y / seeds;
				}
				y[a] = (float)ym;
				gain[a] = a ? (y[a] - y[0]) / dy : 1;
				if (a) {
					gmin = fminf(gmin, gain[a]);
					gmax = fmaxf(gmax, gain[a]);
					smin = fminf(smin, y[a] - y[a - 1]);
					worst = fmaxf(worst, fabsf(y[a] - y[0] - dy) / fmaxf(0.15f * dy, 0.1f));
				}
			}
			/* Noise: one frame at 10 degrees under 16 noise seeds. */
			double n1 = 0, n2 = 0;
			for (int sd = 0; sd < 16; sd++) {
				const float dy = 0.41f * sinf(10 * 3.14159265f / 180) * wpx;
				const real_eye s =
				    reading_eye(k, cx, cy, dy, fup, cs[c].sclera, cs[c].iris, cs[c].tex, 500 + (uint32_t)sd);
				draw_real_eye(img, W, H, &s, eye);
				jpeg_like(img, W, H, 75);
				const float v = cf_refine_pupil(img, W, W, H, eye, 6, (cf_pt){s.ix - 1 * k, iy0 + 0.3f * dy - 1 * k}).y;
				n1 += v;
				n2 += (double)v * v;
			}
			const double nsd = sqrt(fmax(n2 / 16 - (n1 / 16) * (n1 / 16), 0));
			char name[64], gl[160];
			snprintf(name, sizeof(name), "%.0f px eye, upper lid %.0f%%%s", wpx, 100 * fup,
				 cs[c].sclera > 0 ? ", low contrast" : "");
			int n = 0;
			for (int a = 1; a <= 15 && n < (int)sizeof(gl) - 6; a++)
				n += snprintf(gl + n, sizeof(gl) - (size_t)n, " %.2f", gain[a]);
			CHECK(worst <= 1, "reading sweep, %s: gain 1..15 deg%s", name, gl);
			CHECK(smin > 0, "reading sweep, %s: refined height steps down by %.3f px between degrees", name, -smin);
			CHECK(nsd < (cs[c].sclera > 0 ? 0.18 : 0.1),
			      "reading sweep, %s: refined height sd %.3f px over 16 noise seeds at 10 deg", name, nsd);
			printf("ok: reading sweep, %s: gain 1..15 deg%s (%.2f..%.2f), lens to 15 deg %.2f of %.2f px, steps "
			       ">= %.2f px a degree, noise sd %.3f px\n",
			       name, gl, gmin, gmax, y[15] - y[0], 0.41f * sinf(15 * 3.14159265f / 180) * wpx, smin, nsd);
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
	 * frame of the new gaze and never overshoots (5%). Every single glance lands within 2 frames: within
	 * 25% of the target (2.2 px, twice the refined noise: a glance that landed still carries its frames'
	 * noise) on the first or second frame, with or without Vision's points, also with the old bimodal
	 * refiner. Reaching 90% on a single glance is up to that noise: with Vision's points 95% of them do on
	 * the first or second frame (every one on 1 px noise), without them 90%. 30 and 15 fps. */
	{
		const char *names[5] = {"1 px noise", "1 px noise", "realistic noise", "realistic noise", "bimodal refiner"};
		for (int fr = 0; fr < 2; fr++)
			for (int c = 0; c < 5; c++) {
				const double fps = fr ? 15.0 : 30.0;
				const int vis = c != 1 && c != 3, flips = c == 4, rnoise = c == 2 || c == 3;
				const real_noise rn = {fps, vis, flips ? 0.4f : 0.0f, 0, rnoise ? 1.2f : 1.0f, vis ? 0.3f : 0.0f,
						       rnoise ? 0.05f : 0.0f};
				int dn, up, late, near;
				float over, quick;
				real_glances(&rn, 11 + c, &dn, &up, &over, &quick, &late, &near);
				const int mlate = 1;
				const float qmin = vis ? 0.95f : 0.9f;
				const int slate = c == 0 ? 1 : 12, snear = 1;
				CHECK(dn <= mlate && up <= mlate && over < 0.05f * 0.10f * RW && quick >= qmin && late <= slate &&
					      near <= snear,
				      "glances (%s, %s, %.0f fps): mean reaches on frame %d going down, %d coming back, mean "
				      "overshoot %.2f px; %.0f%% of single glances by the second frame, slowest frame %d (within "
				      "25%%: %d)",
				      vis ? "Vision" : "no Vision", names[c], fps, dn, up, over, 100.0f * quick, late, near);
				printf("ok: glances (%s, %s, %.0f fps): mean reaches 90%% on frame %d down / %d back, overshoot "
				       "%.2f px; %.0f%% of single glances by the second frame, slowest frame %d (within 25%%: "
				       "%d)\n",
				       vis ? "Vision" : "no Vision", names[c], fps, dn, up, over, 100.0f * quick, late, near);
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
	 * fully. With the eyes (changed on purpose: it used to be a fade after the eye had landed, seen as
	 * the iris rolling on for half a second) that happens within the saccade, hidden by the eye's own
	 * jump: the shift is where it ends up from the first or second frame of the new gaze (with the old
	 * refiner, whose flips can hide the move for a frame, the third), and moves no more than 1.25 px a frame
	 * after that at 30 fps (2.5 px at 15: the same speed). Turning the head is a gradual fade: never more
	 * than that a frame, only ever shrinking while letting go (no wobble) and only growing while taking over
	 * (up to the reading's own jitter, 0.3 px). */
	{
		const real_gaze aways[3] = {{0.30f, 0.0f, 0.0f}, {0.10f, 0.5f, 0.0f}, {0.10f, 0.0f, 40.0f}};
		const char *names[3] = {"keyboard", "sideways", "head turn"};
		for (int fr = 0; fr < 2; fr++)
			for (int nz = 0; nz < 2; nz++)
				for (int a = 0; a < 3; a++) {
					const double fps = fr ? 15.0 : 30.0;
					const real_noise rn = {fps, 1, nz ? 0.4f : 0.0f, 0, nz ? 1.0f : 1.2f, 0.3f, nz ? 0.0f : 0.05f};
					const float bound = fr ? 2.5f : 1.25f;
					const int head = a == 2, land_max = nz ? 2 : 1;
					look_away_result w = {{0, 0}, 0, 0, 0, 0, 0, 1e9f};
					for (unsigned seed = 1; seed <= 4; seed++) {
						look_away_result r;
						real_look_away(&rn, aways[a], 40 + seed, &r);
						for (int k = 0; k < 2; k++)
							w.land[k] = r.land[k] > w.land[k] ? r.land[k] : w.land[k];
						w.step = fmaxf(w.step, r.step);
						w.step_after = fmaxf(w.step_after, r.step_after);
						w.against_out = fmaxf(w.against_out, r.against_out);
						w.against_in = fmaxf(w.against_in, r.against_in);
						w.left = fmaxf(w.left, r.left);
						w.back = fminf(w.back, r.back);
					}
					const int ok = head ? w.step <= bound && w.against_out < 0.05f && w.against_in < 0.3f
							    : w.land[0] <= land_max && w.land[1] <= land_max && w.step_after <= bound;
					CHECK(ok && w.left < 0.2f && w.back > 0.9f * 0.10f * RW,
					      "look away (%s, %s, %.0f fps): handed over on frame %d away / %d back, largest step %.2f px "
					      "(%.2f after), against the hand-off %.2f px out / %.2f px in, %.2f px left away, %.2f px "
					      "corrected back",
					      names[a], nz ? "bimodal refiner" : "realistic noise", fps, w.land[0], w.land[1], w.step,
					      w.step_after, w.against_out, w.against_in, w.left, w.back);
					if (head)
						printf("ok: look away and back (%s, %s, %.0f fps): a fade, largest step %.2f px, against it "
						       "%.2f out / %.2f in, %.2f px left away, %.2f of %.2f px corrected back\n",
						       names[a], nz ? "bimodal refiner" : "realistic noise", fps, w.step, w.against_out,
						       w.against_in, w.left, w.back, 0.10f * RW);
					else
						printf("ok: look away and back (%s, %s, %.0f fps): handed over on frame %d away / %d back, "
						       "then steps of %.2f px at most, %.2f px left away, %.2f of %.2f px corrected back\n",
						       names[a], nz ? "bimodal refiner" : "realistic noise", fps, w.land[0], w.land[1],
						       w.step_after, w.left, w.back, 0.10f * RW);
				}
	}

	/* 32. A notes calibration that came out small (0.013 eye widths, 1.1 px: notes just under the lens, or the
	 * lens step taken looking a little low) still corrects reading lower down (0.05 and 0.065 below the
	 * lens): the far fade starts at least FAR_MARGIN (and 5x the landmark noise) past the notes, so neither
	 * reading there nor the noise switches the correction off. At least 90% of the glance is corrected,
	 * without shaking (under 0.05 px RMS frame to frame); a keyboard glance (0.3) still lets go. Realistic
	 * noise with Vision, 30 and 15 fps. */
	{
		cf_gaze_params ps = real_params();
		ps.notes_offset = -0.013f;
		ps.notes_open = (RH - R_LID * 0.013f * RW) / RW;
		const float depths[2] = {0.05f, 0.065f};
		for (int fr = 0; fr < 2; fr++)
			for (int k = 0; k < 2; k++) {
				const double fps = fr ? 15.0 : 30.0, dt1 = 1.0 / fps;
				const real_noise rn = {fps, 1, 0.0f, 0, 1.2f, 0.3f, 0.05f};
				const real_gaze lens = {0, 0, 0}, read = {depths[k], 0, 0}, kb = {0.30f, 0, 0};
				float lo = 1e9f, worst_rms = 0, left = 0;
				for (unsigned seed = 1; seed <= 3; seed++) {
					cf_gaze_init(&g);
					rng_seed(70 + seed + 10 * k);
					double tt = 100.0, s2 = 0, sum = 0;
					float prev = 0;
					int c = 0;
					const int n_lens = (int)(2 * fps), n_read = (int)(3 * fps), n_kb = (int)(1.5 * fps);
					for (int i = 0; i < n_lens + n_read + n_kb; i++, tt += dt1) {
						const int ph = i < n_lens ? 0 : (i < n_lens + n_read ? 1 : 2);
						cf_face f = noisy_real(&rn, ph == 0 ? lens : (ph == 1 ? read : kb));
						const int nn = cf_gaze_update(&g, &f, &ps, tt, out);
						const float sy = nn ? out[0].sy : 0.0f;
						if (ph == 1 && i >= n_lens + n_read / 2) {
							s2 += (double)(sy - prev) * (sy - prev);
							sum += -sy;
							c++;
						}
						if (ph == 2)
							left = nn ? shift_px(&out[0]) : 0.0f;
						prev = sy;
					}
					lo = fminf(lo, (float)(sum / c));
					worst_rms = fmaxf(worst_rms, (float)sqrt(s2 / c));
				}
				const float want = depths[k] * RW;
				CHECK(lo > 0.9f * want && worst_rms < 0.05f && left < 0.2f,
				      "small notes calibration (0.013), reading %.3f below the lens (%.0f fps): corrected %.2f of %.2f px, "
				      "%.3f px RMS; keyboard leaves %.2f px",
				      depths[k], fps, lo, want, worst_rms, left);
				printf("ok: small notes calibration (0.013 eye widths), reading %.3f below the lens (%.0f fps): %.0f%% "
				       "corrected, %.3f px RMS; the keyboard still lets go (%.2f px left)\n",
				       depths[k], fps, 100.0f * lo / want, worst_rms, left);
			}
	}

	/* 33. Calibrating with real noise: the result is the median of the steady frames. Frames whose lids are
	 * well off their usual height (a squint too slight to be held as a half blink, dragging the pupil
	 * points 2.6 px down) are left out, on one frame in five; the result then stays within 0.004 eye widths
	 * (0.3 px) of the true offset and openness, and is steady (no hint). The spread reported is about the
	 * landmark noise. A gaze that moves on halfway through (0.04 eye widths, 3.4 px), or lids that keep
	 * moving on half the frames, is flagged unsteady (the plugin asks to redo the step). */
	{
		for (int sc = 0; sc < 3; sc++) {
			cf_gaze_params pc = real_params();
			pc.has_notes = 0;
			pc.baseline_open = 0;
			const real_noise rn = {30.0, 1, 0.0f, 0, 1.2f, 0.3f, 0.05f};
			cf_calib_result res;
			memset(&res, 0, sizeof(res));
			float worst_off = 0, worst_open = 0, spread = 0;
			int hints_or = 0, hints_and = ~0;
			for (unsigned seed = 1; seed <= 6; seed++) {
				cf_gaze_init(&g);
				rng_seed(90 + seed + 10 * sc);
				double tt = 100.0;
				const real_gaze lens = {0, 0, 0};
				for (int i = 0; i < 30; i++, tt += dt) {
					cf_face f = noisy_real(&rn, lens);
					cf_gaze_update(&g, &f, &pc, tt, out);
				}
				cf_gaze_start_calibration(&g, CF_CALIB_LENS);
				int got = 0, i = 0;
				for (; i < 200 && !got; i++, tt += dt) {
					/* (sample k: the calibration skips its first CF_CALIB_WARMUP_S) */
					const int k = i - (int)(CF_CALIB_WARMUP_S * 30.0) - 1;
					const real_gaze gz = {sc == 1 && k >= CF_CALIB_SAMPLES / 2 ? 0.04f : 0.0f, 0, 0};
					cf_face f = noisy_real(&rn, gz);
					const int squint = sc == 2 ? (i % 2 == 0) : (i % 5 == 0);
					if (squint) /* the upper lid 2.6 px lower, dragging both pupil points down as far */
						for (int e = 0; e < 2; e++) {
							cf_pt *pts = e == 0 ? f.left_eye : f.right_eye;
							for (int q = 0; q < 16; q++) {
								const float s = sinf((float)q / 16.0f * 6.2831853f);
								if (s < -1e-3f)
									pts[q].y += 2.6f * -s;
							}
							(e == 0 ? &f.left_pupil : &f.right_pupil)->y += 2.6f;
							(e == 0 ? &f.left_pupil_vision : &f.right_pupil_vision)->y += 1.3f;
						}
					cf_gaze_update(&g, &f, &pc, tt, out);
					got = cf_gaze_take_calibration(&g, &res);
				}
				CHECK(got == 1, "calibration (case %d) didn't finish", sc);
				const int h = cf_gaze_calib_hints(&res, 0.0f);
				hints_or |= h;
				hints_and &= h;
				const float true_off = sc == 1 ? -0.02f : 0.0f; /* (half the samples at each) */
				worst_off = fmaxf(worst_off, fabsf(res.offset - true_off));
				worst_open = fmaxf(worst_open, fabsf(res.openness - RH / RW));
				spread = fmaxf(spread, res.spread);
			}
			if (sc == 0) {
				CHECK(worst_off < 0.004f && worst_open < 0.004f && hints_or == 0 && spread > 0.004f && spread < 0.02f,
				      "calibration with a squint on one frame in five: offset off by %.4f, openness by %.4f, spread "
				      "%.4f, hints %d",
				      worst_off, worst_open, spread, hints_or);
				printf("ok: calibration leaves out frames with the lids off (one in five): offset within %.4f, "
				       "openness within %.4f eye widths, spread %.4f (the noise), no hint\n",
				       worst_off, worst_open, spread);
			} else {
				CHECK(hints_and & CF_CALIB_HINT_UNSTEADY, "calibration with %s should be flagged unsteady (hints %d)",
				      sc == 1 ? "the gaze moving on" : "the lids moving on half the frames", hints_and);
				printf("ok: calibration with %s is flagged unsteady\n",
				       sc == 1 ? "the gaze moving on halfway (0.04 eye widths)" : "the lids moving on half the frames");
			}
		}
	}

	/* 34. A small notes calibration is taken (notes 2-3 cm under the lens at arm's length are a real glance of
	 * about 0.02 eye widths), with a hint for the panel; only one under CF_NOTES_MIN_GLANCE (or above the
	 * lens) is refused. Through the calibration itself: lens, then notes 0.015 below it, realistic noise. */
	{
		cf_calib_result r = {.target = CF_CALIB_NOTES, .offset = -0.013f, .spread = 0.01f, .drift = 0.002f};
		CHECK(cf_gaze_notes_valid(0.0f, -0.013f) && (cf_gaze_calib_hints(&r, 0.0f) & CF_CALIB_HINT_SMALL_GLANCE),
		      "a 0.013 notes glance should be taken, with the small-glance hint");
		r.offset = -0.06f;
		CHECK(cf_gaze_calib_hints(&r, 0.0f) == 0, "a 0.06 notes glance should need no hint (%d)",
		      cf_gaze_calib_hints(&r, 0.0f));
		CHECK(!cf_gaze_notes_valid(0.0f, -0.007f) && !cf_gaze_notes_valid(0.0f, 0.02f),
		      "a notes glance of 0.007 (or above the lens) should be refused");
		cf_gaze_params pc = real_params();
		pc.has_notes = 0;
		const real_noise rn = {30.0, 1, 0.0f, 0, 1.2f, 0.3f, 0.05f};
		cf_calib_result lens, notes;
		memset(&lens, 0, sizeof(lens));
		memset(&notes, 0, sizeof(notes));
		cf_gaze_init(&g);
		rng_seed(123);
		double tt = 100.0;
		for (int step = 0; step < 2; step++) {
			cf_gaze_start_calibration(&g, step ? CF_CALIB_NOTES : CF_CALIB_LENS);
			int got = 0;
			const real_gaze gz = {step ? 0.015f : 0.0f, 0, 0};
			for (int i = 0; i < 200 && !got; i++, tt += dt) {
				cf_face f = noisy_real(&rn, gz);
				cf_gaze_update(&g, &f, &pc, tt, out);
				got = cf_gaze_take_calibration(&g, step ? &notes : &lens);
			}
			if (!step) {
				pc.baseline = lens.offset;
				pc.baseline_open = lens.openness;
			}
		}
		const int h = cf_gaze_calib_hints(&notes, lens.offset);
		const float gl = lens.offset - notes.offset;
		CHECK(cf_gaze_notes_valid(lens.offset, notes.offset) && h == CF_CALIB_HINT_SMALL_GLANCE &&
			      fabsf(gl - 0.015f) < 0.01f, /* (a single calibration: 0.003 typical error in this noise) */
		      "notes 0.015 under the lens: measured %.4f, valid %d, hints %d", gl,
		      cf_gaze_notes_valid(lens.offset, notes.offset), h);
		printf("ok: a small notes calibration (0.015, measured %.4f) is taken with a hint; 0.007 is refused\n", gl);
	}

	/* 35. A new lens calibration (the baseline moving 0.02 eye widths, 1.7 px) while reading: everything the
	 * glance channel keeps moves along, so the correction takes the new calibration on the next frame and
	 * then holds steady (it used to read as an unbacked jump of the gaze, voted on and glided in). Realistic
	 * noise with Vision, 30 fps. */
	{
		const real_noise rn = {30.0, 1, 0.0f, 0, 1.2f, 0.3f, 0.05f};
		cf_gaze_params pb = real_params();
		pb.baseline = 0.02f; /* (the notes then read 0.12 below it) */
		cf_gaze_init(&g);
		rng_seed(321);
		double tt = 100.0;
		float before = 0, worst = 0, first = 0, prev = 0;
		for (int i = 0; i < 150; i++, tt += dt) {
			if (i == 90)
				pb.baseline = 0.0f;
			cf_face f = noisy_real(&rn, at_notes());
			const int nn = cf_gaze_update(&g, &f, &pb, tt, out);
			const float c = nn ? -out[0].sy : 0.0f;
			if (i == 89)
				before = c;
			if (i == 90)
				first = c;
			if (i > 90)
				worst = fmaxf(worst, fabsf(c - prev));
			prev = c;
		}
		CHECK(fabsf(before - 0.12f * RW) < 0.5f && fabsf(first - 0.10f * RW) < 0.5f && worst < 0.15f,
		      "new lens calibration while reading: %.2f px before, %.2f px on the next frame (expected %.2f), then "
		      "steps up to %.2f px",
		      before, first, 0.10f * RW, worst);
		printf("ok: a new lens calibration is taken on the next frame (%.2f -> %.2f px), then steps of %.2f px at most\n",
		       before, first, worst);
	}

	/* 36. Vision's points move 0.3, 0.6 or 1.0 of the glance (eyes and lighting differ), with their gain
	 * known from the two calibrations (as the plugin keeps them) or learnt from the glances themselves:
	 * every single glance still lands within 25% on the first or second frame, the mean reaches 90% by
	 * the second frame without overshooting, and reading steadily at the notes stays as calm as check 28
	 * asks. Realistic noise, 30 and 15 fps. */
	{
		const float ratios[3] = {0.3f, 0.6f, 1.0f};
		float worst_rms = 0, worst_big = 0;
		int worst_near = 0, worst_mean = 0;
		for (int cal = 0; cal < 2; cal++)
			for (int k = 0; k < 3; k++)
				for (int fr = 0; fr < 2; fr++) {
					const double fps = fr ? 15.0 : 30.0;
					r_vis = ratios[k];
					real_vis_cal = cal;
					const real_noise rn = {fps, 1, 0.0f, 0, 1.2f, 0.3f, 0.05f};
					int dn, up, late, near;
					float over, quick, rms, rms_x, big, wob, mean;
					real_glances(&rn, 41 + k, &dn, &up, &over, &quick, &late, &near);
					real_steady(&rn, 51 + k, &rms, &rms_x, &big, &wob, &mean);
					CHECK(dn <= 1 && up <= 1 && over < 0.05f * 0.10f * RW && near <= 1 && rms < 0.06f && big < 0.25f &&
						      fabsf(mean - 0.10f * RW) < 0.6f,
					      "Vision ratio %.1f (%s, %.0f fps): mean glance reaches 90%% on frame %d / %d back, "
					      "overshoot %.2f px, single glances within 25%% by frame %d; reading: %.3f px RMS, "
					      "largest step %.2f, mean %.2f px",
					      ratios[k], cal ? "calibrated" : "learnt", fps, dn, up, over, near, rms, big, mean);
					worst_rms = fmaxf(worst_rms, rms);
					worst_big = fmaxf(worst_big, big);
					worst_near = near > worst_near ? near : worst_near;
					worst_mean = dn > worst_mean ? dn : (up > worst_mean ? up : worst_mean);
				}
		r_vis = 0.6f;
		real_vis_cal = 0;
		printf("ok: Vision ratio 0.3/0.6/1.0, calibrated or learnt: glances land by frame %d (mean) / %d (each, "
		       "within 25%%); reading %.3f px RMS, largest step %.2f px\n",
		       worst_mean, worst_near, worst_rms, worst_big);
	}

	/* 37. Vision's points alone move while the eyes and the refined points stay (a refit, a lid or a smile
	 * dragging them, head pitch): a step of 1.5 or 3 px, or a drift of 0.5 or 1 px/s for 4 s. The step is
	 * not believed (the refined readings didn't follow it), the drift is taken into the offset between the
	 * two as it shows: the correction stays within 0.8 px of where it was for a step and 1.6 / 2.5 px for
	 * the drifts (it used to follow a 3 px step fully, and a 1 px/s drift by 4 px), and never steps faster
	 * than the slew limit. Realistic noise at the notes, 30 and 15 fps. */
	{
		const float amount[4] = {1.5f, 3.0f, 0.5f, 1.0f}, limit[4] = {0.8f, 0.8f, 1.6f, 2.5f};
		float worst[4] = {0, 0, 0, 0}, worst_step = 0;
		for (int fr = 0; fr < 2; fr++)
			for (int k = 0; k < 4; k++)
				for (unsigned seed = 0; seed < 4; seed++) {
					const double fps = fr ? 15.0 : 30.0, dts = 1.0 / fps;
					const real_noise rn = {fps, 1, 0.0f, 0, 1.2f, 0.3f, 0.05f};
					const cf_gaze_params pr = real_params();
					cf_gaze_init(&g);
					rng_seed(61 + 7 * seed + k);
					double tt = 100.0, ref = 0;
					float prev = 0, dev = 0, stp = 0;
					const int n0 = (int)(4.0 * fps), n1 = n0 + (int)(8.0 * fps);
					for (int i = 0; i < n1; i++, tt += dts) {
						const double ts = (i - n0) * dts;
						rx.vdrag = i < n0 ? 0.0f
							 : (k < 2 ? amount[k] : amount[k] * (float)fmin(ts, 4.0)) / RW;
						cf_face f = noisy_real(&rn, at_notes());
						const int nn = cf_gaze_update(&g, &f, &pr, tt, out);
						const float c = nn ? -out[0].sy : 0.0f;
						if (i >= n0 - (int)fps && i < n0)
							ref += c / (int)fps;
						if (i >= n0) {
							dev = fmaxf(dev, fabsf(c - (float)ref));
							stp = fmaxf(stp, fabsf(c - prev));
						}
						prev = c;
					}
					rx = rx_none;
					const float slew = 0.3f * (float)(30.0 / fps) + 0.1f;
					CHECK(dev < limit[k] && stp < slew,
					      "Vision alone %s %.1f px%s (%.0f fps, seed %u): the correction moved %.2f px (largest step "
					      "%.2f)",
					      k < 2 ? "stepping" : "drifting", amount[k], k < 2 ? "" : "/s", fps, seed, dev, stp);
					worst[k] = fmaxf(worst[k], dev);
					worst_step = fmaxf(worst_step, stp * (float)(fps / 30.0));
				}
		printf("ok: Vision alone stepping 1.5 / 3 px moves the correction %.2f / %.2f px, drifting 0.5 / 1 px/s "
		       "%.2f / %.2f px, no step past the slew limit (%.2f px per 1/30 s)\n",
		       worst[0], worst[1], worst[2], worst[3], worst_step);
	}

	/* 38. The face lost for a moment (1-4 frames at 30 fps, 1-2 at 15: under LOST_HOLD) exactly while the
	 * gaze moves between the lens and the notes: when it is back, the correction lands on the new gaze
	 * within 2 frames (within 25%), both ways, as a glance would, and once that landing has settled (3
	 * frames) it moves no faster than the slew limit (it used to glide for a second or push the wrong way).
	 * Lost for longer (0.3 s), the correction fades out and comes back in within 0.5 s of the face, easing
	 * (no step past 1 px per 1/30 s). Realistic noise with Vision, 30 and 15 fps. */
	{
		int worst_land = 0;
		float worst_after = 0, worst_long = 0;
		int worst_back = 0;
		for (int fr = 0; fr < 2; fr++)
			for (int lost = 1; lost <= 5; lost++)
				for (int dir = 0; dir < 2; dir++)
					for (unsigned seed = 0; seed < 6; seed++) {
						const double fps = fr ? 15.0 : 30.0, dts = 1.0 / fps;
						const int n_lost = lost == 5 ? (int)(0.3 * fps) : lost;
						if (lost < 5 && n_lost * dts >= 0.19)
							continue;
						const real_noise rn = {fps, 1, 0.0f, 0, 1.2f, 0.3f, 0.05f};
						const cf_gaze_params pr = real_params();
						const real_gaze lens = {0, 0, 0};
						cf_gaze_init(&g);
						rng_seed(71 + 5 * seed + lost + 13 * dir);
						double tt = 100.0;
						const int n0 = (int)(3.0 * fps);
						const float target = dir == 0 ? 0.10f * RW : 0.0f;
						int land = -1;
						float prev = 0, after = 0, big = 0;
						for (int i = 0; i < n0 + n_lost + (int)fps; i++, tt += dts) {
							const int phase = i < n0 ? 0 : (i < n0 + n_lost ? 1 : 2);
							cf_face f = noisy_real(&rn, (phase == 0) == (dir == 0) ? lens : at_notes());
							if (phase == 1)
								f.found = 0;
							const int nn = cf_gaze_update(&g, &f, &pr, tt, out);
							const float c = nn ? -out[0].sy : 0.0f;
							if (phase == 2) {
								const int k = i - n0 - n_lost;
								if (land < 0 && fabsf(c - target) <= 0.25f * 0.10f * RW)
									land = k;
								if (land >= 0 && k > land + 3)
									after = fmaxf(after, fabsf(c - prev));
								big = fmaxf(big, fabsf(c - prev));
							}
							prev = c;
						}
						if (lost < 5) {
							CHECK(land >= 0 && land <= 1 && after < 0.3f * (float)(30.0 / fps) + 0.15f,
							      "face lost %d frames across a glance %s (%.0f fps, seed %u): landed on frame %d "
							      "after it was back, steps after %.2f px",
							      n_lost, dir == 0 ? "to the notes" : "to the lens", fps, seed, land, after);
							worst_land = land > worst_land ? land : worst_land;
							worst_after = fmaxf(worst_after, after * (float)(fps / 30.0));
						} else {
							CHECK(land >= 0 && land * dts <= 0.5 && big < 1.0f * (float)(30.0 / fps),
							      "face lost 0.3 s across a glance %s (%.0f fps, seed %u): back on frame %d, "
							      "largest step %.2f px",
							      dir == 0 ? "to the notes" : "to the lens", fps, seed, land, big);
							worst_back = land > worst_back ? land : worst_back;
							worst_long = fmaxf(worst_long, big * (float)(fps / 30.0));
						}
					}
		printf("ok: face lost 1-4 frames across a glance: lands by frame %d of it being back, then steps of %.2f "
		       "px per 1/30 s at most; lost 0.3 s: back by frame %d, steps up to %.2f px per 1/30 s\n",
		       worst_land, worst_after, worst_back, worst_long);
	}

	/* 39. Eyes closed for longer than a blink's hold (0.55, 0.8 and 1.0 s) while reading, the points
	 * dragged by the lids: the correction holds, and past the hold eases out while they are shut. When they
	 * open it is back on the notes correction within 0.25 s, easing in about as fast as the lids open
	 * (never more than 70% of it in one frame; it used to restart from one noisy frame, 1.2-3.7 px steps
	 * with the eyes open, and then fade in over half a second). Realistic noise with Vision, 30 and 15 fps. */
	{
		const double closed[3] = {0.55, 0.8, 1.0};
		float worst_step = 0;
		double worst_back = 0;
		for (int fr = 0; fr < 2; fr++)
			for (int k = 0; k < 3; k++)
				for (unsigned seed = 0; seed < 4; seed++) {
					const double fps = fr ? 15.0 : 30.0, dts = 1.0 / fps;
					const real_noise rn = {fps, 1, 0.0f, 0, 1.2f, 0.3f, 0.05f};
					const cf_gaze_params pr = real_params();
					cf_gaze_init(&g);
					rng_seed(81 + 3 * seed + k);
					double tt = 100.0, back = -1;
					const int n0 = (int)(3.0 * fps), nc = (int)(closed[k] * fps + 0.5);
					float prev = 0, big = 0;
					for (int i = 0; i < n0 + nc + (int)(1.5 * fps); i++, tt += dts) {
						const int shut = i >= n0 && i < n0 + nc;
						rx.open = shut ? 0.05f : 1.0f;
						rx.drag[0] = rx.drag[1] = shut ? 0.08f : 0.0f;
						rx.vdrag = shut ? 0.04f : 0.0f;
						cf_face f = noisy_real(&rn, at_notes());
						const int nn = cf_gaze_update(&g, &f, &pr, tt, out);
						const float c = nn ? -out[0].sy : 0.0f;
						if (i >= n0 + nc)
							big = fmaxf(big, fabsf(c - prev));
						if (i >= n0 + nc && back < 0 && fabsf(c - 0.10f * RW) < 1.0f)
							back = (i - n0 - nc) * dts;
						prev = c;
					}
					rx = rx_none;
					CHECK(back >= 0 && back <= 0.25 + 1e-6 && big < 0.7f * 0.10f * RW,
					      "eyes closed %.2f s while reading (%.0f fps, seed %u): back on the notes correction "
					      "%.2f s after they opened, largest step %.2f px",
					      closed[k], fps, seed, back, big);
					worst_step = fmaxf(worst_step, big);
					worst_back = fmax(worst_back, back);
				}
		printf("ok: eyes closed 0.55-1 s while reading: back on the notes correction %.2f s after they open at "
		       "most, easing in (largest step %.2f px)\n",
		       worst_back, worst_step);
	}

	/* 40. A blink with the pupil points dragged by the lids (a generic profile: the lids close halfway over
	 * 2 frames and reopen over 6; one eye's refined point pulled down 0.10 eye widths, the other first up
	 * 0.04, then down 0.10, then up again as they open; Vision's 0.045 down), the output shown one frame
	 * late (lag 1: what frame i computes shows on frame i+1). The drags never back a change: with the
	 * gaze unchanged the correction stays within 0.5 px (the noise's own wander). A gaze change during the
	 * blink (notes -> lens, or lens -> notes; blinks often come with glances) may show as the lids close,
	 * its size read through the drags, but never the wrong way and never more than 35% past it; it is most
	 * of the way there (within 45%) 0.15 s after the lids are 85% open (one frame more: the first such frame
	 * may still carry the drag) and within 25% by 0.3 s. Looking at the notes with lowered lids and up to the
	 * lens as they rise is a saccade: it lands within 2 frames. Realistic noise with Vision, 30 and 15 fps. */
	{
		float worst_still = 0, worst_early = 0;
		double worst_land = 0, worst_near = 0;
		int worst_rise = 0;
		for (int fr = 0; fr < 2; fr++)
			for (int sc = 0; sc < 4; sc++)
				for (unsigned seed = 0; seed < 4; seed++) {
					const double fps = fr ? 15.0 : 30.0, dts = 1.0 / fps;
					const real_noise rn = {fps, 1, 0.0f, 0, 1.2f, 0.3f, 0.05f};
					const cf_gaze_params pr = real_params();
					const real_gaze lens = {0, 0, 0};
					cf_gaze_init(&g);
					rng_seed(91 + 5 * seed + sc);
					double tt = 100.0;
					const int n0 = (int)(3.0 * fps), n1 = n0 + (int)(1.0 * fps);
					const real_gaze from = sc == 2 ? lens : at_notes();
					const real_gaze to = sc == 1 || sc == 3 ? lens : at_notes();
					const float target = sc == 1 || sc == 3 ? 0.0f : 0.10f * RW;
					float steady = 0, prev = 0, still = 0, early = 0, k_next = 1;
					double open_t = -1, land_t = -1, near_t = -1;
					int rise = -1;
					float outs[64], ks[64];
					int no = 0;
					for (int i = 0; i < n1; i++, tt += dts) {
						const double ts = (i - n0) * dts; /* time into the blink */
						float c01 = 0;                  /* how far closed (0..1) */
						if (sc < 3 && ts >= 0) {
							if (ts < 0.067)
								c01 = (float)(ts / 0.067);
							else if (ts < 0.1)
								c01 = 1;
							else if (ts < 0.3)
								c01 = 1 - (float)((ts - 0.1) / 0.2);
						}
						rx.open = sc == 3 ? (ts < 0 ? 0.9f : 1.0f) : 1.0f - 0.5f * c01;
						rx.drag[0] = 0.10f * c01;
						rx.drag[1] = c01 > 0 ? (ts < 0.04 ? -0.04f : (ts > 0.2 ? -0.08f * c01 : 0.10f * c01)) : 0.0f;
						rx.vdrag = 0.045f * c01;
						const real_gaze gz = sc == 3 ? (ts < 0 ? from : to) : (ts < 0.05 ? from : to);
						cf_face f = noisy_real(&rn, gz);
						const int nn = cf_gaze_update(&g, &f, &pr, tt, out);
						const float c = nn ? -out[0].sy : 0.0f;
						if (i == n0 - 1)
							steady = c;
						if (i >= n0 && no < 64) {
							outs[no] = c;
							ks[no] = rx.open;
							no++;
						}
						prev = c;
					}
					rx = rx_none;
					(void)prev;
					(void)k_next;
					/* lag 1: outs[j] shows on frame j + 1, whose lids are ks[j + 1] */
					for (int j = 0; j + 1 < no; j++) {
						const float shown = outs[j], lids = ks[j + 1];
						const double ts = (j + 1) * dts;
						if (sc == 0)
							still = fmaxf(still, fabsf(shown - steady));
						if ((sc == 1 || sc == 2) && lids < 0.85f && open_t < 0)
							early = fmaxf(early, fmaxf(fminf(steady, target) - shown, shown - fmaxf(steady, target)));
						if ((sc == 1 || sc == 2) && lids >= 0.85f && open_t < 0 && ts > 0.1)
							open_t = ts;
						if ((sc == 1 || sc == 2) && land_t < 0 && fabsf(shown - target) <= 0.25f * 0.10f * RW)
							land_t = ts;
						if ((sc == 1 || sc == 2) && near_t < 0 && fabsf(shown - target) <= 0.45f * 0.10f * RW)
							near_t = ts;
						if (sc == 3 && rise < 0 && fabsf(shown - target) <= 0.25f * 0.10f * RW)
							rise = j;
					}
					if (sc == 0) {
						CHECK(still < 0.5f, "blink with dragged points, gaze still (%.0f fps, seed %u): the "
						      "correction moved %.2f px", fps, seed, still);
						worst_still = fmaxf(worst_still, still);
					} else if (sc < 3) {
						CHECK(early < 0.35f * 0.10f * RW && open_t > 0 && near_t >= 0 && land_t >= 0 &&
							      near_t - open_t <= 0.15 + dts + 1e-6 && land_t - open_t <= 0.3 + 1e-6,
						      "blink with a glance %s (%.0f fps, seed %u): %.2f px past where it was or went "
						      "before the lids were 85%% open (at %.3f s), within 45%% at %.3f s, 25%% at %.3f s",
						      sc == 1 ? "to the lens" : "to the notes", fps, seed, early, open_t, near_t, land_t);
						worst_early = fmaxf(worst_early, early);
						worst_near = fmax(worst_near, near_t - open_t);
						worst_land = fmax(worst_land, land_t - open_t);
					} else {
						CHECK(rise >= 0 && rise <= 1, "looking up to the lens as the lids rise (%.0f fps, seed "
						      "%u): landed on frame %d", fps, seed, rise);
						worst_rise = rise > worst_rise ? rise : worst_rise;
					}
				}
		printf("ok: blinks with lid-dragged points (lag 1): gaze still moves the correction %.2f px; a glance "
		       "during the blink goes %.2f px past either end before the lids are 85%% open, is within 45%% "
		       "%.3f s and 25%% %.3f s after; looking up with the lids rising lands on frame %d\n",
		       worst_still, worst_early, worst_near, worst_land, worst_rise);
	}

	/* 41. The refined points departing from Vision's without an eye movement (patterns seen on the user's
	 * camera, generic sizes): (a) one eye's point sits 0.05 eye widths low, then the other's drops 0.03;
	 * (b) one eye's point 0.06 low jumps 0.08 up in one frame; (d) the two disagree by 0.09 for 3 s, then
	 * the low one spikes 0.05 for a frame; (c) the lids drop a little while both refined points fall
	 * 0.05-0.09 for 0.15 s. Vision's points don't move, so nothing backs a change: the output never steps
	 * faster than the slew limit. (c') The same with Vision's points dragged 0.015 too (1.3 px, 2.2 px of
	 * gaze with the calibrated gain): that is what a short look down looks like, and may show as one, but
	 * when the points come back so does the correction, at once (within 1.5 px, a few frames' noise, no
	 * later than 0.1 s and a frame after the first frame without the drags), not gliding back over half a
	 * second. Realistic noise with Vision, 25, 30 and 15 fps. */
	{
		const double fpss[3] = {25.0, 30.0, 15.0};
		float worst[5] = {0, 0, 0, 0, 0};
		double worst_back = 0;
		for (int fr = 0; fr < 3; fr++)
			for (int pat = 0; pat < 5; pat++)
				for (unsigned seed = 0; seed < 4; seed++) {
					const double fps = fpss[fr], dts = 1.0 / fps;
					const real_noise rn = {fps, 1, 0.0f, 0, 1.2f, 0.3f, 0.05f};
					real_vis_cal = pat == 4;
					const cf_gaze_params pr = real_params();
					real_vis_cal = 0;
					cf_gaze_init(&g);
					rng_seed(101 + 7 * seed + pat);
					double tt = 100.0;
					const int n0 = (int)(5.0 * fps), n1 = n0 + (int)(3.0 * fps);
					float prev = 0, big = 0, steady = 0, dev = 0;
					double back = -1, clean_t = -1;
					for (int i = 0; i < n1; i++, tt += dts) {
						const double tp = (i - n0) * dts;
						rx = rx_none;
						if (pat == 0) {
							rx.drag[1] = tp >= -0.4 ? 0.05f : 0.0f;
							rx.drag[0] = tp >= 0 ? 0.03f : 0.0f;
						} else if (pat == 1) {
							rx.drag[1] = tp >= -1.0 ? (tp >= 0 ? -0.02f : 0.06f) : 0.0f;
						} else if (pat == 2 || pat == 4) {
							if (tp >= 0 && tp < 0.15) {
								rx.open = 0.97f;
								rx.drag[0] = tp < 0.04 ? 0.05f : 0.09f;
								rx.drag[1] = tp < 0.04 ? 0.05f : 0.03f;
								rx.vdrag = pat == 4 ? 0.015f : 0.0f;
							}
						} else {
							rx.drag[1] = tp >= -3.0 ? (tp >= 0 && tp < 0.99 * dts ? 0.04f : 0.09f) : 0.0f;
						}
						cf_face f = noisy_real(&rn, at_notes());
						const int nn = cf_gaze_update(&g, &f, &pr, tt, out);
						const float c = nn ? -out[0].sy : 0.0f;
						if (i == n0 - (int)(1.2 * fps))
							steady = c;
						if (i > n0 - (int)(1.2 * fps)) {
							big = fmaxf(big, fabsf(c - prev));
							dev = fmaxf(dev, fabsf(c - steady));
						}
						if (tp >= 0.15 - 1e-6 && clean_t < 0)
							clean_t = tp; /* the first frame without the drags */
						if (clean_t >= 0 && back < 0 && fabsf(c - steady) < 1.5f)
							back = tp - clean_t;
						prev = c;
					}
					rx = rx_none;
					const float slew = 0.3f * (float)(30.0 / fps) + 0.1f;
					if (pat == 4) {
						CHECK(dev < 1.5f || (back >= 0 && back <= 0.1 + dts + 1e-6),
						      "refined points falling with the lids, Vision's a little (%.0f fps, seed %u): the "
						      "correction moved %.2f px, back %.3f s after the points",
						      fps, seed, dev, back);
						worst[pat] = fmaxf(worst[pat], dev);
						worst_back = fmax(worst_back, back);
					} else {
						CHECK(big < slew, "refined points departing from Vision's, pattern %c (%.0f fps, seed %u): "
						      "a step of %.2f px", "abcd"[pat], fps, seed, big);
						worst[pat] = fmaxf(worst[pat], big * (float)(fps / 30.0));
					}
				}
		printf("ok: refined points departing from Vision's: patterns a, b, c, d step at most %.2f / %.2f / %.2f / "
		       "%.2f px per 1/30 s; c with Vision's dragged too moves the correction %.2f px and is back %.3f s "
		       "after the points\n",
		       worst[0], worst[1], worst[2], worst[3], worst[4], worst_back);
	}
	test_pose();
	test_head_lids();
	test_side_scan();
	test_vision_tail();
	test_learn_notes();

	if (failures) {
		printf("%d FAILED\n", failures);
		return 1;
	}
	printf("ALL CHECKS PASSED\n");
	return 0;
}
