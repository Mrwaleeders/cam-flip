/*
 * Head pose checks (src/pose.c), and the gaze model's fade driven by it. Included by test_gaze.c
 * (uses its CHECK, face_at, gauss and rng_seed), so the one test binary runs everything:
 *   cc -O2 -Wall -Wextra -Werror -Isrc tests/test_gaze.c src/gaze.c src/warp.c src/pose.c -lm
 *
 * The faces are 3D landmark sets (eye outlines and nose points, in mm) rotated in pitch and yaw
 * about the neck, projected through a pinhole camera 60 cm away (so perspective is in) and rolled in
 * the image. Their depths differ from the typical ones pose.c assumes (flat and deep noses, shallow
 * and deep-set eye corners) and some are asymmetric: the estimate must still be monotonic, continuous
 * and symmetric, and only its scale may be off. (Even the typical face reads turns ~1.2x: at 60 cm
 * the near eye looks bigger and a head turning about the neck moves off the camera's axis. Seen
 * orthographically it reads them to within 4%.)
 */
#include "pose.h"

typedef struct {
	float nose_dz;   /* extra depth of every nose point (mm): flatter or more prominent nose */
	float corner_d;  /* outer eye corner this far behind the inner one (mm) */
	float nose_dx;   /* nose off the middle (mm), asymmetric face */
	float eye_dw;    /* left eye this much wider than the right one (mm) */
	float canthal;   /* outer corners this much higher than the inner ones (mm) */
	int eye_points;  /* points per eye outline (Vision: 6 or 8) */
} synth_face;

typedef struct {
	cf_pt eye[2][16];
	int ne;
	cf_pt nose[16];
	int nn;
} synth_lm;

#define SF_IOD 63.0f   /* mm between the eye centres */
#define SF_DIST 600.0f /* mm from the camera to the eyes */

/* Builds the landmarks of `sf` tilted by pitch (head down positive), then turned by yaw, about a pivot
 * in the neck, seen by a camera with focal length `focal` px, the point between the eyes landing near
 * (ox, oy); then rolled (a tilt to the shoulder, which keeps the face square to the camera) about that
 * point. Degrees. */
static void synth_project(const synth_face *sf, float yaw, float pitch, float roll, float focal, float ox, float oy,
			  synth_lm *out)
{
	float P[48][3];
	int n = 0;
	const int ne = sf->eye_points;
	for (int e = 0; e < 2; e++) {
		const float side = e == 0 ? -1.0f : 1.0f; /* image left eye first */
		const float w = 29.0f + (e == 0 ? 0.5f : -0.5f) * sf->eye_dw, h = 10.0f;
		for (int i = 0; i < ne; i++) {
			const float a = 6.2831853f * (float)i / (float)ne;
			const float u = cosf(a); /* -1 .. 1 along the eye, +1 on the image's right */
			const float out_u = side * u; /* +1 at the outer corner */
			P[n][0] = side * SF_IOD / 2 + u * w / 2;
			P[n][1] = h / 2 * sinf(a) - out_u * sf->canthal / 2;
			/* inner corner forward, outer back; the lids bulge over the eyeball in between */
			P[n][2] = -out_u * sf->corner_d / 2 + 3.0f * (1.0f - u * u);
			n++;
		}
	}
	/* Nose: crest from the bridge to the tip, and the outline of the lower nose. */
	static const float nose[13][3] = {
		{0, 6, 13},   {0, 16, 18},  {0, 26, 23},  {0, 36, 28},  /* crest */
		{-17, 37, 6}, {-14, 44, 10}, {-7, 47, 15}, {0, 47, 18}, {7, 47, 15}, {14, 44, 10}, {17, 37, 6},
		{-8, 40, 20}, {8, 40, 20}};
	for (int i = 0; i < 13; i++) {
		P[n][0] = nose[i][0] + sf->nose_dx;
		P[n][1] = nose[i][1];
		P[n][2] = nose[i][2] + sf->nose_dz;
		n++;
	}
	const float d2r = 0.017453293f;
	const float cp = cosf(pitch * d2r), sp = sinf(pitch * d2r), cy = cosf(yaw * d2r), sy = sinf(yaw * d2r);
	const float cr = cosf(roll * d2r), sr = sinf(roll * d2r);
	const float pv[3] = {0, 70, -90}; /* neck pivot, mm */
	for (int i = 0; i < n; i++) {
		const float x = P[i][0] - pv[0], y = P[i][1] - pv[1], z = P[i][2] - pv[2];
		const float y1 = y * cp + z * sp, z1 = -y * sp + z * cp;
		const float x2 = x * cy + z1 * sy, z2 = -x * sy + z1 * cy;
		const float X = x2 + pv[0], Y = y1 + pv[1], Z = z2 + pv[2];
		const float s = focal / (SF_DIST - Z);
		const float u = X * s, v = Y * s;
		const cf_pt q = {ox + u * cr - v * sr, oy + u * sr + v * cr};
		if (i < ne)
			out->eye[0][i] = q;
		else if (i < 2 * ne)
			out->eye[1][i - ne] = q;
		else
			out->nose[i - 2 * ne] = q;
	}
	out->ne = ne;
	out->nn = 13;
}

static void synth_noise(synth_lm *lm, float sd)
{
	for (int e = 0; e < 2; e++)
		for (int i = 0; i < lm->ne; i++) {
			lm->eye[e][i].x += gauss(sd);
			lm->eye[e][i].y += gauss(sd);
		}
	for (int i = 0; i < lm->nn; i++) {
		lm->nose[i].x += gauss(sd);
		lm->nose[i].y += gauss(sd);
	}
}

static int synth_measure(const synth_lm *lm, cf_head *m)
{
	return cf_head_measure(lm->eye[0], lm->ne, lm->eye[1], lm->ne, lm->nose, lm->nn, m);
}

/* focal 1400 px at 60 cm: eyes ~147 px apart, as on a 1080p MacBook camera. */
#define SF_FOCAL 1400.0f

static cf_head synth_head(const synth_face *sf, float yaw, float pitch, float roll)
{
	synth_lm lm;
	synth_project(sf, yaw, pitch, roll, SF_FOCAL, 960, 480, &lm);
	cf_head m;
	synth_measure(&lm, &m);
	return m;
}

/* The face's reference: measured while looking into the camera (yaw0, pitch0). */
static cf_head_ref synth_ref(const synth_face *sf, float yaw0, float pitch0)
{
	cf_head s[CF_HEAD_MIN_SAMPLES];
	for (int i = 0; i < CF_HEAD_MIN_SAMPLES; i++)
		s[i] = synth_head(sf, yaw0, pitch0, 0);
	return cf_head_reference(s, CF_HEAD_MIN_SAMPLES);
}

static void synth_angles(const synth_face *sf, const cf_head_ref *ref, float yaw, float pitch, float roll, float *ey,
			 float *ep)
{
	const cf_head m = synth_head(sf, yaw, pitch, roll);
	cf_head_angles(&m, ref, ey, ep);
}

/* A gaze-model face (test_gaze.c's synthetic eyes glancing at the notes) carrying the head measures of
 * `sf` at this pose, with landmark noise. */
static cf_face pose_face(const synth_face *sf, float yaw, float pitch, float noise_px)
{
	cf_face f = face_at(960, 500, 0.25f, 16, 0);
	f.has_pose = 0;
	synth_lm lm;
	synth_project(sf, yaw, pitch, 0, SF_FOCAL, 960, 480, &lm);
	if (noise_px > 0)
		synth_noise(&lm, noise_px);
	f.has_head_geom = synth_measure(&lm, &f.head);
	return f;
}

/* Runs a head movement through the gaze model at 30 fps: yaw and pitch ramp linearly (smoothly in and
 * out) from (y0, p0) to (y1, p1) over `secs`, then hold for `hold` s. Tracks the correction factor. */
typedef struct {
	float max_step;   /* largest frame-to-frame change of the factor */
	float against;    /* largest move of the factor against the direction of the fade */
	float end;        /* factor at the end */
	int half_frame;   /* first frame the factor crossed 0.5 (-1: never) */
	int span;         /* frames from its first crossing 0.9 (0.1 fading in) to its first crossing 0.1 (0.9) */
} fade_stats;

static fade_stats run_head(cf_gaze *g, const cf_gaze_params *p, double *t, const synth_face *sf, float y0, float p0,
			   float y1, float p1, float secs, float hold, float noise_px, int fading_out)
{
	fade_stats s = {0, 0, 0, -1, -1};
	int started = -1;
	float prev = g->factor;
	const int frames = (int)((secs + hold) * 30.0f + 0.5f);
	cf_eye out[2];
	for (int i = 0; i < frames; i++, *t += 1.0 / 30.0) {
		float k = fminf((float)i / (secs * 30.0f), 1.0f);
		k = k * k * (3 - 2 * k); /* a head turn starts and ends gently */
		const cf_face f = pose_face(sf, y0 + (y1 - y0) * k, p0 + (p1 - p0) * k, noise_px);
		cf_gaze_update(g, &f, p, *t, out);
		const float fa = g->factor;
		s.max_step = fmaxf(s.max_step, fabsf(fa - prev));
		s.against = fmaxf(s.against, fading_out ? fa - prev : prev - fa);
		if (s.half_frame < 0 && (fading_out ? fa < 0.5f : fa > 0.5f))
			s.half_frame = i;
		if (started < 0 && (fading_out ? fa < 0.9f : fa > 0.1f))
			started = i;
		if (s.span < 0 && (fading_out ? fa < 0.1f : fa > 0.9f))
			s.span = i - started;
		prev = fa;
	}
	s.end = prev;
	return s;
}

static void test_pose(void)
{
	const synth_face faces[4] = {
		{0, 9, 0, 0, 2, 8},          /* typical, symmetric, 8 points per eye */
		{-5, 6, 0, 0, 1, 6},         /* flat nose, shallow corners, 6 points per eye like Vision's */
		{6, 13, 0, 0, 3, 8},         /* prominent nose, deep-set corners */
		{0, 9, 2.5f, 1.2f, 2.5f, 6}, /* asymmetric: nose 2.5 mm off the middle, one eye wider */
	};
	const char *names[4] = {"typical", "flat", "deep", "asymmetric"};

	/* 1. Sweeps through yaw and through pitch, noise-free: monotonic, continuous, symmetric, roughly to
	 * scale, and never folding back below the fade-out angle however far the head goes. The typical depths
	 * pose.c assumes fit the typical and asymmetric faces; for the flat and deep ones the scale is off
	 * (pitch most: past a few degrees it rests on the corners' depth), which only moves where the fade
	 * happens: their scale is checked up to 20 deg, the fade's range. (Beyond ~45 deg
	 * of yaw the far eye's synthetic outline is hidden behind the nose in a real face; the sweep stops
	 * there and only checks that the reading stays past the fade.) */
	for (int fi = 0; fi < 4; fi++) {
		const synth_face *sf = &faces[fi];
		const int typical = fi == 0 || fi == 3;
		const cf_head_ref ref = synth_ref(sf, 0, 0);
		CHECK(ref.ok, "%s: reference learnt", names[fi]);
		float zy, zp;
		synth_angles(sf, &ref, 0, 0, 0, &zy, &zp);
		CHECK(fabsf(zy) < 0.05f && fabsf(zp) < 0.05f, "%s: calibrated pose reads (%.2f, %.2f), not 0", names[fi], zy,
		      zp);
		for (int axis = 0; axis < 2; axis++) {
			const float lim = axis == 0 ? 42.0f : 55.0f, far = axis == 0 ? 60.0f : 70.0f;
			const float fade_hi = axis == 0 ? CF_YAW_FADE_HI : CF_PITCH_FADE_HI;
			float prev = -1e9f, min_inc = 1e9f, max_step = 0, lo_ratio = 1e9f, hi_ratio = 0, worst_sym = 0;
			float other = 0, beyond = 1e9f;
			for (float a = -far; a <= far + 1e-3f; a += 0.25f) {
				float ey, ep;
				synth_angles(sf, &ref, axis == 0 ? a : 0, axis == 1 ? a : 0, 0, &ey, &ep);
				const float e = axis == 0 ? ey : ep;
				if (fabsf(a) > lim) {
					beyond = fminf(beyond, a > 0 ? e : -e);
					prev = -1e9f;
					continue;
				}
				other = fmaxf(other, fabsf(axis == 0 ? ep : ey));
				if (prev > -1e8f) {
					min_inc = fminf(min_inc, e - prev);
					if (fabsf(e) <= 45)
						max_step = fmaxf(max_step, fabsf(e - prev));
				}
				prev = e;
				if (fabsf(a) >= 5 && fabsf(a) <= (typical ? 30 : 20)) {
					lo_ratio = fminf(lo_ratio, e / a);
					hi_ratio = fmaxf(hi_ratio, e / a);
				}
				if (a > 0 && a <= 20) {
					float my, mp;
					synth_angles(sf, &ref, axis == 0 ? -a : 0, axis == 1 ? -a : 0, 0, &my, &mp);
					const float m = axis == 0 ? my : mp;
					worst_sym = fmaxf(worst_sym, fabsf(e + m) / a);
				}
			}
			const char *an = axis == 0 ? "yaw" : "pitch";
			CHECK(min_inc >= -1e-3f, "%s %s: not monotonic (a 0.25 deg step changed it by %.3f)", names[fi], an,
			      min_inc);
			CHECK(max_step <= 0.75f, "%s %s: jumps %.2f deg for a 0.25 deg step", names[fi], an, max_step);
			if (typical)
				CHECK(lo_ratio > 0.65f && hi_ratio < 1.35f, "%s %s: reads %.2f..%.2f x the angle (5..30 deg)",
				      names[fi], an, lo_ratio, hi_ratio);
			else
				CHECK(lo_ratio > 0.5f && hi_ratio < 2.0f, "%s %s: reads %.2f..%.2f x the angle (5..20 deg)", names[fi],
				      an, lo_ratio, hi_ratio);
			CHECK(worst_sym < (fi == 0 ? 0.10f : 0.30f), "%s %s: the two ways differ by %.0f%% of the angle",
			      names[fi], an, 100 * worst_sym);
			CHECK(other < (axis == 0 ? 6.0f : 3.0f), "%s: a pure %s reads up to %.1f deg of %s", names[fi], an, other,
			      axis == 0 ? "pitch" : "yaw");
			CHECK(beyond > fade_hi, "%s %s: %.0f..%.0f deg reads as little as %.1f deg (fade gone at %.0f)",
			      names[fi], an, lim, far, beyond, fade_hi);
			printf("ok: %s face, %s -%.0f..%.0f deg: monotonic (smallest 0.25 deg step %.3f), largest step %.2f, reads "
			       "%.2f..%.2f x the angle, the two ways within %.0f%%, cross-talk %.1f deg; %.0f..%.0f deg reads %.1f or "
			       "more\n",
			       names[fi], an, lim, lim, min_inc, max_step, lo_ratio, hi_ratio, 100 * worst_sym, other, lim, far,
			       beyond);
		}
	}

	/* 2. Roll, size and position change nothing; a turn and a tilt together are both seen. */
	{
		const synth_face *sf = &faces[3];
		const cf_head_ref ref = synth_ref(sf, 0, 0);
		float worst_roll = 0, worst_scale = 0, worst_shift = 0;
		const float poses[5][2] = {{0, 0}, {20, 0}, {-25, 10}, {10, -20}, {-15, 25}};
		for (int k = 0; k < 5; k++) {
			float y0, p0;
			synth_angles(sf, &ref, poses[k][0], poses[k][1], 0, &y0, &p0);
			for (float roll = -30; roll <= 30; roll += 15) {
				float y, p;
				synth_angles(sf, &ref, poses[k][0], poses[k][1], roll, &y, &p);
				worst_roll = fmaxf(worst_roll, fmaxf(fabsf(y - y0), fabsf(p - p0)));
			}
			for (int sc = 0; sc < 2; sc++) {
				synth_lm lm;
				synth_project(sf, poses[k][0], poses[k][1], 0, sc ? 3.0f * SF_FOCAL : 0.4f * SF_FOCAL, 960, 480, &lm);
				cf_head m;
				synth_measure(&lm, &m);
				float y, p;
				cf_head_angles(&m, &ref, &y, &p);
				worst_scale = fmaxf(worst_scale, fmaxf(fabsf(y - y0), fabsf(p - p0)));
				for (int e = 0; e < 2; e++)
					for (int i = 0; i < lm.ne; i++) {
						lm.eye[e][i].x += 333;
						lm.eye[e][i].y -= 177;
					}
				for (int i = 0; i < lm.nn; i++) {
					lm.nose[i].x += 333;
					lm.nose[i].y -= 177;
				}
				synth_measure(&lm, &m);
				float y2, p2;
				cf_head_angles(&m, &ref, &y2, &p2);
				worst_shift = fmaxf(worst_shift, fmaxf(fabsf(y2 - y), fabsf(p2 - p)));
			}
		}
		CHECK(worst_roll < 0.3f, "head roll changed the pose by %.2f deg", worst_roll);
		CHECK(worst_scale < 0.3f, "face size changed the pose by %.2f deg", worst_scale);
		CHECK(worst_shift < 0.01f, "moving the face in the image changed the pose by %.3f deg", worst_shift);
		float y, p, y2, p2;
		synth_angles(sf, &ref, 25, 20, 0, &y, &p);
		synth_angles(sf, &ref, -25, -20, 0, &y2, &p2);
		CHECK(y > 18 && y < 40 && p > 12 && p < 30 && y2 < -18 && y2 > -40 && p2 < -12 && p2 > -30,
		      "turned 25 and tilted 20 deg reads (%.1f, %.1f), the other way (%.1f, %.1f)", y, p, y2, p2);
		printf("ok: head roll changes the pose by %.2f deg at most, face size by %.2f, position by %.3f; turned 25 and "
		       "tilted 20 deg reads (%.1f, %.1f), (-25, -20) reads (%.1f, %.1f)\n",
		       worst_roll, worst_scale, worst_shift, y, p, y2, p2);
	}

	/* 3. Landmark noise (0.6 px on eyes 147 px apart): raw per frame, and after the light smoothing that
	 * feeds the fade. Plus a one-frame landmark glitch and a gap. */
	{
		rng_seed(7);
		const synth_face *sf = &faces[0];
		const cf_head_ref ref = synth_ref(sf, 0, 0);
		cf_pose ps;
		memset(&ps, 0, sizeof(ps));
		double sy = 0, sp = 0, ry = 0, rp = 0;
		float worst = 0;
		const int N = 900;
		for (int i = 0; i < N; i++) {
			synth_lm lm;
			synth_project(sf, 0, 0, 0, SF_FOCAL, 960, 480, &lm);
			synth_noise(&lm, 0.6f);
			cf_head m;
			synth_measure(&lm, &m);
			float y, p;
			cf_head_angles(&m, &ref, &y, &p);
			ry += (double)y * y;
			rp += (double)p * p;
			cf_pose_update(&ps, &m, &ref, i / 30.0, &y, &p);
			if (i >= 15) {
				sy += (double)y * y;
				sp += (double)p * p;
				worst = fmaxf(worst, fmaxf(fabsf(y), fabsf(p)));
			}
		}
		const float raw_y = (float)sqrt(ry / N), raw_p = (float)sqrt(rp / N);
		const float sm_y = (float)sqrt(sy / (N - 15)), sm_p = (float)sqrt(sp / (N - 15));
		CHECK(sm_y < 0.8f && sm_p < 0.8f && worst < CF_YAW_FADE_LO / 3,
		      "noisy landmarks: smoothed pose RMS yaw %.2f pitch %.2f deg, worst %.1f deg", sm_y, sm_p, worst);
		/* A single frame with the nose 15 px off (a landmark glitch): the median of three ignores it. */
		cf_pose_update(&ps, &(cf_head){0}, &ref, N / 30.0, &(float){0}, &(float){0}); /* no nose this frame */
		float before = ps.yaw, glitch = 0;
		for (int i = 0; i < 10; i++) {
			synth_lm lm;
			synth_project(sf, 0, 0, 0, SF_FOCAL, 960, 480, &lm);
			if (i == 3)
				for (int k = 0; k < lm.nn; k++)
					lm.nose[k].x += 15;
			cf_head m;
			synth_measure(&lm, &m);
			float y, p;
			cf_pose_update(&ps, &m, &ref, (N + 1 + i) / 30.0, &y, &p);
			glitch = fmaxf(glitch, fabsf(y - before));
		}
		CHECK(glitch < 0.5f, "a one-frame landmark glitch moved the pose %.2f deg", glitch);
		/* No measures: held for CF_POSE_HOLD, then gone. */
		float y, p;
		const double t0 = (N + 11) / 30.0;
		const int held = cf_pose_update(&ps, NULL, &ref, t0 + 0.3, &y, &p);
		const int gone = !cf_pose_update(&ps, NULL, &ref, t0 + 0.6, &y, &p);
		CHECK(held && gone, "missing measures: held %d, released %d", held, gone);
		printf("ok: landmark noise 0.6 px: pose RMS raw yaw %.2f pitch %.2f deg, smoothed %.2f / %.2f (worst %.1f); a "
		       "one-frame glitch moves it %.2f deg; gaps held for %.1f s\n",
		       raw_y, raw_p, sm_y, sm_p, worst, glitch, CF_POSE_HOLD);
	}

	/* 4. Without a calibration: a typical face. Yaw still works both ways (a few degrees smaller), pitch is
	 * not guessed. Degenerate landmarks are refused. */
	{
		const synth_face *sf = &faces[0];
		float prev = -1e9f, min_inc = 1e9f, sym = 0, pitch_max = 0;
		for (float a = -45; a <= 45; a += 0.5f) {
			float y, p, m, mp;
			synth_angles(sf, NULL, a, 0, 0, &y, &p);
			synth_angles(sf, NULL, -a, 0, 0, &m, &mp);
			if (prev > -1e8f)
				min_inc = fminf(min_inc, y - prev);
			prev = y;
			pitch_max = fmaxf(pitch_max, fabsf(p));
			if (a >= 20)
				sym = fmaxf(sym, fabsf(y + m) / a);
		}
		float y30, p30, y0, p0;
		float dummy;
		synth_angles(sf, NULL, 30, 0, 0, &y30, &dummy);
		synth_angles(sf, NULL, 0, 30, 0, &dummy, &p30);
		synth_angles(sf, NULL, 0, 0, 0, &y0, &p0);
		CHECK(min_inc >= 0 && sym < 0.35f && pitch_max == 0 && p30 == 0 && y30 > 15 && fabsf(y0) < 5,
		      "uncalibrated: yaw monotonic %.3f, sides differ %.0f%%, pitch %.1f, 30 deg reads %.1f, front %.1f",
		      min_inc, 100 * sym, pitch_max, y30, y0);
		cf_head m;
		const cf_pt few[2] = {{1, 1}, {2, 2}};
		const cf_pt eye[4] = {{100, 100}, {110, 98}, {120, 100}, {110, 102}};
		cf_pt nan_eye[4];
		memcpy(nan_eye, eye, sizeof(eye));
		nan_eye[2].x = NAN;
		const int bad = cf_head_measure(few, 2, eye, 4, eye, 4, &m) + cf_head_measure(eye, 4, eye, 4, eye, 4, &m) +
				cf_head_measure(nan_eye, 4, eye, 4, eye, 4, &m) + cf_head_measure(eye, 4, eye, 4, eye, 0, &m);
		cf_head s[CF_HEAD_MIN_SAMPLES];
		for (int i = 0; i < CF_HEAD_MIN_SAMPLES; i++)
			s[i] = synth_head(sf, 0, 0, 0);
		const cf_head_ref few_ref = cf_head_reference(s, CF_HEAD_MIN_SAMPLES - 1);
		CHECK(bad == 0 && !few_ref.ok, "degenerate landmarks accepted (%d) or a reference from too few samples (%d)",
		      bad, few_ref.ok);
		printf("ok: uncalibrated: yaw monotonic, the two ways within %.0f%%, turned 30 deg reads %.1f, front %.1f, "
		       "pitch not guessed; degenerate landmarks refused\n",
		       100 * sym, y30, y0);
	}

	/* 5. The gaze model: correction locked on while the head is still (noisy landmarks); a gradual head turn
	 * left or right, or a tilt down or up, fades it out and looking back fades it in, smoothly, the same
	 * either way. */
	{
		const cf_gaze_params pp = {.strength = 1.0f, .max_glance = 0.15f, .head_ref = synth_ref(&faces[0], 0, 0)};
		cf_gaze g;
		double t = 0;
		rng_seed(11);
		cf_gaze_init(&g);
		fade_stats s = run_head(&g, &pp, &t, &faces[0], 0, 0, 0, 0, 1.0f, 10.0f, 0.6f, 0);
		float lo = 1;
		{
			cf_eye out[2];
			for (int i = 0; i < 300; i++, t += 1.0 / 30.0) {
				const cf_face f = pose_face(&faces[0], 0, 0, 0.6f);
				cf_gaze_update(&g, &f, &pp, t, out);
				lo = fminf(lo, g.factor);
			}
		}
		CHECK(s.end > 0.97f && lo > 0.97f, "still head with noisy landmarks: correction %.2f, dipped to %.2f", s.end,
		      lo);
		printf("ok: still head, noisy landmarks: correction stays locked on (lowest %.3f over 10 s)\n", lo);

		const struct {
			const char *name;
			float yaw, pitch;
		} moves[4] = {{"turn right", 40, 0}, {"turn left", -40, 0}, {"tilt down", 0, 45}, {"tilt up", 0, -45}};
		int half_out[4], half_in[4];
		for (int k = 0; k < 4; k++) {
			{
				cf_gaze_init(&g);
				t = 100.0 * (k + 1);
				run_head(&g, &pp, &t, &faces[0], 0, 0, 0, 0, 0.5f, 1.0f, 0.6f, 0);
				const fade_stats o = run_head(&g, &pp, &t, &faces[0], 0, 0, moves[k].yaw, moves[k].pitch, 1.2f, 1.0f,
							      0.6f, 1);
				const fade_stats i = run_head(&g, &pp, &t, &faces[0], moves[k].yaw, moves[k].pitch, 0, 0, 1.2f, 1.0f,
							      0.6f, 0);
				half_out[k] = o.half_frame;
				half_in[k] = i.half_frame;
				/* The factor's own ease (gaze.c) bounds a step to ~0.125 per frame at 30 fps; the pose must
				 * not add to it, and a gradual turn must give a fade over several frames. */
				CHECK(o.end < 0.03f && i.end > 0.97f && o.max_step < 0.13f && i.max_step < 0.13f &&
					      o.against < 0.01f && i.against < 0.01f && o.half_frame > 0 && i.half_frame > 0 &&
					      o.span >= 6 && i.span >= 6,
				      "%s: let go to %.2f and back to %.2f; largest step %.3f out, %.3f in; against the fade %.3f / %.3f; "
				      "0.9 to 0.1 in %d / %d frames",
				      moves[k].name, o.end, i.end, o.max_step, i.max_step, o.against, i.against, o.span, i.span);
				printf("ok: gradual head %s (40-45 deg over 1.2 s and back): lets go to %.3f (half way at frame %d, "
				       "0.9 to 0.1 over %d frames), takes over to %.3f (half way at frame %d, over %d frames); largest "
				       "step %.3f / %.3f per frame, never against the fade by more than %.4f\n",
				       moves[k].name, o.end, o.half_frame, o.span, i.end, i.half_frame, i.span, o.max_step, i.max_step,
				       fmaxf(o.against, i.against));
			}
		}
		CHECK(abs(half_out[0] - half_out[1]) <= 2 && abs(half_in[0] - half_in[1]) <= 2 &&
			      abs(half_out[2] - half_out[3]) <= 4 && abs(half_in[2] - half_in[3]) <= 4,
		      "fades not symmetric: right/left at frames %d/%d out, %d/%d in; down/up %d/%d out, %d/%d in", half_out[0],
		      half_out[1], half_in[0], half_in[1], half_out[2], half_out[3], half_in[2], half_in[3]);
		printf("ok: fades are symmetric: half way at frames %d/%d (right/left) and %d/%d (down/up) going, %d/%d and "
		       "%d/%d coming back\n",
		       half_out[0], half_out[1], half_out[2], half_out[3], half_in[0], half_in[1], half_in[2], half_in[3]);

		/* A sudden pose change (the head already turned on the next frame) is still a fade, not a step. */
		cf_gaze_init(&g);
		run_head(&g, &pp, &t, &faces[0], 0, 0, 0, 0, 0.5f, 1.0f, 0, 0);
		const fade_stats jo = run_head(&g, &pp, &t, &faces[0], 40, 0, 40, 0, 0.1f, 1.0f, 0, 1);
		const fade_stats ji = run_head(&g, &pp, &t, &faces[0], 0, 0, 0, 0, 0.1f, 1.0f, 0, 0);
		CHECK(jo.max_step < 0.13f && ji.max_step < 0.13f && jo.end < 0.03f && ji.end > 0.97f && jo.span >= 6 &&
			      ji.span >= 6,
		      "sudden head turn: largest factor step %.3f out, %.3f in; 0.9 to 0.1 in %d / %d frames", jo.max_step,
		      ji.max_step, jo.span, ji.span);
		printf("ok: a sudden head turn still fades (largest step %.3f out, %.3f in; over %d / %d frames)\n", jo.max_step,
		       ji.max_step, jo.span, ji.span);
	}

	/* 6. The lens calibration learns the reference: someone whose "looking into the camera" is turned 10
	 * and tilted 6 deg (camera off to the side and high, or just how they sit) is locked on there, and
	 * turning 30 deg either way from there lets go. Uncalibrated, pitch is left alone. */
	{
		const synth_face *sf = &faces[3];
		cf_gaze_params pp = {.strength = 1.0f, .max_glance = 0.15f};
		cf_gaze g;
		cf_gaze_init(&g);
		double t = 1000;
		cf_gaze_start_calibration(&g, CF_CALIB_LENS);
		cf_calib_result res;
		memset(&res, 0, sizeof(res));
		int got = 0;
		cf_eye out[2];
		for (int i = 0; i < 300 && !got; i++, t += 1.0 / 30.0) {
			cf_face f = pose_face(sf, 10, 6, 0.5f);
			f.left_pupil.y -= 4; /* looking into the lens */
			f.right_pupil.y -= 4;
			cf_gaze_update(&g, &f, &pp, t, out);
			got = cf_gaze_take_calibration(&g, &res);
		}
		const cf_head_ref want = synth_ref(sf, 10, 6);
		CHECK(got == 1 && res.head.ok && fabsf(res.head.side - want.side) < 0.01f &&
			      fabsf(res.head.vert - want.vert) < 0.01f,
		      "lens calibration: head reference %d (side %.3f vert %.3f, expected %.3f %.3f)", res.head.ok,
		      res.head.side, res.head.vert, want.side, want.vert);
		pp.baseline = res.offset;
		pp.baseline_open = res.openness;
		pp.head_ref = res.head;
		cf_gaze_init(&g);
		const fade_stats st = run_head(&g, &pp, &t, sf, 10, 6, 10, 6, 0.5f, 3.0f, 0.5f, 0);
		const fade_stats r = run_head(&g, &pp, &t, sf, 10, 6, 40, 6, 1.0f, 1.0f, 0.5f, 1);
		run_head(&g, &pp, &t, sf, 40, 6, 10, 6, 1.0f, 1.0f, 0.5f, 0);
		const fade_stats l = run_head(&g, &pp, &t, sf, 10, 6, -20, 6, 1.0f, 1.0f, 0.5f, 1);
		CHECK(st.end > 0.97f && r.end < 0.03f && l.end < 0.03f,
		      "calibrated at (10, 6) deg: locked %.2f there, %.2f turned to 40, %.2f turned to -20", st.end, r.end,
		      l.end);
		/* Without the reference: the same tilt is not taken for a look away, a turn still is. */
		cf_gaze_params np = pp;
		np.head_ref.ok = 0;
		cf_gaze_init(&g);
		const fade_stats nt = run_head(&g, &np, &t, &faces[0], 0, 0, 0, 35, 0.5f, 2.0f, 0.5f, 0);
		run_head(&g, &np, &t, &faces[0], 0, 35, 0, 0, 0.5f, 1.0f, 0.5f, 0);
		const fade_stats ny = run_head(&g, &np, &t, &faces[0], 0, 0, 40, 0, 1.0f, 1.0f, 0.5f, 1);
		CHECK(nt.end > 0.97f && ny.end < 0.03f, "uncalibrated: tilted 35 deg %.2f (should stay), turned 40 %.2f",
		      nt.end, ny.end);
		printf("ok: lens calibration learns the head (side %.3f vert %.3f eyes %.3f tilt %.3f): locked on at its own "
		       "pose (%.3f), lets go turned 30 deg either way (%.3f, %.3f); uncalibrated, only turns let go\n",
		       res.head.side, res.head.vert, res.head.eyes, res.head.tilt, st.end, r.end, l.end);
	}
}
