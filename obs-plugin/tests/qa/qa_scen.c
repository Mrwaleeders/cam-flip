/* Scenarios for qa_head.c (included by it). */

static script SC;
static trace TR;

static gen gen_default(double fps)
{
	gen G = {.fps = fps, .kp = 0.6, .ky = 0.3, .tau = 0.35, .up_drop = 0.005, .lo_rise = 0.012};
	return G;
}

/* Lens for `secs`, everything at rest. */
static void at_lens(script *sc, gen *G, head_st *h, double secs, int phase, int sacc)
{
	const double kp = G->kp, ky = G->ky;
	G->kp = G->ky = 0;
	const double lens[3] = {0, 0, G->D};
	hold_tgt(sc, G, h, lens, secs, phase, sacc, 0);
	G->kp = kp;
	G->ky = ky;
}

/* Reading lines y0..y1 mm under the lens, `width` mm wide, nfix fixations of fix_s each. Phase 10 + line. */
static void reading(script *sc, gen *G, head_st *h, const world *w, double y0, double y1, int lines, double width,
		    double fix_s, int nfix)
{
	for (int L = 0; L < lines; L++) {
		const double y = lines > 1 ? y0 + (y1 - y0) * L / (lines - 1) : y0;
		const double v = total_v(w, y);
		for (int k = 0; k < nfix; k++) {
			const double x = nfix > 1 ? -width / 2 + width * k / (nfix - 1) : 0;
			const double tg[3] = {x, y, w->D};
			hold_tgt(sc, G, h, tg, fix_s, 10 + L, 1, 1.0);
		}
		(void)v;
	}
}

/* Index range of a phase. */
static void phase_range(const script *sc, int ph, int *i0, int *i1)
{
	*i0 = -1;
	*i1 = -1;
	for (int i = 0; i < sc->n; i++)
		if (sc->s[i].phase == ph) {
			if (*i0 < 0)
				*i0 = i;
			*i1 = i + 1;
		}
}

/* ---- S1: reading the whole notes area */
typedef struct {
	float res_top, res_mid, res_bot;   /* mean G_seen on the top / middle / bottom line (px) */
	float gt_top, gt_bot;              /* G_true there */
	float res_min, res_max;            /* over all reading fixations (after 0.2 s) */
	float keep_bot;                    /* lift / G_true on the bottom line */
	steps st;
	float land_worst;                  /* s: worst landing time after a line change */
	float over_back;                   /* back at the lens: most the seen eye goes above it (px) */
	float left_back;                   /* lift left 1.5 s after back at the lens */
	float lid_mean, low_mean, lift_mean;
	float side_ratio;                  /* sd of the seen sideways sweep / the true one, reading fixations */
} s1_res;

static s1_res s1_reading(world w, calib_opts co, gen G, unsigned long long sd, int verbose)
{
	s1_res r;
	memset(&r, 0, sizeof(r));
	const cf_gaze_params p = calibrate(&w, &co);
	SC.n = 0;
	head_st h = {0, 0};
	G.t = 0;
	G.D = w.D;
	at_lens(&SC, &G, &h, 2.0, 0, 0);
	const int lines = 12;
	reading(&SC, &G, &h, &w, 50, 150, lines, 175, 0.28, 6);
	at_lens(&SC, &G, &h, 2.5, 2, 1);
	run(&w, &p, &SC, G.fps, sd, &TR);
	const int skip = (int)lround(0.15 * G.fps);
	r.res_min = 1e9f;
	r.res_max = -1e9f;
	float land_worst = 0;
	double ht1 = 0, ht2 = 0, hs1 = 0, hs2 = 0;
	int hc = 0;
	for (int L = 0; L < lines; L++) {
		int i0, i1;
		phase_range(&SC, 10 + L, &i0, &i1);
		/* fixations: skip the first `skip` frames after each saccade */
		double s = 0, sg = 0, sl = 0, sli = 0, slo = 0;
		int c = 0;
		for (int i = i0; i < i1; i++) {
			if (near_sacc(&SC, i, skip))
				continue;
			s += TR.gs[i];
			sg += TR.gt[i];
			sl += TR.lift[i];
			sli += TR.lid[i];
			slo += TR.low[i];
			c++;
			ht1 += TR.ht[i];
			ht2 += (double)TR.ht[i] * TR.ht[i];
			hs1 += TR.hs[i];
			hs2 += (double)TR.hs[i] * TR.hs[i];
			hc++;
			r.res_min = fminf(r.res_min, TR.gs[i]);
			r.res_max = fmaxf(r.res_max, TR.gs[i]);
		}
		const float m = c ? (float)(s / c) : 0, mg = c ? (float)(sg / c) : 0;
		if (L == 0) {
			r.res_top = m;
			r.gt_top = mg;
		}
		if (L == lines / 2)
			r.res_mid = m;
		if (L == lines - 1) {
			r.res_bot = m;
			r.gt_bot = mg;
			r.keep_bot = mg > 0 ? (float)(sl / c) / mg : 0;
		}
		r.lift_mean += (float)(sl / c) / lines;
		r.lid_mean += (float)(sli / c) / lines;
		r.low_mean += (float)(slo / c) / lines;
		/* landing: from the line's first frame to within 0.3 px of the lift the line settles at (its last 3
		 * fixations' mean, frames after `skip`), staying there for the first fixation */
		const int fix_n = (int)lround(0.28 * G.fps);
		double fin = 0;
		int fc = 0;
		for (int i = i1 - 3 * fix_n; i < i1; i++)
			if (!near_sacc(&SC, i, skip)) {
				fin += TR.lift[i];
				fc++;
			}
		fin /= fc ? fc : 1;
		int land = -1;
		for (int i = i0; i < i0 + fix_n; i++)
			if (fabsf(TR.lift[i] - (float)fin) < 0.3f) {
				land = i - i0;
				break;
			}
		const float lt = land < 0 ? 0.28f : (float)(land / G.fps);
		land_worst = fmaxf(land_worst, lt);
		if (verbose)
			printf("      line %2d (%5.1f deg down, head %4.1f, eyes %4.1f): true %5.2f px, lift %5.2f, seen %+5.2f; "
			       "upper lid %.2f lower lid %.2f; lands in %.2f s\n",
			       L, total_v(&w, 50 + 100.0 * L / (lines - 1)), SC.s[i1 - 1].pitch, TR.ev[i1 - 1], mg,
			       (float)(sl / c), m, (float)(sli / c), (float)(slo / c), lt);
	}
	r.land_worst = land_worst;
	{
		const double vt = ht2 / hc - (ht1 / hc) * (ht1 / hc), vs = hs2 / hc - (hs1 / hc) * (hs1 / hc);
		r.side_ratio = vt > 0 ? (float)sqrt(fmax(vs, 0) / vt) : 0;
	}
	int i0, i1;
	phase_range(&SC, 10, &i0, &i1);
	int e0, e1;
	phase_range(&SC, 10 + lines - 1, &e0, &e1);
	r.st = step_stats(&SC, &TR, i0, e1, skip > 2 ? 2 : skip, G.fps);
	phase_range(&SC, 2, &i0, &i1);
	r.over_back = -minf_range(TR.gs, i0, i1);
	r.left_back = fabsf(TR.lift[i1 - 1]);
	return r;
}

static void print_s1(const char *name, const s1_res *r)
{
	printf("  %-38s seen top/mid/bot %+5.2f %+5.2f %+5.2f px (true %4.1f..%4.1f, bottom kept %3.0f%%), range %+5.2f..%+5.2f;"
	       " steps lift %.2f lid %.2f low %.2f side %.2f (rms %.3f, jerk %.2f); land %.2f s; back: %.2f above, %.2f left\n",
	       name, r->res_top, r->res_mid, r->res_bot, r->gt_top, r->gt_bot, 100 * r->keep_bot, r->res_min, r->res_max,
	       r->st.step_lift, r->st.step_lid, r->st.step_low, r->st.step_side, r->st.rms_lift, r->st.jerk, r->land_worst,
	       r->over_back, r->left_back);
}

/* ---- S2: eyes on the lens, head tilting down */
typedef struct {
	float lift_end, seen_end, over; /* lift held at the end, G_seen there, most above the lens (px) */
	float pitch_read;
} s2_res;

static s2_res s2_lens_head_down(world w, calib_opts co, double pitch, double ramp_s, double fps, unsigned long long sd)
{
	s2_res r = {0};
	const cf_gaze_params p = calibrate(&w, &co);
	SC.n = 0;
	const double dt = 1.0 / fps;
	for (int i = 0; i < (int)(2 * fps); i++) {
		st s = {.tgt = {0, 0, w.D}};
		add(&SC, &s, 0);
	}
	const int nr = (int)lround(ramp_s * fps);
	for (int i = 0; i < nr + (int)(3 * fps); i++) {
		const double k = i < nr ? (double)(i + 1) / nr : 1.0;
		const double pk = pitch * k * k * (3 - 2 * k);
		st s = {.pitch = pk, .tgt = {0, 0, w.D}, .phase = 1};
		add(&SC, &s, 0);
	}
	(void)dt;
	run(&w, &p, &SC, fps, sd, &TR);
	const int n = SC.n;
	r.lift_end = mean_range(TR.lift, n - (int)fps, n);
	r.seen_end = mean_range(TR.gs, n - (int)fps, n);
	r.over = -minf_range(TR.gs, 0, n);
	r.pitch_read = TR.pitch_meas[n - 1];
	return r;
}

/* ---- S3: from reading to somewhere else and back (keyboard, side), eyes first or head first */
typedef struct {
	float wrong_v, wrong_h;   /* largest frame where the seen eye moved against the real gaze (px) */
	float step_away;          /* largest step of the lift while away, outside saccades (per 1/30 s) */
	float lift_away_end;      /* lift left at the end of the away phase */
	float relift;             /* largest rise of the lift while away after it first fell under half (px) */
	float lift_at_jump;       /* lift on the first away frame (saccade) and before it */
	float lift_before;
	float back_over, back_land; /* back on the notes: most above, and s to land within 0.3 px */
	float min_seen_away;      /* most the seen eye was above the lens while away */
	float side_away_end;
} s3_res;

typedef struct {
	double tgt[3];       /* away target (mm) */
	double head_p, head_y; /* where the head ends up */
	double head_tau;
	int head_first;      /* 1: the head moves first (eyes still in the head), then the eyes jump */
	double away_s;
	double lid_extra;    /* lids drop extra at the target (eye widths) */
} away_spec;

static s3_res s3_away(world w, calib_opts co, gen G, const away_spec *a, unsigned long long sd, int verbose)
{
	s3_res r = {0};
	const cf_gaze_params p = calibrate(&w, &co);
	SC.n = 0;
	head_st h = {0, 0};
	G.t = 0;
	G.D = w.D;
	at_lens(&SC, &G, &h, 1.5, 0, 0);
	const double v_read = total_v(&w, 100);
	const double notes[3] = {0, 100, w.D};
	hold_tgt(&SC, &G, &h, notes, 2.5, 1, 1, 1.0);
	const double dt = 1.0 / G.fps;
	const int na = (int)lround(a->away_s * G.fps);
	/* the eyes riding along in the head while it moves first: the target they would see (eyes fixed in the
	 * head at their reading angles) */
	double ev0, eh0;
	eye_angles(&w, &SC.s[SC.n - 1], 0, notes, &ev0, &eh0);
	int jumped = 0;
	for (int i = 0; i < na; i++) {
		st s = SC.s[SC.n - 1];
		s.phase = 3;
		int sacc = 0;
		if (a->head_first) {
			/* head ramps over 0.3 s first, the eyes ride along in the head; then they jump */
			const double k = fmin((i + 1) * dt / 0.3, 1.0);
			const double kk = k * k * (3 - 2 * k);
			h.pitch = 0.6 * v_read + kk * (a->head_p - 0.6 * v_read);
			h.yaw = kk * a->head_y;
			s.pitch = h.pitch;
			s.yaw = h.yaw;
			if (!jumped && i * dt >= 0.3) {
				jumped = 1;
				sacc = 1;
			}
			if (jumped) {
				memcpy(s.tgt, a->tgt, sizeof(s.tgt));
			} else { /* a point straight along the eyes' line in the head, far off */
				const double dh[3] = {sin(eh0 * D2R) * 1000, sin(ev0 * D2R) * cos(eh0 * D2R) * 1000,
						      cos(ev0 * D2R) * cos(eh0 * D2R) * 1000};
				double dw[3], C[3], Cw[3] = {0, 0, 0};
				vrot(&s, dh, dw, 0);
				C[0] = -0.5 * IOD;
				C[1] = 0;
				C[2] = 6 - IRIS_R;
				rot(&w, &s, C, Cw);
				for (int c = 0; c < 3; c++)
					s.tgt[c] = Cw[c] + dw[c];
			}
		} else {
			/* eyes jump at once; the head follows with lag, the eyes turning back in the head */
			head_step(&h, a->head_p, a->head_y, a->head_tau, dt);
			s.pitch = h.pitch;
			s.yaw = h.yaw;
			memcpy(s.tgt, a->tgt, sizeof(s.tgt));
			sacc = i == 0;
		}
		s.up_drop = 0.005 + a->lid_extra;
		s.lo_rise = 0.012;
		add(&SC, &s, sacc);
	}
	/* back to the notes: eyes jump, head follows back */
	for (int i = 0; i < (int)(2.5 * G.fps); i++) {
		head_step(&h, 0.6 * v_read, 0, 0.3, dt);
		st s = {.pitch = h.pitch, .yaw = h.yaw, .tgt = {0, 100, w.D}, .up_drop = 0.005, .lo_rise = 0.012, .phase = 4};
		add(&SC, &s, i == 0);
	}
	run(&w, &p, &SC, G.fps, sd, &TR);
	int i0, i1;
	phase_range(&SC, 3, &i0, &i1);
	r.lift_before = TR.lift[i0 - 1];
	r.lift_at_jump = TR.lift[i0];
	float wv = 0, wh = 0, step = 0, relift = 0, low_seen = 1e9f;
	int fell = 0;
	float min_lift = 1e9f;
	for (int i = i0; i < i1; i++) {
		const float dgt = TR.gt[i] - TR.gt[i - 1], dgs = TR.gs[i] - TR.gs[i - 1];
		/* the real gaze moving down (or still): the seen one must not move up */
		if (dgt >= -0.05f && dgs < 0)
			wv = fmaxf(wv, -dgs);
		if (dgt <= 0.05f && dgs > 0)
			wv = fmaxf(wv, dgs);
		const float dht = TR.ht[i] - TR.ht[i - 1], dhs = TR.hs[i] - TR.hs[i - 1];
		if (fabs(a->tgt[0]) > 0) {
			if (a->tgt[0] > 0 ? (dht >= -0.05f && dhs < 0) : (dht <= 0.05f && dhs > 0))
				wh = fmaxf(wh, fabsf(dhs));
		}
		if (!SC.sacc[i])
			step = fmaxf(step, fabsf(TR.lift[i] - TR.lift[i - 1]) * (float)(G.fps / 30.0));
		if (TR.lift[i] < 0.5f * r.lift_before)
			fell = 1;
		min_lift = fminf(min_lift, TR.lift[i]);
		if (fell)
			relift = fmaxf(relift, TR.lift[i] - min_lift);
		low_seen = fminf(low_seen, TR.gs[i]);
	}
	r.wrong_v = wv;
	r.wrong_h = wh;
	r.step_away = step;
	r.lift_away_end = TR.lift[i1 - 1];
	r.side_away_end = TR.side[i1 - 1];
	r.relift = relift;
	r.min_seen_away = low_seen;
	int b0, b1;
	phase_range(&SC, 4, &b0, &b1);
	const float fin = mean_range(TR.lift, b1 - (int)G.fps, b1);
	r.back_over = -minf_range(TR.gs, b0, b1);
	r.back_land = -1;
	for (int i = b0; i < b1; i++)
		if (fabsf(TR.lift[i] - fin) < 0.3f) {
			r.back_land = (float)((i - b0) / G.fps);
			break;
		}
	if (verbose) {
		printf("      frame: true-down seen-down lift lid low side | head p/y | eyes v/h | factor lock\n");
		for (int i = i0 - 2; i < i1 && i < i0 + (int)(1.6 * G.fps); i++)
			printf("      %4d %6.2f %6.2f %6.2f %5.2f %5.2f %6.2f | %5.1f %5.1f | %5.1f %5.1f | %.2f %.2f | dl %.3f ul %.3f "
			       "hv %.3f vo %.3f sl %.3f hh %.3f dn %.3f ff %.2f%s\n",
			       i - i0, TR.gt[i], TR.gs[i], TR.lift[i], TR.lid[i], TR.low[i], TR.side[i], SC.s[i].pitch,
			       SC.s[i].yaw, TR.ev[i], TR.eh[i], TR.factor[i], TR.lockf[i], TR.dl[i], TR.ul[i], TR.hv[i], TR.vo[i],
			       TR.sl[i], TR.hh[i], TR.down[i], TR.ff[i], SC.sacc[i] ? " *" : "");
	}
	return r;
}

static void print_s3(const char *name, const s3_res *r)
{
	printf("  %-44s lift %.2f -> %.2f on the jump, %.2f at the end (re-lift %.2f); steps %.2f; wrong-way v %.2f h %.2f;"
	       " seen min %+.2f; back: %.2f above, lands %.2f s\n",
	       name, r->lift_before, r->lift_at_jump, r->lift_away_end, r->relift, r->step_away, r->wrong_v, r->wrong_h,
	       r->min_seen_away, r->back_over, r->back_land);
}

/* ---- S4: holding a line while the head nods / turns / blinks; noise */
typedef struct {
	float lift_mean, lift_sd, seen_sd, rms_step, max_step, lid_rms, low_rms, max_lid_step, max_low_step;
	float blink_dev;
} s4_res;

static s4_res s4_hold(world w, calib_opts co, gen G, double secs, int blinks, unsigned long long sd)
{
	s4_res r = {0};
	const cf_gaze_params p = calibrate(&w, &co);
	SC.n = 0;
	head_st h = {0, 0};
	G.t = 0;
	G.D = w.D;
	at_lens(&SC, &G, &h, 1.5, 0, 0);
	const double notes[3] = {0, 100, w.D};
	hold_tgt(&SC, &G, &h, notes, 1.5, 1, 1, 1.0);
	const int start = SC.n;
	hold_tgt(&SC, &G, &h, notes, secs, 2, 0, 1.0);
	if (blinks) {
		/* blink every 1.5 s: lids 0.4 0.9 0.95 0.9 0.6 0.25 0 over 7 frames at 30 fps, the pupil point dragged */
		static const double prof[7] = {0.4, 0.9, 0.95, 0.9, 0.6, 0.25, 0.0};
		const double k = G.fps / 30.0;
		for (int b = start + (int)(0.5 * G.fps); b + 12 < SC.n; b += (int)(1.5 * G.fps))
			for (int j = 0; j < (int)(7 * k); j++) {
				const double sh = prof[(int)(j / k)];
				SC.s[b + j].shut = sh;
				SC.s[b + j].drag = 0.04 * sh;
			}
	}
	run(&w, &p, &SC, G.fps, sd, &TR);
	const int i0 = start + (int)(0.5 * G.fps), i1 = SC.n;
	r.lift_mean = mean_range(TR.lift, i0, i1);
	double s2 = 0, sg2 = 0, ms = 0, sg = 0, d2 = 0, dl2 = 0, db2 = 0;
	float mx = 0, mxl = 0, mxb = 0;
	for (int i = i0; i < i1; i++) {
		s2 += (TR.lift[i] - r.lift_mean) * (TR.lift[i] - r.lift_mean);
		sg += TR.gs[i];
		sg2 += TR.gs[i] * TR.gs[i];
		const float d = TR.lift[i] - TR.lift[i - 1];
		d2 += d * d;
		mx = fmaxf(mx, fabsf(d));
		dl2 += (TR.lid[i] - TR.lid[i - 1]) * (TR.lid[i] - TR.lid[i - 1]);
		db2 += (TR.low[i] - TR.low[i - 1]) * (TR.low[i] - TR.low[i - 1]);
		mxl = fmaxf(mxl, fabsf(TR.lid[i] - TR.lid[i - 1]));
		mxb = fmaxf(mxb, fabsf(TR.low[i] - TR.low[i - 1]));
		ms = fmax(ms, fabs(TR.lift[i] - r.lift_mean));
	}
	const int c = i1 - i0;
	const float k = (float)(G.fps / 30.0);
	r.lift_sd = (float)sqrt(s2 / c);
	r.seen_sd = (float)sqrt(fmax(sg2 / c - (sg / c) * (sg / c), 0));
	r.rms_step = (float)sqrt(d2 / c) * k;
	r.max_step = mx * k;
	r.lid_rms = (float)sqrt(dl2 / c) * k;
	r.low_rms = (float)sqrt(db2 / c) * k;
	r.max_lid_step = mxl * k;
	r.max_low_step = mxb * k;
	r.blink_dev = (float)ms;
	return r;
}

static void print_s4(const char *name, const s4_res *r)
{
	printf("  %-46s lift %.2f px, sd %.3f (seen sd %.3f); step rms %.3f max %.3f; lid rms %.3f max %.3f; low rms %.3f "
	       "max %.3f; most off %.2f\n",
	       name, r->lift_mean, r->lift_sd, r->seen_sd, r->rms_step, r->max_step, r->lid_rms, r->max_lid_step, r->low_rms,
	       r->max_low_step, r->blink_dev);
}

/* ---- Warp checks with the gaze model's own eyes */
static void warp_checks(void)
{
	printf("\nW. Warp with gaze.c's eye sizing (75 x 23 px opening, 8-point outline)\n");
	/* Build eyes as emit() would: ellipse from the outline, shifts as given. */
	const float w = 75, hh = 23;
	const float lifts[6] = {1.5f, 3.0f, 4.7f, 6.0f, 7.5f, 9.4f};
	for (int k = 0; k < 6; k++) {
		for (int withlow = 0; withlow < 2; withlow++) {
			cf_eye e;
			memset(&e, 0, sizeof(e));
			e.cx = 400.3f;
			e.cy = 300.6f;
			for (int i = 0; i < 8; i++) {
				const float a = 6.2831853f * i / 8.0f;
				e.aperture[i][0] = e.cx + 0.5f * w * cosf(a);
				e.aperture[i][1] = e.cy + 0.5f * hh * sinf(a);
			}
			e.aperture_count = 8;
			float ext_y = 0.5f * hh;
			e.rx = fmaxf(0.75f * w, 0.5f * w + 0.25f * w);
			e.ry = fmaxf(fmaxf(1.6f * hh, 0.5f * w), ext_y + 1.1f * hh);
			const float cap = fminf(0.38f * e.ry, 0.5f * (e.ry - ext_y));
			const float a = fminf(lifts[k], cap);
			e.sy = -a;
			/* upper lid: LID_SHIFT x lift + the lid restoration (0.01 eye widths = 0.75 px) */
			e.lid_sy = -(0.45f * a + fminf(0.75f, 0.5f * cap));
			e.low_sy = withlow ? fminf(0.75f, 0.25f * cap) : 0.0f;
			/* field on a 1 px lattice over the ellipse */
			enum { NX = 140, NY = 100 };
			static float fx[NX * NY], fy[NX * NY];
			const float x0 = e.cx - 70, y0 = e.cy - 50;
			cf_warp_debug_grid(&e, x0, y0, 1.0f, NX, NY, fx, fy);
			float det_min = 9, jyy_min = 9;
			for (int j = 1; j < NY - 1; j++)
				for (int i = 1; i < NX - 1; i++) {
					const float jxx = 1 - 0.5f * (fx[j * NX + i + 1] - fx[j * NX + i - 1]);
					const float jxy = -0.5f * (fx[(j + 1) * NX + i] - fx[(j - 1) * NX + i]);
					const float jyx = -0.5f * (fy[j * NX + i + 1] - fy[j * NX + i - 1]);
					const float jyy = 1 - 0.5f * (fy[(j + 1) * NX + i] - fy[(j - 1) * NX + i]);
					det_min = fminf(det_min, jxx * jyy - jxy * jyx);
					jyy_min = fminf(jyy_min, jyy);
				}
			/* where the lower rim (just under the bottom outline point) and the upper lid edge end up:
			 * output(p) = input(p - d(p)); find p whose source is the rim, along the centre column */
			float rim_moved = 0, top_moved = 0, iris_moved = 0;
			const int ic = (int)lroundf(e.cx - x0);
			const float rim_src = e.cy + 0.5f * hh, top_src = e.cy - 0.5f * hh, iris_src = e.cy;
			for (int pass = 0; pass < 3; pass++) {
				const float src = pass == 0 ? rim_src : pass == 1 ? top_src : iris_src;
				/* scan output rows (finely, interpolating the field) for p - d(p) = src */
				float best = 0;
				for (float py = e.cy - 30; py < e.cy + 30; py += 0.01f) {
					const float gy = py - y0;
					const int j = (int)floorf(gy);
					if (j < 0 || j + 1 >= NY)
						continue;
					const float fr = gy - j;
					const float d = fy[j * NX + ic] * (1 - fr) + fy[(j + 1) * NX + ic] * fr;
					if (fabsf(py - d - src) < 0.006f) {
						best = py - src;
						break;
					}
				}
				if (pass == 0)
					rim_moved = best;
				else if (pass == 1)
					top_moved = best;
				else
					iris_moved = best;
			}
			/* untouched outside the ellipse */
			float outside = 0;
			for (int j = 0; j < NY; j++)
				for (int i = 0; i < NX; i++) {
					const float px = x0 + i, py = y0 + j;
					const float u = (px - e.cx) / e.rx, v = (py - e.cy) / e.ry;
					if (u * u + v * v >= 1.0f)
						outside = fmaxf(outside, fmaxf(fabsf(fx[j * NX + i]), fabsf(fy[j * NX + i])));
				}
			printf("  iris lift %4.1f (cap %.1f), upper lid %.2f, lower lid %.2f: centre +%.2f, upper edge +%.2f, lower rim "
			       "%+.2f (up = +): opening %+.2f px vs unwarped; min Jacobian %.3f (vertical %.3f); outside %.4f\n",
			       a, cap, -e.lid_sy, e.low_sy, -iris_moved, -top_moved, -rim_moved, (-top_moved) - (-rim_moved),
			       det_min, jyy_min, outside);
		}
	}
}

/* Prints a stretch of the last run. */
static void dump(int i0, int i1)
{
	printf("      frame: true-down seen-down lift lid low side | head p (meas) y | eyes v h | factor lock\n");
	for (int i = i0 < 0 ? 0 : i0; i < i1 && i < SC.n; i++)
		printf("      %4d %6.2f %6.2f %6.2f %5.2f %5.2f %6.2f | %5.1f (%5.1f) %5.1f | %5.1f %5.1f | %.2f %.2f%s\n", i,
		       TR.gt[i], TR.gs[i], TR.lift[i], TR.lid[i], TR.low[i], TR.side[i], SC.s[i].pitch, TR.pitch_meas[i],
		       SC.s[i].yaw, TR.ev[i], TR.eh[i], TR.factor[i], TR.lockf[i], SC.sacc[i] ? " *" : "");
}

static int want(const char *sel, char c) { return !sel || strchr(sel, c) != NULL; }

int main(int argc, char **argv)
{
	const char *sel = argc > 1 ? argv[1] : NULL;
	const int verbose = argc > 2;
	const calib_opts co0 = {.notes_y = 100, .kp = 0.6, .strength = 1.0, .fps = 30};
	const double dists[2] = {500, 700};

	if (want(sel, 'a')) {
		printf("S1. Reading 12 lines 5-15 cm under the lens (175 mm wide, 6 fixations of 0.28 s a line), head 0.6 of "
		       "the pitch (lag 0.35 s), 0.3 of the turn; noise eye 0.3 nose 0.3 pupil 0.3 px\n");
		for (int di = 0; di < 2; di++) {
			world w = W0(dists[di]);
			printf("  -- %.1f m (eye %.0f px)\n", dists[di] / 1000, 29 * w.focal / dists[di]);
			s1_res r = s1_reading(w, co0, gen_default(30), 1, verbose);
			print_s1("30 fps, 100%", &r);
			if (verbose && di == 0) {
				int a0, a1;
				phase_range(&SC, 2, &a0, &a1);
				dump(a0 - 3, a0 + 20);
				phase_range(&SC, 19, &a0, &a1);
				dump(a0 - 2, a0 + 16);
			}
		}
	}
	if (want(sel, 'b')) {
		printf("\nS2. Eyes on the lens, head tilting down (0.6 m); iris_z 0 = gaze.c's eye model, 6 = iris 6 mm in front "
		       "of the corners (rotating eyeball)\n");
		const double pitches[4] = {3, 6, 10, 15};
		for (int m = 0; m < 3; m++) {
			world w = W0(600);
			w.iris_z = m == 0 ? 0 : m == 1 ? 6 : 9;
			for (int k = 0; k < 4; k++)
				for (int q = 0; q < 2; q++) {
					const s2_res r = s2_lens_head_down(w, co0, pitches[k], q ? 0.3 : 1.0, 30, 3 + k);
					printf("  iris_z %.0f, head %4.1f deg in %.1f s (reads %4.1f): lift held %5.2f px, seen %+5.2f "
					       "(above the lens up to %.2f)\n",
					       w.iris_z, pitches[k], q ? 0.3 : 1.0, r.pitch_read, r.lift_end, r.seen_end, r.over);
					if (verbose && m == 0 && k == 2 && q == 0)
						dump(55, 100);
				}
		}
		printf("  reading with the rotating-eyeball model (iris_z 6):\n");
		for (int di = 0; di < 2; di++) {
			world w = W0(dists[di]);
			w.iris_z = 6;
			s1_res r = s1_reading(w, co0, gen_default(30), 1, 0);
			char nm[64];
			snprintf(nm, sizeof(nm), "%.1f m, iris_z 6", dists[di] / 1000);
			print_s1(nm, &r);
		}
	}
	if (want(sel, 'c')) {
		printf("\nS3. From reading (10 cm line) away and back, 0.6 m, 30 fps\n");
		const world w = W0(600);
		const away_spec specs[] = {
			/* keyboard 30 cm down, 15 cm nearer: ~34 deg */
			{{0, 300, 450}, 22, 0, 0.25, 0, 2.0, 0.03},
			{{0, 300, 450}, 28, 0, 0.25, 0, 2.0, 0.03},
			{{0, 300, 450}, 22, 0, 0.25, 1, 2.0, 0.03},
			/* desk / phone 20 cm down: ~18 deg, head does most */
			{{0, 200, 600}, 13, 0, 0.25, 0, 2.0, 0.015},
			{{0, 250, 550}, 20, 0, 0.30, 0, 2.0, 0.02},
			/* side: 30 deg to the image's right and left at lens height, head 20 */
			{{346, 0, 600}, 0, 20, 0.25, 0, 2.0, 0},
			{{-346, 0, 600}, 0, -20, 0.25, 0, 2.0, 0},
			{{346, 0, 600}, 0, 20, 0.25, 1, 2.0, 0},
			/* side, 15 deg, eyes only */
			{{160, 0, 600}, 0, 0, 0.25, 0, 2.0, 0},
		};
		const char *names[] = {"keyboard 34 deg, eyes first, head -> 22",  "keyboard 34 deg, eyes first, head -> 28",
				       "keyboard 34 deg, head first -> 22",        "desk 18 deg, eyes first, head -> 13",
				       "desk 24 deg, eyes first, head -> 20",      "side 30 deg right, eyes first, head 20",
				       "side 30 deg left, eyes first, head 20",    "side 30 deg right, head first 20",
				       "side 15 deg right, eyes only"};
		for (int k = 0; k < (int)(sizeof(specs) / sizeof(specs[0])); k++) {
			const s3_res r = s3_away(w, co0, gen_default(30), &specs[k], 11 + k, verbose && strchr(argv[2], '0' + k));
			print_s3(names[k], &r);
		}
	}
	if (want(sel, 'd')) {
		printf("\nS4. Holding the 10 cm line 6 s: noise, nods, turns, blinks (0.6 m)\n");
		const double noises[4] = {0.2, 0.4, 0.6, 0.8};
		for (int k = 0; k < 4; k++)
			for (int old = 0; old < 2; old++) {
				world w = W0(600);
				w.sd_nose = noises[k];
				w.sd_eye = noises[k];
				calib_opts co = co0;
				co.old_head = old;
				co.old_lids = old;
				const s4_res r = s4_hold(w, co, gen_default(30), 6, 0, 21 + k);
				char nm[80];
				snprintf(nm, sizeof(nm), "landmarks %.1f px%s", noises[k], old ? " (no head/lid calib)" : "");
				print_s4(nm, &r);
			}
		for (int k = 0; k < 4; k++) {
			world w = W0(600);
			w.sd_nose = noises[k];
			w.sd_eye = 0.3;
			const s4_res r = s4_hold(w, co0, gen_default(30), 6, 0, 31 + k);
			char nm[80];
			snprintf(nm, sizeof(nm), "nose %.1f px, eyes 0.3", noises[k]);
			print_s4(nm, &r);
		}
		{
			world w = W0(600);
			gen G = gen_default(30);
			G.nod_amp = 3;
			G.nod_hz = 2;
			s4_res r = s4_hold(w, co0, G, 4, 0, 41);
			print_s4("nod +-3 deg 2 Hz (eyes hold the line)", &r);
			calib_opts cold = co0;
			cold.old_head = cold.old_lids = 1;
			r = s4_hold(w, cold, G, 4, 0, 41);
			print_s4("nod +-3 deg 2 Hz, no head/lid calib", &r);
			G.nod_amp = 6;
			G.nod_hz = 1;
			r = s4_hold(w, co0, G, 4, 0, 42);
			print_s4("nod +-6 deg 1 Hz", &r);
			r = s4_hold(w, cold, G, 4, 0, 42);
			print_s4("nod +-6 deg 1 Hz, no head/lid calib", &r);
			G = gen_default(30);
			G.turn_amp = 8;
			G.turn_hz = 1;
			r = s4_hold(w, co0, G, 4, 0, 43);
			print_s4("turn +-8 deg 1 Hz", &r);
			G = gen_default(30);
			r = s4_hold(w, co0, G, 6, 1, 44);
			print_s4("blinks every 1.5 s (pupil dragged 0.04)", &r);
		}
	}
	if (want(sel, 'e')) {
		printf("\nS5. Reading (0.5 m) and the keyboard at 15/25/30 fps, strength 100/200%%, old calibrations\n");
		const double fpss[3] = {15, 25, 30};
		const away_spec kb = {{0, 300, 450}, 22, 0, 0.25, 0, 2.0, 0.03};
		for (int f = 0; f < 3; f++)
			for (int s = 0; s < 2; s++) {
				world w = W0(500);
				calib_opts co = co0;
				co.fps = fpss[f];
				co.strength = s ? 2.0 : 1.0;
				s1_res r = s1_reading(w, co, gen_default(fpss[f]), 51 + f, 0);
				char nm[64];
				snprintf(nm, sizeof(nm), "reading %2.0f fps %3.0f%%", fpss[f], 100 * co.strength);
				print_s1(nm, &r);
				const s3_res k = s3_away(w, co, gen_default(fpss[f]), &kb, 61 + f, 0);
				snprintf(nm, sizeof(nm), "keyboard %2.0f fps %3.0f%%", fpss[f], 100 * co.strength);
				print_s3(nm, &k);
			}
		for (int o = 1; o < 4; o++) {
			world w = W0(500);
			calib_opts co = co0;
			co.old_lids = o & 1;
			co.old_head = (o >> 1) & 1;
			s1_res r = s1_reading(w, co, gen_default(30), 71, 0);
			const char *nm = o == 1 ? "old calib: no lid baselines" : o == 2 ? "old calib: no head ref" : "old calib: neither";
			print_s1(nm, &r);
		}
	}
	if (want(sel, 'g')) {
		printf("\nS6. Refined pupil under-reading the eyes' turn (meas_gain), 0.6 m\n");
		const double gains[2] = {0.5, 0.7};
		for (int k = 0; k < 2; k++)
			for (int s = 0; s < 2; s++) {
				world w = W0(600);
				w.meas_gain = gains[k];
				calib_opts co = co0;
				co.strength = s ? 1.0 / gains[k] : 1.0;
				const s2_res a = s2_lens_head_down(w, co, 6, 1.0, 30, 5);
				const s2_res b = s2_lens_head_down(w, co, 10, 1.0, 30, 6);
				s1_res r = s1_reading(w, co, gen_default(30), 7, 0);
				char nm[80];
				snprintf(nm, sizeof(nm), "gain %.1f, strength %3.0f%%", gains[k], 100 * co.strength);
				printf("  %s: eyes on the lens, head 6 / 10 deg down: lift held %.2f / %.2f px (above the lens)\n", nm,
				       a.lift_end, b.lift_end);
				print_s1(nm, &r);
			}
	}
	if (sel && want(sel, 'h')) {
		printf("\nS3b. Side looks at lens height from reading: what sets off the sideways damping (0.6 m)\n");
		const double xs[4] = {346, -346, 250, -250};
		const double yaws[3] = {20, 10, 0};
		for (int v = 0; v < 4; v++)
			for (int k = 0; k < 4; k++)
				for (int y = 0; y < 3; y++) {
					world w = W0(600);
					if (v & 1)
						w.sd_eye = w.sd_nose = w.sd_pupil = w.sd_vis = 0;
					if (v & 2)
						w.vis_gain = 1.0;
					const away_spec a = {{xs[k], 0, 600}, 0, xs[k] > 0 ? yaws[y] : -yaws[y], 0.25, 0, 2.0, 0};
					const s3_res r = s3_away(w, co0, gen_default(30), &a, 11 + k, 0);
					char nm[96];
					snprintf(nm, sizeof(nm), "x %+4.0f, head yaw %+3.0f%s%s", xs[k], a.head_y, v & 1 ? ", no noise" : "",
						 v & 2 ? ", vision gain 1" : "");
					print_s3(nm, &r);
				}
	}
	if (sel && want(sel, 'k')) {
		printf("\nS1b. Reading (0.5 m) with the head tilted more than the gaze (eyes up in the head on the top lines)\n");
		const double kps[4] = {0.6, 1.0, 1.4, 2.0};
		for (int k = 0; k < 4; k++) {
			const world w = W0(500);
			gen G = gen_default(30);
			G.kp = kps[k];
			calib_opts co = co0;
			co.kp = kps[k];
			const s1_res r = s1_reading(w, co, G, 1, 0);
			char nm[64];
			snprintf(nm, sizeof(nm), "head %.1f x the gaze's tilt", kps[k]);
			print_s1(nm, &r);
			printf("    sideways sweep seen/true %.2f; mean lift %.2f, upper lid %.2f, lower lid %.2f px\n", r.side_ratio,
			       r.lift_mean, r.lid_mean, r.low_mean);
		}
	}
	if (sel && want(sel, 'i')) {
		printf("\nS3c. Side looks at lens height from reading, 40 noise seeds each (0.6 m): wrong-way sideways px\n");
		const double xs[4] = {346, -346, 250, -250}, ys[4] = {20, -20, 0, 0};
		for (int k = 0; k < 8; k++) {
			const world w = W0(600);
			calib_opts co = co0;
			co.old_head = k >= 4;
			const away_spec a = {{xs[k & 3], 0, 600}, 0, ys[k & 3], 0.25, 0, 2.0, 0};
			float worst = 0, worst_v = 0, sum = 0;
			int over1 = 0, over3 = 0, worst_sd = 0;
			for (int sd = 0; sd < 40; sd++) {
				const s3_res r = s3_away(w, co, gen_default(30), &a, 1000 + 37 * sd, 0);
				if (r.wrong_h > worst)
					worst_sd = sd;
				worst = fmaxf(worst, r.wrong_h);
				worst_v = fmaxf(worst_v, r.wrong_v);
				sum += r.wrong_h;
				over1 += r.wrong_h > 1.0f;
				over3 += r.wrong_h > 3.0f;
			}
			printf("  x %+4.0f, head yaw %+3.0f%s: wrong-way h mean %.2f, worst %.2f, runs over 1 px %d, over 3 px %d of 40; "
			       "wrong-way v worst %.2f\n",
			       xs[k & 3], ys[k & 3], co.old_head ? " (no head ref)" : "", sum / 40, worst, over1, over3, worst_v);
			if (argc > 2 && strchr(argv[2], '0' + k))
				s3_away(w, co, gen_default(30), &a, 1000 + 37 * worst_sd, 1);
		}
	}
	if (want(sel, 'w'))
		warp_checks();
	return 0;
}
