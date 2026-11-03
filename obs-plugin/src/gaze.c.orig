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
#define LID_SHIFT 0.45f       /* upper lid follows the iris by this fraction (real lids track vertical gaze) */
#define AMOUNT_MAX_RY 0.38f   /* vertical iris shift limit, fraction of ry (well inside the fold limit) */
#define HSHIFT_MAX_RX 0.30f   /* sideways iris shift limit, fraction of rx */
#define READ_DAMP 0.65f       /* fraction of the sideways reading excursion cancelled at full strength */
#define SIDE_FADE_LO 0.30f    /* sideways distance from the notes centre (eye widths) where fading starts */
#define SIDE_FADE_HI 0.45f
#define FADE_IN_TAU 0.15      /* s: per stage of the fade in (10-90% in ~0.5 s) ... */
#define FADE_OUT_TAU 0.10     /* s: ... and out (~0.35 s): looking away, back, turning the head */
#define PREDICT_MAX_DT 0.10   /* s: never extrapolate further than this */
#define VEL_TAU 0.05         /* s: smoothing of the velocity used for prediction */
#define LOST_HOLD 0.20        /* s: a face missing for less than this (detector dropout) is held, not faded */
#define LOST_FADE_TAU 0.02    /* s: ... missing for longer: the correction lets go this fast */
#define ROLL_MIN_CUTOFF 1.0   /* Hz: head roll smoothing (One Euro) of the push direction when still */
#define ROLL_BETA 3.0         /* ... per rad/s of roll speed: a real head tilt is followed quickly */
#define RESYNC_JUMP 0.03f     /* eye widths: after a blink, a gaze change bigger than this is taken at once */
/* Vertical saccade detection. Changes of the raw glance below `lo` are landmark noise (smoothed),
 * changes above `hi` are a saccade (a glance to or from the notes) and are taken at once. Both
 * follow the measured landmark noise: as tight as clean tracking allows, wider when it is noisy. */
#define DOWN_NOISE_INIT 0.008f /* eye widths: mean frame-to-frame change assumed before measuring */
#define DOWN_NOISE_TAU 1.5     /* s: how fast the noise estimate adapts */
#define DOWN_LO_K 2.0f         /* lo = this x mean noise change ... */
#define DOWN_LO_MIN 0.01f      /* ... clamped to these */
#define DOWN_LO_MAX 0.03f
#define DOWN_HI_MIN 0.025f     /* hi = 2 lo, clamped to these */
#define DOWN_HI_MAX 0.05f
#define DOWN_CUTOFF 1.2       /* Hz: low-pass for noise-sized vertical changes */
#define GLANCE_PUPIL 0.02f    /* eye widths: lids dropping with the pupil moving down this much = a glance */
#define GLANCE_MIN_R 0.6f     /* ... but only while the lids are still this open (relative): lower is a blink */
#define GLANCE_BIG 0.06f      /* eye widths: a pupil drop this big (and at least the lid drop) is a glance even
			       * with the lids half shut (keyboard): far more than a lid can drag the point */
#define GLANCE_LID_RATIO 0.5f /* ... and only if the pupil fell at least this fraction of the lid drop: the
			       * lid follows a glance about 1:1, a pupil point dragged by a closing lid moves less */
#define HALF_ENTER 0.60f      /* relative openness: a quick drop below this is a half blink (held) ... */
#define HALF_DROP 0.12f       /* ... when the lids just fell at least this much (recent peak minus now) */
#define HALF_DECAY 1.2f       /* per second: how fast the recent peak forgets (slower drops, like a smile, never count) */
#define HALF_EXIT 0.80f       /* reopened above this (or close to where they were before) ends it */
#define HALF_MAX 0.50         /* s: lids that stay half shut this long are a squint: released, measured normally */
#define HALF_SLOW_TAU 0.08    /* s: smoothing of the openness level a half blink drops from */
#define HALF_STEP 0.06f       /* relative openness: a single-frame drop this big (below HALF_ENTER) starts one too */
#define LID_NOISE_INIT 0.02f  /* mean frame-to-frame change of the relative openness assumed before measuring */
#define LID_STEADY_STEP 0.05f /* relative openness per frame: lids moving more than this are not steady */
#define LID_STEADY_TIME 0.10  /* s: the lid cue needs the lids steady this long */
#define LID_STEADY_MAX 0.08f  /* ... the steadiness threshold widens with lid noise up to this, no further */
#define ROLLBACK_WIN 0.15     /* s: a blink or half blink undoes glance changes made this shortly before it ... */
#define ROLLBACK_PUPIL 0.03f  /* eye widths: ... and by more than a dragged pupil point (~1 px) moves */
#define ROLLBACK_SLIDE_MAX 0.08f /* eye widths: the pupil now this much past the rolled-back state is a real glance */
#define ROLLBACK_GLANCE 0.6f  /* ... by frames whose lids moved, unless the pupil moved at least this fraction
			       * of the lid move (the lid following a glance): a glance seen is kept */
#define EASE_STEP_PX 0.3f     /* px per 1/30 s: largest output change after a half-blink hold times out */
#define EASE_MIN 0.15         /* s: ... kept up at least this long (covers the lid cue waking up) */
#define LID_GATE 1.05f        /* lid cue only once the pupil already reads past the limit (x this) */
/* Outlier-robust gaze (see gate()). The refined pupil can jump between two spots ~10 px apart from one
 * frame to the next while the iris moves smoothly; read as saccades, every jump showed and the irises
 * shook. A change bigger than the noise band now shows at once only when something independent backs
 * it up; otherwise it has to persist, and a reading that comes straight back never shows. */
#define LR_TAU 2.0            /* s: learning of the usual left-minus-right reading and its noise */
#define LR_K 4.0f             /* the two eyes disagreeing by more than this x their usual noise ... */
#define LR_MIN 0.025f         /* ... (clamped to these, eye widths) is an outlier in one eye: the other eye's */
#define LR_MAX 0.07f          /* ... reading is used (the eyes move together, up/down and sideways) */
#define LR_NOISE_INIT 0.012f
#define VIS_FRAC 0.2f         /* Vision's own pupil point backs a change up when it moved the same way by this
			       * fraction of it (it is coarse: it under-reads glances) ... */
#define VIS_K 3.0f            /* ... and by more than this x its own frame-to-frame noise ... */
#define VIS_K_SMALL 4.5f      /* ... (this x for a change short of a saccade, which noisy refined readings
			       * reach now and then: both sensors' noise must not line up into a jump) ... */
#define VIS_MIN 0.008f        /* ... and at least this (eye widths: 0.7 px on an 86 px eye) */
#define VIS_NOISE_INIT 0.003f
#define LIDC_FRAC 0.3f        /* (no Vision points) the upper lid backs up a vertical change when it moved with
			       * it by this fraction of it ... */
#define LIDC_K 3.0f           /* ... and by more than this x its noise */
#define ANCHOR_TAU 0.3        /* s: between jumps, "where the backing-up sensors were" follows them this fast ... */
#define ANCHOR_HOLD 1.0       /* ... and Vision's point this slowly while it is well away (it moved) */
#define CLEAN_TIME 1.5        /* s: the measurement counts as clean this long after the last outlier */
#define PERSIST_LONG 5        /* frames an unbacked change must stay put while outliers are about */
#define VOTE_FRAMES 24.0f     /* frames: memory of the vote between two levels (see gate()) ... */
#define VOTE_SLOW 72.0f       /* ... and of the longer vote, which must agree ... */
#define VOTE_Z 1.0f           /* ... by this many standard deviations of chance (see vote()) */
#define VIS_AGREE 0.35f       /* a refined reading this far (x the move) from where Vision's point says the
			       * gaze went is not taken with the saccade: Vision's estimate is */
#define LANDING_FRAMES 3      /* frames after a jump that still count as landing it */
#define EYES_K 6.0f           /* (no Vision) both eyes back a change up only past this x the glance noise */
#define VOTE_RATIO 1.3f       /* the other level wins with this many times the accepted level's share ... */
#define VOTE_MIN 0.3f         /* ... and at least this share of the recent readings */
#define VOTE_NEW 0.1f         /* a reading at neither level replaces an other level weaker than this */
#define VIS_GAIN_INIT 1.5f    /* refined change per Vision change until learned ... */
#define VIS_GAIN_ERR_INIT 0.3f /* ... and its typical (log) error until measured */
#define VIS_OFF_TAU 2.0       /* s: memory of the refined-minus-Vision offset between changes (see gate()) */
#define REVERT_TIME 0.15      /* s: a jump undone this soon was an outlier */
#define SLEW_PX 0.3f          /* px per 1/30 s: fastest the shown gaze moves without a backed-up saccade */
#define SIDE_LO 0.02f         /* eye widths: sideways noise band (or 2x the noise, up to twice this) ... */
#define SIDE_HI 0.06f         /* ... and saccade size (or twice the band) */
#define SIDE_CUTOFF 3.0       /* Hz: low-pass for noise-sized sideways changes */
#define DOWN_OUT_CUTOFF 1.5   /* Hz: second low-pass stage on what the output shows, vertical ... */
#define SIDE_OUT_CUTOFF 2.0   /* ... and sideways (see slew()) */

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

static float clampf(float x, float lo, float hi)
{
	return x < lo ? lo : (x > hi ? hi : x);
}

/* ---- Outlier-robust gaze channels (the vertical glance and the sideways position) */

/* Combines the two eyes' readings of one channel. The eyes move together, so one eye disagreeing with
 * the other far more than usual is an outlier in that eye: the one further from where this frame's
 * reading should be (where the accepted value is, plus whatever Vision's point says moved). The
 * combined reading then comes from the other eye alone (plus their usual difference), blending over
 * from 70% of the outlier threshold so there is no switching edge. e[] gets each eye's deviation from
 * the accepted value. */
static float lr_combine(cf_robust *c, const cf_one_euro *f, float a, float b, int vis, float v, int learn,
			double t, double dt, float e[2], int *outlier)
{
	const float d = a - b - c->lr_bias;
	const float thr = clampf(LR_K * c->lr_noise, LR_MIN, LR_MAX);
	float r = 0.5f * (a + b);
	*outlier = 0;
	e[0] = e[1] = 0;
	if (f->ready) {
		e[0] = a - ((float)f->value + 0.5f * c->lr_bias);
		e[1] = b - ((float)f->value - 0.5f * c->lr_bias);
		float expect = 0;
		if (vis && c->have_v) {
			const float dv = v - c->v_anchor;
			if (fabsf(dv) >= fmaxf(VIS_K * c->v_noise, VIS_MIN))
				expect = dv * c->v_gain;
		}
		const float k = smoothstepf(0.7f * thr, thr, fabsf(d));
		if (k > 0) {
			const float one = fabsf(e[0] - expect) <= fabsf(e[1] - expect) ? a - 0.5f * c->lr_bias
										      : b + 0.5f * c->lr_bias;
			r += k * (one - r);
		}
		if (fabsf(d) > thr) {
			*outlier = 1;
			if (fabsf(d) > 1.5f * thr)
				c->out_t = t; /* far past what noise does: the measurement isn't clean */
		}
	}
	if (learn) {
		const float k = (float)(1.0 - exp(-dt / LR_TAU));
		c->lr_bias += k * clampf(d, -thr, thr); /* an outlier moves it no more than noise would */
		if (!*outlier)
			c->lr_noise += k * (fabsf(d) - c->lr_noise);
	}
	return r;
}

typedef struct {
	float lo, hi;           /* noise band; saccade size (eye widths) */
	double alpha;           /* low-pass weight for a change inside the noise band */
	int vis;                /* Vision's own pupil reading available this frame ... */
	float v;                /* ... and its value, in the channel's units */
	int lid;                /* the upper lid may back up changes (vertical channel) */
	float open, open_noise; /* mean eye openness and its frame-to-frame noise (eye widths) */
	float e[2];             /* each eye's own deviation from the accepted value (less their usual difference) */
	int lr_out;             /* one eye's reading was replaced this frame */
	float resync;           /* > 0 on the first frame after a blink: both eyes moving this much is enough */
	float eyes_min;         /* both eyes moving alike back a change up only from this size (far past noise) */
	float noise;            /* typical frame-to-frame change of the reading (landmark noise) */
} gate_in;

/* A new accepted level: no other levels yet, and the vote counts readings from now on (no head start:
 * if it was a wrong level, the readings since show it as soon as they can). */
static void new_level(cf_robust *c)
{
	c->have_alt[0] = c->have_alt[1] = 0;
	c->w_alt[0] = c->w_alt[1] = c->s_alt[0] = c->s_alt[1] = 0;
	c->w_acc = c->s_acc = 0;
}

/* The accepted level moved to offset o_new (see vote()) from o_old, Vision's point not explaining the move:
 * if that is a known other level, the two swap places (with their shares of the readings); otherwise the
 * old accepted level becomes an other level (in place of a weaker one), and the new one starts afresh. */
static void move_level(cf_robust *c, float o_old, float o_new, float tol)
{
	for (int j = 0; j < 2; j++)
		if (c->have_alt[j] && fabsf(c->alt[j] - o_new) <= tol) {
			const float w = c->w_alt[j], sw = c->s_alt[j];
			c->alt[j] = o_old;
			c->w_alt[j] = c->w_acc;
			c->s_alt[j] = c->s_acc;
			c->w_acc = w;
			c->s_acc = sw;
			return;
		}
	const int j = !c->have_alt[0] ? 0 : !c->have_alt[1] ? 1 : (c->s_alt[0] <= c->s_alt[1] ? 0 : 1);
	if (!c->have_alt[j] || c->s_alt[j] < c->s_acc) {
		c->alt[j] = o_old;
		c->have_alt[j] = 1;
		c->w_alt[j] = c->w_acc;
		c->s_alt[j] = c->s_acc;
	}
	c->w_acc = c->s_acc = 0;
}

/*
 * The vote between levels, for a refiner that flips between spots: which level did each eye's own reading
 * go to (half a vote each)? Each eye's reading, not the combined one: that leans towards the accepted
 * value when the eyes disagree, and would keep a wrong level winning. A reading near the accepted level
 * (or another known one) counts for it; one a saccade away from all of them starts a new level in place
 * of a weak one. Two other levels are kept, so readings split between two spots on either side of a wrong
 * accepted value still settle on one. Returns 1, with the winning level in *to, when another level got
 * clearly more of the readings, recently and over the longer memory (a short run of one level is how a
 * 60/40 flip looks now and then), and still gets them: it is where the eye is. The old accepted value
 * takes its place as another level.
 * The other levels are kept as offsets from u, where Vision's point says the gaze is (v_gain x its reading;
 * 0 without it), and the accepted level is at acc (where it is now, moved on by what Vision's point says
 * moved since it was taken). A refiner flipping between spots does so at the same offsets wherever the
 * eyes look, so with Vision the vote goes on across saccades: a glance doesn't wipe out what the readings
 * so far showed, and a wrong level (or a false switch, which a short run of flips can cause) is set right
 * from all of them.
 */
static int vote(cf_robust *c, float x0, float acc, float u, const gate_in *in, float *to)
{
	float v_acc = 0, v_alt[2] = {0, 0};
	/* Levels closer than a saccade are one spot: merged (an other level next to the accepted one is just
	 * the accepted one; two other levels side by side become one). */
	for (int j = 0; j < 2; j++)
		if (c->have_alt[j] && fabsf(c->alt[j] + u - acc) < in->hi)
			c->have_alt[j] = 0;
	if (c->have_alt[0] && c->have_alt[1] && fabsf(c->alt[0] - c->alt[1]) < in->hi) {
		const float w0 = c->s_alt[0] + 1e-6f, w1 = c->s_alt[1] + 1e-6f;
		c->alt[0] = (c->alt[0] * w0 + c->alt[1] * w1) / (w0 + w1);
		c->w_alt[0] += c->w_alt[1];
		c->s_alt[0] += c->s_alt[1];
		c->have_alt[1] = 0;
	}
	for (int j = 0; j < 2; j++)
		if (!c->have_alt[j])
			c->w_alt[j] = c->s_alt[j] = 0;
	for (int k = 0; k < 2; k++) {
		const float rk = x0 + in->e[k];
		/* The nearest known level; a reading counts for it within a saccade's size, or 40% of the gap to
		 * the next level if that is more. */
		const float lv[3] = {acc, c->alt[0] + u, c->alt[1] + u};
		const int has[3] = {1, c->have_alt[0], c->have_alt[1]};
		int best = 0;
		for (int j = 1; j < 3; j++)
			if (has[j] && fabsf(rk - lv[j]) < fabsf(rk - lv[best]))
				best = j;
		float gap = 1e9f;
		for (int j = 0; j < 3; j++)
			if (has[j] && j != best)
				gap = fminf(gap, fabsf(lv[j] - lv[best]));
		const float tol = fmaxf(in->hi, gap < 1e8f ? 0.4f * gap : 0.0f);
		if (fabsf(rk - lv[best]) > tol) {
			if (fabsf(rk - acc) <= in->hi)
				continue; /* between levels, not a saccade away: no vote */
			const int j = !c->have_alt[0] ? 0 : !c->have_alt[1] ? 1 : (c->w_alt[0] <= c->w_alt[1] ? 0 : 1);
			if (c->have_alt[j] && c->w_alt[j] >= VOTE_NEW)
				continue;
			c->alt[j] = rk - u; /* a new other level */
			c->have_alt[j] = 1;
			c->w_alt[j] = c->s_alt[j] = 0;
			best = j + 1;
		}
		if (best == 0) {
			v_acc += 0.5f;
		} else {
			v_alt[best - 1] += 0.5f;
			c->alt[best - 1] += 0.15f * (rk - u - c->alt[best - 1]);
		}
	}
	/* By frames, not seconds: each frame is one more sample, as sure at 15 fps as at 30. */
	const float kv = 1.0f - expf(-1.0f / VOTE_FRAMES), ks = 1.0f - expf(-1.0f / VOTE_SLOW);
	c->w_acc += kv * (v_acc - c->w_acc);
	c->s_acc += ks * (v_acc - c->s_acc);
	for (int j = 0; j < 2; j++) {
		c->w_alt[j] += kv * (v_alt[j] - c->w_alt[j]);
		c->s_alt[j] += ks * (v_alt[j] - c->s_alt[j]);
	}
	const int j = c->w_alt[0] >= c->w_alt[1] ? 0 : 1;
	/* Over the longer memory, the lead must also be well past what chance gives that many readings (they
	 * are s/ks readings' worth): a level only just taken has too few readings to tell 60/40 apart. */
	const float lead = c->s_alt[j] - c->s_acc;
	if (!(c->have_alt[j] && v_alt[j] > 0 && c->w_alt[j] >= VOTE_MIN && c->w_alt[j] >= VOTE_RATIO * c->w_acc &&
	      lead > 0 && lead * lead >= VOTE_Z * VOTE_Z * ks * (c->s_alt[j] + c->s_acc)))
		return 0;
	*to = c->alt[j] + u;
	const float w = c->w_alt[j], sw = c->s_alt[j];
	c->alt[j] = acc - u;
	c->w_alt[j] = c->w_acc;
	c->s_alt[j] = c->s_acc;
	c->w_acc = w;
	c->s_acc = sw;
	return 1;
}

/*
 * Moves the accepted value f->value towards this frame's combined reading m. Changes inside the noise
 * band are low-passed (with Vision's point, as a reading it makes steadier: see mf). A bigger change is
 * taken at once only when it is backed up:
 *  - Vision's own pupil point (if given) moved the same way by a fair part of it. Vision's point is
 *    coarse but never jumps the way the refined one can, so with it nothing else may back a change up;
 *  - without Vision's points: both eyes moved alike by a saccade's size while the measurement has been
 *    clean (no outlier for CLEAN_TIME), or the upper lid moved with it (it follows a vertical glance).
 * Otherwise a change short of a saccade is averaged like noise if Vision's point (which would have shown
 * it) is given, and without it has to stay put for a frame (PERSIST_LONG frames while outliers are about)
 * before it is followed; a reading that comes back first was an outlier and changed nothing. A
 * saccade-sized change that nothing backs up, while outliers are about (or Vision's point says nothing
 * moved), is settled by a vote: the readings keep going to two levels (a refiner flipping between two
 * spots), and the level that recently got clearly more of them wins.
 * Returns 1 when the change may show at once; anything else only glides there (see slew()).
 */
static int gate(cf_robust *c, cf_one_euro *f, float m, const gate_in *in, double t, double dt)
{
	const float x0 = (float)f->value;
	int jump = 0, big = 0, switched = 0, landed = 0, averaged = 0, away = 0, followed = 0;
	/* Where Vision's point says the gaze is (the vote's reference, see vote()), and how far it says the gaze
	 * moved since the accepted value was taken. */
	const float g0 = c->v_gain, u = in->vis ? g0 * in->v : 0.0f;
	const float shift = in->vis && c->have_v ? g0 * (in->v - c->v_anchor) : 0.0f;
	const int init = !f->ready || t - f->t > 0.5;
	if (in->vis != c->have_v)
		new_level(c); /* the levels' reference changed */
	if (init) {
		f->value = m;
		f->deriv = 0;
		f->ready = 1;
		c->steady_n = 0;
		c->held_big = 0;
		c->jump_weak = 0;
		new_level(c);
		jump = landed = 1;
	} else {
		const float e = m - x0, ae = fabsf(e);
		const int s = e > 0 ? 1 : -1;
		const int clean = t - c->out_t > CLEAN_TIME;
		/* Right after a jump the accepted value carries that one frame's noise: the next readings are
		 * averaged in (1/2, 1/3, ...: a running mean of the readings taken since, outliers left out)
		 * until that is no quicker than the usual low-pass, and for the first few of them anything short
		 * of a saccade counts as that noise (it is the same landing). */
		const double a_lp = fmax(in->alpha, 1.0 / (double)(c->settle + 1));
		const int landing = c->settle <= LANDING_FRAMES;
		const float lo = landing ? in->hi : in->lo;
		big = ae > in->hi;
		/* The reading stayed put since last frame, which was already away from the accepted value (this is
		 * not a one-frame excursion; a reading just past the noise band, next to last frame's one just
		 * inside it, is not "staying put"). */
		const int steady = c->away && c->have_prev && fabsf(m - c->m_prev) <= fmaxf(lo, 0.5f * ae);
		/* What a noise-sized change is averaged as. With Vision's point, the reading made steadier by it:
		 * Vision's point is several times less noisy frame to frame (in the user's clip 0.3 px against the
		 * refined 1.2-2 px), but coarse and offset from the refined one. So the offset between them,
		 * averaged since the last change taken, puts Vision's reading (u + v_off) where the refined one
		 * says, and the two are weighted by their noise. What is averaged then wanders far less with the
		 * refined reading's noise, and a refined outlier pulls it only by its small share; a real small
		 * move shows in both. Without Vision's point it is the refined reading itself, and so it is while
		 * a landing is still being averaged: the offset is only being learnt then (from the same readings). */
		float mf = m;
		if (in->vis && c->have_v && c->v_off_n > 0 && a_lp <= in->alpha) {
			const float n_p = c->v_noise * g0, var_m = in->noise * in->noise, var_p = n_p * n_p;
			if (var_m + var_p > 0)
				mf = m + var_m / (var_m + var_p) * (u + c->v_off - m);
		}
		const float ef = clampf(mf - x0, -lo, lo); /* (pulling no harder than the band's edge) */
		/* The vote between levels (see vote()): a level the readings clearly keep going to wins. */
		float vote_to;
		if (c->jump_weak && c->out_t > c->jump_t && t - c->jump_t <= REVERT_TIME) {
			/* The last jump only had both eyes moving alike to back it (no Vision point, lids still), and
			 * outliers showed up right after it: it was most likely both eyes reading wrong at once. */
			f->value = c->jump_from;
			jump = landed = 1;
			c->jump_weak = 0;
			c->held_big = 0;
			c->steady_n = 0;
		} else if (vote(c, x0, x0 + shift, u, in, &vote_to)) {
			f->value = vote_to;
			switched = 1;
			c->held_big = 0;
			c->steady_n = 0;
		} else if (ae <= lo) {
			f->value += a_lp * ef;
			averaged = 1;
			jump = landing; /* still landing the saccade: shown as it settles, not slewed */
			if (c->held_big)
				c->out_t = t; /* a saccade-sized reading came straight back: an outlier */
			c->held_big = 0;
			c->steady_n = 0;
		} else {
			const float v_min = fmaxf((big ? VIS_K : VIS_K_SMALL) * c->v_noise, VIS_MIN);
			const int by_vision = in->vis && c->have_v &&
					      (in->v - c->v_anchor) * (float)s >= fmaxf(VIS_FRAC * ae, v_min);
			const int by_eyes = !in->vis && clean && !in->lr_out &&
					    ae > fmaxf(in->resync > 0 ? in->resync : in->hi, in->eyes_min) &&
					    in->e[0] * (float)s > in->lo && in->e[1] * (float)s > in->lo;
			const int by_lids = !in->vis && in->lid && c->have_lid && big &&
					    (c->lid_anchor - in->open) * (float)s >=
						    fmaxf(LIDC_FRAC * ae, LIDC_K * in->open_noise);
			if (by_vision || by_eyes || by_lids) {
				/* Straight back to where the last jump came from, a few frames later: that jump was an
				 * outlier after all (two eyes at once, and no Vision point to tell). */
				const int back = t - c->jump_t <= REVERT_TIME &&
						 fabsf(m - c->jump_from) <= fmaxf(in->lo, 0.3f * fabsf(c->jump_size));
				if (back)
					c->out_t = t;
				float to = m;
				if (by_vision) {
					/* Vision's point is coarse but never flips: where it says the gaze went (it
					 * under-reads by v_gain, learned here). A refined reading far from that, e.g.
					 * a flip landing with the saccade, is not taken: Vision's estimate is, and
					 * the next readings settle it. */
					const float dv = in->v - c->v_anchor, pred = x0 + dv * c->v_gain;
					if (fabsf(m - pred) > fmaxf(in->hi, VIS_AGREE * fabsf(pred - x0))) {
						to = pred;
					} else {
						/* Both say where the gaze landed: weighted by how far off each tends
						 * to be. The refined reading carries this frame's landmark noise;
						 * Vision's estimate its own noise (scaled up by v_gain) and the error
						 * of v_gain itself, which grows with the size of the move. So the
						 * landing doesn't carry one noisy frame's reading. */
						const float n_p = c->v_noise * c->v_gain;
						const float n_g = c->v_gain_err * fabsf(pred - x0);
						const float var_m = in->noise * in->noise;
						const float var_p = n_p * n_p + n_g * n_g;
						if (var_m + var_p > 0)
							to = m + var_m / (var_m + var_p) * (pred - m);
					}
					/* Learnt from backed saccades the eyes agree on, roughly as Vision said (not
					 * a flip, or a wrong level it moved from, which would teach a wrong gain),
					 * each moving it only so far, and how far off it was. */
					const float far = fabsf(pred - x0);
					if (!in->lr_out && big && dv * e > 0 && fabsf(m - pred) <= 0.7f * far) {
						const float r = logf(e / dv / c->v_gain);
						c->v_gain_err += 0.3f * (fminf(fabsf(r), 0.5f) - c->v_gain_err);
						c->v_gain *= expf(0.3f * clampf(r, -0.25f, 0.25f));
					}
					c->v_gain = clampf(c->v_gain, 0.5f, 5.0f);
				}
				c->jump_t = t;
				c->jump_from = x0;
				c->jump_size = to - x0;
				c->jump_weak = by_eyes && !by_lids && !back; /* (going back is not a guess) */
				f->value = to;
				jump = landed = 1;
				c->steady_n = 0;
				c->held_big = 0;
			} else if (in->vis && !big) {
				/* Past the noise band but short of a saccade, and Vision's point, which shows any real
				 * move this big, saw none: a larger draw of the noise. Averaged like the rest of it (as
				 * the reading Vision's point steadies), its pull capped at the band's edge, so a run of
				 * them can't carry the value off. */
				f->value += a_lp * ef;
				averaged = 1;
				if (c->held_big)
					c->out_t = t;
				c->held_big = 0;
				c->steady_n = 0;
			} else {
				away = 1;
				if (c->held_big && !steady)
					c->out_t = t;
				c->steady_n = steady ? c->steady_n + 1 : 0;
				if (big && !clean) {
					c->held_big = 1; /* saccade-sized, unbacked, outliers about: left to the vote */
				} else if (c->steady_n >= (clean && !(big && in->vis) ? 1 : PERSIST_LONG)) {
					/* Persisting: followed like before (small changes partly, more the longer
					 * they stay, saccade-sized ones fully), but shown at once only when nothing
					 * could have said otherwise, and it was saccade-sized on both frames (noise
					 * wandering off the band and then a little further is not a saccade). */
					const float k = big ? 1.0f : smoothstepf(0.0f, 3.0f, (float)c->steady_n);
					const float k_e = smoothstepf(in->lo, in->hi, ae);
					const float a = (float)a_lp + (1.0f - (float)a_lp) * k * k_e;
					f->value += a * e;
					jump = landed = clean && !in->vis && big && c->held_big;
					c->jump_weak &= !landed;
					c->held_big = 0;
					followed = 1;
				} else {
					c->held_big = big;
				}
			}
		}
	}
	f->t = t;
	c->m_prev = m;
	c->have_prev = 1;
	c->away = away;
	c->followed = followed;
	const int took = jump || switched || fabsf((float)f->value - x0) > in->hi;
	if (took && !switched && !init) {
		if (in->vis && c->have_v) {
			/* The levels are offsets from where Vision's point says the gaze is: the accepted one only
			 * changes when the gaze moved more (or less) than Vision's point says. */
			const float o_old = x0 - g0 * c->v_anchor, o_new = (float)f->value - u;
			if (fabsf(o_new - o_old) > in->hi)
				move_level(c, o_old, o_new, in->hi);
		} else {
			new_level(c); /* the gaze moved: the old level is history, not one to vote back to */
		}
	}
	/* Only a saccade shown at once has a landing to settle (a glide or a vote arrives already averaged);
	 * it counts the readings averaged in since. */
	c->settle = landed ? 1 : (averaged && c->settle < 1000 ? c->settle + 1 : c->settle);

	/* Where the backing-up sensors were: reset on a taken change, otherwise following them slowly. A move
	 * of Vision's point well past its noise is kept a while longer (the refined reading may only show it
	 * a frame or two later, both eyes having read wrong when it happened). */
	const float ka = (float)(1.0 - exp(-dt / ANCHOR_TAU));
	if (in->vis) {
		if (c->have_v && !big && !took) {
			const float kn = (float)(1.0 - exp(-dt / DOWN_NOISE_TAU));
			c->v_noise += kn * (fminf(fabsf(in->v - c->v_last), 0.03f) - c->v_noise);
		}
		const int v_moved = c->have_v && fabsf(in->v - c->v_anchor) > fmaxf(VIS_K * c->v_noise, VIS_MIN);
		const float kh = v_moved ? (float)(1.0 - exp(-dt / ANCHOR_HOLD)) : ka;
		c->v_anchor = (!c->have_v || took) ? in->v : c->v_anchor + kh * (in->v - c->v_anchor);
		c->v_last = in->v;
		/* The refined-minus-Vision offset (see mf above): restarted from the accepted value on a taken
		 * change (with the gain as just learnt), then a running mean of the readings averaged in (never an
		 * outlier, or a flipping refiner's other level), each pulling no further than the band's edge,
		 * until that is no quicker than VIS_OFF_TAU. */
		const float u1 = c->v_gain * in->v;
		if (took || !c->have_v || c->v_off_n == 0) {
			c->v_off = (float)f->value - u1;
			c->v_off_n = 1;
		} else if (averaged && !in->lr_out) {
			const float k = fmaxf((float)(1.0 - exp(-dt / VIS_OFF_TAU)), 1.0f / (float)(c->v_off_n + 1));
			c->v_off += k * clampf(m - u1 - c->v_off, -in->hi, in->hi);
			c->v_off_n += c->v_off_n < 1000;
		}
	} else {
		c->v_off_n = 0;
	}
	c->have_v = in->vis;
	if (in->lid)
		c->lid_anchor = (!c->have_lid || took) ? in->open : c->lid_anchor + ka * (in->open - c->lid_anchor);
	c->have_lid = in->lid;
	return jump;
}

/* Low-pass weight for noise-sized changes: never weaker smoothing per frame than at 30 fps, so a slower
 * camera (10-15 fps in dim light) doesn't show more jitter per frame. */
static double noise_alpha(double cutoff, double dt)
{
	return fmin(oe_alpha(cutoff, dt), oe_alpha(cutoff, 1.0 / 30.0));
}

/* The value the output uses: the accepted one, through a second low-pass stage (weight alpha) and moving
 * by at most `step` per frame, unless a backed-up change (or a restored state) lets it jump. One low-pass
 * stage passes a fixed share of each frame's noise straight into that frame's step, a fine shimmer; the
 * second leaves a slow, smooth wander of a fraction of the size for little more lag. A saccade skips
 * both. With `pass`, a change that persisted (a slow, steady sweep: reading along a line, which the
 * sideways damping must keep up with) is not treated as noise: this frame's move of the accepted value is
 * passed on whole (any lag left from before still closes gently). And noise or an outlier that got
 * through can only ever nudge the eyes by a fraction of a pixel per frame. */
static double slew(cf_robust *c, double x, int snap, float step, double alpha, int pass)
{
	if (snap || !c->have_shown) {
		c->shown = x;
	} else {
		double d = alpha * (x - c->shown);
		if (pass && c->followed)
			d += (1.0 - alpha) * (x - c->shown_x);
		c->shown += clampf((float)d, -step, step);
	}
	c->shown_x = x;
	c->have_shown = 1;
	return c->shown;
}

/* ---- Measurements */

typedef struct {
	cf_pt center;    /* corner midpoint, moved along the corner line to the middle of the outline */
	float width;     /* corner to corner */
	float height;    /* extent of the outline across the corner line */
	float offset;    /* pupil above the corner line, eye widths (positive = up) */
	float side;      /* pupil along the corner line from the corner midpoint, eye widths */
	float openness;  /* height / width */
	int has_v;       /* Vision's own pupil point given (and finite): */
	float v_offset, v_side; /* ... the same two measures from it */
} eye_measure;

/* axis: unit vector from this face's left eye towards its right eye (image space).
 * up: unit vector perpendicular to it, pointing to the top of the head.
 * vpupil: Vision's own pupil point, or NULL. */
static int measure(const cf_pt *pts, int count, cf_pt pupil, const cf_pt *vpupil, cf_pt axis, cf_pt up,
		   eye_measure *m)
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
	m->has_v = 0;
	if (vpupil) {
		const float vx = vpupil->x - mid.x, vy = vpupil->y - mid.y;
		m->v_offset = (vx * nx + vy * ny) / w;
		m->v_side = (vx * ux + vy * uy) / w;
		m->has_v = isfinite(m->v_offset) && isfinite(m->v_side); /* broken: just not used */
	}
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

/* Sets g->axis and g->up (the directions the iris is pushed along) from this frame's eye axis.
 * The raw axis wobbles by a degree or so with landmark noise, which made the push direction
 * wobble, so its roll angle is One Euro smoothed: steady when the head is still, quick to follow
 * a real tilt (at 30 deg/s it trails by ~2 deg, a tenth of a pixel on a few-pixel shift). Only
 * the output direction is smoothed: the measurements use the frame's own axis, so a head roll
 * never reads as a glance. The angle is unwrapped against the filtered one, so the axis keeps
 * the orientation the measurements used. */
static void smooth_axes(cf_gaze *g, cf_pt axis, double t)
{
	double a = atan2(axis.y, axis.x);
	if (g->roll.ready) {
		while (a - g->roll.value > M_PI)
			a -= 2.0 * M_PI;
		while (a - g->roll.value < -M_PI)
			a += 2.0 * M_PI;
	}
	const double roll = oe_filter(&g->roll, a, t, ROLL_MIN_CUTOFF, ROLL_BETA);
	const float ax = (float)cos(roll), ay = (float)sin(roll);
	cf_pt u = {-ay, ax};
	if (u.y > 0) {
		u.x = -u.x;
		u.y = -u.y;
	}
	g->axis.x = ax;
	g->axis.y = ay;
	g->up = u;
}

void cf_gaze_init(cf_gaze *g)
{
	memset(g, 0, sizeof(*g));
	g->up.x = 0;
	g->up.y = -1;
	g->axis.x = 1;
	g->axis.y = 0;
	g->open_ref = DEFAULT_OPEN;
	g->far_fade = 1.0f;
	g->down_noise = DOWN_NOISE_INIT;
	g->side_noise = DOWN_NOISE_INIT;
	g->lid_noise = LID_NOISE_INIT;
	cf_robust *rb[2] = {&g->rb_down, &g->rb_side};
	for (int i = 0; i < 2; i++) {
		rb[i]->lr_noise = LR_NOISE_INIT;
		rb[i]->v_noise = VIS_NOISE_INIT;
		rb[i]->v_gain = VIS_GAIN_INIT;
		rb[i]->v_gain_err = VIS_GAIN_ERR_INIT;
		rb[i]->out_t = -1e9; /* clean until an outlier is seen */
	}
}

int cf_gaze_notes_valid(float baseline, float notes_offset)
{
	const float glance = baseline - notes_offset;
	return isfinite(glance) && glance > CF_NOTES_MIN_GLANCE;
}

static float finite_or(float v, float fallback)
{
	return isfinite(v) ? v : fallback;
}

/* A NaN/inf setting would poison the filters for good (g->down never recovers): replace it. */
static cf_gaze_params sanitize(const cf_gaze_params *in)
{
	cf_gaze_params p = *in;
	p.strength = clampf(finite_or(p.strength, 0.0f), 0.0f, 1.0f);
	p.max_glance = finite_or(p.max_glance, 0.15f);
	p.baseline = finite_or(p.baseline, 0.0f);
	p.baseline_open = finite_or(p.baseline_open, 0.0f);
	p.notes_open = finite_or(p.notes_open, 0.0f);
	if (!isfinite(p.notes_offset) || !isfinite(p.notes_side) || !cf_gaze_notes_valid(p.baseline, p.notes_offset)) {
		p.has_notes = 0;
		p.notes_offset = p.notes_side = p.notes_open = 0.0f;
	}
	return p;
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
	g->calib_head_n = 0;
}

static void finish_calibration(cf_gaze *g)
{
	g->calib_result = median(g->calib_samples, g->calib_count);
	g->calib_open_result = median(g->calib_open, g->calib_count);
	g->calib_side_result = median(g->calib_side, g->calib_count);
	g->calib_head_result = cf_head_reference(g->calib_head, g->calib_head_n);
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
	out->head = g->calib_head_result;
	return 1;
}

/* ---- Blinks */

/* Moves the correction strength towards `target` through two equal first-order stages (a critically
 * damped ease): it starts moving gently and arrives gently, an S-curve with no kink at either end, so a
 * fade never shows as a snap. Exact per stage for any frame time, so it looks the same at 15 fps as at
 * 30. Fades in a little slower than it fades out. */
static void ease_factor(cf_gaze *g, float target, double dt)
{
	const double tau = target > g->factor_mid ? FADE_IN_TAU : FADE_OUT_TAU;
	const float a = (float)(1.0 - exp(-dt / tau));
	g->factor_mid += a * (target - g->factor_mid);
	g->factor += a * (g->factor_mid - g->factor);
}

/*
 * r: openness relative to the user's open eye (1 = normal). rate: how fast r falls, per second.
 * A blink closes the lids to a slit within ~2 frames; looking down lowers them part way and
 * then they stay. So: below BLINK_ENTER = blink; dropping fast = hold until it's clear which
 * one it is; a drop that stops above BLINK_ENTER is released as a glance (or squint).
 */
/* step: how much r fell since the last frame. The half-blink and steadiness thresholds widen with
 * the measured lid noise, so noisy landmarks don't keep starting holds. */
static void update_blink(cf_gaze *g, float r, float r2, float slow_drop, float rate, float step, double t)
{
	/* Where the lids count as open again: HALF_EXIT, or a little under where they were before the
	 * hold (lids lowered by a smile never get back to HALF_EXIT). */
	if (!g->blink && !g->closing && !g->half)
		g->open_r = g->r_peak;
	const float reopen = fminf(HALF_EXIT, g->open_r - 0.08f);
	if (g->blink) {
		const double len = t - g->hold_start;
		if (r > BLINK_EXIT || (r > BLINK_ENTER + 0.05f && fabsf(rate) < 1.0f && len > 0.1)) {
			g->blink = 0;
			if (r > reopen) {
				g->settle_until = t + SETTLE_TIME;
			} else { /* still half shut: the pupil point is still unreliable */
				g->half = 1;
				g->half_start = t;
			}
		}
		return;
	}
	if (g->half) {
		if (r < BLINK_ENTER) {
			g->half = 0;
			g->blink = 1;
		} else if (r > reopen) {
			g->half = 0;
			g->settle_until = t + SETTLE_TIME;
		} else if (t - g->half_start > HALF_MAX) {
			g->half = 0; /* a squint, not a half blink: measure normally from here on */
			g->ease = 1; /* ... handing over gently: the eyes are still */
			g->ease_min_until = t + EASE_MIN;
			/* The lids are where they will stay: not a fresh drop next frame. */
			g->r_peak = g->open_r = r;
			g->r_slow = g->r_slow_prev[0] = g->r_slow_prev[1] = r2;
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
	/* Half blink: the lids fell quickly to well below open but not shut. Vision's pupil point slides
	 * with the upper lid, so it is held like a blink. */
	/* Three ways to see the quick drop: already closing fast; a big one-frame step (clean landmarks);
	 * or the two-frame mean openness well under its recent level, which is what still works with
	 * noisy landmarks (averaging halves the noise; a half blink keeps the lids down for frames). */
	if (r < HALF_ENTER && (g->closing || g->r_peak - r > fmaxf(HALF_DROP, 4.0f * g->lid_noise) ||
			       step > fmaxf(HALF_STEP, 3.0f * g->lid_noise) ||
			       (r2 < HALF_ENTER && slow_drop > fmaxf(HALF_DROP, 1.4f * g->lid_noise)))) {
		g->half = 1;
		g->half_start = t;
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
	return g->blink || g->closing || g->half || t < g->settle_until;
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

int cf_gaze_update(cf_gaze *g, const cf_face *face, const cf_gaze_params *p_in, double t, cf_eye out[2])
{
	const cf_gaze_params ps = sanitize(p_in);
	const cf_gaze_params *p = &ps;
	double dt = g->last_t > 0 ? t - g->last_t : 1.0 / 30.0;
	if (dt <= 0 || dt > 0.5)
		dt = 1.0 / 30.0;
	g->last_t = t;

	calibration_tick(g, t);

	cf_pt axis = {1, 0}, up = {0, -1};
	eye_measure m[2];
	const int given_v = face->has_vision_pupils;
	const int ok = face->found && face_axes(face, &axis, &up) &&
		       measure(face->left_eye, face->left_count, face->left_pupil,
			       given_v ? &face->left_pupil_vision : NULL, axis, up, &m[0]) &&
		       measure(face->right_eye, face->right_count, face->right_pupil,
			       given_v ? &face->right_pupil_vision : NULL, axis, up, &m[1]);
	g->face_found = ok;

	if (!ok) {
		g->tracking = 0;
		g->have_rel_open = 0;
		g->have_down_raw = 0;
		g->have_side_raw = 0;
		g->blink = g->closing = g->half = 0;
		g->ease = 0;
		g->hist_n = 0;
		g->rb_down.steady_n = g->rb_side.steady_n = 0;
		g->rb_down.held_big = g->rb_side.held_big = 0;
		g->rb_down.away = g->rb_side.away = g->rb_down.followed = g->rb_side.followed = 0;
		g->rb_down.have_prev = g->rb_side.have_prev = 0;
		g->rb_down.have_v = g->rb_side.have_v = 0;
		g->rb_down.have_lid = 0;
		/* The detector often misses a frame or two: hold the correction exactly as it was through
		 * that instead of letting the eyes dip and climb back. Lost for longer, the person may well
		 * have moved and the spot being warped may no longer be their eyes: let go quickly, like a
		 * saccade (gone ~0.1 s after the hold), instead of a slow glide down at a stale spot. */
		const int brief = g->have_geometry && g->cx[0].ready && t - g->cx[0].t < LOST_HOLD;
		const float old = g->factor;
		if (!brief) {
			g->factor -= (float)(1.0 - exp(-dt / LOST_FADE_TAU)) * g->factor;
			if (g->factor < 0.05f)
				g->factor = 0; /* the last few percent are invisible: just stop */
		}
		g->factor_mid = g->factor;
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
	smooth_axes(g, axis, t);

	/* Both eyes' readings combined, an outlier in one eye replaced by the other eye's reading. Everything
	 * below (blinks, calibration, the glance) uses these. Learned only while nothing is held (the
	 * previous frame's state: this frame's is known only later). */
	const int learn_lr = !holding(g, t);
	const int vis = m[0].has_v && m[1].has_v;
	const float v_down = vis ? -0.5f * (m[0].v_offset + m[1].v_offset) : 0.0f; /* down positive, like the glance */
	const float v_side = vis ? 0.5f * (m[0].v_side + m[1].v_side) : 0.0f;
	float e_down[2], e_side[2];
	int lr_down, lr_side;
	const float down_raw = lr_combine(&g->rb_down, &g->down, p->baseline - m[0].offset, p->baseline - m[1].offset,
					  vis, v_down, learn_lr, t, dt, e_down, &lr_down);
	const float side =
		lr_combine(&g->rb_side, &g->side, m[0].side, m[1].side, vis, v_side, learn_lr, t, dt, e_side, &lr_side);
	const float offset = p->baseline - down_raw;
	const float openness = fminf(m[0].openness, m[1].openness);
	/* The noise band and saccade size of the glance follow the measured landmark noise: as tight as
	 * clean tracking allows, wider when it is noisy. */
	const float down_lo = clampf(DOWN_LO_K * g->down_noise, DOWN_LO_MIN, DOWN_LO_MAX);
	const float down_hi = clampf(2.0f * down_lo, DOWN_HI_MIN, DOWN_HI_MAX);

	/* Lids: relative to the calibrated (or learned) open eye. */
	const float ref = p->baseline_open > 0 ? p->baseline_open : g->open_ref;
	const float r = openness / fmaxf(ref, 0.05f);
	const float rate = g->have_rel_open ? (g->last_rel_open - r) / (float)dt : 0.0f;
	if (!g->have_rel_open) {
		g->r_peak = r;
		g->r_slow = g->r_slow_prev[0] = g->r_slow_prev[1] = r;
		g->last_r = r;
	}
	/* Glance state before this frame, for undoing a late-caught blink (see ROLLBACK_WIN). */
	if (g->hist_n == CF_HIST)
		memmove(&g->hist[0], &g->hist[1], sizeof(g->hist[0]) * (CF_HIST - 1));
	else
		g->hist_n++;
	g->hist[g->hist_n - 1].t = t;
	g->hist[g->hist_n - 1].down = g->down.value;
	g->hist[g->hist_n - 1].ready = g->down.ready;
	g->hist[g->hist_n - 1].far = g->far_fade;
	g->hist[g->hist_n - 1].suspect = 0; /* set below, once this frame's lids and pupil are known */
	const float step = g->have_rel_open ? g->last_rel_open - r : 0.0f;
	/* Lid noise, learned while nothing is held (blinks are not noise). */
	if (g->have_rel_open && !holding(g, t))
		g->lid_noise += (float)(1.0 - exp(-dt / DOWN_NOISE_TAU)) * (fminf(fabsf(step), 0.15f) - g->lid_noise);
	g->last_rel_open = r;
	g->have_rel_open = 1;
	const int was_held = g->blink || g->half;
	const float r2 = 0.5f * (r + g->last_r); /* two-frame mean: half the landmark noise */
	const float slow_drop = fmaxf(g->r_slow, fmaxf(g->r_slow_prev[0], g->r_slow_prev[1])) - r2;
	update_blink(g, r, r2, slow_drop, rate, step, t);
	g->r_peak = fmaxf(r, g->r_peak - HALF_DECAY * (float)dt);
	g->r_slow_prev[1] = g->r_slow_prev[0];
	g->r_slow_prev[0] = g->r_slow;
	if (!g->blink && !g->half && !g->closing)
		g->r_slow += (float)(1.0 - exp(-dt / HALF_SLOW_TAU)) * (0.5f * (r + g->last_r) - g->r_slow);
	g->last_r = r;
	/* Lids dropping fast while the pupil drops with them is a glance down (the upper lid follows the
	 * eye), not a blink: a blink leaves the pupil where it was or loses it upwards. Holding the glance
	 * at its old value here showed the whole glance for a frame and then slid the correction in.
	 * Only while the lids are still fairly open: as they close, Vision's pupil point slides down with
	 * the upper lid, which looked like a glance. */
	const float pupil_drop = g->down.ready ? down_raw - (float)g->down.value : 0.0f;
	const float lid_drop = (g->open_r - r) * fmaxf(ref, 0.05f); /* eye widths */
	/* (Both thresholds sit above the pupil noise, so a noisy reading doesn't end a hold.) */
	const int glance = (r > GLANCE_MIN_R && pupil_drop > fmaxf(GLANCE_PUPIL, 2.0f * g->down_noise) &&
			    pupil_drop >= GLANCE_LID_RATIO * lid_drop) ||
			   (pupil_drop > GLANCE_BIG + 2.0f * g->down_noise && pupil_drop >= lid_drop);
	if ((g->closing || g->half) && !g->blink && g->down.ready && glance) {
		g->closing = g->half = 0;
		g->r_peak = r; /* the lids are where the glance put them: not a fresh drop next frame */
		g->r_slow = g->r_slow_prev[0] = g->r_slow_prev[1] = r;
	}
	/* ...and if the lids then shut (or half shut) after all, it was a blink, possibly caught a frame or
	 * two late: undo what the frames just before it changed, so the held correction is the one from
	 * before the lids started to move. */
	const int held_now = g->blink || g->half;
	int rolled_back = 0; /* the glance was restored to an earlier state: shown as it was, at once */
	if (held_now && !was_held) {
		/* Walk back over the run of suspect frames just before this one (the lids already moving,
		 * the pupil point dragged along) and restore the state from before the first of them. A
		 * frame with steady lids, or whose pupil moved like a glance, ends the walk: what it saw is
		 * kept. (The newest entry is this frame's own, which has changed nothing yet.) */
		int from = -1;
		for (int k = g->hist_n - 2; k >= 0 && t - g->hist[k].t <= ROLLBACK_WIN && g->hist[k].suspect; k--)
			from = k;
		/* Never undo a glance the pupil is still clearly showing, more than a lid could drag it (a
		 * keyboard glance with the lids dropping): the correction would come back mid-glance. */
		if (from >= 0 && g->hist[from].ready && down_raw - (float)g->hist[from].down > ROLLBACK_SLIDE_MAX)
			from = -1;
		if (from >= 0) {
			if (g->hist[from].ready && g->down.ready)
				g->down.value = g->hist[from].down;
			g->far_fade = g->hist[from].far;
			rolled_back = 1;
		}
		g->ease = 0;
	}
	const int hold = holding(g, t);
	/* Lids steady enough to read a far glance from: not shut, half shut or moving, and settled for
	 * LID_STEADY_TIME (half-shut lids that just stopped falling still have the pupil point slid). */
	if (held_now || g->closing || fabsf(step) > clampf(3.0f * g->lid_noise, LID_STEADY_STEP, LID_STEADY_MAX) ||
	    slow_drop > fmaxf(LID_STEADY_STEP, 1.4f * g->lid_noise)) /* just fell (seen through the noise) */
		g->lid_move_t = t;
	const int lids_steady = t - g->lid_move_t >= LID_STEADY_TIME;
	int glance_trusted; /* this frame's change may be kept by a rollback; finished once far_fade is known */
	int lids_clear;     /* ... because the lids were clearly steady (not just a glance-like pupil move) */
	float glance_step;
	{
		/* Is what this frame is about to change trustworthy (see the rollback above)? Yes if the lids
		 * were clearly steady, or the pupil clearly moved like a glance (at least ROLLBACK_GLANCE of the
		 * lid drop, and well above the pupil noise). Anything unclear (noisy lids that may be starting
		 * to close) is suspect, so a blink caught late still undoes it. */
		/* Lids falling this frame, eye widths: the raw step or twice the two-frame-mean drop, whichever
		 * is larger (two noisy estimates; erring towards "the lids moved"). */
		const float lid_fall = fmaxf(fmaxf(step, 2.0f * slow_drop), 0.0f); /* relative openness */
		const float lid_step = lid_fall * fmaxf(ref, 0.05f);
		const float pupil_step = down_raw - g->last_down_raw;
		const int clearly_steady = !g->closing && !held_now && fabsf(step) < LID_STEADY_STEP &&
					   lid_fall < LID_STEADY_STEP;
		const int clearly_glance = pupil_step > fmaxf(ROLLBACK_PUPIL, 3.0f * g->down_noise) &&
					   pupil_step >= ROLLBACK_GLANCE * lid_step + 3.0f * g->down_noise;
		glance_trusted = clearly_steady || clearly_glance;
		lids_clear = clearly_steady;
		glance_step = pupil_step;
	}
	/* Landmark noise: mean frame-to-frame change of the raw glance while the lids are steady. A change of a
	 * saccade's size is a saccade or an outlier, not noise: left out (a refiner jumping between two spots
	 * on many frames would otherwise read as huge noise and open the noise band right up). */
	if (!hold && g->have_down_raw && fabsf(down_raw - g->last_down_raw) <= down_hi) {
		const float ch = fabsf(down_raw - g->last_down_raw);
		g->down_noise += (float)(1.0 - exp(-dt / DOWN_NOISE_TAU)) * (ch - g->down_noise);
	}
	const float down_jump = fabsf(down_raw - g->last_down_raw); /* this frame's raw pupil move */
	g->last_down_raw = down_raw;
	g->have_down_raw = !hold;
	/* The same for the sideways reading: its noise band and saccade size widen with it (never tighter than
	 * SIDE_LO and SIDE_HI). */
	const float side_lo = clampf(DOWN_LO_K * g->side_noise, SIDE_LO, 2.0f * SIDE_LO);
	const float side_hi = fmaxf(SIDE_HI, 2.0f * side_lo);
	if (!hold && g->have_side_raw && fabsf(side - g->last_side_raw) <= side_hi) {
		const float kn = (float)(1.0 - exp(-dt / DOWN_NOISE_TAU));
		g->side_noise += kn * (fabsf(side - g->last_side_raw) - g->side_noise);
	}
	g->last_side_raw = side;
	g->have_side_raw = !hold;

	/* Learn the open-eye openness when uncalibrated: follows wide-open quickly, closing slowly. */
	if (p->baseline_open <= 0 && !hold && fabsf(down_raw) < 0.08f && r > 0.6f) {
		const double tau = openness > g->open_ref ? 1.0 : 5.0;
		g->open_ref += (float)(1.0 - exp(-dt / tau)) * (openness - g->open_ref);
		g->open_ref = clampf(g->open_ref, 0.15f, 0.6f);
	}

	/* Head pose: from this frame's landmarks (pose.c), against the face the lens calibration learnt (a
	 * typical face while a lens calibration is learning it). The detector's own angles only without
	 * landmark measures. */
	float yaw_deg = 0, pitch_deg = 0;
	const cf_head_ref *head_ref = g->calibrating && g->calib_target == CF_CALIB_LENS ? NULL : &p->head_ref;
	if (!cf_pose_update(&g->pose, face->has_head_geom ? &face->head : NULL, head_ref, t, &yaw_deg, &pitch_deg) &&
	    face->has_pose) {
		yaw_deg = face->yaw_deg;
		pitch_deg = face->pitch_deg;
	}
	float abs_yaw = fabsf(yaw_deg), abs_pitch = fabsf(pitch_deg);
	abs_yaw = isfinite(abs_yaw) ? abs_yaw : 0; /* broken pose: ignore it rather than poison the factor */
	abs_pitch = isfinite(abs_pitch) ? abs_pitch : 0;

	/* The first moments after starting are skipped: the eyes are still on the Calibrate button. */
	const int calib_warm = g->calibrating && t - g->calib_start >= CF_CALIB_WARMUP_S;
	if (calib_warm && !hold && abs_yaw < 15 && g->calib_count < 64) {
		g->calib_samples[g->calib_count] = offset;
		g->calib_open[g->calib_count] = openness;
		g->calib_side[g->calib_count] = side;
		g->calib_count++;
		if (face->has_head_geom && g->calib_head_n < 64)
			g->calib_head[g->calib_head_n++] = face->head;
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
	 * jitter disappears, quick enough to follow a real glance, frozen through blinks.
	 * Saccade-aware: glances jump between the lens and the notes in a frame or two, and any lag there
	 * shows as a dip on the way down and an overshoot above the lens on the way back. But only a jump
	 * something else backs up is taken at once (see gate()): a refined pupil reading that jumps for a
	 * frame and comes back changes nothing. */
	double down;
	int snap_down = rolled_back, snap_side = 0;
	if (!hold) {
		const gate_in gi = {
			.lo = down_lo,
			.hi = down_hi,
			.alpha = noise_alpha(DOWN_CUTOFF, dt),
			.vis = vis,
			.v = v_down,
			.lid = 1,
			.open = 0.5f * (m[0].openness + m[1].openness),
			.open_noise = g->lid_noise * fmaxf(ref, 0.05f),
			.e = {e_down[0], e_down[1]},
			.lr_out = lr_down,
			/* First look after a blink: eyes often move during a blink (blinks go with gaze shifts).
			 * A clear change is taken at once instead of sliding there from the held value. Only a
			 * real blink (lids shut), not a noise-triggered "closing" hold, may resync. */
			.resync = g->was_holding ? fmaxf(RESYNC_JUMP, 3.0f * g->down_noise) : 0.0f,
			.eyes_min = EYES_K * g->down_noise,
			.noise = g->down_noise,
		};
		g->was_holding = 0;
		snap_down |= gate(&g->rb_down, &g->down, down_raw, &gi, t, dt);
		down = g->down.value;
	} else {
		down = g->down.ready ? g->down.value : 0;
		g->was_holding |= g->blink || g->half;
		/* (cut short by a blink: no outlier) */
		g->rb_down.steady_n = g->rb_down.held_big = g->rb_down.away = g->rb_down.followed = 0;
	}

	/* Sideways position for reading damping. Eyes move in saccades: a return sweep jumps a
	 * fifth of an eye width in one or two frames. A lagging filter would let that jump show at
	 * full size and then drift back (a visible snap), so: jumps bigger than landmark noise are
	 * followed at once (when backed up, as above), and only noise-sized changes are smoothed. */
	const float side_c = p->has_notes ? p->notes_side : 0.0f;
	double side_f;
	if (!hold) {
		const gate_in gs = {
			.lo = side_lo,
			.hi = side_hi,
			.alpha = noise_alpha(SIDE_CUTOFF, dt),
			.vis = vis,
			.v = v_side,
			.e = {e_side[0], e_side[1]},
			.lr_out = lr_side,
			.eyes_min = EYES_K * g->side_noise,
			.noise = g->side_noise,
		};
		snap_side = gate(&g->rb_side, &g->side, side, &gs, t, dt);
		side_f = g->side.value;
	} else {
		side_f = g->side.ready ? g->side.value : side_c;
		g->rb_side.steady_n = g->rb_side.held_big = g->rb_side.away = g->rb_side.followed = 0;
	}

	/* Output slew guard: what the warp shows follows the accepted gaze, but without a backed-up saccade
	 * it moves at most SLEW_PX per 1/30 s (whatever got past the checks above can only nudge the eyes).
	 * Fades act on top of this, unlimited (they are smooth by design). Sideways, the shift is
	 * READ_DAMP of the gaze, so the gaze may move that much faster for the same shift. */
	const float wmax = fmaxf(fmaxf((float)g->rx[0].value, (float)g->rx[1].value), 1.0f);
	const float slew_step = SLEW_PX * (float)fmin(dt * 30.0, 1.0) / wmax; /* per frame, at 30 fps or slower */
	down = slew(&g->rb_down, down, snap_down, slew_step, noise_alpha(DOWN_OUT_CUTOFF, dt), 0);
	side_f = slew(&g->rb_side, side_f, snap_side, slew_step / READ_DAMP, noise_alpha(SIDE_OUT_CUTOFF, dt), 1);
	const float side_dev = (float)side_f - side_c;

	/* Glance limit: just past the notes when calibrated, else the manual setting. */
	const int notes = p->has_notes; /* sanitize() cleared it if the notes aren't below the lens */
	const float notes_glance = p->baseline - p->notes_offset;
	const float limit = notes ? 1.4f * notes_glance : p->max_glance;
	const float band = fmaxf(0.5f * limit, 0.04f);
	/* Second cue: lids drop when looking down. When the pupil already reads close to the limit, a big
	 * lid drop means a far glance (keyboard) even if the pupil landmark under-reads it. It can only
	 * switch the effect off, never add correction. Measured against the lids at the notes when that is
	 * known, so lids lowered by a smile or a squint while reading don't count, and gated well above
	 * the notes, so reading itself never triggers it. */
	float far = (float)down;
	/* The pupil must read past the gate for two frames running: one frame past it is as likely a
	 * noisy reading, or Vision's pupil point dragged by a lid just starting to close. */
	g->past_gate = (!hold && far > LID_GATE * limit) ? g->past_gate + 1 : 0;
	if (lids_steady && g->past_gate >= 2) {
		float lid_glance;
		if (notes && p->notes_open > 0)
			lid_glance = notes_glance + LID_GLANCE_K * (p->notes_open - openness) / fmaxf(ref, 0.05f);
		else
			lid_glance = LID_GLANCE_K * (1.0f - r);
		far = fmaxf(far, lid_glance);
	}

	/* Everything that should gently switch the effect off. Each term is 1 when it's fine. */
	const float f_yaw = 1.0f - smoothstepf(CF_YAW_FADE_LO, CF_YAW_FADE_HI, abs_yaw);
	const float f_pitch = 1.0f - smoothstepf(CF_PITCH_FADE_LO, CF_PITCH_FADE_HI, abs_pitch);
	const float f_side = 1.0f - smoothstepf(SIDE_FADE_LO, SIDE_FADE_HI, fabsf(side_dev));
	/* Far fade: how much the gaze is still on the notes rather than past them (keyboard). It goes into
	 * the eased factor below, with the shift held at its pre-look-away size while it fades, so letting go
	 * and taking over are gradual and the visible eye only ever moves the same way as the real gaze.
	 * Held while the lids are shut or closing (the lid cue is meaningless then), recomputed as soon as
	 * they are open again (also during the settle time after a blink, from the fresh lids). */
	if (!g->blink && !g->closing && !g->half)
		g->far_fade = 1.0f - smoothstepf(limit, limit + band, far);
	const float f_far = g->far_fade;
	/* A glance-like move with the lids moving that also switched the correction off (past the notes) is
	 * only trusted when it is far bigger than a dragged pupil point: a noisy reading of a slide can look
	 * like a small glance, but not like a keyboard glance. */
	if (glance_trusted && !lids_clear && g->far_fade < g->hist[g->hist_n - 1].far - 0.3f &&
	    glance_step < GLANCE_BIG + 2.0f * g->down_noise)
		glance_trusted = 0;
	g->hist[g->hist_n - 1].suspect = !glance_trusted;
	float f_size = smoothstepf(0.05f, 0.09f, face->face_width / fmaxf((float)face->image_width, 1.0f));
	f_size = isfinite(f_size) ? f_size : 0;
	/* No "down <= 0 -> off" here: the shift is already 0 at the lens (it is proportional to the
	 * glance), and switching the factor off there made every glance at the notes start from 0 and
	 * ease in over half a second, so the eyes visibly dipped and climbed back each time. */
	/* Looking away (past the notes, or sideways out of them) lets go, and looking back takes over again,
	 * through the same smooth fade as a head turn: an S-curve over a few hundred ms, never a snap. While
	 * the gaze is away (lock < 1), the shift fades from what it was when the gaze left (d_lock,
	 * side_lock), not from the far-away gaze itself, so it only ever shrinks, without a jump or a wobble
	 * on the way. Back in, it follows the gaze again at once (scaled by the fading-in factor). */
	const float lock = f_side * f_far;
	const float target = f_yaw * f_pitch * f_size * lock;
	if (hold) {
		/* Blink: hold the correction steady (the gaze terms are held too). Only head-based fades
		 * (reliable while the eyes are closed) may lower it; eyes closed for long fade out. */
		if (g->blink && t - g->hold_start > BLINK_HOLD_MAX)
			ease_factor(g, 0, dt);
		else if (target < g->factor)
			ease_factor(g, target, dt);
		else
			g->factor_mid = g->factor;
	} else {
		ease_factor(g, target, dt);
	}
	g->last_target = target;

	const float d_live = (float)(down > 0 ? down : 0);
	if (lock >= 0.999f || !g->have_lock) {
		g->d_lock = d_live;
		g->side_lock = side_dev;
		g->have_lock = 1;
	}
	const float d = g->d_lock;
	/* Sideways damping only engages as the gaze drops towards the notes, so looking around at lens
	 * height (at the other person) stays untouched. */
	const float engage = smoothstepf(0.0f, fmaxf(0.35f * limit, 0.01f), d);
	/* After a half-blink hold timed out the eyes are still, but the measurement taking over (and the
	 * lid cue waking up 0.1 s later) can differ from the held output: move there at most EASE_STEP_PX
	 * per frame instead of in one jump. A real saccade ends it (that change is shown at once): the pupil
	 * jumped this frame, and the jump was backed up (an outlier jump does not end it). */
	if (g->ease && (hold || (snap_down && down_jump > fmaxf(DOWN_HI_MIN, 3.0f * g->down_noise))))
		g->ease = 0;
	int converged = 1;
	const float max_step = EASE_STEP_PX * (float)(dt * 30.0);
	for (int i = 0; i < 2; i++) {
		const float w = (float)g->rx[i].value;
		float a = d * w * g->factor;
		float h = -READ_DAMP * g->side_lock * w * g->factor * engage;
		if (g->ease) {
			converged &= fabsf(a - g->amount_unit[i]) <= max_step && fabsf(h - g->hshift_unit[i]) <= max_step;
			a = g->amount_unit[i] + clampf(a - g->amount_unit[i], -max_step, max_step);
			h = g->hshift_unit[i] + clampf(h - g->hshift_unit[i], -max_step, max_step);
		}
		g->amount_unit[i] = a;
		g->hshift_unit[i] = h;
	}
	if (g->ease && converged && t >= g->ease_min_until)
		g->ease = 0;
	return finish_output(g, p, out);
}

int cf_gaze_predict(const cf_gaze *g, const cf_gaze_params *p_in, double t_now, cf_eye out[2])
{
	if (!g->last_n || !g->have_geometry)
		return 0;
	const cf_gaze_params ps = sanitize(p_in);
	const cf_gaze_params *p = &ps;
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
	out->factor = (g->down.ready && g->down.value > 0) ? g->factor : 0.0f; /* includes looking away */
}

/* ---- Iris centre refinement
 *
 * The gaze model turns the refined point's vertical position into a shift of the iris, so any
 * frame-to-frame jitter or flip here shows as shaking eyes. Everything below is therefore a
 * continuous function of the pixels and the landmarks: soft weights, fractional pixel coverage and
 * blends, no argmax and no on/off thresholds, so a sub-pixel change gives a sub-pixel answer.
 *
 * The iris (radius ~0.2 eye widths) is usually cut by both lids, so its vertical centre shows only
 * in its left and right edges (the limbus arcs); the dark area itself just fills the opening.
 *  1. Lid curves through the outline, pinned at the corners. Vision gives 6 points per eye, a coarse
 *     hexagon that cuts off the curved lids' corners; the curves follow the lids between them.
 *  2. Iris and sclera grey levels (percentiles inside the outline, away from the lash line) and a
 *     linear darkness ramp between them, so an edge's position comes out sub-pixel, by area.
 *  3. Iris column: mean shift on the darkness profile along the eye, from Vision's point.
 *  4. Darkness centroid across the eye in those columns: steady, but drawn towards the middle of
 *     the opening when the lids cut the iris. The fallback.
 *  5. Per pixel row, the iris' left and right edges: dark width out from the centre (a running
 *     minimum, so lashes or a dark corner past a gap of sclera don't count; bright patches near the
 *     centre, catchlights and the screen's reflection on the cornea, bridged), kept only where
 *     sclera shows just beyond and the lids (landmarks) are clear. A circle fit through these edge
 *     points (Kasa, linear least squares, refitted with weights that discount points off the circle)
 *     gives the centre, unbiased by the lids, from either or both arcs.
 *  6. The fit is blended in by how well it is determined (few rows when squinting), and the result
 *     towards Vision's point when the eye is nearly closed, shows no contrast or no iris.
 */

#define RP_IRIS_R 0.20f      /* iris radius, eye widths */
#define RP_CELLS_W 100       /* eyes wider than this many pixels are sampled in s x s blocks */
#define RP_MAXCELLS 6144     /* grid cells over the eye's bounding box */
#define RP_MAXROW 160        /* grid cells per row */
#define RP_NB 64             /* grey histogram bins */
#define RP_UP0 0.04f         /* lash line: weight 0 within this many eye widths under the upper lid ... */
#define RP_UP1 0.08f         /* ... 1 from here (levels, iris column and centroid) */
#define RP_LO0 0.02f         /* same over the lower lid */
#define RP_LO1 0.045f
#define RP_EUP0 0.03f        /* lid clearance for edge points, the image check does the rest */
#define RP_EUP1 0.06f
#define RP_ELO0 0.015f
#define RP_ELO1 0.035f
#define RP_P_IRIS 0.20f      /* grey percentiles inside the outline: iris level ... */
#define RP_P_SCLERA 0.90f    /* ... and sclera level */
#define RP_RAMP0 0.15f       /* darkness 1 below iris + this x (sclera - iris), 0 above ... */
#define RP_RAMP1 0.85f       /* ... iris + this x (sclera - iris), linear in between */
#define RP_CONTRAST0 18.0f   /* sclera - iris grey levels: below this, no refinement ... */
#define RP_CONTRAST1 36.0f   /* ... full refinement from here */
#define RP_OPEN0 0.09f       /* openness (outline height / width): below this, Vision's point ... */
#define RP_OPEN1 0.14f
#define RP_FILL_R 0.45f      /* bright patches near the centre: bridged if dark within this x R further out ... */
#define RP_FILL0 0.50f       /* ... for cells up to this x R from the centre, fading out ... */
#define RP_FILL1 0.75f       /* ... by here (R units) ... */
#define RP_FILLCAP0 0.80f    /* ... and dark counted only up to this x R from the centre, fading out ... */
#define RP_FILLCAP1 0.90f    /* ... by here, so lashes past the iris are never reached */
#define RP_CLEAR0 0.75f      /* an edge point needs this much bright sclera just beyond it ... */
#define RP_CLEAR1 0.95f      /* ... to count fully */
#define RP_MAXPTS 256        /* edge points */
#define RP_TAU 0.06          /* circle refit: points this far off the circle (R units) count half */
#define RP_SIG0 0.06f        /* circle fit's nominal error (R units): trusted fully below this ... */
#define RP_SIG1 0.15f        /* ... not at all above */

typedef struct {
	float c0x, c0y; /* corner midpoint */
	float ux, uy;   /* along the eye, unit */
	float nx, ny;   /* across the eye, towards the lower lid */
	float ihw;      /* 1 / half width */
	float up[2], lo[2]; /* lid curves: upper -(1-t^2)(up0 + up1 t), lower (1-t^2)(lo0 + lo1 t) */
} rp_frame;

static inline float rp_ss(float e0, float inv, float x) /* smoothstep, inv = 1 / (e1 - e0) */
{
	float t = (x - e0) * inv;
	t = t < 0 ? 0 : (t > 1 ? 1 : t);
	return t * t * (3 - 2 * t);
}

/* For a point (dx, dy) from the corner midpoint: how far below the upper lid (du) and above the
 * lower lid (dl) it is, in pixels across the eye. 0 outside the corners' span. */
static inline int rp_lids(const rp_frame *e, float dx, float dy, float *du, float *dl)
{
	const float t = (dx * e->ux + dy * e->uy) * e->ihw, b = dx * e->nx + dy * e->ny;
	if (!(t > -1 && t < 1))
		return 0;
	const float f = 1 - t * t;
	*du = b + f * (e->up[0] + e->up[1] * t);
	*dl = f * (e->lo[0] + e->lo[1] * t) - b;
	return 1;
}

/* Soft outline mask from rp_lids' distances, cells of size fs. */
static inline float rp_mask(float du, float dl, float ifs)
{
	return clampf(du * ifs + 0.5f, 0, 1) * clampf(dl * ifs + 0.5f, 0, 1);
}

/* Weighted circle fit through edge points (Kasa: x^2 + y^2 = A x + B y + C, centre (A/2, B/2)),
 * refitted a few times with Cauchy weights on the distance from the circle, so a few stray points
 * (a reflection reaching the iris edge, lashes touching it) lose their pull smoothly. Coordinates in
 * iris radii. Returns 0 if degenerate; else the centre, the sum of the final weights and how well
 * the centre's height is pinned down: its error for an edge error of 0.03 radii per point. */
static int rp_circle(const float *px, const float *py, const float *om, int n, float *cx, float *cy, float *wsum,
		     float *sig)
{
	double a = 0, b = 0, r = 0;
	int ok = 0;
	for (int it = 0; it < 4; it++) {
		double S[3][3] = {{0}}, Y[3] = {0}, ws = 0;
		for (int i = 0; i < n; i++) {
			double wi = om[i];
			if (it) {
				const double d = (hypot(px[i] - a, py[i] - b) - r) / RP_TAU;
				wi /= 1 + d * d;
			}
			const double v = (double)px[i] * px[i] + (double)py[i] * py[i], r3[3] = {px[i], py[i], 1};
			for (int u = 0; u < 3; u++) {
				for (int q = 0; q < 3; q++)
					S[u][q] += wi * r3[u] * r3[q];
				Y[u] += wi * r3[u] * v;
			}
			ws += wi;
		}
		const double a11 = S[0][0], a12 = S[0][1], a13 = S[0][2], a22 = S[1][1], a23 = S[1][2], a33 = S[2][2];
		const double c11 = a22 * a33 - a23 * a23, c12 = a13 * a23 - a12 * a33, c13 = a12 * a23 - a13 * a22;
		const double c22 = a11 * a33 - a13 * a13, c23 = a12 * a13 - a11 * a23, c33 = a11 * a22 - a12 * a12;
		const double det = a11 * c11 + a12 * c12 + a13 * c13;
		if (!(det > 1e-9))
			break;
		const double A = (c11 * Y[0] + c12 * Y[1] + c13 * Y[2]) / det;
		const double B = (c12 * Y[0] + c22 * Y[1] + c23 * Y[2]) / det;
		const double C = (c13 * Y[0] + c23 * Y[1] + c33 * Y[2]) / det;
		a = 0.5 * A;
		b = 0.5 * B;
		r = sqrt(fmax(C + a * a + b * b, 0));
		*cx = (float)a;
		*cy = (float)b;
		*wsum = (float)ws;
		*sig = (float)(0.03 * sqrt(c22 / det));
		ok = 1;
	}
	return ok;
}

cf_pt cf_refine_pupil(const uint8_t *grey, int stride, int width, int height, const cf_pt *eye, int count,
		      cf_pt guess)
{
	if (!grey || !eye || count < 4 || count > CF_MAX_EYE_POINTS || width <= 0 || height <= 0 || stride < width)
		return guess;

	/* 1. Frame and lids. Corners: the farthest pair of outline points. */
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
	if (!(w >= 8 && w < 4096))
		return guess;
	rp_frame e;
	e.ux = (eye[ib].x - eye[ia].x) / w;
	e.uy = (eye[ib].y - eye[ia].y) / w;
	if (e.ux < 0) {
		e.ux = -e.ux;
		e.uy = -e.uy;
	}
	e.nx = -e.uy;
	e.ny = e.ux;
	e.ihw = 2 / w;
	e.c0x = 0.5f * (eye[ia].x + eye[ib].x);
	e.c0y = 0.5f * (eye[ia].y + eye[ib].y);
	/* Least squares for each lid's two coefficients from the points on its side. */
	double M[2][5] = {{0}};
	int nside[2] = {0, 0};
	float lo = 0, hi = 0;
	for (int k = 0; k < count; k++) {
		const float dx = eye[k].x - e.c0x, dy = eye[k].y - e.c0y;
		const float t = (dx * e.ux + dy * e.uy) * e.ihw, b = dx * e.nx + dy * e.ny;
		lo = fminf(lo, b);
		hi = fmaxf(hi, b);
		if (k == ia || k == ib || !(fabsf(t) < 0.97f) || b == 0)
			continue;
		const int z = b > 0;
		const double f0 = 1.0 - (double)t * t, f1 = f0 * t, bb = fabs((double)b);
		M[z][0] += f0 * f0;
		M[z][1] += f0 * f1;
		M[z][2] += f1 * f1;
		M[z][3] += f0 * bb;
		M[z][4] += f1 * bb;
		nside[z]++;
	}
	if (!nside[0] || !nside[1])
		return guess;
	float *coef[2] = {e.up, e.lo};
	for (int z = 0; z < 2; z++) {
		const double det = M[z][0] * M[z][2] - M[z][1] * M[z][1];
		if (nside[z] >= 2 && det > 1e-3 * M[z][0] * M[z][2]) {
			coef[z][0] = (float)((M[z][3] * M[z][2] - M[z][4] * M[z][1]) / det);
			coef[z][1] = (float)((M[z][4] * M[z][0] - M[z][3] * M[z][1]) / det);
		} else {
			coef[z][0] = (float)(M[z][3] / M[z][0]);
			coef[z][1] = 0;
		}
	}
	/* Nearly closed: Vision's point, faded in (a switch would be a jump). */
	float conf = smoothstepf(RP_OPEN0, RP_OPEN1, (hi - lo) / w);
	if (!(conf > 0))
		return guess;

	/* Grid: the lids' bounding box in s x s pixel blocks (block means: a big eye is low-passed, not
	 * aliased; blocks aligned to multiples of s, so they don't move with the eye). */
	float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f;
	for (int k = 0; k <= 16; k++) {
		const float t = -1 + (float)k / 8.0f, f = 1 - t * t;
		const float bs[2] = {-f * (e.up[0] + e.up[1] * t), f * (e.lo[0] + e.lo[1] * t)};
		for (int z = 0; z < 2; z++) {
			const float x = e.c0x + 0.5f * w * t * e.ux + bs[z] * e.nx;
			const float y = e.c0y + 0.5f * w * t * e.uy + bs[z] * e.ny;
			minx = fminf(minx, x);
			maxx = fmaxf(maxx, x);
			miny = fminf(miny, y);
			maxy = fmaxf(maxy, y);
		}
	}
	if (!(maxx > 0 && maxy > 0 && minx < (float)width && miny < (float)height && maxx - minx < 2 * w &&
	      maxy - miny < 2 * w))
		return guess;
	int s = 1 + (int)(w / RP_CELLS_W), x0, y0, gw, gh;
	for (;;) {
		x0 = (int)fmaxf(minx - 1, 0) / s * s;
		y0 = (int)fmaxf(miny - 1, 0) / s * s;
		const int x1 = (int)fminf(ceilf(maxx + 1), (float)width), y1 = (int)fminf(ceilf(maxy + 1), (float)height);
		gw = (x1 - x0 + s - 1) / s;
		gh = (y1 - y0 + s - 1) / s;
		if (gw < 3 || gh < 3)
			return guess;
		if (gw * gh <= RP_MAXCELLS && gw <= RP_MAXROW)
			break;
		s++;
	}
	const float fs = (float)s, ifs = 1 / fs;
	float val[RP_MAXCELLS]; /* block mean grey, then darkness (0 outside the outline) */
	float qw[RP_MAXCELLS];  /* outline mask x lid bands (away from the lash line) */
	if (s == 1) {
		for (int j = 0; j < gh; j++) {
			const uint8_t *row = grey + (size_t)(y0 + j) * (size_t)stride + x0;
			for (int i = 0; i < gw; i++)
				val[j * gw + i] = row[i];
		}
	} else {
		for (int j = 0; j < gh; j++) {
			const int py0 = y0 + j * s, py1 = py0 + s > height ? height : py0 + s;
			for (int i = 0; i < gw; i++) {
				const int px0 = x0 + i * s, px1 = px0 + s > width ? width : px0 + s;
				unsigned sum = 0;
				for (int y = py0; y < py1; y++) {
					const uint8_t *row = grey + (size_t)y * (size_t)stride;
					for (int x = px0; x < px1; x++)
						sum += row[x];
				}
				val[j * gw + i] = (float)sum / (float)((py1 - py0) * (px1 - px0));
			}
		}
	}
	/* Cell centres relative to the corner midpoint: (gx0 + i fs, gy0 + j fs). */
	const float gx0 = (float)x0 + 0.5f * fs - e.c0x, gy0 = (float)y0 + 0.5f * fs - e.c0y;

	/* 2. Grey levels: weighted histogram inside the outline, away from the lash line. */
	const float U0 = RP_UP0 * w, iU = 1 / ((RP_UP1 - RP_UP0) * w);
	const float B0 = RP_LO0 * w, iB = 1 / ((RP_LO1 - RP_LO0) * w);
	float hist[RP_NB] = {0}, qsum = 0;
	for (int j = 0; j < gh; j++) {
		const float dy = gy0 + (float)j * fs;
		for (int i = 0; i < gw; i++) {
			const int c = j * gw + i;
			float du, dl, m = 0, q = 0;
			if (rp_lids(&e, gx0 + (float)i * fs, dy, &du, &dl)) {
				m = rp_mask(du, dl, ifs);
				if (m > 0)
					q = m * rp_ss(U0, iU, du) * rp_ss(B0, iB, dl);
			}
			qw[c] = q;
			if (m <= 0)
				val[c] = -1; /* outside the outline */
			if (q <= 0)
				continue;
			const float v = val[c] * (RP_NB / 256.0f) - 0.5f; /* linear split between bins */
			const int k = (int)v;
			if (v <= 0) {
				hist[0] += q;
			} else if (k >= RP_NB - 1) {
				hist[RP_NB - 1] += q;
			} else {
				const float fr = v - (float)k;
				hist[k] += q * (1 - fr);
				hist[k + 1] += q * fr;
			}
			qsum += q;
		}
	}
	const float R = RP_IRIS_R * w;
	if (!(qsum * fs * fs >= 0.3f * R * R))
		return guess;
	float lev[2];
	const float pct[2] = {RP_P_IRIS, RP_P_SCLERA};
	for (int z = 0; z < 2; z++) {
		const float target = pct[z] * qsum;
		float cum = 0;
		int k = 0;
		while (k < RP_NB - 1 && cum + hist[k] < target)
			cum += hist[k++];
		const float fr = hist[k] > 0 ? clampf((target - cum) / hist[k], 0, 1) : 0.5f;
		lev[z] = ((float)k + fr) * (256.0f / RP_NB);
	}
	const float contrast = lev[1] - lev[0];
	conf *= smoothstepf(RP_CONTRAST0, RP_CONTRAST1, contrast); /* nothing to see: Vision's point */
	if (!(conf > 0))
		return guess;
	const float th0 = lev[0] + RP_RAMP0 * contrast, ith = 1 / ((RP_RAMP1 - RP_RAMP0) * contrast);
	for (int c = 0; c < gw * gh; c++)
		val[c] = val[c] < 0 ? 0 : 1 - clampf((val[c] - th0) * ith, 0, 1);

	/* 3. Iris column: mean shift along the eye on the darkness profile (away from the lash line),
	 * from Vision's point. */
	float prof[2 * RP_MAXROW + 8];
	const int np = (int)(w * ifs) + 4;
	if (np > 2 * RP_MAXROW + 8)
		return guess;
	memset(prof, 0, sizeof(float) * (size_t)np);
	const float pa0 = -0.5f * w - fs; /* profile bin k is at pa0 + k fs along the eye */
	for (int j = 0; j < gh; j++) {
		const float dy = gy0 + (float)j * fs;
		for (int i = 0; i < gw; i++) {
			const int c = j * gw + i;
			const float d = val[c] * qw[c];
			if (d <= 0)
				continue;
			const float p = ((gx0 + (float)i * fs) * e.ux + dy * e.uy - pa0) * ifs;
			const int k = (int)p;
			if (p >= 0 && k + 1 < np) {
				const float fr = p - (float)k;
				prof[k] += d * (1 - fr);
				prof[k + 1] += d * fr;
			}
		}
	}
	float xc = (guess.x - e.c0x) * e.ux + (guess.y - e.c0y) * e.uy;
	if (!isfinite(xc))
		return guess;
	xc = clampf(xc, -0.3f * w, 0.3f * w);
	const float H = 1.7f * R, iH = 2 / H; /* kernel: flat to H/2, smooth to 0 at H */
	for (int it = 0; it < 4; it++) {
		double sw = 0, sa = 0;
		int k0 = (int)floorf((xc - H - pa0) * ifs), k1 = (int)ceilf((xc + H - pa0) * ifs);
		k0 = k0 < 0 ? 0 : k0;
		k1 = k1 > np - 1 ? np - 1 : k1;
		for (int k = k0; k <= k1; k++) {
			const float a = pa0 + (float)k * fs;
			const float kk = 1 - rp_ss(0.5f * H, iH, fabsf(a - xc));
			sw += prof[k] * kk;
			sa += prof[k] * kk * a;
		}
		if (!(sw > 1e-3))
			break;
		xc = (float)(sa / sw);
	}

	/* 4. Darkness centroid across the eye in the iris columns. */
	const float Hc = 1.6f * R, iHc = 4 / Hc; /* flat to 0.75 Hc */
	double cs = 0, cb = 0;
	for (int j = 0; j < gh; j++) {
		const float dy = gy0 + (float)j * fs;
		for (int i = 0; i < gw; i++) {
			const int c = j * gw + i;
			const float d = val[c] * qw[c];
			if (d <= 0)
				continue;
			const float dx = gx0 + (float)i * fs;
			const float kk = 1 - rp_ss(0.75f * Hc, iHc, fabsf(dx * e.ux + dy * e.uy - xc));
			cs += d * kk;
			cb += d * kk * (dx * e.nx + dy * e.ny);
		}
	}
	conf *= smoothstepf(0.05f, 0.12f, (float)cs * fs * fs / (3.14159265f * R * R)); /* no iris seen */
	if (!(conf > 0) || !(cs > 0))
		return guess;
	const float bc = (float)(cb / cs);
	float by = bc;

	/* 5. Limbus edge points per grid row, and a circle through them. */
	{
		const float X0 = e.c0x + xc * e.ux + bc * e.nx, Y0 = e.c0y + xc * e.uy + bc * e.ny; /* centre so far */
		const float pc = (X0 - (float)x0) * ifs - 0.5f; /* X0 in cell index units (centres at integers) */
		const int ic = (int)floorf(pc), icx = (int)floorf(pc + 0.5f);
		const float fr = pc - (float)ic;
		const float pw = Hc * ifs, pn = 0.1f * R * ifs, pf = 0.4f * R * ifs; /* window, clearance */
		const int nf = (int)ceilf(RP_FILL_R * R * ifs);                      /* bridging reach, cells */
		const int nwin = (int)ceilf(pw + pf) + nf + 3;
		const int lo_i = ic - nwin < 0 ? 0 : ic - nwin, hi_i = ic + 1 + nwin > gw - 1 ? gw - 1 : ic + 1 + nwin;
		const float EU0 = RP_EUP0 * w, iEU = 1 / ((RP_EUP1 - RP_EUP0) * w);
		const float EL0 = RP_ELO0 * w, iEL = 1 / ((RP_ELO1 - RP_ELO0) * w);
		const float iR = 1 / R;
		/* Per column: how much a cell is bridged (fz), and how much of its darkness may bridge
		 * cells nearer the centre (fc). Both fade with the distance from the centre column. */
		float fz[RP_MAXROW], fc[RP_MAXROW];
		{
			const float z0 = RP_FILL0 * R * ifs, iz = 1 / ((RP_FILL1 - RP_FILL0) * R * ifs);
			const float k0 = RP_FILLCAP0 * R * ifs, ik = 1 / ((RP_FILLCAP1 - RP_FILLCAP0) * R * ifs);
			for (int i = lo_i; i <= hi_i; i++) {
				fz[i] = 1 - rp_ss(z0, iz, fabsf((float)i - pc));
				fc[i] = 1 - rp_ss(k0, ik, fabsf((float)i - pc));
			}
		}
		float ptx[RP_MAXPTS], pty[RP_MAXPTS], ptw[RP_MAXPTS];
		int npt = 0;
		for (int j = 0; ic >= 0 && ic + 1 < gw && j < gh; j++) {
			const float Yj = (float)(y0 + j * s) + 0.5f * fs, dy = Yj - e.c0y;
			float du, dl;
			/* The row must cross the iris itself: not far from the centre so far, and its centre
			 * column clear of the lid bands (lashes are dark too) ... */
			float core = 1 - smoothstepf(1.0f * R, 1.25f * R, fabsf(Yj - Y0));
			core *= rp_lids(&e, X0 - e.c0x, dy, &du, &dl) ? rp_ss(EU0, iEU, du) * rp_ss(EL0, iEL, dl) : 0;
			if (core <= 0)
				continue;
			const float *src = val + j * gw;
			float row[RP_MAXROW], mr[RP_MAXROW];
			for (int i = lo_i; i <= hi_i; i++) {
				mr[i] = rp_lids(&e, gx0 + (float)i * fs, dy, &du, &dl) ? rp_mask(du, dl, ifs) : 0;
				row[i] = src[i] * mr[i];
			}
			/* ... and be dark at the centre column, bright patches there bridged (dark both sides). */
			{
				float ml = row[ic], mg = row[ic + 1];
				for (int k = 1; k <= nf; k++) {
					if (ic - k >= lo_i && row[ic - k] > ml)
						ml = row[ic - k];
					if (ic + 1 + k <= hi_i && row[ic + 1 + k] > mg)
						mg = row[ic + 1 + k];
				}
				core *= smoothstepf(0.5f, 0.85f, fmaxf(row[ic] * (1 - fr) + row[ic + 1] * fr, fminf(ml, mg)));
			}
			if (core <= 0)
				continue;
			for (int z = 0; z < 2; z++) {
				const int step = z ? 1 : -1;
				/* Dark width out from the centre within the window: running minimum, so dark
				 * things past a gap of sclera don't count. Cells cut by the centre or the window
				 * end count by their coverage, also in the minimum. Near the centre a cell counts
				 * as dark as the darkest cell within reach further out (bright patches inside the
				 * iris bridged; the outer edge itself is not moved, nothing past it is dark). */
				const float lim = z ? pc + pw : pc - pw;
				float W = 0, rm = 1;
				for (int i = icx; i >= lo_i && i <= hi_i; i += step) {
					const float cov = fminf((float)i + 0.5f, z ? lim : pc) - fmaxf((float)i - 0.5f, z ? pc : lim);
					if (cov <= 0) {
						if (i != icx)
							break;
						continue;
					}
					float d = row[i];
					if (fz[i] > 0) {
						float m = d;
						for (int k = 1, ii = i + step; k <= nf && ii >= lo_i && ii <= hi_i; k++, ii += step)
							m = fmaxf(m, row[ii] * fc[ii]);
						d += fz[i] * (m - d);
					}
					const float de = 1 - cov * (1 - d);
					rm = de < rm ? de : rm;
					W += rm * cov;
				}
				/* Sclera just beyond the edge: inside the outline and bright. */
				const float pe = pc + (float)step * W;
				const float r0 = z ? pe + pn : pe - pf, r1 = z ? pe + pf : pe - pn;
				float clear = 0;
				for (int i = (int)floorf(r0 + 0.5f); i <= (int)ceilf(r1 - 0.5f); i++) {
					if (i < lo_i || i > hi_i)
						continue;
					const float cov = fminf((float)i + 0.5f, r1) - fmaxf((float)i - 0.5f, r0);
					if (cov > 0)
						clear += (mr[i] - row[i]) * cov;
				}
				clear /= r1 - r0;
				const float Wpx = W * fs;
				/* The lids (landmarks) clear of the edge point too. */
				float band = 0;
				if (rp_lids(&e, X0 + (float)step * Wpx - e.c0x, dy, &du, &dl))
					band = rp_ss(EU0, iEU, du) * rp_ss(EL0, iEL, dl);
				const float om = core * band * smoothstepf(RP_CLEAR0, RP_CLEAR1, clear) *
						 smoothstepf(0.25f * R, 0.45f * R, Wpx);
				if (om <= 0 || npt >= RP_MAXPTS)
					continue;
				ptx[npt] = (float)step * Wpx * iR; /* R units around (X0, Y0) */
				pty[npt] = (Yj - Y0) * iR;
				ptw[npt++] = om;
			}
		}
		float fcx, fcy, fw, sig;
		if (npt >= 3 && rp_circle(ptx, pty, ptw, npt, &fcx, &fcy, &fw, &sig)) {
			/* Few rows (squinting) or one short arc: less trusted. */
			const float g = (1 - smoothstepf(RP_SIG0, RP_SIG1, sig)) * smoothstepf(3, 6, fw);
			const float fx = X0 + fcx * R - e.c0x, fy = Y0 + fcy * R - e.c0y;
			const float bf = clampf(fx * e.nx + fy * e.ny, bc - R, bc + R);
			if (isfinite(bf))
				by = bc + g * (bf - bc);
		}
	}

	/* 6. Result, faded towards Vision's point when unsure, or far from it. */
	cf_pt res = {e.c0x + xc * e.ux + by * e.nx, e.c0y + xc * e.uy + by * e.ny};
	conf *= 1 - smoothstepf(0.3f, 0.45f, hypotf(res.x - guess.x, res.y - guess.y) / w);
	if (!(conf > 0) || !isfinite(res.x) || !isfinite(res.y))
		return guess;
	res.x = guess.x + conf * (res.x - guess.x);
	res.y = guess.y + conf * (res.y - guess.y);
	return res;
}
