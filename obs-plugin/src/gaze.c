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
#define FAR_MARGIN 0.04f      /* eye widths: the far fade starts at least this far past the calibrated notes ... */
#define FAR_NOISE_K 5.0f      /* ... and this many times the landmark noise: a small notes calibration (notes just
			       * under the lens, or a lens step taken looking a little low) must not switch the
			       * correction off for reading a little lower, or for noise */
#define FAR_NOISE_TAU 10.0    /* s: (that noise's recent peak, forgotten this slowly: the fade start holds still) */
#define CALIB_OPEN_K 3.0f     /* calibration frames whose openness is off its median by this x its spread ... */
#define CALIB_OPEN_MIN 0.02f  /* ... and by this at least (height / width) are left out */
#define CALIB_DRIFT_MAX 0.02f /* eye widths: a calibration whose halves differ more than this is unsteady ... */
#define CALIB_SPREAD_MAX 0.04f /* ... and so is one whose samples spread more than this ... */
#define CALIB_OPEN_SPREAD_MAX 0.015f /* ... or whose openness does (RMS, height / width: 3x its landmark noise) */
#define BLINK_ENTER 0.40f     /* relative openness (vs. open-eye) below which the lids count as closed */
#define BLINK_EXIT 0.50f      /* ... and above which the blink is over (hysteresis) */
#define CLOSING_RATE 5.0f     /* relative openness per second: lids dropping this fast may be a blink */
#define CLOSING_MAX 0.15      /* s: lids that stop dropping above BLINK_ENTER are a glance or a squint */
#define BLINK_HOLD_MAX 0.40   /* s: a "blink" longer than this is eyes closed: fade out */
#define SETTLE_TIME 0.07      /* s: ignore the pupil this long after the lids reopen ... */
#define SETTLE_OPEN 0.85f     /* ... unless (or until) they are back this open (relative): measured from there */
#define REACQ_TIME 0.25       /* s: after a blink, a change taken this soon is shown as a glide ... */
#define GLIDE_TIME 0.15       /* ... over this long (an S-curve), never as a jump with the eyes open */
#define LID_GLANCE_K 0.30f    /* eye widths of downward gaze per unit of lid drop (far-glance cue only) */
#define LID_SHIFT 0.45f       /* upper lid follows the iris by this fraction (real lids track vertical gaze) */
#define AMOUNT_MAX_RY 0.38f   /* vertical iris shift limit, fraction of ry (well inside the fold limit) */
#define HSHIFT_MAX_RX 0.30f   /* sideways iris shift limit, fraction of rx */
#define READ_DAMP 0.65f       /* fraction of the sideways reading excursion cancelled at full strength */
#define SIDE_FADE_LO 0.30f    /* sideways distance from the notes centre (eye widths) where fading starts */
#define SIDE_FADE_HI 0.45f
#define FADE_IN_TAU 0.15      /* s: per stage of the fade in (10-90% in ~0.5 s) ... */
#define FADE_OUT_TAU 0.10     /* s: ... and out (~0.35 s): looking away, back, turning the head */
#define REOPEN_FADE_TAU 0.04  /* s: per stage, back in after eyes closed for long (10-90% in ~0.13 s) */
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
#define LID_VIS_RATIO 0.5f    /* ... and Vision's point at least this fraction of it (it under-reads a glance
			       * by less than that; a lid drags it far less than the lid moves) */
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
#define VIS_CONSIST 0.6f      /* Vision's own pupil point backs a saccade up when it moved the same way by this
			       * fraction of what v_gain says it should (consistent in size; the gain taken as
			       * up to twice its typical error larger) ... */
#define VIS_FRAC_MIN 0.2f     /* ... and by this fraction of the change at least (coarse as it is, it never
			       * under-reads a glance by 5x) ... */
#define VIS_K 3.0f            /* ... and by more than this x its own frame-to-frame noise ... */
#define VIS_MIN 0.008f        /* ... and at least this (eye widths: 0.7 px on an 86 px eye) */
#define VIS_NOISE_INIT 0.003f
#define LIDC_FRAC 0.3f        /* (no Vision points) the upper lid backs up a vertical change when it moved with
			       * it by this fraction of it ... */
#define LIDC_K 3.0f           /* ... and by more than this x its noise */
#define ANCHOR_TAU 0.3        /* s: between jumps, "where the backing-up sensors were" follows them this fast ... */
#define ANCHOR_HOLD 1.0       /* ... and Vision's point this slowly just after it jumped (by more than its noise
			       * in a frame), VIS_HOLD long: the refined reading may show that a frame or two late */
#define VIS_HOLD 0.15         /* s */
#define CLEAN_TIME 1.5        /* s: the measurement counts as clean this long after the last outlier */
#define PERSIST_LONG 5        /* frames an unbacked change must stay put while outliers are about */
#define VOTE_FRAMES 24.0f     /* frames: memory of the vote between two levels (see gate()) ... */
#define VOTE_SLOW 72.0f       /* ... and of the longer vote, which must agree ... */
#define VOTE_Z 1.0f           /* ... by this many standard deviations of chance (see vote()) */
#define VIS_AGREE 0.35f       /* a refined reading this far (x the move) from where Vision's point says the
			       * gaze went is not taken with the saccade: Vision's estimate is */
#define LANDING_FRAMES 3      /* frames after a jump that still count as landing it */
#define VIS_MID_K 1.5f        /* a move short of a saccade's size needs Vision's point to move this x its threshold */
#define EYES_K 6.0f           /* (no Vision) both eyes back a change up only past this x the glance noise */
#define VOTE_RATIO 1.3f       /* the other level wins with this many times the accepted level's share ... */
#define VOTE_MIN 0.3f         /* ... and at least this share of the recent readings */
#define VOTE_NEW 0.1f         /* a reading at neither level replaces an other level weaker than this */
#define VIS_GAIN_INIT 2.0f    /* refined change per Vision change until calibrated or learned (seen: 1-3.3) ... */
#define VIS_GAIN_ERR_INIT 0.6f /* ... and its typical (log) error until then: anything in that range */
#define VIS_GAIN_ERR_CAL 0.08f /* ... and once calibrated (see use_vis_calib()) */
#define VIS_GAIN_ERR_MIN 0.04f /* learning never makes it surer than this ... */
#define VIS_GAIN_Q 0.002f     /* ... and each saccade learnt from leaves this much (log, squared) room to change */
#define VIS_GAIN_MIN 0.5f
#define VIS_GAIN_MAX 5.0f
#define VIS_CAL_MIN 0.004f    /* eye widths: Vision's point must have moved this much between the calibrations ... */
#define VIS_CAL_MIN_FRAC 0.1f /* ... and this fraction of the refined move, for a usable gain */
#define LEARN_N 6             /* readings averaged in after a backed saccade before its size is learnt from ... */
#define LEARN_MIN_HI 2.0f     /* ... if it moved this x the saccade size at least: a smaller one is only taken when
			       * its reading overshot (noise past the saccade size), and would teach a gain too big */
#define VIS_CAL_DR_ERR 0.004f /* eye widths: typical error of the refined move between the two calibrations */
#define VIS_OFF_TAU 3.0       /* s: memory of the refined-minus-Vision offset between changes (see gate()) ... */
#define VIS_RES_TAU 0.5       /* ... unless the readings' recent mean (over this) keeps off it ... */
#define VIS_RES_K 3.0f        /* ... by this x its standard error: then it follows over VIS_RES_TAU */
#define VIS_PEND_TIME 0.1     /* s: a jump of Vision's point alone waits this long for the refined reading */
#define LR_TAU_OUT 10.0       /* s: the eyes disagreeing past the outlier threshold is learnt this slowly */
#define FACE_GONE 1.0         /* s: the face missing this long: what Vision's point said before is stale */
#define REVERT_TIME 0.15      /* s: a jump undone this soon was an outlier */
#define VIS_REVERT_TIME 0.4   /* s: a jump Vision's point backed, undone by both sensors this soon, is undone at once */
#define SLEW_PX 0.3f          /* px per 1/30 s: fastest the shown gaze moves without a backed-up saccade */
#define SIDE_LO 0.02f         /* eye widths: sideways noise band (or 2x the noise, up to twice this) ... */
#define SIDE_HI 0.06f         /* ... and saccade size (or twice the band) */
#define SIDE_CUTOFF 3.0       /* Hz: low-pass for noise-sized sideways changes */
#define DOWN_OUT_CUTOFF 1.5   /* Hz: second low-pass stage on what the output shows, vertical ... */
#define SIDE_OUT_CUTOFF 2.0   /* ... and sideways (see slew()) */

/* ---- Tuning: head and lids (see head_lids()) */
#define EYE_K 0.41f           /* eyeball radius / eye width (12 mm / 29 mm): an eye turning by a moves the iris
			       * EYE_K sin(a) eye widths against its corners */
#define HEAD_PITCH_MAX 18.0f  /* deg: a head tilted down is made up for up to here (CF_PITCH_FADE_LO) */
#define HEAD_YAW_K 0.5f       /* share of the head's turn made up for sideways */
#define HEAD_YAW_MAX 15.0f    /* deg: ... up to here (CF_YAW_FADE_LO) */
#define HEAD_MIN_CUTOFF 0.5   /* Hz: One Euro smoothing of the head's angles while it is still ... */
#define HEAD_BETA 0.2         /* ... and how much quicker per deg/s it moves (head_angles()) */
#define HEAD_SLEW_PX 0.4f     /* px per 1/30 s: fastest a head term moves the iris */
#define LID_TAU 0.15          /* s: smoothing of the lid heights, each of two stages */
#define LID_SLEW_PX 0.12f     /* px per 1/30 s: fastest the lid restoration moves a lid */
#define LID_DEAD 0.002f       /* eye widths: lid changes this small are landmark noise, not restored */
#define UP_DROP_MAX 0.035f    /* eye widths: the most upper-lid drop restored (~2.6 px on a 75 px eye) ... */
#define LO_RISE_MAX 0.025f    /* ... and lower-lid rise */
#define VOR_TAU 0.5           /* s: how long the head's move can explain a catch-up of the glance ... */
#define VOR_GLIDE_PX 0.2f     /* ... px per 1/30 s it is glided in at (glide_catch_up()) ... */
#define VOR_RAW_TAU 0.03      /* s: ... and smoothing of the raw glance it is checked against */

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
		/* An outlier moves it no more than noise would, and far more slowly: one eye's point departing
		 * from the other's (slipping) for seconds is not learnt as their usual difference, which would
		 * then average the two again; the other eye stays the one used. */
		const float k = (float)(1.0 - exp(-dt / LR_TAU)), k_out = (float)(1.0 - exp(-dt / LR_TAU_OUT));
		c->lr_bias += (*outlier ? k_out : k) * clampf(d, -thr, thr);
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
	int lid_drag;           /* a lid is dragging the pupil points this frame: nothing backs a change up */
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

/* Vision's gain from one backed saccade: the refined reading moved dx (with variance var_dx) and Vision's dv.
 * The measure is weighed by how sure it is against how sure the gain already is (a Kalman filter on its
 * log): the first saccades teach an uncalibrated gain most of what they show, later ones (and a calibrated
 * gain) only adjust it. One far off what the gain says, once that is known, is something else (a level the
 * refiner flipped to, a lid) and teaches nothing. The offset and the other levels keep where they are
 * (v: Vision's reading now). */
static void learn_gain(cf_robust *c, float dx, float dv, float var_dx, float v)
{
	if (!(dx * dv > 0))
		return;
	const float r = logf(dx / (c->v_gain * dv)), var_r = var_dx / (dx * dx), pe = c->v_gain_err * c->v_gain_err;
	if (!(r * r <= 9.0f * (pe + var_r)))
		return;
	const float kg = pe / (pe + var_r);
	const float g1 = clampf(c->v_gain * expf(kg * r), VIS_GAIN_MIN, VIS_GAIN_MAX);
	c->v_gain_err = fmaxf(sqrtf((1.0f - kg) * pe + VIS_GAIN_Q), VIS_GAIN_ERR_MIN);
	c->v_off += (c->v_gain - g1) * c->v_ref;
	for (int j = 0; j < 2; j++)
		c->alt[j] += (c->v_gain - g1) * v;
	c->v_gain = g1;
}

/*
 * Moves the accepted value f->value towards this frame's combined reading m. Changes inside the noise
 * band are low-passed (with Vision's point, as a reading it makes steadier: see mf). A saccade-sized
 * change is taken at once only when it is backed up:
 *  - Vision's own pupil point (if given) moved the same way, by about as much as its gain says it should
 *    (see v_need). Vision's point is coarse but never jumps the way the refined one can, so with it
 *    nothing else may back a change up;
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
	int jump = 0, big = 0, switched = 0, landed = 0, averaged = 0, away = 0, followed = 0, by_v = 0, mid = 0;
	const float g0 = c->v_gain, u = in->vis ? g0 * in->v : 0.0f;
	const float v_thr = fmaxf(VIS_K * c->v_noise, VIS_MIN);
	const int vis = in->vis && c->have_v; /* Vision's point now, and last frame */
	const int have_off = vis && c->v_off_n > 0;
	/* A gap with no readings (a stalled camera, the face lost a moment) starts afresh from this reading,
	 * unless Vision's point is known from before it: the change since is then checked like any other. */
	const int init = !f->ready || (t - f->t > 0.5 && !vis);
	if (in->vis != c->have_v)
		new_level(c); /* the levels' reference changed */
	if (!have_off)
		c->v_pend = 0;
	/* Vision's point jumped (past its noise, in one frame) and the two eyes' readings did not both move
	 * with it by half as much (with the gain taken as up to twice its typical error smaller: an unsure
	 * gain may overstate the move): for VIS_PEND_TIME that jump is not believed, and the reading it steadies
	 * (mf) stays where it was. Vision's point refits now and then, and a lid or a smile drags it; a real
	 * move shows in the refined reading too. Then, what of it stayed is taken out of the offset between the
	 * two unless the refined readings followed it (see the end). A saccade is backed up by it all the same
	 * (see by_vision). */
	if (have_off && !init && fabsf(in->v - c->v_last) > v_thr) {
		const float step = g0 * (in->v - c->v_last), half = 0.5f * step * step * expf(-2.0f * c->v_gain_err);
		if (in->lr_out || in->e[0] * step < half || in->e[1] * step < half) {
			/* (another jump while one is pending: the wait starts again, from before the first one) */
			if (!c->v_pend)
				c->v_pend_v0 = c->v_anchor;
			c->v_pend = 1;
			c->v_pend_t = t;
			c->v_pend_vs = c->v_pend_rs = 0;
			c->v_pend_n = 0;
		}
	}
	/* Where Vision's point says the gaze is: v_gain x its reading plus the offset between the two readings,
	 * averaged since the last change taken (est; what averaging already took in of a move counts once),
	 * and the same with Vision's reading from before a jump not believed yet (base). Without the offset
	 * yet: the accepted value, moved on by what Vision's point says moved since it was taken. */
	const float est = have_off ? u + c->v_off : (vis ? x0 + g0 * (in->v - c->v_anchor) : x0);
	const float base = c->v_pend ? g0 * c->v_pend_v0 + c->v_off : est;
	if (init) {
		f->value = m;
		f->deriv = 0;
		f->ready = 1;
		c->steady_n = 0;
		c->held_big = 0;
		c->jump_weak = 0;
		c->learn = 0;
		new_level(c);
		jump = landed = 1;
	} else {
		const float e = m - x0, ae = fabsf(e);
		const int s = e > 0 ? 1 : -1;
		const int clean = t - c->out_t > CLEAN_TIME;
		/* Right after a jump the accepted value carries that one frame's noise: the next readings are
		 * averaged in (1/2, 1/3, ...: a running mean of the readings taken since, outliers left out)
		 * until that is no quicker than the usual low-pass, and for the first few frames after it
		 * anything short of a saccade counts as that noise (it is the same landing). */
		const double a_lp = fmax(in->alpha, 1.0 / (double)(c->settle + 1));
		if (c->land_n < 1000)
			c->land_n++;
		const int landing = c->land_n <= LANDING_FRAMES;
		const float lo = landing ? in->hi : in->lo;
		big = ae > in->hi;
		/* The reading stayed put since last frame, which was already away from the accepted value (this is
		 * not a one-frame excursion; a reading just past the noise band, next to last frame's one just
		 * inside it, is not "staying put"). */
		const int steady = c->away && c->have_prev && fabsf(m - c->m_prev) <= fmaxf(lo, 0.5f * ae);
		/* What a noise-sized change is averaged as. With Vision's point, the reading made steadier by it:
		 * Vision's point is several times less noisy frame to frame (in the user's clip 0.3 px against the
		 * refined 1.2-2 px), but coarse and offset from the refined one. So base puts Vision's reading
		 * where the refined one says, and the two are weighted by their noise. What is averaged then
		 * wanders far less with the refined reading's noise, and a refined outlier pulls it only by its
		 * small share; a real small move shows in both. The refined reading's share pulls no further than
		 * twice its usual noise: a refined point that departs from Vision's (one eye's point slipping, both
		 * sliding with a lid) shifts it only that far, and only the offset can follow such a departure,
		 * over seconds. Without Vision's point it is the refined reading itself, and so it is while a
		 * landing is still being averaged: the offset is only being learnt then (from the same readings). */
		float mf = m;
		const float lim = 2.0f * in->noise;
		if (have_off && a_lp <= in->alpha) {
			const float n_p = c->v_noise * g0, var_m = in->noise * in->noise, var_p = n_p * n_p;
			if (var_m + var_p > 0)
				mf = base + var_p / (var_m + var_p) * clampf(m - base, -lim, lim);
		}
		if (c->v_pend) { /* did the refined reading follow Vision's jump? */
			const float cap = fmaxf(lim, 1.5f * fabsf(est - base));
			c->v_pend_rs += clampf(m - base, -cap, cap);
			c->v_pend_vs += in->v;
			c->v_pend_n++;
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
		} else if (c->jump_vis && vis && !in->lr_out && ae > lo && t - c->jump_t <= VIS_REVERT_TIME &&
			   fabsf(m - c->jump_from) <= fmaxf(in->lo, 0.3f * fabsf(c->jump_size)) &&
			   (in->v - c->jump_v0) * (c->jump_v1 - c->jump_v0) <=
				   0.25f * (c->jump_v1 - c->jump_v0) * (c->jump_v1 - c->jump_v0)) {
			/* Both eyes' readings straight back where the last jump Vision's point backed came from, and
			 * Vision's point back at least halfway to where it was: a short look there and back, or the
			 * points dragged there and back together (a lid, a squint) that only looked like one. Either
			 * way it goes back at once, not by averaging over half a second; a gain learnt from that jump
			 * is unlearnt. */
			f->value = c->jump_from;
			c->v_gain = c->jump_gain;
			c->v_gain_err = c->jump_gain_err;
			jump = landed = 1;
			c->jump_vis = 0;
			c->held_big = 0;
			c->steady_n = 0;
		} else if (vote(c, x0, est, u, in, &vote_to)) {
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
			/* A lid dragging the points moves Vision's along (and the lid itself) the same way: no backing
			 * then; and right after a blink, not with the two eyes disagreeing either (one still dragged). */
			const int may = !in->lid_drag && !(in->resync > 0 && in->lr_out);
			/* Vision's point must have moved the same way, past its noise, and by about as much as its
			 * gain says a move this size moves it (VIS_CONSIST of it, the gain taken as up to twice its
			 * typical error larger), and VIS_FRAC_MIN of it at least: a refined point departing from it
			 * (one eye slipping, both sliding with a lid) while it moves a little is not a glance. A
			 * saccade-sized change at once; one short of that (a line change while reading: in noisy
			 * tracking the saccade size is 4 px) only when Vision's point clearly jumped too (VIS_MID_K x
			 * its noise threshold, and believed: see v_pend), both eyes moved its way, and the reading
			 * stayed there a second frame: noise that happens to line up with a small move of Vision's
			 * point (or with a refit of it) is averaged like the rest. */
			const float frac = fmaxf(VIS_CONSIST / (g0 * expf(2.0f * c->v_gain_err)), VIS_FRAC_MIN);
			const float v_need = fmaxf(frac * ae, v_thr);
			const float dvs = vis ? (in->v - c->v_anchor) * (float)s : 0.0f;
			mid = may && vis && !big && !c->v_pend && !in->lr_out && dvs >= fmaxf(v_need, VIS_MID_K * v_thr) &&
			      in->e[0] * (float)s > 0 && in->e[1] * (float)s > 0;
			const int mid_held = mid && c->mid == s && c->have_prev && fabsf(m - c->m_prev) <= fmaxf(lo, 0.5f * ae);
			/* A jump of Vision's point that the refined readings haven't followed for two frames, by half
			 * as much on average (pending, see v_pend), is backed up only by both eyes' readings following
			 * it: one eye's outlier lining up with it is no saccade. */
			const int stale = c->v_pend && c->v_pend_n >= 3 &&
					  c->v_pend_rs / (float)c->v_pend_n * (float)s < 0.5f * fabsf(est - base) &&
					  (in->lr_out || in->e[0] * (float)s <= 0 || in->e[1] * (float)s <= 0);
			const int by_vision = may && vis && dvs >= v_need && (big || mid_held) && !stale;
			const int by_eyes = may && !in->vis && clean && !in->lr_out &&
					    ae > fmaxf(in->resync > 0 ? in->resync : in->hi, in->eyes_min) &&
					    in->e[0] * (float)s > in->lo && in->e[1] * (float)s > in->lo;
			const int by_lids = may && !in->vis && in->lid && c->have_lid && big &&
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
					/* Where the gaze landed: the refined reading and Vision's estimate (est), weighted
					 * by how far off each tends to be. The refined reading carries this frame's
					 * landmark noise; Vision's estimate its own noise (scaled up by v_gain) and the
					 * error of v_gain itself, which grows with the size of the move. So the landing
					 * doesn't carry one noisy frame's reading, and while the gain is unsure (not
					 * calibrated, little learnt) the refined reading decides. A reading far further
					 * from Vision's estimate than the two tend to be apart (3 sigma), or, with the two
					 * eyes' readings disagreeing (one may be an outlier landing with the saccade, e.g.
					 * a flip), far from it at all, is not mixed in: the estimate is taken alone, and
					 * the next readings settle it. */
					by_v = 1;
					c->jump_v0 = c->v_anchor;
					c->jump_v1 = in->v;
					c->jump_gain = c->v_gain;
					c->jump_gain_err = c->v_gain_err;
					const float n_p = c->v_noise * g0, n_g = c->v_gain_err * fabsf(est - x0);
					const float var_m = in->noise * in->noise, var_p = n_p * n_p + n_g * n_g;
					const int agree = !in->lr_out && fabsf(in->e[0] - in->e[1]) <= in->hi;
					const float tol = agree ? fmaxf(in->hi, 3.0f * sqrtf(var_m + var_p))
								: fmaxf(in->hi, VIS_AGREE * fabsf(est - x0) + 2.0f * sqrtf(var_p));
					if (fabsf(m - est) > tol)
						to = est;
					else if (var_m + var_p > 0)
						to = m + var_m / (var_m + var_p) * (est - m);
					/* Its size teaches v_gain: at once while the gain is unsure (from this frame's
					 * reading, if both eyes read the move alike), and again once the landing is
					 * averaged in (see the end). */
					c->learn = 1;
					c->learn_x = have_off ? c->v_off + g0 * c->v_ref : x0;
					c->learn_v = have_off ? c->v_ref : c->v_anchor;
					c->learn_n = have_off ? c->v_off_n : 1;
					if (agree && c->v_gain_err > 2.0f * VIS_GAIN_ERR_CAL &&
					    fabsf(m - c->learn_x) > LEARN_MIN_HI * in->hi)
						learn_gain(c, m - c->learn_x, in->v - c->learn_v,
							   var_m * (1.0f + 1.0f / (float)c->learn_n) + n_p * n_p, in->v);
				}
				c->jump_t = t;
				c->jump_from = x0;
				c->jump_size = to - x0;
				c->jump_weak = by_eyes && !by_lids && !back; /* (going back is not a guess) */
				c->jump_vis = by_vision && !back;
				f->value = to;
				jump = landed = 1;
				c->steady_n = 0;
				c->held_big = 0;
			} else if (in->vis && !big) {
				/* Past the noise band but short of a saccade, and Vision's point, which shows any real
				 * move this big, saw none (or not one this size): a larger draw of the noise, or the
				 * refined point departing from where the eye is. Averaged like the rest of it (as the
				 * reading Vision's point steadies), its pull capped at the band's edge, so a run of them
				 * can't carry the value off. */
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
				if (big && (!clean || in->vis)) {
					/* Saccade-sized and unbacked, while outliers are about or Vision's point (which
					 * would have shown it) saw no such move: left to the vote, which takes a level only
					 * once the readings have clearly kept going there; never followed for persisting. */
					c->held_big = 1;
				} else if (c->steady_n >= (clean ? 1 : PERSIST_LONG)) {
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
	c->mid = mid && !jump ? (m > x0 ? 1 : -1) : 0;
	c->m_prev = m;
	c->have_prev = 1;
	c->away = away;
	c->followed = followed;
	const int took = jump || switched || fabsf((float)f->value - x0) > in->hi;
	if (took && !switched && !init) {
		if (vis) {
			/* The levels are offsets from where Vision's point says the gaze is: the accepted one only
			 * changes when the gaze moved more (or less) than Vision's point says. Not with a saccade it
			 * backed up: what differs then is the error of its gain, and the other levels (a refiner's
			 * spots, at fixed offsets from where the eye is) move along with the accepted one. */
			const float o_old = est - u, o_new = (float)f->value - u;
			if (by_v) {
				for (int j = 0; j < 2; j++)
					c->alt[j] += o_new - o_old;
			} else if (fabsf(o_new - o_old) > in->hi) {
				move_level(c, o_old, o_new, in->hi);
			}
		} else {
			new_level(c); /* the gaze moved: the old level is history, not one to vote back to */
		}
	}
	/* Only a saccade shown at once has a landing to settle (a glide or a vote arrives already averaged): it
	 * counts the readings averaged in since (for the running mean) and the frames since (the landing). */
	c->settle = landed ? 1 : (averaged && c->settle < 1000 ? c->settle + 1 : c->settle);
	if (landed)
		c->land_n = 0;
	else if (switched || followed)
		c->land_n = LANDING_FRAMES + 1; /* (arrived already averaged: nothing to land) */
	if (init || switched || followed || (landed && !by_v))
		c->learn = 0; /* (that saccade's landing is no clean measure any more) */

	/* Where the backing-up sensors were: reset on a taken change, otherwise following them slowly. A jump
	 * of Vision's point well past its noise is kept a while longer (the refined reading may only show it
	 * a frame or two later, both eyes having read wrong when it happened), but only just after it jumped:
	 * Vision's point drifting off slowly must not pile up into a move that backs a later, unrelated jump
	 * of the refined point. */
	const float ka = (float)(1.0 - exp(-dt / ANCHOR_TAU));
	if (in->vis) {
		if (c->have_v && !big && !took) {
			const float kn = (float)(1.0 - exp(-dt / DOWN_NOISE_TAU));
			c->v_noise += kn * (fminf(fabsf(in->v - c->v_last), 0.03f) - c->v_noise);
		}
		if (c->have_v && fabsf(in->v - c->v_last) > v_thr)
			c->v_jump_t = t;
		const int v_moved = c->have_v && t - c->v_jump_t <= VIS_HOLD && fabsf(in->v - c->v_anchor) > v_thr;
		const float kh = v_moved ? (float)(1.0 - exp(-dt / ANCHOR_HOLD)) : ka;
		c->v_anchor = (!c->have_v || took) ? in->v : c->v_anchor + kh * (in->v - c->v_anchor);
		c->v_last = in->v;
		/* The refined-minus-Vision offset (see mf above) and Vision's mean: restarted from the accepted
		 * value on a taken change, then running means of the readings averaged in (never an outlier, or a
		 * flipping refiner's other level), each pulling the offset no further than the band's edge, until
		 * that is no quicker than VIS_OFF_TAU. Held while a jump of Vision's point is pending: once
		 * VIS_PEND_TIME is over, it is kept if the refined readings since moved half as much at least, or
		 * else taken out of the offset (and out of the anchor: it backs nothing up). */
		const float u1 = c->v_gain * in->v;
		if (took || !c->have_v || c->v_off_n == 0) {
			c->v_off = (float)f->value - u1;
			c->v_ref = in->v;
			c->v_off_n = 1;
			c->v_res = 0;
			c->v_fast = in->v;
			c->v_pend = 0;
		} else if (c->v_pend) {
			if (t - c->v_pend_t >= VIS_PEND_TIME - 1e-6 && c->v_pend_n >= 2) {
				/* What of the jump stayed (Vision's mean since, against its level before), and what the
				 * refined readings did meanwhile. */
				const float dv = c->v_pend_vs / (float)c->v_pend_n - c->v_pend_v0;
				const float dr = c->v_pend_rs / (float)c->v_pend_n, dg = c->v_gain * dv;
				if (fabsf(dv) > v_thr && dr * dg < 0.5f * dg * dg) {
					c->v_off -= dg;
					c->v_ref += dv;
					c->v_anchor = in->v; /* (nothing moved) */
				}
				c->v_pend = 0;
			}
		} else if (averaged && !in->lr_out) {
			/* Vision's point drifting off slowly while the refined one stays shows as the readings' recent
			 * mean (v_res) staying off the offset by more than its noise allows, with Vision's own recent
			 * mean (v_fast) off its longer one the matching way: the offset then follows that share of it
			 * quicker, as fast as VIS_RES_TAU once it is twice that far off. The other way round (the
			 * refined point departing, Vision's still) it doesn't: Vision's point says where the eye is,
			 * and only VIS_OFF_TAU follows the refined one. A one-off departure (a lid dragging both
			 * points, a refit) is past the band's edge, not averaged, and builds no such mean. */
			const float r = clampf(m - u1 - c->v_off, -in->hi, in->hi), n_p = c->v_noise * c->v_gain;
			const float kf = (float)(1.0 - exp(-dt / VIS_RES_TAU));
			c->v_res += kf * (r - c->v_res);
			c->v_fast += kf * (in->v - c->v_fast);
			const float se = sqrtf((in->noise * in->noise + n_p * n_p) * kf / (2.0f - kf));
			const float w = se > 0 ? clampf(fabsf(c->v_res) / (VIS_RES_K * se) - 1.0f, 0.0f, 1.0f) : 0.0f;
			const float share =
				c->v_res != 0 ? clampf(c->v_gain * (c->v_ref - c->v_fast) / c->v_res, 0.0f, 1.0f) : 0.0f;
			const float k = fmaxf(fmaxf((float)(1.0 - exp(-dt / VIS_OFF_TAU)), 1.0f / (float)(c->v_off_n + 1)),
					      w * share * kf);
			c->v_off += k * r;
			c->v_ref += k * (in->v - c->v_ref);
			c->v_off_n += c->v_off_n < 1000;
		}
		/* Learning v_gain (see learn_gain()): once LEARN_N readings since a backed saccade are averaged
		 * in, its size is measured again from the means, far surer than from one frame. */
		if (c->learn && !took && !c->v_pend && c->v_off_n >= LEARN_N) {
			c->learn = 0;
			const float x1 = c->v_off + c->v_gain * c->v_ref, dx = x1 - c->learn_x, dv = c->v_ref - c->learn_v;
			const float n_p = c->v_noise * c->v_gain;
			if (fabsf(dx) > LEARN_MIN_HI * in->hi)
				learn_gain(c, dx, dv, (in->noise * in->noise + n_p * n_p) *
							      (1.0f / (float)c->v_off_n + 1.0f / (float)c->learn_n), in->v);
		}
	} else {
		c->v_off_n = 0;
		c->v_pend = 0;
		c->learn = 0;
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

/* What the output shows (see slew()). A jump right after a blink (`reacq`) becomes a glide instead: an
 * S-curve from where the output was to the accepted value (which may still settle meanwhile), over
 * GLIDE_TIME, the first step on the frame of the jump. */
static double show(cf_robust *c, double x, int snap, int reacq, double t, double dt, float step, double alpha,
		   int pass)
{
	if (snap && reacq && c->have_shown && !c->gliding) {
		c->gliding = 1;
		c->glide_from = c->shown;
		c->glide_t0 = t;
	}
	if (!c->gliding)
		return slew(c, x, snap, step, alpha, pass);
	const double u = fmin((t - c->glide_t0 + dt) / fmax(GLIDE_TIME, dt), 1.0);
	c->shown = c->glide_from + u * u * (3.0 - 2.0 * u) * (x - c->glide_from);
	c->shown_x = x;
	c->gliding = u < 1.0;
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
	float upper, lower; /* upper lid above, lower lid below the corner line, eye widths (upper + lower = openness) */
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
	m->upper = hi / w;
	m->lower = -lo / w;
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
	g->lock_f = g->lock_mid = 1.0f;
	g->down_noise = g->noise_slow = DOWN_NOISE_INIT;
	g->side_noise = DOWN_NOISE_INIT;
	g->lid_noise = LID_NOISE_INIT;
	cf_robust *rb[2] = {&g->rb_down, &g->rb_side};
	for (int i = 0; i < 2; i++) {
		rb[i]->lr_noise = LR_NOISE_INIT;
		rb[i]->v_noise = VIS_NOISE_INIT;
		rb[i]->v_gain = VIS_GAIN_INIT;
		rb[i]->v_gain_err = VIS_GAIN_ERR_INIT;
		rb[i]->out_t = -1e9; /* clean until an outlier is seen */
		rb[i]->v_jump_t = -1e9;
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
	p.strength = clampf(finite_or(p.strength, 0.0f), 0.0f, CF_STRENGTH_MAX);
	/* Lid heights at the lens: both known and plausible, or neither (lids then not restored). */
	p.baseline_upper = finite_or(p.baseline_upper, 0.0f);
	p.baseline_lower = finite_or(p.baseline_lower, 0.0f);
	if (!(p.baseline_upper > 0.0f && p.baseline_upper < 1.0f && p.baseline_lower > 0.0f && p.baseline_lower < 1.0f))
		p.baseline_upper = p.baseline_lower = 0.0f;
	p.max_glance = finite_or(p.max_glance, 0.15f);
	p.baseline = finite_or(p.baseline, 0.0f);
	p.baseline_open = finite_or(p.baseline_open, 0.0f);
	p.notes_open = finite_or(p.notes_open, 0.0f);
	if (!isfinite(p.notes_offset) || !isfinite(p.notes_side) || !cf_gaze_notes_valid(p.baseline, p.notes_offset)) {
		p.has_notes = 0;
		p.notes_offset = p.notes_side = p.notes_open = 0.0f;
	}
	if (!p.has_notes || !isfinite(p.vis_baseline) || !isfinite(p.vis_notes))
		p.has_vis_calib = 0;
	return p;
}

/* Vision's gain from the two calibrations: how far the refined pupil moved between the lens and the notes,
 * per how far Vision's point moved. Measured on many steady frames each, so it is far surer than what one
 * saccade shows: taken as known (VIS_GAIN_ERR_CAL), and both channels start from it (sideways the eye can
 * only be assumed to under-read alike, so that one is still learnt as unsure). Set again only when the
 * calibration changes. */
static void use_vis_calib(cf_gaze *g, const cf_gaze_params *p)
{
	if (!p->has_vis_calib)
		return;
	const float dr = p->baseline - p->notes_offset, dv = p->vis_baseline - p->vis_notes;
	if (!(dv > VIS_CAL_MIN_FRAC * dr && dv > VIS_CAL_MIN))
		return; /* Vision's point barely moved: no usable gain */
	const float gain = dr / dv;
	if (!(gain >= VIS_GAIN_MIN && gain <= VIS_GAIN_MAX) || gain == g->vis_gain_cal)
		return;
	g->vis_gain_cal = gain;
	cf_robust *rb[2] = {&g->rb_down, &g->rb_side};
	for (int i = 0; i < 2; i++) {
		rb[i]->v_gain = gain;
		rb[i]->v_off_n = 0; /* the offset and levels go with the gain: start them afresh */
		rb[i]->learn = 0;
		rb[i]->v_pend = 0;
		new_level(rb[i]);
	}
	/* (a small notes glance measures the gain less surely: the two medians carry their noise) */
	g->rb_down.v_gain_err = fmaxf(VIS_GAIN_ERR_CAL, VIS_CAL_DR_ERR / dr);
}

/* A new lens calibration moved where "looking into the lens" reads (baseline, by delta): everything the
 * glance channel keeps (the accepted and shown glance, the offset from Vision's point, the levels, the
 * history) is measured from the old one. All of it moves along, so the change doesn't read as the gaze
 * jumping (an unbacked jump, voted on and glided in over a second or more). */
static void rebase_down(cf_gaze *g, float delta)
{
	cf_robust *c = &g->rb_down;
	if (g->down.ready)
		g->down.value += delta;
	c->v_off += delta;
	c->learn_x += delta;
	for (int j = 0; j < 2; j++)
		c->alt[j] += delta;
	c->jump_from += delta;
	c->m_prev += delta;
	c->shown += delta;
	c->shown_x += delta;
	c->glide_from += delta;
	g->last_down_raw += delta;
	for (int k = 0; k < g->hist_n; k++)
		g->hist[k].down += delta;
	g->d_lock = fmaxf(g->d_lock + delta, 0.0f);
	g->d_live_prev = fmaxf(g->d_live_prev + delta, 0.0f);
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
	g->calib_vis_count = 0;
}

/* Median of n values, leaving them as they are. */
static float median_of(const float *v, int n)
{
	float tmp[64];
	memcpy(tmp, v, (size_t)n * sizeof(float));
	return median(tmp, n);
}

/* A standard deviation that a few wild values don't move: 1.4826 x the median distance from the median. */
static float robust_sd(const float *v, int n, float med)
{
	float d[64];
	for (int i = 0; i < n; i++)
		d[i] = fabsf(v[i] - med);
	return 1.4826f * median(d, n);
}

/* The medians of the samples. Frames whose lids were well off their usual height during the calibration (a
 * partial blink or a squint too slight to be held, a lid half over the iris) read the pupil off: the lid
 * drags the landmarks and covers part of the iris. They are left out, unless that leaves too few. */
static void finish_calibration(cf_gaze *g)
{
	const int n = g->calib_count;
	const float open_med = median_of(g->calib_open, n);
	const float open_tol = fmaxf(CALIB_OPEN_K * robust_sd(g->calib_open, n, open_med), CALIB_OPEN_MIN);
	int lids_off = 0;
	for (int i = 0; i < n; i++)
		lids_off += fabsf(g->calib_open[i] - open_med) > open_tol;
	const int trim = n - lids_off >= CF_CALIB_MIN_SAMPLES;
	float off[64], side[64], open[64], vis[64], up[64], lo[64];
	int k = 0, kv = 0;
	for (int i = 0; i < n; i++) {
		if (trim && fabsf(g->calib_open[i] - open_med) > open_tol)
			continue;
		off[k] = g->calib_samples[i];
		side[k] = g->calib_side[i];
		open[k] = g->calib_open[i];
		up[k] = g->calib_upper[i]; /* head and lids */
		lo[k] = g->calib_lower[i];
		k++;
		if (isfinite(g->calib_vis[i]))
			vis[kv++] = g->calib_vis[i];
	}
	g->calib_result = median_of(off, k);
	g->calib_open_result = median_of(open, k);
	g->calib_side_result = median_of(side, k);
	g->calib_upper_result = median_of(up, k); /* head and lids */
	g->calib_lower_result = median_of(lo, k);
	g->calib_head_result = cf_head_reference(g->calib_head, g->calib_head_n);
	/* How steady it was (see cf_calib_result): each half of the time (in order) about its own median, and how
	 * far apart those are; and the lids over all the frames. */
	const int h = k / 2;
	const float m1 = median_of(off, h), m2 = median_of(off + h, k - h);
	g->calib_spread = 0.5f * (robust_sd(off, h, m1) + robust_sd(off + h, k - h, m2));
	g->calib_drift = fabsf(m2 - m1);
	float o2 = 0;
	for (int i = 0; i < n; i++)
		o2 += (g->calib_open[i] - open_med) * (g->calib_open[i] - open_med);
	g->calib_open_spread = sqrtf(o2 / (float)n);
	/* Vision's point, if it was given on most of the frames used. */
	g->calib_vis_count = 2 * kv >= k ? kv : 0;
	g->calib_vis_result = g->calib_vis_count ? median_of(vis, kv) : 0.0f;
	g->calibrating = 0;
	g->calib_done = 1;
}

int cf_gaze_calib_hints(const cf_calib_result *res, float baseline)
{
	int hints = 0;
	/* The gaze moved on: the halves differ by more than CALIB_DRIFT_MAX, and by more than their noise could
	 * make them (twice the spread: the halves' medians differ by about half the spread on their own). Or it
	 * moved about a lot, or the lids kept moving. */
	if (!(res->drift <= fmaxf(CALIB_DRIFT_MAX, 2.0f * res->spread)) || !(res->spread <= CALIB_SPREAD_MAX) ||
	    !(res->open_spread <= CALIB_OPEN_SPREAD_MAX))
		hints |= CF_CALIB_HINT_UNSTEADY;
	if (res->target == CF_CALIB_NOTES && baseline - res->offset < CF_NOTES_SMALL_GLANCE)
		hints |= CF_CALIB_HINT_SMALL_GLANCE;
	return hints;
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
	out->has_vis = g->calib_vis_count > 0;
	out->vis_offset = g->calib_vis_result;
	out->spread = g->calib_spread;
	out->drift = g->calib_drift;
	out->open_spread = g->calib_open_spread;
	out->upper = g->calib_upper_result; /* head and lids */
	out->lower = g->calib_lower_result;
	return 1;
}

/* ---- Blinks */

/* Moves the correction strength towards `target` through two equal first-order stages (a critically
 * damped ease): it starts moving gently and arrives gently, an S-curve with no kink at either end, so a
 * fade never shows as a snap. Exact per stage for any frame time, so it looks the same at 15 fps as at
 * 30. Fades in a little slower than it fades out; after eyes closed for long, back in about as quickly
 * as the lids open (reopen_fast): what was let go of then went while they were shut. */
static void ease_factor(cf_gaze *g, float target, double dt)
{
	const double tau = target > g->factor_mid ? (g->reopen_fast ? REOPEN_FADE_TAU : FADE_IN_TAU) : FADE_OUT_TAU;
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
	/* After a hold the pupil points are still pulled by the lids for a moment: the glance stays frozen a
	 * little longer, but only while the lids are not yet back near open. Once they are, it is measured
	 * again at once (any change since the blink then glides in, see REACQ_TIME), so a correction held
	 * wrong is never shown for frames with the eyes open, to jump once the time is up. */
	const float settled = fminf(SETTLE_OPEN, g->open_r - 0.05f);
	if (t < g->settle_until && r >= settled)
		g->settle_until = t;
	if (g->blink) {
		const double len = t - g->hold_start;
		if (r > BLINK_EXIT || (r > BLINK_ENTER + 0.05f && fabsf(rate) < 1.0f && len > 0.1)) {
			g->blink = 0;
			if (r > reopen) {
				g->settle_until = r < settled ? t + SETTLE_TIME : t;
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
			g->settle_until = r < settled ? t + SETTLE_TIME : t;
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

/* ---- Head and lids
 *
 * What the viewer reads as gaze is where the head points plus where the eyes point in the head. The glance
 * (g->down, g->side) measures only the second, against the lens calibration. While reading notes under the
 * camera the head tilts down as well (on the user's clips ~6 deg, IQR 3, up to 19, against ~1.5 px of iris
 * movement in a 75 px eye), so cancelling the eyes alone left them looking that far below the lens. An eye
 * turning by an angle a moves the iris EYE_K sin(a) eye widths against its corners, so a head tilted down by p
 * is made up for by EYE_K sin(p) more lift: the total is how far the eyes would have to turn up in the head
 * to meet the lens. It is summed with the glance before clamping at 0, so a head tilted down while the eyes
 * look into the lens (the glance then reads up by just as much) gets no correction. Only down, and only up
 * to HEAD_PITCH_MAX: a head tilted further looks elsewhere, and the pitch fade lets go there.
 *
 * The head's turn is made up for sideways in the same way, but only HEAD_YAW_K of it: with typical face
 * depths the turn reads 0.9-1.7x the true angle across faces (tests/test_pose.c), the sideways correction
 * itself damps only READ_DAMP of the eyes' scanning, and a head following a line while the eyes stay on the
 * viewer is what eye contact looks like anyway; half keeps the eyes nearer the lens when the head turns with
 * the text, and can never swing them past it.
 *
 * Lids: reading lowers the upper lid and raises the lower one (~0.01 eye widths each on the user's clips), and
 * a narrower opening alone reads as looking down. Against the lens calibration's lid heights the drop and
 * the rise are restored while correcting: the upper lid is lifted by its drop where that is more than it
 * already follows the iris (LID_SHIFT of the glance's lift, see emit()), the lower lid lowered by its rise.
 *
 * All terms are smoothed (head_angles(), LID_TAU) and move by at most HEAD_SLEW_PX / LID_SLEW_PX a frame,
 * are frozen while a blink is held (the lids are meaningless then), and reach the eyes only through the
 * eased factor, so they never snap.
 */

static float med3f(float a, float b, float c)
{
	return fmaxf(fminf(a, b), fminf(fmaxf(a, b), c));
}

/* The head's angles for the head terms, degrees, against the lens calibration (0 without one, as pose.c).
 * The pose's own angles are smoothed for a fade and run ~4 frames behind a moving head, while the eyes
 * turning against it to hold a spot show at once, so the lift would run ahead of the head and lag back.
 * These are the median of the pose's last three measures (a one-frame glitch never shows), through a One
 * Euro filter: smoothed harder than the pose's while the head is still, a frame or two behind it while it
 * moves. (Landmark noise still passes more than it does into the glance: see check H9 in tests/test_gaze.c.) */
static void head_angles(cf_gaze *g, const cf_gaze_params *p, double t, float *yaw, float *pitch)
{
	const cf_pose *ps = &g->pose;
	*yaw = *pitch = 0;
	if (!ps->ready || ps->hist_n < 1 || ps->hist_n > 3)
		return;
	cf_head md = ps->hist[ps->hist_n - 1];
	if (ps->hist_n == 3) {
		const cf_head *h = ps->hist;
		md.side = med3f(h[0].side, h[1].side, h[2].side);
		md.vert = med3f(h[0].vert, h[1].vert, h[2].vert);
		md.eyes = med3f(h[0].eyes, h[1].eyes, h[2].eyes);
		md.tilt = med3f(h[0].tilt, h[1].tilt, h[2].tilt);
	}
	float y = 0, pt = 0;
	cf_head_angles(&md, &p->head_ref, &y, &pt);
	if (!isfinite(y) || !isfinite(pt))
		return;
	*yaw = (float)oe_filter(&g->head_yf, y, t, HEAD_MIN_CUTOFF, HEAD_BETA);
	*pitch = (float)oe_filter(&g->head_pf, pt, t, HEAD_MIN_CUTOFF, HEAD_BETA);
}

static void head_lids(cf_gaze *g, const cf_gaze_params *p, const eye_measure m[2], int hold, double t, double dt)
{
	if (hold)
		return; /* blink, half blink, closing or settling: everything stays where it was */
	const float w = fmaxf(0.5f * (float)(g->rx[0].value + g->rx[1].value), 1.0f);
	const float k30 = (float)fmin(dt * 30.0, 1.0); /* per frame at 30 fps or slower */
	const float deg = (float)(M_PI / 180.0);

	/* The head (a glitch that got past the median still moves the terms only so fast). */
	float yaw, pitch;
	head_angles(g, p, t, &yaw, &pitch);
	const float hv = EYE_K * sinf(clampf(pitch, 0.0f, HEAD_PITCH_MAX) * deg);
	/* (yaw is in image terms, the sideways shift along the eye axis, whose direction depends on which eye is
	 * labelled left) */
	const float hh = HEAD_YAW_K * EYE_K * sinf(clampf(yaw, -HEAD_YAW_MAX, HEAD_YAW_MAX) * deg) *
			 (g->axis.x >= 0 ? 1.0f : -1.0f);
	const float sh = HEAD_SLEW_PX * k30 / w;
	g->head_v += clampf(hv - g->head_v, -sh, sh);
	g->head_h += clampf(hh - g->head_h, -sh, sh);

	/* The lids, both eyes' mean. */
	const float up = 0.5f * (m[0].upper + m[1].upper), lo = 0.5f * (m[0].lower + m[1].lower);
	if (!g->have_lids) {
		g->lid_up = g->lid_up1 = up;
		g->lid_lo = g->lid_lo1 = lo;
		g->have_lids = 1;
	} else {
		/* Two stages: the lid heights come from the outline's extreme points, as noisy as single landmarks,
		 * and one stage of the same lag would pass several times more of that into each frame's step. */
		const float al = (float)(1.0 - exp(-dt / LID_TAU));
		g->lid_up1 += al * (up - g->lid_up1);
		g->lid_lo1 += al * (lo - g->lid_lo1);
		g->lid_up += al * (g->lid_up1 - g->lid_up);
		g->lid_lo += al * (g->lid_lo1 - g->lid_lo);
	}
	float ud = 0, lr = 0;
	if (p->baseline_upper > 0 && p->baseline_lower > 0) { /* sanitize(): both calibrated, or neither */
		ud = clampf(p->baseline_upper - g->lid_up - LID_DEAD, 0.0f, UP_DROP_MAX);
		lr = clampf(p->baseline_lower - g->lid_lo - LID_DEAD, 0.0f, LO_RISE_MAX);
	}
	const float sl = LID_SLEW_PX * k30 / w;
	g->up_drop += clampf(ud - g->up_drop, -sl, sl);
	g->lo_rise += clampf(lr - g->lo_rise, -sl, sl);
}

/* The glance catching up after the head moved. Eyes holding a spot while the head tilts turn the other way in
 * the head by as much, so the glance and the head term change alike in opposite directions and the lift stays
 * put. But the glance passes the gate, which keeps a slow drift like that back as noise until it has grown
 * past the noise band and then takes it in one frame, while the head term follows at once: the lift would
 * ramp with the head and then jump back by a pixel or two. So a jump of the lift back against the way the
 * head just moved it is glided in at VOR_GLIDE_PX a frame instead, as far as all three agree: the head moved
 * the lift that far (within the last VOR_TAU), and the raw glance had already moved that far ahead of the
 * accepted one. The rest of a jump (a saccade) still shows at once. eye: the glance the lift is made from
 * (both signs). Returns the offset still to glide in (eye widths), held through blinks. */
static float glide_catch_up(cf_gaze *g, float eye, float down_raw, int hold, double t, double dt)
{
	if (hold || !isfinite(down_raw))
		return g->vor_off;
	const float w = fmaxf(0.5f * (float)(g->rx[0].value + g->rx[1].value), 1.0f);
	if (g->have_vor && !(t - g->vor_t <= 0.5)) { /* the face lost, or a long hold: what moved meanwhile is unknown */
		g->have_vor = 0;
		g->vor_acc = 0;
	}
	if (!g->have_vor) {
		g->vor_raw = down_raw;
	} else {
		const float dh = g->head_v_lock - g->vor_h, dd = (eye - g->vor_d) + dh;
		g->vor_acc = g->vor_acc * (float)exp(-(t - g->vor_t) / VOR_TAU) + dh;
		if (dd * g->vor_acc < 0) { /* back against the head's move: the glance catching up, as far as due */
			const float due = fmaxf(g->vor_gap * (dd > 0 ? 1.0f : -1.0f), 0.0f);
			const float take = copysignf(fminf(fminf(fabsf(dd), fabsf(g->vor_acc)), due), dd);
			g->vor_acc += take;
			g->vor_off -= take;
		}
		g->vor_raw += (float)(1.0 - exp(-dt / VOR_RAW_TAU)) * (down_raw - g->vor_raw);
	}
	const float step = VOR_GLIDE_PX * (float)fmin(dt * 30.0, 1.0) / w;
	g->vor_off -= clampf(g->vor_off, -step, step);
	g->vor_gap = g->vor_raw - eye;
	g->vor_d = eye;
	g->vor_h = g->head_v_lock;
	g->vor_t = t;
	g->have_vor = 1;
	return g->vor_off;
}

/* ---- Output */

/* Fills out[] from the filtered state. The shifts before strength are in g->amount_unit
 * (vertical, along face up), g->hshift_unit (sideways, along the eye axis), and the lid terms
 * g->lid_unit (upper lid, beyond following the iris) and g->low_unit (lower lid, down). */
static void emit(const cf_gaze *g, const cf_gaze_params *p, cf_eye out[2])
{
	const float strength = clampf(p->strength, 0.0f, CF_STRENGTH_MAX);
	/* The lids are restored at most to the lens calibration's opening: an overshooting strength (over 1,
	 * making up for a pupil reading that under-reads) lifts the iris, and the upper lid with it, further,
	 * but never opens the eyes wider than they are when looking into the lens. */
	const float lid_strength = fminf(strength, 1.0f);
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
		const float cap = fminf(AMOUNT_MAX_RY * e->ry, 0.5f * (e->ry - ext_y));
		amount = fminf(amount, cap);
		amount = fmaxf(amount, 0.0f);
		/* Sideways no more than 100%: it already takes out READ_DAMP of the eyes' scanning, and more would show
		 * them sweeping the wrong way. */
		float hs = g->hshift_unit[i] * fminf(strength, 1.0f);
		hs = clampf(hs, -HSHIFT_MAX_RX * e->rx, HSHIFT_MAX_RX * e->rx);

		e->sx = g->up.x * amount + g->axis.x * hs;
		e->sy = g->up.y * amount + g->axis.y * hs;
		/* The upper lid follows vertical eye movement, not sideways movement; plus what is left of its
		 * reading drop (head and lids). The lower lid moves down by its reading rise. */
		const float lid = amount * LID_SHIFT + clampf(g->lid_unit[i] * lid_strength, 0.0f, 0.5f * cap);
		const float low = clampf(g->low_unit[i] * lid_strength, 0.0f, 0.25f * cap);
		e->lid_sx = g->up.x * lid;
		e->lid_sy = g->up.y * lid;
		e->low_sx = -g->up.x * low;
		e->low_sy = -g->up.y * low;
	}
}

static int finish_output(cf_gaze *g, const cf_gaze_params *p, cf_eye out[2])
{
	emit(g, p, out);
	int n = g->factor * g->lock_f < 0.005f ? 0 : 2;
	if (n && fabsf(g->amount_unit[0]) + fabsf(g->amount_unit[1]) + fabsf(g->hshift_unit[0]) +
			     fabsf(g->hshift_unit[1]) + fabsf(g->lid_unit[0]) + fabsf(g->lid_unit[1]) +
			     fabsf(g->low_unit[0]) + fabsf(g->low_unit[1]) <
		     0.01f)
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
	use_vis_calib(g, p);
	if (g->have_base && p->baseline != g->base_prev)
		rebase_down(g, p->baseline - g->base_prev);
	g->base_prev = p->baseline;
	g->have_base = 1;
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
		/* Where Vision's point was stays known through a short loss: the first frame back is checked
		 * against it like any other, so a glance made meanwhile is taken at once. After a long one it is
		 * stale (the person may have moved): everything starts afresh then. */
		if (!g->cx[0].ready || t - g->cx[0].t > FACE_GONE) {
			g->rb_down.have_v = g->rb_side.have_v = 0;
			g->have_lids = 0; /* the lids' smoothing starts afresh from the face coming back */
		}
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
			g->lid_unit[i] *= ratio; /* head and lids */
			g->low_unit[i] *= ratio;
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
	/* A glance with the lids following, or a lid dragging the pupil points (a blink starting)? A lid drags
	 * each eye's refined point its own way (one down, the other not, or even up), and Vision's point only
	 * part of the way, while a glance moves both eyes alike and the lids by less than the eyes. So: with
	 * Vision's point, it must have dropped by at least LID_VIS_RATIO of the lid drop, and the pupil drop is
	 * the combined reading (which leans on the eye Vision's point moved with); without it, both refined
	 * points must have dropped, alike: the smaller drop counts. */
	const int have_vis = vis && g->rb_down.have_v;
	const float v_drop = have_vis ? v_down - g->rb_down.v_anchor : 0.0f; /* Vision's, from the accepted gaze */
	const float v_step = have_vis ? v_down - g->rb_down.v_last : 0.0f;   /* ... and since the last frame */
	const int eyes_agree = !lr_down;
	const float eyes_drop = g->down.ready ? fminf(e_down[0], e_down[1]) : 0.0f;
	const float pupil_drop = !g->down.ready ? 0.0f
				 : have_vis	    ? down_raw - (float)g->down.value
				 : eyes_agree	    ? eyes_drop
						    : 0.0f;
	const float lid_drop = (g->open_r - r) * fmaxf(ref, 0.05f); /* eye widths */
	const int vis_lid = have_vis && v_drop < LID_VIS_RATIO * lid_drop; /* Vision's point: a lid, not a glance */
	/* (Both thresholds sit above the pupil noise, so a noisy reading doesn't end a hold.) */
	const int glance = !vis_lid && ((r > GLANCE_MIN_R && pupil_drop > fmaxf(GLANCE_PUPIL, 2.0f * g->down_noise) &&
					 pupil_drop >= GLANCE_LID_RATIO * lid_drop) ||
					(pupil_drop > GLANCE_BIG + 2.0f * g->down_noise && pupil_drop >= lid_drop));
	if ((g->closing || g->half) && !g->blink && g->down.ready && glance) {
		g->closing = g->half = 0;
		g->was_holding &= 2; /* (that "half blink" was this glance: nothing to resync or glide in) */
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
	int lid_drag;       /* the lids are dragging the pupil points this frame */
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
		/* Moved with the lids like a glance, as for `glance` above (this frame's moves): Vision's point
		 * by at least LID_VIS_RATIO of the lid's, or without it both refined points. */
		const int like_glance = have_vis ? v_step >= LID_VIS_RATIO * lid_step
						 : eyes_agree && eyes_drop >= ROLLBACK_GLANCE * lid_step;
		const int clearly_glance = pupil_step > fmaxf(ROLLBACK_PUPIL, 3.0f * g->down_noise) &&
					   pupil_step >= ROLLBACK_GLANCE * lid_step + 3.0f * g->down_noise && like_glance;
		glance_trusted = clearly_steady || clearly_glance;
		lids_clear = clearly_steady;
		glance_step = pupil_step;
		/* The lids clearly fell this frame and the pupils didn't move with them like a glance: the points
		 * are being dragged by a lid (a blink starting). Nothing may back a change up this frame. This
		 * frame's own lid fall against this frame's pupil moves (a glance spread over two frames has the
		 * lids falling on both, and the pupils moving with them on both). */
		const float lid_now = fmaxf(step, 0.0f) * fmaxf(ref, 0.05f);
		const int drag_like_glance = have_vis ? v_step >= LID_VIS_RATIO * lid_now
						      : eyes_agree && eyes_drop >= ROLLBACK_GLANCE * lid_now;
		lid_drag = step > fmaxf(LID_STEADY_STEP, 3.0f * g->lid_noise) && !drag_like_glance;
	}
	/* Landmark noise: mean frame-to-frame change of the raw glance while the lids are steady. A change of a
	 * saccade's size is a saccade or an outlier, not noise: left out (a refiner jumping between two spots
	 * on many frames would otherwise read as huge noise and open the noise band right up). */
	if (!hold && g->have_down_raw && fabsf(down_raw - g->last_down_raw) <= down_hi) {
		const float ch = fabsf(down_raw - g->last_down_raw);
		g->down_noise += (float)(1.0 - exp(-dt / DOWN_NOISE_TAU)) * (ch - g->down_noise);
		g->noise_slow = fmaxf(g->down_noise, g->noise_slow + (float)(1.0 - exp(-dt / FAR_NOISE_TAU)) *
								     (g->down_noise - g->noise_slow));
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

	/* Calibration samples: each frame's own readings (the accepted gaze would carry one frame's lid or outlier
	 * into the next few). The first moments after starting are skipped: the eyes are still on the Calibrate
	 * button. */
	const int calib_warm = g->calibrating && t - g->calib_start >= CF_CALIB_WARMUP_S;
	if (calib_warm && !hold && abs_yaw < 15 && g->calib_count < 64) {
		g->calib_samples[g->calib_count] = p->baseline - down_raw;
		g->calib_open[g->calib_count] = openness;
		g->calib_side[g->calib_count] = side;
		g->calib_upper[g->calib_count] = 0.5f * (m[0].upper + m[1].upper); /* head and lids */
		g->calib_lower[g->calib_count] = 0.5f * (m[0].lower + m[1].lower);
		g->calib_vis[g->calib_count] = vis ? -v_down : NAN; /* (up positive, like offset) */
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
			.lid_drag = lid_drag,
		};
		if (g->was_holding)
			g->reacq_until = t + REACQ_TIME; /* the first look after a blink: see below */
		g->was_holding = 0;
		snap_down |= gate(&g->rb_down, &g->down, down_raw, &gi, t, dt);
		down = g->down.value;
	} else {
		down = g->down.ready ? g->down.value : 0;
		g->was_holding |= g->blink ? 2 : g->half ? 1 : 0;
		/* (cut short by a blink: no outlier) */
		g->rb_down.steady_n = g->rb_down.held_big = g->rb_down.away = g->rb_down.followed = 0;
		/* Held, not lost: the first reading after a long blink is checked like any other (resync), not
		 * taken as a fresh start. */
		if (g->down.ready)
			g->down.t = t;
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
		if (g->side.ready)
			g->side.t = t;
	}

	/* Output slew guard: what the warp shows follows the accepted gaze, but without a backed-up saccade
	 * it moves at most SLEW_PX per 1/30 s (whatever got past the checks above can only nudge the eyes).
	 * Fades act on top of this, unlimited (they are smooth by design). Sideways, the shift is
	 * READ_DAMP of the gaze, so the gaze may move that much faster for the same shift. */
	const float wmax = fmaxf(fmaxf((float)g->rx[0].value, (float)g->rx[1].value), 1.0f);
	const float slew_step = SLEW_PX * (float)fmin(dt * 30.0, 1.0) / wmax; /* per frame, at 30 fps or slower */
	/* Just after a blink, a change taken at once (the gaze moved while the eyes were shut, or what was held
	 * was off) glides in over GLIDE_TIME instead: it may only be measured once the lids are nearly open,
	 * and a jump then would show with the eyes open. Most of it lands while the lids are still opening. */
	const int reacq = t < g->reacq_until;
	const double down_was = g->rb_down.have_shown ? g->rb_down.shown : down;
	const double side_was = g->rb_side.have_shown ? g->rb_side.shown : side_f;
	down = show(&g->rb_down, down, snap_down, reacq && !rolled_back, t, dt, slew_step,
		    noise_alpha(DOWN_OUT_CUTOFF, dt), 0);
	side_f = show(&g->rb_side, side_f, snap_side, reacq, t, dt, slew_step / READ_DAMP,
		      noise_alpha(SIDE_OUT_CUTOFF, dt), 1);
	const float side_dev = (float)side_f - side_c;

	/* Glance limit: just past the notes when calibrated (by 40%, and by FAR_MARGIN and FAR_NOISE_K x the
	 * landmark noise at least: reading lower down the page, or a calibration that came out small, stays
	 * corrected), else the manual setting. */
	const int notes = p->has_notes; /* sanitize() cleared it if the notes aren't below the lens */
	const float notes_glance = p->baseline - p->notes_offset;
	const float limit = notes ? fmaxf(1.4f * notes_glance,
					  notes_glance + fmaxf(FAR_MARGIN, FAR_NOISE_K * g->noise_slow))
				  : p->max_glance;
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

	/* Everything that should switch the effect off. Each term is 1 when it's fine. The head's (f_yaw,
	 * f_pitch, f_size) fade it gently; the gaze's (f_side, f_far) let go with a saccade (see lock_f). */
	const float f_yaw = 1.0f - smoothstepf(CF_YAW_FADE_LO, CF_YAW_FADE_HI, abs_yaw);
	const float f_pitch = 1.0f - smoothstepf(CF_PITCH_FADE_LO, CF_PITCH_FADE_HI, abs_pitch);
	const float f_side = 1.0f - smoothstepf(SIDE_FADE_LO, SIDE_FADE_HI, fabsf(side_dev));
	/* Far fade: how much the gaze is still on the notes rather than past them (keyboard). It goes into
	 * the gaze's lock below (lock_f), with the shift no bigger than at the limit while it fades, so the
	 * visible eye only ever moves the same way as the real gaze. Held while the lids are shut or closing
	 * (the lid cue is meaningless then), recomputed as soon as they are open again (also during the
	 * settle time after a blink, from the fresh lids). */
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
	const float target = f_yaw * f_pitch * f_size;
	if (hold) {
		/* Blink: hold the correction steady (the gaze terms are held too). Only head-based fades
		 * (reliable while the eyes are closed) may lower it; eyes closed for long fade out. */
		if (g->blink && t - g->hold_start > BLINK_HOLD_MAX) {
			ease_factor(g, 0, dt);
			g->reopen_fast = 1;
		} else if (target < g->factor) {
			ease_factor(g, target, dt);
		} else {
			g->factor_mid = g->factor;
		}
	} else {
		ease_factor(g, target, dt);
		if (g->factor >= 0.98f * target)
			g->reopen_fast = 0;
	}
	/* Looking away (past the notes, or sideways out of them) lets go of the correction, and looking back
	 * takes over again (lock_f). With a saccade (the shown gaze jumping by a saccade's size this frame) that
	 * happens at once, inside the eye's own jump, where it can't be seen: the eye lands where a real one
	 * would, instead of landing with the old correction and then rolling the rest of the way. Otherwise (a
	 * slow drift past the notes, the lids saying the gaze went far, the gaze gliding in after a blink) it
	 * eases like a head turn, an S-curve over a few hundred ms. Held through blinks. */
	const float lock = f_side * f_far;
	const int glide = g->rb_down.gliding || g->rb_side.gliding;
	const int sacc = (snap_down && fabs(down - down_was) > down_hi) || (snap_side && fabs(side_f - side_was) > side_hi);
	if (!hold) {
		if (sacc) {
			g->lock_f = g->lock_mid = lock;
		} else {
			const float a = (float)(1.0 - exp(-dt / (lock > g->lock_mid ? FADE_IN_TAU : FADE_OUT_TAU)));
			g->lock_mid += a * (lock - g->lock_mid);
			g->lock_f += a * (g->lock_mid - g->lock_f);
		}
	}
	g->last_target = target * lock;

	/* The glance and sideways position the shift is made from. While the gaze is on the notes, the live
	 * ones; while it is away (lock < 1), no further than the edge of the band where the correction applies
	 * (the limit, SIDE_FADE_LO), and never growing unless with a saccade (or moving with a glide after a
	 * blink): so a fade only ever lets go, without a wobble, and how much is corrected is continuous in how
	 * far the gaze went. Back on the notes without a saccade, they follow the live ones' moves, and what is
	 * left of the gap the fade made closes at twice the output slew limit (no snap). */
	const float d_live = (float)(down > 0 ? down : 0);
	const float d_cap = fminf(d_live, limit), s_cap = clampf(side_dev, -SIDE_FADE_LO, SIDE_FADE_LO);
	const float d_move = d_live - g->d_live_prev, s_move = side_dev - g->side_live_prev;
	/* The head's tilt and turn, and the lids (see head_lids()), held through blinks like the glance; and
	 * the eyes' glance up in the head, which offsets a head tilted down (the eyes holding the lens while the
	 * head tilts). Followed while the gaze is on the notes; while it is away, never growing (like the
	 * glance), so a fade only ever lets go. */
	head_lids(g, p, m, hold, t, dt);
	const float up_live = fminf((float)down, 0.0f);
	if (lock >= 0.999f || !g->have_lock) {
		g->head_v_lock = g->head_v;
		g->head_h_lock = g->head_h;
		g->up_lock = up_live;
		g->up_drop_lock = g->up_drop;
		g->lo_rise_lock = g->lo_rise;
	} else {
		g->head_v_lock = fminf(g->head_v_lock, g->head_v);
		g->head_h_lock = fabsf(g->head_h) < fabsf(g->head_h_lock) ? g->head_h : g->head_h_lock;
		g->up_lock = fminf(g->up_lock, up_live);
		g->up_drop_lock = fminf(g->up_drop_lock, g->up_drop);
		g->lo_rise_lock = fminf(g->lo_rise_lock, g->lo_rise);
	}
	if (!g->have_lock || (lock >= 0.999f && sacc)) {
		g->d_lock = d_live;
		g->side_lock = side_dev;
		g->have_lock = 1;
	} else if (lock >= 0.999f) {
		const float gd = g->d_live_prev - g->d_lock, gs = g->side_live_prev - g->side_lock;
		const float sd = 2.0f * slew_step, ss = sd / READ_DAMP;
		g->d_lock = fmaxf(d_live - (gd > 0 ? fmaxf(gd - sd, 0.0f) : fminf(gd + sd, 0.0f)), 0.0f);
		g->side_lock = side_dev - (gs > 0 ? fmaxf(gs - ss, 0.0f) : fminf(gs + ss, 0.0f));
	} else if (sacc) {
		g->d_lock = d_cap;
		g->side_lock = s_cap;
	} else {
		g->d_lock = fminf(glide ? fmaxf(g->d_lock + d_move, 0.0f) : g->d_lock, d_cap);
		const float s1 = glide ? g->side_lock + s_move : g->side_lock;
		g->side_lock = fabsf(s_cap) < fabsf(s1) ? s_cap : s1;
	}
	g->d_live_prev = d_live;
	g->side_live_prev = side_dev;
	/* How far the eyes must turn up to meet the lens: their glance plus the head's tilt down (with the glance's
	 * catch-up after a head move glided in, see glide_catch_up()). */
	const float eye = g->d_lock + g->up_lock;
	const float d = fmaxf(eye + g->head_v_lock + glide_catch_up(g, eye, down_raw, hold, t, dt), 0.0f);
	/* Sideways damping only engages as the gaze drops towards the notes, so looking around at lens
	 * height (at the other person) stays untouched. The lids are restored only then too. */
	const float engage = smoothstepf(0.0f, fmaxf(0.35f * limit, 0.01f), d);
	/* Upper lid: beyond following the iris lift by LID_SHIFT (emit()), lifted by what is left of its drop
	 * after the share of the glance's own lift (the share it dropped with); lower lid: by its rise. */
	const float lid_up_x = fmaxf(g->up_drop_lock - LID_SHIFT * fmaxf(g->d_lock, 0.0f), 0.0f) * engage;
	const float lid_lo_x = g->lo_rise_lock * engage;
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
		float a = d * w * g->factor * g->lock_f;
		float h = -READ_DAMP * (g->side_lock + g->head_h_lock) * w * g->factor * g->lock_f * engage;
		float l = lid_up_x * w * g->factor * g->lock_f, b = lid_lo_x * w * g->factor * g->lock_f;
		if (g->ease) {
			converged &= fabsf(a - g->amount_unit[i]) <= max_step && fabsf(h - g->hshift_unit[i]) <= max_step &&
				     fabsf(l - g->lid_unit[i]) <= max_step && fabsf(b - g->low_unit[i]) <= max_step;
			a = g->amount_unit[i] + clampf(a - g->amount_unit[i], -max_step, max_step);
			h = g->hshift_unit[i] + clampf(h - g->hshift_unit[i], -max_step, max_step);
			l = g->lid_unit[i] + clampf(l - g->lid_unit[i], -max_step, max_step);
			b = g->low_unit[i] + clampf(b - g->low_unit[i], -max_step, max_step);
		}
		g->amount_unit[i] = a;
		g->hshift_unit[i] = h;
		g->lid_unit[i] = l;
		g->low_unit[i] = b;
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
	out->factor = (g->down.ready && g->down.value > 0) ? g->factor * g->lock_f : 0.0f; /* includes looking away */
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
#define RP_EUP0 0.01f        /* lid clearance for edge points: just inside the landmarks' lids, since */
#define RP_EUP1 0.03f        /* the image check (sclera beyond the edge) keeps lashes out; looking down, */
#define RP_ELO0 0.005f       /* the rows near the lids are the ones that tell where the centre is */
#define RP_ELO1 0.02f
#define RP_P_IRIS 0.20f      /* grey percentiles inside the outline: iris level ... */
#define RP_P_SCLERA 0.90f    /* ... and sclera level */
#define RP_RAMP0 0.30f       /* darkness 1 below iris + this x (sclera - iris), 0 above ... (well above the */
                             /* iris level, so a light or textured iris body reads fully dark: the edge */
                             /* widths are integrals of darkness and would shrink, row by row, with it) */
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
#define RP_MAXPTS 512        /* edge points */
#define RP_TAU 0.06          /* circle refit: points this far off the circle (R units) count half */
#define RP_SIG0 0.12f        /* circle fit's nominal error (R units): trusted fully below this ... */
#define RP_SIG1 0.30f        /* ... not at all above (the centroid it falls back on reads a lowered gaze short) */

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

static cf_pt rp_refine_s(const uint8_t *grey, int stride, int width, int height, const rp_frame *ep, float w,
			 float conf, cf_pt guess, int s_in)
{
	const rp_frame e = *ep;
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
	int s = s_in, x0, y0, gw, gh;
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
	float hist[RP_NB] = {0}, qsum = 0, msum = 0;
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
			q *= 1 - smoothstepf(220.0f, 240.0f, val[c]);
			qw[c] = q;
			msum += m;
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
	{
		const float area = (4.0f / 3.0f) * 0.5f * w * (e.up[0] + e.lo[0]); /* integral of the opening */
		conf *= smoothstepf(0.8f, 0.95f, area > 0 ? msum * fs * fs / area : 0); /* eye cut by the image edge */
		if (!(conf > 0))
			return guess;
	}
	conf *= smoothstepf(0.3f, 0.36f, qsum * fs * fs / (R * R)); /* too little of the eye between the lash lines */
	if (!(conf > 0))
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
		const float nff = RP_FILL_R * R * ifs;
		const int nf = (int)ceilf(nff);                                      /* bridging reach, cells */
		const float nfl = nff - (float)(nf - 1);                             /* weight of the last one */
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
		/* Edge rows: every pixel row (blocks only along the row), so the vertical sampling does not
		 * change with the block size. */
		const int rpitch = (s + 1) / 2;
		const int yend = y0 + gh * s < height ? y0 + gh * s : height;
		int pya = (int)floorf((Y0 - 1.25f * R - 1) / (float)rpitch) * rpitch, pyb = (int)ceilf(Y0 + 1.25f * R) + 1;
		while (pya < y0) /* rows on a fixed image grid, so they don't move with the eye */
			pya += rpitch;
		pyb = pyb > yend - 1 ? yend - 1 : pyb;
		for (int py = pya; ic >= 0 && ic + 1 < gw && py <= pyb; py += rpitch) {
			const float Yj = (float)py + 0.5f, dy = Yj - e.c0y;
			float du, dl;
			/* The row must cross the iris itself: not far from the centre so far, and its centre
			 * column clear of the lid bands (lashes are dark too) ... */
			float core = 1 - smoothstepf(1.0f * R, 1.25f * R, fabsf(Yj - Y0));
			core *= rp_lids(&e, X0 - e.c0x, dy, &du, &dl) ? rp_ss(EU0, iEU, du) * rp_ss(EL0, iEL, dl) : 0;
			if (core <= 0)
				continue;
			/* Grey levels 1/4 1/2 1/4 over the pixel row and its neighbours: a bright patch's or a lid's
			 * edge across the rows is spread over two or three of them, so the edge points don't
			 * change in steps as the eye moves by fractions of a row. */
			const uint8_t *grow = grey + (size_t)py * (size_t)stride;
			const uint8_t *gup = grey + (size_t)(py > 0 ? py - 1 : py) * (size_t)stride;
			const uint8_t *gdn = grey + (size_t)(py + 1 < height ? py + 1 : py) * (size_t)stride;
			float row[RP_MAXROW], mr[RP_MAXROW];
			for (int i = lo_i; i <= hi_i; i++) {
				mr[i] = rp_lids(&e, gx0 + (float)i * fs, dy, &du, &dl) ? rp_mask(du, dl, 1.0f) : 0;
				if (mr[i] > 0) {
					const int px0 = x0 + i * s, px1 = px0 + s > width ? width : px0 + s;
					unsigned sum = 0;
					for (int x = px0; x < px1; x++)
						sum += 2u * grow[x] + gup[x] + gdn[x];
					const float gv = 0.25f * (float)sum / (float)(px1 - px0);
					row[i] = (1 - clampf((gv - th0) * ith, 0, 1)) * mr[i];
				} else {
					row[i] = 0;
				}
			}
			/* ... and be dark at the centre column, bright patches there bridged (dark both sides). */
			{
				float ml = row[ic], mg = row[ic + 1];
				for (int k = 1; k <= nf; k++) {
					const float wk = k == nf ? nfl : 1;
					if (ic - k >= lo_i && row[ic - k] * wk > ml)
						ml = row[ic - k] * wk;
					if (ic + 1 + k <= hi_i && row[ic + 1 + k] * wk > mg)
						mg = row[ic + 1 + k] * wk;
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
							m = fmaxf(m, row[ii] * fc[ii] * (k == nf ? nfl : 1));
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
				ptw[npt++] = om * (float)rpitch * ifs;
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

cf_pt cf_refine_pupil(const uint8_t *grey, int stride, int width, int height, const cf_pt *eye, int count,
		      cf_pt guess)
{
	if (!grey || !eye || count < 4 || count > CF_MAX_EYE_POINTS || width <= 0 || height <= 0 || stride < width ||
	    !isfinite(guess.x) || !isfinite(guess.y))
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

	{
		const float sf = w / RP_CELLS_W;
		const int s0 = 1 + (int)sf;
		const float tb = smoothstepf(0.94f, 1.0f, sf - floorf(sf));
		const cf_pt r0 = tb < 1 ? rp_refine_s(grey, stride, width, height, &e, w, conf, guess, s0) : guess;
		if (!(tb > 0))
			return r0;
		const cf_pt r1 = rp_refine_s(grey, stride, width, height, &e, w, conf, guess, s0 + 1);
		if (!(tb < 1) || (r0.x == r1.x && r0.y == r1.y))
			return r1;
		return (cf_pt){r0.x + tb * (r1.x - r0.x), r0.y + tb * (r1.y - r0.y)};
	}
}

