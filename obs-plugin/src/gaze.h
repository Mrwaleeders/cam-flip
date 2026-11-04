/*
 * Cam Flip: turns face landmarks into eye-contact warps that feel natural.
 *
 * Pure C with no Apple or OBS dependencies, so the behaviour can be tested anywhere.
 *  - Eye positions are smoothed with a One Euro filter: steady when you're still,
 *    no lag when your head moves, so the correction never slides off your eyes.
 *  - The correction strength fades in and out smoothly (never snaps) when you turn your
 *    head, look far down or sideways, blink, or leave the frame.
 */
#pragma once

#include <stdint.h>

#include "pose.h"
#include "warp.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CF_MAX_EYE_POINTS 32
#define CF_HIST 6
#define CF_CALIB_SAMPLES 30
#define CF_CALIB_MIN_SAMPLES 10
#define CF_CALIB_TIMEOUT_FRAMES 300 /* about 10 s at 30 fps */
#define CF_CALIB_WARMUP_S 0.6        /* s: samples ignored after starting (eyes still on the button) */

/* cf_pt (image pixels, origin top left) is in pose.h. */

typedef struct {
	int found;
	cf_pt left_eye[CF_MAX_EYE_POINTS];
	int left_count;
	cf_pt right_eye[CF_MAX_EYE_POINTS];
	int right_count;
	cf_pt left_pupil, right_pupil; /* iris centres (refined) */
	/* Vision's own pupil points before refinement: coarse but steady, a second opinion on eye
	 * movement. has_vision_pupils 0 = not given (then only the refined points are used). */
	cf_pt left_pupil_vision, right_pupil_vision;
	int has_vision_pupils;
	/* Head pose measured from the landmarks on every frame (pose.c, cf_head_measure): used for the
	 * head-turn fade whenever given. has_pose / yaw_deg / pitch_deg are the detector's own angles,
	 * used only while there are no head measures. */
	int has_head_geom;
	cf_head head;
	int has_pose;
	float yaw_deg, pitch_deg;
	float face_width; /* pixels */
	int image_width, image_height;
} cf_face;

typedef struct {
	float strength;      /* 0..1, how much of a downward glance to cancel */
	float max_glance;    /* how far down is still corrected before fading out (manual mode) */
	float baseline;      /* gaze offset when looking into the lens (lens calibration) */
	float baseline_open; /* eye openness when looking into the lens; 0 = unknown, use a default */
	int has_notes;       /* 1 when the "look at your notes" calibration has been done */
	float notes_offset;  /* gaze offset when looking at the notes (notes calibration) */
	float notes_side;    /* sideways gaze offset (eye widths) at the middle of the notes (notes calibration) */
	float notes_open;    /* eye openness when looking at the notes (notes calibration); 0 = unknown */
	cf_head_ref head_ref; /* the face looking into the lens (lens calibration); ok 0 = a typical face */

	/* Vision's own pupil point at the two calibration steps (cf_calib_result.vis_offset of each), so the model
	 * knows how much of a glance it shows (it under-reads glances, by a factor that differs per person and
	 * camera). has_vis_calib: 1 when both are from calibrations (then has_notes must be set too). */
	int has_vis_calib;
	float vis_baseline; /* at the lens */
	float vis_notes;    /* at the notes */

	/* ---- Head and lid compensation (gaze.c, "Head and lids") */
	/* How far the upper lid sits above, and the lower lid below, the line through the eye corners when
	 * looking into the lens, eye widths (lens calibration, mean of both eyes). 0 = unknown (calibrated by
	 * an older version): the lids then only follow the iris as before, nothing is restored. */
	float baseline_upper, baseline_lower;

	/* How much of the eyes' sideways scanning along a line is taken out, 0..1 of what is measured (0: none,
	 * the default). Tracking runs a frame behind the video, so the counter-move for the jump back to the next
	 * line comes a frame after the eye's own jump: a one-frame twitch at every line return. Taking it in
	 * gently instead (a glide) left 0.77-0.98 of the scan (the eye sliding back across the line after each
	 * return), so there is no smooth middle way: the scanning is taken out (with that twitch) or left alone.
	 * Damping is centred on where the eyes have been lately (seconds), never on a calibrated spot, so it
	 * never adds a sideways offset the eyes don't have. */
	float scan_damp;

	/* 1 when has_notes / notes_* / vis_notes come from cf_gaze_take_learnt_notes() (the notes learnt from the
	 * reading itself), not from a notes calibration. Learnt notes only ever widen the correction: the glance
	 * limit is never below max_glance, and Vision's gain from them is only re-applied when it moved by more than
	 * a small tolerance (they are refined as the user reads). A model with nothing learnt yet starts from them. */
	int notes_learnt;
	/* How much reading the learnt notes rest on (cf_learnt_notes.seconds), s; 0 = unknown. A model starting from
	 * them counts them as that much reading (10 s at least, a minute at most): the more there was, the more new
	 * reading it takes to move them. */
	float notes_seconds;
} cf_gaze_params;

/* "Eye contact strength" goes up to this (200%): over 1 the correction overshoots, so it can make up for
 * a pupil reading that under-reads glances. The warp's own limits (AMOUNT_MAX_RY, the slope limit) still
 * apply, and the lid restoration never goes past the lens-calibrated opening. */
#define CF_STRENGTH_MAX 2.0f

/* A notes calibration is only usable when the notes are clearly below the lens: at least this many
 * eye widths of downward gaze (baseline - notes_offset). Smaller or reversed means the user wasn't
 * looking at notes under the camera (or the lens calibration is off). Kept this low on purpose: notes
 * 2-3 cm under the lens at arm's length are a real glance of about 0.02. */
#define CF_NOTES_MIN_GLANCE 0.01f

/* 1 when a notes calibration at notes_offset is meaningfully below the lens baseline. */
int cf_gaze_notes_valid(float baseline, float notes_offset);

enum cf_calib_target { CF_CALIB_LENS = 0, CF_CALIB_NOTES = 1 };

typedef struct {
	enum cf_calib_target target;
	float offset;   /* median gaze offset while calibrating (same units as baseline) */
	float openness; /* median eye openness while calibrating */
	float side;     /* median sideways gaze offset while calibrating (eye widths, along the eye axis) */
	cf_head_ref head; /* the head while calibrating (ok 0: too few frames with head measures) */

	/* Vision's own pupil point while calibrating (cf_face.has_vision_pupils): has_vis 1 when it was given on
	 * most frames, and its median offset (same units and direction as offset). Saved as the params'
	 * vis_baseline / vis_notes. */
	int has_vis;
	float vis_offset;

	/* How steady it was while calibrating: the samples' spread about the median of their half of the time
	 * (a robust standard deviation, eye widths: what is left of the landmark noise, or the gaze moving
	 * about), how far the second half's median moved from the first half's (the gaze moving on), and the
	 * root-mean-square deviation of the openness from its median (the lids moving: a squint coming and
	 * going). See cf_gaze_calib_hints(). */
	float spread;
	float drift;
	float open_spread;

	/* ---- Head and lid compensation: median upper / lower lid height above / below the corner line while
	 * calibrating (eye widths, mean of both eyes): baseline_upper / baseline_lower for a lens calibration */
	float upper, lower;
} cf_calib_result;

/* Hints about a finished calibration that was still taken (bit flags, 0 = looks fine), for the panel:
 *  CF_CALIB_HINT_UNSTEADY: the gaze moved while calibrating (or the lids kept moving), so the result may be
 *    off: redo that step, keeping still.
 *  CF_CALIB_HINT_SMALL_GLANCE (notes): the notes read less than CF_NOTES_SMALL_GLANCE below the lens. Right
 *    for notes just under the camera; otherwise the camera step was probably taken looking a little low
 *    (at the screen instead of into the lens).
 * baseline: the camera calibration the notes are measured from (unused for a camera result). */
#define CF_CALIB_HINT_UNSTEADY 1
#define CF_CALIB_HINT_SMALL_GLANCE 2
#define CF_NOTES_SMALL_GLANCE 0.03f
int cf_gaze_calib_hints(const cf_calib_result *res, float baseline);

/* What the filter panel shows. */
typedef struct {
	int face_found;
	int blinking;
	float glance;    /* how far below the lens the gaze is, same units as baseline */
	float factor;    /* 0..1, current correction strength after fades */
	/* Plain-word reasons the correction is (mostly) off, for the panel. Each is 0 unless the face is found. */
	int too_far;     /* the face is too small in the picture (the face-size fade is below half) */
	int turned;      /* the head is turned or tilted away from the camera (its fades are below half) */
	int reading;     /* the eyes are reading right now (the notes learner's detector) */
} cf_gaze_status;

/* ---- Notes learnt from reading (no notes step)
 *
 * While the model runs it watches for reading: the eyes sweeping along a line, then jumping back to the start
 * of the next one (a sawtooth sideways), with the gaze held below the lens. From those frames it learns the same
 * measures a notes calibration takes, and keeps following them when they move (the laptop tilted, the notes
 * moved, a new seat). The caller saves what it learns and passes it back in the params (has_notes, notes_offset,
 * notes_side, notes_open, vis_notes, notes_seconds, with notes_learnt 1), so it survives restarts. */
typedef struct {
	int has;          /* 1 once enough reading has been seen */
	float offset;     /* gaze offset while reading (same units as baseline, like cf_calib_result.offset) */
	float side;       /* sideways gaze while reading (eye widths, like cf_calib_result.side) */
	float open;       /* eye openness while reading (like cf_calib_result.openness) */
	int has_vis;      /* Vision's own pupil point was given while reading ... */
	float vis_offset; /* ... and its offset then (like cf_calib_result.vis_offset) */
	float seconds;    /* how much reading the estimate rests on (s) */
	unsigned serial;  /* changes each time the estimate moved enough to be worth saving */
} cf_learnt_notes;

/* Checks a finished lens calibration ("look at the camera") before it is used. reading: the learnt notes, or
 * NULL / has 0 when none are known yet. Returns CF_LOOK_OK, or why the look should be redone:
 *  CF_LOOK_UNSTEADY: the eyes or lids moved far more than a held look does (not just a bit unsteady);
 *  CF_LOOK_LOW: it is not clearly above where the eyes read (by CF_NOTES_MIN_GLANCE), so it was a look at the
 *    screen, not into the lens. Only said when the learnt notes rest on CF_LOOK_READING_MIN seconds of reading
 *    at least: a few seconds of it may be off. */
#define CF_LOOK_OK 0
#define CF_LOOK_UNSTEADY 1
#define CF_LOOK_LOW 2
#define CF_LOOK_READING_MIN 10.0f
int cf_gaze_check_lens_look(const cf_calib_result *res, const cf_learnt_notes *reading);

typedef struct {
	double value, deriv, t;
	int ready;
} cf_one_euro;

/* One gaze channel (vertical glance or sideways position) made robust against measurement outliers:
 * a change bigger than noise is only taken at once when another sensor backs it up (see gaze.c). */
typedef struct {
	float lr_bias;    /* typical left-eye minus right-eye reading (eye widths) */
	float lr_noise;   /* mean |left - right - lr_bias| while they agree: what disagreement is noise */
	float v_anchor;   /* Vision's reading where the accepted gaze is (follows it slowly, reset on a jump) */
	float v_last;     /* Vision's reading last frame */
	float v_noise;    /* mean frame-to-frame change of Vision's reading */
	float v_gain;     /* refined change per Vision change (Vision's point is coarse: it under-reads) */
	float v_gain_err; /* typical error of v_gain (log ratio), seen on the saccades it was learnt from */
	int have_v;
	double v_jump_t;  /* when Vision's reading last jumped (moved past its noise in one frame) */
	float v_off;     /* the reading minus v_gain x Vision's, averaged since the last change taken ... */
	int v_off_n;      /* ... over this many readings (0: none yet) */
	float v_ref;      /* ... and Vision's reading, averaged over the same readings */
	float v_res;      /* recent mean of the readings minus the offset, and of Vision's (see gate()) */
	float v_fast;
	/* A jump of Vision's point that the refined reading has not (yet) shown: not believed for VIS_PEND_TIME
	 * (the fused reading stays where it was), then taken out of the offset unless the refined reading
	 * followed it. */
	int v_pend;       /* one is pending */
	double v_pend_t;  /* when it jumped ... */
	float v_pend_v0;  /* ... from where (Vision's reading) */
	float v_pend_vs;  /* Vision's readings since (sum) ... */
	float v_pend_rs;  /* ... and the refined ones, less where they were before it (sum, clamped) */
	int v_pend_n;
	/* Learning v_gain from a backed saccade, once its landing has been averaged: the refined and Vision
	 * means before it (v_off + v_gain x v_ref, v_ref) ... */
	int learn;
	float learn_x, learn_v;
	int learn_n;      /* ... (over this many readings) */
	/* While outliers are about: up to two other levels the readings keep going to (as offsets from where
	 * Vision's point says the gaze is, see vote()), and the share of readings at the accepted level and at
	 * each, recently (w_) and over a longer memory (s_). */
	float alt[2], w_alt[2], s_alt[2];
	int have_alt[2];
	float w_acc, s_acc;
	float lid_anchor; /* eye openness where the accepted gaze is (vertical channel) */
	int have_lid;
	float m_prev;     /* last frame's combined reading */
	int have_prev;
	int steady_n;     /* frames running the reading has been away from the accepted value, but steady */
	int held_big;     /* last frame's reading was a saccade-sized change that was not taken */
	int away;         /* last frame's reading was past the noise band and not taken at once */
	int followed;     /* this frame's reading persisted away and the accepted value followed it */
	double jump_t;    /* the last change taken at once: when, ... */
	float jump_from, jump_size; /* ... from where, and how big (to see it come straight back) */
	int jump_weak;    /* ... and only both eyes moving alike backed it up (no Vision point, lids still) */
	int jump_vis;     /* ... or Vision's point did: where it was before (jump_v0) and after (jump_v1), and the
			   * gain before that jump taught it anything */
	float jump_v0, jump_v1, jump_gain, jump_gain_err;
	int settle;       /* readings averaged in since that change: the first few are averaged in quickly */
	int land_n;       /* frames since that change: the first LANDING_FRAMES are still its landing */
	int mid;          /* last frame's reading was a move short of a saccade that Vision's point backs (its sign) */
	double out_t;     /* last time an outlier was seen (the measurement counts as clean a while after) */
	double shown;     /* the value the output uses: the accepted one, smoothed and slew-limited (see slew()) */
	double shown_x;   /* the accepted value it last followed */
	int have_shown;
	int gliding;      /* shown is gliding (an S-curve) from glide_from, started at glide_t0, to the accepted value */
	double glide_from, glide_t0;
	/* Vision's frame-to-frame noise tail (see VIS_TAIL_P); a saccade Vision's point backs alone, waiting a
	 * frame (its sign); and the last jump's landing: its readings so far (the first is where it landed) and
	 * when it was taken */
	float v_tail;
	int vwait;
	float land_buf[8];
	int land_k;
	double land_t;
	int land_mode;    /* (how its landing is shown: like the jump, gate()'s return) */
	int rel_s, rel_n; /* frames running both readings have been off the accepted value, and which way */
	/* The last pending jump of Vision's point taken out of the offset (see gate()): by how much (refit_dv,
	 * of Vision's reading), from where (refit_v0, refit_anchor) and when; Vision's point going back undoes it. */
	float refit_dv, refit_v0, refit_anchor;
	double refit_t;
} cf_robust;

/* ---- The notes learner's own state (gaze.c, "Notes learnt from reading"). The last few seconds of frames that may
 * be reading, the sideways jumps found in them, the reading lines between those jumps, and the lines the estimate
 * rests on. */
#define CF_LEARN_FRAMES 512 /* frames kept: over 13 s at the learner's frame rate (at most ~38 a second) */
#define CF_LEARN_JUMPS 32   /* sideways jumps remembered, for the share going the returns' way */
#define CF_LEARN_LINES 24   /* reading lines the estimate rests on (about a minute's worth) */
#define CF_LEARN_BLOCKS 20  /* 1 s blocks the sideways noise is measured over */

typedef struct {
	double t;
	float off, side, open; /* as cf_calib_result's offset, side and openness, for this one frame */
	float vis;             /* Vision's point's offset, NAN when it was not given */
} cf_learn_frame;

/* One reading line: its frames' medians and how much reading it holds (w, seconds; w_vis of them with Vision's
 * point). Also used for the learnt notes passed in the params (a seed). */
typedef struct {
	float w, off, side, open, vis, w_vis;
	float sd; /* robust standard deviation of the vertical over the line (eye widths) */
	double t; /* when it ended */
} cf_learn_line;

/* A sideways jump: where the eyes were before it and after (5-frame medians), at which frames and when. */
typedef struct {
	int sign;
	unsigned first, last; /* the first frame after it (last: of the last jump merged into it) */
	double t_first, t_last;
	float before, after;
} cf_learn_jump;

typedef struct {
	cf_learn_frame f[CF_LEARN_FRAMES]; /* frame k is f[k % CF_LEARN_FRAMES] */
	unsigned n;                       /* frames stored so far */
	unsigned run;                     /* the first frame since the last break (a long gap) */
	double last_t;                    /* when the last frame was stored */
	double gate_t;                    /* when a frame last passed the checks (stored or not) */
	/* The sideways noise: robust standard deviations over 1 s blocks, and their 25th percentile */
	float block_sd[CF_LEARN_BLOCKS];
	int blocks, block_k;
	unsigned block_from;
	float sigma;
	/* Jumps: the one waiting to be merged with a quick follow-up the same way (a corrective saccade), the last
	 * frame a jump was found at, and the recent ones' times and signs */
	cf_learn_jump pend;
	int have_pend;
	unsigned last_jump;
	int have_last_jump;
	double jump_t[CF_LEARN_JUMPS];
	int jump_sign[CF_LEARN_JUMPS];
	int jumps;
	/* Per way a return can go (0: towards -side, 1: towards +side): the last jump that way (and whether one went the
	 * other way since), how many valid lines in a row ended with one (when the first began, the last ended), and the
	 * last valid line not yet used */
	struct {
		int have, crossed;
		cf_learn_jump at;
		int chain;
		double chain_t0, line_t;
		int have_line;
		cf_learn_line line;
		int have_prev;
		cf_learn_line prev; /* the last valid line (for the level check between two lines) */
	} way[2];
	/* The reading spell: the way its returns go (-1, +1; 0: none), a count of spells, when it began, when its
	 * last line ended and how many returns it has had */
	int spell;
	unsigned spell_id;
	double spell_t0, spell_t1;
	int spell_returns;
	/* The estimate: the lines it rests on, oldest first, at most LEARN_MEMORY seconds of them in all */
	cf_learn_line line[CF_LEARN_LINES];
	int lines;
	cf_learnt_notes est;   /* the current estimate (has: any lines) */
	cf_learnt_notes pub;   /* the last one handed out, or the learnt notes from the params (has_pub) */
	int has_pub;
	double pub_t;
	int no_seed;           /* forgotten: not seeded again until the params stop carrying learnt notes */
} cf_notes_learner;

typedef struct {
	cf_one_euro cx[2], cy[2], rx[2], ry[2], eye_h[2];
	cf_one_euro down;
	cf_one_euro roll;  /* head roll (angle of the eye axis), smoothed: sets g->up and g->axis */
	float factor;      /* current, smoothed 0..1 (head turn, face size and losing the face fade it) */
	float factor_mid;  /* the first of its two easing stages */
	/* The glance and sideways position the shift is made from: the live ones while the gaze is on the
	 * notes; while it is away, no further than the edge of the band where the correction applies, and
	 * never growing, so a look-away only ever lets go. */
	float d_lock, side_lock;
	int have_lock;
	/* Looking away (past the notes, or sideways out of them): 0..1, how much of the correction applies. It
	 * switches with a saccade (hidden by the eye's own jump), and eases like `factor` otherwise. */
	float lock_f, lock_mid;
	float last_target; /* for diagnostics */
	cf_pt up;          /* smoothed face "up" direction */
	double last_t;
	int have_geometry;

	int calibrating;
	int calib_count;
	int calib_frames; /* frames seen since calibration started, for the timeout */
	int calib_have_start;
	double calib_start; /* timestamp of the first frame after starting, for the warm-up */
	int calib_failed;
	float calib_samples[64];
	int calib_done;
	float calib_result;

	/* Blink hold, two-point calibration, reading damping and background prediction. */
	cf_one_euro side;       /* filtered sideways gaze offset (eye widths), for reading damping */
	float open_ref;         /* learned open-eye openness, used when params->baseline_open is 0 */
	float last_rel_open;    /* previous relative openness, to see lids closing fast */
	int have_rel_open;
	int blink;              /* lids closed: correction and geometry are held */
	int closing;            /* lids dropping fast: held until it is clear whether it is a blink */
	int half;               /* half blink: lids dropped quickly well below open but not shut. Held like a
				 * blink until they reopen, or released after HALF_MAX (a squint) */
	double half_start;
	float r_peak;           /* recent highest relative openness (decays): how far the lids just dropped */
	float open_r;           /* relative openness before the current hold: where "reopened" is */
	float lid_noise;        /* mean frame-to-frame change of the relative openness while steady: landmark noise */
	double lid_move_t;      /* last time the lids moved (or were held): the lid cue waits for them to settle */
	double hold_start;      /* when the current blink/closing hold began */
	double settle_until;    /* glance stays frozen until then, after a blink */
	double reacq_until;     /* after a blink: a change taken before then is shown as a glide (see GLIDE_TIME) */
	int face_found;
	int tracking;           /* last update saw the face (prediction may extrapolate) */
	cf_pt vel[2];           /* eye centre velocity (px/s) for prediction: quick to follow starts and stops */
	int was_holding;        /* a real blink (2) or half blink (1) happened during the current hold: resync the
				 * glance after it */
	float far_fade;         /* 0..1, "looking far past the notes" fade, applied directly (not eased) */
	float down_noise;       /* typical frame-to-frame change of the raw glance (eye widths): landmark noise */
	float last_down_raw;
	int have_down_raw;
	float side_noise;       /* the same for the raw sideways gaze */
	float last_side_raw;
	int have_side_raw;
	cf_robust rb_down, rb_side; /* outlier rejection and output slew guard for the glance and sideways gaze */
	/* The last few frames' glance state, so a blink or half blink caught a frame or two late (the lids
	 * dragging Vision's pupil point down on the way) can be undone back to before the lids moved. */
	struct {
		double t, down;
		float far;
		int ready;
		int suspect; /* that frame's lids were moving and its pupil move didn't look like a glance */
	} hist[CF_HIST];
	int hist_n;
	float r_slow;           /* relative openness, lightly smoothed (two-frame mean, then ~0.08 s) */
	float r_slow_prev[2];   /* ... its values one and two frames ago (the level before a drop began) */
	float last_r;
	int past_gate;          /* consecutive frames the pupil read past the lid-cue gate */
	int ease;               /* easing the output after a half-blink hold timed out (eyes still) */
	double ease_min_until;
	enum cf_calib_target calib_target;
	float calib_open[64];
	float calib_side[64];
	float calib_open_result, calib_side_result;
	float ap[2][CF_MAX_APERTURE][2]; /* smoothed eye outline, relative to the eye centre */
	int ap_count[2];
	float amount_unit[2];   /* vertical iris shift before strength (px) */
	float hshift_unit[2];   /* sideways iris shift before strength (px, along the eye axis) */
	cf_pt axis;             /* smoothed eye axis (unit, from the left eye towards the right eye) */
	cf_eye last_out[2];
	int last_n;

	/* Head pose (pose.c): the smoothed landmark measures, and the calibration's samples of them. */
	cf_pose pose;
	cf_head calib_head[64];
	int calib_head_n;
	cf_head_ref calib_head_result;
	/* Vision's pupil point while calibrating, and the gain the calibrations gave (see cf_gaze_params). */
	float calib_vis[64];
	int calib_vis_count;
	float calib_vis_result;
	float vis_gain_cal; /* the calibrated gain v_gain was last set to (0: none) */
	float calib_spread, calib_drift, calib_open_spread; /* see cf_calib_result */
	float base_prev; /* the lens baseline the glance state is measured from (it moves with a new calibration) */
	int have_base;
	float d_live_prev, side_live_prev; /* last frame's live glance and sideways position (see d_lock) */
	float noise_slow; /* down_noise's recent peak (forgotten over FAR_NOISE_TAU): sets where the far fade starts */
	int reopen_fast;  /* the correction faded out with the eyes closed for long: it comes back in quickly */

	/* ---- Head and lid compensation (gaze.c, "Head and lids"). What the viewer reads as gaze is the head's
	 * direction plus the eyes' direction in the head. The glance above is only the second; these add the
	 * head (its tilt down and turn, against the lens calibration, as the iris shift that would make up for
	 * them) and restore the lids, which narrow while reading. All in eye widths, smoothed, slew limited,
	 * held through blinks and, like d_lock / side_lock, held while the gaze is away (so a look-away only
	 * ever fades them). */
	cf_one_euro head_pf, head_yf;    /* the head's pitch and yaw for these (see head_angles() in gaze.c) */
	float head_v, head_h;            /* iris-shift equivalent of the head's tilt down / turn (eye widths) */
	float head_v_lock, head_h_lock;
	float up_lock;                   /* the eyes' glance up in the head (<= 0), offsetting head_v_lock */
	float lid_up, lid_lo;            /* upper / lower lid above / below the corner line now (smoothed) */
	float lid_up1, lid_lo1;          /* ... the first of the two smoothing stages */
	int have_lids;
	float up_drop, lo_rise;          /* how far the upper lid has dropped / the lower lid risen against the
					  * lens calibration (smoothed, slew limited; 0 when not calibrated) */
	float up_drop_lock, lo_rise_lock;
	float lid_unit[2];               /* upper-lid lift beyond LID_SHIFT x the iris lift, before strength (px) */
	float low_unit[2];               /* lower-lid shift down, before strength (px) */
	float calib_upper[64], calib_lower[64];
	float calib_upper_result, calib_lower_result;
	/* The glance's catch-up after the head moved (see glide_catch_up() in gaze.c), eye widths. */
	float vor_acc;                   /* how far the head term moved lately (decaying), less what was glided */
	float vor_off;                   /* the part of a catch-up still being glided in */
	float vor_d, vor_h;              /* the glance (d_lock + up_lock) and head_v_lock last frame */
	float vor_raw, vor_gap;          /* the raw glance (lightly smoothed), and how far the glance lagged it */
	double vor_t;
	int have_vor;
	/* Vision's sideways gain (see side_vision_scale() in gaze.c): its sideways point is scaled by svk about
	 * svk_ref before the sideways channel uses it, svk measured from how the refined and Vision's readings
	 * move together (running means, the refined one's variance and their covariance, eye widths). */
	float svk, svk_ref;
	float svk_m, svk_v, svk_mm, svk_mv;
	int have_svk;
	int side_novis; /* Vision's point hardly moves sideways: not used there */
	float side_centre; /* where the sideways damping is centred (where the eyes have been lately) */
	int have_centre;
	/* The head's tilt down as an iris shift, uncapped (head_lids()): past HEAD_READ_MAX, for the far cue; all
	 * of it, and how far it moved lately (decaying), to tell the eyes catching up with the head from a look. */
	float head_far, head_full, head_move;
	int have_head_full;
	float engage_f; /* how far the sideways damping (and the lid restoration) is engaged, 0..1 */
	int have_engage;

	/* ---- Status reasons (cf_gaze_status.too_far / turned) and the notes learnt from reading. The learner's
	 * own state goes here too (see "Notes learnt from reading" in gaze.c). */
	float st_size_f, st_turn_f; /* this frame's face-size fade and head-turn fade (f_size, f_yaw x f_pitch) */
	int st_reading;
	cf_learnt_notes learnt;     /* the current estimate */
	unsigned learnt_taken;      /* the serial last handed out by cf_gaze_take_learnt_notes() */
	cf_notes_learner learn;     /* the notes learner's own state */
} cf_gaze;

void cf_gaze_init(cf_gaze *g);

/* Feeds one frame. Returns how many eyes to warp (0 or 2) and fills `out`. */
int cf_gaze_update(cf_gaze *g, const cf_face *face, const cf_gaze_params *p, double t_seconds, cf_eye out[2]);

/*
 * Background tracking: the worker thread calls cf_gaze_update() with each analysed frame
 * (t_seconds = that frame's timestamp); the video thread then calls cf_gaze_predict() for the
 * frame it is about to output, which moves the eye regions forward to t_now using the tracked
 * velocity so the warp stays on the eyes even though the analysis is a little behind.
 * The caller must serialise access to `g` (one mutex around both calls is enough).
 * Returns 0 or 2 like cf_gaze_update().
 */
int cf_gaze_predict(const cf_gaze *g, const cf_gaze_params *p, double t_now, cf_eye out[2]);

void cf_gaze_get_status(const cf_gaze *g, cf_gaze_status *out);

/*
 * Moves Vision's pupil guess onto the real iris centre using the image itself (dark, round
 * region inside the eye outline). grey is 8-bit brightness, image pixels, origin top left.
 * Returns the guess unchanged if it can't do better.
 */
cf_pt cf_refine_pupil(const uint8_t *grey, int stride, int width, int height, const cf_pt *eye, int count,
		      cf_pt guess);

void cf_gaze_start_calibration(cf_gaze *g, enum cf_calib_target target);
/* Returns 1 once (and clears the flag) when a calibration has just finished, -1 once when it gave
 * up because no steady, front-facing face was seen for CF_CALIB_TIMEOUT_FRAMES, otherwise 0. */
int cf_gaze_take_calibration(cf_gaze *g, cf_calib_result *out);
/* Drops a calibration that is running (nothing is taken from it). */
void cf_gaze_cancel_calibration(cf_gaze *g);

/* Returns 1 (copying them to out) when the learnt notes (see cf_learnt_notes) changed enough to be saved since
 * the last call, else 0. Rare by design: the first time after three lines (8 s) of reading or so, then at most
 * every 18 s, and only on a real change. Learnt notes the model started from (the params) count as taken. */
int cf_gaze_take_learnt_notes(cf_gaze *g, cf_learnt_notes *out);
/* Forgets what was learnt (the notes are learnt afresh). The caller also stops passing the old ones. */
void cf_gaze_forget_learnt_notes(cf_gaze *g);

#ifdef __cplusplus
}
#endif
