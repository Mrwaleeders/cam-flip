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

	/* ---- Head and lid compensation (gaze.c, "Head and lids") */
	/* How far the upper lid sits above, and the lower lid below, the line through the eye corners when
	 * looking into the lens, eye widths (lens calibration, mean of both eyes). 0 = unknown (calibrated by
	 * an older version): the lids then only follow the iris as before, nothing is restored. */
	float baseline_upper, baseline_lower;
} cf_gaze_params;

/* "Eye contact strength" goes up to this (200%): over 1 the correction overshoots, so it can make up for
 * a pupil reading that under-reads glances. The warp's own limits (AMOUNT_MAX_RY, the slope limit) still
 * apply, and the lid restoration never goes past the lens-calibrated opening. */
#define CF_STRENGTH_MAX 2.0f

/* A notes calibration is only usable when the notes are clearly below the lens: at least this many
 * eye widths of downward gaze (baseline - notes_offset). Smaller or reversed means the user wasn't
 * looking at notes under the camera (or the lens calibration is off). */
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
	/* ---- Head and lid compensation: median upper / lower lid height above / below the corner line while
	 * calibrating (eye widths, mean of both eyes): baseline_upper / baseline_lower for a lens calibration */
	float upper, lower;
} cf_calib_result;

/* What the filter panel shows. */
typedef struct {
	int face_found;
	int blinking;
	float glance;    /* how far below the lens the gaze is, same units as baseline */
	float factor;    /* 0..1, current correction strength after fades */
} cf_gaze_status;

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
	float v_off;      /* the reading minus v_gain x Vision's, averaged since the last change taken ... */
	int v_off_n;      /* ... over this many readings (0: none yet) */
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
	int settle;       /* readings since that change: the first few are averaged in quickly */
	double out_t;     /* last time an outlier was seen (the measurement counts as clean a while after) */
	double shown;     /* the value the output uses: the accepted one, smoothed and slew-limited (see slew()) */
	double shown_x;   /* the accepted value it last followed */
	int have_shown;
} cf_robust;

typedef struct {
	cf_one_euro cx[2], cy[2], rx[2], ry[2], eye_h[2];
	cf_one_euro down;
	cf_one_euro roll;  /* head roll (angle of the eye axis), smoothed: sets g->up and g->axis */
	float factor;      /* current, smoothed 0..1 (head turn, face size, and looking away all fade it) */
	float factor_mid;  /* the first of its two easing stages */
	/* The glance and sideways position the shift is made from: the live ones while the gaze is on the
	 * notes, held while it is away, so a look-away fades out from where it left. */
	float d_lock, side_lock;
	int have_lock;
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
	int face_found;
	int tracking;           /* last update saw the face (prediction may extrapolate) */
	cf_pt vel[2];           /* eye centre velocity (px/s) for prediction: quick to follow starts and stops */
	int was_holding;        /* a real blink happened during the current hold: resync the glance after it */
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

	/* ---- Head and lid compensation (gaze.c, "Head and lids"). What the viewer reads as gaze is the head's
	 * direction plus the eyes' direction in the head. The glance above is only the second; these add the
	 * head (its tilt down and turn, against the lens calibration, as the iris shift that would make up for
	 * them) and restore the lids, which narrow while reading. All in eye widths, smoothed, slew limited,
	 * held through blinks and, like d_lock / side_lock, held while the gaze is away (so a look-away only
	 * ever fades them). */
	cf_one_euro head_pf, head_yf;    /* the head's pitch and yaw for these (see head_angles() in gaze.c) */
	float head_v, head_h;            /* iris-shift equivalent of the head's tilt down / turn (eye widths) */
	float head_v_lock, head_h_lock;
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
	float vor_d, vor_h;              /* d_lock and head_v_lock last frame */
	float vor_raw, vor_gap;          /* the raw glance (lightly smoothed), and how far d_lock lagged it */
	double vor_t;
	int have_vor;
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

#ifdef __cplusplus
}
#endif
