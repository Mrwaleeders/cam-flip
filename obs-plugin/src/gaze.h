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

typedef struct {
	float x, y; /* image pixels, origin top left */
} cf_pt;

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
} cf_gaze_params;

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

typedef struct {
	cf_one_euro cx[2], cy[2], rx[2], ry[2], eye_h[2];
	cf_one_euro down;
	cf_one_euro roll;  /* head roll (angle of the eye axis), smoothed: sets g->up and g->axis */
	float factor;      /* current, smoothed 0..1 */
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
