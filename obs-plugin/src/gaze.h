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
#define CF_CALIB_SAMPLES 30
#define CF_CALIB_MIN_SAMPLES 10
#define CF_CALIB_TIMEOUT_FRAMES 300 /* about 10 s at 30 fps */

typedef struct {
	float x, y; /* image pixels, origin top left */
} cf_pt;

typedef struct {
	int found;
	cf_pt left_eye[CF_MAX_EYE_POINTS];
	int left_count;
	cf_pt right_eye[CF_MAX_EYE_POINTS];
	int right_count;
	cf_pt left_pupil, right_pupil;
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
} cf_gaze_params;

enum cf_calib_target { CF_CALIB_LENS = 0, CF_CALIB_NOTES = 1 };

typedef struct {
	enum cf_calib_target target;
	float offset;   /* median gaze offset while calibrating (same units as baseline) */
	float openness; /* median eye openness while calibrating */
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
	float factor;      /* current, smoothed 0..1 */
	float last_target; /* for diagnostics */
	cf_pt up;          /* smoothed face "up" direction */
	double last_t;
	int have_geometry;

	int calibrating;
	int calib_count;
	int calib_frames; /* frames seen since calibration started, for the timeout */
	int calib_failed;
	float calib_samples[64];
	int calib_done;
	float calib_result;
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
