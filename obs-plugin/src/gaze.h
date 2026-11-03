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

#include "warp.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CF_MAX_EYE_POINTS 32
#define CF_CALIB_SAMPLES 30
#define CF_CALIB_MIN_SAMPLES 10
#define CF_CALIB_TIMEOUT_FRAMES 300 /* about 10 s at 30 fps */
#define CF_LOST_GRACE_S 0.1 /* a face missing for less than this is a detection dropout: hold the warp */

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
	float strength;   /* 0..1, how much of a downward glance to cancel */
	float max_glance; /* how far down (in eye heights) is still corrected before fading out */
	float baseline;   /* pupil offset when looking into the lens (from calibration) */
} cf_gaze_params;

typedef struct {
	double value, deriv, t;
	int ready;
} cf_one_euro;

typedef struct {
	cf_one_euro cx[2], cy[2], rx[2], ry[2], eye_h[2];
	cf_one_euro down;
	cf_one_euro roll;  /* head roll (angle of the line between the eyes), smoothed */
	float factor;     /* current, smoothed 0..1 */
	float last_target; /* for diagnostics */
	cf_pt up;          /* smoothed face "up" direction */
	double last_t;
	double lost_time;  /* how long the face has been missing, in seconds (0 while it's found) */
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

void cf_gaze_start_calibration(cf_gaze *g);
/* Returns 1 once (and clears the flag) when a calibration has just finished, -1 once when it gave
 * up because no steady, front-facing face was seen for CF_CALIB_TIMEOUT_FRAMES, otherwise 0. */
int cf_gaze_take_calibration(cf_gaze *g, float *baseline_out);

#ifdef __cplusplus
}
#endif
