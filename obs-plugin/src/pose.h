/*
 * Cam Flip: head pose (yaw and pitch) from the face landmarks of every frame.
 *
 * Pure C with no Apple or OBS dependencies, so it can be tested anywhere (tests/test_pose.c).
 *
 * Apple Vision's own head angles are not usable for a smooth fade: on a MacBook camera its yaw reads
 * only 0 or 45 degrees (never negative) and its pitch is always 0, and it comes only from full
 * detections. This measures the head pose continuously instead, on every frame (tracked or full), from
 * landmarks that are fixed to the skull and so do not move with expressions, gaze or blinks: the eye
 * corners and the nose. The geometry (see pose.c for the full model and its constants):
 *
 *  - The nose sits in front of the plane of the eyes. Turning the head moves it sideways relative to
 *    the eyes by about depth x tan(yaw) inter-eye distances; tilting it moves it up or down.
 *  - Each eye's outer corner sits further back than its inner corner. Turning the head makes the eye
 *    turning towards the camera look wider than the other; tilting it moves both outer corners up (head
 *    down) or down (head up) relative to the inner ones.
 *
 * Every measure is taken in the frame the two eyes define (origin between them, along the line through
 * them, in inter-eye distances), so moving, rolling or scaling the face changes none of them. A rigid
 * 3D model built from the person's own frontal measures (learnt by the "look into the camera"
 * calibration) and typical depths turns them into yaw and pitch in degrees, in closed form (no search,
 * so the angles are continuous in the landmarks).
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	float x, y; /* image pixels, origin top left */
} cf_pt;

/* Most landmark points used per region (more are ignored). */
#define CF_HEAD_MAX_POINTS 64

/* What the landmarks of one frame say about the head, all in the eyes' frame (see cf_head_measure). */
typedef struct {
	float side;  /* nose centre along the eye line from the point between the eyes, inter-eye distances
		      * (towards the image's right is positive) */
	float vert;  /* nose centre across the eye line, inter-eye distances (towards the chin is positive) */
	float eyes;  /* (left width - right width) / (sum), eyes as they appear left and right in the image */
	float tilt;  /* how much higher the outer corners are than the inner ones, per eye width (mean of
		      * both eyes; a head roll changes neither eye's tilt relative to the eye line) */
} cf_head;

/* The person's face looking into the camera, from the lens calibration. ok 0 = not calibrated yet
 * (or calibrated by an older version): a typical face is assumed, see cf_head_angles(). */
typedef struct {
	int ok;
	float side, vert, eyes, tilt;
} cf_head_ref;

/* The correction fades out as the head turns (yaw) or tilts up or down (pitch) further than this
 * from the calibrated pose, degrees: all there up to _LO, gone from _HI. The same both ways. */
#define CF_YAW_FADE_LO 15.0f
#define CF_YAW_FADE_HI 28.0f
#define CF_PITCH_FADE_LO 18.0f
#define CF_PITCH_FADE_HI 32.0f

/* The lens calibration needs this many frames with head measures to learn the reference. */
#define CF_HEAD_MIN_SAMPLES 8

/*
 * Measures one frame. eye_a / eye_b: the two eye outlines (either order, at least 3 points each), nose:
 * points of the nose (outline, crest; any rigid nose points, at least 1). Image pixels, y down.
 * Returns 1 and fills `out`, or 0 if the landmarks are unusable (too few, degenerate, NaN).
 */
int cf_head_measure(const cf_pt *eye_a, int na, const cf_pt *eye_b, int nb, const cf_pt *nose, int nn,
		    cf_head *out);

/*
 * Head yaw and pitch, degrees, relative to the calibrated pose `ref` (positive yaw: the face turns
 * towards the image's right; positive pitch: the head tilts down). Without a reference (ref NULL or
 * not ok), a typical face is assumed: yaw is then known to a few degrees only, so it is reported a
 * little smaller (by CF_HEAD_DEFAULT_YAW_MARGIN), and pitch, which depends on where the camera sits
 * and on the face far more, is reported as 0 (not measured). Continuous in the measures.
 */
#define CF_HEAD_DEFAULT_YAW_MARGIN 4.0f
void cf_head_angles(const cf_head *m, const cf_head_ref *ref, float *yaw_deg, float *pitch_deg);

/* The reference from calibration samples (the median of each measure). ok 0 if fewer than
 * CF_HEAD_MIN_SAMPLES. Reorders `samples`. */
cf_head_ref cf_head_reference(cf_head *samples, int n);

/*
 * Per-frame pose, lightly smoothed: it drives a fade, not geometry, so it only needs to be steady and
 * free of single-frame jumps. Each measure goes through a median of the last three frames (a one-frame
 * landmark glitch never shows) and then a short exponential average (CF_POSE_TAU). Missing measures
 * (no nose found) hold the last pose for up to CF_POSE_HOLD; after a longer gap it starts afresh.
 * A zeroed struct is a valid, empty state.
 */
#define CF_POSE_TAU 0.10  /* s */
#define CF_POSE_HOLD 0.5  /* s */
typedef struct {
	cf_head hist[3]; /* the last raw measures, newest last */
	int hist_n;
	cf_head smooth;  /* their medians, averaged */
	double t;        /* time of the last measure */
	int ready;
	float yaw, pitch; /* the last output, degrees (for diagnostics) */
} cf_pose;

/* m: this frame's measures, or NULL. Returns 1 and the smoothed pose relative to `ref` (see
 * cf_head_angles), or 0 if there is no recent measure. */
int cf_pose_update(cf_pose *p, const cf_head *m, const cf_head_ref *ref, double t, float *yaw_deg,
		   float *pitch_deg);

#ifdef __cplusplus
}
#endif
