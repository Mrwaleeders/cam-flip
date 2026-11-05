/*
 * Cam Flip: the "personal model" eye engine, pixel side (pure C, no Apple or OBS dependencies).
 *
 * A small learned model (trained on the user's own face, see tools/model/) looks at each eye in a
 * fixed-size, corner-aligned patch and says how to change it so the iris ends up at a target
 * position: a flow field (where each patch pixel should take its content from), a brightness
 * residual (what a warp can't make, e.g. the sclera above a lowered lower lid), and a mask for the
 * residual. This file
 *   - finds each eye's patch frame from its outline (cf_ml_eye_frame),
 *   - cuts the patch the model reads out of the brightness channel (cf_ml_crop),
 *   - applies the model's answer to the real frame at its native resolution (cf_ml_apply): the flow
 *     moves the full-resolution pixels (bicubic), the residual is added on top, and everything fades
 *     to nothing at the edge of the eye's ellipse. Pixels outside the ellipses are never touched.
 *
 * Left and right eyes share one model: the eye further right in the image is mirrored into the patch,
 * so the inner corner is always on the same side.
 */
#pragma once

#include "warp.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CF_ML_W 128            /* patch width (model input) */
#define CF_ML_H 64             /* patch height */
#define CF_ML_CORNER_L 24.0f   /* patch x of the outer/inner eye corners: the eye is 80 patch px wide */
#define CF_ML_CORNER_R 104.0f
#define CF_ML_CORNER_Y 34.0f   /* patch y of the corner line (a little below the middle: more lid above) */
#define CF_ML_MAX_FLOW 16.0f   /* largest flow the model can output, patch px */

/* Image -> patch affine map: patch = M * (x, y, 1). inv maps back. */
typedef struct {
	float m[6];   /* px = m0 x + m1 y + m2, py = m3 x + m4 y + m5 */
	float inv[6];
	int flip;     /* 1 if mirrored (the eye on the right of the image) */
} cf_ml_frame;

/* Patch frame of an eye from its two corners (image pixels). flip mirrors it. */
void cf_ml_frame_from_corners(float ax, float ay, float bx, float by, int flip, cf_ml_frame *out);

/* Patch frame of an eye from its outline (cf_eye aperture): the corners are the outline's extremes
 * along its long axis. Returns 0 if the eye has no outline. */
int cf_ml_eye_frame(const cf_eye *e, int flip, cf_ml_frame *out);

void cf_ml_to_patch(const cf_ml_frame *f, float x, float y, float *px, float *py);
void cf_ml_to_image(const cf_ml_frame *f, float px, float py, float *x, float *y);

/* Cuts the model's input out of the brightness channel: CF_ML_H x CF_ML_W floats in 0..1, rows top
 * to bottom (bilinear; outside the frame repeats the edge). */
void cf_ml_crop(const cf_channel *luma, const cf_ml_frame *f, float *out);

/* Where the iris centre sits across the eye in a cropped patch (patch x), from the iris's own left-right
 * symmetry in a band of rows just below the pupil centre py: the x within +-search of x_guess about which
 * the brightness profile is most mirror-like. Sub-pixel. Unlike the tracked pupil, which reads a sideways
 * move at only about half its size on real eyes, this follows the iris itself. Returns x_guess if the band
 * is off the patch. */
float cf_ml_iris_x(const float *patch, float x_guess, float py, float search);

/* The model's answer for one eye, each CF_ML_H x CF_ML_W, row major. */
typedef struct {
	const float *flow_x, *flow_y; /* patch px: output(p) = input(p + flow(p)) */
	const float *res;             /* brightness to add, 0..1 units (only to brightness channels) */
	const float *mask;            /* 0..1, how much of res to add */
} cf_ml_out;

/*
 * Applies the model's answer for each eye to every channel of a frame, in place. alpha (0..1, per
 * eye) blends between the original and the model's result. Only pixels inside each eye's ellipse
 * (cf_eye cx, cy, rx, ry) change, fading to zero at its edge. The brightness residual goes only to
 * the channels whose bit is set in res_channels (see cf_ml_residual_channels).
 */
void cf_ml_apply(const cf_channel *channels, int count, const cf_eye *eyes, const cf_ml_frame *frames,
		 const cf_ml_out *outs, const float *alpha, int eye_count, unsigned res_channels);

/* Which of cf_channels()' channels carry brightness for a format: Y for YUV formats, R, G and B (not
 * alpha) for RGB ones. */
unsigned cf_ml_residual_channels(enum cf_format format);

#ifdef __cplusplus
}
#endif
