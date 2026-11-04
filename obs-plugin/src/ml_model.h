/*
 * Cam Flip: runs the personal eye model (Core ML) on two eye patches at a time.
 *
 * Model contract (tools/model/train.py exports it):
 *   inputs   patch  [2, 1, CF_ML_H, CF_ML_W]  brightness 0..1 (ml_eye.h cf_ml_crop)
 *            target [2, 2]                    where the iris should end up, patch px / (W, H)
 *   outputs  flow   [2, 2, CF_ML_H, CF_ML_W]  patch px (x then y)
 *            res    [2, 1, CF_ML_H, CF_ML_W]
 *            mask   [2, 1, CF_ML_H, CF_ML_W]
 * The model file is the user's own (trained on their face) and lives outside the repo, by default
 * ~/Library/Application Support/cam-flip/eye-model.mlpackage.
 */
#pragma once

#include "ml_eye.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cf_ml_model cf_ml_model;

enum cf_ml_compute { CF_ML_ALL = 0, CF_ML_CPU = 1, CF_ML_CPU_GPU = 2, CF_ML_CPU_ANE = 3 };

/* Loads (compiling it first if it's an .mlpackage / .mlmodel) the model. NULL on failure; the reason
 * is written to err (if given). */
cf_ml_model *cf_ml_load(const char *path, enum cf_ml_compute compute, char *err, int err_size);

/* Default model path (static buffer). */
const char *cf_ml_default_path(void);

/*
 * Runs the model on n (1 or 2) eyes. patches: n x CF_ML_H x CF_ML_W; targets: n x 2 (normalised).
 * Writes flow_x, flow_y, res, mask for each eye into buffers owned by the model; outs[e] points at
 * them until the next call. Returns 0 on failure.
 */
int cf_ml_run(cf_ml_model *m, const float *patches, const float *targets, int n, cf_ml_out outs[2]);

void cf_ml_free(cf_ml_model *m);

/*
 * One frame through the model, in place: for each eye (cf_eye from cf_gaze_predict, with its outline)
 * cuts the patch from luma (a brightness channel at full resolution), asks the model to put the iris at
 * target_img[e] (image px), and blends the answer in with alpha[e]. Two eyes: the one further right in
 * the image is mirrored (as in training). ms (if given) gets the model's own run time. Returns the
 * number of eyes corrected (0 if the model failed or no eye had an outline).
 */
int cf_ml_correct(cf_ml_model *m, const cf_channel *luma, const cf_channel *channels, int count,
		  unsigned res_channels, const cf_eye *eyes, int n, const float (*target_img)[2], const float *alpha,
		  double *ms);

#ifdef __cplusplus
}
#endif
