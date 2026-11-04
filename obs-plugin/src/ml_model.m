/* Cam Flip: Core ML runner for the personal eye model. See ml_model.h. */
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "ml_model.h"

#include <stdio.h>
#include <string.h>

enum { PN = CF_ML_W * CF_ML_H };

struct cf_ml_model {
	void *model;    /* MLModel, retained */
	void *in_patch; /* MLMultiArray [2,1,H,W] float32, retained */
	void *in_target;
	int has_lens;
	float lens[2][2]; /* iris looking into the lens, patch px: eye left in the image, mirrored eye */
	float flow_x[2][PN], flow_y[2][PN], res[2][PN], mask[2][PN];
};

static void set_err(char *err, int size, NSString *msg)
{
	if (err && size > 0)
		snprintf(err, (size_t)size, "%s", msg.UTF8String ? msg.UTF8String : "error");
}

const char *cf_ml_default_path(void)
{
	static char path[1024];
	NSString *p = [NSHomeDirectory()
		stringByAppendingPathComponent:@"Library/Application Support/cam-flip/eye-model.mlpackage"];
	snprintf(path, sizeof(path), "%s", p.UTF8String);
	return path;
}

cf_ml_model *cf_ml_load(const char *path, enum cf_ml_compute compute, char *err, int err_size)
{
	@autoreleasepool {
		NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
		NSError *e = nil;
		if (![url.pathExtension isEqualToString:@"mlmodelc"]) {
			/* Compile once and keep the compiled copy next to the source (it's the user's own folder). */
			NSURL *compiled = [[url URLByDeletingPathExtension] URLByAppendingPathExtension:@"mlmodelc"];
			NSDictionary *sa = [[NSFileManager defaultManager] attributesOfItemAtPath:url.path error:nil];
			NSDictionary *ca = [[NSFileManager defaultManager] attributesOfItemAtPath:compiled.path error:nil];
			if (!ca || [ca.fileModificationDate compare:sa.fileModificationDate] == NSOrderedAscending) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
				NSURL *tmp = [MLModel compileModelAtURL:url error:&e];
#pragma clang diagnostic pop
				if (!tmp) {
					set_err(err, err_size, [NSString stringWithFormat:@"compile failed: %@", e.localizedDescription]);
					return NULL;
				}
				[[NSFileManager defaultManager] removeItemAtURL:compiled error:nil];
				if (![[NSFileManager defaultManager] moveItemAtURL:tmp toURL:compiled error:&e])
					compiled = tmp; /* couldn't keep it: use the temporary copy */
			}
			url = compiled;
		}
		MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
		switch (compute) {
		case CF_ML_CPU:
			cfg.computeUnits = MLComputeUnitsCPUOnly;
			break;
		case CF_ML_CPU_GPU:
			cfg.computeUnits = MLComputeUnitsCPUAndGPU;
			break;
		case CF_ML_CPU_ANE:
			cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
			break;
		default:
			cfg.computeUnits = MLComputeUnitsAll;
			break;
		}
		MLModel *model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&e];
		if (!model) {
			set_err(err, err_size, [NSString stringWithFormat:@"load failed: %@", e.localizedDescription]);
			return NULL;
		}
		MLMultiArray *ip = [[MLMultiArray alloc] initWithShape:@[@2, @1, @(CF_ML_H), @(CF_ML_W)]
							      dataType:MLMultiArrayDataTypeFloat32
								 error:&e];
		MLMultiArray *it = [[MLMultiArray alloc] initWithShape:@[@2, @2] dataType:MLMultiArrayDataTypeFloat32 error:&e];
		if (!ip || !it) {
			set_err(err, err_size, @"cannot allocate inputs");
			return NULL;
		}
		cf_ml_model *m = calloc(1, sizeof(*m));
		m->model = (__bridge_retained void *)model;
		m->in_patch = (__bridge_retained void *)ip;
		m->in_target = (__bridge_retained void *)it;
		NSString *lens = model.modelDescription.metadata[MLModelCreatorDefinedKey][@"camflip.lens"];
		if (lens && sscanf(lens.UTF8String, "%f %f %f %f", &m->lens[0][0], &m->lens[0][1], &m->lens[1][0],
				   &m->lens[1][1]) == 4)
			m->has_lens = 1;
		return m;
	}
}

/* Copies output [2, C, H, W] channel c of batch b into dst, whatever its type and strides. */
static int read_plane(MLMultiArray *a, int b, int c, float *dst)
{
	if (a.shape.count != 4 || a.shape[2].intValue != CF_ML_H || a.shape[3].intValue != CF_ML_W)
		return 0;
	const NSInteger sb = a.strides[0].integerValue, sc = a.strides[1].integerValue, sh = a.strides[2].integerValue,
			sw = a.strides[3].integerValue;
	__block int ok = 1;
	[a getBytesWithHandler:^(const void *bytes, NSInteger size) {
		(void)size;
		const NSInteger base = b * sb + c * sc;
		if (a.dataType == MLMultiArrayDataTypeFloat32) {
			const float *p = bytes;
			for (int y = 0; y < CF_ML_H; y++)
				for (int x = 0; x < CF_ML_W; x++)
					dst[y * CF_ML_W + x] = p[base + y * sh + x * sw];
		} else if (a.dataType == MLMultiArrayDataTypeFloat16) {
			const __fp16 *p = bytes;
			for (int y = 0; y < CF_ML_H; y++)
				for (int x = 0; x < CF_ML_W; x++)
					dst[y * CF_ML_W + x] = (float)p[base + y * sh + x * sw];
		} else {
			ok = 0;
		}
	}];
	return ok;
}

int cf_ml_run(cf_ml_model *m, const float *patches, const float *targets, int n, cf_ml_out outs[2])
{
	if (!m || n < 1 || n > 2)
		return 0;
	@autoreleasepool {
		MLModel *model = (__bridge MLModel *)m->model;
		MLMultiArray *ip = (__bridge MLMultiArray *)m->in_patch, *it = (__bridge MLMultiArray *)m->in_target;
		/* The model always takes two eyes: with one, the second slot repeats it. */
		[ip getMutableBytesWithHandler:^(void *bytes, NSInteger size, NSArray<NSNumber *> *strides) {
			(void)size;
			(void)strides; /* freshly allocated: contiguous */
			float *pp = bytes;
			for (int b = 0; b < 2; b++)
				memcpy(pp + (size_t)b * PN, patches + (size_t)(b < n ? b : 0) * PN, sizeof(float) * PN);
		}];
		[it getMutableBytesWithHandler:^(void *bytes, NSInteger size, NSArray<NSNumber *> *strides) {
			(void)size;
			(void)strides;
			float *tp = bytes;
			for (int b = 0; b < 2; b++) {
				const int s = b < n ? b : 0;
				tp[b * 2] = targets[s * 2];
				tp[b * 2 + 1] = targets[s * 2 + 1];
			}
		}];
		NSError *e = nil;
		MLDictionaryFeatureProvider *in = [[MLDictionaryFeatureProvider alloc]
			initWithDictionary:@{@"patch": [MLFeatureValue featureValueWithMultiArray:ip],
					     @"target": [MLFeatureValue featureValueWithMultiArray:it]}
				     error:&e];
		if (!in)
			return 0;
		id<MLFeatureProvider> out = [model predictionFromFeatures:in error:&e];
		if (!out)
			return 0;
		MLMultiArray *flow = [out featureValueForName:@"flow"].multiArrayValue;
		MLMultiArray *res = [out featureValueForName:@"res"].multiArrayValue;
		MLMultiArray *mask = [out featureValueForName:@"mask"].multiArrayValue;
		if (!flow || !res || !mask)
			return 0;
		for (int b = 0; b < n; b++) {
			if (!read_plane(flow, b, 0, m->flow_x[b]) || !read_plane(flow, b, 1, m->flow_y[b]) ||
			    !read_plane(res, b, 0, m->res[b]) || !read_plane(mask, b, 0, m->mask[b]))
				return 0;
			outs[b].flow_x = m->flow_x[b];
			outs[b].flow_y = m->flow_y[b];
			outs[b].res = m->res[b];
			outs[b].mask = m->mask[b];
		}
		return 1;
	}
}

void cf_ml_free(cf_ml_model *m)
{
	if (!m)
		return;
	CFRelease(m->model);
	CFRelease(m->in_patch);
	CFRelease(m->in_target);
	free(m);
}

#include <mach/mach_time.h>

static int mirrored(const cf_eye *eyes, int n, int e)
{
	return n == 2 && eyes[e].cx > eyes[1 - e].cx;
}

int cf_ml_lens_target(const cf_ml_model *m, const cf_eye *eyes, int n, int e, float out[2])
{
	if (!m || !m->has_lens || e < 0 || e >= n || n > 2)
		return 0;
	const int flip = mirrored(eyes, n, e);
	cf_ml_frame fr;
	if (!cf_ml_eye_frame(&eyes[e], flip, &fr))
		return 0;
	cf_ml_to_image(&fr, m->lens[flip][0], m->lens[flip][1], &out[0], &out[1]);
	return 1;
}

int cf_ml_correct(cf_ml_model *m, const cf_channel *luma, const cf_channel *channels, int count,
		  unsigned res_channels, const cf_eye *eyes, int n, const float (*target_img)[2], const float *alpha,
		  double *ms)
{
	if (!m || n < 1)
		return 0;
	if (n > 2)
		n = 2;
	static float patches[2][PN];
	float targets[2][2];
	cf_ml_frame fr[2];
	cf_eye use[2];
	float a[2];
	int k = 0;
	for (int e = 0; e < n; e++) {
		const int flip = mirrored(eyes, n, e);
		if (!cf_ml_eye_frame(&eyes[e], flip, &fr[k]))
			continue;
		use[k] = eyes[e];
		a[k] = alpha[e];
		cf_ml_crop(luma, &fr[k], patches[k]);
		float px, py;
		cf_ml_to_patch(&fr[k], target_img[e][0], target_img[e][1], &px, &py);
		targets[k][0] = px / (float)CF_ML_W;
		targets[k][1] = py / (float)CF_ML_H;
		k++;
	}
	if (!k)
		return 0;
	cf_ml_out outs[2];
	const uint64_t t0 = mach_absolute_time();
	const int ok = cf_ml_run(m, &patches[0][0], &targets[0][0], k, outs);
	if (ms) {
		mach_timebase_info_data_t tb;
		mach_timebase_info(&tb);
		*ms = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
	}
	if (!ok)
		return 0;
	cf_ml_apply(channels, count, use, fr, outs, a, k, res_channels);
	return k;
}
