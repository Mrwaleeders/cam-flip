/*
 * Cam Flip for OBS: a video filter for your camera that
 *   - mirrors the video, so everyone in your call sees it mirrored, and
 *   - keeps eye contact when you glance slightly down at notes under the camera.
 *
 * It runs directly on each camera frame as it arrives (in the camera's own pixel format),
 * with no queues, copies of the whole frame, or extra GPU passes, so it adds no frame of delay
 * and doesn't change image quality outside the eyes.
 */
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Vision/Vision.h>

#include <obs-module.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "gaze.h"
#include "warp.h"

#ifndef CAMFLIP_VERSION
#define CAMFLIP_VERSION "dev"
#endif

#ifndef CAMFLIP_TEST
OBS_DECLARE_MODULE()
#endif

MODULE_EXPORT const char *obs_module_name(void)
{
	return "Cam Flip";
}

MODULE_EXPORT const char *obs_module_description(void)
{
	return "Mirror your camera for everyone and keep eye contact while reading notes.";
}

#define S_MIRROR "mirror"
#define S_EYE_CONTACT "eye_contact"
#define S_STRENGTH "strength"
#define S_MAX_GLANCE "max_glance"
#define S_BASELINE "baseline"

struct camflip {
	obs_source_t *context;
	pthread_mutex_t lock;

	/* settings (guarded by lock) */
	bool mirror;
	bool eye_contact;
	float strength;
	float max_glance;
	float baseline;
	bool calibrate_request;

	/* video thread only */
	cf_gaze gaze;
	void *handler; /* VNSequenceRequestHandler, retained */
	void *request; /* VNDetectFaceLandmarksRequest, retained */
	uint8_t *gray;
	size_t gray_size;
	uint8_t *bgra;
	size_t bgra_size;
	bool vision_needs_bgra;
	bool vision_grey_ok; /* Vision has accepted a grey frame at least once */
	bool warned_format;
	bool warned_vision;
	bool warned_flip;
	unsigned long vision_runs;     /* for diagnostics/tests */
	unsigned long vision_failures;
};

static enum cf_format to_cf_format(enum video_format f)
{
	switch (f) {
	case VIDEO_FORMAT_I420:
		return CF_FMT_I420;
	case VIDEO_FORMAT_NV12:
		return CF_FMT_NV12;
	case VIDEO_FORMAT_YUY2:
		return CF_FMT_YUY2;
	case VIDEO_FORMAT_YVYU:
		return CF_FMT_YVYU;
	case VIDEO_FORMAT_UYVY:
		return CF_FMT_UYVY;
	case VIDEO_FORMAT_RGBA:
		return CF_FMT_RGBA;
	case VIDEO_FORMAT_BGRA:
		return CF_FMT_BGRA;
	case VIDEO_FORMAT_BGRX:
		return CF_FMT_BGRX;
	case VIDEO_FORMAT_Y800:
		return CF_FMT_Y800;
	case VIDEO_FORMAT_I444:
		return CF_FMT_I444;
	case VIDEO_FORMAT_I422:
		return CF_FMT_I422;
	default:
		return CF_FMT_UNSUPPORTED;
	}
}

/* ---------------------------------------------------------------- Vision */

static uint8_t *ensure(uint8_t **buf, size_t *size, size_t need)
{
	if (*size < need) {
		free(*buf);
		*buf = malloc(need);
		*size = *buf ? need : 0;
	}
	return *buf;
}

/* Runs Apple Vision on the frame's brightness and fills `face` (image pixels, origin top left). */
static void detect_face(struct camflip *f, struct obs_source_frame *frame, const cf_channel *channels, int count,
			enum cf_format fmt, cf_face *face)
{
	memset(face, 0, sizeof(*face));
	const int w = (int)frame->width, h = (int)frame->height;
	face->image_width = w;
	face->image_height = h;

	@autoreleasepool {
		/* Grey view of the frame. For planar formats this is the luma plane itself (no copy). */
		uint8_t *grey = NULL;
		size_t grey_stride = 0;
		if (fmt == CF_FMT_NV12 || fmt == CF_FMT_I420 || fmt == CF_FMT_I422 || fmt == CF_FMT_I444 ||
		    fmt == CF_FMT_Y800) {
			grey = frame->data[0];
			grey_stride = frame->linesize[0];
		} else {
			/* Packed YUV: copy the luma bytes out. RGB: green is a good stand-in for brightness. */
			const cf_channel *src = &channels[0];
			if (fmt == CF_FMT_RGBA || fmt == CF_FMT_BGRA || fmt == CF_FMT_BGRX)
				src = &channels[1];
			if (!ensure(&f->gray, &f->gray_size, (size_t)w * h))
				return;
			cf_extract_luma(src, f->gray);
			grey = f->gray;
			grey_stride = (size_t)w;
		}
		(void)count;

		CVPixelBufferRef pb = NULL;
		if (!f->vision_needs_bgra) {
			CVPixelBufferCreateWithBytes(kCFAllocatorDefault, (size_t)w, (size_t)h, kCVPixelFormatType_OneComponent8,
						     grey, grey_stride, NULL, NULL, NULL, &pb);
		} else {
			/* Fallback if this macOS version's Vision rejects 8-bit grey buffers. */
			if (!ensure(&f->bgra, &f->bgra_size, (size_t)w * h * 4))
				return;
			for (int y = 0; y < h; y++) {
				const uint8_t *s = grey + (size_t)y * grey_stride;
				uint32_t *d = (uint32_t *)(f->bgra + (size_t)y * w * 4);
				for (int x = 0; x < w; x++)
					d[x] = 0xff000000u | ((uint32_t)s[x] * 0x010101u);
			}
			CVPixelBufferCreateWithBytes(kCFAllocatorDefault, (size_t)w, (size_t)h, kCVPixelFormatType_32BGRA,
						     f->bgra, (size_t)w * 4, NULL, NULL, NULL, &pb);
		}
		if (!pb)
			return;

		VNSequenceRequestHandler *handler = (__bridge VNSequenceRequestHandler *)f->handler;
		VNDetectFaceLandmarksRequest *request = (__bridge VNDetectFaceLandmarksRequest *)f->request;
		NSError *error = nil;
		BOOL ok = [handler performRequests:@[request]
				   onCVPixelBuffer:pb
				       orientation:kCGImagePropertyOrientationUp
					     error:&error];
		CVPixelBufferRelease(pb);

		f->vision_runs++;
		if (!ok) {
			f->vision_failures++;
			/* Only fall back to the slower BGRA path if grey input has never worked. A one-off
			 * failure after grey frames have been accepted is not a format problem. */
			if (!f->vision_needs_bgra && !f->vision_grey_ok) {
				f->vision_needs_bgra = true;
				blog(LOG_INFO, "[cam-flip] Vision rejected grey frames, switching to BGRA input");
			} else if (!f->warned_vision) {
				f->warned_vision = true;
				const char *why = error.localizedDescription.UTF8String;
				blog(LOG_WARNING, "[cam-flip] face tracking failed: %s", why ? why : "unknown error");
			}
			return;
		}
		if (!f->vision_needs_bgra)
			f->vision_grey_ok = true;

		VNFaceObservation *best = nil;
		for (VNFaceObservation *obs in request.results) {
			if (!best || obs.boundingBox.size.width > best.boundingBox.size.width)
				best = obs;
		}
		VNFaceLandmarks2D *lm = best.landmarks;
		if (!best || !lm.leftEye || !lm.rightEye || !lm.leftPupil || !lm.rightPupil)
			return;

		const CGSize size = CGSizeMake(w, h);
		VNFaceLandmarkRegion2D *regions[2] = {lm.leftEye, lm.rightEye};
		cf_pt *outs[2] = {face->left_eye, face->right_eye};
		int *counts[2] = {&face->left_count, &face->right_count};
		for (int e = 0; e < 2; e++) {
			const CGPoint *pts = [regions[e] pointsInImageOfSize:size];
			int n = (int)regions[e].pointCount;
			if (n > CF_MAX_EYE_POINTS)
				n = CF_MAX_EYE_POINTS;
			for (int i = 0; i < n; i++) {
				outs[e][i].x = (float)pts[i].x;
				outs[e][i].y = (float)(h - pts[i].y); /* Vision's origin is bottom left */
			}
			*counts[e] = n;
		}
		const CGPoint *lp = [lm.leftPupil pointsInImageOfSize:size];
		const CGPoint *rp = [lm.rightPupil pointsInImageOfSize:size];
		if (lm.leftPupil.pointCount < 1 || lm.rightPupil.pointCount < 1)
			return;
		face->left_pupil = (cf_pt){(float)lp[0].x, (float)(h - lp[0].y)};
		face->right_pupil = (cf_pt){(float)rp[0].x, (float)(h - rp[0].y)};

		face->face_width = (float)(best.boundingBox.size.width * w);
		if (best.yaw != nil) {
			face->has_pose = 1;
			face->yaw_deg = (float)(best.yaw.doubleValue * 180.0 / M_PI);
			face->pitch_deg = best.pitch != nil ? (float)(best.pitch.doubleValue * 180.0 / M_PI) : 0.0f;
		}
		face->found = 1;
	}
}

/* ---------------------------------------------------------------- Filter */

static const char *camflip_get_name(void *unused)
{
	(void)unused;
	return "Cam Flip (mirror + eye contact)";
}

static void camflip_update(void *data, obs_data_t *settings)
{
	struct camflip *f = data;
	pthread_mutex_lock(&f->lock);
	f->mirror = obs_data_get_bool(settings, S_MIRROR);
	f->eye_contact = obs_data_get_bool(settings, S_EYE_CONTACT);
	f->strength = (float)obs_data_get_double(settings, S_STRENGTH) / 100.0f;
	f->max_glance = (float)obs_data_get_double(settings, S_MAX_GLANCE) / 100.0f;
	/* The calibration result lives in the settings too (see store_baseline), so a saved value
	 * comes back on load and the Defaults button resets it. */
	f->baseline = (float)obs_data_get_double(settings, S_BASELINE);
	pthread_mutex_unlock(&f->lock);
}

/* A calibration result on its way from the video thread to the source's settings. */
struct baseline_task {
	obs_weak_source_t *source;
	struct camflip *f;
	float baseline;
};

/* Runs on the UI thread, where settings are normally edited, so it never races the properties
 * window. Writing into the settings means the value survives a crash (OBS saves settings with
 * the scene collection) without needing a .save callback. */
static void store_baseline_task(void *param)
{
	struct baseline_task *t = param;
	obs_source_t *source = obs_weak_source_get_source(t->source);
	if (source) {
		obs_data_t *settings = obs_source_get_settings(source);
		obs_data_set_double(settings, S_BASELINE, t->baseline);
		obs_data_release(settings);

		/* An update that ran between calibrating and now read the old value from the settings.
		 * Holding a reference to the source keeps the filter (and f) alive here. */
		pthread_mutex_lock(&t->f->lock);
		t->f->baseline = t->baseline;
		pthread_mutex_unlock(&t->f->lock);
		obs_source_release(source);
	}
	obs_weak_source_release(t->source);
	bfree(t);
}

/* Called on the video thread when a calibration finishes. */
static void store_baseline(struct camflip *f, float baseline)
{
	if (!f->context) /* no real source, e.g. in tests */
		return;
	struct baseline_task *t = bzalloc(sizeof(*t));
	t->source = obs_source_get_weak_source(f->context);
	t->f = f;
	t->baseline = baseline;
	obs_queue_task(OBS_TASK_UI, store_baseline_task, t, false);
}

static void *camflip_create(obs_data_t *settings, obs_source_t *source)
{
	struct camflip *f = bzalloc(sizeof(*f));
	f->context = source;
	pthread_mutex_init(&f->lock, NULL);
	cf_gaze_init(&f->gaze);

	@autoreleasepool {
		VNDetectFaceLandmarksRequest *request = [[VNDetectFaceLandmarksRequest alloc] init];
		request.constellation = VNRequestFaceLandmarksConstellation76Points;
		f->request = (void *)CFBridgingRetain(request);
		f->handler = (void *)CFBridgingRetain([[VNSequenceRequestHandler alloc] init]);
	}

	camflip_update(f, settings);
	return f;
}

static void camflip_destroy(void *data)
{
	struct camflip *f = data;
	if (f->request)
		CFBridgingRelease(f->request);
	if (f->handler)
		CFBridgingRelease(f->handler);
	free(f->gray);
	free(f->bgra);
	pthread_mutex_destroy(&f->lock);
	bfree(f);
}

static void camflip_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, S_MIRROR, true);
	obs_data_set_default_bool(settings, S_EYE_CONTACT, true);
	obs_data_set_default_double(settings, S_STRENGTH, 70.0);
	obs_data_set_default_double(settings, S_MAX_GLANCE, 45.0);
	obs_data_set_default_double(settings, S_BASELINE, 0.05);
}

static bool camflip_calibrate_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	(void)props;
	(void)property;
	struct camflip *f = data;
	if (!f)
		return false;
	pthread_mutex_lock(&f->lock);
	f->calibrate_request = true;
	pthread_mutex_unlock(&f->lock);
	blog(LOG_INFO, "[cam-flip] calibrating: look into the camera for one second");
	return false;
}

static obs_properties_t *camflip_properties(void *data)
{
	(void)data;
	obs_properties_t *props = obs_properties_create();

	obs_property_t *p = obs_properties_add_bool(props, S_MIRROR, "Mirror video (everyone in the call sees it mirrored)");
	obs_property_set_long_description(
		p, "Call apps only mirror your own preview. This mirrors the video that is actually sent.");

	obs_properties_add_bool(props, S_EYE_CONTACT, "Eye contact");

	p = obs_properties_add_float_slider(props, S_STRENGTH, "Eye contact strength", 0.0, 100.0, 1.0);
	obs_property_float_set_suffix(p, "%");

	p = obs_properties_add_float_slider(props, S_MAX_GLANCE, "Correct glances up to", 20.0, 80.0, 1.0);
	obs_property_float_set_suffix(p, "% of an eye height");
	obs_property_set_long_description(
		p, "Glances further down than this (like at your keyboard) fade back to your real eyes so it stays natural.");

	p = obs_properties_add_button2(props, "calibrate", "Calibrate: look into the camera, then click",
				       camflip_calibrate_clicked, data);
	obs_property_set_long_description(p, "Takes about a second. Redo it if you move your screen or chair a lot.");

	obs_properties_add_text(props, "info",
				"Tip: in Teams/Zoom/Meet pick “OBS Virtual Camera” and turn off “Mirror my video”, "
				"so your own preview matches what others see.",
				OBS_TEXT_INFO);
	return props;
}

static struct obs_source_frame *camflip_filter_video(void *data, struct obs_source_frame *frame)
{
	struct camflip *f = data;

	pthread_mutex_lock(&f->lock);
	const bool mirror = f->mirror;
	const bool eye_contact = f->eye_contact;
	const bool calibrate = f->calibrate_request;
	f->calibrate_request = false;
	cf_gaze_params params = {.strength = f->strength, .max_glance = f->max_glance, .baseline = f->baseline};
	pthread_mutex_unlock(&f->lock);

	const enum cf_format fmt = to_cf_format(frame->format);
	cf_channel channels[8];
	const int count = cf_channels(fmt, frame->data, frame->linesize, (int)frame->width, (int)frame->height, channels);
	if (count == 0) {
		if (!f->warned_format) {
			f->warned_format = true;
			blog(LOG_WARNING,
			     "[cam-flip] camera pixel format %d not supported; set the camera's Video Format to NV12 or "
			     "UYVY in its properties",
			     (int)frame->format);
		}
		return frame;
	}

	if (frame->flip) {
		/* Upside-down (bottom-up) frames: face tracking and the eye warp assume top-down rows, so
		 * eye contact is skipped. Say so once, and don't leave a calibration silently pending. */
		if (eye_contact && !f->warned_flip) {
			f->warned_flip = true;
			blog(LOG_WARNING, "[cam-flip] eye contact is unavailable: this camera delivers upside-down "
					  "frames. Mirroring still works");
		}
		if (calibrate || f->gaze.calibrating) {
			cf_gaze_init(&f->gaze);
			blog(LOG_WARNING, "[cam-flip] calibration cancelled: this camera delivers upside-down frames, "
					  "so eye contact is unavailable");
		}
	} else if (calibrate) {
		cf_gaze_start_calibration(&f->gaze);
	}

	/* Eye contact first (on the camera's original orientation), then mirror. */
	if ((eye_contact || f->gaze.calibrating) && !frame->flip) {
		cf_face face;
		detect_face(f, frame, channels, count, fmt, &face);
		cf_eye eyes[2];
		const double t = (double)frame->timestamp / 1e9;
		const int n = cf_gaze_update(&f->gaze, &face, &params, t, eyes);

		float baseline;
		const int calibrated = cf_gaze_take_calibration(&f->gaze, &baseline);
		if (calibrated > 0) {
			pthread_mutex_lock(&f->lock);
			f->baseline = baseline;
			pthread_mutex_unlock(&f->lock);
			store_baseline(f, baseline);
			blog(LOG_INFO, "[cam-flip] calibrated (baseline %.3f)", baseline);
		} else if (calibrated < 0) {
			blog(LOG_WARNING, "[cam-flip] calibration gave up: no steady, front-facing face seen. "
					  "Look straight into the camera and click Calibrate again");
		}

		if (eye_contact && n > 0)
			cf_warp_frame(channels, count, eyes, n);
	}

	if (mirror)
		cf_mirror_frame(channels, count);

	return frame;
}

static struct obs_source_info camflip_filter = {
	.id = "camflip_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_ASYNC_VIDEO,
	.get_name = camflip_get_name,
	.create = camflip_create,
	.destroy = camflip_destroy,
	.update = camflip_update,
	.get_defaults = camflip_defaults,
	.get_properties = camflip_properties,
	.filter_video = camflip_filter_video,
};

bool obs_module_load(void)
{
	obs_register_source(&camflip_filter);
	blog(LOG_INFO, "[cam-flip] loaded (version %s)", CAMFLIP_VERSION);
	return true;
}
