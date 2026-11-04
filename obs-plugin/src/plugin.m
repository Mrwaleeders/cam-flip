/*
 * Cam Flip for OBS: a video filter for your camera that
 *   - mirrors the video, so everyone in your call sees it mirrored, and
 *   - keeps eye contact when you glance slightly down at notes under the camera.
 *
 * Threads:
 *   - Video thread (filter_video): copies the frame's brightness into a spare buffer, asks the
 *     gaze model where the eyes are *now* (cf_gaze_predict), warps the eyes and mirrors. It never
 *     waits for face tracking, so the filter adds no delay and never stalls OBS.
 *   - Tracking thread (one per filter): runs Apple Vision on the newest copied frame (older
 *     frames are dropped, never queued; frames 2560 px and wider are shrunk first), refines the
 *     pupils on the full-resolution copy and feeds cf_gaze_update().
 *   The gaze model is shared by both and guarded by `gaze_lock`; everything else that both
 *   threads touch is guarded by `lock`. The two locks are never held together.
 *
 * Pixels are processed in the camera's own format, with no conversion, so nothing outside the
 * eyes changes.
 */
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Vision/Vision.h>

#include <obs-module.h>
#include <math.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
#define S_SHOW_EYES "show_eyes"
#define S_BASELINE "baseline"
#define S_BASELINE_OPEN "baseline_open"
#define S_NOTES_OFFSET "notes_offset"
#define S_NOTES_SIDE "notes_side"
#define S_HAS_NOTES "has_notes"
#define S_NOTES_OPEN "notes_open"
/* The face looking into the camera, learnt by step 1 (pose.c's cf_head_ref): head turns and tilts are
 * measured from it. No S_HEAD_REF (step 1 done by an older version): a typical face is assumed. */
#define S_HEAD_REF "head_ref"
#define S_HEAD_SIDE "head_side"
#define S_HEAD_VERT "head_vert"
#define S_HEAD_EYES "head_eyes"
#define S_HEAD_TILT "head_tilt"
/* The lids looking into the camera, learnt by step 1 (upper lid above / lower lid below the eye corners'
 * line, eye widths): what reading narrows is restored from these. 0 (older step 1): not restored. */
#define S_BASELINE_UPPER "baseline_upper"
#define S_BASELINE_LOWER "baseline_lower"
#define S_CALIB_VERSION "calib_version"

/* Vision's own pupil point at each calibration step (cf_calib_result.vis_offset), which tells the gaze model how
 * much of a glance Vision's point shows. S_VIS_CAL: bit 1 = taken at the lens, bit 2 = at the notes. */
#define S_VIS_CAL "vis_cal"
#define S_VIS_BASELINE "vis_baseline"
#define S_VIS_NOTES "vis_notes"

/* Version 2: gaze offsets are in eye widths. Older calibrations are in other units. */
#define CALIB_VERSION 2
#define DEFAULT_BASELINE 0.0 /* eye widths; pupil level with the eye corners */

/* Internal hotkey name for a filter with the default name; other filters add their own name,
 * see hotkey_names(). Key bindings are saved under this name, so it must stay stable. */
#define HOTKEY_NAME "camflip.toggle_eye_contact"
#define HOTKEY_DESC "Cam Flip: toggle eye contact"
#define FILTER_NAME "Cam Flip (mirror + eye contact)"

#define FULL_DETECT_NS 1000000000ull /* re-run full face detection at least this often */
#define LOG_EVERY_NS 10000000000ull  /* timing summary in the log */
#define REFRESH_NS 1000000000ull     /* settings panel status refresh, at most this often */
#define CALIB_STALL_NS 12000000000ull /* calibration gives up after this long without camera frames */

/* Vision runs on frames up to twice this wide at full size (the pupil needs the detail). Wider
 * frames (1440p, 4K) are shrunk by the whole factor that keeps them at least this wide: Vision's
 * time grows with the pixel count. 2048 px stays full size; 2560 -> 1280, 3840 -> 1280. */
#define VISION_MIN_W 1280

static uint64_t now_ns(void)
{
	return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

typedef struct {
	uint8_t *data; /* 8-bit brightness, tightly packed (stride = w) */
	size_t cap;
	int w, h;
	double t; /* frame timestamp, seconds */
} grey_frame;

enum calib_state { CALIB_IDLE = 0, CALIB_LENS = 1, CALIB_NOTES = 2 };

struct camflip {
	obs_source_t *context;

	/* ---- guarded by `lock` ---- */
	pthread_mutex_t lock;
	pthread_cond_t wake; /* worker waits on this for a new frame or stop */

	/* settings */
	bool mirror;
	bool eye_contact;
	bool show_eyes;
	float strength;
	float max_glance;
	float baseline;
	float baseline_open;
	float notes_offset;
	float notes_side;
	float notes_open; /* eye openness at the notes; 0 = unknown (calibrated by an older version) */
	cf_head_ref head_ref; /* the face looking into the camera; ok 0 = not learnt (older step 1) */
	float baseline_upper, baseline_lower; /* lids looking into the camera; 0 = not learnt (older step 1) */
	bool has_notes;
	bool notes_stale; /* notes no longer below the (newer) lens calibration: not used until redone */
	bool lens_done;
	bool needs_recalib; /* old settings in old units were reset */

	/* calibration requests (from the panel) and progress (from the worker) */
	int calib_request; /* 0 none, else CALIB_LENS / CALIB_NOTES */
	int calib_running; /* CALIB_IDLE / CALIB_LENS / CALIB_NOTES */
	int calib_failed;  /* CALIB_IDLE or the step that gave up */
	bool calib_notes_bad; /* last notes calibration was not below the camera: rejected */
	bool calib_no_frames; /* last calibration gave up: no camera frames arrived */
	bool frames_flipped;  /* the camera sends upside-down frames (frame->flip): not tracked */
	bool calib_cancel;    /* worker: drop the calibration the gaze model may still be running */
	uint64_t calib_progress_ns; /* when the calibration was requested or last got a frame */
	const char *calib_done_msg;

	/* Vision's pupil point at the calibrations (see S_VIS_CAL), and whether calib_done_msg is a hint that
	 * the last calibration may be off (cf_gaze_calib_hints) */
	int vis_cal;
	float vis_baseline, vis_notes;
	bool calib_hint;

	/* frame hand-off: two buffers, the worker owns `busy`, the newest waiting one is `pending` */
	grey_frame frames[2];
	int busy;
	int pending;
	bool stop;

	/* status for the panel */
	bool st_face;
	float st_factor;
	double ema_vision_ms, ema_warp_ms, ema_fps;
	double last_frame_t; /* worker, for the analysed-fps estimate */

	/* timing summary for the log (video thread logs it) */
	double win_vision_ms;
	unsigned long win_vision_n, win_full_n;
	double win_warp_ms;
	unsigned long win_warp_n;
	uint64_t win_start;

	/* diagnostics (tests) */
	unsigned long vision_runs; /* frames analysed */
	unsigned long vision_failures;
	unsigned long full_detections;
	unsigned long tracked_runs;
#ifdef CAMFLIP_TEST
	/* Diagnostics for tests and tools/diag_video.m: Vision's own pupil points before refinement,
	 * its landmark confidence, and whether the last pass was a full detection. */
	cf_pt dbg_raw_pupil[2];
	float dbg_conf;
	cf_face seen_face; /* what Vision saw on the last analysed frame (full-resolution pixels) */
	bool test_clock;   /* detect_face times re-detection by frame timestamps, not the wall clock */
#endif

	obs_hotkey_id hotkey;

	/* ---- guarded by `gaze_lock` ---- */
	pthread_mutex_t gaze_lock;
	cf_gaze gaze;

	/* ---- worker thread only ---- */
	pthread_t worker;
	bool worker_started;
	void *handler;   /* VNSequenceRequestHandler, retained */
	void *request;   /* VNDetectFaceLandmarksRequest, retained */
	void *last_face; /* VNFaceObservation to track from on the next frame, retained, or NULL */
	grey_frame small; /* shrunk copy for Vision, for frames at least 2 x VISION_MIN_W wide */
	uint8_t *bgra;
	size_t bgra_size;
	bool vision_needs_bgra;
	bool vision_grey_ok; /* Vision has accepted a grey frame at least once */
	bool warned_vision;
	bool tracking_disabled;
	int track_errors;
	uint64_t last_detect_ns;
	int last_w, last_h;
	/* geometry learned at the last full detection, used to move the face box with the head */
	double ref_off_x, ref_off_y; /* box centre minus landmark centroid, in box widths */
	double ref_aspect;           /* box height / width, pixels */
	double ref_width_per_iod;    /* box width / distance between the eyes */
	double ref_side, ref_vert;   /* nose position relative to the eyes (head pose proxy) */
	bool ref_has_nose;
	double pose_roll, pose_yaw, pose_pitch;
	bool pose_has_roll, pose_has_yaw, pose_has_pitch;
	unsigned int last_sig;
	uint64_t last_refresh;

	/* ---- video thread only ---- */
	bool warned_format;
	bool warned_flip;
	bool vt_was_on;

#ifdef CAMFLIP_TEST
	bool test_pause; /* guarded by `lock`: worker leaves frames alone */
#endif
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

static bool is_rgb(enum cf_format fmt)
{
	return fmt == CF_FMT_RGBA || fmt == CF_FMT_BGRA || fmt == CF_FMT_BGRX;
}

static uint8_t *ensure(uint8_t **buf, size_t *size, size_t need)
{
	if (*size < need) {
		free(*buf);
		*buf = malloc(need);
		*size = *buf ? need : 0;
	}
	return *buf;
}

static cf_gaze_params params_locked(const struct camflip *f)
{
	cf_gaze_params p;
	memset(&p, 0, sizeof(p));
	p.strength = f->strength;
	p.max_glance = f->max_glance;
	p.baseline = f->baseline;
	p.baseline_open = f->baseline_open;
	p.has_notes = f->has_notes && !f->notes_stale ? 1 : 0;
	p.notes_offset = f->notes_offset;
	p.notes_side = f->notes_side;
	p.notes_open = f->notes_open;
	p.head_ref = f->head_ref;
	/* Vision's gain from both calibrations, once both steps were taken with Vision's point */
	p.has_vis_calib = (f->vis_cal & 3) == 3 ? 1 : 0;
	p.vis_baseline = f->vis_baseline;
	p.vis_notes = f->vis_notes;
	p.baseline_upper = f->baseline_upper;
	p.baseline_lower = f->baseline_lower;
	return p;
}

/* Notes are only usable while they are clearly below the lens. A newer camera calibration can
 * move the lens baseline past them: they are then ignored (and the panel asks to redo step 2)
 * until step 2 is redone or the camera calibration moves back. */
static void check_notes_locked(struct camflip *f)
{
	f->notes_stale = f->has_notes && !cf_gaze_notes_valid(f->baseline, f->notes_offset);
}

/*
 * A calibration that gets no camera frames (camera off or in use elsewhere, source hidden) would
 * say "Calibrating…" forever: give up after CALIB_STALL_NS without a frame. Checked by the worker
 * (which wakes up for this while a calibration is pending) and whenever the panel is built.
 * Returns true if it gave up just now; the worker then drops any half-done calibration.
 */
static bool calib_stall_check_locked(struct camflip *f, uint64_t now)
{
	if (!f->calib_request && !f->calib_running)
		return false;
	if (now < f->calib_progress_ns || now - f->calib_progress_ns < CALIB_STALL_NS)
		return false;
	f->calib_request = 0;
	f->calib_running = CALIB_IDLE;
	f->calib_failed = CALIB_IDLE;
	f->calib_no_frames = true;
	f->calib_cancel = true;
	return true;
}

static void log_calib_stalled(bool flipped)
{
	if (flipped)
		blog(LOG_WARNING, "[cam-flip] calibration didn't finish: the camera sends upside-down (flipped) "
				  "frames, which eye contact can't track yet");
	else
		blog(LOG_WARNING,
		     "[cam-flip] calibration didn't finish: no camera frames for %d s. Make sure the camera "
		     "is on and try again",
		     (int)(CALIB_STALL_NS / 1000000000ull));
}

/* The camera sends upside-down frames, which can't be tracked: give up a requested or running
 * calibration straight away (the panel shows the flipped-frames message for it). The worker drops
 * whatever the gaze model started on an earlier frame (calib_cancel). Returns true if it cancelled. */
static bool calib_cancel_flipped(struct camflip *f)
{
	pthread_mutex_lock(&f->lock);
	const bool cancel = f->calib_request || f->calib_running;
	if (cancel) {
		f->calib_request = 0;
		f->calib_running = CALIB_IDLE;
		f->calib_failed = CALIB_IDLE;
		f->calib_no_frames = true; /* no usable frames; with frames_flipped the panel says why */
		f->calib_cancel = true;
	}
	pthread_mutex_unlock(&f->lock);
	if (cancel)
		log_calib_stalled(true);
	return cancel;
}

/* ------------------------------------------------- settings written from other threads */

/*
 * Calibration results and the hotkey change settings outside the settings panel. They are
 * written on OBS's UI thread (where the panel itself writes them), through obs_source_update,
 * so the saved settings, this filter and an open panel all agree. Holds only a weak reference,
 * so it is safe if the filter is deleted before the task runs.
 */
struct ui_task {
	obs_weak_source_t *weak;
	obs_data_t *changes; /* may be NULL: just refresh the panel */
};

static void ui_task_run(void *param)
{
	struct ui_task *t = param;
	obs_source_t *src = obs_weak_source_get_source(t->weak);
	if (src) {
		if (t->changes)
			obs_source_update(src, t->changes);
		obs_source_update_properties(src);
		obs_source_release(src);
	}
	if (t->changes)
		obs_data_release(t->changes);
	obs_weak_source_release(t->weak);
	bfree(t);
}

/* Runs on OBS's destroy-task thread, so the UI task below is posted, not run inline. */
static void ui_task_hop(void *param)
{
	obs_queue_task(OBS_TASK_UI, ui_task_run, param, false);
}

/* Takes ownership of `changes`. */
static void queue_ui(struct camflip *f, obs_data_t *changes)
{
	if (!f->context) { /* tests: no OBS source */
		if (changes)
			obs_data_release(changes);
		return;
	}
	struct ui_task *t = bzalloc(sizeof(*t));
	t->weak = obs_source_get_weak_source(f->context);
	t->changes = changes;
	/* Never run it inline on the UI thread: OBS's UI task handler and its "update_properties"
	 * handler are direct calls there, so a button callback (or a hotkey handled on the UI thread)
	 * would rebuild the settings panel while Qt is still inside the clicked widget, which frees
	 * that widget under it. Bounce through another thread so it runs from the event loop. */
	if (obs_in_task_thread(OBS_TASK_UI))
		obs_queue_task(OBS_TASK_DESTROY, ui_task_hop, t, false);
	else
		obs_queue_task(OBS_TASK_UI, ui_task_run, t, false);
}

/* ---------------------------------------------------------------- Vision (worker thread) */

static bool region_centroid(VNFaceLandmarkRegion2D *r, CGSize size, CGPoint *out)
{
	if (!r || r.pointCount == 0)
		return false;
	const CGPoint *p = [r pointsInImageOfSize:size];
	double sx = 0, sy = 0;
	for (NSUInteger i = 0; i < r.pointCount; i++) {
		sx += p[i].x;
		sy += p[i].y;
	}
	out->x = sx / (double)r.pointCount;
	out->y = sy / (double)r.pointCount;
	return true;
}

/* Landmark geometry, in Vision's image pixels (origin bottom left). */
typedef struct {
	CGPoint centroid, left, right, nose;
	double iod; /* distance between the eye centres */
	bool has_nose;
	double side, vert; /* nose relative to the eyes, in eye distances: changes as the head turns */
} face_geom;

static bool measure_geom(VNFaceLandmarks2D *lm, CGSize size, face_geom *g)
{
	memset(g, 0, sizeof(*g));
	if (!region_centroid(lm.allPoints, size, &g->centroid) || !region_centroid(lm.leftEye, size, &g->left) ||
	    !region_centroid(lm.rightEye, size, &g->right))
		return false;
	const double ax = g->right.x - g->left.x, ay = g->right.y - g->left.y;
	g->iod = sqrt(ax * ax + ay * ay);
	if (g->iod < 4.0)
		return false;
	g->has_nose = region_centroid(lm.nose, size, &g->nose);
	if (g->has_nose) {
		const double ux = ax / g->iod, uy = ay / g->iod;
		const double dx = g->nose.x - (g->left.x + g->right.x) / 2, dy = g->nose.y - (g->left.y + g->right.y) / 2;
		g->side = (dx * ux + dy * uy) / g->iod;
		g->vert = (-dx * uy + dy * ux) / g->iod;
	}
	return true;
}

static void set_last_face(struct camflip *f, VNFaceObservation *obs)
{
	if (f->last_face)
		CFBridgingRelease(f->last_face);
	f->last_face = obs ? (void *)CFBridgingRetain(obs) : NULL;
}

/* Face box for the next frame: the box from the last full detection, moved and scaled with
 * the landmarks so it follows the head between detections. */
static void plan_tracking(struct camflip *f, const face_geom *g, int w, int h)
{
	const double bw = f->ref_width_per_iod * g->iod;
	const double bh = bw * f->ref_aspect;
	const double cx = g->centroid.x + f->ref_off_x * bw;
	const double cy = g->centroid.y + f->ref_off_y * bw;
	if (cx < 0 || cy < 0 || cx > w || cy > h || bw < 8 || bh < 8) {
		set_last_face(f, nil);
		return;
	}
	const CGRect box = CGRectMake((cx - bw / 2) / w, (cy - bh / 2) / h, bw / w, bh / h);
	VNFaceObservation *next = [VNFaceObservation
		faceObservationWithRequestRevision:VNDetectFaceRectanglesRequestRevision3
				       boundingBox:box
					      roll:(f->pose_has_roll ? @(f->pose_roll) : nil)
					       yaw:(f->pose_has_yaw ? @(f->pose_yaw) : nil)
					     pitch:(f->pose_has_pitch ? @(f->pose_pitch) : nil)];
	set_last_face(f, next);
}

/*
 * One Vision pass. `track` nil = full face detection, otherwise only the landmarks inside that
 * face box (much cheaper). Returns the face with eye landmarks, or nil.
 */
static VNFaceObservation *vision_pass(struct camflip *f, CVPixelBufferRef pb, VNFaceObservation *track,
				      bool *vision_error)
{
	VNSequenceRequestHandler *handler = (__bridge VNSequenceRequestHandler *)f->handler;
	VNDetectFaceLandmarksRequest *request = (__bridge VNDetectFaceLandmarksRequest *)f->request;
	request.inputFaceObservations = track ? @[track] : nil;
	NSError *error = nil;
	BOOL ok = [handler performRequests:@[request]
			   onCVPixelBuffer:pb
			       orientation:kCGImagePropertyOrientationUp
				     error:&error];
	*vision_error = !ok;
	if (!ok) {
		pthread_mutex_lock(&f->lock);
		f->vision_failures++;
		pthread_mutex_unlock(&f->lock);
		if (!track) {
			/* Only fall back to the slower BGRA path if grey input has never worked. A
			 * one-off failure after grey frames have been accepted is not a format problem. */
			if (!f->vision_needs_bgra && !f->vision_grey_ok) {
				f->vision_needs_bgra = true;
				blog(LOG_INFO, "[cam-flip] Vision rejected grey frames, switching to BGRA input");
			} else if (!f->warned_vision) {
				f->warned_vision = true;
				const char *why = error.localizedDescription.UTF8String;
				blog(LOG_WARNING, "[cam-flip] face tracking failed: %s", why ? why : "unknown error");
			}
		}
		return nil;
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
		return nil;
	/* In tracking mode Vision always answers for the box we gave it, face or not. */
	if (track && lm.confidence < 0.4f)
		return nil;
	return best;
}

/*
 * The image Vision analyses: the frame itself, or for frames at least 2 x VISION_MIN_W wide a copy
 * shrunk by a whole factor (as large as possible while staying at least VISION_MIN_W wide) (box filter: each output pixel is the mean of a scale x scale block).
 * Pixel x of the copy covers full-resolution pixels [x * scale, (x + 1) * scale), so Vision's
 * normalised coordinates map back to full resolution with an image size of (w * scale, h * scale).
 */
static const grey_frame *vision_input(struct camflip *f, const grey_frame *g, int *scale)
{
	*scale = 1;
	const int k = g->w / VISION_MIN_W;
	if (k <= 1)
		return g;
	const int sw = g->w / k, sh = g->h / k;
	if (sh < 1 || !ensure(&f->small.data, &f->small.cap, (size_t)sw * sh))
		return g;
	const unsigned area = (unsigned)(k * k);
	for (int y = 0; y < sh; y++) {
		const uint8_t *row = g->data + (size_t)y * k * g->w;
		uint8_t *d = f->small.data + (size_t)y * sw;
		for (int x = 0; x < sw; x++) {
			const uint8_t *s = row + (size_t)x * k;
			unsigned sum = 0;
			for (int r = 0; r < k; r++, s += g->w)
				for (int c = 0; c < k; c++)
					sum += s[c];
			d[x] = (uint8_t)((sum + area / 2) / area);
		}
	}
	f->small.w = sw;
	f->small.h = sh;
	f->small.t = g->t;
	*scale = k;
	return &f->small;
}

static CVPixelBufferRef make_pixel_buffer(struct camflip *f, const grey_frame *g)
{
	CVPixelBufferRef pb = NULL;
	if (!f->vision_needs_bgra) {
		CVPixelBufferCreateWithBytes(kCFAllocatorDefault, (size_t)g->w, (size_t)g->h,
					     kCVPixelFormatType_OneComponent8, g->data, (size_t)g->w, NULL, NULL, NULL,
					     &pb);
	} else {
		/* Fallback if this macOS version's Vision rejects 8-bit grey buffers. */
		if (!ensure(&f->bgra, &f->bgra_size, (size_t)g->w * g->h * 4))
			return NULL;
		for (int y = 0; y < g->h; y++) {
			const uint8_t *s = g->data + (size_t)y * g->w;
			uint32_t *d = (uint32_t *)(f->bgra + (size_t)y * g->w * 4);
			for (int x = 0; x < g->w; x++)
				d[x] = 0xff000000u | ((uint32_t)s[x] * 0x010101u);
		}
		CVPixelBufferCreateWithBytes(kCFAllocatorDefault, (size_t)g->w, (size_t)g->h, kCVPixelFormatType_32BGRA,
					     f->bgra, (size_t)g->w * 4, NULL, NULL, NULL, &pb);
	}
	return pb;
}

/* Runs Apple Vision on a grey frame and fills `face` (full-resolution image pixels, origin top
 * left). */
static void detect_face(struct camflip *f, const grey_frame *g, cf_face *face, bool *did_full)
{
	memset(face, 0, sizeof(*face));
	const int w = g->w, h = g->h;
	face->image_width = w;
	face->image_height = h;
	*did_full = false;

	int scale = 1;
	const grey_frame *vg = vision_input(f, g, &scale);
	/* Everything below is in full-resolution pixels: Vision's normalised results are mapped with
	 * this size. It equals (w, h) unless the frame was shrunk and w or h isn't a multiple of the
	 * factor; then the last column or row (less than `scale` pixels) is simply not analysed. */
	const int vw = vg->w * scale, vh = vg->h * scale;

	CVPixelBufferRef pb = make_pixel_buffer(f, vg);
	if (!pb)
		return;

	uint64_t now = now_ns();
#ifdef CAMFLIP_TEST
	if (f->test_clock)
		now = (uint64_t)(g->t * 1e9) + 1; /* offline tools: frame time, so runs are repeatable */
#endif
	if (w != f->last_w || h != f->last_h) {
		set_last_face(f, nil);
		f->last_w = w;
		f->last_h = h;
	}
	VNFaceObservation *track = nil;
	if (f->last_face && !f->tracking_disabled && now - f->last_detect_ns < FULL_DETECT_NS)
		track = (__bridge VNFaceObservation *)f->last_face;

	bool vision_error = false;
	VNFaceObservation *best = nil;
	if (track) {
		best = vision_pass(f, pb, track, &vision_error);
		if (vision_error) {
			if (++f->track_errors >= 3 && !f->tracking_disabled) {
				f->tracking_disabled = true;
				blog(LOG_INFO, "[cam-flip] fast face tracking unavailable, using full detection on every frame");
			}
		} else {
			f->track_errors = 0;
		}
		if (!best)
			track = nil; /* lost the face: full detection on this same frame */
	}
	if (!track) {
		*did_full = true;
		best = vision_pass(f, pb, nil, &vision_error);
	}
	CVPixelBufferRelease(pb);

	const CGSize size = CGSizeMake(vw, vh);
	face_geom geom;
	if (!best || !measure_geom(best.landmarks, size, &geom)) {
		set_last_face(f, nil);
		return;
	}

	if (!track) {
		/* Fresh detection: remember how the detector's box sits around the landmarks. */
		const CGRect bb = best.boundingBox;
		const double bw = bb.size.width * vw, bh = bb.size.height * vh;
		const double bcx = (bb.origin.x + bb.size.width / 2) * vw, bcy = (bb.origin.y + bb.size.height / 2) * vh;
		f->ref_off_x = (bcx - geom.centroid.x) / bw;
		f->ref_off_y = (bcy - geom.centroid.y) / bw;
		f->ref_aspect = bh / bw;
		f->ref_width_per_iod = bw / geom.iod;
		f->ref_side = geom.side;
		f->ref_vert = geom.vert;
		f->ref_has_nose = geom.has_nose;
		f->pose_has_roll = best.roll != nil;
		f->pose_roll = best.roll.doubleValue;
		f->pose_has_yaw = best.yaw != nil;
		f->pose_yaw = best.yaw.doubleValue;
		f->pose_has_pitch = best.pitch != nil;
		f->pose_pitch = best.pitch.doubleValue;
		f->last_detect_ns = now;
	}
	plan_tracking(f, &geom, vw, vh);
	/* Tracked passes fit the landmarks with the head pose of the last full detection. If the head
	 * has turned or tilted noticeably since, detect again on the next frame so the landmarks (which
	 * the head-turn fade is measured from, below) stay good. */
	if (track && f->ref_has_nose && geom.has_nose &&
	    (fabs(geom.side - f->ref_side) > 0.05 || fabs(geom.vert - f->ref_vert) > 0.08))
		f->last_detect_ns = 0;

	VNFaceLandmarks2D *lm = best.landmarks;
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
			outs[e][i].y = (float)(vh - pts[i].y); /* Vision's origin is bottom left */
		}
		*counts[e] = n;
	}
	/* Head pose, on every frame (tracked or full), from the eye outlines and the nose's outline and
	 * crest (pose.c). Vision's own yaw only reads 0 or 45 degrees and its pitch 0, and only on full
	 * detections: it is kept below only as a fallback for frames without a nose. */
	{
		cf_pt nose[CF_HEAD_MAX_POINTS];
		int nn = 0;
		VNFaceLandmarkRegion2D *nose_regions[2] = {lm.nose, lm.noseCrest};
		for (int r = 0; r < 2; r++) {
			VNFaceLandmarkRegion2D *region = nose_regions[r];
			if (!region || region.pointCount < 1)
				continue;
			const CGPoint *pts = [region pointsInImageOfSize:size];
			if (!pts)
				continue;
			for (NSUInteger i = 0; i < region.pointCount && nn < CF_HEAD_MAX_POINTS; i++) {
				nose[nn].x = (float)pts[i].x;
				nose[nn].y = (float)(vh - pts[i].y);
				nn++;
			}
		}
		face->has_head_geom = cf_head_measure(face->left_eye, face->left_count, face->right_eye,
						      face->right_count, nose, nn, &face->head);
	}
	if (lm.leftPupil.pointCount < 1 || lm.rightPupil.pointCount < 1)
		return;
	const CGPoint *lp = [lm.leftPupil pointsInImageOfSize:size];
	const CGPoint *rp = [lm.rightPupil pointsInImageOfSize:size];
	face->left_pupil = (cf_pt){(float)lp[0].x, (float)(vh - lp[0].y)};
	face->right_pupil = (cf_pt){(float)rp[0].x, (float)(vh - rp[0].y)};
	face->left_pupil_vision = face->left_pupil;
	face->right_pupil_vision = face->right_pupil;
	face->has_vision_pupils = 1;
#ifdef CAMFLIP_TEST
	f->dbg_raw_pupil[0] = face->left_pupil;
	f->dbg_raw_pupil[1] = face->right_pupil;
	f->dbg_conf = lm.confidence;
#endif

	/* Vision's pupil point is coarse; move it onto the real iris centre, using the full-resolution
	 * frame even when Vision saw a shrunk copy (all coordinates here are full-resolution). */
	face->left_pupil =
		cf_refine_pupil(g->data, w, w, h, face->left_eye, face->left_count, face->left_pupil);
	face->right_pupil =
		cf_refine_pupil(g->data, w, w, h, face->right_eye, face->right_count, face->right_pupil);

	face->face_width = (float)(best.boundingBox.size.width * vw);
	if (best.yaw != nil) {
		face->has_pose = 1;
		face->yaw_deg = (float)(best.yaw.doubleValue * 180.0 / M_PI);
		face->pitch_deg = best.pitch != nil ? (float)(best.pitch.doubleValue * 180.0 / M_PI) : 0.0f;
	}
	face->found = 1;
}

/* Short code for "what the status panel shows", so it is refreshed only on real changes.
 * Only discrete states: each refresh rebuilds the whole panel (and drops a slider the user is
 * dragging), so it must not follow values that change while you look around. */
static unsigned int status_sig_locked(const struct camflip *f)
{
	return (unsigned)f->calib_running | ((unsigned)f->calib_failed << 2) | ((unsigned)f->st_face << 4) |
	       ((unsigned)f->lens_done << 5) | ((unsigned)f->has_notes << 6) | ((unsigned)f->eye_contact << 7) |
	       ((unsigned)f->calib_notes_bad << 8) | ((unsigned)f->calib_no_frames << 9) |
	       ((unsigned)f->notes_stale << 10) | ((unsigned)(f->lens_done && !f->head_ref.ok) << 11);
}

/*
 * Takes a finished calibration. Returns the settings to save, or NULL if it was rejected: a notes
 * calibration that isn't clearly below the camera (looked elsewhere, or notes above the lens)
 * would be ignored by the gaze logic, so it is refused instead of claiming it worked.
 */
static obs_data_t *apply_calibration_locked(struct camflip *f, const cf_calib_result *res, bool *notes_bad)
{
	*notes_bad = res->target == CF_CALIB_NOTES && !cf_gaze_notes_valid(f->baseline, res->offset);
	if (*notes_bad) {
		f->calib_notes_bad = true;
		return NULL;
	}
	obs_data_t *changes = obs_data_create();
	if (res->target == CF_CALIB_NOTES) {
		f->notes_offset = res->offset;
		f->notes_side = res->side;
		f->notes_open = res->openness;
		f->has_notes = true;
		obs_data_set_double(changes, S_NOTES_OFFSET, res->offset);
		obs_data_set_double(changes, S_NOTES_SIDE, res->side);
		obs_data_set_double(changes, S_NOTES_OPEN, res->openness);
		obs_data_set_bool(changes, S_HAS_NOTES, true);
		f->calib_done_msg = "Notes calibrated.";
	} else {
		f->baseline = res->offset;
		f->baseline_open = res->openness;
		f->lens_done = true;
		obs_data_set_double(changes, S_BASELINE, res->offset);
		obs_data_set_double(changes, S_BASELINE_OPEN, res->openness);
		/* The head as it was while looking into the camera: turns are measured from here. Not learnt
		 * (no nose seen) leaves ok 0, a typical face, and the panel asks to redo the step. */
		f->head_ref = res->head;
		obs_data_set_bool(changes, S_HEAD_REF, res->head.ok != 0);
		obs_data_set_double(changes, S_HEAD_SIDE, res->head.side);
		obs_data_set_double(changes, S_HEAD_VERT, res->head.vert);
		obs_data_set_double(changes, S_HEAD_EYES, res->head.eyes);
		obs_data_set_double(changes, S_HEAD_TILT, res->head.tilt);
		/* The lids as they are while looking into the camera: what reading narrows is restored from here. */
		f->baseline_upper = res->upper;
		f->baseline_lower = res->lower;
		obs_data_set_double(changes, S_BASELINE_UPPER, res->upper);
		obs_data_set_double(changes, S_BASELINE_LOWER, res->lower);
		f->calib_done_msg = "Camera calibrated.";
	}
	/* Vision's pupil point at this step, and a hint if the result may be off (the gaze or lids moved while
	 * calibrating, or notes that read barely below the camera). */
	const int vis_bit = res->target == CF_CALIB_NOTES ? 2 : 1;
	f->vis_cal = res->has_vis ? (f->vis_cal | vis_bit) : (f->vis_cal & ~vis_bit);
	if (res->target == CF_CALIB_NOTES)
		f->vis_notes = res->vis_offset;
	else
		f->vis_baseline = res->vis_offset;
	obs_data_set_int(changes, S_VIS_CAL, f->vis_cal);
	obs_data_set_double(changes, res->target == CF_CALIB_NOTES ? S_VIS_NOTES : S_VIS_BASELINE, res->vis_offset);
	const int hints = cf_gaze_calib_hints(res, f->baseline);
	f->calib_hint = hints != 0;
	if (hints & CF_CALIB_HINT_UNSTEADY)
		f->calib_done_msg = res->target == CF_CALIB_NOTES
					    ? "Notes calibrated, but your eyes moved while calibrating: if the effect looks off, "
					      "redo step 2, keeping your gaze on the middle of your notes."
					    : "Camera calibrated, but your eyes moved while calibrating: if the effect looks off, "
					      "redo step 1, looking straight into the lens.";
	else if (hints & CF_CALIB_HINT_SMALL_GLANCE)
		f->calib_done_msg = "Notes calibrated, but they read barely below the camera. If your notes are further "
				    "down, redo step 1 looking straight into the lens (not at the screen), then step 2.";

	/* A new camera calibration can leave the saved notes above (or level with) the lens. */
	check_notes_locked(f);
	f->needs_recalib = false;
	obs_data_set_int(changes, S_CALIB_VERSION, CALIB_VERSION);
	return changes;
}

static void process_frame(struct camflip *f, const grey_frame *g)
{
	pthread_mutex_lock(&f->lock);
	const cf_gaze_params params = params_locked(f);
	const bool cancel = f->calib_cancel;
	f->calib_cancel = false;
	const int start = f->calib_request;
	f->calib_request = 0;
	if (start) {
		f->calib_running = start;
		f->calib_failed = CALIB_IDLE;
		f->calib_notes_bad = false;
		f->calib_no_frames = false;
		f->calib_done_msg = NULL;
		f->calib_hint = false;
	}
	if (f->calib_running)
		f->calib_progress_ns = now_ns(); /* frames are arriving: not stalled */
	pthread_mutex_unlock(&f->lock);

	if (start || cancel) {
		pthread_mutex_lock(&f->gaze_lock);
		if (start)
			cf_gaze_start_calibration(&f->gaze, start == CALIB_NOTES ? CF_CALIB_NOTES : CF_CALIB_LENS);
		else
			cf_gaze_init(&f->gaze); /* drop a calibration that gave up for lack of frames */
		pthread_mutex_unlock(&f->gaze_lock);
	}

	const uint64_t t0 = now_ns();
	cf_face face;
	bool did_full = false;
	detect_face(f, g, &face, &did_full);

	cf_eye eyes[2];
	cf_calib_result res;
	memset(&res, 0, sizeof(res));
	cf_gaze_status st;
	memset(&st, 0, sizeof(st));
	pthread_mutex_lock(&f->gaze_lock);
	cf_gaze_update(&f->gaze, &face, &params, g->t, eyes);
	const int calibrated = cf_gaze_take_calibration(&f->gaze, &res);
	cf_gaze_get_status(&f->gaze, &st);
	pthread_mutex_unlock(&f->gaze_lock);
	const double ms = (double)(now_ns() - t0) / 1e6;

	obs_data_t *changes = NULL;
	pthread_mutex_lock(&f->lock);
	f->vision_runs++;
	if (did_full)
		f->full_detections++;
	else
		f->tracked_runs++;
	f->win_vision_ms += ms;
	f->win_vision_n++;
	if (did_full)
		f->win_full_n++;
	f->ema_vision_ms = f->ema_vision_ms > 0 ? f->ema_vision_ms * 0.9 + ms * 0.1 : ms;
	const double dt = g->t - f->last_frame_t;
	if (f->last_frame_t > 0 && dt > 0 && dt < 1.0)
		f->ema_fps = f->ema_fps > 0 ? f->ema_fps * 0.9 + 0.1 / dt : 1.0 / dt;
	f->last_frame_t = g->t;
	f->st_face = st.face_found != 0;
	f->st_factor = st.factor;
#ifdef CAMFLIP_TEST
	f->seen_face = face;
#endif

	const int step = f->calib_running;
	const bool was_stale = f->notes_stale;
	const float notes_glance = f->baseline - res.offset;
	bool notes_bad = false;
	/* step is CALIB_IDLE only if the calibration gave up (no frames) while this frame was analysed:
	 * its result is from long after the click, so it is dropped. */
	const bool usable = calibrated != 0 && step != CALIB_IDLE;
	if (usable)
		f->calib_running = CALIB_IDLE;
	if (usable && calibrated > 0)
		changes = apply_calibration_locked(f, &res, &notes_bad);
	else if (usable && calibrated < 0)
		f->calib_failed = step;
	const bool became_stale = f->notes_stale && !was_stale;

	/* Refresh an open settings panel, at most once a second and only when something changed. */
	bool refresh = false;
	const unsigned int sig = status_sig_locked(f);
	const uint64_t now = now_ns();
	if (changes || notes_bad || (sig != f->last_sig && now - f->last_refresh >= REFRESH_NS)) {
		f->last_sig = sig;
		f->last_refresh = now;
		refresh = true;
	}
	pthread_mutex_unlock(&f->lock);

	if (notes_bad) {
		blog(LOG_WARNING, "[cam-flip] notes calibration rejected: gaze %.3f eye widths below the camera (needs "
				  "more than %.3f). Look at your notes below the camera and try again",
		     notes_glance, CF_NOTES_MIN_GLANCE);
	} else if (usable && calibrated > 0) {
		if (res.target == CF_CALIB_NOTES)
			blog(LOG_INFO, "[cam-flip] notes calibrated (offset %.3f, side %.3f eye widths)", res.offset,
			     res.side);
		else
			blog(LOG_INFO,
			     "[cam-flip] camera calibrated (baseline %.3f eye widths, openness %.3f; head %s: side %.3f "
			     "vert %.3f eyes %.3f tilt %.3f)",
			     res.offset, res.openness, res.head.ok ? "learnt" : "not learnt (no nose seen)", res.head.side,
			     res.head.vert, res.head.eyes, res.head.tilt);
		/* How steady it was, and the hint the panel shows (see cf_gaze_calib_hints) */
		blog(LOG_INFO, "[cam-flip] calibration spread %.4f, drift %.4f eye widths, lids %.4f (hints %d)",
		     res.spread, res.drift, res.open_spread, cf_gaze_calib_hints(&res, notes_glance + res.offset));
		if (became_stale)
			blog(LOG_WARNING, "[cam-flip] the notes calibration is no longer below the new camera "
					  "calibration and is not used: please redo step 2 (notes)");
	} else if (usable && calibrated < 0) {
		blog(LOG_WARNING, "[cam-flip] calibration gave up: no steady, front-facing face seen. "
				  "Light your face, hold still and click the calibration button again");
	}
	if (refresh)
		queue_ui(f, changes); /* persists calibration immediately, then refreshes the panel */
	else if (changes)
		obs_data_release(changes);
}

/* Waits on `wake` for at most a second. Called with `lock` held. */
static void wait_a_second_locked(struct camflip *f)
{
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts); /* pthread_cond_timedwait's clock on macOS */
	ts.tv_sec += 1;
	pthread_cond_timedwait(&f->wake, &f->lock, &ts);
}

static void *worker_main(void *data)
{
	struct camflip *f = data;
	pthread_setname_np("cam-flip tracking");
	pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

	pthread_mutex_lock(&f->lock);
	for (;;) {
		while (!f->stop && (f->pending < 0
#ifdef CAMFLIP_TEST
				    || f->test_pause
#endif
				    )) {
			if (!f->calib_request && !f->calib_running) {
				pthread_cond_wait(&f->wake, &f->lock);
				continue;
			}
			/* A calibration is waiting for frames: check once a second that they still come. */
			if (calib_stall_check_locked(f, now_ns())) {
				f->calib_cancel = false;
				const bool flipped = f->frames_flipped;
				pthread_mutex_unlock(&f->lock);
				pthread_mutex_lock(&f->gaze_lock);
				cf_gaze_init(&f->gaze); /* drop the half-done calibration */
				pthread_mutex_unlock(&f->gaze_lock);
				log_calib_stalled(flipped);
				queue_ui(f, NULL); /* show it in an open panel */
				pthread_mutex_lock(&f->lock);
				continue;
			}
			wait_a_second_locked(f);
		}
		if (f->stop)
			break;
		const int idx = f->pending;
		f->pending = -1;
		f->busy = idx;
		pthread_mutex_unlock(&f->lock);

		@autoreleasepool {
			process_frame(f, &f->frames[idx]);
		}

		pthread_mutex_lock(&f->lock);
		f->busy = -1;
	}
	pthread_mutex_unlock(&f->lock);
	return NULL;
}

/* ---------------------------------------------------------------- Video thread helpers */

/* Copies the frame's brightness into the free buffer and hands it to the worker. Any older
 * frame still waiting is dropped: the worker always analyses the newest one. */
static void submit_frame(struct camflip *f, const cf_channel *luma, double t)
{
	pthread_mutex_lock(&f->lock);
	const int idx = f->busy == 0 ? 1 : 0;
	if (f->pending == idx)
		f->pending = -1;
	pthread_mutex_unlock(&f->lock);

	grey_frame *g = &f->frames[idx];
	const int w = luma->width, h = luma->height;
	if (!ensure(&g->data, &g->cap, (size_t)w * h))
		return;
	if (luma->step == 1) {
		for (int y = 0; y < h; y++)
			memcpy(g->data + (size_t)y * w, luma->base + (size_t)y * luma->linesize, (size_t)w);
	} else {
		cf_extract_luma(luma, g->data);
	}
	g->w = w;
	g->h = h;
	g->t = t;

	pthread_mutex_lock(&f->lock);
	f->pending = idx;
	pthread_cond_signal(&f->wake);
	pthread_mutex_unlock(&f->lock);
}

static void plot(const cf_channel *c, float x, float y, uint8_t v)
{
	const int ix = (int)lrintf(x), iy = (int)lrintf(y);
	if (ix < 0 || iy < 0 || ix >= c->width || iy >= c->height)
		return;
	c->base[(size_t)iy * c->linesize + (size_t)ix * c->step] = v;
}

static void draw_line(const cf_channel *c, float x0, float y0, float x1, float y1, uint8_t v)
{
	const int n = (int)fmaxf(fabsf(x1 - x0), fabsf(y1 - y0)) + 1;
	for (int i = 0; i <= n; i++) {
		const float a = (float)i / (float)n;
		plot(c, x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, v);
	}
}

/*
 * Thin outlines in the brightness channel (before mirroring, so they mirror with the video), all
 * from the same prediction the warp uses for this frame, so they sit exactly where the warp
 * works: white ellipse = the area that may change, black = the eye opening the iris is moved
 * inside. Nothing is drawn while there is nothing to correct (for example looking into the lens).
 */
static void draw_overlay(const cf_channel *luma, const cf_eye *eyes, int n)
{
	for (int e = 0; e < n; e++) {
		const cf_eye *eye = &eyes[e];
		const int steps = (int)fmaxf(24.0f, 6.3f * fmaxf(eye->rx, eye->ry));
		for (int i = 0; i < steps; i++) {
			const float a = 6.2831853f * (float)i / (float)steps;
			plot(luma, eye->cx + eye->rx * cosf(a), eye->cy + eye->ry * sinf(a), 235);
		}
		const int ac = eye->aperture_count < CF_MAX_APERTURE ? eye->aperture_count : CF_MAX_APERTURE;
		for (int i = 0; i < ac; i++) {
			const int j = (i + 1) % ac;
			draw_line(luma, eye->aperture[i][0], eye->aperture[i][1], eye->aperture[j][0],
				  eye->aperture[j][1], 16);
		}
	}
}

/* ---------------------------------------------------------------- Filter */

static const char *camflip_get_name(void *unused)
{
	(void)unused;
	return FILTER_NAME;
}

static void camflip_update(void *data, obs_data_t *settings)
{
	struct camflip *f = data;
	pthread_mutex_lock(&f->lock);
	f->mirror = obs_data_get_bool(settings, S_MIRROR);
	f->eye_contact = obs_data_get_bool(settings, S_EYE_CONTACT);
	f->show_eyes = obs_data_get_bool(settings, S_SHOW_EYES);
	f->strength = (float)obs_data_get_double(settings, S_STRENGTH) / 100.0f;
	f->max_glance = (float)obs_data_get_double(settings, S_MAX_GLANCE) / 100.0f; /* eye widths */
	f->baseline = (float)obs_data_get_double(settings, S_BASELINE);
	f->baseline_open = (float)obs_data_get_double(settings, S_BASELINE_OPEN);
	f->notes_offset = (float)obs_data_get_double(settings, S_NOTES_OFFSET);
	f->notes_side = (float)obs_data_get_double(settings, S_NOTES_SIDE);
	f->notes_open = (float)obs_data_get_double(settings, S_NOTES_OPEN);
	f->has_notes = obs_data_get_bool(settings, S_HAS_NOTES);
	f->vis_cal = (int)obs_data_get_int(settings, S_VIS_CAL);
	f->vis_baseline = (float)obs_data_get_double(settings, S_VIS_BASELINE);
	f->vis_notes = (float)obs_data_get_double(settings, S_VIS_NOTES);
	f->lens_done = f->baseline_open > 0.0f;
	memset(&f->head_ref, 0, sizeof(f->head_ref));
	f->head_ref.ok = f->lens_done && obs_data_get_bool(settings, S_HEAD_REF) ? 1 : 0;
	f->head_ref.side = (float)obs_data_get_double(settings, S_HEAD_SIDE);
	f->head_ref.vert = (float)obs_data_get_double(settings, S_HEAD_VERT);
	f->head_ref.eyes = (float)obs_data_get_double(settings, S_HEAD_EYES);
	f->head_ref.tilt = (float)obs_data_get_double(settings, S_HEAD_TILT);
	f->baseline_upper = (float)obs_data_get_double(settings, S_BASELINE_UPPER);
	f->baseline_lower = (float)obs_data_get_double(settings, S_BASELINE_LOWER);
	check_notes_locked(f);
	if (f->has_notes || f->lens_done)
		f->needs_recalib = false;
	pthread_mutex_unlock(&f->lock);
}

/* Calibrations saved by older versions use other units: drop them (once) and ask again. */
static bool migrate_calibration(obs_data_t *settings)
{
	/* No default for this key (OBS applies defaults before create), so a missing key really
	 * means "saved by an older version or never calibrated". */
	if (obs_data_has_user_value(settings, S_CALIB_VERSION) &&
	    obs_data_get_int(settings, S_CALIB_VERSION) == CALIB_VERSION)
		return false;
	const bool had_old = obs_data_has_user_value(settings, S_BASELINE);
	obs_data_unset_user_value(settings, S_BASELINE);
	obs_data_unset_user_value(settings, S_BASELINE_OPEN);
	obs_data_unset_user_value(settings, S_NOTES_OFFSET);
	obs_data_unset_user_value(settings, S_NOTES_SIDE);
	obs_data_unset_user_value(settings, S_NOTES_OPEN);
	obs_data_unset_user_value(settings, S_HAS_NOTES);
	obs_data_unset_user_value(settings, S_HEAD_REF);
	obs_data_unset_user_value(settings, S_HEAD_SIDE);
	obs_data_unset_user_value(settings, S_HEAD_VERT);
	obs_data_unset_user_value(settings, S_HEAD_EYES);
	obs_data_unset_user_value(settings, S_HEAD_TILT);
	obs_data_unset_user_value(settings, S_VIS_CAL);
	obs_data_unset_user_value(settings, S_VIS_BASELINE);
	obs_data_unset_user_value(settings, S_VIS_NOTES);
	obs_data_unset_user_value(settings, S_BASELINE_UPPER);
	obs_data_unset_user_value(settings, S_BASELINE_LOWER);
	if (obs_data_has_user_value(settings, S_MAX_GLANCE))
		obs_data_unset_user_value(settings, S_MAX_GLANCE); /* also in different units now */
	obs_data_set_int(settings, S_CALIB_VERSION, CALIB_VERSION);
	if (had_old)
		blog(LOG_INFO, "[cam-flip] calibration from an older version was reset; please calibrate again");
	return had_old;
}

static void camflip_renamed(void *data, calldata_t *cd);

static void *camflip_create(obs_data_t *settings, obs_source_t *source)
{
	struct camflip *f = bzalloc(sizeof(*f));
	f->context = source;
	f->busy = -1;
	f->pending = -1;
	f->hotkey = OBS_INVALID_HOTKEY_ID;
	pthread_mutex_init(&f->lock, NULL);
	pthread_mutex_init(&f->gaze_lock, NULL);
	pthread_cond_init(&f->wake, NULL);
	cf_gaze_init(&f->gaze);

	@autoreleasepool {
		VNDetectFaceLandmarksRequest *request = [[VNDetectFaceLandmarksRequest alloc] init];
		request.constellation = VNRequestFaceLandmarksConstellation76Points;
		f->request = (void *)CFBridgingRetain(request);
		f->handler = (void *)CFBridgingRetain([[VNSequenceRequestHandler alloc] init]);
	}

	const bool reset = migrate_calibration(settings);
	camflip_update(f, settings);
	f->needs_recalib = reset;
	if (source)
		signal_handler_connect(obs_source_get_signal_handler(source), "rename", camflip_renamed, f);

	if (pthread_create(&f->worker, NULL, worker_main, f) == 0)
		f->worker_started = true;
	else
		blog(LOG_ERROR, "[cam-flip] could not start the tracking thread; eye contact is off");
	return f;
}

static void hotkey_unregister(struct camflip *f)
{
	/* Never call into the hotkey system while holding `lock`: the hotkey thread holds its own
	 * lock while calling camflip_hotkey, which takes `lock`. */
	pthread_mutex_lock(&f->lock);
	const obs_hotkey_id id = f->hotkey;
	f->hotkey = OBS_INVALID_HOTKEY_ID;
	pthread_mutex_unlock(&f->lock);
	if (id != OBS_INVALID_HOTKEY_ID)
		obs_hotkey_unregister(id); /* waits for a running callback to finish */
}

static void camflip_destroy(void *data)
{
	struct camflip *f = data;
	if (f->context)
		signal_handler_disconnect(obs_source_get_signal_handler(f->context), "rename", camflip_renamed, f);
	hotkey_unregister(f);

	pthread_mutex_lock(&f->lock);
	f->stop = true;
	pthread_cond_broadcast(&f->wake);
	pthread_mutex_unlock(&f->lock);
	if (f->worker_started)
		pthread_join(f->worker, NULL);

	@autoreleasepool {
		set_last_face(f, nil);
		if (f->request)
			CFBridgingRelease(f->request);
		if (f->handler)
			CFBridgingRelease(f->handler);
	}
	for (int i = 0; i < 2; i++)
		free(f->frames[i].data);
	free(f->small.data);
	free(f->bgra);
	pthread_cond_destroy(&f->wake);
	pthread_mutex_destroy(&f->gaze_lock);
	pthread_mutex_destroy(&f->lock);
	bfree(f);
}

static void camflip_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, S_MIRROR, true);
	obs_data_set_default_bool(settings, S_EYE_CONTACT, true);
	obs_data_set_default_bool(settings, S_SHOW_EYES, false);
	/* Full strength: with the notes calibrated, anything less leaves part of every glance visible.
	 * The slider stays so it can be turned down. */
	obs_data_set_default_double(settings, S_STRENGTH, 100.0);
	obs_data_set_default_double(settings, S_MAX_GLANCE, 15.0);
	obs_data_set_default_double(settings, S_BASELINE, DEFAULT_BASELINE);
	obs_data_set_default_double(settings, S_BASELINE_OPEN, 0.0);
	obs_data_set_default_double(settings, S_NOTES_OFFSET, 0.0);
	obs_data_set_default_double(settings, S_NOTES_SIDE, 0.0);
	obs_data_set_default_double(settings, S_NOTES_OPEN, 0.0);
	obs_data_set_default_bool(settings, S_HAS_NOTES, false);
	obs_data_set_default_bool(settings, S_HEAD_REF, false);
	obs_data_set_default_double(settings, S_HEAD_SIDE, 0.0);
	obs_data_set_default_double(settings, S_HEAD_VERT, 0.0);
	obs_data_set_default_double(settings, S_HEAD_EYES, 0.0);
	obs_data_set_default_double(settings, S_HEAD_TILT, 0.0);
	obs_data_set_default_int(settings, S_VIS_CAL, 0);
	obs_data_set_default_double(settings, S_VIS_BASELINE, 0.0);
	obs_data_set_default_double(settings, S_VIS_NOTES, 0.0);
	obs_data_set_default_double(settings, S_BASELINE_UPPER, 0.0);
	obs_data_set_default_double(settings, S_BASELINE_LOWER, 0.0);
	/* S_CALIB_VERSION deliberately has no default, see migrate_calibration(). */
}

/* ---------------------------------------------------------------- Hotkey */

static void camflip_hotkey(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed)
{
	(void)id;
	(void)hotkey;
	struct camflip *f = data;
	if (!pressed)
		return;
	pthread_mutex_lock(&f->lock);
	f->eye_contact = !f->eye_contact;
	const bool on = f->eye_contact;
	pthread_mutex_unlock(&f->lock);
	blog(LOG_INFO, "[cam-flip] eye contact %s (hotkey)", on ? "on" : "off");

	obs_data_t *changes = obs_data_create();
	obs_data_set_bool(changes, S_EYE_CONTACT, on);
	queue_ui(f, changes);
}

/*
 * Hotkey names. OBS saves key bindings under the internal name, so it must be unique per filter on
 * a camera (two Cam Flip filters on one camera would otherwise share one binding) and stable across
 * restarts. It comes from the filter's name, which OBS keeps unique among a source's filters. A
 * filter with the default name keeps the plain name earlier versions used, so its existing binding
 * still works. Without a name, a counter keeps it unique (for this session only).
 */
static void hotkey_names(const char *filter_name, char *name, size_t name_size, char *desc, size_t desc_size)
{
	if (filter_name && *filter_name && strcmp(filter_name, FILTER_NAME) == 0) {
		snprintf(name, name_size, "%s", HOTKEY_NAME);
		snprintf(desc, desc_size, "%s", HOTKEY_DESC);
	} else if (filter_name && *filter_name) {
		snprintf(name, name_size, "%s.%s", HOTKEY_NAME, filter_name);
		snprintf(desc, desc_size, "%s (%s)", HOTKEY_DESC, filter_name);
	} else {
		static long counter;
		const long n = __atomic_add_fetch(&counter, 1, __ATOMIC_RELAXED);
		snprintf(name, name_size, "%s.%ld", HOTKEY_NAME, n);
		snprintf(desc, desc_size, "%s (%ld)", HOTKEY_DESC, n);
	}
}

/* `keys`: bindings to carry over (from obs_hotkey_save), or NULL to use the saved ones. */
static void hotkey_register(struct camflip *f, obs_source_t *parent, const char *filter_name, obs_data_array_t *keys)
{
	char name[512], desc[512];
	hotkey_names(filter_name, name, sizeof(name), desc, sizeof(desc));
	const obs_hotkey_id id = obs_hotkey_register_source(parent, name, desc, camflip_hotkey, f);
	if (keys && id != OBS_INVALID_HOTKEY_ID)
		obs_hotkey_load(id, keys);
	pthread_mutex_lock(&f->lock);
	f->hotkey = id;
	pthread_mutex_unlock(&f->lock);
}

/*
 * The hotkey belongs to the camera source this filter sits on, so it is listed under the
 * camera's name in Settings › Hotkeys. (OBS loads saved filters as private sources, which
 * can't have hotkeys of their own.) OBS saves and restores the key binding together with the
 * camera source's own hotkeys, so no save/load code is needed here.
 */
static void camflip_filter_add(void *data, obs_source_t *parent)
{
	struct camflip *f = data;
	hotkey_unregister(f);
	if (!parent)
		return;
	hotkey_register(f, parent, f->context ? obs_source_get_name(f->context) : NULL, NULL);
}

/* Renaming the filter renames its hotkey (the name is derived from it) and keeps its keys. */
static void camflip_renamed(void *data, calldata_t *cd)
{
	struct camflip *f = data;
	pthread_mutex_lock(&f->lock);
	const obs_hotkey_id old = f->hotkey;
	pthread_mutex_unlock(&f->lock);
	obs_source_t *parent = f->context ? obs_filter_get_parent(f->context) : NULL;
	if (old == OBS_INVALID_HOTKEY_ID || !parent)
		return;
	obs_data_array_t *keys = obs_hotkey_save(old);
	hotkey_unregister(f);
	hotkey_register(f, parent, calldata_string(cd, "new_name"), keys);
	obs_data_array_release(keys);
}

static void camflip_filter_remove(void *data, obs_source_t *parent)
{
	(void)parent;
	hotkey_unregister(data);
}

/* ---------------------------------------------------------------- Settings panel */

static bool request_calibration(struct camflip *f, int step)
{
	if (!f)
		return false;
	pthread_mutex_lock(&f->lock);
	f->calib_request = step;
	f->calib_running = step; /* shown straight away; the worker starts it on the next frame */
	f->calib_failed = CALIB_IDLE;
	f->calib_notes_bad = false;
	f->calib_no_frames = false;
	f->calib_progress_ns = now_ns();
	f->calib_done_msg = NULL;
	f->calib_hint = false;
	pthread_cond_signal(&f->wake); /* the worker starts watching for missing frames */
	pthread_mutex_unlock(&f->lock);
	blog(LOG_INFO, "[cam-flip] calibrating: %s for a second or two",
	     step == CALIB_NOTES ? "look at the middle of your notes" : "look into the camera");
	/* Returning true only re-lays out the panel's current properties; rebuild them so the status
	 * shows "Calibrating…" straight away. */
	queue_ui(f, NULL);
	return false;
}

static bool camflip_calib_lens_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	(void)props;
	(void)property;
	return request_calibration(data, CALIB_LENS);
}

static bool camflip_calib_notes_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	(void)props;
	(void)property;
	return request_calibration(data, CALIB_NOTES);
}

static bool camflip_forget_notes_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	(void)props;
	(void)property;
	struct camflip *f = data;
	if (!f)
		return false;
	pthread_mutex_lock(&f->lock);
	f->has_notes = false;
	f->notes_offset = 0.0f;
	f->notes_side = 0.0f;
	f->notes_open = 0.0f;
	f->vis_cal &= ~2; /* (Vision's point at the notes goes with them) */
	f->notes_stale = false;
	f->calib_notes_bad = false;
	f->calib_done_msg = NULL;
	f->calib_hint = false;
	pthread_mutex_unlock(&f->lock);
	obs_data_t *changes = obs_data_create();
	obs_data_set_bool(changes, S_HAS_NOTES, false);
	obs_data_set_double(changes, S_NOTES_OFFSET, 0.0);
	obs_data_set_double(changes, S_NOTES_SIDE, 0.0);
	obs_data_set_double(changes, S_NOTES_OPEN, 0.0);
	obs_data_set_int(changes, S_VIS_CAL, f->vis_cal);
	queue_ui(f, changes); /* saves it and rebuilds the panel */
	blog(LOG_INFO, "[cam-flip] notes calibration cleared");
	return false;
}

static void status_text(struct camflip *f, char *out, size_t size, bool *warn)
{
	pthread_mutex_lock(&f->lock);
	const bool stalled = calib_stall_check_locked(f, now_ns());
	const bool eye_contact = f->eye_contact, face = f->st_face, lens = f->lens_done;
	const bool notes = f->has_notes && !f->notes_stale, notes_stale = f->notes_stale;
	const bool head = f->head_ref.ok != 0;
	const bool lids = f->baseline_upper > 0 && f->baseline_lower > 0;
	const bool vis_notes = (f->vis_cal & 2) != 0;
	const bool recalib = f->needs_recalib;
	const int running = f->calib_running, failed = f->calib_failed;
	const bool notes_bad = f->calib_notes_bad, no_frames = f->calib_no_frames, flipped = f->frames_flipped;
	const char *done = f->calib_done_msg;
	const bool hint = f->calib_hint;
	const double vms = f->ema_vision_ms, fps = f->ema_fps, wms = f->ema_warp_ms;
	pthread_mutex_unlock(&f->lock);
	if (stalled)
		log_calib_stalled(flipped);

	char calib[480], track[160], speed[200];
	*warn = false;
	if (running == CALIB_LENS)
		snprintf(calib, sizeof(calib), "Calibrating: keep looking into the camera…");
	else if (running == CALIB_NOTES)
		snprintf(calib, sizeof(calib), "Calibrating: keep looking at the middle of your notes…");
	else if (no_frames && flipped) {
		*warn = true;
		snprintf(calib, sizeof(calib),
			 "Calibration didn't finish: this camera sends upside-down (flipped) frames, which eye "
			 "contact can't track yet.");
	} else if (no_frames) {
		*warn = true;
		snprintf(calib, sizeof(calib),
			 "Calibration didn't finish: no camera frames. Make sure the camera is on and try again.");
	} else if (failed) {
		*warn = true;
		snprintf(calib, sizeof(calib),
			 "Calibration gave up: no steady, front-facing face seen. Light your face and try step %d again.",
			 failed == CALIB_NOTES ? 2 : 1);
	} else if (notes_bad) {
		*warn = true;
		snprintf(calib, sizeof(calib),
			 "Notes calibration failed: look at your notes below the camera and try again.");
	} else if (recalib) {
		*warn = true;
		snprintf(calib, sizeof(calib), "Please calibrate again (this version measures your eyes differently).");
	} else if (notes_stale) {
		*warn = true;
		snprintf(calib, sizeof(calib), "%s%sNotes calibration no longer matches: please redo step 2.",
			 done ? done : "", done ? " " : "");
	} else {
		/* A step done by an older version (or step 1 without the nose in view): head tilt and lids
		 * aren't measured for this face, or the notes step lacks Vision's point, so the correction is
		 * weaker or less sure than it could be. */
		const bool redo1 = lens && (!head || !lids), redo2 = notes && !vis_notes;
		const char *redo = redo1 && notes ? " Redo steps 1 and 2 once so your head tilt and eyelids are measured."
				   : redo1	    ? " Redo step 1 so your head tilt and eyelids are measured."
				   : redo2	    ? " Redo step 2 once so it measures how far your eyes move."
						    : "";
		*warn = hint && done;
		snprintf(calib, sizeof(calib), "%s%sCalibration: camera %s, notes %s.%s", done ? done : "",
			 done ? " " : "", lens ? "done" : "not done", notes ? "done" : "not done", redo);
	}

	if (!eye_contact)
		snprintf(track, sizeof(track), "Eye contact is off.");
	else if (flipped)
		snprintf(track, sizeof(track),
			 "Eye contact can't run: this camera sends upside-down (flipped) frames. Mirroring still works.");
	else if (!face)
		snprintf(track, sizeof(track), "No face found. Light your face and look towards the camera.");
	else
		snprintf(track, sizeof(track), "Face found.");

	if (vms > 0)
		snprintf(speed, sizeof(speed),
			 "Speed: eye tracking %.1f ms per frame (%.0f frames/s), in the background. "
			 "Warp and mirror %.2f ms.",
			 vms, fps, wms);
	else
		snprintf(speed, sizeof(speed), "Speed: warp and mirror %.2f ms per frame.", wms);

	snprintf(out, size, "%s\n%s\n%s", calib, track, speed);
}

static obs_properties_t *camflip_properties(void *data)
{
	struct camflip *f = data;
	bool has_notes = false, notes_used = false;
	if (f) {
		pthread_mutex_lock(&f->lock);
		has_notes = f->has_notes;
		notes_used = f->has_notes && !f->notes_stale;
		pthread_mutex_unlock(&f->lock);
	}

	obs_properties_t *props = obs_properties_create();

	obs_property_t *p = obs_properties_add_bool(props, S_MIRROR, "Mirror video (everyone in the call sees it mirrored)");
	obs_property_set_long_description(
		p, "Call apps only mirror your own preview. This mirrors the video that is actually sent.");

	obs_properties_add_bool(props, S_EYE_CONTACT, "Eye contact");

	/* Up to CF_STRENGTH_MAX (200%): over 100% the correction overshoots, for when glances still show. */
	p = obs_properties_add_float_slider(props, S_STRENGTH, "Eye contact strength", 0.0, 100.0 * CF_STRENGTH_MAX, 1.0);
	obs_property_float_set_suffix(p, "%");
	obs_property_set_long_description(
		p, "100% cancels the glance at your notes as it is measured. If you still look like you're reading, "
		   "go higher (up to 200%) to lift your eyes further; lower keeps some of the natural movement. "
		   "The damping of your eyes' sideways scanning along a line doesn't change with it (above 25%).");

	p = obs_properties_add_float_slider(props, S_MAX_GLANCE, "Correct glances up to", 5.0, 30.0, 1.0);
	obs_property_float_set_suffix(p, "%");
	obs_property_set_long_description(
		p, "How far down you can look and still be corrected. Glances further down (like at your keyboard) "
		   "fade back to your real eyes so it stays natural. Not used once step 2 (notes) is calibrated: "
		   "then the correction is matched to your notes and fades out just beyond them.");
	if (notes_used)
		obs_property_set_enabled(p, false);

	p = obs_properties_add_button2(props, "calibrate_lens", "1. Click, then look into the camera",
				       camflip_calib_lens_clicked, data);
	obs_property_set_long_description(p, "After clicking, look straight into the camera lens until the status says it's "
					     "done (a second or two). Redo it if you move your screen or chair.");

	p = obs_properties_add_button2(props, "calibrate_notes", "2. Click, then look at your notes",
				       camflip_calib_notes_clicked, data);
	obs_property_set_long_description(
		p, "After clicking, look at the middle of the line you usually read, with your head still, until the "
		   "status says it's done (a second or two). The correction is then matched exactly to your notes.");

	if (has_notes) {
		p = obs_properties_add_button2(props, "forget_notes", "Forget notes calibration",
					       camflip_forget_notes_clicked, data);
		obs_property_set_long_description(p, "Go back to using the \"Correct glances up to\" slider.");
	}

	if (f) {
		char text[880]; /* (calib, track and speed in status_text, and their line breaks) */
		bool warn = false;
		status_text(f, text, sizeof(text), &warn);
		/* No long description: OBS then shows the description itself as the info text. */
		p = obs_properties_add_text(props, "status", text, OBS_TEXT_INFO);
		if (warn)
			obs_property_text_set_info_type(p, OBS_TEXT_INFO_WARNING);
	}

	p = obs_properties_add_bool(props, S_SHOW_EYES, "Show tracked eyes (for tuning)");
	obs_property_set_long_description(
		p, "While a glance is being corrected, draws thin outlines where the correction works: white = "
		   "the area that may change, black = the eye opening your iris is moved inside. Nothing is drawn "
		   "while you look into the camera. Turn it off before a call.");

	obs_properties_add_text(props, "info",
				"Tip: in Teams/Zoom/Meet pick “OBS Virtual Camera” and turn off “Mirror my video”, "
				"so your own preview matches what others see. Toggle eye contact with a hotkey: "
				"Settings › Hotkeys › “" HOTKEY_DESC "” under your camera.",
				OBS_TEXT_INFO);
	return props;
}

/* ---------------------------------------------------------------- Video */

static struct obs_source_frame *camflip_filter_video(void *data, struct obs_source_frame *frame)
{
	struct camflip *f = data;
	const uint64_t start = now_ns();

	pthread_mutex_lock(&f->lock);
	const bool mirror = f->mirror;
	const bool eye_contact = f->eye_contact;
	const bool show_eyes = f->show_eyes;
	const bool calibrating = f->calib_request != 0 || f->calib_running != 0;
	const cf_gaze_params params = params_locked(f);
	f->frames_flipped = frame->flip; /* such frames aren't tracked (below): the panel says why */
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
	/* Brightness channel: the luma plane, or green for RGB cameras. */
	const cf_channel *luma = &channels[is_rgb(fmt) && count > 1 ? 1 : 0];

	/* Turning eye contact back on starts from a clean slate (fades in), not from stale tracking. */
	if (eye_contact && !f->vt_was_on && !calibrating) {
		pthread_mutex_lock(&f->gaze_lock);
		cf_gaze_init(&f->gaze);
		pthread_mutex_unlock(&f->gaze_lock);
	}
	f->vt_was_on = eye_contact;

	if (frame->flip) {
		/* Upside-down (bottom-up) frames aren't tracked (below): the panel says why; log it once.
		 * A calibration can't run on them either: cancel it at once, with the panel saying why,
		 * instead of showing "Calibrating…" until it times out for lack of frames. */
		if (eye_contact && !f->warned_flip) {
			f->warned_flip = true;
			blog(LOG_WARNING, "[cam-flip] eye contact is unavailable: this camera sends upside-down "
					  "(flipped) frames. Mirroring still works");
		}
		if (calibrating && calib_cancel_flipped(f))
			queue_ui(f, NULL); /* show it in an open panel */
	}

	/* Eye contact first (on the camera's original orientation), then mirror. */
	if ((eye_contact || calibrating) && !frame->flip && f->worker_started) {
		const double t = (double)frame->timestamp / 1e9;
		submit_frame(f, luma, t);

		cf_eye eyes[2];
		int n = 0;
		if (eye_contact) {
			pthread_mutex_lock(&f->gaze_lock);
			n = cf_gaze_predict(&f->gaze, &params, t, eyes);
			pthread_mutex_unlock(&f->gaze_lock);
			if (n > 0)
				cf_warp_frame(channels, count, eyes, n);
		}
		if (show_eyes)
			draw_overlay(luma, eyes, n);
	}

	if (mirror)
		cf_mirror_frame(channels, count);

	/* Timing: the video thread part, plus a summary in the log every 10 s. */
	const uint64_t end = now_ns();
	const double ms = (double)(end - start) / 1e6;
	bool log_now = false;
	double vis_ms = 0, warp_ms = 0, secs = 0;
	unsigned long vis_n = 0, full_n = 0;
	pthread_mutex_lock(&f->lock);
	f->ema_warp_ms = f->ema_warp_ms > 0 ? f->ema_warp_ms * 0.95 + ms * 0.05 : ms;
	f->win_warp_ms += ms;
	f->win_warp_n++;
	if (!f->win_start)
		f->win_start = end;
	if (end - f->win_start >= LOG_EVERY_NS) {
		log_now = true;
		secs = (double)(end - f->win_start) / 1e9;
		vis_ms = f->win_vision_n ? f->win_vision_ms / f->win_vision_n : 0;
		vis_n = f->win_vision_n;
		full_n = f->win_full_n;
		warp_ms = f->win_warp_n ? f->win_warp_ms / f->win_warp_n : 0;
		f->win_vision_ms = f->win_warp_ms = 0;
		f->win_vision_n = f->win_full_n = f->win_warp_n = 0;
		f->win_start = end;
	}
	pthread_mutex_unlock(&f->lock);
	if (log_now) {
		if (vis_n)
			blog(LOG_INFO,
			     "[cam-flip] tracking %.1f ms avg (%.0f fps analysed, %lu full face detections), warp %.2f ms",
			     vis_ms, (double)vis_n / secs, full_n, warp_ms);
		else
			blog(LOG_INFO, "[cam-flip] tracking idle, warp %.2f ms", warp_ms);
	}
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
	.filter_add = camflip_filter_add,
	.filter_remove = camflip_filter_remove,
};

bool obs_module_load(void)
{
	obs_register_source(&camflip_filter);
	blog(LOG_INFO, "[cam-flip] loaded (version %s)", CAMFLIP_VERSION);
	return true;
}
