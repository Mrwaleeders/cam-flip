/*
 * Diagnostics: renders a recording through Cam Flip's real tracking, gaze model and warp, the way
 * the plugin runs live, so a change can be judged on a real face, frame by frame, before it ships.
 *
 *   render_video <in.mov> <out-dir> [options]
 *
 * Writes into <out-dir>:
 *   corrected.mov  the full frame with eye contact applied (no mirror), high bitrate H.264
 *   compare.mov    an eye close-up: the original on top, corrected below, 3x enlarged
 *   frames.csv     per analysed frame: t, warp count, eye centres and shifts, pupils seen in the
 *                  output (--measure), for numbers
 *
 * Options
 *   --lag N            the warp for frame k uses tracking from frame k-N, moved forward by
 *                      cf_gaze_predict (live it is 1: the worker analyses frame k after it is shown)
 *   --dedupe MODE      OBS records at 60 fps from a 25 or 30 fps camera, so some frames repeat; the
 *                      filter never sees the repeats. "cadence" (default) finds the camera rate and phase
 *                      from the frame-to-frame differences; a number X skips frames differing from the last
 *                      kept one by less than X grey levels (drops real frames while still: avoid); 0 keeps
 *                      every frame.
 *   --baseline B --baseline-open O --notes-offset N --notes-side S --notes-open P
 *   --head-side S --head-vert V --head-eyes E --head-tilt K
 *                      calibration, as saved in the filter's settings (OBS scene JSON); without the
 *                      --head-* values the head-turn fade assumes a typical face (as the plugin does
 *                      after a camera calibration by an older version)
 *   --lens a-b --notes a-b   or calibrate on these time ranges (seconds) of the clip instead (--lens
 *                      also learns the head looking into the camera)
 *   --strength S       0..2 (default 1; over 1 overshoots, like the 200% slider)
 *   --scan-damp D      0..1 (default 0): the "Hide reading along a line" slider
 *   --measure          also track the corrected output and report how much the iris seen by the
 *                      viewer shakes (RMS of value minus its centred 5-frame mean, pixels)
 *   --zoom Z           compare.mov enlargement (default 3)
 *   --learn            print the notes learnt from the reading as they are handed out (and the final
 *                      estimate) and, unless the notes are given (--notes, --notes-offset), pass them back
 *                      in the params the way the plugin saves them (as tools/replay.c --learn)
 *
 * Nothing leaves this Mac. tools/render.sh builds and runs it.
 */
#define CAMFLIP_TEST 1
#include "../src/plugin.m"

#import <AVFoundation/AVFoundation.h>

typedef struct {
	double t;
	int ok;
	double in_y[2], out_y[2], in_x[2], out_x[2]; /* pupil minus eye-outline centroid */
	double shift_y[2], cx[2], cy[2];
	int dok;
	double dark_in[2], dark_out[2]; /* dark_centroid y, original and corrected frame */
} meas_t;

static double shake5(const double *x, const int *ok, int n)
{
	double s = 0;
	int c = 0;
	for (int i = 2; i < n - 2; i++) {
		if (!(ok[i - 2] && ok[i - 1] && ok[i] && ok[i + 1] && ok[i + 2]))
			continue;
		const double m = (x[i - 2] + x[i - 1] + x[i] + x[i + 1] + x[i + 2]) / 5.0;
		s += (x[i] - m) * (x[i] - m);
		c++;
	}
	return c ? sqrt(s / c) : NAN;
}

static cf_pt centroid_of(const cf_pt *p, int n)
{
	cf_pt c = {0, 0};
	for (int i = 0; i < n; i++) {
		c.x += p[i].x;
		c.y += p[i].y;
	}
	if (n) {
		c.x /= (float)n;
		c.y /= (float)n;
	}
	return c;
}

/* Independent iris measure for the summary (not the plugin's refiner): darkness-weighted centroid
 * of the pixels inside the eye outline, relative to the outline's centroid. Biased by the lids, but
 * smooth, so the frame-to-frame change of (corrected - original) is the shake the warp adds. */
static int cmp_u8(const void *a, const void *b)
{
	return (int)*(const uint8_t *)a - (int)*(const uint8_t *)b;
}

static int dark_centroid(const uint8_t *img, size_t stride, int w, int h, const cf_pt *eye, int n, cf_pt *out)
{
	if (n < 4)
		return 0;
	float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
	for (int i = 0; i < n; i++) {
		x0 = fminf(x0, eye[i].x);
		x1 = fmaxf(x1, eye[i].x);
		y0 = fminf(y0, eye[i].y);
		y1 = fmaxf(y1, eye[i].y);
	}
	const int ix0 = (int)fmaxf(0, floorf(x0)), ix1 = (int)fminf((float)w - 1, ceilf(x1));
	const int iy0 = (int)fmaxf(0, floorf(y0)), iy1 = (int)fminf((float)h - 1, ceilf(y1));
	if (ix1 - ix0 < 8 || iy1 - iy0 < 3 || (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > 400 * 200)
		return 0;
	static uint8_t vals[400 * 200];
	static float xs[400 * 200], ys[400 * 200];
	int m = 0;
	for (int y = iy0; y <= iy1; y++)
		for (int x = ix0; x <= ix1; x++) {
			const float px = (float)x + 0.5f, py = (float)y + 0.5f;
			int inside = 0;
			for (int k = 0, j = n - 1; k < n; j = k++)
				if ((eye[k].y > py) != (eye[j].y > py) &&
				    px < (eye[j].x - eye[k].x) * (py - eye[k].y) / (eye[j].y - eye[k].y) + eye[k].x)
					inside = !inside;
			if (!inside)
				continue;
			vals[m] = img[(size_t)y * stride + x];
			xs[m] = px;
			ys[m] = py;
			m++;
		}
	if (m < 30)
		return 0;
	static uint8_t sorted[400 * 200];
	memcpy(sorted, vals, (size_t)m);
	qsort(sorted, (size_t)m, 1, cmp_u8);
	const float lo = sorted[m / 20], hi = sorted[m * 7 / 10];
	const float thr = lo + 0.5f * (hi - lo);
	if (!(thr > lo + 2))
		return 0;
	double sw = 0, sx = 0, sy = 0;
	for (int i = 0; i < m; i++) {
		float wt = (thr - (float)vals[i]) / (thr - lo);
		wt = wt < 0 ? 0 : (wt > 1 ? 1 : wt);
		wt *= wt;
		sw += wt;
		sx += wt * xs[i];
		sy += wt * ys[i];
	}
	if (sw < 1)
		return 0;
	const cf_pt c = centroid_of(eye, n);
	out->x = (float)(sx / sw) - c.x;
	out->y = (float)(sy / sw) - c.y;
	return 1;
}

/* RMS of the frame-to-frame change, over consecutive pairs where both are ok. */
static double step_rms(const double *x, const int *ok, int n)
{
	double s = 0;
	int c = 0;
	for (int i = 1; i < n; i++)
		if (ok[i] && ok[i - 1]) {
			s += (x[i] - x[i - 1]) * (x[i] - x[i - 1]);
			c++;
		}
	return c ? sqrt(s / c) : NAN;
}

/* Finds which recorded frames repeat an earlier camera frame (OBS records at 60 fps from a 25 or
 * 30 fps camera). Model: recorded frame i shows camera frame floor((t_i - t_0) * f + phase); it is
 * new when that index changes. Searches the camera rate f (common rates, with a little clock drift)
 * and the phase for the labelling where new frames differ from their predecessor most clearly more
 * than repeats do. Returns 1 and fills is_dup when a cadence is clear (ratio >= 1.5), else 0. */
static int find_cadence(const double *diff, const double *t, int n, unsigned char *is_dup, double *fps_out,
			double *ratio_out)
{
	static const double rates[] = {30.0, 29.97, 25.0, 24.0, 20.0, 15.0, 12.0, 10.0};
	double best = 0, best_f = 0, best_ph = 0;
	if (n < 10)
		return 0;
	const double rec_fps = (n - 1) / fmax(t[n - 1] - t[0], 1e-6);
	for (size_t r = 0; r < sizeof(rates) / sizeof(rates[0]); r++) {
		if (rates[r] > rec_fps * 0.95)
			continue;
		for (int d = -15; d <= 15; d++) {
			const double f = rates[r] * (1.0 + d * 0.0002);
			for (int p = 0; p < 80; p++) {
				const double ph = p / 80.0;
				double sn = 0, sr = 0;
				int cn = 0, cr = 0;
				long prev = (long)floor((t[0] - t[0]) * f + ph);
				for (int i = 1; i < n; i++) {
					const long idx = (long)floor((t[i] - t[0]) * f + ph);
					if (idx != prev) {
						sn += diff[i];
						cn++;
					} else {
						sr += diff[i];
						cr++;
					}
					prev = idx;
				}
				if (!cn || !cr)
					continue;
				const double ratio = (sn / cn) / fmax(sr / cr, 1e-3);
				if (ratio > best) {
					best = ratio;
					best_f = f;
					best_ph = ph;
				}
			}
		}
	}
	*fps_out = best_f;
	*ratio_out = best;
	if (best < 1.5)
		return 0;
	long prev = (long)floor(best_ph);
	is_dup[0] = 0;
	for (int i = 1; i < n; i++) {
		const long idx = (long)floor((t[i] - t[0]) * best_f + best_ph);
		is_dup[i] = idx == prev;
		prev = idx;
	}
	return 1;
}

/* Openness and gaze offsets of one eye, the way the gaze model measures them: relative to the line
 * through the two outline points farthest apart (the corners), in eye widths, positive = up. */
static void eye_metrics(const cf_pt *eye, int n, cf_pt refined, cf_pt vision, double *open, double *off_r, double *off_v)
{
	*open = *off_r = *off_v = NAN;
	if (n < 4)
		return;
	int a = 0, b = 1;
	float best = -1;
	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++) {
			const float dx = eye[j].x - eye[i].x, dy = eye[j].y - eye[i].y;
			if (dx * dx + dy * dy > best) {
				best = dx * dx + dy * dy;
				a = i;
				b = j;
			}
		}
	if (eye[a].x > eye[b].x) {
		const int tmp = a;
		a = b;
		b = tmp;
	}
	const double w = sqrt(best);
	if (!(w > 2))
		return;
	const double ux = (eye[b].x - eye[a].x) / w, uy = (eye[b].y - eye[a].y) / w;
	const double nx = uy, ny = -ux; /* up (image y grows downwards) */
	double lo = 0, hi = 0;
	for (int i = 0; i < n; i++) {
		const double v = (eye[i].x - eye[a].x) * nx + (eye[i].y - eye[a].y) * ny;
		lo = fmin(lo, v);
		hi = fmax(hi, v);
	}
	*open = (hi - lo) / w;
	*off_r = ((refined.x - eye[a].x) * nx + (refined.y - eye[a].y) * ny) / w;
	*off_v = ((vision.x - eye[a].x) * nx + (vision.y - eye[a].y) * ny) / w;
}

static AVAssetWriter *make_writer(NSString *path, int w, int h, double bitrate, AVAssetWriterInput **in_out,
				  AVAssetWriterInputPixelBufferAdaptor **ad_out)
{
	[[NSFileManager defaultManager] removeItemAtPath:path error:nil];
	NSError *err = nil;
	AVAssetWriter *wr = [AVAssetWriter assetWriterWithURL:[NSURL fileURLWithPath:path] fileType:AVFileTypeQuickTimeMovie
							 error:&err];
	if (!wr)
		return nil;
	NSDictionary *vs = @{
		AVVideoCodecKey: AVVideoCodecTypeH264,
		AVVideoWidthKey: @(w),
		AVVideoHeightKey: @(h),
		AVVideoCompressionPropertiesKey: @{AVVideoAverageBitRateKey: @(bitrate)},
	};
	AVAssetWriterInput *in = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:vs];
	in.expectsMediaDataInRealTime = NO;
	NSDictionary *pa = @{
		(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange),
		(id)kCVPixelBufferWidthKey: @(w),
		(id)kCVPixelBufferHeightKey: @(h),
	};
	AVAssetWriterInputPixelBufferAdaptor *ad =
		[AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:in
										  sourcePixelBufferAttributes:pa];
	[wr addInput:in];
	*in_out = in;
	*ad_out = ad;
	return wr;
}

static void append(AVAssetWriterInput *in, AVAssetWriterInputPixelBufferAdaptor *ad, CVPixelBufferRef pb, double t)
{
	while (!in.readyForMoreMediaData)
		usleep(1000);
	[ad appendPixelBuffer:pb withPresentationTime:CMTimeMakeWithSeconds(t, 600)];
}

/* Copies the region (x0, y0, cw, ch) of an NV12 frame, enlarged z times (nearest), into dst at row
 * offset dy (even). */
static void blit_zoom(const uint8_t *sy, size_t sys, const uint8_t *suv, size_t suvs, int x0, int y0, int cw, int ch,
		      int z, uint8_t *dy_, size_t dys, uint8_t *duv, size_t duvs, int dy)
{
	for (int y = 0; y < ch * z; y++) {
		const uint8_t *s = sy + (size_t)(y0 + y / z) * sys + x0;
		uint8_t *d = dy_ + (size_t)(dy + y) * dys;
		for (int x = 0; x < cw * z; x++)
			d[x] = s[x / z];
	}
	for (int y = 0; y < ch * z / 2; y++) {
		const int sy2 = (y0 + (2 * y) / z) / 2;
		const uint8_t *s = suv + (size_t)sy2 * suvs;
		uint8_t *d = duv + (size_t)(dy / 2 + y) * duvs;
		for (int x = 0; x < cw * z / 2; x++) {
			const int sx2 = (x0 + (2 * x) / z) / 2;
			d[2 * x] = s[2 * sx2];
			d[2 * x + 1] = s[2 * sx2 + 1];
		}
	}
}

static int parse_range(const char *s, double r[2])
{
	return sscanf(s, "%lf-%lf", &r[0], &r[1]) == 2;
}

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "usage: render_video <in> <out-dir> [--lag N] [--dedupe X] [--baseline B] "
				"[--baseline-open O] [--notes-offset N] [--notes-side S] [--notes-open P] [--head-side S] "
				"[--head-vert V] [--head-eyes E] [--head-tilt K] [--baseline-upper U] [--baseline-lower L] "
				"[--vis-baseline A] [--vis-notes B] [--lens a-b] [--notes a-b] [--strength S] [--scan-damp D] "
				"[--measure] [--zoom Z] [--learn]\n");
		return 2;
	}
	int lag = 1, measure = 0, zoom = 3, learn = 0;
	double dedupe = -1, lens[2] = {0, 0}, notes_r[2] = {0, 0};
	cf_gaze_params p = {.strength = 1.0f, .max_glance = 0.15f};
	int vis_opt = 0; /* --vis-baseline (1) and --vis-notes (2) given */
	for (int i = 3; i < argc; i++) {
		const char *a = argv[i];
		const char *v = i + 1 < argc ? argv[i + 1] : "0";
		if (!strcmp(a, "--lag")) { lag = atoi(v); i++; }
		else if (!strcmp(a, "--dedupe")) { dedupe = !strcmp(v, "cadence") ? -1 : atof(v); i++; }
		else if (!strcmp(a, "--baseline")) { p.baseline = (float)atof(v); i++; }
		else if (!strcmp(a, "--baseline-open")) { p.baseline_open = (float)atof(v); i++; }
		else if (!strcmp(a, "--notes-offset")) { p.notes_offset = (float)atof(v); p.has_notes = 1; i++; }
		else if (!strcmp(a, "--notes-side")) { p.notes_side = (float)atof(v); i++; }
		else if (!strcmp(a, "--notes-open")) { p.notes_open = (float)atof(v); i++; }
		else if (!strcmp(a, "--head-side")) { p.head_ref.side = (float)atof(v); p.head_ref.ok = 1; i++; }
		else if (!strcmp(a, "--head-vert")) { p.head_ref.vert = (float)atof(v); p.head_ref.ok = 1; i++; }
		else if (!strcmp(a, "--head-eyes")) { p.head_ref.eyes = (float)atof(v); p.head_ref.ok = 1; i++; }
		else if (!strcmp(a, "--head-tilt")) { p.head_ref.tilt = (float)atof(v); p.head_ref.ok = 1; i++; }
		else if (!strcmp(a, "--baseline-upper")) { p.baseline_upper = (float)atof(v); i++; }
		else if (!strcmp(a, "--baseline-lower")) { p.baseline_lower = (float)atof(v); i++; }
		else if (!strcmp(a, "--vis-baseline")) { p.vis_baseline = (float)atof(v); vis_opt |= 1; i++; }
		else if (!strcmp(a, "--vis-notes")) { p.vis_notes = (float)atof(v); vis_opt |= 2; i++; }
		else if (!strcmp(a, "--lens")) { parse_range(v, lens); i++; }
		else if (!strcmp(a, "--notes")) { parse_range(v, notes_r); i++; }
		else if (!strcmp(a, "--strength")) { p.strength = (float)atof(v); i++; }
		else if (!strcmp(a, "--scan-damp")) { p.scan_damp = (float)atof(v); i++; }
		else if (!strcmp(a, "--zoom")) { zoom = atoi(v) < 1 ? 1 : atoi(v); i++; }
		else if (!strcmp(a, "--measure")) measure = 1;
		else if (!strcmp(a, "--learn")) learn = 1;
		else { fprintf(stderr, "unknown option %s\n", a); return 2; }
	}
	p.has_vis_calib = vis_opt == 3 && p.has_notes; /* as plugin.m: both steps taken with Vision's point */
	if (lag < 0)
		lag = 0;
	/* Learnt notes are passed back unless notes were given (--notes-offset, --notes). */
	const int learn_feed = learn && !p.has_notes && !(notes_r[1] > notes_r[0]);

	@autoreleasepool {
		NSString *dir = [NSString stringWithUTF8String:argv[2]];
		[[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil
								error:nil];
		AVURLAsset *asset = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
							options:nil];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		NSArray<AVAssetTrack *> *tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
#pragma clang diagnostic pop
		if (tracks.count == 0) {
			fprintf(stderr, "no video track (remux .mkv to .mp4 first)\n");
			return 1;
		}

		obs_data_t *settings = obs_data_create();
		camflip_defaults(settings);
		struct camflip *f = camflip_create(settings, NULL);
		struct camflip *f_out = measure ? camflip_create(settings, NULL) : NULL;
		f->test_clock = true;
		if (f_out)
			f_out->test_clock = true;

		/* Pass 1 (only if calibrating from time ranges): run the calibration like the buttons do. */
		int vis_cal = 0; /* Vision's point given at step 1 (bit 1) and step 2 (bit 2), as in plugin.m */
		for (int step = 0; step < 2; step++) {
			const double *r = step == 0 ? lens : notes_r;
			if (!(r[1] > r[0]))
				continue;
			NSError *err = nil;
			AVAssetReader *rd = [AVAssetReader assetReaderWithAsset:asset error:&err];
			AVAssetReaderTrackOutput *o = [AVAssetReaderTrackOutput
				assetReaderTrackOutputWithTrack:tracks[0]
						 outputSettings:@{(id)kCVPixelBufferPixelFormatTypeKey:
									@(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)}];
			[rd addOutput:o];
			[rd startReading];
			cf_gaze cg;
			cf_gaze_init(&cg);
			cf_gaze_start_calibration(&cg, step == 0 ? CF_CALIB_LENS : CF_CALIB_NOTES);
			grey_frame g = {0};
			double t0 = -1;
			int got = 0;
			cf_calib_result res;
			CMSampleBufferRef sb;
			while (!got && (sb = [o copyNextSampleBuffer])) {
				CVPixelBufferRef pb = CMSampleBufferGetImageBuffer(sb);
				const double ts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb));
				if (t0 < 0)
					t0 = ts;
				const double t = ts - t0;
				if (t >= r[0] && t <= r[1]) {
					CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
					const int w = (int)CVPixelBufferGetWidthOfPlane(pb, 0), h = (int)CVPixelBufferGetHeightOfPlane(pb, 0);
					ensure(&g.data, &g.cap, (size_t)w * h);
					const uint8_t *src = CVPixelBufferGetBaseAddressOfPlane(pb, 0);
					for (int y = 0; y < h; y++)
						memcpy(g.data + (size_t)y * w, src + (size_t)y * CVPixelBufferGetBytesPerRowOfPlane(pb, 0), (size_t)w);
					CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
					g.w = w;
					g.h = h;
					g.t = t;
					cf_face face;
					bool full;
					detect_face(f, &g, &face, &full);
					cf_eye tmp[2];
					cf_gaze_update(&cg, &face, &p, t, tmp);
					got = cf_gaze_take_calibration(&cg, &res);
				}
				CFRelease(sb);
			}
			free(g.data);
			if (got > 0 && step == 0) {
				p.baseline = res.offset;
				p.baseline_open = res.openness;
				p.head_ref = res.head;
				vis_cal = res.has_vis ? (vis_cal | 1) : (vis_cal & ~1);
				p.vis_baseline = res.vis_offset;
				p.has_vis_calib = vis_cal == 3 && p.has_notes;
				p.baseline_upper = res.upper;
				p.baseline_lower = res.lower;
				printf("lens calibration: baseline %.4f open %.4f; head %s: side %.4f vert %.4f eyes %.4f tilt "
				       "%.4f\n",
				       res.offset, res.openness, res.head.ok ? "learnt" : "not learnt", res.head.side,
				       res.head.vert, res.head.eyes, res.head.tilt);
			} else if (got > 0) {
				p.has_notes = 1;
				p.notes_offset = res.offset;
				p.notes_side = res.side;
				p.notes_open = res.openness;
				vis_cal = res.has_vis ? (vis_cal | 2) : (vis_cal & ~2);
				p.vis_notes = res.vis_offset;
				p.has_vis_calib = vis_cal == 3;
				printf("notes calibration: offset %.4f side %.4f open %.4f\n", res.offset, res.side, res.openness);
			} else {
				printf("%s calibration failed (%d)\n", step == 0 ? "lens" : "notes", got);
			}
		}
		printf("params: strength %.2f baseline %.4f open %.4f notes %d offset %.4f side %.4f open %.4f, lag %d\n",
		       p.strength, p.baseline, p.baseline_open, p.has_notes, p.notes_offset, p.notes_side, p.notes_open, lag);
		if (p.head_ref.ok)
			printf("head looking into the camera: side %.4f vert %.4f eyes %.4f tilt %.4f\n", p.head_ref.side,
			       p.head_ref.vert, p.head_ref.eyes, p.head_ref.tilt);
		else
			printf("head looking into the camera: not given, a typical face is assumed (head turns only)\n");

		/* Repeated frames, by cadence: frame-to-frame differences alternate small (repeat) / large
		 * (new camera frame). Per frame, look at a 1 s window: if one parity's mean difference is well
		 * below the other's, frames of that parity are repeats. */
		int nd = 0;
		unsigned char *is_dup = NULL;
		if (dedupe < 0) {
			NSError *e0 = nil;
			AVAssetReader *rd = [AVAssetReader assetReaderWithAsset:asset error:&e0];
			AVAssetReaderTrackOutput *o = [AVAssetReaderTrackOutput
				assetReaderTrackOutputWithTrack:tracks[0]
						 outputSettings:@{(id)kCVPixelBufferPixelFormatTypeKey:
									@(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)}];
			[rd addOutput:o];
			[rd startReading];
			size_t dcap = 4096;
			double *diff = calloc(dcap, sizeof(double)), *tsv = calloc(dcap, sizeof(double));
			uint8_t *last = NULL;
			int lw = 0, lh = 0;
			CMSampleBufferRef sb0;
			while ((sb0 = [o copyNextSampleBuffer])) {
				CVPixelBufferRef pb = CMSampleBufferGetImageBuffer(sb0);
				CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
				const int w = (int)CVPixelBufferGetWidthOfPlane(pb, 0), h = (int)CVPixelBufferGetHeightOfPlane(pb, 0);
				const size_t st = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
				const uint8_t *py = CVPixelBufferGetBaseAddressOfPlane(pb, 0);
				if (!last) {
					lw = w;
					lh = h;
					last = calloc((size_t)((w + 3) / 4) * ((h + 3) / 4), 1);
				}
				double sad = 0;
				long cnt = 0;
				for (int y = 0; y < lh && y < h; y += 4)
					for (int x = 0; x < lw && x < w; x += 4, cnt++) {
						const size_t li = (size_t)(y / 4) * ((lw + 3) / 4) + (size_t)(x / 4);
						sad += abs((int)py[(size_t)y * st + x] - (int)last[li]);
						last[li] = py[(size_t)y * st + x];
					}
				CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
				const double ts0 = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb0));
				CFRelease(sb0);
				if ((size_t)nd == dcap) {
					dcap *= 2;
					diff = realloc(diff, dcap * sizeof(double));
					tsv = realloc(tsv, dcap * sizeof(double));
				}
				tsv[nd] = ts0;
				diff[nd] = nd == 0 ? 1e9 : sad / (double)(cnt ? cnt : 1);
				nd++;
			}
			free(last);
			is_dup = calloc((size_t)nd + 1, 1);
			double cam_fps = 0, ratio = 0;
			const int found = find_cadence(diff, tsv, nd, is_dup, &cam_fps, &ratio);
			int ndup = 0;
			for (int i = 0; i < nd; i++)
				ndup += is_dup[i];
			free(diff);
			free(tsv);
			if (found)
				printf("cadence: camera %.2f fps (new frames differ %.1fx more than repeats)\n", cam_fps, ratio);
			else
				printf("cadence: no repeat pattern found (best ratio %.2f): keeping every frame\n", ratio);
			printf("cadence: %d of %d frames are repeats\n", ndup, nd);
		}

		NSError *err = nil;
		AVAssetReader *reader = [AVAssetReader assetReaderWithAsset:asset error:&err];
		AVAssetReaderTrackOutput *out = [AVAssetReaderTrackOutput
			assetReaderTrackOutputWithTrack:tracks[0]
					 outputSettings:@{(id)kCVPixelBufferPixelFormatTypeKey:
								@(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)}];
		[reader addOutput:out];
		if (![reader startReading]) {
			fprintf(stderr, "cannot read video\n");
			return 1;
		}

		cf_gaze gaze;
		cf_gaze_init(&gaze);
		enum { QMAX = 16 };
		cf_face q[QMAX]; /* faces waiting to be fed, newest last */
		double qt[QMAX];
		int qn = 0;

		AVAssetWriter *wfull = nil, *wcmp = nil;
		AVAssetWriterInput *ifull = nil, *icmp = nil;
		AVAssetWriterInputPixelBufferAdaptor *afull = nil, *acmp = nil;
		FILE *csv = fopen([[dir stringByAppendingPathComponent:@"frames.csv"] UTF8String], "w");
		/* in_*: refined pupil y minus outline centroid (px). l/r_open, l/r_ref, l/r_vis: openness and
		 * the refined and Vision pupil offsets (eye widths, + = up) the gaze model saw on this frame.
		 * yaw, pitch: the smoothed head pose (pose.c, degrees from the calibrated one) its fade used. */
		fprintf(csv, "t,n,factor,cx0,cy0,sx0,sy0,cx1,cy1,sx1,sy1,in_ly,in_ry,out_ly,out_ry,"
			     "l_open,l_ref,l_vis,r_open,r_ref,r_vis,yaw,pitch\n");

		size_t cap = 4096, nm = 0;
		meas_t *m = calloc(cap, sizeof(meas_t));
		uint8_t *prev = NULL;
		int frames = 0, kept = 0, warped = 0;
		double t0 = -1, track_ms = 0;
		int crop_x = -1, crop_y = 0, crop_w = 0, crop_h = 0;
		grey_frame g = {0}, g2 = {0};
		CMSampleBufferRef sb;
		while ((sb = [out copyNextSampleBuffer])) {
			@autoreleasepool {
				CVPixelBufferRef src = CMSampleBufferGetImageBuffer(sb);
				const double ts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb));
				if (t0 < 0)
					t0 = ts;
				const double t = ts - t0;
				frames++;
				CVPixelBufferLockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
				const int w = (int)CVPixelBufferGetWidthOfPlane(src, 0), h = (int)CVPixelBufferGetHeightOfPlane(src, 0);
				const size_t sys = CVPixelBufferGetBytesPerRowOfPlane(src, 0), suvs = CVPixelBufferGetBytesPerRowOfPlane(src, 1);
				const uint8_t *sy = CVPixelBufferGetBaseAddressOfPlane(src, 0), *suv = CVPixelBufferGetBaseAddressOfPlane(src, 1);

				/* Duplicate frames (60 fps recording of a 30 fps camera): the filter never sees them. */
				bool dup = false;
				const int fi = frames - 1;
				if (!prev)
					prev = malloc((size_t)w * h);
				else if (dedupe < 0)
					dup = is_dup && fi < nd && is_dup[fi];
				else if (dedupe > 0) {
					double sad = 0;
					long cnt = 0;
					for (int y = 0; y < h; y += 4)
						for (int x = 0; x < w; x += 4, cnt++)
							sad += abs((int)sy[(size_t)y * sys + x] - (int)prev[(size_t)y * w + x]);
					dup = sad / (double)cnt < dedupe;
				}
				if (dup) {
					CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
					CFRelease(sb);
					continue;
				}
				kept++;
				for (int y = 0; y < h; y++)
					memcpy(prev + (size_t)y * w, sy + (size_t)y * sys, (size_t)w);

				/* Writable copy, the frame OBS hands the filter. */
				CVPixelBufferRef dst = NULL;
				CVPixelBufferCreate(NULL, (size_t)w, (size_t)h, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, NULL, &dst);
				CVPixelBufferLockBaseAddress(dst, 0);
				uint8_t *dy = CVPixelBufferGetBaseAddressOfPlane(dst, 0), *duv = CVPixelBufferGetBaseAddressOfPlane(dst, 1);
				const size_t dys = CVPixelBufferGetBytesPerRowOfPlane(dst, 0), duvs = CVPixelBufferGetBytesPerRowOfPlane(dst, 1);
				for (int y = 0; y < h; y++)
					memcpy(dy + (size_t)y * dys, sy + (size_t)y * sys, (size_t)w);
				for (int y = 0; y < (h + 1) / 2; y++)
					memcpy(duv + (size_t)y * duvs, suv + (size_t)y * suvs, (size_t)((w + 1) / 2 * 2));

				/* Track this frame (the worker would do it after the frame is shown). */
				ensure(&g.data, &g.cap, (size_t)w * h);
				memcpy(g.data, prev, (size_t)w * h);
				g.w = w;
				g.h = h;
				g.t = t;
				cf_face face;
				bool full;
				const uint64_t s0 = now_ns();
				detect_face(f, &g, &face, &full);
				track_ms += (double)(now_ns() - s0) / 1e6;
				if (qn == QMAX) {
					memmove(q, q + 1, sizeof(q[0]) * (QMAX - 1));
					memmove(qt, qt + 1, sizeof(qt[0]) * (QMAX - 1));
					qn--;
				}
				q[qn] = face;
				qt[qn] = t;
				qn++;
				/* Feed everything except the newest `lag` frames. */
				while (qn > lag) {
					cf_eye tmp[2];
					cf_gaze_update(&gaze, &q[0], &p, qt[0], tmp);
					cf_learnt_notes ln;
					if (learn && cf_gaze_take_learnt_notes(&gaze, &ln)) {
						printf("learn: notes handed out at %.1f s: offset %.4f side %.4f open %.4f vis %.4f (%s), "
						       "%.4f below the baseline %.4f (%.0f s of reading)\n",
						       qt[0], ln.offset, ln.side, ln.open, ln.vis_offset, ln.has_vis ? "given" : "none",
						       p.baseline - ln.offset, p.baseline, ln.seconds);
						if (learn_feed) { /* as the plugin saves them */
							p.has_notes = 1;
							p.notes_offset = ln.offset;
							p.notes_side = ln.side;
							p.notes_open = ln.open;
							p.notes_learnt = 1;
							p.notes_seconds = ln.seconds;
							p.vis_notes = ln.vis_offset;
							p.has_vis_calib = ln.has_vis && ((vis_cal & 1) || (vis_opt & 1));
						}
					}
					memmove(q, q + 1, sizeof(q[0]) * (QMAX - 1));
					memmove(qt, qt + 1, sizeof(qt[0]) * (QMAX - 1));
					qn--;
				}

				/* Warp, exactly like camflip_filter_video. */
				cf_eye eyes[2];
				memset(eyes, 0, sizeof(eyes));
				const int n = cf_gaze_predict(&gaze, &p, t, eyes);
				cf_gaze_status st;
				cf_gaze_get_status(&gaze, &st);
				if (n > 0) {
					uint8_t *planes[2] = {dy, duv};
					const uint32_t ls[2] = {(uint32_t)dys, (uint32_t)duvs};
					cf_channel ch[8];
					const int cc = cf_channels(CF_FMT_NV12, planes, ls, w, h, ch);
					cf_warp_frame(ch, cc, eyes, n);
					warped++;
				}

				/* What the viewer sees: track the corrected frame too. */
				cf_face seen;
				memset(&seen, 0, sizeof(seen));
				if (f_out) {
					ensure(&g2.data, &g2.cap, (size_t)w * h);
					for (int y = 0; y < h; y++)
						memcpy(g2.data + (size_t)y * w, dy + (size_t)y * dys, (size_t)w);
					g2.w = w;
					g2.h = h;
					g2.t = t;
					bool full2;
					detect_face(f_out, &g2, &seen, &full2);
				}
				if (nm == cap) {
					cap *= 2;
					m = realloc(m, cap * sizeof(meas_t));
				}
				meas_t *mm = &m[nm++];
				memset(mm, 0, sizeof(*mm));
				mm->t = t;
				mm->ok = face.found && face.left_count >= 4 && face.right_count >= 4 && (!f_out || seen.found);
				const cf_pt *ein[2] = {face.left_eye, face.right_eye};
				const int nin[2] = {face.left_count, face.right_count};
				const cf_pt pin[2] = {face.left_pupil, face.right_pupil};
				const cf_pt pout[2] = {seen.left_pupil, seen.right_pupil};
				for (int e = 0; e < 2; e++) {
					const cf_pt c = centroid_of(ein[e], nin[e]);
					mm->in_y[e] = pin[e].y - c.y;
					mm->in_x[e] = pin[e].x - c.x;
					mm->out_y[e] = pout[e].y - c.y;
					mm->out_x[e] = pout[e].x - c.x;
					mm->shift_y[e] = n > e ? eyes[e].sy : 0;
					mm->cx[e] = n > e ? eyes[e].cx : c.x;
					mm->cy[e] = n > e ? eyes[e].cy : c.y;
				}
				{
					cf_pt a, b, c2, d2;
					mm->dok = face.found && dark_centroid(sy, sys, w, h, face.left_eye, face.left_count, &a) &&
						  dark_centroid(dy, dys, w, h, face.left_eye, face.left_count, &b) &&
						  dark_centroid(sy, sys, w, h, face.right_eye, face.right_count, &c2) &&
						  dark_centroid(dy, dys, w, h, face.right_eye, face.right_count, &d2);
					if (mm->dok) {
						mm->dark_in[0] = a.y;
						mm->dark_out[0] = b.y;
						mm->dark_in[1] = c2.y;
						mm->dark_out[1] = d2.y;
					}
				}
				double em[2][3];
				eye_metrics(face.left_eye, face.left_count, face.left_pupil, face.left_pupil_vision, &em[0][0],
					    &em[0][1], &em[0][2]);
				eye_metrics(face.right_eye, face.right_count, face.right_pupil, face.right_pupil_vision,
					    &em[1][0], &em[1][1], &em[1][2]);
				fprintf(csv,
					"%.4f,%d,%.3f,%.2f,%.2f,%.3f,%.3f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
					"%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f\n",
					t, n, st.factor, eyes[0].cx, eyes[0].cy, eyes[0].sx, eyes[0].sy, eyes[1].cx, eyes[1].cy,
					eyes[1].sx, eyes[1].sy, mm->in_y[0], mm->in_y[1], mm->out_y[0], mm->out_y[1], em[0][0],
					em[0][1], em[0][2], em[1][0], em[1][1], em[1][2], gaze.pose.yaw, gaze.pose.pitch);

				/* Writers, sized on the first frame. The close-up crop is fixed from the first face. */
				if (!wfull) {
					wfull = make_writer([dir stringByAppendingPathComponent:@"corrected.mov"], w, h, 40e6, &ifull, &afull);
					[wfull startWriting];
					[wfull startSessionAtSourceTime:kCMTimeZero];
				}
				if (crop_x < 0 && face.found && face.left_count && face.right_count) {
					float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
					for (int e = 0; e < 2; e++)
						for (int i = 0; i < nin[e]; i++) {
							x0 = fminf(x0, ein[e][i].x);
							x1 = fmaxf(x1, ein[e][i].x);
							y0 = fminf(y0, ein[e][i].y);
							y1 = fmaxf(y1, ein[e][i].y);
						}
					const float span = x1 - x0;
					crop_w = ((int)(span * 1.5f) + 7) & ~7;
					crop_h = ((int)(span * 0.45f) + 7) & ~7;
					crop_x = ((int)((x0 + x1) / 2 - crop_w / 2)) & ~1;
					crop_y = ((int)((y0 + y1) / 2 - crop_h / 2)) & ~1;
					crop_x = crop_x < 0 ? 0 : (crop_x + crop_w > w ? (w - crop_w) & ~1 : crop_x);
					crop_y = crop_y < 0 ? 0 : (crop_y + crop_h > h ? (h - crop_h) & ~1 : crop_y);
					wcmp = make_writer([dir stringByAppendingPathComponent:@"compare.mov"], crop_w * zoom,
							   crop_h * zoom * 2, 30e6, &icmp, &acmp);
					[wcmp startWriting];
					[wcmp startSessionAtSourceTime:kCMTimeZero];
				}
				append(ifull, afull, dst, t);
				if (wcmp) {
					CVPixelBufferRef cb = NULL;
					CVPixelBufferCreate(NULL, (size_t)(crop_w * zoom), (size_t)(crop_h * zoom * 2),
							    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, NULL, &cb);
					CVPixelBufferLockBaseAddress(cb, 0);
					uint8_t *cy = CVPixelBufferGetBaseAddressOfPlane(cb, 0), *cuv = CVPixelBufferGetBaseAddressOfPlane(cb, 1);
					const size_t cys = CVPixelBufferGetBytesPerRowOfPlane(cb, 0), cuvs = CVPixelBufferGetBytesPerRowOfPlane(cb, 1);
					blit_zoom(sy, sys, suv, suvs, crop_x, crop_y, crop_w, crop_h, zoom, cy, cys, cuv, cuvs, 0);
					blit_zoom(dy, dys, duv, duvs, crop_x, crop_y, crop_w, crop_h, zoom, cy, cys, cuv, cuvs, crop_h * zoom);
					CVPixelBufferUnlockBaseAddress(cb, 0);
					append(icmp, acmp, cb, t);
					CVPixelBufferRelease(cb);
				}
				CVPixelBufferUnlockBaseAddress(dst, 0);
				CVPixelBufferRelease(dst);
				CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
				CFRelease(sb);
			}
		}
		fclose(csv);
		for (int k = 0; k < 2; k++) {
			AVAssetWriter *wr = k ? wcmp : wfull;
			AVAssetWriterInput *in = k ? icmp : ifull;
			if (!wr)
				continue;
			[in markAsFinished];
			dispatch_semaphore_t sem = dispatch_semaphore_create(0);
			[wr finishWritingWithCompletionHandler:^{
				dispatch_semaphore_signal(sem);
			}];
			dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
			if (wr.status != AVAssetWriterStatusCompleted)
				fprintf(stderr, "writing failed: %s\n", wr.error.localizedDescription.UTF8String);
		}

		/* Summary */
		const int nn = (int)nm;
		int *ok = calloc((size_t)nn + 1, sizeof(int)), *okw = calloc((size_t)nn + 1, sizeof(int));
		double *buf[10];
		for (int s = 0; s < 10; s++)
			buf[s] = calloc((size_t)nn + 1, sizeof(double));
		for (int i = 0; i < nn; i++) {
			ok[i] = m[i].ok;
			okw[i] = m[i].ok && m[i].shift_y[0] != 0;
			buf[0][i] = m[i].in_y[0];
			buf[1][i] = m[i].in_y[1];
			buf[2][i] = m[i].out_y[0];
			buf[3][i] = m[i].out_y[1];
			buf[4][i] = m[i].in_y[0] + m[i].shift_y[0]; /* intended seen position */
			buf[5][i] = m[i].shift_y[0];
			buf[6][i] = m[i].cx[0];
			buf[7][i] = m[i].cy[0];
			buf[8][i] = m[i].out_x[0];
			buf[9][i] = m[i].in_x[0];
		}
		{
			int *dk = calloc((size_t)nn + 1, sizeof(int)), *dko = calloc((size_t)nn + 1, sizeof(int));
			double *din = calloc((size_t)nn + 1, sizeof(double)), *dout = calloc((size_t)nn + 1, sizeof(double)),
			       *dadd = calloc((size_t)nn + 1, sizeof(double)), *dadd1 = calloc((size_t)nn + 1, sizeof(double));
			for (int i = 0; i < nn; i++) {
				dko[i] = m[i].dok;
				din[i] = m[i].dark_in[0];
				dout[i] = m[i].dark_out[0];
				dadd[i] = m[i].shift_y[0];
				dadd1[i] = m[i].shift_y[1];
				dk[i] = m[i].ok && m[i].shift_y[0] != 0; /* warped frames */
			}
			printf("IRIS (dark-pixel centroid, independent of the plugin), y px: frame-to-frame RMS step original %.2f\n",
			       step_rms(din, dko, nn));
			/* The original iris moves smoothly, so the visible shake is the frame-to-frame change of the
			 * applied shift. Whole clip, and the worst 1 s window (shaking comes in bursts). */
			double worst = 0, worst_t = 0;
			for (int a = 0; a < nn; a++) {
				int b = a;
				while (b < nn && m[b].t < m[a].t + 1.0)
					b++;
				if (b - a >= 8) {
					const double r = step_rms(dadd + a, dk + a, b - a);
					if (r > worst) {
						worst = r;
						worst_t = m[a].t;
					}
				}
			}
			printf("SHAKE = frame-to-frame RMS change of the applied vertical shift (px, lower is better): "
			       "whole clip L %.2f R %.2f, worst 1 s window %.2f at %.1f s\n",
			       step_rms(dadd, dk, nn), step_rms(dadd1, dk, nn), worst, worst_t);
			free(dk);
			free(dko);
			free(din);
			free(dout);
			free(dadd);
			free(dadd1);
		}
		printf("frames %d read, %d kept (camera frames), %d warped, mean tracking %.1f ms\n", frames, kept, warped,
		       kept ? track_ms / kept : 0);
		printf("pupil-in-eye y shake, original video (px): L %.2f R %.2f | x L %.2f\n", shake5(buf[0], ok, nn),
		       shake5(buf[1], ok, nn), shake5(buf[9], ok, nn));
		printf("while warped: shift y %.2f | warp centre x %.2f y %.2f | intended seen y %.2f\n",
		       shake5(buf[5], okw, nn), shake5(buf[6], okw, nn), shake5(buf[7], okw, nn), shake5(buf[4], okw, nn));
		if (f_out)
			printf("SEEN by tracking the corrected video (px): y L %.2f R %.2f, x L %.2f (original y L %.2f, same frames)\n",
			       shake5(buf[2], okw, nn), shake5(buf[3], okw, nn), shake5(buf[8], okw, nn), shake5(buf[0], okw, nn));
		if (learn) {
			const cf_learnt_notes *e = &gaze.learn.est;
			if (e->has)
				printf("learn: final estimate offset %.4f side %.4f open %.4f vis %.4f (%s), %.4f below the "
				       "baseline (%.0f s of reading)\n",
				       e->offset, e->side, e->open, e->vis_offset, e->has_vis ? "given" : "none",
				       p.baseline - e->offset, e->seconds);
			else
				printf("learn: nothing learnt\n");
		}
		printf("wrote %s/{corrected.mov,compare.mov,frames.csv}\n", argv[2]);
		camflip_destroy(f);
		if (f_out)
			camflip_destroy(f_out);
		obs_data_release(settings);
	}
	return 0;
}
