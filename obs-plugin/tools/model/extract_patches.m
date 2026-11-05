/*
 * Turns a capture session (capture_pairs.m) into training data for the personal eye model: runs Cam
 * Flip's own face tracking on every frame and cuts each eye out as the model sees it (ml_eye.h: a
 * CF_ML_W x CF_ML_H brightness patch aligned on the eye corners, the eye on the right of the image
 * mirrored).
 *
 *   tools/model/extract.sh <capture-folder>
 *
 * Writes into the capture folder:
 *   patches.u8   one CF_ML_H x CF_ML_W byte patch per row of eyes.csv, in order
 *   eyes.csv     per patch: frame, t, eye (0 = left in the image), flip, the corners (image px), the
 *                refined and Vision pupils (patch px), openness (outline height / width), face width
 * Nothing leaves this Mac.
 */
#define CAMFLIP_TEST 1
#include "../../src/plugin.m"
#include "../../src/ml_eye.h"

#import <AVFoundation/AVFoundation.h>

static void corners(const cf_pt *p, int n, cf_pt *a, cf_pt *b)
{
	float best = -1;
	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++) {
			const float d = (p[i].x - p[j].x) * (p[i].x - p[j].x) + (p[i].y - p[j].y) * (p[i].y - p[j].y);
			if (d > best) {
				best = d;
				*a = p[i];
				*b = p[j];
			}
		}
}

static float openness(const cf_pt *p, int n, cf_pt a, cf_pt b)
{
	/* spread of the outline across the corner line, over its length */
	const float dx = b.x - a.x, dy = b.y - a.y, len = sqrtf(dx * dx + dy * dy);
	if (len < 1)
		return 0;
	float lo = 0, hi = 0;
	for (int i = 0; i < n; i++) {
		const float d = ((p[i].x - a.x) * -dy + (p[i].y - a.y) * dx) / len;
		lo = fminf(lo, d);
		hi = fmaxf(hi, d);
	}
	return (hi - lo) / len;
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: extract_patches <capture-folder>\n");
		return 2;
	}
	@autoreleasepool {
		NSString *dir = [NSString stringWithUTF8String:argv[1]];
		AVURLAsset *asset =
			[AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"video.mov"]]
					    options:nil];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		NSArray<AVAssetTrack *> *tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
#pragma clang diagnostic pop
		if (!tracks.count) {
			fprintf(stderr, "no video in %s\n", argv[1]);
			return 1;
		}
		NSError *err = nil;
		AVAssetReader *rd = [AVAssetReader assetReaderWithAsset:asset error:&err];
		AVAssetReaderTrackOutput *o = [AVAssetReaderTrackOutput
			assetReaderTrackOutputWithTrack:tracks[0]
					 outputSettings:@{(id)kCVPixelBufferPixelFormatTypeKey:
								@(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)}];
		[rd addOutput:o];
		[rd startReading];

		obs_data_t *settings = obs_data_create();
		camflip_defaults(settings);
		struct camflip *f = camflip_create(settings, NULL);
		f->test_clock = true;

		FILE *pf = fopen([dir stringByAppendingPathComponent:@"patches.u8"].UTF8String, "wb");
		FILE *csv = fopen([dir stringByAppendingPathComponent:@"eyes.csv"].UTF8String, "w");
		fprintf(csv, "frame,t,eye,flip,ax,ay,bx,by,pupil_x,pupil_y,vis_x,vis_y,open,face_w\n");
		grey_frame g = {0};
		static float patch[CF_ML_W * CF_ML_H];
		static uint8_t bytes[CF_ML_W * CF_ML_H];
		int frame = 0, rows = 0, found = 0;
		double t0 = -1;
		CMSampleBufferRef sb;
		while ((sb = [o copyNextSampleBuffer])) {
			@autoreleasepool {
				CVPixelBufferRef pb = CMSampleBufferGetImageBuffer(sb);
				const double ts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb));
				if (t0 < 0)
					t0 = ts;
				const double t = ts - t0;
				CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
				const int w = (int)CVPixelBufferGetWidthOfPlane(pb, 0), h = (int)CVPixelBufferGetHeightOfPlane(pb, 0);
				const size_t st = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
				uint8_t *py = CVPixelBufferGetBaseAddressOfPlane(pb, 0);
				ensure(&g.data, &g.cap, (size_t)w * h);
				for (int y = 0; y < h; y++)
					memcpy(g.data + (size_t)y * w, py + (size_t)y * st, (size_t)w);
				CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
				CFRelease(sb);
				g.w = w;
				g.h = h;
				g.t = t;
				cf_face face;
				bool full;
				detect_face(f, &g, &face, &full);
				if (face.found && face.left_count >= 4 && face.right_count >= 4) {
					found++;
					cf_channel ch[8];
					uint8_t *planes[1] = {g.data};
					const uint32_t ls[1] = {(uint32_t)w};
					cf_channels(CF_FMT_Y800, planes, ls, w, h, ch);
					const cf_pt *pts[2] = {face.left_eye, face.right_eye};
					const int cnt[2] = {face.left_count, face.right_count};
					const cf_pt pup[2] = {face.left_pupil, face.right_pupil};
					const cf_pt vis[2] = {face.left_pupil_vision, face.right_pupil_vision};
					float mx[2];
					for (int e = 0; e < 2; e++) {
						mx[e] = 0;
						for (int i = 0; i < cnt[e]; i++)
							mx[e] += pts[e][i].x / (float)cnt[e];
					}
					for (int e = 0; e < 2; e++) {
						const int flip = mx[e] > mx[1 - e];
						cf_pt a, b;
						corners(pts[e], cnt[e], &a, &b);
						cf_ml_frame fr;
						cf_ml_frame_from_corners(a.x, a.y, b.x, b.y, flip, &fr);
						cf_ml_crop(&ch[0], &fr, patch);
						for (int i = 0; i < CF_ML_W * CF_ML_H; i++)
							bytes[i] = (uint8_t)lrintf(fminf(255, fmaxf(0, patch[i] * 255)));
						fwrite(bytes, 1, sizeof(bytes), pf);
						float ppx, ppy, vpx = NAN, vpy = NAN;
						cf_ml_to_patch(&fr, pup[e].x, pup[e].y, &ppx, &ppy);
						if (face.has_vision_pupils)
							cf_ml_to_patch(&fr, vis[e].x, vis[e].y, &vpx, &vpy);
						fprintf(csv, "%d,%.4f,%d,%d,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.4f,%.1f\n", frame, t,
							flip, flip, a.x, a.y, b.x, b.y, ppx, ppy, vpx, vpy, openness(pts[e], cnt[e], a, b),
							face.face_width);
						rows++;
					}
				}
				frame++;
				if (frame % 300 == 0)
					printf("  %d frames…\n", frame);
			}
		}
		fclose(pf);
		fclose(csv);
		free(g.data);
		printf("%d frames, face found on %d, %d eye patches written to %s\n", frame, found, rows, argv[1]);
	}
	return 0;
}
