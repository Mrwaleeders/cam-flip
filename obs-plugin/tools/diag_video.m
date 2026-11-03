/*
 * Diagnostics: runs Cam Flip's real face tracking (Apple Vision + pupil refinement, exactly as in
 * the plugin) over every frame of a recorded video and writes one CSV row per frame.
 *
 *   diag_video <video.mp4|.mov> <out.csv>
 *
 * Columns: frame, t (s), ms (tracking time), found, full (1 = full face detection, 0 = tracked),
 * conf (Vision landmark confidence), yaw, pitch (deg), face_w (px), image w, h,
 * Vision pupils (lx, ly, rx, ry), refined pupils (lx, ly, rx, ry), then the left eye outline
 * (count, x0, y0, ...) and the right eye outline (count, x0, y0, ...), then the head measures from
 * the landmarks (pose.c): head (1 = measured), side, vert, eyes, tilt. Pixels, origin top left.
 * yaw and pitch are Apple Vision's own angles (coarse; the plugin's fade uses the head measures).
 *
 * The CSV holds only numbers (no image data). tools/diag.sh builds and runs this on a Mac.
 */
#define CAMFLIP_TEST 1
#include "../src/plugin.m"

#import <AVFoundation/AVFoundation.h>

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "usage: diag_video <video> <out.csv>\n");
		return 2;
	}
	@autoreleasepool {
		NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
		AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		NSArray<AVAssetTrack *> *tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
#pragma clang diagnostic pop
		if (tracks.count == 0) {
			fprintf(stderr, "no video track in %s (OBS .mkv files must be remuxed to .mp4 first)\n", argv[1]);
			return 1;
		}
		NSError *err = nil;
		AVAssetReader *reader = [AVAssetReader assetReaderWithAsset:asset error:&err];
		NSDictionary *opts = @{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)};
		AVAssetReaderTrackOutput *out = [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:tracks[0]
										  outputSettings:opts];
		out.alwaysCopiesSampleData = NO;
		[reader addOutput:out];
		if (![reader startReading]) {
			fprintf(stderr, "cannot read video: %s\n", reader.error.localizedDescription.UTF8String);
			return 1;
		}

		obs_data_t *settings = obs_data_create();
		camflip_defaults(settings);
		struct camflip *f = camflip_create(settings, NULL);
		if (f)
			f->test_clock = true;
		FILE *csv = fopen(argv[2], "w");
		if (!f || !csv) {
			fprintf(stderr, "setup failed\n");
			return 1;
		}
		fprintf(csv, "frame,t,ms,found,full,conf,yaw,pitch,face_w,w,h,vlx,vly,vrx,vry,lx,ly,rx,ry,"
			     "nl,left...,nr,right...,head,side,vert,eyes,tilt\n");

		grey_frame g;
		memset(&g, 0, sizeof(g));
		int frame = 0, found_n = 0;
		double t0 = -1;
		double total_ms = 0;
		CMSampleBufferRef sb;
		while ((sb = [out copyNextSampleBuffer])) {
			CVPixelBufferRef pb = CMSampleBufferGetImageBuffer(sb);
			double t = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb));
			if (t0 < 0)
				t0 = t;
			CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
			const int w = (int)CVPixelBufferGetWidthOfPlane(pb, 0), h = (int)CVPixelBufferGetHeightOfPlane(pb, 0);
			const size_t stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
			const uint8_t *src = CVPixelBufferGetBaseAddressOfPlane(pb, 0);
			ensure(&g.data, &g.cap, (size_t)w * h);
			for (int y = 0; y < h; y++)
				memcpy(g.data + (size_t)y * w, src + (size_t)y * stride, (size_t)w);
			CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
			CFRelease(sb);
			g.w = w;
			g.h = h;
			g.t = t - t0;

			cf_face face;
			bool full = false;
			const uint64_t s = now_ns();
			detect_face(f, &g, &face, &full);
			const double ms = (double)(now_ns() - s) / 1e6;
			total_ms += ms;
			found_n += face.found;

			fprintf(csv, "%d,%.6f,%.2f,%d,%d,%.3f,%.2f,%.2f,%.1f,%d,%d", frame, g.t, ms, face.found, full ? 1 : 0,
				f->dbg_conf, face.yaw_deg, face.pitch_deg, face.face_width, w, h);
			fprintf(csv, ",%.3f,%.3f,%.3f,%.3f", f->dbg_raw_pupil[0].x, f->dbg_raw_pupil[0].y,
				f->dbg_raw_pupil[1].x, f->dbg_raw_pupil[1].y);
			fprintf(csv, ",%.3f,%.3f,%.3f,%.3f", face.left_pupil.x, face.left_pupil.y, face.right_pupil.x,
				face.right_pupil.y);
			fprintf(csv, ",%d", face.left_count);
			for (int i = 0; i < face.left_count; i++)
				fprintf(csv, ",%.3f,%.3f", face.left_eye[i].x, face.left_eye[i].y);
			fprintf(csv, ",%d", face.right_count);
			for (int i = 0; i < face.right_count; i++)
				fprintf(csv, ",%.3f,%.3f", face.right_eye[i].x, face.right_eye[i].y);
			fprintf(csv, ",%d,%.5f,%.5f,%.5f,%.5f", face.has_head_geom, face.head.side, face.head.vert,
				face.head.eyes, face.head.tilt);
			fprintf(csv, "\n");
			frame++;
		}
		fclose(csv);
		printf("frames %d, face found in %d, mean tracking time %.1f ms\n", frame, found_n,
		       frame ? total_ms / frame : 0);
		camflip_destroy(f);
		obs_data_release(settings);
	}
	return 0;
}
