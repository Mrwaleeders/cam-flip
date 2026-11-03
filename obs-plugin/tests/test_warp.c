/*
 * Checks for the pixel code, run in CI (and runnable anywhere with a C compiler):
 *   cc -O2 -Isrc tests/test_warp.c src/warp.c -lm -o test_warp && ./test_warp
 *
 * For every supported camera format:
 *  - mirroring is an exact flip and mirroring twice restores the original,
 *  - pixels outside the eye regions are never changed (no quality loss),
 *  - the eye region content moves up by the requested amount.
 * With an eye outline (aperture), sized like gaze.c sizes it: the pupil moves by at least 85% of
 * typical notes-glance shifts, delivery only grows with the request up to gaze.c's cap, the upper
 * lid moves less than the iris and the lower lid less still, the eye corners stay put, the field
 * changes smoothly from frame to frame (no popping), it is never steep enough to smear, horizontal
 * shifts keep the iris out of the corners, bicubic keeps edges sharper than bilinear, and a 1080p
 * frame is fast.
 */
#include "warp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Test hook in warp.c (deliberately not in the public header). */
int cf_warp_debug_field(const cf_eye *e, float x, float y, float *dx, float *dy, float *max_slope, float *scale);

static int failures = 0;
#define CHECK(cond, ...)                         \
	do {                                     \
		if (!(cond)) {                   \
			printf("FAIL: " __VA_ARGS__); \
			printf("\n");            \
			failures++;              \
			return;                  \
		}                                \
	} while (0)

typedef struct {
	enum cf_format fmt;
	const char *name;
	int planes;
	int bpp_num, bpp_den; /* bytes per pixel in plane 0 */
} fmt_info;

static const fmt_info formats[] = {
	{CF_FMT_NV12, "NV12", 2, 1, 1}, {CF_FMT_I420, "I420", 3, 1, 1}, {CF_FMT_UYVY, "UYVY", 1, 2, 1},
	{CF_FMT_YUY2, "YUY2", 1, 2, 1}, {CF_FMT_BGRA, "BGRA", 1, 4, 1}, {CF_FMT_I444, "I444", 3, 1, 1},
	{CF_FMT_I422, "I422", 3, 1, 1}, {CF_FMT_Y800, "Y800", 1, 1, 1},
};

enum { W = 320, H = 240 };

typedef struct {
	uint8_t *data[4];
	uint32_t linesize[4];
	size_t size[4];
} frame;

static void alloc_frame(const fmt_info *f, frame *fr)
{
	memset(fr, 0, sizeof(*fr));
	fr->linesize[0] = W * f->bpp_num / f->bpp_den + 16; /* padding, like real frames */
	fr->size[0] = (size_t)fr->linesize[0] * H;
	if (f->fmt == CF_FMT_NV12) {
		fr->linesize[1] = W + 8;
		fr->size[1] = (size_t)fr->linesize[1] * (H / 2);
	} else if (f->fmt == CF_FMT_I420 || f->fmt == CF_FMT_I422 || f->fmt == CF_FMT_I444) {
		int cw = f->fmt == CF_FMT_I444 ? W : W / 2;
		int chh = f->fmt == CF_FMT_I420 ? H / 2 : H;
		fr->linesize[1] = fr->linesize[2] = cw + 8;
		fr->size[1] = fr->size[2] = (size_t)(cw + 8) * chh;
	}
	for (int i = 0; i < 4; i++)
		if (fr->size[i]) {
			fr->data[i] = malloc(fr->size[i]);
			for (size_t k = 0; k < fr->size[i]; k++)
				fr->data[i][k] = (uint8_t)(rand() & 0xff);
		}
}

static void copy_frame(const frame *src, frame *dst)
{
	*dst = *src;
	for (int i = 0; i < 4; i++)
		if (src->size[i]) {
			dst->data[i] = malloc(src->size[i]);
			memcpy(dst->data[i], src->data[i], src->size[i]);
		}
}

static void free_frame(frame *fr)
{
	for (int i = 0; i < 4; i++)
		free(fr->data[i]);
}

static void test_mirror(const fmt_info *f)
{
	frame a, b;
	alloc_frame(f, &a);
	copy_frame(&a, &b);
	cf_channel ca[8], cb[8];
	int n = cf_channels(f->fmt, a.data, a.linesize, W, H, ca);
	cf_channels(f->fmt, b.data, b.linesize, W, H, cb);
	CHECK(n > 0, "%s: no channels", f->name);
	cf_mirror_frame(cb, n);
	for (int c = 0; c < n; c++)
		for (int y = 0; y < ca[c].height; y++)
			for (int x = 0; x < ca[c].width; x++) {
				uint8_t va = ca[c].base[(size_t)y * ca[c].linesize + (size_t)x * ca[c].step];
				uint8_t vb = cb[c].base[(size_t)y * cb[c].linesize + (size_t)(ca[c].width - 1 - x) * cb[c].step];
				CHECK(va == vb, "%s: mirror not exact (channel %d, %d,%d)", f->name, c, x, y);
			}
	cf_mirror_frame(cb, n);
	for (int i = 0; i < 4; i++)
		if (a.size[i])
			CHECK(memcmp(a.data[i], b.data[i], a.size[i]) == 0, "%s: mirror twice != original", f->name);
	printf("ok: %s mirror is exact\n", f->name);
	free_frame(&a);
	free_frame(&b);
}

static void test_warp(const fmt_info *f)
{
	frame a, b;
	alloc_frame(f, &a);
	copy_frame(&a, &b);
	cf_channel ca[8], cb[8];
	int n = cf_channels(f->fmt, a.data, a.linesize, W, H, ca);
	cf_channels(f->fmt, b.data, b.linesize, W, H, cb);

	cf_eye eyes[2];
	memset(eyes, 0, sizeof(eyes)); /* no aperture: legacy whole-ellipse shift */
	/* (big enough that a 6 px shift stays under the slope limit, which applies here too) */
	eyes[0].cx = 100, eyes[0].cy = 120, eyes[0].rx = 45, eyes[0].ry = 40, eyes[0].sy = -6;
	eyes[1] = eyes[0];
	eyes[1].cx = 220;
	cf_warp_frame(cb, n, eyes, 2);

	/* Outside the (channel-scaled) ellipses nothing may change. Padding bytes too. */
	for (int c = 0; c < n; c++) {
		for (int y = 0; y < ca[c].height; y++)
			for (int x = 0; x < ca[c].width; x++) {
				int inside = 0;
				for (int e = 0; e < 2; e++) {
					float dx = ((float)x + 0.5f - eyes[e].cx * ca[c].scale_x) / (eyes[e].rx * ca[c].scale_x);
					float dy = ((float)y + 0.5f - eyes[e].cy * ca[c].scale_y) / (eyes[e].ry * ca[c].scale_y);
					if (sqrtf(dx * dx + dy * dy) < 1.0f)
						inside = 1;
				}
				if (inside)
					continue;
				size_t off = (size_t)y * ca[c].linesize + (size_t)x * ca[c].step;
				CHECK(ca[c].base[off] == cb[c].base[off], "%s: pixel outside eyes changed (ch %d, %d,%d)",
				      f->name, c, x, y);
			}
	}
	/* Every byte not belonging to any channel sample (row padding) is untouched as well. */
	if (f->fmt == CF_FMT_NV12 || f->fmt == CF_FMT_I420 || f->fmt == CF_FMT_Y800)
		for (int y = 0; y < H; y++)
			CHECK(memcmp(a.data[0] + (size_t)y * a.linesize[0] + W, b.data[0] + (size_t)y * b.linesize[0] + W,
				     a.linesize[0] - W) == 0,
			      "%s: row padding changed", f->name);

	/* Content at the eye centre moved up by 6 px: new(cx, cy) == old(cx, cy + 6) on the luma channel. */
	const cf_channel *la = &ca[0], *lb = &cb[0];
	int matches = 0, total = 0;
	for (int y = 117; y <= 123; y++)
		for (int x = 96; x <= 104; x++) {
			total++;
			uint8_t now = lb->base[(size_t)y * lb->linesize + (size_t)x * lb->step];
			uint8_t was = la->base[(size_t)(y + 6) * la->linesize + (size_t)x * la->step];
			if (abs((int)now - (int)was) <= 1)
				matches++;
		}
	CHECK(matches >= total * 9 / 10, "%s: eye centre did not move up by 6 px (%d/%d)", f->name, matches, total);
	printf("ok: %s warp moves only the eyes\n", f->name);
	free_frame(&a);
	free_frame(&b);
}

/* ---- Eye outline (aperture) warp ---------------------------------------------------------- */

/* An almond-shaped eye opening of width w and height h, sized exactly like gaze.c's emit() does:
 * rx = max(0.75 w, outline reach + 0.25 w), ry = max(1.6 h, 0.5 w, outline reach + 1.1 h), and the
 * upper lid follows the vertical part of the iris shift by 0.35. */
static cf_eye make_eye(float cx, float cy, float w, float h, float sx, float sy)
{
	cf_eye e;
	memset(&e, 0, sizeof(e));
	e.cx = cx, e.cy = cy;
	e.sx = sx, e.sy = sy;
	e.lid_sx = 0, e.lid_sy = 0.35f * sy;
	const int n = 16;
	e.aperture_count = n;
	float ext_x = 0.5f * w, ext_y = 0.5f * h;
	for (int k = 0; k < n; k++) {
		const float th = 3.14159265f * (float)(k % 8) / 8.0f;
		if (k < 8) { /* upper lid, left corner to right */
			e.aperture[k][0] = cx - w / 2 * cosf(th);
			e.aperture[k][1] = cy - h / 2 * sinf(th);
		} else { /* lower lid, right corner back to left */
			e.aperture[k][0] = cx + w / 2 * cosf(th);
			e.aperture[k][1] = cy + h / 2 * sinf(th);
		}
		ext_x = fmaxf(ext_x, fabsf(e.aperture[k][0] - cx));
		ext_y = fmaxf(ext_y, fabsf(e.aperture[k][1] - cy));
	}
	e.rx = fmaxf(0.75f * w, ext_x + 0.25f * w);
	e.ry = fmaxf(fmaxf(1.6f * h, 0.5f * w), ext_y + 1.1f * h);
	return e;
}

/* The largest vertical shift gaze.c asks for with this eye (AMOUNT_MAX_RY and the reach limit). */
static float gaze_cap(const cf_eye *e, float h)
{
	return fminf(0.38f * e->ry, 0.5f * (e->ry - 0.5f * h));
}

/* Same byte-for-byte "nothing outside the ellipses changes" check, with outlines and sideways shifts. */
static void test_warp_aperture_untouched(const fmt_info *f)
{
	frame a, b;
	alloc_frame(f, &a);
	copy_frame(&a, &b);
	cf_channel ca[8], cb[8];
	int n = cf_channels(f->fmt, a.data, a.linesize, W, H, ca);
	cf_channels(f->fmt, b.data, b.linesize, W, H, cb);
	/* Two normal eyes, one hanging over the frame corner, and degenerate / broken inputs that must
	 * change nothing outside their ellipses (and must not crash). */
	cf_eye eyes[5] = {make_eye(100, 120, 60, 22, -10.8f, -7), make_eye(221.3f, 117.6f, 60, 22, 6, -9),
			  make_eye(6, 5, 50, 18, -5, -8), make_eye(160, 200, 40, 14, 3, -5), make_eye(260, 40, 40, 14, 0, -5)};
	for (int k = 0; k < eyes[3].aperture_count; k++)
		eyes[3].aperture[k][0] = 160, eyes[3].aperture[k][1] = 200; /* collapsed outline */
	eyes[4].aperture[3][1] = NAN;					   /* broken outline */
	const int ne = 5;
	cf_warp_frame(cb, n, eyes, ne);
	int changed = 0;
	for (int c = 0; c < n; c++)
		for (int y = 0; y < ca[c].height; y++)
			for (int x = 0; x < ca[c].width; x++) {
				int inside = 0;
				for (int e = 0; e < ne; e++) {
					float dx = ((float)x + 0.5f - eyes[e].cx * ca[c].scale_x) / (eyes[e].rx * ca[c].scale_x);
					float dy = ((float)y + 0.5f - eyes[e].cy * ca[c].scale_y) / (eyes[e].ry * ca[c].scale_y);
					if (sqrtf(dx * dx + dy * dy) < 1.0f)
						inside = 1;
				}
				size_t off = (size_t)y * ca[c].linesize + (size_t)x * ca[c].step;
				if (inside) {
					changed += ca[c].base[off] != cb[c].base[off];
					continue;
				}
				CHECK(ca[c].base[off] == cb[c].base[off], "%s: aperture warp changed a pixel outside (ch %d, %d,%d)",
				      f->name, c, x, y);
			}
	if (f->fmt == CF_FMT_NV12 || f->fmt == CF_FMT_I420 || f->fmt == CF_FMT_Y800)
		for (int y = 0; y < H; y++)
			CHECK(memcmp(a.data[0] + (size_t)y * a.linesize[0] + W, b.data[0] + (size_t)y * b.linesize[0] + W,
				     a.linesize[0] - W) == 0,
			      "%s: row padding changed", f->name);
	CHECK(changed > 500, "%s: aperture warp changed almost nothing (%d)", f->name, changed);
	printf("ok: %s aperture warp leaves everything outside the eyes byte-identical\n", f->name);
	free_frame(&a);
	free_frame(&b);
}

/* A grey Y800 test image of the eye: a mid-dark iris disc (radius iris_r) with a dark pupil (radius
 * iris_r / 3, darker than PUPIL_T), and dark dots at both corners. */
enum { BG = 170, PUPIL_T = 60 };
static uint8_t *eye_image(const cf_eye *e, float iris_x, float iris_y, float iris_r, int lid_mark)
{
	uint8_t *img = malloc(W * H);
	for (int y = 0; y < H; y++)
		for (int x = 0; x < W; x++) {
			float v = BG;
			/* anti-aliased discs */
			float d = sqrtf(((float)x + 0.5f - iris_x) * ((float)x + 0.5f - iris_x) +
					((float)y + 0.5f - iris_y) * ((float)y + 0.5f - iris_y));
			float cov = iris_r + 0.5f - d;
			cov = cov < 0 ? 0 : (cov > 1 ? 1 : cov);
			v = v * (1 - cov) + 90 * cov;
			cov = iris_r / 3 + 0.5f - d;
			cov = cov < 0 ? 0 : (cov > 1 ? 1 : cov);
			v = v * (1 - cov) + 15 * cov;
			img[y * W + x] = (uint8_t)(v + 0.5f);
		}
	/* corner dots, 3x3, centred on the outline's end points */
	const float cxs[2] = {e->aperture[0][0], e->aperture[8][0]}, cys[2] = {e->aperture[0][1], e->aperture[8][1]};
	for (int k = 0; k < 2; k++)
		for (int y = -1; y <= 1; y++)
			for (int x = -1; x <= 1; x++)
				img[((int)cys[k] + y) * W + (int)cxs[k] + x] = 20;
	/* lash line: a 2-px dark band just above the upper outline, in the middle third of the eye */
	if (lid_mark) {
		const int top = (int)lrintf(e->cy - (e->aperture[4][1] > e->cy ? -1 : 1) * fabsf(e->aperture[4][1] - e->cy));
		for (int x = (int)e->cx - 6; x <= (int)e->cx + 6; x++)
			for (int y = top - 3; y <= top - 2; y++)
				img[y * W + x] = 30;
	}
	return img;
}

/* Darkness-weighted centroid of the pixels darker than thr in a window. */
static void dark_centroid(const uint8_t *img, int x0, int y0, int x1, int y1, int thr, float *mx, float *my)
{
	double sw = 0, sx = 0, sy = 0;
	for (int y = y0; y <= y1; y++)
		for (int x = x0; x <= x1; x++) {
			const double w = thr - img[y * W + x];
			if (w <= 0)
				continue;
			sw += w;
			sx += w * x;
			sy += w * y;
		}
	*mx = sw > 0 ? (float)(sx / sw) : 0;
	*my = sw > 0 ? (float)(sy / sw) : 0;
}

/* Darkness-weighted centroid of a window. */
static void centroid(const uint8_t *img, int x0, int y0, int x1, int y1, float *mx, float *my)
{
	double sw = 0, sx = 0, sy = 0;
	for (int y = y0; y <= y1; y++)
		for (int x = x0; x <= x1; x++) {
			double w = BG - img[y * W + x];
			if (w < 0)
				w = 0;
			sw += w;
			sx += w * x;
			sy += w * y;
		}
	*mx = sw > 0 ? (float)(sx / sw) : 0;
	*my = sw > 0 ? (float)(sy / sw) : 0;
}

static void warp_y800(uint8_t *img, const cf_eye *e)
{
	uint8_t *d[4] = {img};
	uint32_t ls[4] = {W};
	cf_channel ch[8];
	int n = cf_channels(CF_FMT_Y800, d, ls, W, H, ch);
	cf_warp_frame(ch, n, e, 1);
}

/* Pupil displacement along the requested shift, for an eye of w x h at (100, 120) whose pupil sits
 * 0.05 w low (looking at notes under the camera), as gaze.c sizes it. */
static float pupil_moved(float w, float h, float sx, float sy, cf_eye *eo, uint8_t **imgo, uint8_t **outo)
{
	cf_eye e = make_eye(100, 120, w, h, sx, sy);
	uint8_t *img = eye_image(&e, 100, 120 + 0.05f * w, 0.21f * w, 0), *out = malloc(W * H);
	memcpy(out, img, W * H);
	warp_y800(out, &e);
	float ax, ay, bx, by;
	/* window inside the opening, clear of the corner dots */
	const int x0 = (int)(100 - 0.3f * w), x1 = (int)(100 + 0.3f * w), y0 = (int)(120 - 0.6f * h), y1 = (int)(120 + 0.8f * h);
	dark_centroid(img, x0, y0, x1, y1, PUPIL_T, &ax, &ay);
	dark_centroid(out, x0, y0, x1, y1, PUPIL_T, &bx, &by);
	const float want = sqrtf(sx * sx + sy * sy);
	if (eo) {
		*eo = e, *imgo = img, *outo = out;
	} else {
		free(img);
		free(out);
	}
	return ((bx - ax) * sx + (by - ay) * sy) / want;
}

static void test_aperture_motion(const char *label, float w, float h, float sx, float sy, float min_frac)
{
	cf_eye e;
	uint8_t *img, *out;
	const float want = sqrtf(sx * sx + sy * sy), got = pupil_moved(w, h, sx, sy, &e, &img, &out);
	printf("    %s: pupil moved %.2f of %.2f px (%.0f%%)\n", label, got, want, 100 * got / want);
	CHECK(got >= min_frac * want && got <= want * 1.05f, "%s: pupil moved %.2f of %.2f px", label, got, want);

	/* Corners: the dots must stay put. */
	float ax, ay, bx, by;
	for (int k = 0; k < 2; k++) {
		const int kx = (int)e.aperture[k * 8][0], ky = (int)e.aperture[k * 8][1];
		centroid(img, kx - 3, ky - 3, kx + 3, ky + 3, &ax, &ay);
		centroid(out, kx - 3, ky - 3, kx + 3, ky + 3, &bx, &by);
		const float cm = sqrtf((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
		printf("    %s: corner %d moved %.3f px\n", label, k, cm);
		CHECK(cm < 0.3f, "%s: eye corner %d moved %.2f px", label, k, cm);
	}

	/* The iris is not dragged into the corners: near each corner the image barely changes. */
	int worst = 0;
	for (int k = 0; k < 2; k++) {
		const float kx = e.aperture[k * 8][0], ky = e.aperture[k * 8][1];
		for (int y = (int)ky - 8; y <= (int)ky + 8; y++)
			for (int x = (int)kx - 8; x <= (int)kx + 8; x++) {
				if (((float)x + 0.5f - kx) * ((float)x + 0.5f - kx) + ((float)y + 0.5f - ky) * ((float)y + 0.5f - ky) >
				    64.0f)
					continue;
				int dlt = abs((int)out[y * W + x] - (int)img[y * W + x]);
				if (abs(x - (int)kx) <= 2 && abs(y - (int)ky) <= 2)
					continue; /* the dot itself (sub-pixel resampling of its edges) */
				worst = dlt > worst ? dlt : worst;
			}
	}
	CHECK(worst < 40, "%s: iris dragged into a corner (change %d)", label, worst);
	printf("ok: %s pupil moves, corners stay put\n", label);
	free(img);
	free(out);
}

/* Up to gaze.c's own cap the delivered shift only grows with the request (never collapses), and
 * at least 85% of it is delivered up to about 0.27 x the opening height (0.1 eye widths). */
static void test_aperture_delivery_range(float w, float h)
{
	cf_eye e = make_eye(100, 120, w, h, 0, 0);
	const float cap = gaze_cap(&e, h);
	float prev = 0, last85 = 0;
	for (float s = 0.5f; s <= cap + 1e-3f; s += 0.5f) {
		const float got = pupil_moved(w, h, 0, -s, NULL, NULL, NULL);
		CHECK(got >= prev - 0.05f, "%.0fx%.0f eye: asking %.1f px delivers %.2f, less than %.2f for %.1f px", w, h, s, got,
		      prev, s - 0.5f);
		if (got >= 0.85f * s && last85 == s - 0.5f)
			last85 = s;
		prev = got;
	}
	printf("    %.0fx%.0f eye: >= 85%% delivered up to %.1f px (%.2f h), %.2f px delivered at gaze's cap %.1f px\n", w, h,
	       last85, last85 / h, prev, cap);
	CHECK(last85 >= 0.25f * h, "%.0fx%.0f eye: 85%% delivery only up to %.1f px", w, h, last85);
	printf("ok: %.0fx%.0f eye delivery grows with the request\n", w, h);
}

/* Lash line above the upper outline and a line below the lower one: how far each moves. */
static void lid_motion(float sy, float *up, float *low)
{
	cf_eye e = make_eye(100, 120, 60, 22, 0, sy);
	uint8_t *img = eye_image(&e, 100, 200, 0, 1), *out = malloc(W * H); /* no iris: lid lines only */
	const int lo = 120 + 11 + 2;					     /* 2 px below the lower outline */
	for (int x = 94; x <= 106; x++)
		img[lo * W + x] = 30;
	memcpy(out, img, W * H);
	warp_y800(out, &e);
	float ax, ay, bx, by;
	centroid(img, 92, 100, 108, 112, &ax, &ay);
	centroid(out, 92, 100, 108, 112, &bx, &by);
	*up = by - ay;
	centroid(img, 92, lo - 4, 108, lo + 4, &ax, &ay);
	centroid(out, 92, lo - 4, 108, lo + 4, &bx, &by);
	*low = by - ay;
	free(img);
	free(out);
}

static void test_aperture_lid(void)
{
	/* A typical notes glance (3.5 px in a 60 x 22 eye) and a big one. The upper lid moves clearly
	 * less than the iris (gaze.c asks for 0.35 of it; the iris layer's soft edge adds some), the
	 * lower lid even less. */
	const float sys[2] = {-3.5f, -6}, up_max[2] = {0.55f, 0.75f}, low_max[2] = {0.4f, 0.55f};
	for (int k = 0; k < 2; k++) {
		float up, low;
		lid_motion(sys[k], &up, &low);
		printf("    iris shift %.1f (lid shift %.2f): upper lid moved %.2f px, lower lid %.2f px\n", sys[k], 0.35f * sys[k],
		       up, low);
		CHECK(up <= 0.25f * sys[k] && up >= up_max[k] * sys[k], "upper lid moved %.2f px for %.1f", up, sys[k]);
		CHECK(low <= 0.0f && low >= low_max[k] * sys[k], "lower lid moved %.2f px for %.1f", low, sys[k]);
	}
	printf("ok: upper lid lifts by less than the iris, lower lid less still\n");
}

/* No popping: as the shift (or the eye position) changes in small steps from frame to frame,
 * the field changes in equally small steps (no quantised mask, blur radius or grid). */
static void test_temporal_smoothness(void)
{
	enum { STEPS = 120, NP = 9 };
	float worst = 0, mean = 0;
	float prev[NP][NP][2];
	for (int s = 0; s <= STEPS; s++) {
		const float t = (float)s / STEPS;
		/* a glance building up with reading damping sweeping sideways, eye drifting by 2.4 px */
		cf_eye e = make_eye(100.3f + 2.4f * t, 120.2f + 1.1f * t, 70, 24, -4 + 8 * t, -5 * t);
		float m = 0;
		for (int j = 0; j < NP; j++)
			for (int i = 0; i < NP; i++) {
				float dx, dy, sl, sc;
				cf_warp_debug_field(&e, e.cx + (i - NP / 2) * 4.4f, e.cy + (j - NP / 2) * 2.6f, &dx, &dy, &sl, &sc);
				if (s > 0)
					m = fmaxf(m, fmaxf(fabsf(dx - prev[j][i][0]), fabsf(dy - prev[j][i][1])));
				prev[j][i][0] = dx, prev[j][i][1] = dy;
			}
		if (s > 0) {
			worst = fmaxf(worst, m);
			mean += m / STEPS;
		}
	}
	printf("    field change per frame: worst %.3f px, mean %.3f px\n", worst, mean);
	CHECK(worst < 2.0f * mean && worst < 0.12f, "field jumps by %.3f px in one step (mean %.3f)", worst, mean);
	/* Head still, glance steady, eye sliding 0.02 px per frame: the field just follows it. */
	worst = 0;
	for (int s = 0; s <= 100; s++) {
		cf_eye e = make_eye(100.3f + 0.02f * s, 120.2f + 0.013f * s, 76.5f, 26.5f, 1.5f, -3.5f);
		for (int j = 0; j < NP; j++)
			for (int i = 0; i < NP; i++) {
				float dx, dy, sl, sc;
				cf_warp_debug_field(&e, e.cx + (i - NP / 2) * 4.4f, e.cy + (j - NP / 2) * 2.6f, &dx, &dy, &sl, &sc);
				if (s > 0)
					worst = fmaxf(worst, fmaxf(fabsf(dx - prev[j][i][0]), fabsf(dy - prev[j][i][1])));
				prev[j][i][0] = dx, prev[j][i][1] = dy;
			}
	}
	printf("    eye sliding 0.02 px per frame: field (relative to the eye) changes by at most %.4f px\n", worst);
	CHECK(worst < 0.01f, "field jitters by %.3f px as the eye slides", worst);
	printf("ok: the field changes smoothly frame to frame\n");
}

/* The field's slope, from the internal measurement and independently by finite differences. */
static void test_aperture_slope(void)
{
	const struct {
		float w, h, sx, sy, roll;
	} cases[] = {
		{60, 22, 0, -3, 0},	   {60, 22, 0, -6, 0},	      {60, 22, 0, -9.2f, 0},   {60, 22, -10.8f, 0, 0},
		{60, 22, 10.8f, -6, 0},   {100, 36, 0, -9, 0},	      {100, 36, -18, 0, 0},    {100, 36, -18, -15, 0},
		{60, 22, 0, -6, 0.3f},	   {40, 10, 0, -5, 0},	      {30, 12, 3, -4, -0.25f}, {100, 20, 0, -8, 0},
	};
	float worst = 0;
	for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
		cf_eye e = make_eye(500, 400, cases[c].w, cases[c].h, cases[c].sx, cases[c].sy);
		/* roll: rotate the outline about the centre (a tilted head) */
		const float cr = cosf(cases[c].roll), sr = sinf(cases[c].roll);
		for (int k = 0; k < e.aperture_count; k++) {
			const float x = e.aperture[k][0] - e.cx, y = e.aperture[k][1] - e.cy;
			e.aperture[k][0] = e.cx + x * cr - y * sr;
			e.aperture[k][1] = e.cy + x * sr + y * cr;
		}
		float dx, dy, slope, scale;
		CHECK(cf_warp_debug_field(&e, e.cx, e.cy, &dx, &dy, &slope, &scale), "slope case %zu: no field", c);
		float fd = 0; /* independent finite differences on a 1 px lattice (pixel centres) */
		static float gx[260][260], gy[260][260];
		const int x0 = (int)(e.cx - e.rx) - 1, y0 = (int)(e.cy - e.ry) - 1;
		const int nx = (int)(2 * e.rx) + 3, ny = (int)(2 * e.ry) + 3;
		for (int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++) {
				float s1, s2;
				cf_warp_debug_field(&e, (float)(x0 + i) + 0.5f, (float)(y0 + j) + 0.5f, &gx[j][i], &gy[j][i], &s1, &s2);
			}
		for (int j = 1; j < ny - 1; j++)
			for (int i = 1; i < nx - 1; i++) {
				fd = fmaxf(fd, 0.5f * fabsf(gx[j][i + 1] - gx[j][i - 1]));
				fd = fmaxf(fd, 0.5f * fabsf(gy[j][i + 1] - gy[j][i - 1]));
				fd = fmaxf(fd, 0.5f * fabsf(gx[j + 1][i] - gx[j - 1][i]));
				fd = fmaxf(fd, 0.5f * fabsf(gy[j + 1][i] - gy[j - 1][i]));
			}
		printf("    eye %3.0fx%-3.0f shift (%5.1f,%5.1f) roll %5.2f: centre moves (%5.2f,%5.2f), max slope %.3f "
		       "(measured %.3f), limit scale %.2f\n",
		       cases[c].w, cases[c].h, cases[c].sx, cases[c].sy, cases[c].roll, dx, dy, slope, fd, scale);
		CHECK(slope < 0.45f && fd < 0.45f, "slope case %zu: displacement slope %.3f / %.3f", c, slope, fd);
		worst = fmaxf(worst, fmaxf(slope, fd));
	}
	printf("ok: displacement slope stays below 0.45 (worst %.3f)\n", worst);
}

/* Catmull-Rom keeps a soft edge sharper than bilinear for a fractional shift. */
static void test_bicubic_sharper(void)
{
	/* Legacy eye: within 0.35 of the centre the shift is exactly sy, here 3.5 px. */
	cf_eye e;
	memset(&e, 0, sizeof(e));
	e.cx = 160, e.cy = 120, e.rx = 60, e.ry = 60, e.sy = -3.5f;
	uint8_t *img = malloc(W * H), *out = malloc(W * H);
	const float edge = 121.3f, soft = 0.9f; /* edge position and width */
	for (int y = 0; y < H; y++)
		for (int x = 0; x < W; x++)
			img[y * W + x] = (uint8_t)(40.5f + 180.0f / (1.0f + expf(-((float)y + 0.5f - edge) / soft)));
	memcpy(out, img, W * H);
	warp_y800(out, &e);
	/* Ideal result: the same edge 3.5 px higher. Bilinear reference: mix of rows y+3 and y+4. */
	double err_cubic = 0, err_lin = 0;
	int step_cubic = 0, step_lin = 0;
	for (int y = 108; y <= 126; y++) {
		const float ideal = 40.5f + 180.0f / (1.0f + expf(-((float)y + 0.5f - (edge - 3.5f)) / soft));
		const float lin = 0.5f * img[(y + 3) * W + 160] + 0.5f * img[(y + 4) * W + 160];
		err_cubic += fabsf(out[y * W + 160] - ideal);
		err_lin += fabsf(lin - ideal);
		step_cubic = abs(out[(y + 1) * W + 160] - out[y * W + 160]) > step_cubic ? abs(out[(y + 1) * W + 160] - out[y * W + 160])
											   : step_cubic;
		const float lin1 = 0.5f * img[(y + 4) * W + 160] + 0.5f * img[(y + 5) * W + 160];
		step_lin = (int)fabsf(lin1 - lin) > step_lin ? (int)fabsf(lin1 - lin) : step_lin;
	}
	printf("    edge after a 3.5 px shift: steepest step bicubic %d vs bilinear %d, error vs ideal %.0f vs %.0f\n",
	       step_cubic, step_lin, err_cubic, err_lin);
	CHECK(step_cubic > step_lin && err_cubic < 0.7 * err_lin, "bicubic not sharper than bilinear");
	printf("ok: bicubic keeps edges sharper than bilinear\n");
	free(img);
	free(out);
}

static double now_us(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static void test_timing(void)
{
	enum { FW = 1920, FH = 1080, REPS = 300 };
	uint8_t *y = malloc((size_t)FW * FH), *uv = malloc((size_t)FW * FH / 2);
	for (size_t k = 0; k < (size_t)FW * FH; k++)
		y[k] = (uint8_t)(rand() & 0xff);
	for (size_t k = 0; k < (size_t)FW * FH / 2; k++)
		uv[k] = (uint8_t)(rand() & 0xff);
	uint8_t *d[4] = {y, uv};
	uint32_t ls[4] = {FW, FW};
	cf_channel ch[8];
	int n = cf_channels(CF_FMT_NV12, d, ls, FW, FH, ch);
	/* A MacBook camera face at arm's length: eyes about 70 px wide and 24 px open, a notes glance. */
	cf_eye eyes[2] = {make_eye(820.3f, 500.6f, 70, 24, 2, -3.5f), make_eye(1100.7f, 505.2f, 70, 24, 2, -3.5f)};
	cf_eye legacy[2];
	memcpy(legacy, eyes, sizeof(eyes));
	legacy[0].aperture_count = legacy[1].aperture_count = 0;
	for (int k = 0; k < 20; k++)
		cf_warp_frame(ch, n, eyes, 2); /* warm up */
	double t0 = now_us();
	for (int k = 0; k < REPS; k++)
		cf_warp_frame(ch, n, eyes, 2);
	double t1 = now_us();
	for (int k = 0; k < REPS; k++)
		cf_warp_frame(ch, n, legacy, 2);
	double t2 = now_us();
	printf("timing: 1080p NV12, two 70x24 eyes (ellipse %.0fx%.0f): %.0f us/frame with outline (bicubic luma), "
	       "%.0f us/frame legacy\n",
	       2 * eyes[0].rx, 2 * eyes[0].ry, (t1 - t0) / REPS, (t2 - t1) / REPS);
	free(y);
	free(uv);
}

static void test_extract_luma(void)
{
	const fmt_info *f = &formats[2]; /* UYVY */
	frame a;
	alloc_frame(f, &a);
	cf_channel c[8];
	cf_channels(f->fmt, a.data, a.linesize, W, H, c);
	uint8_t *grey = malloc(W * H);
	cf_extract_luma(&c[0], grey);
	for (int y = 0; y < H; y++)
		for (int x = 0; x < W; x++)
			CHECK(grey[y * W + x] == a.data[0][(size_t)y * a.linesize[0] + (size_t)x * 2 + 1], "UYVY luma extract");
	printf("ok: UYVY luma extraction\n");
	free(grey);
	free_frame(&a);
}

int main(void)
{
	srand(42);
	for (size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); i++) {
		test_mirror(&formats[i]);
		test_warp(&formats[i]);
		test_warp_aperture_untouched(&formats[i]);
	}
	/* Pupil delivery for notes glances (0.04-0.07 eye widths) with gaze.c's sizing, including the
	 * sideways reading damping, and for a big glance. */
	test_aperture_motion("small upward shift", 60, 22, 0, -2, 0.85f);
	test_aperture_motion("notes glance", 60, 22, 0, -3.5f, 0.85f);
	test_aperture_motion("notes glance, 70 px eye", 70, 24, 0, -4, 0.85f);
	test_aperture_motion("notes glance, 80 px eye", 80, 28, 0, -4.5f, 0.85f);
	test_aperture_motion("large upward shift", 60, 22, 0, -6, 0.85f);
	test_aperture_motion("sideways reading-scan damping", 60, 22, 4, -3.5f, 0.85f);
	test_aperture_motion("sideways reading-scan damping, left", 60, 22, -4, -3.5f, 0.85f);
	test_aperture_motion("sideways shift (0.3 rx)", 60, 22, -10.8f, 0, 0.5f);
	test_aperture_motion("diagonal shift", 60, 22, -6, -6, 0.75f);
	test_aperture_delivery_range(60, 22);
	test_aperture_delivery_range(70, 24);
	test_aperture_delivery_range(80, 28);
	test_aperture_lid();
	test_temporal_smoothness();
	test_aperture_slope();
	test_bicubic_sharper();
	test_extract_luma();
	test_timing();
	if (failures) {
		printf("%d FAILED\n", failures);
		return 1;
	}
	printf("ALL CHECKS PASSED\n");
	return 0;
}
