/*
 * Checks for the pixel code, run in CI (and runnable anywhere with a C compiler):
 *   cc -O2 -Isrc tests/test_warp.c src/warp.c -lm -o test_warp && ./test_warp
 *
 * For every supported camera format:
 *  - mirroring is an exact flip and mirroring twice restores the original,
 *  - pixels outside the eye regions are never changed (no quality loss),
 *  - the eye region content moves up by the requested amount.
 */
#include "warp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

	cf_eye eyes[2] = {
		{100, 120, 30, 20, 0, -6},
		{220, 120, 30, 20, 0, -6},
	};
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
	}
	test_extract_luma();
	if (failures) {
		printf("%d FAILED\n", failures);
		return 1;
	}
	printf("ALL CHECKS PASSED\n");
	return 0;
}
