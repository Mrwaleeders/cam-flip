#include "warp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static cf_channel make(uint8_t *base, int linesize, int step, int w, int h, int image_w, int image_h, int luma)
{
	cf_channel c;
	c.base = base;
	c.linesize = linesize;
	c.step = step;
	c.width = w;
	c.height = h;
	c.scale_x = (float)w / (float)image_w;
	c.scale_y = (float)h / (float)image_h;
	c.is_luma = luma;
	return c;
}

int cf_channels(enum cf_format f, uint8_t *const d[], const uint32_t ls[], int w, int h, cf_channel o[8])
{
	const int hw = (w + 1) / 2, hh = (h + 1) / 2;
	switch (f) {
	case CF_FMT_I420:
		o[0] = make(d[0], ls[0], 1, w, h, w, h, 1);
		o[1] = make(d[1], ls[1], 1, hw, hh, w, h, 0);
		o[2] = make(d[2], ls[2], 1, hw, hh, w, h, 0);
		return 3;
	case CF_FMT_NV12:
		o[0] = make(d[0], ls[0], 1, w, h, w, h, 1);
		o[1] = make(d[1], ls[1], 2, hw, hh, w, h, 0);
		o[2] = make(d[1] + 1, ls[1], 2, hw, hh, w, h, 0);
		return 3;
	case CF_FMT_I444:
		for (int i = 0; i < 3; i++)
			o[i] = make(d[i], ls[i], 1, w, h, w, h, i == 0);
		return 3;
	case CF_FMT_I422:
		o[0] = make(d[0], ls[0], 1, w, h, w, h, 1);
		o[1] = make(d[1], ls[1], 1, hw, h, w, h, 0);
		o[2] = make(d[2], ls[2], 1, hw, h, w, h, 0);
		return 3;
	case CF_FMT_Y800:
		o[0] = make(d[0], ls[0], 1, w, h, w, h, 1);
		return 1;
	case CF_FMT_YUY2:
	case CF_FMT_YVYU:
		o[0] = make(d[0], ls[0], 2, w, h, w, h, 1);
		o[1] = make(d[0] + 1, ls[0], 4, w / 2, h, w, h, 0);
		o[2] = make(d[0] + 3, ls[0], 4, w / 2, h, w, h, 0);
		return 3;
	case CF_FMT_UYVY:
		o[0] = make(d[0] + 1, ls[0], 2, w, h, w, h, 1);
		o[1] = make(d[0], ls[0], 4, w / 2, h, w, h, 0);
		o[2] = make(d[0] + 2, ls[0], 4, w / 2, h, w, h, 0);
		return 3;
	case CF_FMT_RGBA:
	case CF_FMT_BGRA:
	case CF_FMT_BGRX:
		for (int i = 0; i < 4; i++)
			o[i] = make(d[0] + i, ls[0], 4, w, h, w, h, 1);
		return 4;
	default:
		return 0;
	}
}

static inline float smoothstep(float e0, float e1, float x)
{
	float t = (x - e0) / (e1 - e0);
	t = t < 0 ? 0 : (t > 1 ? 1 : t);
	return t * t * (3 - 2 * t);
}

static inline int clampi(int v, int lo, int hi)
{
	return v < lo ? lo : (v > hi ? hi : v);
}

static void warp_one(const cf_channel *ch, const cf_eye *e)
{
	/* Eye geometry in this channel's sample grid. */
	const float cx = e->cx * ch->scale_x, cy = e->cy * ch->scale_y;
	const float rx = e->rx * ch->scale_x, ry = e->ry * ch->scale_y;
	const float sx = e->sx * ch->scale_x, sy = e->sy * ch->scale_y;
	if (rx < 1.0f || ry < 1.0f || (fabsf(sx) < 0.01f && fabsf(sy) < 0.01f))
		return;

	const int x0 = clampi((int)floorf(cx - rx), 0, ch->width - 1);
	const int x1 = clampi((int)ceilf(cx + rx), 0, ch->width - 1);
	const int y0 = clampi((int)floorf(cy - ry), 0, ch->height - 1);
	const int y1 = clampi((int)ceilf(cy + ry), 0, ch->height - 1);

	/* Snapshot of the source area (destination box grown by the displacement). */
	const int pad = (int)ceilf(fmaxf(fabsf(sx), fabsf(sy))) + 2;
	const int wx0 = clampi(x0 - pad, 0, ch->width - 1), wx1 = clampi(x1 + pad, 0, ch->width - 1);
	const int wy0 = clampi(y0 - pad, 0, ch->height - 1), wy1 = clampi(y1 + pad, 0, ch->height - 1);
	const int ww = wx1 - wx0 + 1, wh = wy1 - wy0 + 1;
	uint8_t *snap = (uint8_t *)malloc((size_t)ww * (size_t)wh);
	if (!snap)
		return;
	for (int y = 0; y < wh; y++) {
		const uint8_t *row = ch->base + (size_t)(wy0 + y) * ch->linesize + (size_t)wx0 * ch->step;
		for (int x = 0; x < ww; x++)
			snap[y * ww + x] = row[x * ch->step];
	}

	for (int y = y0; y <= y1; y++) {
		uint8_t *row = ch->base + (size_t)y * ch->linesize;
		const float py = (float)y + 0.5f;
		const float dy = (py - cy) / ry;
		for (int x = x0; x <= x1; x++) {
			const float px = (float)x + 0.5f;
			const float dx = (px - cx) / rx;
			const float len = sqrtf(dx * dx + dy * dy);
			if (len >= 1.0f)
				continue;
			const float w = 1.0f - smoothstep(0.35f, 1.0f, len);
			/* Sample position, in snapshot coordinates. */
			float fx = px - sx * w - 0.5f - (float)wx0;
			float fy = py - sy * w - 0.5f - (float)wy0;
			fx = fx < 0 ? 0 : (fx > ww - 1 ? (float)(ww - 1) : fx);
			fy = fy < 0 ? 0 : (fy > wh - 1 ? (float)(wh - 1) : fy);
			const int ix = (int)fx, iy = (int)fy;
			const int ix1 = ix + 1 < ww ? ix + 1 : ix, iy1 = iy + 1 < wh ? iy + 1 : iy;
			const float ax = fx - (float)ix, ay = fy - (float)iy;
			const float top = snap[iy * ww + ix] * (1 - ax) + snap[iy * ww + ix1] * ax;
			const float bot = snap[iy1 * ww + ix] * (1 - ax) + snap[iy1 * ww + ix1] * ax;
			const float v = top * (1 - ay) + bot * ay;
			row[(size_t)x * ch->step] = (uint8_t)(v + 0.5f);
		}
	}
	free(snap);
}

void cf_warp_channel(const cf_channel *ch, const cf_eye *eyes, int eye_count)
{
	for (int i = 0; i < eye_count; i++)
		warp_one(ch, &eyes[i]);
}

void cf_mirror_channel(const cf_channel *ch)
{
	for (int y = 0; y < ch->height; y++) {
		uint8_t *row = ch->base + (size_t)y * ch->linesize;
		for (int a = 0, b = ch->width - 1; a < b; a++, b--) {
			uint8_t *pa = row + (size_t)a * ch->step, *pb = row + (size_t)b * ch->step;
			uint8_t t = *pa;
			*pa = *pb;
			*pb = t;
		}
	}
}

void cf_warp_frame(const cf_channel *channels, int count, const cf_eye *eyes, int eye_count)
{
	for (int i = 0; i < count; i++)
		cf_warp_channel(&channels[i], eyes, eye_count);
}

void cf_mirror_frame(const cf_channel *channels, int count)
{
	/* Contiguous 1-byte channels can be flipped faster, but this is already well under a millisecond. */
	for (int i = 0; i < count; i++)
		cf_mirror_channel(&channels[i]);
}

void cf_extract_luma(const cf_channel *luma, uint8_t *out)
{
	for (int y = 0; y < luma->height; y++) {
		const uint8_t *row = luma->base + (size_t)y * luma->linesize;
		uint8_t *dst = out + (size_t)y * luma->width;
		for (int x = 0; x < luma->width; x++)
			dst[x] = row[(size_t)x * luma->step];
	}
}
