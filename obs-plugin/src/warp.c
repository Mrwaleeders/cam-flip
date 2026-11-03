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

/*
 * Displacement field of one eye. It is computed once per frame, in full-resolution image pixels,
 * and shared by every channel (subsampled chroma planes look it up at their own scale, so the
 * eye outline is automatically scaled per channel).
 *
 * The field is sampled on a grid of nodes cs image pixels apart (1 for small eyes, 2 for a
 * typical 1080p eye, up to 4 for huge ones: the field is smooth, so this costs nothing visible
 * and saves most of the time) and read back with bilinear interpolation. d(p) is how far the content at p has moved: output(p) = input(p - d).
 *
 * With an eye outline (aperture) the field is built in layers:
 *   iris weight: 1 on the part of the opening whose source p - S is itself inside the opening,
 *                so the shifted eyeball never pulls in lid skin from below or from a corner;
 *   lid weight:  1 on a band above the upper outline (the upper lid and its lashes);
 * both are smoothed with a double box blur whose radius grows with the shift (that bounds their
 * slope), renormalised so the middle of the iris gets the full shift, and combined as
 * S * iris + L * lid * (1 - iris). Then the corners and the outer ellipse are pinned: with
 * "room" = distance to the nearest pinned place, the displacement is softly limited to
 * CF_BUILD_SLOPE * room, which brings it to exactly zero there along the gentlest possible
 * ramp, however large the shift. Last, the field's steepest slope is measured; if it is still
 * above CF_MAX_SLOPE (only for extreme shifts in tiny eyes) the whole field is scaled down, so
 * the warp can never visibly stretch, smear or fold anything.
 */
#define CF_MAX_SLOPE 0.4f   /* hard limit, a safety net */
#define CF_BUILD_SLOPE 0.3f /* what the construction aims for */

typedef struct {
	float ox, oy;  /* image position of node (0, 0) (pixel-centre coordinates) */
	int cs;        /* node spacing, image pixels */
	int gw, gh;    /* nodes per row, rows */
	float *dx, *dy;
	float maxd;      /* largest |d| component, image pixels */
	float max_slope; /* steepest |partial derivative| of the final field */
	float scale;     /* < 1 when the slope limit kicked in */
	void *mem;
	size_t snap_size; /* bytes available at snap, enough for any channel no finer than the image */
	uint8_t *snap;
} cf_field;

/* Running box sum of radius r along each row (samples beyond the ends count as 0). */
static void box_rows(const float *in, float *out, int w, int h, int r)
{
	const float inv = 1.0f / (float)(2 * r + 1);
	for (int j = 0; j < h; j++) {
		const float *a = in + (size_t)j * w;
		float *b = out + (size_t)j * w;
		float s = 0;
		for (int i = 0; i <= r && i < w; i++)
			s += a[i];
		/* Split so the steady-state loop has no bounds tests. */
		const int lo = r < w ? r : w;                   /* i < lo: nothing leaves the window yet */
		const int hi = w - r - 1 > lo ? w - r - 1 : lo; /* lo <= i < hi: one in, one out */
		int i = 0;
		for (; i < lo; i++) {
			b[i] = s * inv;
			if (i + r + 1 < w)
				s += a[i + r + 1];
		}
		for (; i < hi; i++) {
			b[i] = s * inv;
			s += a[i + r + 1] - a[i - r];
		}
		for (; i < w; i++) {
			b[i] = s * inv;
			if (i + r + 1 < w)
				s += a[i + r + 1];
			s -= a[i - r];
		}
	}
}

/* The same down each column, with running sums for all columns at once so memory is walked
 * row by row (and the inner loops vectorise). */
static void box_cols(const float *in, float *out, int w, int h, int r, float *acc)
{
	const float inv = 1.0f / (float)(2 * r + 1);
	for (int i = 0; i < w; i++)
		acc[i] = 0;
	for (int j = 0; j <= r && j < h; j++)
		for (int i = 0; i < w; i++)
			acc[i] += in[(size_t)j * w + i];
	for (int j = 0; j < h; j++) {
		float *o = out + (size_t)j * w;
		for (int i = 0; i < w; i++)
			o[i] = acc[i] * inv;
		if (j + r + 1 < h) {
			const float *add = in + (size_t)(j + r + 1) * w;
			for (int i = 0; i < w; i++)
				acc[i] += add[i];
		}
		if (j - r >= 0) {
			const float *sub = in + (size_t)(j - r) * w;
			for (int i = 0; i < w; i++)
				acc[i] -= sub[i];
		}
	}
}

/* Two box passes per axis (a smooth, tent-like kernel), in place. The slope of a blurred step
 * of height 1 is at most 1 / (2r + 1): that is the bound the field relies on. */
static void blur2(float *buf, float *tmp, float *acc, int w, int h, int r)
{
	if (r < 1)
		return;
	box_rows(buf, tmp, w, h, r);
	box_rows(tmp, buf, w, h, r);
	box_cols(buf, tmp, w, h, r, acc);
	box_cols(tmp, buf, w, h, r, acc);
}

static void field_free(cf_field *f)
{
	free(f->mem);
	f->mem = NULL;
}

/* One allocation per eye per frame: grid, scratch, and the snapshot used by every channel. */
static int field_alloc(cf_field *f, const cf_eye *e)
{
	memset(f, 0, sizeof(*f));
	if (!(e->rx >= 1.0f && e->ry >= 1.0f) || e->rx > 4096.0f || e->ry > 4096.0f)
		return 0;
	f->cs = clampi((int)(fminf(e->rx, e->ry) / 14.0f), 1, 4); /* about 14+ nodes across the short radius */
	const float cs = (float)f->cs;
	f->ox = floorf(e->cx - e->rx) + 0.5f - cs;
	f->oy = floorf(e->cy - e->ry) + 0.5f - cs;
	f->gw = (int)ceilf((e->cx + e->rx + cs - f->ox) / cs) + 1;
	f->gh = (int)ceilf((e->cy + e->ry + cs - f->oy) / cs) + 1;
	const size_t cells = (size_t)f->gw * (size_t)f->gh;
	/* Snapshot pad: the largest possible displacement plus the bicubic footprint. */
	const float smax = fmaxf(fmaxf(fabsf(e->sx), fabsf(e->sy)), fmaxf(fabsf(e->lid_sx), fabsf(e->lid_sy)));
	const int pad = (int)ceilf(smax) + 4;
	f->snap_size = (size_t)(2 * (int)ceilf(e->rx) + 3 + 2 * pad) * (size_t)(2 * (int)ceilf(e->ry) + 3 + 2 * pad);
	const size_t bytes = cells * 3 * sizeof(float) + (size_t)f->gw * 2 * sizeof(float) + f->snap_size;
	f->mem = malloc(bytes);
	if (!f->mem)
		return 0;
	f->dx = (float *)f->mem;
	f->dy = f->dx + cells;
	f->snap = (uint8_t *)(f->dy + 2 * cells + (size_t)f->gw * 2);
	return 1;
}

static void field_legacy(cf_field *f, const cf_eye *e)
{
	for (int j = 0; j < f->gh; j++) {
		const float dyn = (f->oy + (float)(j * f->cs) - e->cy) / e->ry;
		for (int i = 0; i < f->gw; i++) {
			const float dxn = (f->ox + (float)(i * f->cs) - e->cx) / e->rx;
			const float len = sqrtf(dxn * dxn + dyn * dyn);
			const float w = len >= 1.0f ? 0.0f : 1.0f - smoothstep(0.35f, 1.0f, len);
			f->dx[(size_t)j * f->gw + i] = e->sx * w;
			f->dy[(size_t)j * f->gw + i] = e->sy * w;
		}
	}
}

static int field_aperture(cf_field *f, const cf_eye *e)
{
	const int gw = f->gw, gh = f->gh, n = e->aperture_count;
	const float cs = (float)f->cs;
	const size_t cells = (size_t)gw * gh;
	float *wi = f->dx, *wl = f->dy, *tmp = f->dy + cells;
	float *top = tmp + cells, *bot = top + gw;
	const float(*ap)[2] = e->aperture;

	/* Vertical extent of the opening at every node column (scanline through the outline). */
	float h_open = 0;
	for (int i = 0; i < gw; i++) {
		const float u = f->ox + (float)i * cs;
		float lo = 1e30f, hi = -1e30f;
		for (int k = 0; k < n; k++) {
			const float ax = ap[k][0], ay = ap[k][1];
			const float bx = ap[(k + 1) % n][0], by = ap[(k + 1) % n][1];
			if ((u >= ax && u < bx) || (u >= bx && u < ax)) {
				const float y = ay + (u - ax) * (by - ay) / (bx - ax);
				lo = fminf(lo, y);
				hi = fmaxf(hi, y);
			}
		}
		top[i] = lo;
		bot[i] = hi;
		if (hi > lo)
			h_open = fmaxf(h_open, hi - lo);
	}

	/* Eye corners: the two outline points farthest apart (works for a tilted head too). */
	int c0 = 0, c1 = 1;
	float best = -1;
	for (int a = 0; a < n; a++)
		for (int b = a + 1; b < n; b++) {
			const float ddx = ap[a][0] - ap[b][0], ddy = ap[a][1] - ap[b][1];
			const float d2 = ddx * ddx + ddy * ddy;
			if (d2 > best) {
				best = d2;
				c0 = a;
				c1 = b;
			}
		}
	const float w_open = sqrtf(best);
	if (h_open < 2.0f || w_open < 4.0f)
		return 0;

	/* Layer masks. */
	const int osi = (int)lrintf(e->sx / cs); /* iris source offset, in node columns */
	const float band = 0.6f * h_open;        /* upper lid band height */
	for (int j = 0; j < gh; j++) {
		const float v = f->oy + (float)j * cs;
		const float sv = v - e->sy;
		for (int i = 0; i < gw; i++) {
			const size_t k = (size_t)j * gw + i;
			const int open = top[i] < bot[i];
			const int si = i - osi;
			wi[k] = (open && v >= top[i] && v <= bot[i] && si >= 0 && si < gw && sv >= top[si] && sv <= bot[si])
					? 1.0f
					: 0.0f;
			wl[k] = (open && v < top[i] && v >= top[i] - band) ? 1.0f : 0.0f;
		}
	}

	/* Smooth both layers. The blur radius grows with the shift, so small shifts stay sharply
	 * layered (iris moves, lids barely) and large ones spread onto the lids and skin rather than
	 * being squeezed into the opening, which would smear. */
	const float smax = fmaxf(fabsf(e->sx), fabsf(e->sy));
	const float lmax = fmaxf(fabsf(e->lid_sx), fabsf(e->lid_sy));
	const float rmax = fmaxf(1.0f, 0.5f * w_open);
	const float r_px = fminf(fmaxf(smax / (2.0f * CF_BUILD_SLOPE), 0.12f * h_open), rmax);
	const float rl_px = fminf(fmaxf(lmax / (2.0f * CF_BUILD_SLOPE), 0.15f * h_open), rmax);
	blur2(wi, tmp, top, gw, gh, (int)fmaxf(1.0f, roundf(r_px / cs))); /* top/bot no longer needed */
	blur2(wl, tmp, top, gw, gh, (int)fmaxf(1.0f, roundf(rl_px / cs)));
	float peak = 0, lpeak = 0;
	for (size_t k = 0; k < cells; k++) {
		peak = wi[k] > peak ? wi[k] : peak;
		lpeak = wl[k] > lpeak ? wl[k] : lpeak;
	}
	const float inv_peak = peak > 0.05f ? 1.0f / peak : 0.0f;
	const float inv_lpeak = lpeak > 0.05f ? 1.0f / lpeak : 0.0f;

	/* Combine, and pin the corners and the outer ellipse. */
	const float k0x = ap[c0][0], k0y = ap[c0][1], k1x = ap[c1][0], k1y = ap[c1][1];
	const float rmin = fminf(e->rx, e->ry);
	const float irx2 = 1.0f / (e->rx * e->rx), iry2 = 1.0f / (e->ry * e->ry);
	for (int j = 0; j < gh; j++) {
		const float py = f->oy + (float)j * cs;
		const float dyn = (py - e->cy) / e->ry;
		for (int i = 0; i < gw; i++) {
			const size_t k = (size_t)j * gw + i;
			const float a = fminf(wi[k] * inv_peak, 1.0f);
			const float b = fminf(wl[k] * inv_lpeak, 1.0f) * (1.0f - a);
			const float tx = e->sx * a + e->lid_sx * b, ty = e->sy * a + e->lid_sy * b;
			const float tm = fmaxf(fabsf(tx), fabsf(ty));
			const float px = f->ox + (float)i * cs;
			const float dxn = (px - e->cx) / e->rx;
			const float len2 = dxn * dxn + dyn * dyn;
			if (tm < 1e-6f || len2 >= 1.0f) {
				wi[k] = wl[k] = 0;
				continue;
			}
			/* Distance to the ellipse, about (1 - len) / |grad len| (exact for a circle); the
			 * direction-dependent factor eases to the short radius at the centre, where it
			 * would otherwise jump. */
			const float len = sqrtf(len2);
			const float gl = sqrtf(dxn * dxn * irx2 + dyn * dyn * iry2);
			const float hr = len > 1e-4f ? len / gl : rmin;
			float room = (1.0f - len) * (rmin + (hr - rmin) * smoothstep(0.0f, 0.6f, len));
			/* Corners: d^2 / (d + 6) has slope <= 1 but stays nearly flat for the first pixels,
			 * so the whole corner (not only its exact point) stays put. */
			const float d0 = (px - k0x) * (px - k0x) + (py - k0y) * (py - k0y);
			const float d1 = (px - k1x) * (px - k1x) + (py - k1y) * (py - k1y);
			const float dc = sqrtf(fminf(d0, d1));
			room = fminf(room, dc * dc / (dc + 6.0f));
			/* Soft min(1, x): x / (1 + x^4)^(1/4), whose slope never exceeds 1. */
			const float x = CF_BUILD_SLOPE * room / tm, x2 = x * x;
			const float g = x > 64.0f ? 1.0f : x / sqrtf(sqrtf(1.0f + x2 * x2));
			wi[k] = tx * g;
			wl[k] = ty * g;
		}
	}
	return 1;
}

/* Builds the field for one eye. Returns 0 when the eye needs no work. */
static int field_build(cf_field *f, const cf_eye *e)
{
	/* Never trust the geometry blindly: a NaN would turn into undefined integer conversions. */
	if (!isfinite(e->cx) || !isfinite(e->cy) || !isfinite(e->sx) || !isfinite(e->sy) || fabsf(e->cx) > 1e6f ||
	    fabsf(e->cy) > 1e6f)
		return 0;
	int has_ap = e->aperture_count >= 4 && e->aperture_count <= CF_MAX_APERTURE && isfinite(e->lid_sx) &&
		     isfinite(e->lid_sy);
	for (int k = 0; has_ap && k < e->aperture_count; k++)
		if (!isfinite(e->aperture[k][0]) || !isfinite(e->aperture[k][1]))
			has_ap = 0; /* a broken outline: fall back to the plain shift */
	const float mag = fmaxf(fmaxf(fabsf(e->sx), fabsf(e->sy)),
				has_ap ? fmaxf(fabsf(e->lid_sx), fabsf(e->lid_sy)) : 0.0f);
	if (!(mag >= 0.01f) || !field_alloc(f, e))
		return 0;
	f->scale = 1.0f;
	if (has_ap) {
		if (!field_aperture(f, e)) { /* degenerate outline (closed eye, bad points): leave it alone */
			field_free(f);
			return 0;
		}
	} else {
		field_legacy(f, e);
	}

	/* Steepest slope (differences between neighbouring nodes, which bound the slope of the
	 * bilinear interpolation between them) and largest displacement. */
	const int gw = f->gw, gh = f->gh;
	const float inv2 = 1.0f / (float)f->cs;
	float slope = 0, maxd = 0;
	for (int j = 0; j < gh; j++) {
		const float *rx = f->dx + (size_t)j * gw, *ry = f->dy + (size_t)j * gw;
		for (int i = 0; i < gw; i++) {
			const float m = fabsf(rx[i]) > fabsf(ry[i]) ? fabsf(rx[i]) : fabsf(ry[i]);
			maxd = m > maxd ? m : maxd;
		}
		for (int i = 0; i < gw - 1; i++) {
			const float a = fabsf(rx[i + 1] - rx[i]), b = fabsf(ry[i + 1] - ry[i]);
			const float m = a > b ? a : b;
			slope = m > slope ? m : slope;
		}
		if (j < gh - 1)
			for (int i = 0; i < gw; i++) {
				const float a = fabsf(rx[i + gw] - rx[i]), b = fabsf(ry[i + gw] - ry[i]);
				const float m = a > b ? a : b;
				slope = m > slope ? m : slope;
			}
	}
	slope *= inv2;
	if (has_ap && slope > CF_MAX_SLOPE) {
		f->scale = CF_MAX_SLOPE / slope;
		for (size_t k = 0; k < (size_t)gw * gh; k++) {
			f->dx[k] *= f->scale;
			f->dy[k] *= f->scale;
		}
		maxd *= f->scale;
		slope = CF_MAX_SLOPE;
	}
	f->max_slope = slope;
	f->maxd = maxd;
	if (maxd < 0.005f) {
		field_free(f);
		return 0;
	}
	return 1;
}

/* Field at fractional node position (gu, gv), bilinear, clamped to the grid. */
static inline void field_at(const cf_field *f, float gu, float gv, float *dx, float *dy)
{
	gu = gu < 0 ? 0 : (gu > (float)(f->gw - 1) ? (float)(f->gw - 1) : gu);
	gv = gv < 0 ? 0 : (gv > (float)(f->gh - 1) ? (float)(f->gh - 1) : gv);
	const int i = (int)gu, j = (int)gv;
	const int i1 = i + 1 < f->gw ? i + 1 : i, j1 = j + 1 < f->gh ? j + 1 : j;
	const float ax = gu - (float)i, ay = gv - (float)j;
	const size_t k00 = (size_t)j * f->gw + i, k01 = (size_t)j * f->gw + i1;
	const size_t k10 = (size_t)j1 * f->gw + i, k11 = (size_t)j1 * f->gw + i1;
	*dx = (f->dx[k00] * (1 - ax) + f->dx[k01] * ax) * (1 - ay) + (f->dx[k10] * (1 - ax) + f->dx[k11] * ax) * ay;
	*dy = (f->dy[k00] * (1 - ax) + f->dy[k01] * ax) * (1 - ay) + (f->dy[k10] * (1 - ax) + f->dy[k11] * ax) * ay;
}

/* Catmull-Rom weights in 1/4096 units for 128 sub-pixel phases (plus phase 1.0), each row summing to
 * exactly 4096 so whole-pixel shifts are exact. Generated from w(t) = the usual cubic. */
static const int16_t cr_lut[129][4] = {
	{0, 4096, 0, 0}, {-16, 4096, 16, 0}, {-31, 4093, 34, 0}, {-46, 4091, 52, -1}, {-60, 4086, 72, -2},
	{-74, 4081, 92, -3}, {-87, 4074, 113, -4}, {-100, 4067, 135, -6}, {-112, 4058, 158, -8},
	{-124, 4047, 182, -9}, {-136, 4037, 207, -12}, {-147, 4024, 233, -14}, {-158, 4011, 259, -16},
	{-168, 3997, 286, -19}, {-178, 3982, 314, -22}, {-187, 3965, 343, -25}, {-196, 3948, 372, -28},
	{-205, 3930, 402, -31}, {-213, 3911, 433, -35}, {-220, 3890, 464, -38}, {-228, 3869, 497, -42},
	{-235, 3848, 529, -46}, {-241, 3824, 563, -50}, {-248, 3801, 597, -54}, {-254, 3776, 632, -58},
	{-259, 3751, 667, -63}, {-264, 3724, 703, -67}, {-269, 3698, 739, -72}, {-273, 3670, 776, -77},
	{-278, 3642, 813, -81}, {-281, 3612, 851, -86}, {-285, 3583, 889, -91}, {-288, 3552, 928, -96},
	{-291, 3521, 967, -101}, {-293, 3488, 1007, -106}, {-296, 3456, 1047, -111}, {-298, 3423, 1087, -116},
	{-299, 3389, 1128, -122}, {-301, 3355, 1169, -127}, {-302, 3319, 1211, -132}, {-302, 3284, 1252, -138},
	{-303, 3247, 1295, -143}, {-303, 3210, 1337, -148}, {-303, 3172, 1380, -153}, {-303, 3136, 1422, -159},
	{-303, 3097, 1466, -164}, {-302, 3058, 1509, -169}, {-301, 3020, 1552, -175}, {-300, 2980, 1596, -180},
	{-299, 2940, 1640, -185}, {-297, 2899, 1684, -190}, {-295, 2859, 1728, -196}, {-293, 2818, 1772, -201},
	{-291, 2777, 1816, -206}, {-289, 2735, 1861, -211}, {-286, 2693, 1905, -216}, {-284, 2650, 1950, -220},
	{-281, 2608, 1994, -225}, {-278, 2566, 2038, -230}, {-274, 2522, 2083, -235}, {-271, 2479, 2127, -239},
	{-267, 2434, 2172, -243}, {-264, 2392, 2216, -248}, {-260, 2348, 2260, -252}, {-256, 2304, 2304, -256},
	{-252, 2260, 2348, -260}, {-248, 2216, 2392, -264}, {-243, 2172, 2434, -267}, {-239, 2127, 2479, -271},
	{-235, 2083, 2522, -274}, {-230, 2038, 2566, -278}, {-225, 1994, 2608, -281}, {-220, 1950, 2650, -284},
	{-216, 1905, 2693, -286}, {-211, 1861, 2735, -289}, {-206, 1816, 2777, -291}, {-201, 1772, 2818, -293},
	{-196, 1728, 2859, -295}, {-190, 1684, 2899, -297}, {-185, 1640, 2940, -299}, {-180, 1596, 2980, -300},
	{-175, 1552, 3020, -301}, {-169, 1509, 3058, -302}, {-164, 1466, 3097, -303}, {-159, 1422, 3136, -303},
	{-153, 1380, 3172, -303}, {-148, 1337, 3210, -303}, {-143, 1295, 3247, -303}, {-138, 1252, 3284, -302},
	{-132, 1211, 3319, -302}, {-127, 1169, 3355, -301}, {-122, 1128, 3389, -299}, {-116, 1087, 3423, -298},
	{-111, 1047, 3456, -296}, {-106, 1007, 3488, -293}, {-101, 967, 3521, -291}, {-96, 928, 3552, -288},
	{-91, 889, 3583, -285}, {-86, 851, 3612, -281}, {-81, 813, 3642, -278}, {-77, 776, 3670, -273},
	{-72, 739, 3698, -269}, {-67, 703, 3724, -264}, {-63, 667, 3751, -259}, {-58, 632, 3776, -254},
	{-54, 597, 3801, -248}, {-50, 563, 3824, -241}, {-46, 529, 3848, -235}, {-42, 497, 3869, -228},
	{-38, 464, 3890, -220}, {-35, 433, 3911, -213}, {-31, 402, 3930, -205}, {-28, 372, 3948, -196},
	{-25, 343, 3965, -187}, {-22, 314, 3982, -178}, {-19, 286, 3997, -168}, {-16, 259, 4011, -158},
	{-14, 233, 4024, -147}, {-12, 207, 4037, -136}, {-9, 182, 4047, -124}, {-8, 158, 4058, -112},
	{-6, 135, 4067, -100}, {-4, 113, 4074, -87}, {-3, 92, 4081, -74}, {-2, 72, 4086, -60}, {-1, 52, 4091, -46},
	{0, 34, 4093, -31}, {0, 16, 4096, -16}, {0, 0, 4096, 0}
};

static void warp_apply(const cf_channel *ch, const cf_eye *e, cf_field *f)
{
	/* Eye geometry in this channel's sample grid. */
	const float sxs = ch->scale_x, sys = ch->scale_y;
	const float cx = e->cx * sxs, cy = e->cy * sys;
	const float rx = e->rx * sxs, ry = e->ry * sys;
	if (rx < 1.0f || ry < 1.0f)
		return;

	const int x0 = clampi((int)floorf(cx - rx), 0, ch->width - 1);
	const int x1 = clampi((int)ceilf(cx + rx), 0, ch->width - 1);
	const int y0 = clampi((int)floorf(cy - ry), 0, ch->height - 1);
	const int y1 = clampi((int)ceilf(cy + ry), 0, ch->height - 1);

	/* Snapshot of the source area (destination box grown by the displacement and the filter),
	 * so the warp never reads its own output. */
	const int pad = (int)ceilf(f->maxd * fmaxf(sxs, sys)) + 3;
	const int wx0 = clampi(x0 - pad, 0, ch->width - 1), wx1 = clampi(x1 + pad, 0, ch->width - 1);
	const int wy0 = clampi(y0 - pad, 0, ch->height - 1), wy1 = clampi(y1 + pad, 0, ch->height - 1);
	const int ww = wx1 - wx0 + 1, wh = wy1 - wy0 + 1;
	uint8_t *snap = f->snap, *own = NULL;
	if ((size_t)ww * (size_t)wh > f->snap_size) { /* not expected: channels are never finer than the image */
		snap = own = (uint8_t *)malloc((size_t)ww * (size_t)wh);
		if (!snap)
			return;
	}
	for (int y = 0; y < wh; y++) {
		const uint8_t *row = ch->base + (size_t)(wy0 + y) * ch->linesize + (size_t)wx0 * ch->step;
		uint8_t *dst = snap + (size_t)y * ww;
		if (ch->step == 1)
			memcpy(dst, row, (size_t)ww);
		else
			for (int x = 0; x < ww; x++)
				dst[x] = row[x * ch->step];
	}

	const int bicubic = ch->is_luma;
	/* Channel sample x (centre x + 0.5) is image point (x + 0.5) / scale: node (that - ox) / cs. */
	const float gsx = 1.0f / (sxs * (float)f->cs), gsy = 1.0f / (sys * (float)f->cs);
	const float gox = -f->ox / (float)f->cs, goy = -f->oy / (float)f->cs;
	const float irx = 1.0f / rx, iry = 1.0f / ry;
	/* Field rows interpolated to this channel row, in the (now free) scratch area of the field. */
	float *rdx = f->dy + (size_t)f->gw * f->gh, *rdy = rdx + f->gw;
	for (int y = y0; y <= y1; y++) {
		uint8_t *row = ch->base + (size_t)y * ch->linesize;
		const float py = (float)y + 0.5f;
		const float dyn = (py - cy) * iry;
		if (dyn * dyn >= 1.0f)
			continue;
		/* Columns inside the ellipse on this row (the exact test is repeated per sample). */
		const float half = rx * sqrtf(1.0f - dyn * dyn);
		const int xa = clampi((int)floorf(cx - half - 0.5f), x0, x1);
		const int xb = clampi((int)ceilf(cx + half - 0.5f), x0, x1);
		float gv = py * gsy + goy;
		gv = gv < 0 ? 0 : (gv > (float)(f->gh - 1) ? (float)(f->gh - 1) : gv);
		const int j = (int)gv, j1 = j + 1 < f->gh ? j + 1 : j;
		const float ay = gv - (float)j;
		const float *d0x = f->dx + (size_t)j * f->gw, *d1x = f->dx + (size_t)j1 * f->gw;
		const float *d0y = f->dy + (size_t)j * f->gw, *d1y = f->dy + (size_t)j1 * f->gw;
		int i0 = (int)fmaxf(0.0f, ((float)xa + 0.5f) * gsx + gox);
		int i1 = (int)fminf((float)(f->gw - 1), ((float)xb + 0.5f) * gsx + gox) + 1;
		i0 = clampi(i0, 0, f->gw - 1);
		i1 = clampi(i1, 0, f->gw - 1);
		for (int i = i0; i <= i1; i++) {
			rdx[i] = d0x[i] + (d1x[i] - d0x[i]) * ay;
			rdy[i] = d0y[i] + (d1y[i] - d0y[i]) * ay;
		}
		for (int x = xa; x <= xb; x++) {
			const float px = (float)x + 0.5f;
			const float dxn = (px - cx) * irx;
			if (dxn * dxn + dyn * dyn >= 1.0f)
				continue;
			float gu = px * gsx + gox;
			gu = gu < (float)i0 ? (float)i0 : (gu > (float)i1 ? (float)i1 : gu);
			int gi = (int)gu;
			if (gi >= i1)
				gi = i1 > i0 ? i1 - 1 : i0;
			const float ax0 = gu - (float)gi;
			const int gi1 = gi + 1 <= i1 ? gi + 1 : gi;
			float ddx = (rdx[gi] + (rdx[gi1] - rdx[gi]) * ax0) * sxs;
			float ddy = (rdy[gi] + (rdy[gi1] - rdy[gi]) * ax0) * sys;
			if (fabsf(ddx) < 0.004f && fabsf(ddy) < 0.004f)
				continue; /* no visible motion: keep the original sample exactly */
			/* Sample position, in snapshot coordinates. */
			float fx = px - ddx - 0.5f - (float)wx0;
			float fy = py - ddy - 0.5f - (float)wy0;
			fx = fx < 0 ? 0 : (fx > ww - 1 ? (float)(ww - 1) : fx);
			fy = fy < 0 ? 0 : (fy > wh - 1 ? (float)(wh - 1) : fy);
			const int ix = (int)fx, iy = (int)fy;
			const float ax = fx - (float)ix, ay = fy - (float)iy;
			float v;
			if (bicubic) {
				/* Fixed point: no per-tap int-to-float conversions. Phase error <= 1/256 px. */
				const int16_t *wx = cr_lut[(int)(ax * 128.0f + 0.5f)], *wy = cr_lut[(int)(ay * 128.0f + 0.5f)];
				int64_t acc = 0;
				if (ix >= 1 && iy >= 1 && ix + 2 < ww && iy + 2 < wh) {
					const uint8_t *s = snap + (size_t)(iy - 1) * ww + (ix - 1);
					for (int r = 0; r < 4; r++, s += ww)
						acc += (int64_t)(s[0] * wx[0] + s[1] * wx[1] + s[2] * wx[2] + s[3] * wx[3]) * wy[r];
				} else { /* at the frame border: repeat the edge samples */
					int cxs[4];
					for (int t = 0; t < 4; t++)
						cxs[t] = clampi(ix - 1 + t, 0, ww - 1);
					for (int r = 0; r < 4; r++) {
						const uint8_t *s = snap + (size_t)clampi(iy - 1 + r, 0, wh - 1) * ww;
						acc += (int64_t)(s[cxs[0]] * wx[0] + s[cxs[1]] * wx[1] + s[cxs[2]] * wx[2] +
								 s[cxs[3]] * wx[3]) *
						       wy[r];
					}
				}
				const int iv = (int)((acc + (1 << 23)) >> 24);
				row[(size_t)x * ch->step] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
				continue;
			} else {
				const int ix1 = ix + 1 < ww ? ix + 1 : ix, iy1 = iy + 1 < wh ? iy + 1 : iy;
				const float t0 = snap[iy * ww + ix] * (1 - ax) + snap[iy * ww + ix1] * ax;
				const float t1 = snap[iy1 * ww + ix] * (1 - ax) + snap[iy1 * ww + ix1] * ax;
				v = t0 * (1 - ay) + t1 * ay;
			}
			row[(size_t)x * ch->step] = (uint8_t)(v + 0.5f);
		}
	}
	free(own);
}

void cf_warp_channel(const cf_channel *ch, const cf_eye *eyes, int eye_count)
{
	for (int i = 0; i < eye_count; i++) {
		cf_field f;
		if (!field_build(&f, &eyes[i]))
			continue;
		warp_apply(ch, &eyes[i], &f);
		field_free(&f);
	}
}

/*
 * Test hook (deliberately not in the public header): builds the field of one eye and reports
 * its displacement at image point (x, y) (pixel-centre coordinates: pixel i spans i..i+1), the
 * steepest slope, and the scale applied by the slope limit. Returns 0 when the eye would not
 * be warped at all.
 */
int cf_warp_debug_field(const cf_eye *e, float x, float y, float *dx, float *dy, float *max_slope, float *scale);
int cf_warp_debug_field(const cf_eye *e, float x, float y, float *dx, float *dy, float *max_slope, float *scale)
{
	cf_field f;
	*dx = *dy = *max_slope = 0;
	*scale = 1;
	if (!field_build(&f, e))
		return 0;
	field_at(&f, (x - f.ox) / (float)f.cs, (y - f.oy) / (float)f.cs, dx, dy);
	*max_slope = f.max_slope;
	*scale = f.scale;
	field_free(&f);
	return 1;
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
	/* The field is built once per eye and shared by every channel. */
	for (int i = 0; i < eye_count; i++) {
		cf_field f;
		if (!field_build(&f, &eyes[i]))
			continue;
		for (int c = 0; c < count; c++)
			warp_apply(&channels[c], &eyes[i], &f);
		field_free(&f);
	}
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
