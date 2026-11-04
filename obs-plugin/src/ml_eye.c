/* Cam Flip: personal-model eye engine, pixel side. See ml_eye.h. */
#include "ml_eye.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void affine_inverse(const float m[6], float inv[6])
{
	const float det = m[0] * m[4] - m[1] * m[3];
	const float id = fabsf(det) > 1e-12f ? 1.0f / det : 0.0f;
	inv[0] = m[4] * id;
	inv[1] = -m[1] * id;
	inv[3] = -m[3] * id;
	inv[4] = m[0] * id;
	inv[2] = -(inv[0] * m[2] + inv[1] * m[5]);
	inv[5] = -(inv[3] * m[2] + inv[4] * m[5]);
}

void cf_ml_frame_from_corners(float ax, float ay, float bx, float by, int flip, cf_ml_frame *out)
{
	/* a is the corner further left in the image. Mirroring (flip) swaps which patch side it lands on. */
	if (bx < ax) {
		float t = ax;
		ax = bx;
		bx = t;
		t = ay;
		ay = by;
		by = t;
	}
	const float rx = flip ? -1.0f : 1.0f; /* reflection of image x */
	const float ux = rx * ax, uy = ay, vx = rx * bx, vy = by;
	const float pax = flip ? CF_ML_CORNER_R : CF_ML_CORNER_L, pbx = flip ? CF_ML_CORNER_L : CF_ML_CORNER_R;
	const float pay = CF_ML_CORNER_Y, pby = CF_ML_CORNER_Y;
	/* Rotation + scale z with z (v - u) = pb - pa, as complex numbers. */
	const float dr = vx - ux, di = vy - uy, nr = pbx - pax, ni = pby - pay;
	const float den = dr * dr + di * di;
	float zr = 1, zi = 0;
	if (den > 1e-9f) {
		zr = (nr * dr + ni * di) / den;
		zi = (ni * dr - nr * di) / den;
	}
	/* patch = Z * (rx x, y) + t */
	out->m[0] = zr * rx;
	out->m[1] = -zi;
	out->m[3] = zi * rx;
	out->m[4] = zr;
	out->m[2] = pax - (zr * ux - zi * uy);
	out->m[5] = pay - (zi * ux + zr * uy);
	out->flip = flip;
	affine_inverse(out->m, out->inv);
}

int cf_ml_eye_frame(const cf_eye *e, int flip, cf_ml_frame *out)
{
	const int n = e->aperture_count;
	if (n < 2)
		return 0;
	/* The two outline points furthest apart are the corners. */
	int ia = 0, ib = 1;
	float best = -1;
	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++) {
			const float dx = e->aperture[i][0] - e->aperture[j][0], dy = e->aperture[i][1] - e->aperture[j][1];
			const float d = dx * dx + dy * dy;
			if (d > best) {
				best = d;
				ia = i;
				ib = j;
			}
		}
	if (best < 4.0f)
		return 0;
	cf_ml_frame_from_corners(e->aperture[ia][0], e->aperture[ia][1], e->aperture[ib][0], e->aperture[ib][1], flip,
				 out);
	return 1;
}

void cf_ml_to_patch(const cf_ml_frame *f, float x, float y, float *px, float *py)
{
	*px = f->m[0] * x + f->m[1] * y + f->m[2];
	*py = f->m[3] * x + f->m[4] * y + f->m[5];
}

void cf_ml_to_image(const cf_ml_frame *f, float px, float py, float *x, float *y)
{
	*x = f->inv[0] * px + f->inv[1] * py + f->inv[2];
	*y = f->inv[3] * px + f->inv[4] * py + f->inv[5];
}

static inline int clampi(int v, int lo, int hi)
{
	return v < lo ? lo : (v > hi ? hi : v);
}

static inline float px_at(const cf_channel *c, int x, int y)
{
	x = clampi(x, 0, c->width - 1);
	y = clampi(y, 0, c->height - 1);
	return c->base[(size_t)y * c->linesize + (size_t)x * c->step];
}

void cf_ml_crop(const cf_channel *luma, const cf_ml_frame *f, float *out)
{
	for (int j = 0; j < CF_ML_H; j++)
		for (int i = 0; i < CF_ML_W; i++) {
			float x, y;
			cf_ml_to_image(f, (float)i, (float)j, &x, &y);
			/* image px -> channel px (pixel centres) */
			const float cx = (x + 0.5f) * luma->scale_x - 0.5f, cy = (y + 0.5f) * luma->scale_y - 0.5f;
			const int x0 = (int)floorf(cx), y0 = (int)floorf(cy);
			const float fx = cx - (float)x0, fy = cy - (float)y0;
			const float v = (px_at(luma, x0, y0) * (1 - fx) + px_at(luma, x0 + 1, y0) * fx) * (1 - fy) +
					(px_at(luma, x0, y0 + 1) * (1 - fx) + px_at(luma, x0 + 1, y0 + 1) * fx) * fy;
			out[j * CF_ML_W + i] = v / 255.0f;
		}
}

float cf_ml_iris_x(const float *patch, float x_guess, float py, float search)
{
	/* Mean of rows py+1 .. py+5: through the iris, below the pupil's catchlight. */
	const int y0 = (int)floorf(py + 1.0f), y1 = y0 + 4;
	if (y0 < 0 || y1 >= CF_ML_H)
		return x_guess;
	float prof[CF_ML_W];
	for (int i = 0; i < CF_ML_W; i++) {
		float s = 0;
		for (int j = y0; j <= y1; j++)
			s += patch[j * CF_ML_W + i];
		prof[i] = s / (float)(y1 - y0 + 1);
	}
	/* Asymmetry about x: mean |p(x+u) - p(x-u)| for u = 2..R (the iris, its edges and some white). */
	enum { R = 20 };
	float cost[129];
	const float step = 0.5f;
	int n = 0;
	for (float x = x_guess - search; x <= x_guess + search + 1e-3f && n < 129; x += step, n++) {
		float c = 0;
		int m = 0;
		for (int u = 2; u <= R; u++) {
			const float a = x - (float)u, b = x + (float)u;
			if (a < 0 || b > CF_ML_W - 1)
				continue;
			const int ia = (int)a, ib = (int)b;
			const float fa = a - (float)ia, fb = b - (float)ib;
			const float va = prof[ia] * (1 - fa) + prof[ia + 1 < CF_ML_W ? ia + 1 : ia] * fa;
			const float vb = prof[ib] * (1 - fb) + prof[ib + 1 < CF_ML_W ? ib + 1 : ib] * fb;
			c += fabsf(va - vb);
			m++;
		}
		cost[n] = m ? c / (float)m : 1e9f;
	}
	int best = 0;
	for (int k = 1; k < n; k++)
		if (cost[k] < cost[best])
			best = k;
	float off = 0;
	if (best > 0 && best < n - 1) {
		const float a = cost[best - 1], b = cost[best], c = cost[best + 1];
		const float d = a - 2 * b + c;
		if (d > 1e-9f)
			off = 0.5f * (a - c) / d;
	}
	return x_guess - search + ((float)best + off) * step;
}

/* Bilinear lookup in a patch-sized map; 0 outside. */
static inline float map_at(const float *m, float px, float py)
{
	if (!m || px < 0 || py < 0 || px > CF_ML_W - 1 || py > CF_ML_H - 1)
		return 0;
	const int x0 = (int)px, y0 = (int)py;
	const int x1 = x0 + 1 < CF_ML_W ? x0 + 1 : x0, y1 = y0 + 1 < CF_ML_H ? y0 + 1 : y0;
	const float fx = px - (float)x0, fy = py - (float)y0;
	return (m[y0 * CF_ML_W + x0] * (1 - fx) + m[y0 * CF_ML_W + x1] * fx) * (1 - fy) +
	       (m[y1 * CF_ML_W + x0] * (1 - fx) + m[y1 * CF_ML_W + x1] * fx) * fy;
}

static inline float cr_weight(float t, int k)
{
	/* Catmull-Rom weights for samples at -1, 0, 1, 2 */
	const float t2 = t * t, t3 = t2 * t;
	switch (k) {
	case 0:
		return 0.5f * (-t3 + 2 * t2 - t);
	case 1:
		return 0.5f * (3 * t3 - 5 * t2 + 2);
	case 2:
		return 0.5f * (-3 * t3 + 4 * t2 + t);
	default:
		return 0.5f * (t3 - t2);
	}
}

/* Bicubic sample of a w x h byte buffer (edges repeat). */
static float bicubic(const uint8_t *buf, int w, int h, float x, float y)
{
	const int x0 = (int)floorf(x), y0 = (int)floorf(y);
	const float fx = x - (float)x0, fy = y - (float)y0;
	float wx[4], wy[4];
	for (int k = 0; k < 4; k++) {
		wx[k] = cr_weight(fx, k);
		wy[k] = cr_weight(fy, k);
	}
	float s = 0;
	for (int j = 0; j < 4; j++) {
		const int yy = clampi(y0 - 1 + j, 0, h - 1);
		float r = 0;
		for (int i = 0; i < 4; i++)
			r += wx[i] * buf[(size_t)yy * w + clampi(x0 - 1 + i, 0, w - 1)];
		s += wy[j] * r;
	}
	return s;
}

static inline float smoothstep(float e0, float e1, float x)
{
	float t = (x - e0) / (e1 - e0);
	t = t < 0 ? 0 : (t > 1 ? 1 : t);
	return t * t * (3 - 2 * t);
}

static void apply_eye(const cf_channel *ch, const cf_eye *e, const cf_ml_frame *f, const cf_ml_out *o, float alpha,
		      int add_res)
{
	if (alpha <= 0 || e->rx <= 0 || e->ry <= 0)
		return;
	if (alpha > 1)
		alpha = 1;
	const float sx = ch->scale_x, sy = ch->scale_y;
	/* patch px per image px, to bound how far the flow can reach in the image */
	const float pscale = sqrtf(fabsf(f->m[0] * f->m[4] - f->m[1] * f->m[3]));
	const float reach = pscale > 1e-6f ? CF_ML_MAX_FLOW / pscale : 0;
	const int bx0 = clampi((int)floorf((e->cx - e->rx) * sx) - 1, 0, ch->width - 1);
	const int bx1 = clampi((int)ceilf((e->cx + e->rx) * sx) + 1, 0, ch->width - 1);
	const int by0 = clampi((int)floorf((e->cy - e->ry) * sy) - 1, 0, ch->height - 1);
	const int by1 = clampi((int)ceilf((e->cy + e->ry) * sy) + 1, 0, ch->height - 1);
	const int mx = (int)ceilf(reach * sx) + 3, my = (int)ceilf(reach * sy) + 3;
	const int tx0 = clampi(bx0 - mx, 0, ch->width - 1), tx1 = clampi(bx1 + mx, 0, ch->width - 1);
	const int ty0 = clampi(by0 - my, 0, ch->height - 1), ty1 = clampi(by1 + my, 0, ch->height - 1);
	const int tw = tx1 - tx0 + 1, th = ty1 - ty0 + 1;
	uint8_t *tmp = malloc((size_t)tw * th);
	if (!tmp)
		return;
	for (int y = 0; y < th; y++)
		for (int x = 0; x < tw; x++)
			tmp[(size_t)y * tw + x] = ch->base[(size_t)(ty0 + y) * ch->linesize + (size_t)(tx0 + x) * ch->step];

	for (int cy = by0; cy <= by1; cy++)
		for (int cx = bx0; cx <= bx1; cx++) {
			const float x = ((float)cx + 0.5f) / sx - 0.5f, y = ((float)cy + 0.5f) / sy - 0.5f;
			const float dx = (x - e->cx) / e->rx, dy = (y - e->cy) / e->ry;
			const float r = sqrtf(dx * dx + dy * dy);
			if (r >= 1)
				continue;
			const float w = alpha * (1 - smoothstep(0.55f, 1.0f, r));
			float px, py;
			cf_ml_to_patch(f, x, y, &px, &py);
			if (px < 0 || py < 0 || px > CF_ML_W - 1 || py > CF_ML_H - 1)
				continue;
			const float qx = px + map_at(o->flow_x, px, py), qy = py + map_at(o->flow_y, px, py);
			float ix, iy;
			cf_ml_to_image(f, qx, qy, &ix, &iy);
			const float scx = (ix + 0.5f) * sx - 0.5f - (float)tx0, scy = (iy + 0.5f) * sy - 0.5f - (float)ty0;
			float v = bicubic(tmp, tw, th, scx, scy);
			if (add_res)
				v += 255.0f * map_at(o->mask, px, py) * map_at(o->res, px, py);
			const float orig = tmp[(size_t)(cy - ty0) * tw + (cx - tx0)];
			float outv = orig + w * (v - orig);
			outv = outv < 0 ? 0 : (outv > 255 ? 255 : outv);
			ch->base[(size_t)cy * ch->linesize + (size_t)cx * ch->step] = (uint8_t)(outv + 0.5f);
		}
	free(tmp);
}

void cf_ml_apply(const cf_channel *channels, int count, const cf_eye *eyes, const cf_ml_frame *frames,
		 const cf_ml_out *outs, const float *alpha, int eye_count, unsigned res_channels)
{
	for (int c = 0; c < count; c++)
		for (int e = 0; e < eye_count; e++)
			apply_eye(&channels[c], &eyes[e], &frames[e], &outs[e], alpha[e], (int)((res_channels >> c) & 1u));
}

unsigned cf_ml_residual_channels(enum cf_format format)
{
	switch (format) {
	case CF_FMT_RGBA:
	case CF_FMT_BGRA:
	case CF_FMT_BGRX:
		return 7u;
	case CF_FMT_UNSUPPORTED:
		return 0u;
	default:
		return 1u;
	}
}
