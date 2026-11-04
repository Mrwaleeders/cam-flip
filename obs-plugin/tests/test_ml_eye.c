/*
 * Checks for the personal-model eye engine's pixel side (runs anywhere with a C compiler):
 *   cc -O2 -Isrc tests/test_ml_eye.c src/ml_eye.c src/warp.c -lm -o test_ml_eye && ./test_ml_eye
 *
 *  - the patch frame puts the eye corners where the model expects them, mirrored for the right eye,
 *  - a "do nothing" answer leaves the frame byte for byte as it was, in every format,
 *  - pixels outside the eye ellipses never change, whatever the model says,
 *  - a flow moves the eye content by the expected amount in the image, at native resolution,
 *  - the brightness residual reaches only brightness channels (never chroma or alpha),
 *  - cropping reads the right pixels, and two eyes on a 1080p frame are fast.
 */
#include "ml_eye.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

enum { N = CF_ML_W * CF_ML_H };
static float zero[N], fx[N], fy[N], res[N], one[N];

static cf_eye make_eye(float cx, float cy, float w)
{
	cf_eye e;
	memset(&e, 0, sizeof(e));
	e.cx = cx;
	e.cy = cy;
	e.rx = w * 0.85f;
	e.ry = w * 0.55f;
	/* six-point outline like Vision's: corners, two upper, two lower */
	const float o[6][2] = {{-0.5f, 0}, {-0.2f, -0.18f}, {0.2f, -0.18f}, {0.5f, 0}, {0.2f, 0.15f}, {-0.2f, 0.15f}};
	e.aperture_count = 6;
	for (int i = 0; i < 6; i++) {
		e.aperture[i][0] = cx + o[i][0] * w;
		e.aperture[i][1] = cy + o[i][1] * w;
	}
	return e;
}

static void test_frame(void)
{
	cf_ml_frame f;
	cf_ml_frame_from_corners(100, 200, 160, 210, 0, &f);
	float px, py;
	cf_ml_to_patch(&f, 100, 200, &px, &py);
	CHECK(fabsf(px - CF_ML_CORNER_L) < 1e-3f && fabsf(py - CF_ML_CORNER_Y) < 1e-3f, "left corner -> %f %f", px, py);
	cf_ml_to_patch(&f, 160, 210, &px, &py);
	CHECK(fabsf(px - CF_ML_CORNER_R) < 1e-3f && fabsf(py - CF_ML_CORNER_Y) < 1e-3f, "right corner -> %f %f", px, py);
	/* up in the image is up in the patch */
	float ux, uy;
	cf_ml_to_patch(&f, 130, 180, &ux, &uy);
	CHECK(uy < CF_ML_CORNER_Y - 10, "above the eye maps above the corner line (%f)", uy);

	cf_ml_frame g;
	cf_ml_frame_from_corners(100, 200, 160, 210, 1, &g);
	cf_ml_to_patch(&g, 100, 200, &px, &py);
	CHECK(fabsf(px - CF_ML_CORNER_R) < 1e-3f && fabsf(py - CF_ML_CORNER_Y) < 1e-3f, "mirrored left corner -> %f %f",
	      px, py);
	cf_ml_to_patch(&g, 130, 180, &ux, &uy);
	CHECK(uy < CF_ML_CORNER_Y - 10, "mirroring keeps up as up (%f)", uy);
	float x, y;
	cf_ml_to_image(&g, 50, 20, &px, &py);
	cf_ml_to_patch(&g, px, py, &x, &y);
	CHECK(fabsf(x - 50) < 1e-3f && fabsf(y - 20) < 1e-3f, "round trip");

	cf_eye e = make_eye(300, 300, 60);
	CHECK(cf_ml_eye_frame(&e, 0, &f), "frame from outline");
	cf_ml_to_patch(&f, 270, 300, &px, &py);
	CHECK(fabsf(px - CF_ML_CORNER_L) < 1e-3f, "outline corner %f", px);
	cf_eye none = e;
	none.aperture_count = 0;
	CHECK(!cf_ml_eye_frame(&none, 0, &f), "no outline, no frame");
	printf("ok frame\n");
}

typedef struct {
	enum cf_format fmt;
	const char *name;
} fmt_t;

static const fmt_t formats[] = {{CF_FMT_NV12, "NV12"}, {CF_FMT_I420, "I420"}, {CF_FMT_UYVY, "UYVY"},
				{CF_FMT_YUY2, "YUY2"}, {CF_FMT_BGRA, "BGRA"}, {CF_FMT_I444, "I444"}};

/* A frame of the format, filled with a pattern. Returns channels; planes in *mem. */
static int make_frame(enum cf_format fmt, int w, int h, uint8_t **mem, size_t *size, cf_channel ch[8])
{
	uint8_t *d[4] = {0};
	uint32_t ls[4] = {0};
	size_t total = 0;
	switch (fmt) {
	case CF_FMT_NV12:
		ls[0] = (uint32_t)w;
		ls[1] = (uint32_t)((w + 1) / 2 * 2);
		total = (size_t)w * h + (size_t)ls[1] * ((h + 1) / 2);
		*mem = malloc(total);
		d[0] = *mem;
		d[1] = *mem + (size_t)w * h;
		break;
	case CF_FMT_I420:
		ls[0] = (uint32_t)w;
		ls[1] = ls[2] = (uint32_t)((w + 1) / 2);
		total = (size_t)w * h + 2 * (size_t)ls[1] * ((h + 1) / 2);
		*mem = malloc(total);
		d[0] = *mem;
		d[1] = d[0] + (size_t)w * h;
		d[2] = d[1] + (size_t)ls[1] * ((h + 1) / 2);
		break;
	case CF_FMT_I444:
		ls[0] = ls[1] = ls[2] = (uint32_t)w;
		total = 3 * (size_t)w * h;
		*mem = malloc(total);
		d[0] = *mem;
		d[1] = d[0] + (size_t)w * h;
		d[2] = d[1] + (size_t)w * h;
		break;
	case CF_FMT_UYVY:
	case CF_FMT_YUY2:
		ls[0] = (uint32_t)(w * 2);
		total = (size_t)ls[0] * h;
		*mem = malloc(total);
		d[0] = *mem;
		break;
	default:
		ls[0] = (uint32_t)(w * 4);
		total = (size_t)ls[0] * h;
		*mem = malloc(total);
		d[0] = *mem;
		break;
	}
	for (size_t i = 0; i < total; i++)
		(*mem)[i] = (uint8_t)((i * 2654435761u) >> 24);
	if (fmt == CF_FMT_BGRA)
		for (size_t i = 3; i < total; i += 4)
			(*mem)[i] = 255;
	*size = total;
	return cf_channels(fmt, d, ls, w, h, ch);
}

static void outs_of(cf_ml_out *o, const float *flx, const float *fly, const float *r, const float *m)
{
	o->flow_x = flx;
	o->flow_y = fly;
	o->res = r;
	o->mask = m;
}

static int inside_any(const cf_eye *e, int n, float x, float y)
{
	for (int i = 0; i < n; i++) {
		const float dx = (x - e[i].cx) / e[i].rx, dy = (y - e[i].cy) / e[i].ry;
		if (dx * dx + dy * dy < 1.0f)
			return 1;
	}
	return 0;
}

static void test_identity_and_outside(void)
{
	const int w = 640, h = 360;
	cf_eye eyes[2] = {make_eye(260, 170, 56), make_eye(380, 172, 56)};
	cf_ml_frame fr[2];
	cf_ml_eye_frame(&eyes[0], 0, &fr[0]);
	cf_ml_eye_frame(&eyes[1], 1, &fr[1]);
	const float alpha[2] = {1, 1};
	for (size_t k = 0; k < sizeof(formats) / sizeof(formats[0]); k++) {
		uint8_t *mem, *orig;
		size_t size;
		cf_channel ch[8];
		const int cc = make_frame(formats[k].fmt, w, h, &mem, &size, ch);
		orig = malloc(size);
		memcpy(orig, mem, size);

		cf_ml_out o[2];
		outs_of(&o[0], zero, zero, zero, zero);
		outs_of(&o[1], zero, zero, zero, zero);
		cf_ml_apply(ch, cc, eyes, fr, o, alpha, 2, cf_ml_residual_channels(formats[k].fmt));
		CHECK(memcmp(mem, orig, size) == 0, "%s: a do-nothing answer changed the frame", formats[k].name);

		/* big flow and residual: nothing outside the ellipses may change */
		outs_of(&o[0], fx, fy, res, one);
		outs_of(&o[1], fx, fy, res, one);
		cf_ml_apply(ch, cc, eyes, fr, o, alpha, 2, cf_ml_residual_channels(formats[k].fmt));
		int changed_in = 0;
		for (int c = 0; c < cc; c++)
			for (int y = 0; y < ch[c].height; y++)
				for (int x = 0; x < ch[c].width; x++) {
					const size_t off = (size_t)(ch[c].base - mem) + (size_t)y * ch[c].linesize +
							   (size_t)x * ch[c].step;
					const float ix = ((float)x + 0.5f) / ch[c].scale_x - 0.5f;
					const float iy = ((float)y + 0.5f) / ch[c].scale_y - 0.5f;
					if (mem[off] != orig[off]) {
						CHECK(inside_any(eyes, 2, ix, iy), "%s: channel %d pixel %d,%d outside the eyes changed",
						      formats[k].name, c, x, y);
						changed_in++;
					}
				}
		CHECK(changed_in > 0, "%s: the model's answer changed nothing", formats[k].name);
		if (formats[k].fmt == CF_FMT_BGRA) {
			for (size_t i = 3; i < size; i += 4)
				CHECK(mem[i] == 255, "BGRA alpha changed");
		}
		free(mem);
		free(orig);
	}
	printf("ok identity / outside untouched (%zu formats)\n", sizeof(formats) / sizeof(formats[0]));
}

/* A dark disc ("iris") on a light background; flow_y = +k patch px everywhere moves it up. */
static void test_flow_moves_iris(void)
{
	const int w = 640, h = 360;
	uint8_t *mem;
	size_t size;
	cf_channel ch[8];
	const int cc = make_frame(CF_FMT_NV12, w, h, &mem, &size, ch);
	const float icx = 300, icy = 200, ir = 9;
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++) {
			const float d = sqrtf((x - icx) * (x - icx) + (y - icy) * (y - icy));
			mem[(size_t)y * w + x] = (uint8_t)(d < ir ? 40 : (d < ir + 1 ? 40 + 160 * (d - ir) : 200));
		}
	cf_eye e = make_eye(icx, icy, 64);
	e.rx = 60;
	e.ry = 40;
	cf_ml_frame fr;
	cf_ml_eye_frame(&e, 0, &fr);
	const float pscale = 80.0f / 64.0f; /* patch px per image px */
	const float k = 5.0f;               /* patch px */
	static float up[N];
	for (int i = 0; i < N; i++)
		up[i] = k;
	cf_ml_out o;
	outs_of(&o, zero, up, zero, zero);
	const float a = 1;
	cf_ml_apply(ch, cc, &e, &fr, &o, &a, 1, 1u);
	double sy = 0, sw = 0;
	for (int y = 150; y < 250; y++)
		for (int x = 250; x < 350; x++) {
			const double dk = 200.0 - mem[(size_t)y * w + x];
			if (dk > 20) {
				sy += dk * y;
				sw += dk;
			}
		}
	const double moved = icy - sy / sw;
	CHECK(fabs(moved - k / pscale) < 0.3, "iris moved up %.2f px, expected %.2f", moved, k / pscale);
	free(mem);
	printf("ok flow moves the iris %.2f px (expected %.2f)\n", moved, k / pscale);
}

static void test_residual_luma_only(void)
{
	const int w = 320, h = 240;
	uint8_t *mem, *orig;
	size_t size;
	cf_channel ch[8];
	const int cc = make_frame(CF_FMT_NV12, w, h, &mem, &size, ch);
	orig = malloc(size);
	memcpy(orig, mem, size);
	cf_eye e = make_eye(160, 120, 60);
	cf_ml_frame fr;
	cf_ml_eye_frame(&e, 0, &fr);
	static float r[N];
	for (int i = 0; i < N; i++)
		r[i] = 0.1f;
	cf_ml_out o;
	outs_of(&o, zero, zero, r, one);
	const float a = 1;
	cf_ml_apply(ch, cc, &e, &fr, &o, &a, 1, 1u);
	CHECK(memcmp(mem + (size_t)w * h, orig + (size_t)w * h, size - (size_t)w * h) == 0, "chroma changed");
	const size_t c = (size_t)120 * w + 160;
	const int diff = (int)mem[c] - (int)orig[c];
	CHECK(orig[c] > 230 || abs(diff - 26) <= 1, "centre brightness +%d, expected +26", diff);
	free(mem);
	free(orig);
	printf("ok residual reaches brightness only\n");
}

static void test_crop(void)
{
	const int w = 320, h = 240;
	uint8_t *img = malloc((size_t)w * h);
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
			img[(size_t)y * w + x] = (uint8_t)(x < 160 ? 50 : 200);
	cf_channel ch[8];
	uint8_t *d[1] = {img};
	uint32_t ls[1] = {(uint32_t)w};
	cf_channels(CF_FMT_Y800, d, ls, w, h, ch);
	static float p[N];
	cf_ml_frame fr;
	cf_ml_frame_from_corners(130, 120, 190, 120, 0, &fr);
	cf_ml_crop(&ch[0], &fr, p);
	/* image x 160 is the middle of the corners: patch x 64 */
	CHECK(fabsf(p[30 * CF_ML_W + 40] - 50 / 255.0f) < 1e-3f && fabsf(p[30 * CF_ML_W + 90] - 200 / 255.0f) < 1e-3f,
	      "crop sides %f %f", p[30 * CF_ML_W + 40], p[30 * CF_ML_W + 90]);
	cf_ml_frame_from_corners(130, 120, 190, 120, 1, &fr);
	cf_ml_crop(&ch[0], &fr, p);
	CHECK(fabsf(p[30 * CF_ML_W + 40] - 200 / 255.0f) < 1e-3f, "mirrored crop");
	free(img);
	printf("ok crop\n");
}

static void test_speed(void)
{
	const int w = 1920, h = 1080;
	uint8_t *mem;
	size_t size;
	cf_channel ch[8];
	const int cc = make_frame(CF_FMT_NV12, w, h, &mem, &size, ch);
	cf_eye eyes[2] = {make_eye(880, 500, 70), make_eye(1040, 502, 70)};
	cf_ml_frame fr[2];
	cf_ml_eye_frame(&eyes[0], 0, &fr[0]);
	cf_ml_eye_frame(&eyes[1], 1, &fr[1]);
	cf_ml_out o[2];
	outs_of(&o[0], fx, fy, res, one);
	outs_of(&o[1], fx, fy, res, one);
	const float alpha[2] = {1, 1};
	static float p[N];
	const int reps = 50;
	const clock_t t0 = clock();
	for (int i = 0; i < reps; i++) {
		cf_ml_crop(&ch[0], &fr[0], p);
		cf_ml_crop(&ch[0], &fr[1], p);
		cf_ml_apply(ch, cc, eyes, fr, o, alpha, 2, 1u);
	}
	const double ms = (double)(clock() - t0) * 1000.0 / CLOCKS_PER_SEC / reps;
	free(mem);
	CHECK(ms < 4.0, "crop + apply took %.2f ms per frame", ms);
	printf("ok speed: crop + apply %.2f ms per 1080p frame (two eyes, without the model)\n", ms);
}

int main(void)
{
	for (int j = 0; j < CF_ML_H; j++)
		for (int i = 0; i < CF_ML_W; i++) {
			fx[j * CF_ML_W + i] = 3.0f * sinf((float)j * 0.3f);
			fy[j * CF_ML_W + i] = 6.0f;
			res[j * CF_ML_W + i] = 0.2f;
			one[j * CF_ML_W + i] = 1.0f;
		}
	test_frame();
	test_identity_and_outside();
	test_flow_moves_iris();
	test_residual_luma_only();
	test_crop();
	test_speed();
	if (failures) {
		printf("%d FAILED\n", failures);
		return 1;
	}
	printf("all ml_eye tests passed\n");
	return 0;
}
