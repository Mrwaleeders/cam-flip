/*
 * Cam Flip: pixel operations on raw camera frames (no OBS or Apple dependencies).
 *
 * Every operation works on one "channel" at a time: a grid of 8-bit samples that may be
 * interleaved with other channels (packed formats like UYVY or BGRA) or sit in its own plane
 * (NV12, I420). That lets one warp and one mirror routine handle every common camera format
 * in its native layout, with no conversion and therefore no quality loss.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum cf_format {
	CF_FMT_UNSUPPORTED = 0,
	CF_FMT_I420,
	CF_FMT_NV12,
	CF_FMT_YUY2, /* Y0 U Y1 V */
	CF_FMT_YVYU, /* Y0 V Y1 U */
	CF_FMT_UYVY, /* U Y0 V Y1 */
	CF_FMT_RGBA,
	CF_FMT_BGRA,
	CF_FMT_BGRX,
	CF_FMT_Y800,
	CF_FMT_I444,
	CF_FMT_I422,
};

typedef struct {
	uint8_t *base;  /* first sample of row 0 */
	int linesize;   /* bytes between rows */
	int step;       /* bytes between neighbouring samples in a row */
	int width;      /* samples per row */
	int height;     /* rows */
	float scale_x;  /* channel width / image width */
	float scale_y;  /* channel height / image height */
	int is_luma;    /* 1 for the channel that carries brightness (or a colour channel in RGB) */
} cf_channel;

#define CF_MAX_APERTURE 32

/* One eye region, in full-resolution image pixels with the origin at the top left. */
typedef struct {
	float cx, cy; /* centre of the region that may change */
	float rx, ry; /* ellipse radii: nothing outside this ellipse is ever changed */
	float sx, sy; /* iris displacement (negative sy moves content up) */
	float lid_sx, lid_sy; /* upper eyelid displacement, usually about a third of the iris one */
	/* Outline of the eye opening (lid edges), in order around the eye, image pixels.
	 * aperture_count 0 means "no outline": the whole ellipse moves by (sx, sy). */
	int aperture_count;
	float aperture[CF_MAX_APERTURE][2];
} cf_eye;

/* Splits a frame into channels. Returns the number of channels (0 if unsupported). */
int cf_channels(enum cf_format format, uint8_t *const data[], const uint32_t linesize[], int width, int height,
		cf_channel out[8]);

/*
 * Moves the content inside each eye ellipse by its displacement, fading smoothly to zero at
 * the ellipse edge. Pixels outside the ellipses are not touched at all.
 */
void cf_warp_channel(const cf_channel *ch, const cf_eye *eyes, int eye_count);

/* Exact horizontal flip, in place. */
void cf_mirror_channel(const cf_channel *ch);

/* Convenience: apply to every channel of a frame. */
void cf_warp_frame(const cf_channel *channels, int count, const cf_eye *eyes, int eye_count);
void cf_mirror_frame(const cf_channel *channels, int count);

/* Copies the brightness of a packed 4:2:2 frame into a tightly packed 8-bit grey buffer. */
void cf_extract_luma(const cf_channel *luma, uint8_t *out);

#ifdef __cplusplus
}
#endif
