/*
 * Replays a landmarks CSV from tools/diag_video.m through the gaze model, the way the plugin runs
 * it live, and measures shaking. Pure C: runs anywhere.
 *
 *   replay <landmarks.csv> [--lag N] [--lens a-b] [--notes a-b] [--strength 0..2] [--scan-damp 0..1] [--out per-frame.csv]
 *          [--baseline B --baseline-open O --notes-offset N --notes-side S --notes-open P]
 *          [--head-side S --head-vert V --head-eyes E --head-tilt K]
 *
 *   build: cc -O2 -Isrc tools/replay.c src/gaze.c src/warp.c src/pose.c -lm -o replay
 *
 * --lag N     the tracking result used for frame k comes from frame k-N (background tracking),
 *             moved forward with cf_gaze_predict(); 0 = analysed in step with the video.
 * --lens a-b  seconds during which the person looked into the lens (calibration step 1)
 * --notes a-b seconds during which they looked at their notes (calibration step 2)
 * --head-*    the head looking into the camera, as saved by step 1 (--lens learns it too); without
 *             it a typical face is assumed for the head-turn fade. CSVs from before the head columns
 *             fall back to Vision's own yaw and pitch.
 *
 * Shake metric: for each signal, the RMS of (value - centred 5-frame mean). Real movement is
 * smooth across 5 frames, so what remains is frame-to-frame jitter, in pixels.
 */
#include "gaze.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	double t;
	int found;
	float ms, conf;
	cf_face face;
	cf_pt vision_pupil[2];
} row_t;

static int parse(const char *path, row_t **rows_out)
{
	FILE *fp = fopen(path, "r");
	if (!fp)
		return -1;
	size_t cap = 4096, n = 0;
	row_t *rows = calloc(cap, sizeof(row_t));
	static char line[65536];
	if (!fgets(line, sizeof(line), fp)) { fclose(fp); free(rows); return 0; } /* header */
	while (fgets(line, sizeof(line), fp)) {
		if (n == cap) {
			cap *= 2;
			rows = realloc(rows, cap * sizeof(row_t));
		}
		row_t *r = &rows[n];
		memset(r, 0, sizeof(*r));
		double v[512];
		int k = 0;
		for (char *tok = strtok(line, ","); tok && k < 512; tok = strtok(NULL, ","))
			v[k++] = atof(tok);
		if (k < 19)
			continue;
		r->t = v[1];
		r->ms = (float)v[2];
		r->found = (int)v[3];
		r->conf = (float)v[5];
		cf_face *f = &r->face;
		f->found = r->found;
		f->has_pose = 1;
		f->yaw_deg = (float)v[6];
		f->pitch_deg = (float)v[7];
		f->face_width = (float)v[8];
		f->image_width = (int)v[9];
		f->image_height = (int)v[10];
		r->vision_pupil[0] = (cf_pt){(float)v[11], (float)v[12]};
		r->vision_pupil[1] = (cf_pt){(float)v[13], (float)v[14]};
		f->left_pupil_vision = r->vision_pupil[0];
		f->right_pupil_vision = r->vision_pupil[1];
		f->has_vision_pupils = 1;
		f->left_pupil = (cf_pt){(float)v[15], (float)v[16]};
		f->right_pupil = (cf_pt){(float)v[17], (float)v[18]};
		int i = 19;
		int nl = (int)v[i++];
		if (nl < 0 || nl > (k - i) / 2)
			nl = 0;
		for (int j = 0; j < nl && j < CF_MAX_EYE_POINTS && i + 2 * j + 1 < k; j++)
			f->left_eye[j] = (cf_pt){(float)v[i + 2 * j], (float)v[i + 2 * j + 1]};
		f->left_count = nl < CF_MAX_EYE_POINTS ? nl : CF_MAX_EYE_POINTS;
		i += 2 * nl;
		if (i < k) {
			int nr = (int)v[i++];
			if (nr < 0 || nr > (k - i) / 2)
				nr = 0;
			for (int j = 0; j < nr && j < CF_MAX_EYE_POINTS && i + 2 * j + 1 < k; j++)
				f->right_eye[j] = (cf_pt){(float)v[i + 2 * j], (float)v[i + 2 * j + 1]};
			f->right_count = nr < CF_MAX_EYE_POINTS ? nr : CF_MAX_EYE_POINTS;
			i += 2 * nr;
		}
		/* Head measures (newer CSVs): used by the head-turn fade instead of Vision's angles. */
		if (i + 4 < k && v[i] > 0.5) {
			f->has_head_geom = 1;
			f->head = (cf_head){(float)v[i + 1], (float)v[i + 2], (float)v[i + 3], (float)v[i + 4]};
		}
		n++;
	}
	fclose(fp);
	*rows_out = rows;
	return (int)n;
}

/* RMS of x minus its centred 5-sample mean, over samples where ok[i] for the whole window. */
static double shake(const double *x, const int *ok, int n)
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

static cf_pt centroid(const cf_pt *p, int n)
{
	cf_pt c = {0, 0};
	for (int i = 0; i < n; i++) {
		c.x += p[i].x;
		c.y += p[i].y;
	}
	if (n) {
		c.x /= n;
		c.y /= n;
	}
	return c;
}

static int in_range(double t, const double r[2])
{
	return r[1] > r[0] && t >= r[0] && t <= r[1];
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: replay <landmarks.csv> [--lag N] [--lens a-b] [--notes a-b] [--strength s] [--scan-damp d] [--out f]\n");
		return 2;
	}
	int lag = 0;
	double lens[2] = {0, 0}, notes[2] = {0, 0};
	float strength = 1.0f, scan_damp = 0.0f;
	const char *outp = NULL;
	float cal[5] = {0, 0, 0, 0, 0}; /* baseline, baseline_open, notes_offset, notes_side, notes_open */
	float lids[2] = {0, 0}, vis[2] = {0, 0}; /* baseline_upper/lower; vis_baseline/notes */
	int vis_opt = 0;
	int have_cal = 0;
	cf_head_ref head = {0, 0, 0, 0, 0};
	for (int i = 2; i < argc; i++) {
		if (!strcmp(argv[i], "--lag") && i + 1 < argc)
			lag = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--lens") && i + 1 < argc)
			sscanf(argv[++i], "%lf-%lf", &lens[0], &lens[1]);
		else if (!strcmp(argv[i], "--notes") && i + 1 < argc)
			sscanf(argv[++i], "%lf-%lf", &notes[0], &notes[1]);
		else if (!strcmp(argv[i], "--strength") && i + 1 < argc)
			strength = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--scan-damp") && i + 1 < argc)
			scan_damp = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--out") && i + 1 < argc)
			outp = argv[++i];
		else if (!strcmp(argv[i], "--baseline") && i + 1 < argc)
			cal[0] = (float)atof(argv[++i]), have_cal |= 1;
		else if (!strcmp(argv[i], "--baseline-open") && i + 1 < argc)
			cal[1] = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--notes-offset") && i + 1 < argc)
			cal[2] = (float)atof(argv[++i]), have_cal |= 2;
		else if (!strcmp(argv[i], "--notes-side") && i + 1 < argc)
			cal[3] = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--notes-open") && i + 1 < argc)
			cal[4] = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--head-side") && i + 1 < argc)
			head.side = (float)atof(argv[++i]), head.ok = 1;
		else if (!strcmp(argv[i], "--head-vert") && i + 1 < argc)
			head.vert = (float)atof(argv[++i]), head.ok = 1;
		else if (!strcmp(argv[i], "--head-eyes") && i + 1 < argc)
			head.eyes = (float)atof(argv[++i]), head.ok = 1;
		else if (!strcmp(argv[i], "--head-tilt") && i + 1 < argc)
			head.tilt = (float)atof(argv[++i]), head.ok = 1;
		else if (!strcmp(argv[i], "--baseline-upper") && i + 1 < argc)
			lids[0] = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--baseline-lower") && i + 1 < argc)
			lids[1] = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--vis-baseline") && i + 1 < argc)
			vis[0] = (float)atof(argv[++i]), vis_opt |= 1;
		else if (!strcmp(argv[i], "--vis-notes") && i + 1 < argc)
			vis[1] = (float)atof(argv[++i]), vis_opt |= 2;
	}
	row_t *rows;
	const int n = parse(argv[1], &rows);
	if (n <= 0) {
		fprintf(stderr, "no rows\n");
		return 1;
	}

	/* Calibrate exactly like the buttons do, on the given time ranges. */
	cf_gaze_params p = {.strength = strength, .max_glance = 0.15f, .head_ref = head, .scan_damp = scan_damp};
	/* Calibration as saved in the filter's settings (OBS scene JSON), if given. */
	if (have_cal & 1) {
		p.baseline = cal[0];
		p.baseline_open = cal[1];
	}
	if (have_cal & 2) {
		p.has_notes = 1;
		p.notes_offset = cal[2];
		p.notes_side = cal[3];
		p.notes_open = cal[4];
	}
	p.baseline_upper = lids[0];
	p.baseline_lower = lids[1];
	p.vis_baseline = vis[0];
	p.vis_notes = vis[1];
	p.has_vis_calib = vis_opt == 3 && p.has_notes; /* as plugin.m: both steps taken with Vision's point */
	int vis_cal = 0; /* Vision's point given at step 1 (bit 1) and step 2 (bit 2), as in plugin.m */
	for (int step = 0; step < 2; step++) {
		const double *r = step == 0 ? lens : notes;
		if (!(r[1] > r[0]))
			continue;
		cf_gaze g;
		cf_gaze_init(&g);
		cf_eye tmp[2];
		cf_gaze_start_calibration(&g, step == 0 ? CF_CALIB_LENS : CF_CALIB_NOTES);
		cf_calib_result res;
		int got = 0;
		for (int i = 0; i < n && !got; i++) {
			if (!in_range(rows[i].t, r))
				continue;
			cf_gaze_update(&g, &rows[i].face, &p, rows[i].t, tmp);
			got = cf_gaze_take_calibration(&g, &res);
		}
		if (got > 0 && step == 0) {
			p.baseline = res.offset;
			p.baseline_open = res.openness;
			p.head_ref = res.head;
			vis_cal = res.has_vis ? (vis_cal | 1) : (vis_cal & ~1);
			p.vis_baseline = res.vis_offset;
			p.has_vis_calib = vis_cal == 3 && p.has_notes;
			p.baseline_upper = res.upper;
			p.baseline_lower = res.lower;
			printf("lens calibration: baseline %.4f open %.4f; head %s: side %.4f vert %.4f eyes %.4f tilt %.4f\n",
			       res.offset, res.openness, res.head.ok ? "learnt" : "not learnt", res.head.side, res.head.vert,
			       res.head.eyes, res.head.tilt);
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

	double *sig[16];
	for (int s = 0; s < 16; s++)
		sig[s] = calloc((size_t)n, sizeof(double));
	int *ok_in = calloc((size_t)n, sizeof(int)), *ok_out = calloc((size_t)n, sizeof(int));
	double align_err = 0;
	int align_n = 0;
	FILE *of = outp ? fopen(outp, "w") : NULL;
	if (of)
		fprintf(of, "t,n,cx0,cy0,sx0,sy0,cx1,cy1,sx1,sy1,eye_cx0,eye_cy0,pupil_y0,vpupil_y0\n");

	cf_gaze g;
	cf_gaze_init(&g);
	int fed = -1;
	for (int k = 0; k < n; k++) {
		/* Background tracking: by the time frame k is output, frames up to k-lag are analysed. */
		const int upto = k - lag;
		while (fed < upto) {
			fed++;
			cf_eye tmp[2];
			cf_gaze_update(&g, &rows[fed].face, &p, rows[fed].t, tmp);
		}
		cf_eye e[2];
		const int ne = cf_gaze_predict(&g, &p, rows[k].t, e);

		const cf_face *f = &rows[k].face;
		ok_in[k] = f->found && f->left_count >= 4 && f->right_count >= 4;
		ok_out[k] = ne == 2 && ok_in[k];
		const cf_pt lc = centroid(f->left_eye, f->left_count), rc = centroid(f->right_eye, f->right_count);
		/* Inputs */
		sig[0][k] = lc.x;
		sig[1][k] = lc.y;
		sig[2][k] = f->left_pupil.y - lc.y;   /* refined pupil, relative to the eye */
		sig[3][k] = rows[k].vision_pupil[0].y - lc.y; /* Vision's pupil, relative to the eye */
		sig[4][k] = f->right_pupil.y - rc.y;
		/* Outputs */
		sig[5][k] = e[0].cx;
		sig[6][k] = e[0].cy;
		sig[7][k] = e[0].sx;
		sig[8][k] = e[0].sy;
		sig[9][k] = e[1].sy;
		/* Iris position as the viewer sees it = real pupil + applied shift */
		sig[10][k] = f->left_pupil.y + (ne ? e[0].sy : 0);
		sig[11][k] = f->left_pupil.x + (ne ? e[0].sx : 0);
		if (ok_out[k]) {
			const double dx = e[0].cx - lc.x, dy = e[0].cy - lc.y;
			align_err += dx * dx + dy * dy;
			align_n++;
		}
		if (of)
			fprintf(of, "%.4f,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", rows[k].t, ne,
				e[0].cx, e[0].cy, e[0].sx, e[0].sy, e[1].cx, e[1].cy, e[1].sx, e[1].sy, lc.x, lc.y,
				f->left_pupil.y, rows[k].vision_pupil[0].y);
	}
	if (of)
		fclose(of);

	int found = 0, warped = 0;
	double ms = 0, conf = 0;
	for (int k = 0; k < n; k++) {
		found += ok_in[k];
		warped += ok_out[k];
		ms += rows[k].ms;
		conf += rows[k].conf;
	}
	printf("frames %d (%.1f s), face %d, warped %d, mean tracking %.1f ms, mean confidence %.2f, lag %d\n", n,
	       rows[n - 1].t - rows[0].t, found, warped, ms / n, conf / n, lag);
	printf("INPUT shake (px):  eye centre x %.2f y %.2f | refined pupil-in-eye y L %.2f R %.2f | Vision pupil-in-eye y L %.2f\n",
	       shake(sig[0], ok_in, n), shake(sig[1], ok_in, n), shake(sig[2], ok_in, n), shake(sig[4], ok_in, n),
	       shake(sig[3], ok_in, n));
	printf("OUTPUT shake (px): warp centre x %.2f y %.2f | shift x %.2f y L %.2f R %.2f\n", shake(sig[5], ok_out, n),
	       shake(sig[6], ok_out, n), shake(sig[7], ok_out, n), shake(sig[8], ok_out, n), shake(sig[9], ok_out, n));
	printf("SEEN iris shake (px, real pupil + shift, while warped): x %.2f y %.2f\n", shake(sig[11], ok_out, n),
	       shake(sig[10], ok_out, n));
	printf("warp centre vs this frame's eye centre: RMS %.2f px\n", align_n ? sqrt(align_err / align_n) : NAN);
	return 0;
}
