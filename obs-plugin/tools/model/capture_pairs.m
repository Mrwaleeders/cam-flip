/*
 * Records the training session for the personal eye model: a full-screen dot to follow, while the
 * camera records at full quality. Head still, eyes only. About 4 minutes.
 *
 *   tools/model/capture.sh --distance 60          (centimetres from your eyes to the screen)
 *   tools/model/capture.sh --distance 60 --reading   (reading-focused session, about 3 minutes)
 *
 * The sequence:
 *   1. a grid of dots over the whole screen (up to 20 cm either side of the camera and down to the
 *      bottom of the screen), each followed by a look into the camera lens: pairs of frames with the
 *      same head pose that differ only in where the eyes look;
 *   2. reading: the dot hops along lines of "text" in the notes area just under the camera, with line
 *      returns, like reading notes;
 *   3. a few looks away (keyboard, off to the side) and blinks, to check the fade later.
 *
 * Writes into the output folder (default ~/Downloads/camflip-diag/captures/<date>-<distance>cm):
 *   video.mov      the camera, H.264 at a high bitrate
 *   targets.csv    when each dot was shown and where (cm from the camera: x right, y down)
 *   session.json   distance, screen size, camera, resolution
 * This is your face: it stays on this Mac. Never add it to the repository.
 * Esc stops early (what was recorded is kept).
 */
#import <AVFoundation/AVFoundation.h>
#import <AppKit/AppKit.h>
#import <CoreMedia/CoreMedia.h>
#import <QuartzCore/QuartzCore.h>

#include <math.h>
#include <stdio.h>

typedef enum { ST_TEXT, ST_LENS, ST_DOT, ST_READ, ST_AWAY } step_kind;

typedef struct {
	step_kind kind;
	double x_cm, y_cm; /* from the camera: x right, y down */
	double dur;
	int line;          /* reading line index (ST_READ), -1 otherwise */
	const char *text;
} step_t;

static step_t *steps;
static int nsteps, cap_steps;

static void add(step_kind k, double x, double y, double dur, int line, const char *text)
{
	if (nsteps == cap_steps) {
		cap_steps = cap_steps ? cap_steps * 2 : 256;
		steps = realloc(steps, sizeof(step_t) * (size_t)cap_steps);
	}
	steps[nsteps++] = (step_t){k, x, y, dur, line, text};
}

/* Screen geometry */
static double pt_per_cm, cam_above_cm = 0.5, screen_w_cm, screen_h_cm;
static int reading_mode;
static NSRect screen_frame;

static NSPoint cm_to_pt(double x_cm, double y_cm)
{
	/* camera at the top centre, cam_above_cm above the top edge of the picture */
	const double x = screen_frame.size.width / 2 + x_cm * pt_per_cm;
	const double y_top = (y_cm - cam_above_cm) * pt_per_cm; /* from the top edge */
	return NSMakePoint(x, screen_frame.size.height - y_top);
}

static void build_sequence(int quick)
{
	add(ST_TEXT, 0, 0, 6, -1,
	    "Sit as you do in calls. Keep your head still and follow the dot with your eyes only.\n"
	    "When the arrow points up, look straight into the camera lens. Blink normally.\nStarting…");
	add(ST_LENS, 0, 0, 3, -1, "Look into the camera lens");

	/* Grid over the whole screen, each dot paired with a look into the lens. */
	const double half = fmin(20.0, screen_w_cm / 2 - 0.8);
	const double bottom = fmin(25.0, screen_h_cm + cam_above_cm - 0.8);
	double xs[16], ys[16];
	int nx = 0, ny = 0;
	/* evenly spaced, about 5 cm apart (10 quick), edge to edge */
	nx = (int)lround(2 * half / (quick ? 10.0 : 5.0)) + 1;
	if (nx > 16)
		nx = 16;
	for (int i = 0; i < nx; i++)
		xs[i] = -half + 2 * half * i / (nx - 1);
	const double yv[] = {1.5, 3, 5, 7, 9, 12, 15, 19, 25};
	for (size_t i = 0; i < sizeof(yv) / sizeof(yv[0]); i++) {
		const double y = fmin(yv[i], bottom);
		if (ny == 0 || y > ys[ny - 1] + 0.5)
			ys[ny++] = y;
		if (quick)
			i++;
	}
	int order[256], no = 0;
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			order[no++] = j * nx + i;
	srand(7);
	for (int i = no - 1; i > 0; i--) {
		const int k = rand() % (i + 1), t = order[i];
		order[i] = order[k];
		order[k] = t;
	}
	for (int i = 0; i < no; i++) {
		add(ST_DOT, xs[order[i] % nx], ys[order[i] / nx], 1.3, -1, NULL);
		add(ST_LENS, 0, 0, 1.0, -1, NULL);
	}

	/* Reading in the notes area: hops along lines, with line returns. */
	add(ST_TEXT, 0, 0, 3, -1, "Now read along the lines: follow the dot.");
	const double bands[] = {4, 7, 10};
	int line = 0;
	for (int b = 0; b < (quick ? 1 : 3); b++) {
		for (int l = 0; l < 3; l++, line++) {
			const double y = bands[b] + l * 0.9;
			for (double x = -9; x <= 9.01; x += 1.6)
				add(ST_READ, x, y, 0.28, line, NULL);
		}
		add(ST_LENS, 0, 0, 1.5, -1, NULL);
	}

	/* Looking away and blinking (to check the fade, not for training). */
	add(ST_AWAY, 0, 0, 2.5, -1, "Look down at your keyboard");
	add(ST_LENS, 0, 0, 1.5, -1, NULL);
	add(ST_AWAY, 0, 0, 2.5, -1, "Look away to the left of the screen");
	add(ST_LENS, 0, 0, 1.5, -1, NULL);
	add(ST_LENS, 0, 0, 2.5, -1, "Look into the lens and blink twice");
	add(ST_TEXT, 0, 0, 2, -1, "Done. Thank you!");
}

/*
 * --reading: about 3 minutes of reading in the notes area, each line followed by a look into the lens, so
 * every reading frame (along the line, the jump back, the next line) is within a couple of seconds of a
 * lens look with the head in the same place. That is what the model needs most: reading to the lens.
 */
static void build_reading_sequence(int quick)
{
	add(ST_TEXT, 0, 0, 6, -1,
	    "Sit as you do in calls. Keep your head still and follow the dot with your eyes only,
"
	    "like reading your notes. When the arrow points up, look straight into the camera lens.
Starting…");
	add(ST_LENS, 0, 0, 3, -1, "Look into the camera lens");
	const double half = fmin(12.0, screen_w_cm / 2 - 0.8);
	const double ys[] = {3, 4.5, 6, 7.5, 9, 10.5, 12};
	const double widths[] = {7, 9, 11};
	const double hop_t[] = {0.25, 0.32, 0.22};
	int line = 0;
	for (int round = 0; round < (quick ? 1 : 3); round++) {
		const double w = fmin(widths[round], half), dt = hop_t[round];
		const double hop = round == 1 ? 1.3 : 1.7;
		for (size_t k = 0; k < sizeof(ys) / sizeof(ys[0]); k++, line++) {
			const double y = ys[(k * 3 + (size_t)round) % (sizeof(ys) / sizeof(ys[0]))];
			for (double x = -w; x <= w + 0.01; x += hop)
				add(ST_READ, x, y, dt, line, NULL);
			/* two lines in a row now and then: the jump back to the next line, as in real reading */
			if (k % 3 == 1) {
				line++;
				for (double x = -w; x <= w + 0.01; x += hop)
					add(ST_READ, x, y + 0.9, dt, line, NULL);
			}
			add(ST_LENS, 0, 0, 1.0, -1, NULL);
		}
	}
	/* Single glances anywhere in the notes area, each back to the lens. */
	srand(11);
	for (int i = 0; i < (quick ? 4 : 24); i++) {
		const double x = -half + 2 * half * (rand() / (double)RAND_MAX), y = 2 + 11 * (rand() / (double)RAND_MAX);
		add(ST_DOT, x, y, 0.8, -1, NULL);
		add(ST_LENS, 0, 0, 0.9, -1, NULL);
	}
	add(ST_TEXT, 0, 0, 2, -1, "Done. Thank you!");
}

/* ---- recording ---- */

@interface Recorder : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property AVCaptureSession *session;
@property AVAssetWriter *writer;
@property AVAssetWriterInput *input;
@property dispatch_queue_t queue;
@property double first_pts; /* host-clock seconds of the first frame, <0 before it */
@property int frames, width, height;
@property double fps;
@property NSString *camera_name;
@property BOOL stopping;
@end

@implementation Recorder
- (BOOL)startWithCamera:(NSString *)want path:(NSString *)path
{
	self.first_pts = -1;
	AVCaptureDevice *dev = nil;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	for (AVCaptureDevice *d in [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo])
		if (want.length && [d.localizedName rangeOfString:want options:NSCaseInsensitiveSearch].location != NSNotFound)
			dev = d;
#pragma clang diagnostic pop
	if (!dev)
		dev = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
	if (!dev) {
		fprintf(stderr, "no camera found\n");
		return NO;
	}
	self.camera_name = dev.localizedName;
	/* Largest format up to 1920 wide that does 30 fps (what OBS gets from it in calls). */
	AVCaptureDeviceFormat *best = nil;
	int bw = 0;
	for (AVCaptureDeviceFormat *f in dev.formats) {
		CMVideoDimensions d = CMVideoFormatDescriptionGetDimensions(f.formatDescription);
		BOOL fps30 = NO;
		for (AVFrameRateRange *r in f.videoSupportedFrameRateRanges)
			if (r.maxFrameRate >= 29.9)
				fps30 = YES;
		if (fps30 && d.width <= 1920 && d.width > bw) {
			best = f;
			bw = d.width;
		}
	}
	NSError *e = nil;
	self.session = [[AVCaptureSession alloc] init];
	AVCaptureDeviceInput *in = [AVCaptureDeviceInput deviceInputWithDevice:dev error:&e];
	if (!in) {
		fprintf(stderr, "cannot open the camera: %s\n", e.localizedDescription.UTF8String);
		return NO;
	}
	[self.session addInput:in];
	if (best && [dev lockForConfiguration:&e]) {
		dev.activeFormat = best;
		dev.activeVideoMinFrameDuration = CMTimeMake(1, 30);
		dev.activeVideoMaxFrameDuration = CMTimeMake(1, 30);
		[dev unlockForConfiguration];
	}
	CMVideoDimensions dim = CMVideoFormatDescriptionGetDimensions(dev.activeFormat.formatDescription);
	self.width = dim.width;
	self.height = dim.height;
	self.fps = 30;

	AVCaptureVideoDataOutput *out = [[AVCaptureVideoDataOutput alloc] init];
	out.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)};
	out.alwaysDiscardsLateVideoFrames = NO;
	self.queue = dispatch_queue_create("capture", DISPATCH_QUEUE_SERIAL);
	[out setSampleBufferDelegate:self queue:self.queue];
	[self.session addOutput:out];

	[[NSFileManager defaultManager] removeItemAtPath:path error:nil];
	self.writer = [AVAssetWriter assetWriterWithURL:[NSURL fileURLWithPath:path] fileType:AVFileTypeQuickTimeMovie
						  error:&e];
	self.input = [AVAssetWriterInput
		assetWriterInputWithMediaType:AVMediaTypeVideo
			       outputSettings:@{
				       AVVideoCodecKey: AVVideoCodecTypeH264,
				       AVVideoWidthKey: @(self.width),
				       AVVideoHeightKey: @(self.height),
				       AVVideoCompressionPropertiesKey: @{AVVideoAverageBitRateKey: @(45e6)}
			       }];
	self.input.expectsMediaDataInRealTime = YES;
	[self.writer addInput:self.input];
	[self.writer startWriting];
	[self.session startRunning];
	return YES;
}

- (double)now
{
	return CMTimeGetSeconds(CMClockGetTime(self.session.synchronizationClock ?: CMClockGetHostTimeClock()));
}

- (void)captureOutput:(AVCaptureOutput *)o didOutputSampleBuffer:(CMSampleBufferRef)sb
       fromConnection:(AVCaptureConnection *)c
{
	(void)o;
	(void)c;
	if (self.stopping)
		return;
	const CMTime pts = CMSampleBufferGetPresentationTimeStamp(sb);
	if (self.first_pts < 0) {
		[self.writer startSessionAtSourceTime:pts];
		self.first_pts = CMTimeGetSeconds(pts);
	}
	if (self.input.readyForMoreMediaData && [self.input appendSampleBuffer:sb])
		self.frames++;
}

- (void)stop:(void (^)(void))done
{
	dispatch_sync(self.queue, ^{
		self.stopping = YES;
	});
	[self.session stopRunning];
	[self.input markAsFinished];
	[self.writer finishWritingWithCompletionHandler:^{
		dispatch_async(dispatch_get_main_queue(), done);
	}];
}
@end

/* ---- the screen ---- */

static Recorder *rec;
static int cur = -1;
static double step_t0;
static FILE *targets;
static NSString *out_dir;
static double distance_cm = 60;

@interface DotView : NSView
@end

@implementation DotView
- (BOOL)acceptsFirstResponder
{
	return YES;
}
- (void)keyDown:(NSEvent *)e
{
	if (e.keyCode == 53) /* Esc */
		[NSApp terminate:nil];
}
- (void)drawRect:(NSRect)r
{
	(void)r;
	[[NSColor colorWithWhite:0.94 alpha:1] setFill];
	NSRectFill(self.bounds);
	if (cur < 0 || cur >= nsteps)
		return;
	const step_t *s = &steps[cur];
	NSDictionary *attr = @{NSFontAttributeName: [NSFont systemFontOfSize:22],
			       NSForegroundColorAttributeName: [NSColor colorWithWhite:0.25 alpha:1]};
	if (s->text) {
		NSString *t = [NSString stringWithUTF8String:s->text];
		NSSize sz = [t sizeWithAttributes:attr];
		[t drawAtPoint:NSMakePoint((self.bounds.size.width - sz.width) / 2, self.bounds.size.height * 0.45)
		    withAttributes:attr];
	}
	if (s->kind == ST_LENS) {
		/* An arrow pointing at the camera. */
		const NSPoint top = cm_to_pt(0, cam_above_cm + 0.2);
		NSBezierPath *p = [NSBezierPath bezierPath];
		const double a = 2.0 * pt_per_cm;
		[p moveToPoint:NSMakePoint(top.x, top.y)];
		[p lineToPoint:NSMakePoint(top.x - a * 0.5, top.y - a * 0.7)];
		[p lineToPoint:NSMakePoint(top.x + a * 0.5, top.y - a * 0.7)];
		[p closePath];
		[[NSColor colorWithRed:0.85 green:0.2 blue:0.2 alpha:1] setFill];
		[p fill];
	} else if (s->kind == ST_DOT || s->kind == ST_READ) {
		const NSPoint c = cm_to_pt(s->x_cm, s->y_cm);
		const double since = CACurrentMediaTime() - step_t0;
		/* a ring that shrinks onto the dot draws the eye to it */
		if (s->kind == ST_DOT) {
			const double rr = 4 + 26 * fmax(0, 1 - since / 0.35);
			NSBezierPath *ring = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(c.x - rr, c.y - rr, 2 * rr, 2 * rr)];
			[[NSColor colorWithWhite:0.4 alpha:0.6] setStroke];
			ring.lineWidth = 2;
			[ring stroke];
		} else {
			/* faint "text" line under the dot */
			[[NSColor colorWithWhite:0.8 alpha:1] setFill];
			const NSPoint l0 = cm_to_pt(-9.5, s->y_cm), l1 = cm_to_pt(9.5, s->y_cm);
			NSRectFill(NSMakeRect(l0.x, l0.y - 4, l1.x - l0.x, 8));
		}
		const double d = 5;
		[[NSColor colorWithWhite:0.1 alpha:1] setFill];
		[[NSBezierPath bezierPathWithOvalInRect:NSMakeRect(c.x - d, c.y - d, 2 * d, 2 * d)] fill];
	}
}
@end

static const char *kind_name(step_kind k)
{
	switch (k) {
	case ST_LENS:
		return "lens";
	case ST_DOT:
		return "dot";
	case ST_READ:
		return "read";
	case ST_AWAY:
		return "away";
	default:
		return "text";
	}
}

static void finish(void)
{
	static int done = 0;
	if (done++)
		return;
	if (cur >= 0 && cur < nsteps && rec.first_pts >= 0) /* close the last row */
		fprintf(targets, "%.4f\n", [rec now] - rec.first_pts);
	fclose(targets);
	NSDictionary *info = @{
		@"distance_cm": @(distance_cm),
		@"screen_w_cm": @(screen_w_cm),
		@"screen_h_cm": @(screen_h_cm),
		@"cam_above_cm": @(cam_above_cm),
		@"camera": rec.camera_name ?: @"",
		@"width": @(rec.width),
		@"height": @(rec.height),
		@"frames": @(rec.frames),
		@"date": [[NSDate date] description],
	};
	NSData *j = [NSJSONSerialization dataWithJSONObject:info options:NSJSONWritingPrettyPrinted error:nil];
	[j writeToFile:[out_dir stringByAppendingPathComponent:@"session.json"] atomically:YES];
	dispatch_semaphore_t sem = dispatch_semaphore_create(0);
	[rec stop:^{
		dispatch_semaphore_signal(sem);
	}];
	/* finishWriting calls back on the main queue: spin the run loop until it does */
	while (dispatch_semaphore_wait(sem, DISPATCH_TIME_NOW))
		[[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.05]];
	printf("recorded %d frames (%dx%d) into %s\n", rec.frames, rec.width, rec.height, out_dir.UTF8String);
}

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property NSWindow *window;
@property DotView *view;
@end

@implementation AppDelegate
- (void)tick:(NSTimer *)t
{
	(void)t;
	if (rec.first_pts < 0) /* wait for the camera's first frame */
		return;
	const double now = CACurrentMediaTime();
	if (cur < 0 || now - step_t0 >= steps[cur].dur) {
		const double host = [rec now] - rec.first_pts;
		if (cur >= 0)
			fprintf(targets, "%.4f\n", host);
		cur++;
		if (cur >= nsteps) {
			[NSApp terminate:nil];
			return;
		}
		step_t0 = now;
		const step_t *s = &steps[cur];
		fprintf(targets, "%d,%s,%.2f,%.2f,%d,%.4f,", cur, kind_name(s->kind), s->x_cm, s->y_cm, s->line, host);
	}
	self.view.needsDisplay = YES;
}

- (void)applicationDidFinishLaunching:(NSNotification *)n
{
	(void)n;
	NSScreen *scr = [NSScreen mainScreen];
	screen_frame = scr.frame;
	const CGDirectDisplayID did = [scr.deviceDescription[@"NSScreenNumber"] unsignedIntValue];
	const CGSize mm = CGDisplayScreenSize(did);
	screen_w_cm = mm.width / 10.0;
	screen_h_cm = mm.height / 10.0;
	pt_per_cm = screen_w_cm > 1 ? screen_frame.size.width / screen_w_cm : 40;
	printf("screen %.1f x %.1f cm, camera assumed %.1f cm above the top edge\n", screen_w_cm, screen_h_cm,
	       cam_above_cm);
	if (reading_mode)
		build_reading_sequence(getenv("CAMFLIP_QUICK") != NULL);
	else
		build_sequence(getenv("CAMFLIP_QUICK") != NULL);
	double total = 0;
	for (int i = 0; i < nsteps; i++)
		total += steps[i].dur;
	printf("%d steps, about %.0f s\n", nsteps, total);

	self.window = [[NSWindow alloc] initWithContentRect:screen_frame
						  styleMask:NSWindowStyleMaskBorderless
						    backing:NSBackingStoreBuffered
						      defer:NO
						     screen:scr];
	self.window.level = NSScreenSaverWindowLevel;
	self.view = [[DotView alloc] initWithFrame:NSMakeRect(0, 0, screen_frame.size.width, screen_frame.size.height)];
	self.window.contentView = self.view;
	[self.window makeKeyAndOrderFront:nil];
	[self.window makeFirstResponder:self.view];
	[NSApp activateIgnoringOtherApps:YES];
	[NSCursor hide];
	[NSTimer scheduledTimerWithTimeInterval:1.0 / 120 target:self selector:@selector(tick:) userInfo:nil repeats:YES];
}

- (void)applicationWillTerminate:(NSNotification *)n
{
	(void)n;
	finish();
}
@end

int main(int argc, char **argv)
{
	@autoreleasepool {
		NSString *camera = @"";
		for (int i = 1; i < argc; i++) {
			const char *v = i + 1 < argc ? argv[i + 1] : "";
			if (!strcmp(argv[i], "--distance")) { distance_cm = atof(v); i++; }
			else if (!strcmp(argv[i], "--out")) { out_dir = [NSString stringWithUTF8String:v]; i++; }
			else if (!strcmp(argv[i], "--camera")) { camera = [NSString stringWithUTF8String:v]; i++; }
			else if (!strcmp(argv[i], "--cam-above")) { cam_above_cm = atof(v); i++; }
			else if (!strcmp(argv[i], "--reading")) reading_mode = 1;
			else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
		}
		if (!out_dir) {
			NSDateFormatter *df = [[NSDateFormatter alloc] init];
			df.dateFormat = @"yyyy-MM-dd-HHmmss";
			out_dir = [NSHomeDirectory() stringByAppendingPathComponent:
				[NSString stringWithFormat:@"Downloads/camflip-diag/captures/%@-%.0fcm%s",
					[df stringFromDate:[NSDate date]], distance_cm, reading_mode ? "-reading" : ""]];
		}
		[[NSFileManager defaultManager] createDirectoryAtPath:out_dir withIntermediateDirectories:YES attributes:nil
								error:nil];
		targets = fopen([out_dir stringByAppendingPathComponent:@"targets.csv"].UTF8String, "w");
		fprintf(targets, "step,kind,x_cm,y_cm,line,t_start,t_end\n");

		rec = [[Recorder alloc] init];
		if (![rec startWithCamera:camera path:[out_dir stringByAppendingPathComponent:@"video.mov"]])
			return 1;
		printf("camera: %s %dx%d\n", rec.camera_name.UTF8String, rec.width, rec.height);

		NSApplication *app = [NSApplication sharedApplication];
		app.activationPolicy = NSApplicationActivationPolicyRegular;
		AppDelegate *d = [[AppDelegate alloc] init];
		app.delegate = d;
		[app run];
	}
	return 0;
}
