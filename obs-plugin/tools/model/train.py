#!/usr/bin/env python3
"""
Trains Cam Flip's personal eye model on your own capture sessions and exports it for the plugin.

    python3 tools/model/train.py ~/Downloads/camflip-diag/captures/<session> [more sessions...]

Each session folder comes from tools/model/capture.sh then tools/model/extract.sh. The model learns,
from pairs of frames with the same head pose, how your eye looks with the iris somewhere else:
given an eye patch (ml_eye.h) and where the iris should end up, it outputs a flow field (moves the
iris and lids), a small brightness residual (what moving pixels can't make, e.g. sclera above a
lowered lower lid) and a mask. The plugin applies that at full resolution (ml_eye.c).

Writes (by default into ~/Library/Application Support/cam-flip/, never the repository):
    eye-model.mlpackage   the Core ML model the plugin and tools/render.sh --engine model load
    eye-model.pt          the PyTorch weights (to continue training)
    preview/              before / model / real patches from held-out frames, to eyeball
Your face is in all of these: keep them on this Mac.

    python3 tools/model/train.py --selftest   trains briefly on synthetic eyes (no face data) and
                                              checks the model learns to move the iris (CI / Linux)

Needs: pip install torch numpy coremltools (coremltools only for the export).
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Must match src/ml_eye.h
W, H = 128, 64
CORNER_L, CORNER_R, CORNER_Y = 24.0, 104.0, 34.0
MAX_FLOW = 16.0
MAX_RES = 0.25

DEFAULT_OUT = os.path.expanduser("~/Library/Application Support/cam-flip")


# ---------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)

    def forward(self, x):
        return F.leaky_relu(self.c2(F.leaky_relu(self.c1(x), 0.1)), 0.1)


class EyeNet(nn.Module):
    """Small U-Net: (patch, target) -> flow (2), residual (1), mask (1). About 0.2 M parameters."""

    def __init__(self):
        super().__init__()
        self.e1 = Block(5, 16)
        self.e2 = Block(16, 32, 2)
        self.e3 = Block(32, 48, 2)
        self.e4 = Block(48, 64, 2)
        self.cond = nn.Linear(2, 64)
        self.d3 = Block(64 + 48, 48)
        self.d2 = Block(48 + 32, 32)
        self.d1 = Block(32 + 16, 16)
        self.head = nn.Conv2d(16, 4, 3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys])[None])

    def forward(self, patch, target):
        b = patch.shape[0]
        t = (target * 2 - 1)  # -1..1 like the coordinate planes
        tmap = t[:, :, None, None].expand(b, 2, H, W)
        x = torch.cat([patch, tmap, self.coords.expand(b, 2, H, W)], 1)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3) + self.cond(t)[:, :, None, None]
        d3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[2:], mode="nearest"), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="nearest"), e1], 1))
        o = self.head(d1)
        flow = torch.tanh(o[:, 0:2]) * MAX_FLOW
        res = torch.tanh(o[:, 2:3]) * MAX_RES
        mask = torch.sigmoid(o[:, 3:4] + 2.0)
        return flow, res, mask


def base_grid(device):
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    return torch.stack([xs, ys]).to(device)  # 2,H,W patch px


def sample(img, g):
    """grid_sample with border padding, written so Apple's GPU (MPS) runs it: MPS has no "border" mode,
    but with align_corners=True, clamping the grid to the image and padding with zeros is the same,
    values and gradients alike."""
    return F.grid_sample(img, g.clamp(-1, 1), mode="bilinear", padding_mode="zeros", align_corners=True)


def render(src, flow, res, mask, grid):
    """What the plugin does (ml_eye.c), in patch space: output(p) = src(p + flow(p)) + mask * res."""
    q = grid[None] + flow
    gx = q[:, 0] / (W - 1) * 2 - 1
    gy = q[:, 1] / (H - 1) * 2 - 1
    warped = sample(src, torch.stack([gx, gy], -1))
    return warped + mask * res


def ellipse_map(cx, cy, rx, ry, inner, outer, device):
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    r = torch.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    return ((r < 1).float() * inner + (r >= 1).float() * outer).to(device)


# ---------------------------------------------------------------- data

class Session:
    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "session.json")) as f:
            self.info = json.load(f)
        self.steps = []
        with open(os.path.join(folder, "targets.csv")) as f:
            for r in csv.DictReader(f):
                if not r.get("t_end"):
                    continue
                self.steps.append((float(r["t_start"]), float(r["t_end"]), r["kind"]))
        rows = list(csv.DictReader(open(os.path.join(folder, "eyes.csv"))))
        n = len(rows)
        self.patches = np.fromfile(os.path.join(folder, "patches.u8"), dtype=np.uint8).reshape(n, H, W)
        g = lambda k: np.array([float(r[k]) for r in rows], dtype=np.float32)
        self.t, self.flip = g("t"), g("flip").astype(int)
        self.px, self.py, self.open = g("pupil_x"), g("pupil_y"), g("open")
        self.ax, self.ay, self.bx, self.by = g("ax"), g("ay"), g("bx"), g("by")
        self.kind = np.array(["none"] * n, dtype=object)
        self.since = np.zeros(n, np.float32)
        self.until = np.zeros(n, np.float32)
        starts = np.array([s[0] for s in self.steps])
        idx = np.searchsorted(starts, self.t, side="right") - 1
        for i in range(n):
            if 0 <= idx[i] < len(self.steps):
                s = self.steps[idx[i]]
                if self.t[i] < s[1]:
                    self.kind[i] = s[2]
                    self.since[i] = self.t[i] - s[0]
                    self.until[i] = s[1] - self.t[i]
        settle = np.where(self.kind == "read", 0.12, 0.3)
        ok = np.isin(self.kind, ["lens", "dot", "read"]) & (self.since >= settle) & (self.until >= 0.03)
        ok &= np.isfinite(self.px) & np.isfinite(self.py)
        for e in (0, 1):  # blinks: much less open than usual at the lens
            sel = (self.flip == e) & (self.kind == "lens") & ok
            med = np.median(self.open[sel]) if sel.any() else np.median(self.open[self.flip == e])
            ok &= ~((self.flip == e) & (self.open < 0.6 * med))
        self.ok = ok
        self.eye_w = np.hypot(self.bx - self.ax, self.by - self.ay)
        print(f"{folder}: {n} patches, {ok.sum()} usable "
              f"(lens {np.sum(ok & (self.kind == 'lens'))}, dots {np.sum(ok & (self.kind == 'dot'))}, "
              f"reading {np.sum(ok & (self.kind == 'read'))})")


class Pairs:
    """Samples (source, destination) pairs: same eye, same session, a few seconds apart, head still."""

    def __init__(self, sessions, val_frac=0.15, val=False, seed=0):
        self.items = []  # (session, src index, candidate dst indices)
        for s in sessions:
            tmax = s.t.max()
            cut = tmax * (1 - val_frac)
            for e in (0, 1):
                idx = np.where(s.ok & (s.flip == e) & ((s.t >= cut) if val else (s.t < cut)))[0]
                if len(idx) < 2:
                    continue
                t = s.t[idx]
                for j, i in enumerate(idx):
                    lo, hi = np.searchsorted(t, s.t[i] - 2.5), np.searchsorted(t, s.t[i] + 2.5)
                    cand = idx[lo:hi]
                    # head still: corners within 6% of the eye width
                    tol = 0.06 * s.eye_w[i]
                    d = np.maximum(np.hypot(s.ax[cand] - s.ax[i], s.ay[cand] - s.ay[i]),
                                   np.hypot(s.bx[cand] - s.bx[i], s.by[cand] - s.by[i]))
                    cand = cand[d < tol]
                    if len(cand):
                        self.items.append((s, i, cand))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def sample(self, batch):
        src = np.zeros((batch, 1, H, W), np.float32)
        dst = np.zeros((batch, 1, H, W), np.float32)
        tgt = np.zeros((batch, 2), np.float32)
        for b in range(batch):
            s, i, cand = self.items[self.rng.integers(len(self.items))]
            r = self.rng.random()
            if r < 0.1:
                j = i  # identity: target where the iris already is -> change nothing
            else:
                lens = cand[s.kind[cand] == "lens"]
                j = self.rng.choice(lens) if (r < 0.65 and len(lens)) else self.rng.choice(cand)
            src[b, 0] = s.patches[i] / 255.0
            dst[b, 0] = s.patches[j] / 255.0
            tgt[b] = (s.px[j] / W, s.py[j] / H)
        return src, dst, tgt


def synthetic_eye(px, py, rng, lid_drop=0.0):
    """A cartoon eye in patch space (no real face data): sclera, iris, pupil, lids that follow the gaze."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    u = (xs - (CORNER_L + CORNER_R) / 2) / ((CORNER_R - CORNER_L) / 2)  # -1..1 between the corners
    down = (py - CORNER_Y) / 10.0
    upper = CORNER_Y - (13 - 4 * down - lid_drop) * np.clip(1 - u ** 2, 0, None)
    lower = CORNER_Y + (9 - 3 * max(down, 0)) * np.clip(1 - u ** 2, 0, None) ** 1.2
    skin = 0.62 + 0.04 * np.sin(xs * 0.11) * np.cos(ys * 0.17)
    img = skin.copy()
    inside = (ys > upper) & (ys < lower)
    img[inside] = 0.88
    r = np.hypot(xs - px, ys - py)
    iris = inside & (r < 11)
    img[iris] = 0.32 + 0.06 * np.cos(np.arctan2(ys - py, xs - px)[iris] * 9)
    img[inside & (r < 4.5)] = 0.08
    img[inside & (np.hypot(xs - px + 3, ys - py + 3) < 1.6)] = 0.97  # catchlight
    img += rng.normal(0, 0.008, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


class SyntheticPairs:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return 10 ** 6

    def sample(self, batch):
        src = np.zeros((batch, 1, H, W), np.float32)
        dst = np.zeros((batch, 1, H, W), np.float32)
        tgt = np.zeros((batch, 2), np.float32)
        for b in range(batch):
            p0 = (64 + self.rng.uniform(-18, 18), CORNER_Y + self.rng.uniform(-3, 10))
            p1 = (64 + self.rng.uniform(-6, 6), CORNER_Y + self.rng.uniform(-3, 1)) if self.rng.random() < 0.7 else p0
            src[b, 0] = synthetic_eye(*p0, self.rng)
            dst[b, 0] = synthetic_eye(*p1, self.rng)
            tgt[b] = (p1[0] / W, p1[1] / H)
        return src, dst, tgt


def augment(src, dst, tgt, rng, device):
    """The same small shift / zoom / tilt and lighting change for both patches (and the target)."""
    b = src.shape[0]
    ang = torch.tensor(rng.uniform(-0.05, 0.05, b), dtype=torch.float32, device=device)
    sc = torch.tensor(rng.uniform(0.95, 1.05, b), dtype=torch.float32, device=device)
    tx = torch.tensor(rng.uniform(-2.5, 2.5, b) / (W / 2), dtype=torch.float32, device=device)
    ty = torch.tensor(rng.uniform(-2.0, 2.0, b) / (H / 2), dtype=torch.float32, device=device)
    # theta maps output normalised coords to input ones (aspect-correct rotation)
    asp = W / H
    th = torch.zeros(b, 2, 3, device=device)
    th[:, 0, 0] = sc * torch.cos(ang)
    th[:, 0, 1] = -sc * torch.sin(ang) / asp
    th[:, 1, 0] = sc * torch.sin(ang) * asp
    th[:, 1, 1] = sc * torch.cos(ang)
    th[:, 0, 2] = tx
    th[:, 1, 2] = ty
    grid = F.affine_grid(th, src.shape, align_corners=True)
    src = sample(src, grid)
    dst = sample(dst, grid)
    # target: input coords -> output coords (inverse of theta)
    n = torch.stack([tgt[:, 0] * W / (W - 1) * 2 - 1, tgt[:, 1] * H / (H - 1) * 2 - 1], 1)
    # 2x2 inverse by hand (torch.linalg.solve isn't on every MPS build)
    a, b_, c, d = th[:, 0, 0], th[:, 0, 1], th[:, 1, 0], th[:, 1, 1]
    r = n - th[:, :, 2]
    det = a * d - b_ * c
    out = torch.stack([(d * r[:, 0] - b_ * r[:, 1]) / det, (-c * r[:, 0] + a * r[:, 1]) / det], 1)
    tgt = torch.stack([(out[:, 0] + 1) / 2 * (W - 1) / W, (out[:, 1] + 1) / 2 * (H - 1) / H], 1)
    gain = torch.tensor(rng.uniform(0.85, 1.15, (b, 1, 1, 1)), dtype=torch.float32, device=device)
    gamma = torch.tensor(rng.uniform(0.85, 1.2, (b, 1, 1, 1)), dtype=torch.float32, device=device)
    src = (src.clamp(1e-4, 1) ** gamma) * gain
    dst = (dst.clamp(1e-4, 1) ** gamma) * gain
    src = src + torch.randn_like(src) * 0.008
    return src.clamp(0, 1), dst.clamp(0, 1), tgt


def iris_centre(img, weight):
    """Darkness-weighted centre inside the eye region (patch px), for a rough 'where is the iris' check."""
    d = ((1 - img) ** 6) * weight
    s = d.sum((1, 2, 3)) + 1e-6
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=img.device),
                            torch.arange(W, dtype=torch.float32, device=img.device), indexing="ij")
    return torch.stack([(d[:, 0] * xs).sum((1, 2)) / s, (d[:, 0] * ys).sum((1, 2)) / s], 1)


# ---------------------------------------------------------------- training

def losses(model, src, dst, tgt, grid, wmap, edge):
    flow, res, mask = model(src, tgt)
    out = render(src, flow, res, mask, grid)
    l_pix = (torch.abs(out - dst) * wmap).mean()
    # Coarse versions of the same comparison: a sharp-image loss only "sees" an iris that is already
    # within a pixel or two of the right place, so on its own training can stall or slide back to
    # "change nothing". Blurred copies give a pull from several pixels away.
    l_coarse = 0.0
    for k in (4, 8):
        po, pd, pw = (F.avg_pool2d(x, k) for x in (out, dst, wmap.expand_as(out)))
        l_coarse = l_coarse + (torch.abs(po - pd) * pw).mean()
    gx = lambda x: x[..., :, 1:] - x[..., :, :-1]
    gy = lambda x: x[..., 1:, :] - x[..., :-1, :]
    l_grad = (torch.abs(gx(out) - gx(dst)) * wmap[..., :, 1:]).mean() + (torch.abs(gy(out) - gy(dst)) * wmap[..., 1:, :]).mean()
    l_tv = torch.abs(gx(flow)).mean() + torch.abs(gy(flow)).mean()
    l_res = torch.abs(mask * res).mean()
    l_edge = ((torch.abs(flow).sum(1, keepdim=True) / MAX_FLOW + torch.abs(mask * res) / MAX_RES) * edge).mean()
    total = l_pix + 2.0 * l_coarse + 0.5 * l_grad + 0.02 * l_tv + 0.05 * l_res + 0.5 * l_edge
    return total, out, l_pix


def evaluate(model, data, grid, wmap, edge, device, batches=20, seed=123):
    model.eval()
    data.rng = np.random.default_rng(seed)
    l_model = l_copy = e_model = e_copy = 0.0
    with torch.no_grad():
        for _ in range(batches):
            s, d, t = [torch.from_numpy(x).to(device) for x in data.sample(32)]
            _, out, lp = losses(model, s, d, t, grid, wmap, edge)
            l_model += lp.item()
            l_copy += (torch.abs(s - d) * wmap).mean().item()
            cd = iris_centre(d, wmap)
            e_model += torch.norm(iris_centre(out, wmap) - cd, dim=1).mean().item()
            e_copy += torch.norm(iris_centre(s, wmap) - cd, dim=1).mean().item()
    model.train()
    k = batches
    return l_model / k, l_copy / k, e_model / k, e_copy / k


def save_preview(model, data, grid, folder, device, n=12):
    os.makedirs(folder, exist_ok=True)
    data.rng = np.random.default_rng(7)
    s, d, t = [torch.from_numpy(x).to(device) for x in data.sample(n)]
    with torch.no_grad():
        flow, res, mask = model(s, t)
        out = render(s, flow, res, mask, grid)
    rows = [np.concatenate([s[i, 0].cpu().numpy(), out[i, 0].cpu().numpy(), d[i, 0].cpu().numpy()], 1) for i in range(n)]
    img = (np.clip(np.concatenate(rows, 0), 0, 1) * 255).astype(np.uint8)
    path = os.path.join(folder, "before-model-real.pgm")
    with open(path, "wb") as f:
        f.write(b"P5 %d %d 255\n" % (img.shape[1], img.shape[0]))
        f.write(img.tobytes())
    try:
        from PIL import Image
        Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(path[:-4] + ".png")
    except ImportError:
        pass
    print(f"preview: {path} (columns: your eye, the model's correction, the real look into the lens)")


class Export(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, patch, target):
        return self.m(patch, target)


def lens_look(sessions):
    """Where the iris sits, in patch px, looking into the lens: (x, y) for the eye left in the image (0) and
    the mirrored one (1). The plugin holds the iris there while reading, so sideways sweeps and line
    returns don't show."""
    out = []
    for e in (0, 1):
        xs = np.concatenate([s.px[s.ok & (s.kind == "lens") & (s.flip == e)] for s in sessions])
        ys = np.concatenate([s.py[s.ok & (s.kind == "lens") & (s.flip == e)] for s in sessions])
        if len(xs) < 20:
            return None
        out += [float(np.median(xs)), float(np.median(ys))]
    return out


def export_coreml(model, path, lens=None):
    import coremltools as ct
    m = Export(model).eval().cpu()
    ex = (torch.rand(2, 1, H, W), torch.rand(2, 2))
    traced = torch.jit.trace(m, ex)
    ml = ct.convert(traced,
                    inputs=[ct.TensorType(name="patch", shape=(2, 1, H, W)), ct.TensorType(name="target", shape=(2, 2))],
                    outputs=[ct.TensorType(name="flow"), ct.TensorType(name="res"), ct.TensorType(name="mask")],
                    convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.macOS13)
    ml.short_description = "Cam Flip personal eye model (private: trained on one person's face)"
    if lens:
        ml.user_defined_metadata["camflip.lens"] = " ".join(f"{v:.3f}" for v in lens)
    ml.save(path)
    print(f"exported {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="start from eye-model.pt in --out")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest-export", metavar="PATH", help="with --selftest: also export the synthetic model here")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)

    if a.selftest:
        train, val = SyntheticPairs(a.seed), SyntheticPairs(1000 + a.seed)
        # Small batches, more steps: same compute as 600 x 48, but more updates to get past the
        # start, where the model hasn't found the iris yet.
        a.steps = min(a.steps, 1200)
        a.batch = min(a.batch, 24)
        out_dir = None
    else:
        if not a.sessions:
            ap.error("give one or more capture folders (or --selftest)")
        sessions = [Session(s) for s in a.sessions]
        lens = lens_look(sessions)
        print("looking into the lens, the iris sits at " +
              (f"({lens[0]:.1f}, {lens[1]:.1f}) and ({lens[2]:.1f}, {lens[3]:.1f}) patch px" if lens else "? (too few lens frames)"))
        train, val = Pairs(sessions), Pairs(sessions, val=True, seed=1)
        print(f"training pairs from {len(train)} source frames, validation {len(val)}")
        if len(train) < 200 or len(val) < 20:
            sys.exit("not enough usable frames: record a full session (tools/model/capture.sh)")
        out_dir = a.out
        os.makedirs(out_dir, exist_ok=True)

    model = EyeNet().to(dev)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters, device {dev}")
    if a.resume and out_dir and os.path.exists(os.path.join(out_dir, "eye-model.pt")):
        model.load_state_dict(torch.load(os.path.join(out_dir, "eye-model.pt"), map_location=dev))
    grid = base_grid(dev)
    mid = (CORNER_L + CORNER_R) / 2
    wmap = ellipse_map(mid, CORNER_Y - 2, 50, 22, 1.0, 0.25, dev)[None, None]
    edge = 1 - ellipse_map(mid, CORNER_Y - 2, 60, 30, 1.0, 0.0, dev)[None, None]
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    rng = np.random.default_rng(a.seed)
    best = None
    t0 = time.time()
    for step in range(1, a.steps + 1):
        s, d, t = [torch.from_numpy(x).to(dev) for x in train.sample(a.batch)]
        s, d, t = augment(s, d, t, rng, dev)
        loss, _, _ = losses(model, s, d, t, grid, wmap, edge)
        opt.zero_grad()
        loss.backward()
        # A single bad batch (e.g. a mislabelled pair) must not throw training off: clip its step.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        sched.step()
        if step % max(1, a.steps // 10) == 0 or step == a.steps:
            lm, lc, em, ec = evaluate(model, val, grid, wmap, edge, dev, batches=5 if a.selftest else 20)
            print(f"step {step}/{a.steps} ({time.time() - t0:.0f} s): held-out difference to the real look "
                  f"{lm:.4f} (uncorrected {lc:.4f}); iris off by {em:.2f} px (uncorrected {ec:.2f})")
            if best is None or lm < best[0]:  # keep the best model seen on held-out frames
                best = (lm, copy.deepcopy(model.state_dict()))

    if best is not None:
        model.load_state_dict(best[1])
    lm, lc, em, ec = evaluate(model, val, grid, wmap, edge, dev)
    if a.selftest:
        ok = em < 0.6 * ec and lm < 0.8 * lc
        print(f"selftest {'passed' if ok else 'FAILED'}: iris error {em:.2f} px vs {ec:.2f} uncorrected, "
              f"difference {lm:.4f} vs {lc:.4f}")
        if ok and a.selftest_export:
            # The synthetic eye looks into the "lens" around the middle, just above the corner line.
            export_coreml(model.cpu(), a.selftest_export, [64.0, CORNER_Y - 1.0, 64.0, CORNER_Y - 1.0])
        sys.exit(0 if ok else 1)
    torch.save(model.state_dict(), os.path.join(out_dir, "eye-model.pt"))
    save_preview(model, val, grid, os.path.join(out_dir, "preview"), dev)
    if not a.no_export:
        export_coreml(model.cpu(), os.path.join(out_dir, "eye-model.mlpackage"), lens)


if __name__ == "__main__":
    main()
