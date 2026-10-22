#!/usr/bin/env python3
"""
experiments.py — Controlled experiments for the paper.

Runs on top of mh_engine.py (same phantom, preprocessing, model and metrics).
Each training run takes ~10 s on an Apple-silicon GPU; the full suite ~3–5 min.

  E1  Multi-seed baseline          : 3 seeds, clean phantom, classification + Grad-CAM verification
  E2  Grad-CAM depth ablation      : last L = 1, 2, 4, 6, 12 transformer blocks (same trained models)
  E3  Shortcut injection           : a synthetic "scanner annotation" marker in the image border,
                                     correlated with the MH label at strength rho, standard and
                                     subtle lesions; tested on clean and label-flipped markers
  E4  Normalisation under device shift : percentile vs min-max vs none, tested in-distribution
                                     and on an unseen low-contrast scanner response
  E0  Metric validation (no training) : controlled saliency maps with known answers

Usage
  python experiments.py                 # everything
  python experiments.py --only E3       # one experiment (E0, E1, E2, E3, E4)
  python experiments.py --seeds 0 1 2   # seeds for E1/E2

Outputs
  outputs/experiments/results.json     every number reported in the paper
  outputs/experiments/*.png            qualitative Grad-CAM panels
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import mh_engine as eng

OUT = eng.OUT_DIR / "experiments"

# Marker = a 10x10 bright block in the top-left corner, fully inside the 16 px border band.
# It imitates burnt-in scanner text / laterality markers known to cause shortcuts.
MARKER = (slice(3, 13), slice(3, 13))
DEPTHS = [1, 2, 4, 6, 12]
SHIFTED_DEVICE = {"gain": (0.25, 0.45), "gamma": (1.5, 2.0), "offset": (0.15, 0.30)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_split(n: int, seed: int, subtle: bool = False, device: dict | None = None,
               marker_rho: float | None = None, marker_on: str = "none"):
    """Balanced phantom split held in memory.

    marker_rho : training-time correlation. The marker is added to an MH scan with
                 probability rho and to a normal scan with probability 1 - rho
                 (rho = 0.5 -> uncorrelated, rho = 1.0 -> perfectly predictive).
    marker_on  : test-time placement when marker_rho is None:
                 "none", "mh" (same as training), "normal" (flipped), "all".
    """
    rng = np.random.default_rng(seed)
    imgs, labels, metas, marked = [], [], [], []
    for label in (0, 1):
        for _ in range(n // 2):
            img, meta = eng.synth_oct(label, rng, subtle=subtle, device=device)
            img = img.copy()
            if marker_rho is not None:
                p = marker_rho if label == 1 else 1.0 - marker_rho
                put = rng.random() < p
            else:
                put = {"none": False, "all": True, "mh": label == 1, "normal": label == 0}[marker_on]
            if put:
                img[MARKER] = 255
            imgs.append(img); labels.append(label); metas.append(meta); marked.append(bool(put))
    return np.stack(imgs), np.array(labels), metas, np.array(marked)


def normalise(img: np.ndarray, mode: str) -> np.ndarray:
    img = img.astype(np.float32)
    if mode == "percentile":
        return eng.percentile_normalize(img)
    if mode == "minmax":
        lo, hi = img.min(), img.max()
        return ((img - lo) / (hi - lo + 1e-6)).astype(np.float32)
    if mode == "none":
        return img / 255.0
    raise ValueError(mode)


class ArrayDS(eng.Dataset):
    def __init__(self, imgs, labels, norm="percentile", augment=None):
        self.imgs, self.labels, self.norm, self.augment = imgs, labels, norm, augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = normalise(self.imgs[i], self.norm)
        if self.augment is not None:
            x = self.augment(x)
        return eng.to_model_tensor(x), int(self.labels[i])


# ---------------------------------------------------------------------------
# Training / inference
# ---------------------------------------------------------------------------
def train_model(imgs, labels, seed: int, device: str, norm: str = "percentile", epochs: int = 4):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model, info = eng.create_model("vit", pretrained=True)
    if info["backbone"] != "vit" or not info["pretrained"]:
        raise RuntimeError("experiments need the pretrained ViT (internet for the first download)")
    model = model.to(device)
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(ArrayDS(imgs, labels, norm, eng.OCTAugment()), batch_size=16,
                        shuffle=True, drop_last=True, generator=g)
    lr = 1e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader), eta_min=lr * 0.01)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    return model.eval(), time.time() - t0


def multi_depth_cams(model, x, depths=DEPTHS):
    """Grad-CAM for the predicted class from every block in one backward pass;
    returns {L: HxW map averaged over the last L blocks}, probs."""
    import torch
    import torch.nn.functional as F
    blocks = model.net.blocks
    acts, grads = {}, {}

    def mk(i):
        def hook(m, inp, out):
            acts[i] = out
            out.register_hook(lambda gr: grads.__setitem__(i, gr))
        return hook

    handles = [blocks[i].norm1.register_forward_hook(mk(i)) for i in range(len(blocks))]
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            xx = x.clone().requires_grad_(True)
            logits = model(xx)
            probs = logits.softmax(1)
            logits[0, int(probs.argmax(1))].backward()
    finally:
        for h in handles:
            h.remove()
    per = {}
    for i in acts:
        A = model.to_spatial(acts[i].detach())
        G = model.to_spatial(grads[i].detach())
        cam = F.relu((G.mean((2, 3), keepdim=True) * A).sum(1, keepdim=True))
        lo, hi = cam.min(), cam.max()
        per[i] = (cam - lo) / (hi - lo + 1e-8)
    n = len(blocks)
    out = {}
    for L in depths:
        cam = torch.stack([per[i] for i in range(n - L, n)]).mean(0)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        out[L] = cam.detach().cpu().numpy().astype(np.float64)
    return out, probs.detach().cpu().numpy()[0]


def evaluate(model, imgs, labels, metas, device, norm="percentile", depths=DEPTHS, keep_idx=()):
    """Classification metrics + Grad-CAM verification for every depth L."""
    import torch
    ps, loc = [], {L: [] for L in depths}
    kept = []
    for i in range(len(imgs)):
        g = normalise(imgs[i], norm)
        x = eng.to_model_tensor(g).unsqueeze(0).to(device)
        cams, prob = multi_depth_cams(model, x, depths)
        ps.append(prob[1])
        gray_for_mask = eng.percentile_normalize(imgs[i].astype(np.float32))   # mask always on normalised anatomy
        for L in depths:
            m = eng.localization_metrics(cams[L], gray_for_mask, fovea_x=metas[i]["fovea_x"])
            m["marker_mass_frac"] = float(cams[L][MARKER].sum() / (cams[L].sum() + 1e-8))
            m["label"] = int(labels[i])
            loc[L].append(m)
        if i in keep_idx:
            kept.append((gray_for_mask, cams, float(prob[1]), int(labels[i])))
    p = np.array(ps)
    cls = eng.classification_metrics(labels, p)
    cls.pop("reliability_bins")
    out = {"classification": cls, "localization": {}}
    for L in depths:
        rows = loc[L]
        mh = [r for r in rows if r["label"] == 1]
        out["localization"][str(L)] = {
            "S_loc": float(np.mean([r["localization_score"] for r in rows])),
            "border_frac": float(np.mean([r["border_mass_frac"] for r in rows])),
            "marker_frac": float(np.mean([r["marker_mass_frac"] for r in rows])),
            "pointing_mh": float(np.mean([r["pointing_hit"] for r in mh])),
            "foveal_mass_mh": float(np.mean([r["foveal_mass_frac"] for r in mh])),
        }
    return out, kept


def summarise(cls):
    return {k: round(float(cls[k]), 4) for k in ("auc", "balanced_accuracy", "sensitivity", "specificity", "mcc", "ece")}


def save_panel(kept, path, L=1, title_prefix=""):
    tiles = []
    for gray, cams, p, label in kept:
        base = (np.repeat(gray[..., None], 3, -1) * 255).astype(np.uint8)
        tiles.append(np.concatenate([base, eng.overlay_cam(gray, cams[L])], 1))
    if tiles:
        Image.fromarray(np.concatenate(tiles, 0)).save(path)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def e0_metric_validation(n_val=100, seed=1000):
    """Known-answer test of the verification metrics with synthetic saliency maps (no model)."""
    imgs, labels, metas, _ = make_split(n_val, seed)
    yy, xx = np.mgrid[:eng.IMG_SIZE, :eng.IMG_SIZE]
    rng = np.random.default_rng(seed)

    def gauss(cx, cy, s):
        return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))

    def smooth_noise():
        z = rng.random((14, 14))
        return np.asarray(Image.fromarray((z * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0

    results = {}
    def run(name, fn):
        rows = []
        for i in range(len(imgs)):
            g = eng.percentile_normalize(imgs[i].astype(np.float32))
            cam = fn(metas[i], g).astype(np.float64) + 1e-6
            r = eng.localization_metrics(cam, g, fovea_x=metas[i]["fovea_x"])
            r["label"] = int(labels[i]); rows.append(r)
        mh = [r for r in rows if r["label"] == 1]
        results[name] = {
            "S_loc": float(np.mean([r["localization_score"] for r in rows])),
            "border_frac": float(np.mean([r["border_mass_frac"] for r in rows])),
            "pointing_mh": float(np.mean([r["pointing_hit"] for r in mh])),
            "foveal_mass_mh": float(np.mean([r["foveal_mass_frac"] for r in mh])),
        }

    fov = lambda m, g: gauss(m["fovea_x"], m["fovea_y"], 15)
    corner = lambda m, g: gauss(8, 8, 10)
    run("oracle_fovea", fov)
    run("off_fovea_tissue_60px", lambda m, g: gauss(m["fovea_x"] + (60 if m["fovea_x"] < 112 else -60), m["fovea_y"], 15))
    run("border_corner_shortcut", corner)
    for a in (0.75, 0.5, 0.25):
        run(f"mix_fovea_{a:.2f}_border", lambda m, g, a=a: a * fov(m, g) / fov(m, g).sum() + (1 - a) * corner(m, g) / corner(m, g).sum())
    run("uniform", lambda m, g: np.ones_like(g, dtype=np.float64))
    run("random_smooth", lambda m, g: smooth_noise())

    # Fovea estimator accuracy (used when no annotation exists), in- and out-of-distribution
    fov_err = {}
    for name, dev in (("in_distribution", None), ("shifted_device", SHIFTED_DEVICE)):
        im2, lb2, me2, _ = make_split(n_val, seed + 1, device=dev)
        errs = np.array([abs(eng.estimate_fovea_x(eng.percentile_normalize(im2[i].astype(np.float32))) - me2[i]["fovea_x"])
                         for i in range(len(im2))])
        fov_err[name] = {"mae_px": float(errs.mean()), "max_px": float(errs.max()),
                         "within_tol_frac": float((errs <= eng.FOVEA_TOL_PX).mean()),
                         "mae_px_normal": float(errs[lb2 == 0].mean()), "mae_px_mh": float(errs[lb2 == 1].mean())}

    # Intensity standardisation: spread of the per-scan RPE-band intensity across scanners
    def rpe_level(img, norm):
        g = normalise(img, norm)
        return float(np.percentile(g, 99.5))      # brightest band ~ RPE
    def tissue_contrast(img, norm):
        g = normalise(img, norm)
        return float(np.percentile(g, 95) - np.percentile(g, 20))
    std = {}
    im_in, _, _, _ = make_split(n_val, seed + 2)
    im_sh, _, _, _ = make_split(n_val, seed + 3, device=SHIFTED_DEVICE)
    for norm in ("none", "minmax", "percentile"):
        c_in = np.array([tissue_contrast(i, norm) for i in im_in])
        c_sh = np.array([tissue_contrast(i, norm) for i in im_sh])
        std[norm] = {"contrast_mean_in": float(c_in.mean()), "contrast_cv_in": float(c_in.std() / c_in.mean()),
                     "contrast_mean_shift": float(c_sh.mean()), "contrast_cv_shift": float(c_sh.std() / c_sh.mean()),
                     "contrast_ratio_shift_vs_in": float(c_sh.mean() / c_in.mean())}
    return {"saliency_controls": results, "fovea_estimator": fov_err, "intensity_standardisation": std,
            "n_scans": n_val}


def e1_e2_baseline(seeds, device, n_train, n_val):
    runs = []
    for s in seeds:
        tr = make_split(n_train, seed=10 * s + 1)
        va = make_split(n_val, seed=10 * s + 2)
        model, t = train_model(tr[0], tr[1], seed=s, device=device)
        h = n_val // 2                                  # first half normal, second half MH
        ev, kept = evaluate(model, va[0], va[1], va[2], device,
                            keep_idx={h, h + 1, 0} if s == seeds[0] else ())
        if kept:
            order = [k for k in kept if k[3] == 1] + [k for k in kept if k[3] == 0]
            save_panel(order, OUT / "e1_gradcam_L1.png", L=1)
            save_panel(order, OUT / "e1_gradcam_L4.png", L=4)
        runs.append({"seed": s, "train_s": round(t, 1), "classification": summarise(ev["classification"]),
                     "confusion": ev["classification"]["confusion_matrix"], "localization": ev["localization"]})
        print(f"  [E1] seed {s}: AUC {ev['classification']['auc']:.3f}  MCC {ev['classification']['mcc']:.3f}  "
              f"S_loc(L=1) {ev['localization']['1']['S_loc']:.3f}  S_loc(L=4) {ev['localization']['4']['S_loc']:.3f}  ({t:.1f}s)")

    def agg(get):
        v = np.array([get(r) for r in runs], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}

    summary = {"classification": {k: agg(lambda r, k=k: r["classification"][k]) for k in runs[0]["classification"]},
               "localization": {str(L): {k: agg(lambda r, L=L, k=k: r["localization"][str(L)][k])
                                         for k in runs[0]["localization"][str(L)]} for L in DEPTHS},
               "train_s": agg(lambda r: r["train_s"])}
    return {"runs": runs, "summary": summary, "n_train": n_train, "n_val": n_val}


def e3_shortcut(device, n_train, n_val, seed=0):
    out = {}
    for subtle in (False, True):
        for rho in (0.5, 1.0):
            key = f"{'subtle' if subtle else 'standard'}_rho{rho}"
            tr = make_split(n_train, seed=101, subtle=subtle, marker_rho=rho)
            model, t = train_model(tr[0], tr[1], seed=seed, device=device)
            res = {"train_s": round(t, 1)}
            for test in ("none", "mh", "normal"):
                va = make_split(n_val, seed=202, subtle=subtle, marker_on=test)
                ev, kept = evaluate(model, va[0], va[1], va[2], device, depths=[1, 4],
                                    keep_idx={0, 1} if (test == "normal" and rho == 1.0) else ())
                res[test] = {"classification": summarise(ev["classification"]),
                             "confusion": ev["classification"]["confusion_matrix"],
                             "localization": ev["localization"]}
                if kept:   # show flipped-marker NORMAL scans (marker present, no hole)
                    normals = [k for k in kept if k[3] == 0][:2]
                    save_panel(normals, OUT / f"e3_{key}_flipped_normals_L1.png", L=1)
                print(f"  [E3] {key:18s} test={test:6s}  BA {ev['classification']['balanced_accuracy']:.3f}  "
                      f"AUC {ev['classification']['auc']:.3f}  border(L=1) {ev['localization']['1']['border_frac']:.3f}  "
                      f"marker(L=1) {ev['localization']['1']['marker_frac']:.3f}  S_loc {ev['localization']['1']['S_loc']:.3f}")
            out[key] = res
    return out


def e4_normalisation(device, n_train, n_val, seed=0):
    out = {}
    tr = make_split(n_train, seed=301)
    va_in = make_split(n_val, seed=302)
    va_sh = make_split(n_val, seed=303, device=SHIFTED_DEVICE)
    for norm in ("percentile", "minmax", "none"):
        model, t = train_model(tr[0], tr[1], seed=seed, device=device, norm=norm)
        res = {"train_s": round(t, 1)}
        for name, va in (("in_distribution", va_in), ("shifted_device", va_sh)):
            ev, _ = evaluate(model, va[0], va[1], va[2], device, norm=norm, depths=[1])
            res[name] = {"classification": summarise(ev["classification"]),
                         "confusion": ev["classification"]["confusion_matrix"],
                         "localization": ev["localization"]["1"]}
            print(f"  [E4] norm={norm:10s} {name:16s}  BA {ev['classification']['balanced_accuracy']:.3f}  "
                  f"AUC {ev['classification']['auc']:.3f}  sens {ev['classification']['sensitivity']:.3f}  "
                  f"spec {ev['classification']['specificity']:.3f}")
        out[norm] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["E0", "E1", "E3", "E4"],
                    help="subset of E0 E1 E3 E4 (E2 is computed inside E1)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--n-train", type=int, default=160)
    ap.add_argument("--n-val", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}

    todo = [x.upper() for x in args.only]
    if "E0" in todo:
        print("[E0] metric validation (no training)")
        results["E0"] = e0_metric_validation()
        print(json.dumps(results["E0"]["saliency_controls"], indent=1))
    if any(x in todo for x in ("E1", "E2", "E3", "E4")):
        import torch
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        torch.set_num_threads(max(1, __import__("os").cpu_count() or 1))
        results["env"] = {"torch": torch.__version__, "device": device}
        print(f"[env] torch {torch.__version__} | device {device}")
        t0 = time.time()
        if "E1" in todo or "E2" in todo:
            print("[E1/E2] multi-seed baseline + Grad-CAM depth ablation")
            results["E1"] = e1_e2_baseline(args.seeds, device, args.n_train, args.n_val)
        if "E3" in todo:
            print("[E3] shortcut injection")
            results["E3"] = e3_shortcut(device, args.n_train, args.n_val)
        if "E4" in todo:
            print("[E4] normalisation under device shift")
            results["E4"] = e4_normalisation(device, args.n_train, args.n_val)
        results["total_s"] = round(time.time() - t0, 1)
    path.write_text(json.dumps(results, indent=2, default=eng._json_default))
    print(f"\n[save] {path}\n[done] send outputs/experiments/results.json (and the PNGs) back for the paper")


if __name__ == "__main__":
    main()
