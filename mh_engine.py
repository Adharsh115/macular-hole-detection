#!/usr/bin/env python3
"""
mh_engine.py — Macular Hole (MH) Detection & Foveal Localization Engine
=========================================================================

End-to-end pipeline for retinal OCT B-scans:

  1. Synthetic OCT dataset generation (layered retina, foveal pit, full-thickness
     macular hole with cystic rims) so the pipeline runs instantly without data.
  2. OCT-specific preprocessing / augmentation:
       * Percentile Intensity Normalisation (p1–p99) -> scanner-contrast invariance
       * Horizontal flip (left/right eye symmetry)
       * Mild vertical shift (retina position varies with scan alignment)
       * Mild gamma jitter (device gain differences)
  3. Method 1 — Classifier
       * Primary  : ViT-Tiny (timm `vit_tiny_patch16_224`, ImageNet-pretrained)
       * Fallback : EfficientNet-B0 (torchvision) + CBAM-style Spatial Attention
       * AdamW + cosine annealing + label smoothing, 3–5 epochs.
  4. Method 2 — Grad-CAM explainability + structural verification
       * Grad-CAM on the last transformer block (token grid -> 14x14 map)
         or on the spatial-attention output for the CNN fallback.
       * Localization Score = M_tissue / (M_tissue + M_border)
       * Foveal pointing game against the known fovea position.
  5. Clinical evaluation report: AUC (95% bootstrap CI), Balanced Accuracy,
     Sensitivity, Specificity, F1, MCC, ECE (+ reliability bins).

Outputs
-------
  mh_model_weights.pth           trained weights + model config
  outputs/performance_report.json
  outputs/gradcam_examples/*.png Grad-CAM overlays on validation scans
  outputs/roc_reliability.png    (only if matplotlib is installed)

Usage
-----
  python mh_engine.py                      # generate data (if missing), train, evaluate
  python mh_engine.py --epochs 5 --regen   # regenerate synthetic data
  python mh_engine.py --backbone effnet    # force CNN fallback
  python mh_engine.py --no-pretrained      # offline: no weight download

IMPORTANT: metrics on the synthetic dataset demonstrate that the pipeline works;
they are NOT evidence of clinical performance. Replace ./data with real labelled
OCT B-scans (e.g. the OCTID dataset's Normal and Macular Hole classes) for any
reportable result. Research use only — not a diagnostic device.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
IMG_SIZE = 224
CLASS_NAMES = ["Normal", "Macular Hole"]          # index 0, index 1
CLASS_DIRS = ["normal", "mh"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
WEIGHTS_PATH = ROOT / "mh_model_weights.pth"
BORDER_PX = 16                                    # width of the "border" band for leakage checks
FOVEA_TOL_PX = 28                                 # pointing-game tolerance (horizontal)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===========================================================================
# 1. Synthetic OCT B-scan generator
# ===========================================================================
def _layer_profile(t: np.ndarray) -> np.ndarray:
    """Map relative retinal depth t (0 = ILM, 1 = RPE) to OCT reflectivity.

    Approximates the classic hyper/hypo-reflective banding:
    NFL (bright) / GCL-IPL / INL (dark) / OPL / ONL (dark) / EZ (bright) / RPE (brightest).
    """
    conds = [
        t < 0.0,                      # vitreous
        t < 0.08,                     # nerve fibre layer
        t < 0.35,                     # GCL + IPL
        t < 0.45,                     # inner nuclear layer
        t < 0.55,                     # outer plexiform layer
        t < 0.83,                     # outer nuclear layer
        t < 0.87,                     # ellipsoid zone
        t < 0.90,                     # interdigitation gap
        t < 1.00,                     # RPE / Bruch
    ]
    vals = [0.04, 0.78, 0.52, 0.28, 0.46, 0.17, 0.82, 0.35, 0.97]
    choroid = 0.40 * np.exp(-np.clip(t - 1.0, 0, None) * 2.5)  # decaying choroid signal
    return np.select(conds, vals, default=choroid)


def synth_oct(label: int, rng: np.random.Generator, size: int = IMG_SIZE,
              subtle: bool = False, device: dict | None = None) -> tuple[np.ndarray, dict]:
    """Generate one synthetic foveal OCT B-scan.

    label 0 -> normal foveal pit, label 1 -> full-thickness macular hole.
    subtle -> narrower, partially reflective hole with fewer secondary cues
              (used to make the true lesion signal harder in experiments).
    device -> override scanner response ranges, e.g. {"gain": (0.25, 0.45),
              "gamma": (1.5, 2.0), "offset": (0.15, 0.3)} for a device shift.
    Returns (uint8 image HxW, metadata with ground-truth fovea location).
    Defaults reproduce the original generator exactly.
    """
    H = W = size
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)[:, None]

    fx = W / 2 + rng.uniform(-20, 20)                   # fovea column (ground truth)
    base_top = 78 + rng.uniform(-14, 14)                 # vertical scan alignment
    tilt = rng.uniform(-0.06, 0.06)                      # scan tilt
    curv = rng.uniform(-0.0012, 0.0012)                  # globe curvature
    thickness = rng.uniform(62, 76)

    top = base_top + tilt * (x - W / 2) + curv * (x - W / 2) ** 2
    rpe = top + thickness
    d = x - fx

    hole_w0 = 0.0
    if label == 0:
        # Normal: smooth foveal depression (pit)
        depth, sigma = rng.uniform(14, 24), rng.uniform(13, 20)
        top = top + depth * np.exp(-(d ** 2) / (2 * sigma ** 2))
    else:
        # Macular hole: elevated, thickened rims around a full-thickness defect
        hole_w0 = rng.uniform(2, 5) if subtle else rng.uniform(6, 14)   # half-width at the apex
        rim_h = rng.uniform(2, 6) if subtle else rng.uniform(6, 14)
        top = top - rim_h * np.exp(-((np.abs(d) - hole_w0 - 7) ** 2) / (2 * 7.0 ** 2))

    t = (y - top[None, :]) / (rpe - top)[None, :]
    img = _layer_profile(t).astype(np.float32)

    if label == 1:
        # Full-thickness defect: hourglass-shaped gap from ILM down to (not through) the RPE
        widen = rng.uniform(2, 6) if subtle else rng.uniform(6, 14)
        half_w = hole_w0 + widen * np.clip(t, 0, 1)                      # wider at base
        defect = (np.abs(d)[None, :] < half_w) & (t >= -0.05) & (t < 0.86)
        fill = 0.22 if subtle else 0.05                                   # subtle: partly reflective
        img[defect] = fill + 0.03 * rng.random(defect.sum())
        # Intraretinal cysts in the hole rims (hypo-reflective ellipses)
        cyst = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(cyst)
        for side in (-1, 1):
            for _ in range(rng.integers(0, 2) if subtle else rng.integers(1, 4)):
                cx = fx + side * (hole_w0 + rng.uniform(8, 20))
                cxi = int(np.clip(cx, 0, W - 1))
                cy = top[cxi] + rng.uniform(0.35, 0.7) * (rpe[cxi] - top[cxi])
                rx, ry = rng.uniform(3, 6), rng.uniform(4, 8)
                draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        cmask = np.asarray(cyst) > 0
        img[cmask] = 0.08
        # Occasional operculum floating above the hole
        if not subtle and rng.random() < 0.4:
            oy = int(top[int(np.clip(fx, 0, W - 1))] - rng.uniform(12, 24))
            op = Image.new("L", (W, H), 0)
            ImageDraw.Draw(op).ellipse([fx - 5, oy - 2, fx + 5, oy + 2], fill=255)
            img[np.asarray(op) > 0] = 0.6

    # Speckle (multiplicative gamma noise) + additive noise, then mild blur
    img = img * rng.gamma(4.0, 1 / 4.0, size=img.shape) + rng.normal(0, 0.03, img.shape)
    img = np.clip(img, 0, 1)
    pil = Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    img = np.asarray(pil).astype(np.float32) / 255.0

    # Scanner-specific gain / gamma / offset (motivates percentile normalisation)
    dev = {"gain": (0.6, 1.3), "gamma": (0.7, 1.4), "offset": (0.0, 0.12), **(device or {})}
    gain, gamma, offset = rng.uniform(*dev["gain"]), rng.uniform(*dev["gamma"]), rng.uniform(*dev["offset"])
    img = np.clip(offset + gain * img ** gamma, 0, 1)

    fxi = int(round(np.clip(fx, 0, W - 1)))
    meta = {
        "label": int(label),
        "fovea_x": float(fx),
        "fovea_y": float((top[fxi] + rpe[fxi]) / 2),
        "hole_half_width_px": float(hole_w0),
    }
    return (img * 255).astype(np.uint8), meta


def build_synthetic_dataset(data_dir: Path, n_train: int, n_val: int, seed: int = 0, force: bool = False) -> None:
    """Create ./data/{train,val}/{normal,mh}/*.png plus metadata.json (fovea ground truth)."""
    marker = data_dir / "metadata.json"
    if marker.exists() and not force:
        print(f"[data] Using existing dataset at {data_dir}")
        return
    rng = np.random.default_rng(seed)
    meta_all = {}
    for split, n in (("train", n_train), ("val", n_val)):
        for label, cdir in enumerate(CLASS_DIRS):
            out = data_dir / split / cdir
            out.mkdir(parents=True, exist_ok=True)
            for f in out.glob("*.png"):
                f.unlink()
            for i in range(n // 2):
                img, meta = synth_oct(label, rng)
                name = f"{cdir}_{split}_{i:04d}.png"
                Image.fromarray(img).save(out / name)
                meta_all[f"{split}/{cdir}/{name}"] = meta
    marker.write_text(json.dumps(meta_all, indent=1))
    print(f"[data] Synthetic OCT dataset written to {data_dir} "
          f"({n_train} train / {n_val} val, balanced)")


# ===========================================================================
# 2. OCT preprocessing and augmentation
# ===========================================================================
def percentile_normalize(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Percentile Intensity Normalisation: clip to [p_lo, p_hi] and rescale to [0, 1].

    Robust to the very different gain/contrast settings of Heidelberg, Zeiss,
    Topcon, Optovue ... devices, and to saturated speckle outliers.
    """
    arr = arr.astype(np.float32)
    p_lo, p_hi = np.percentile(arr, [lo, hi])
    if p_hi - p_lo < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - p_lo) / (p_hi - p_lo), 0.0, 1.0).astype(np.float32)


def load_gray(src) -> np.ndarray:
    """Load an image (path / PIL / file-like) as grayscale float32 HxW resized to IMG_SIZE."""
    img = src if isinstance(src, Image.Image) else Image.open(src)
    img = img.convert("L")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img).astype(np.float32)


def to_model_tensor(gray01: np.ndarray) -> torch.Tensor:
    """Grayscale [0,1] HxW -> ImageNet-normalised 3xHxW tensor (pretrained backbones expect RGB)."""
    t = torch.from_numpy(np.ascontiguousarray(gray01)).float().unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


class OCTAugment:
    """Retina-aware augmentation applied AFTER percentile normalisation."""

    def __init__(self, p_flip: float = 0.5, max_vshift: float = 0.08, gamma_range=(0.85, 1.15)):
        self.p_flip, self.max_vshift, self.gamma_range = p_flip, max_vshift, gamma_range

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if random.random() < self.p_flip:                      # OD/OS mirror symmetry
            x = x[:, ::-1]
        shift = int(round(random.uniform(-self.max_vshift, self.max_vshift) * x.shape[0]))
        if shift:                                              # mild vertical shift, zero fill
            out = np.zeros_like(x)
            if shift > 0:
                out[shift:] = x[:-shift]
            else:
                out[:shift] = x[-shift:]
            x = out
        g = random.uniform(*self.gamma_range)                  # mild device-gain jitter
        return np.clip(x, 0, 1) ** g


def preprocess(src) -> tuple[torch.Tensor, np.ndarray]:
    """Inference preprocessing. Returns (1x3xHxW tensor, display image in [0,1])."""
    gray01 = percentile_normalize(load_gray(src))
    return to_model_tensor(gray01).unsqueeze(0), gray01


class OCTFolder(Dataset):
    def __init__(self, root: Path, split: str, augment: OCTAugment | None = None):
        self.items: list[tuple[Path, int]] = []
        for label, cdir in enumerate(CLASS_DIRS):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
                self.items += [(p, label) for p in sorted((root / split / cdir).glob(ext))]
        if not self.items:
            raise FileNotFoundError(f"No images found under {root / split}")
        self.augment, self.root = augment, root

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        x = percentile_normalize(load_gray(path))
        if self.augment is not None:
            x = self.augment(x)
        return to_model_tensor(x), label, str(path.relative_to(self.root))


# ===========================================================================
# 3. Models: ViT-Tiny (primary) and EfficientNet-B0 + Spatial Attention (fallback)
# ===========================================================================
class SpatialAttention(nn.Module):
    """CBAM-style spatial attention: re-weights each location of the feature map."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(desc))


class EffNetSA(nn.Module):
    def __init__(self, pretrained: bool, num_classes: int = 2):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)
        self.features = base.features                     # B x 1280 x 7 x 7
        self.attn = SpatialAttention()
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(1280, num_classes))

    def forward(self, x):
        return self.head(self.attn(self.features(x)))


class MHNet(nn.Module):
    """Unified wrapper exposing the Grad-CAM target layer and a token->grid converter."""

    def __init__(self, backbone: str, pretrained: bool):
        super().__init__()
        self.backbone_kind = backbone
        if backbone == "vit":
            import timm
            self.net = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained,
                                         num_classes=2, drop_path_rate=0.1)
            # Multi-layer Grad-CAM: inputs (norm1) of the last 4 transformer blocks.
            # Last-block-only Grad-CAM is noisy on ViTs; averaging deeper blocks gives
            # more spatially coherent maps.
            self.cam_layers = [blk.norm1 for blk in self.net.blocks[-4:]]
            self.n_prefix = getattr(self.net, "num_prefix_tokens", 1)
        elif backbone == "effnet":
            self.net = EffNetSA(pretrained=pretrained)
            self.cam_layers = [self.net.attn]                 # attention-weighted 7x7 features
            self.n_prefix = 0
        else:
            raise ValueError(backbone)

    def forward(self, x):
        return self.net(x)

    def to_spatial(self, t: torch.Tensor) -> torch.Tensor:
        """Convert hooked activations/gradients to B x C x h x w."""
        if t.dim() == 4:
            return t
        tokens = t[:, self.n_prefix:, :]                      # drop [CLS] token
        b, n, c = tokens.shape
        s = int(math.isqrt(n))
        return tokens.reshape(b, s, s, c).permute(0, 3, 1, 2)


def _has_timm() -> bool:
    try:
        import timm  # noqa: F401
        return True
    except ImportError:
        return False


def create_model(backbone: str = "auto", pretrained: bool = True) -> tuple[MHNet, dict]:
    """Build the classifier with clean fallbacks.

    auto: ViT-Tiny (pretrained) -> EfficientNet-B0+SA (pretrained) -> EfficientNet-B0+SA (scratch).
    A ViT trained from scratch on a few hundred images is unreliable, so if
    ViT weights cannot be downloaded we switch to the CNN rather than keep the ViT.
    """
    order = {"auto": ["vit", "effnet"], "vit": ["vit"], "effnet": ["effnet"]}[backbone]
    for kind in order:
        if kind == "vit" and not _has_timm():
            print("[model] timm not installed -> falling back to EfficientNet-B0 + Spatial Attention")
            continue
        try:
            model = MHNet(kind, pretrained=pretrained)
            print(f"[model] {kind} backbone ready (ImageNet weights: {pretrained})")
            return model, {"backbone": kind, "pretrained": pretrained}
        except Exception as e:  # typically: no internet for weight download
            print(f"[model] could not load pretrained {kind}: {type(e).__name__}: {e}")
            if kind == "vit" and backbone == "vit":
                print("[model] training ViT from scratch (expect weaker results)")
                return MHNet("vit", pretrained=False), {"backbone": "vit", "pretrained": False}
    print("[model] using EfficientNet-B0 + Spatial Attention from scratch (offline mode)")
    return MHNet("effnet", pretrained=False), {"backbone": "effnet", "pretrained": False}


def load_trained_model(path: Path = WEIGHTS_PATH, device: str = "cpu") -> tuple[MHNet, dict]:
    """Rebuild the architecture WITHOUT downloading weights, then load the fine-tuned state."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MHNet(ckpt["config"]["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["config"]


# ===========================================================================
# 4. Grad-CAM + structural verification
# ===========================================================================
class GradCAM:
    """Gradient-weighted Class Activation Mapping (Selvaraju et al., ICCV 2017).

    For ViT the hooked tensors are the (B, 1+N, C) token sequences entering each of
    the last 4 blocks; patch tokens are reshaped to a 14x14 grid, a Grad-CAM map is
    computed per block, and the normalised maps are averaged (multi-layer Grad-CAM).
    Channel weights are the spatially averaged gradients of the target logit.
    """

    def __init__(self, model: MHNet):
        self.model = model
        self.acts: dict = {}
        self.grads: dict = {}
        self.handles = [layer.register_forward_hook(self._make_hook(i))
                        for i, layer in enumerate(model.cam_layers)]

    def _make_hook(self, i: int):
        def hook(module, inp, out):
            self.acts[i] = out
            if out.requires_grad:
                out.register_hook(lambda g: self.grads.__setitem__(i, g))
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    @staticmethod
    def _norm(cam: torch.Tensor) -> torch.Tensor:
        lo = cam.flatten(1).min(1)[0].view(-1, 1, 1, 1)
        hi = cam.flatten(1).max(1)[0].view(-1, 1, 1, 1)
        return (cam - lo) / (hi - lo + 1e-8)

    def __call__(self, x: torch.Tensor, target: int | None = None):
        """Returns (cams B x H x W in [0,1], probs B x 2, explained class indices)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.acts, self.grads = {}, {}
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            probs = logits.softmax(1)
            idx = probs.argmax(1) if target is None else torch.full((x.size(0),), target, device=x.device)
            logits.gather(1, idx[:, None]).sum().backward()
        maps = []
        for i in self.acts:
            if i not in self.grads:
                continue
            A = self.model.to_spatial(self.acts[i].detach())
            G = self.model.to_spatial(self.grads[i].detach())
            weights = G.mean(dim=(2, 3), keepdim=True)
            maps.append(self._norm(F.relu((weights * A).sum(1, keepdim=True))))
        cam = torch.stack(maps).mean(0)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = self._norm(cam)[:, 0]
        return cam.cpu().numpy(), probs.detach().cpu().numpy(), idx.cpu().numpy()


def retina_mask(gray01: np.ndarray, thr: float = 0.35) -> np.ndarray:
    """Coarse retinal tissue mask: smoothed bright pixels, filled vertically per column
    between the ILM and the deepest bright band (so hypo-reflective ONL is included)."""
    sm = np.asarray(Image.fromarray((gray01 * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)))
    bright = sm.astype(np.float32) / 255.0 > thr
    mask = np.zeros_like(bright)
    for c in range(bright.shape[1]):
        rows = np.flatnonzero(bright[:, c])
        if rows.size:
            mask[rows[0]: rows[-1] + 1, c] = True
    return mask


def estimate_fovea_x(gray01: np.ndarray) -> int:
    """Fovea column ≈ deepest ILM position in the central half of the scan
    (the pit in a normal eye; the bottom of the defect in a macular hole)."""
    sm = np.asarray(Image.fromarray((gray01 * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)))
    bright = sm.astype(np.float32) / 255.0 > 0.35
    H, W = bright.shape
    first = np.array([np.argmax(bright[:, c]) if bright[:, c].any() else 0 for c in range(W)], dtype=float)
    first = np.convolve(first, np.ones(9) / 9, mode="same")
    lo, hi = W // 4, 3 * W // 4
    return int(lo + np.argmax(first[lo:hi]))


def localization_metrics(cam: np.ndarray, gray01: np.ndarray, fovea_x: float | None = None) -> dict:
    """Structural verification of a Grad-CAM map.

    Localization Score = M_tissue / (M_tissue + M_border)
      M_tissue: CAM mass on retinal tissue (excluding the border band)
      M_border: CAM mass inside a BORDER_PX band along the image edges, where no
                pathology exists — high values indicate shortcut learning
                (edge artefacts, burnt-in text, padding).
    Foveal pointing game: is the CAM peak within FOVEA_TOL_PX of the fovea column?
    """
    H, W = cam.shape
    border = np.zeros((H, W), bool)
    border[:BORDER_PX], border[-BORDER_PX:], border[:, :BORDER_PX], border[:, -BORDER_PX:] = True, True, True, True
    tissue = retina_mask(gray01) & ~border
    cam = cam.astype(np.float64)
    total = float(cam.sum()) + 1e-8
    m_tissue, m_border = float(cam[tissue].sum()), float(cam[border].sum())
    fx = estimate_fovea_x(gray01) if fovea_x is None else fovea_x
    foveal_win = np.zeros((H, W), bool)
    foveal_win[:, max(0, int(fx - FOVEA_TOL_PX)): int(fx + FOVEA_TOL_PX) + 1] = True
    py, px = np.unravel_index(np.argmax(cam), cam.shape)
    return {
        "localization_score": m_tissue / (m_tissue + m_border + 1e-8),
        "tissue_mass_frac": m_tissue / total,
        "border_mass_frac": m_border / total,
        "foveal_mass_frac": float(cam[foveal_win & tissue].sum() / total),
        "cam_peak_xy": [int(px), int(py)],
        "fovea_x": float(fx),
        "pointing_hit": bool(abs(px - fx) <= FOVEA_TOL_PX),
    }


def jet(v: np.ndarray) -> np.ndarray:
    """Dependency-free 'jet' colormap: [0,1] -> HxWx3 float."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], -1)


def overlay_cam(gray01: np.ndarray, cam: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Blend CAM over the scan; low-attention regions stay grayscale for readability."""
    base = np.repeat(gray01[..., None], 3, -1)
    a = alpha * np.clip(cam * 1.3, 0, 1)[..., None] ** 0.8
    return (np.clip((1 - a) * base + a * jet(cam), 0, 1) * 255).astype(np.uint8)


# ===========================================================================
# 5. Clinical metrics (NumPy only — no scikit-learn dependency)
# ===========================================================================
def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC via the Mann–Whitney U statistic (ties handled with average ranks)."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i: j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def roc_curve_points(y: np.ndarray, s: np.ndarray) -> tuple[list, list]:
    thr = np.r_[np.inf, np.sort(np.unique(s))[::-1]]
    P, N = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    tpr = [float(((s >= t) & (y == 1)).sum() / P) for t in thr]
    fpr = [float(((s >= t) & (y == 0)).sum() / N) for t in thr]
    return fpr, tpr


def expected_calibration_error(y: np.ndarray, p_pos: np.ndarray, n_bins: int = 10):
    """ECE over the confidence of the predicted class (Guo et al., 2017)."""
    pred = (p_pos >= 0.5).astype(int)
    conf = np.where(pred == 1, p_pos, 1 - p_pos)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)             # binary: confidence >= 0.5
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            bins.append({"bin": [float(lo), float(hi)], "n": int(m.sum()),
                         "accuracy": float(correct[m].mean()), "confidence": float(conf[m].mean())})
    return float(ece), bins


def classification_metrics(y: np.ndarray, p_pos: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p_pos >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    ece, bins = expected_calibration_error(y, p_pos)
    return {
        "auc": roc_auc(y, p_pos), "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens, "specificity": spec, "precision_ppv": prec,
        "npv": tn / (tn + fn) if tn + fn else 0.0, "f1": f1, "mcc": mcc,
        "ece": ece, "accuracy": (tp + tn) / len(y),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "reliability_bins": bins, "threshold": thr,
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn, n_boot: int = 1000, seed: int = 0):
    """Percentile bootstrap 95% CI (stratified by class to keep both classes present)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        vals.append(fn(y[idx], p[idx]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


# ===========================================================================
# 6. Training and evaluation
# ===========================================================================
@dataclass
class TrainConfig:
    epochs: int = 4
    batch_size: int = 16
    lr: float | None = None            # chosen per backbone if None
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    time_budget_s: float = 110.0       # stop starting new epochs after this


def train(model: MHNet, info: dict, cfg: TrainConfig, device: str):
    train_ds = OCTFolder(DATA_DIR, "train", augment=OCTAugment())
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)
    if cfg.lr is None:
        cfg.lr = (1e-4 if info["backbone"] == "vit" else 3e-4) if info["pretrained"] else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=cfg.lr * 0.01)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    print(f"[train] {len(train_ds)} images | {cfg.epochs} epochs | lr {cfg.lr:g} | "
          f"AdamW wd {cfg.weight_decay} | cosine | label smoothing {cfg.label_smoothing}")
    history, t0 = [], time.time()
    for epoch in range(1, cfg.epochs + 1):
        if time.time() - t0 > cfg.time_budget_s:
            print(f"[train] time budget reached, stopping before epoch {epoch}")
            break
        model.train()
        run_loss, run_correct, n = 0.0, 0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run_loss += loss.item() * x.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        history.append({"epoch": epoch, "loss": run_loss / n, "train_acc": run_correct / n,
                        "elapsed_s": round(time.time() - t0, 1)})
        print(f"[train] epoch {epoch}/{cfg.epochs}  loss {run_loss / n:.4f}  "
              f"acc {run_correct / n:.3f}  ({time.time() - t0:.1f}s)")
    return history, time.time() - t0


@torch.no_grad()
def predict_split(model: MHNet, split: str, device: str):
    ds = OCTFolder(DATA_DIR, split)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    ys, ps, names = [], [], []
    model.eval()
    for x, y, nm in loader:
        ps.append(model(x.to(device)).softmax(1)[:, 1].cpu().numpy())
        ys.append(y.numpy()); names += list(nm)
    return np.concatenate(ys), np.concatenate(ps), names


def evaluate_localization(model: MHNet, device: str, max_images: int = 40, n_save: int = 6):
    """Grad-CAM over validation scans: localization score, border leakage, pointing game."""
    meta_path = DATA_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ds = OCTFolder(DATA_DIR, "val")
    cam_engine = GradCAM(model)
    save_dir = OUT_DIR / "gradcam_examples"
    save_dir.mkdir(parents=True, exist_ok=True)
    rows, saved = [], 0
    idxs = np.linspace(0, len(ds) - 1, min(max_images, len(ds))).astype(int)
    for i in idxs:
        path, label = ds.items[i]
        x, gray = preprocess(path)
        cams, probs, cls = cam_engine(x.to(device))
        key = str(path.relative_to(DATA_DIR)).replace(os.sep, "/")
        gt_fx = meta.get(key, {}).get("fovea_x")
        m = localization_metrics(cams[0], gray, fovea_x=gt_fx)
        m.update({"file": key, "label": label, "p_mh": float(probs[0, 1]), "fovea_source": "ground_truth" if gt_fx is not None else "estimated"})
        rows.append(m)
        if saved < n_save and (label == 1 or saved % 2 == 1):
            side = np.concatenate([(np.repeat(gray[..., None], 3, -1) * 255).astype(np.uint8),
                                   overlay_cam(gray, cams[0])], axis=1)
            Image.fromarray(side).save(save_dir / f"{Path(key).stem}_gradcam.png")
            saved += 1
    cam_engine.remove()
    mh = [r for r in rows if r["label"] == 1]
    summary = {
        "n_images": len(rows),
        "mean_localization_score": float(np.mean([r["localization_score"] for r in rows])),
        "mean_border_mass_frac": float(np.mean([r["border_mass_frac"] for r in rows])),
        "mean_foveal_mass_frac_mh": float(np.mean([r["foveal_mass_frac"] for r in mh])) if mh else None,
        "pointing_game_acc_mh": float(np.mean([r["pointing_hit"] for r in mh])) if mh else None,
        "border_px": BORDER_PX, "fovea_tolerance_px": FOVEA_TOL_PX,
    }
    summary["passes_shortcut_check"] = bool(summary["mean_localization_score"] >= 0.8
                                            and summary["mean_border_mass_frac"] <= 0.10)
    return summary, rows


def save_figures(y, p, metrics) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fpr, tpr = roc_curve_points(y, p)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(fpr, tpr, lw=2, label=f"AUC = {metrics['auc']:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", c="gray", lw=1)
    ax[0].set(xlabel="1 − Specificity", ylabel="Sensitivity", title="ROC — MH vs Normal")
    ax[0].legend(loc="lower right")
    bins = metrics["reliability_bins"]
    ax[1].plot([0.5, 1], [0.5, 1], "--", c="gray", lw=1)
    ax[1].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", lw=2)
    ax[1].set(xlabel="Confidence", ylabel="Accuracy", title=f"Reliability (ECE = {metrics['ece']:.3f})")
    fig.tight_layout()
    out = OUT_DIR / "roc_reliability.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def print_report(m: dict, ci: dict, loc: dict, info: dict, train_s: float) -> None:
    bar = "=" * 66
    print(f"\n{bar}\n  MACULAR HOLE DETECTION — CLINICAL EVALUATION REPORT (validation)\n{bar}")
    print(f"  Backbone           : {info['backbone']}  (ImageNet init: {info['pretrained']})")
    print(f"  Training time      : {train_s:.1f} s")
    print(f"  Decision threshold : p(MH) >= {m['threshold']}")
    print("-" * 66)
    print(f"  AUC-ROC            : {m['auc']:.4f}   95% CI [{ci['auc'][0]:.3f}, {ci['auc'][1]:.3f}]")
    print(f"  Balanced Accuracy  : {m['balanced_accuracy']:.4f}   95% CI "
          f"[{ci['balanced_accuracy'][0]:.3f}, {ci['balanced_accuracy'][1]:.3f}]")
    print(f"  Sensitivity (TPR)  : {m['sensitivity']:.4f}")
    print(f"  Specificity (TNR)  : {m['specificity']:.4f}")
    print(f"  PPV / NPV          : {m['precision_ppv']:.4f} / {m['npv']:.4f}")
    print(f"  F1-Score           : {m['f1']:.4f}")
    print(f"  MCC                : {m['mcc']:.4f}")
    print(f"  ECE (10 bins)      : {m['ece']:.4f}")
    cm = m["confusion_matrix"]
    print(f"  Confusion matrix   : TP {cm['tp']}  FN {cm['fn']}  |  FP {cm['fp']}  TN {cm['tn']}")
    print("-" * 66)
    print("  GRAD-CAM STRUCTURAL VERIFICATION")
    print(f"  Localization Score : {loc['mean_localization_score']:.4f}  (M_tissue / (M_tissue + M_border))")
    print(f"  Border mass        : {loc['mean_border_mass_frac']:.4f}  (outer {loc['border_px']} px band)")
    if loc["pointing_game_acc_mh"] is not None:
        print(f"  Foveal pointing    : {loc['pointing_game_acc_mh']:.4f}  (MH scans, ±{loc['fovea_tolerance_px']} px)")
        print(f"  Foveal CAM mass    : {loc['mean_foveal_mass_frac_mh']:.4f}  (MH scans)")
    print(f"  Shortcut check     : {'PASS' if loc['passes_shortcut_check'] else 'REVIEW'}")
    print(bar)
    print("  Note: synthetic-data metrics validate the pipeline, not clinical accuracy.")
    print(bar + "\n")


def _json_default(o):
    """Make NumPy / torch scalars JSON-serialisable."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description="Macular Hole detection + Grad-CAM localization engine")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n-train", type=int, default=160, help="synthetic train images (balanced)")
    ap.add_argument("--n-val", type=int, default=60, help="synthetic val images (balanced)")
    ap.add_argument("--regen", action="store_true", help="regenerate the synthetic dataset")
    ap.add_argument("--backbone", choices=["auto", "vit", "effnet"], default="auto")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weight download")
    ap.add_argument("--time-budget", type=float, default=110.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[env] torch {torch.__version__} | device {device}")
    OUT_DIR.mkdir(exist_ok=True)

    has_real = all(any((DATA_DIR / s / c).glob("*")) for s in ("train", "val") for c in CLASS_DIRS) \
        if DATA_DIR.exists() else False
    if args.regen or not has_real:
        build_synthetic_dataset(DATA_DIR, args.n_train, args.n_val, force=args.regen or not has_real)
    else:
        print(f"[data] Using existing images in {DATA_DIR}")

    model, info = create_model(args.backbone, pretrained=not args.no_pretrained)
    model = model.to(device)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, time_budget_s=args.time_budget)
    history, train_s = train(model, info, cfg, device)

    y, p, names = predict_split(model, "val", device)
    metrics = classification_metrics(y, p)
    ci = {"auc": bootstrap_ci(y, p, roc_auc),
          "balanced_accuracy": bootstrap_ci(y, p, lambda a, b: classification_metrics(a, b)["balanced_accuracy"], n_boot=500)}
    loc_summary, loc_rows = evaluate_localization(model, device)
    print_report(metrics, ci, loc_summary, info, train_s)

    torch.save({"state_dict": model.state_dict(),
                "config": {**info, "img_size": IMG_SIZE, "class_names": CLASS_NAMES,
                           "threshold": metrics["threshold"], "normalization": "percentile_p1_p99"}},
               WEIGHTS_PATH)
    fpr, tpr = roc_curve_points(y, p)
    fig = save_figures(y, p, metrics)
    report = {
        "task": "Macular Hole vs Normal — retinal OCT B-scan",
        "model": info, "train_config": cfg.__dict__, "training_history": history,
        "training_time_s": round(train_s, 1), "n_val": int(len(y)),
        "metrics": metrics, "ci95": ci, "roc_curve": {"fpr": fpr, "tpr": tpr},
        "gradcam_localization": loc_summary, "gradcam_per_image": loc_rows,
        "figures": {"roc_reliability": fig, "gradcam_examples": str(OUT_DIR / "gradcam_examples")},
        "data": {"root": str(DATA_DIR), "synthetic": (DATA_DIR / "metadata.json").exists()},
        "disclaimer": "Research prototype. Synthetic-data results do not indicate clinical performance.",
    }
    (OUT_DIR / "performance_report.json").write_text(json.dumps(report, indent=2, default=_json_default))
    print(f"[save] weights -> {WEIGHTS_PATH}")
    print(f"[save] report  -> {OUT_DIR / 'performance_report.json'}")
    print(f"[save] Grad-CAM examples -> {OUT_DIR / 'gradcam_examples'}")
    print("[next] run:  python app.py   then open http://127.0.0.1:5050")


if __name__ == "__main__":
    main()
