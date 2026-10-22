# Macular Hole Detection & Foveal Localization Engine

ViT-Tiny classifier (Normal vs. Macular Hole) for retinal OCT B-scans, with Grad-CAM
explainability, a structural localization check, a full clinical metrics report and a
Flask web dashboard.

## Run it (≈2 minutes on a laptop CPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python mh_engine.py
python app.py
```

First run downloads ImageNet weights for ViT-Tiny (~22 MB). With no internet, use
`python mh_engine.py --no-pretrained` (switches to EfficientNet-B0 + spatial attention trained from scratch).

Useful flags: `--epochs 5`, `--regen` (new synthetic data), `--backbone vit|effnet|auto`,
`--n-train 240 --n-val 80`, `--time-budget 110`.

## What gets produced

| File | Contents |
|---|---|
| `mh_model_weights.pth` | fine-tuned weights + config (backbone, threshold, normalisation) |
| `outputs/performance_report.json` | AUC with 95% bootstrap CI, balanced accuracy, sensitivity, specificity, PPV/NPV, F1, MCC, ECE + reliability bins, ROC points, Grad-CAM localization stats per image |
| `outputs/gradcam_examples/*.png` | scan / Grad-CAM side-by-side for validation images |
| `outputs/roc_reliability.png` | ROC curve + reliability diagram (if matplotlib is installed) |

## Method summary

- **Preprocessing:** percentile intensity normalisation (p1–p99) for scanner-contrast invariance;
  augmentations: horizontal flip, ±8% vertical shift, mild gamma jitter.
- **Classifier:** `vit_tiny_patch16_224` (timm), AdamW (wd 0.05), cosine annealing, label smoothing 0.1.
  Fallback: EfficientNet-B0 + CBAM-style spatial attention.
- **Grad-CAM:** on the last transformer block (patch tokens reshaped to a 14×14 grid).
- **Localization Score** = M_tissue / (M_tissue + M_border), where M_border is CAM mass in the
  outer 16 px band. Low scores flag shortcut learning (edge artefacts, burnt-in text).
  Plus a **foveal pointing game**: CAM peak within ±28 px of the fovea column.

## Using real data

Put images in `data/train/normal`, `data/train/mh`, `data/val/normal`, `data/val/mh`
(PNG/JPG/TIFF/BMP) and delete `data/metadata.json`. The engine detects the folders and trains
on them as-is. The OCTID dataset includes Normal and Macular Hole B-scans. With real data, split
by patient, not by image, or the metrics will be inflated.

## Important caveat

The bundled data is synthetic, and the task it poses is easy, so expect near-perfect scores.
Those numbers show the pipeline works; they say nothing about clinical accuracy, and the
model will not give meaningful results on real OCT scans until it's trained on real data.
Research prototype only — not a medical device.
