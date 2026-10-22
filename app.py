#!/usr/bin/env python3
"""
app.py — Interactive web demo for the Macular Hole Detection & Localization Engine.

Run `python mh_engine.py` first (creates mh_model_weights.pth), then:
    python app.py            ->  http://127.0.0.1:5050

Endpoints
  GET  /            dashboard (templates/index.html)
  POST /predict     multipart form field "file" -> JSON prediction + Grad-CAM images
  GET  /samples     a few validation scans for one-click demo
  GET  /sample/<p>  serve a sample scan
  GET  /health      model status

Research prototype — not a medical device.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError

import mh_engine as eng

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024          # 16 MB upload limit
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, CONFIG, CAM = None, {}, None
CAM_LOCK = threading.Lock()          # Grad-CAM hooks hold per-call state
if eng.WEIGHTS_PATH.exists():
    MODEL, CONFIG = eng.load_trained_model(eng.WEIGHTS_PATH, DEVICE)
    CAM = eng.GradCAM(MODEL)
    print(f"[app] loaded {CONFIG.get('backbone')} model from {eng.WEIGHTS_PATH}")
else:
    print("[app] WARNING: mh_model_weights.pth not found — run `python mh_engine.py` first.")

REPORT_PATH = eng.OUT_DIR / "performance_report.json"


def _png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _clinical_assessment(p_mh: float, loc: dict, threshold: float) -> dict:
    """Turn model output + Grad-CAM verification into reviewer-facing flags."""
    warnings = []
    disruption = p_mh >= threshold
    if disruption:
        warnings.append({"level": "critical",
                         "text": "Structural foveal disruption detected — pattern consistent with a "
                                 "full-thickness macular hole. Refer for retinal specialist review."})
    if 0.35 <= p_mh <= 0.65:
        warnings.append({"level": "caution",
                         "text": "Borderline confidence — manual grading of the B-scan recommended."})
    if loc["localization_score"] < 0.8 or loc["border_mass_frac"] > 0.10:
        warnings.append({"level": "caution",
                         "text": "Model attention is not well confined to retinal tissue "
                                 "(possible artefact or out-of-distribution scan)."})
    if disruption and not loc["pointing_hit"]:
        warnings.append({"level": "caution",
                         "text": "Attention peak lies away from the estimated fovea — verify the "
                                 "defect location manually."})
    return {"foveal_disruption": disruption, "warnings": warnings}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else None
    return jsonify({
        "model_loaded": MODEL is not None, "device": DEVICE, "config": CONFIG,
        "val_metrics": None if report is None else {
            k: report["metrics"][k] for k in ("auc", "balanced_accuracy", "sensitivity", "specificity", "mcc", "ece")},
        "synthetic_training_data": None if report is None else report["data"]["synthetic"],
    })


@app.route("/samples")
def samples():
    out = []
    for cdir in eng.CLASS_DIRS:
        files = sorted((eng.DATA_DIR / "val" / cdir).glob("*.png"))[:3]
        out += [{"path": f"{cdir}/{f.name}", "label": eng.CLASS_NAMES[eng.CLASS_DIRS.index(cdir)]} for f in files]
    return jsonify(out)


@app.route("/sample/<cdir>/<name>")
def sample(cdir, name):
    if cdir not in eng.CLASS_DIRS or "/" in name or ".." in name:
        abort(404)
    path = eng.DATA_DIR / "val" / cdir / name
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not trained yet. Run `python mh_engine.py` first."}), 503
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify({"error": "No file uploaded (form field 'file')."}), 400
    if Path(f.filename).suffix.lower() not in ALLOWED:
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED))}"}), 400
    try:
        img = Image.open(io.BytesIO(f.read()))
        img.load()
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Could not decode the image."}), 400

    t0 = time.time()
    x, gray = eng.preprocess(img)                                  # percentile normalisation
    with CAM_LOCK:
        cams, probs, cls = CAM(x.to(DEVICE))                       # ViT inference + Grad-CAM
    cam, p_mh = cams[0], float(probs[0, 1])
    loc = eng.localization_metrics(cam, gray)
    threshold = float(CONFIG.get("threshold", 0.5))
    pred = int(p_mh >= threshold)
    assessment = _clinical_assessment(p_mh, loc, threshold)
    latency = (time.time() - t0) * 1000

    original = (np.repeat(gray[..., None], 3, -1) * 255).astype(np.uint8)
    return jsonify({
        "prediction": eng.CLASS_NAMES[pred],
        "is_macular_hole": bool(pred),
        "confidence": round(100 * (p_mh if pred else 1 - p_mh), 2),
        "probabilities": {"normal": round(1 - p_mh, 4), "macular_hole": round(p_mh, 4)},
        "explained_class": eng.CLASS_NAMES[int(cls[0])],
        "images": {"original": _png_b64(original), "gradcam": _png_b64(eng.overlay_cam(gray, cam))},
        "localization": {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in loc.items()},
        "clinical": assessment,
        "model": {"backbone": CONFIG.get("backbone"), "threshold": threshold, "device": DEVICE},
        "latency_ms": round(latency, 1),
        "disclaimer": "Research prototype — not for diagnostic use.",
    })


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large (max 16 MB)."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
