"""
Build script for Notebook 13 — Validation Case Study (Interactive).

The lab's central pattern: a unified model adapter behind two `#@param`
dropdowns — model and test image — wired to Lab 2's metrics. The point is
that *model picker + automated metrics + a thoughtful test set* is the
skeleton of any real validation study.

Run:
    python build_notebook_13.py
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent


class CellBuilder:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.idx = 0
        self.cells = []

    def md(self, source: str):
        self.cells.append({
            "cell_type": "markdown",
            "id": f"{self.prefix}-md-{self.idx:03d}",
            "metadata": {},
            "source": source.lstrip("\n"),
        })
        self.idx += 1
        return self

    def code(self, source: str):
        self.cells.append({
            "cell_type": "code",
            "id": f"{self.prefix}-code-{self.idx:03d}",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": source.lstrip("\n"),
        })
        self.idx += 1
        return self


def build_notebook(cells, name: str):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = OUT_DIR / f"{name}.ipynb"
    out_path.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {out_path.name} ({out_path.stat().st_size} bytes, {len(cells)} cells)")


# ---------------------------------------------------------------------------
# Title and orientation
# ---------------------------------------------------------------------------
def section_title(b):
    b.md("""<!-- colab-badge -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/main/notebooks/13_validation_case_study.ipynb)

*Click the badge to open this notebook in Google Colab. For best performance, switch to a GPU runtime: Runtime → Change runtime type → T4 GPU.*""")

    b.md("""# Notebook 13 — Validation Case Study (Interactive Model Picker)

**Status.** Extension lab — post-workshop self-paced. Recommended after Notebooks 01, 02, 03b.
**Estimated time.** 30–45 minutes for the core walk-through; longer if you explore many model × image combinations.

**Learning goals.**

1. Pick a pretrained segmentation model from a unified menu and run it on a chosen test image.
2. Compare multiple models side-by-side on the same image.
3. Compute Lab 2's validation metrics (IoU, Dice, count error, instance match) automatically when ground truth is available.
4. Use **inter-model agreement** as a validation signal when no ground truth exists.
5. Load any model from the BioImage Model Zoo on the fly via its registry ID.

> **Why this lab matters.** Real validation in the wild looks like this: you don't get to pick a "ground-truth-rich" benchmark. You have your own data, a handful of candidate models, and a question you want answered. This notebook is the skeleton of that workflow: model picker + automated metrics + a thoughtful test set.

> **A note on the form widgets.** Several cells below use `#@param` comments. In **Google Colab** these render as interactive form widgets (sliders, dropdowns) at the top of the cell. In **other environments** (JupyterLab, VS Code, the JB rendered HTML) they appear as plain Python comments — edit the values directly and re-run the cell.""")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def section_setup(b):
    b.md("""## Setup

Install the libraries we'll dispatch to. The setup cell installs everything up front; individual model weights download lazily on first use, so you can pick a single model from the menu without waiting for all of them.""")

    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

# Install in one pass. csbdeep is StarDist's backend; bioimageio.core is the BiMZ Python interface.
%pip install --quiet "cellpose>=3.0" "stardist>=0.8" csbdeep \\
    "git+https://github.com/facebookresearch/segment-anything.git" \\
    bioimageio.core scikit-image matplotlib pandas tifffile

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import data as skdata
from matplotlib.colors import ListedColormap

print("Imports OK.")""")

    b.code("""# GPU detection — informational only; each library handles fallback automatically
try:
    import torch
    print("torch          :", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device         :", torch.cuda.get_device_name(0))
except ImportError:
    print("torch not installed (cellpose will install it)")""")


# ---------------------------------------------------------------------------
# Helper functions (lifted from Lab 2, with attribution)
# ---------------------------------------------------------------------------
def section_helpers(b):
    b.md("""## Helpers from Lab 2

These are the same metric helpers we built in Notebook 02. Repeated here for self-containment so this lab can run in isolation. If you've already worked through Lab 2, skim and move on.""")

    b.code("""# Categorical colormap for instance labels (background black, then qualitative)
_label_cmap = ListedColormap(['black'] + plt.get_cmap('tab20')(np.linspace(0, 1, 20)).tolist())


def show_mask(ax, mask, title=""):
    \"\"\"Render an integer-label mask with one color per instance.\"\"\"
    n_lbl = max(int(mask.max()) if mask.size else 0, 1)
    ax.imshow(mask, cmap=_label_cmap, vmin=0, vmax=20)
    ax.set_title(f"{title}  ({n_lbl} obj)" if title else f"{n_lbl} obj")
    ax.axis('off')


def iou_dice_binary(pred_mask, gt_mask):
    pred_bin = pred_mask > 0
    gt_bin = gt_mask > 0
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    return iou, dice


def match_instances(pred_mask, gt_mask, iou_threshold=0.5):
    \"\"\"Match each predicted instance to a GT instance by IoU >= threshold.

    Returns matched count, false positives, false negatives, precision, recall.
    \"\"\"
    pred_ids = [i for i in np.unique(pred_mask) if i != 0]
    gt_ids = [i for i in np.unique(gt_mask) if i != 0]
    matched_pred, matched_gt = set(), set()
    for p in pred_ids:
        pb = (pred_mask == p)
        best_iou, best_g = 0.0, None
        for g in gt_ids:
            if g in matched_gt:
                continue
            gb = (gt_mask == g)
            inter = (pb & gb).sum()
            union = (pb | gb).sum()
            if union == 0:
                continue
            iv = inter / union
            if iv > best_iou:
                best_iou, best_g = iv, g
        if best_iou >= iou_threshold:
            matched_pred.add(p)
            matched_gt.add(best_g)
    tp = len(matched_pred)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def biological_summary(pred, gt=None):
    pred_count = int(np.unique(pred[pred > 0]).size)
    out = {"predicted_count": pred_count}
    if gt is not None:
        gt_count = int(np.unique(gt[gt > 0]).size)
        out["true_count"] = gt_count
        out["count_error"] = pred_count - gt_count
    return out


print("Helpers defined.")""")


# ---------------------------------------------------------------------------
# Test images — registry + ground truth where available
# ---------------------------------------------------------------------------
def section_test_images(b):
    b.md("""## Test image registry

We curate a small set of test images that span the conditions a real validation study would face: in-distribution, slightly out-of-distribution, and deliberately out-of-distribution. Some have ground truth (the synthetic ones, where we control the generator); others don't, and that's the point — most real images don't.""")

    b.code("""from scipy.ndimage import gaussian_filter

def make_synth_easy(seed=0, size=200, n_cells=12):
    \"\"\"In-distribution: round, well-separated cells. Cellpose-friendly.\"\"\"
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=float)
    truth = np.zeros((size, size), dtype=np.int32)
    centers = rng.uniform(20, size - 20, (n_cells, 2))
    radii = rng.uniform(10, 18, n_cells)
    for idx, ((cy, cx), r) in enumerate(zip(centers, radii), start=1):
        Y, X = np.ogrid[:size, :size]
        circle = (Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2
        img[circle] = rng.uniform(0.6, 1.0)
        truth[circle] = idx
    img = gaussian_filter(img, sigma=1.0) + rng.normal(0, 0.05, img.shape)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img, truth


def make_synth_ood(seed=2, size=200, n_cells=18):
    \"\"\"Out-of-distribution: irregular shapes, dense, varying intensity.\"\"\"
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=float)
    truth = np.zeros((size, size), dtype=np.int32)
    centers = rng.uniform(15, size - 15, (n_cells, 2))
    for idx, (cy, cx) in enumerate(centers, start=1):
        Y, X = np.ogrid[:size, :size]
        ry = rng.uniform(6, 14)
        rx = rng.uniform(6, 14) * rng.uniform(0.7, 1.4)
        ang = rng.uniform(0, np.pi)
        Yr = (Y - cy) * np.cos(ang) + (X - cx) * np.sin(ang)
        Xr = -(Y - cy) * np.sin(ang) + (X - cx) * np.cos(ang)
        ellipse = (Yr / ry) ** 2 + (Xr / rx) ** 2 <= 1
        img[ellipse] = rng.uniform(0.3, 0.9)
        truth[ellipse] = idx  # may overwrite for overlap (intentional)
    img = gaussian_filter(img, sigma=0.8) + rng.normal(0, 0.08, img.shape)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img, truth


# Build the registry. Each entry: (image, ground_truth_or_None, modality_hint)
TEST_IMAGES = {
    "synthetic — in-distribution (12 cells)": (*make_synth_easy(), "fluorescence"),
    "synthetic — OOD (18 irregular cells)":   (*make_synth_ood(),  "fluorescence"),
    "skimage — single cell (cell)":           (skdata.cell(),                   None, "fluorescence"),
    "skimage — human mitosis (nuclei)":       (skdata.human_mitosis(),          None, "fluorescence"),
    "skimage — cells3d mid-Z (nuclei)":       (skdata.cells3d()[30, 1],         None, "fluorescence"),
}

print("Test images registered:")
for name, (img, gt, mod) in TEST_IMAGES.items():
    truth_note = f"truth={gt.max()} obj" if gt is not None else "no truth"
    print(f"  {name:<45}  shape={img.shape}  ({mod}, {truth_note})")""")

    b.code("""# Visualize the registry so you know what you're picking from
fig, axes = plt.subplots(1, len(TEST_IMAGES), figsize=(3 * len(TEST_IMAGES), 3.2))
for ax, (name, (img, gt, mod)) in zip(axes, TEST_IMAGES.items()):
    p1, p99 = np.percentile(img, [1, 99])
    ax.imshow(img, cmap='gray', vmin=p1, vmax=p99)
    truth_tag = "✓ truth" if gt is not None else "no truth"
    ax.set_title(f"{name.split(' — ')[1]}\\n{truth_tag}", fontsize=9)
    ax.axis('off')
plt.tight_layout(); plt.show()""")


# ---------------------------------------------------------------------------
# Unified model adapter
# ---------------------------------------------------------------------------
def section_adapter(b):
    b.md("""## The unified model adapter

Each candidate model has its own Python API and its own assumptions about input shape, channel order, normalization, and output format. The `run_model` function below hides those differences behind a uniform interface: pass it a model ID and an image, get back an integer-labeled instance mask.

Models are loaded **lazily** on first request — picking just one from the menu doesn't pay the cost of downloading all of them.""")

    b.code("""# Lazy model cache. Each entry is populated on first request.
_MODEL_CACHE = {}

def _load_cellpose(model_type):
    if ("cellpose", model_type) in _MODEL_CACHE:
        return _MODEL_CACHE[("cellpose", model_type)]
    from cellpose import models, core
    use_gpu = core.use_gpu()
    m = models.CellposeModel(gpu=use_gpu, model_type=model_type)
    _MODEL_CACHE[("cellpose", model_type)] = m
    return m


def _load_stardist(model_name):
    if ("stardist", model_name) in _MODEL_CACHE:
        return _MODEL_CACHE[("stardist", model_name)]
    from stardist.models import StarDist2D
    m = StarDist2D.from_pretrained(model_name)
    _MODEL_CACHE[("stardist", model_name)] = m
    return m


def _load_sam():
    if ("sam",) in _MODEL_CACHE:
        return _MODEL_CACHE[("sam",)]
    import urllib.request
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    ckpt = "sam_vit_b_01ec64.pth"
    if not os.path.exists(ckpt):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        print(f"Downloading SAM checkpoint (~360 MB)...")
        urllib.request.urlretrieve(url, ckpt)
    sam = sam_model_registry["vit_b"](checkpoint=ckpt)
    gen = SamAutomaticMaskGenerator(sam)
    _MODEL_CACHE[("sam",)] = gen
    return gen


def _to_rgb_uint8(img):
    \"\"\"SAM expects 3-channel uint8.\"\"\"
    arr = img.astype(np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    arr = np.clip((arr - p1) / max(p99 - p1, 1e-8), 0, 1)
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def _normalize_for_stardist(img):
    from csbdeep.utils import normalize
    return normalize(img, 1, 99.8)


def run_model(model_id, img):
    \"\"\"Unified dispatch. Returns (mask, meta_dict).

    mask: int32 array, 0 = background, 1..N = instance labels
    meta: dict with timing and any model-specific notes
    \"\"\"
    import time
    t0 = time.time()

    if model_id == "Cellpose-SAM (cyto3)":
        m = _load_cellpose("cyto3")
        mask, _, _ = m.eval(img, diameter=None, channels=[0, 0])

    elif model_id == "Cellpose-SAM (nuclei)":
        m = _load_cellpose("nuclei")
        mask, _, _ = m.eval(img, diameter=None, channels=[0, 0])

    elif model_id == "StarDist 2D (Versatile fluorescent nuclei)":
        m = _load_stardist("2D_versatile_fluo")
        labels, _ = m.predict_instances(_normalize_for_stardist(img))
        mask = labels.astype(np.int32)

    elif model_id == "StarDist 2D (Versatile H&E nuclei)":
        m = _load_stardist("2D_versatile_he")
        # H&E model expects 3-channel input; convert grayscale to RGB
        rgb = _to_rgb_uint8(img)
        labels, _ = m.predict_instances(_normalize_for_stardist(rgb))
        mask = labels.astype(np.int32)

    elif model_id == "SAM (automatic everything)":
        gen = _load_sam()
        rgb = _to_rgb_uint8(img)
        anns = gen.generate(rgb)
        # SAM auto-mask returns a list of dicts with 'segmentation' bool arrays
        mask = np.zeros(img.shape[:2], dtype=np.int32)
        for i, a in enumerate(sorted(anns, key=lambda x: -x.get('area', 0)), start=1):
            mask[a['segmentation']] = i

    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    elapsed = time.time() - t0
    return mask, {"elapsed_s": elapsed, "n_objects": int(mask.max()) if mask.size else 0}


# Available built-in models (BiMZ ad-hoc loader is in a later cell)
BUILTIN_MODELS = [
    "Cellpose-SAM (cyto3)",
    "Cellpose-SAM (nuclei)",
    "StarDist 2D (Versatile fluorescent nuclei)",
    "StarDist 2D (Versatile H&E nuclei)",
    "SAM (automatic everything)",
]
print(f"Adapter ready. {len(BUILTIN_MODELS)} built-in models available.")""")


# ---------------------------------------------------------------------------
# Pick a model and an image
# ---------------------------------------------------------------------------
def section_pick(b):
    b.md("""## Pick a model and an image

Two dropdowns, one prediction. Pick from the menu, run the cell, look at the output.

**Predict before you run.** Look at the dropdowns. If you pick *Cellpose-SAM (cyto3)* on the synthetic in-distribution image, what do you expect — a tight match to the truth, or merges and splits at boundaries? If you pick *StarDist H&E* on a fluorescence image, what do you expect to happen?

The point is to commit to a hypothesis before the cell runs, then update from the result.""")

    b.code("""# @title Pick model and image { run: \"auto\" }
model_id = "Cellpose-SAM (cyto3)"  # @param ["Cellpose-SAM (cyto3)", "Cellpose-SAM (nuclei)", "StarDist 2D (Versatile fluorescent nuclei)", "StarDist 2D (Versatile H&E nuclei)", "SAM (automatic everything)"]
test_image = "synthetic — in-distribution (12 cells)"  # @param ["synthetic — in-distribution (12 cells)", "synthetic — OOD (18 irregular cells)", "skimage — single cell (cell)", "skimage — human mitosis (nuclei)", "skimage — cells3d mid-Z (nuclei)"]

img, gt, mod = TEST_IMAGES[test_image]
print(f"Image  : {test_image}  shape={img.shape}  modality={mod}")
print(f"Model  : {model_id}")
print(f"Truth  : {'available (' + str(int(gt.max())) + ' objects)' if gt is not None else 'not available'}")
print()

mask, meta = run_model(model_id, img)
print(f"Result : {meta['n_objects']} objects detected in {meta['elapsed_s']:.1f}s")

# Visualize
p1, p99 = np.percentile(img, [1, 99])
fig, axes = plt.subplots(1, 3 if gt is not None else 2, figsize=(13, 4.5))
axes[0].imshow(img, cmap='gray', vmin=p1, vmax=p99); axes[0].set_title("Image"); axes[0].axis('off')
show_mask(axes[1], mask, f"{model_id}")
if gt is not None:
    show_mask(axes[2], gt, "ground truth")
plt.tight_layout(); plt.show()""")

    b.md("""**What you should be seeing.** A reasonable mask if model and image are well-matched (e.g., Cellpose-SAM cyto3 on synthetic-in-distribution); a degraded mask if the model's training distribution doesn't cover the image (e.g., StarDist H&E on fluorescence). The synthetic OOD image will make even a well-chosen model produce confident-but-wrong output — exactly the failure mode the lecture and Lab 1 surfaced.

**Try this.** Run several model × image combinations. Note where the mask looks visually correct, where the count looks right but boundaries look wrong, and where both look wrong. Those three failure modes correspond to different validation challenges.""")


# ---------------------------------------------------------------------------
# Compare all models on the same image
# ---------------------------------------------------------------------------
def section_compare_all(b):
    b.md("""## Compare all built-in models on the same image

For any single image, the most useful diagnostic is a side-by-side of all candidate models. Differences between models on the same image surface the **inductive biases** each model carries — what kinds of objects it expects, what failures it tolerates.""")

    b.code("""# @title Run all models on one image { run: \"auto\" }
test_image_compare = "synthetic — OOD (18 irregular cells)"  # @param ["synthetic — in-distribution (12 cells)", "synthetic — OOD (18 irregular cells)", "skimage — single cell (cell)", "skimage — human mitosis (nuclei)", "skimage — cells3d mid-Z (nuclei)"]

img_c, gt_c, mod_c = TEST_IMAGES[test_image_compare]
print(f"Image: {test_image_compare}\\n")

results = {}
for mid in BUILTIN_MODELS:
    try:
        mask_i, meta_i = run_model(mid, img_c)
        results[mid] = (mask_i, meta_i)
        print(f"  {mid:<46}: {meta_i['n_objects']:>3} objects  ({meta_i['elapsed_s']:.1f}s)")
    except Exception as e:
        results[mid] = (None, {"error": str(e)})
        print(f"  {mid:<46}: ERROR — {e}")

# Visualize side-by-side
n = len(BUILTIN_MODELS) + 1 + (1 if gt_c is not None else 0)
fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.5))
p1c, p99c = np.percentile(img_c, [1, 99])
axes[0].imshow(img_c, cmap='gray', vmin=p1c, vmax=p99c); axes[0].set_title("Image"); axes[0].axis('off')
for i, mid in enumerate(BUILTIN_MODELS, start=1):
    mask_i, meta_i = results[mid]
    if mask_i is not None:
        show_mask(axes[i], mask_i, mid.split(" (")[0])
    else:
        axes[i].text(0.5, 0.5, "ERROR", ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_title(mid.split(" (")[0]); axes[i].axis('off')
if gt_c is not None:
    show_mask(axes[-1], gt_c, "ground truth")
plt.tight_layout(); plt.show()""")

    b.md("""**Reading the comparison.** Models will disagree. *Where* they disagree is informative:

- All five models agree on the obvious cells → high confidence those are real.
- Models disagree on borderline cells → flag for human review.
- One model is very different from the others → check whether that model's training distribution matches your data.

This is the cheapest validation signal you can get without ground truth: ensemble disagreement is a proxy for uncertainty.""")


# ---------------------------------------------------------------------------
# Validate against ground truth (synthetic only)
# ---------------------------------------------------------------------------
def section_validate_truth(b):
    b.md("""## Validate against ground truth (when you have it)

For the synthetic images we built, we have full ground truth. The cell below runs the model picker against the truth and reports Lab 2's metrics: pixel IoU, pixel Dice, instance precision/recall at IoU=0.5, count error.

If you skipped Lab 2, the metrics tell you four different things about the same prediction:

- **IoU / Dice** — pixel-level overlap. High when the masks roughly cover the right area.
- **Instance precision** — fraction of predicted objects that match a true object.
- **Instance recall** — fraction of true objects that the model found.
- **Count error** — predicted count minus true count. The biological metric.

A model can score high on one and low on another. That's the metrics-vs-biology gap.""")

    b.code("""# @title Validate one model against ground truth { run: \"auto\" }
model_id_v = "Cellpose-SAM (cyto3)"  # @param ["Cellpose-SAM (cyto3)", "Cellpose-SAM (nuclei)", "StarDist 2D (Versatile fluorescent nuclei)", "StarDist 2D (Versatile H&E nuclei)", "SAM (automatic everything)"]
test_image_v = "synthetic — in-distribution (12 cells)"  # @param ["synthetic — in-distribution (12 cells)", "synthetic — OOD (18 irregular cells)"]

img_v, gt_v, _ = TEST_IMAGES[test_image_v]
if gt_v is None:
    print("This image has no ground truth — pick a synthetic image for this cell.")
else:
    mask_v, meta_v = run_model(model_id_v, img_v)
    iou_v, dice_v = iou_dice_binary(mask_v, gt_v)
    inst_v = match_instances(mask_v, gt_v, iou_threshold=0.5)
    bio_v = biological_summary(mask_v, gt_v)

    print(f"Model : {model_id_v}")
    print(f"Image : {test_image_v}")
    print()
    print(f"  Pixel IoU                 : {iou_v:.3f}")
    print(f"  Pixel Dice                : {dice_v:.3f}")
    print(f"  Instance precision @0.5   : {inst_v['precision']:.3f}  ({inst_v['tp']} TP, {inst_v['fp']} FP)")
    print(f"  Instance recall    @0.5   : {inst_v['recall']:.3f}  ({inst_v['tp']} TP, {inst_v['fn']} FN)")
    print(f"  Count error (pred − true) : {bio_v['count_error']:+d}  ({bio_v['predicted_count']} vs {bio_v['true_count']})")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    p1v, p99v = np.percentile(img_v, [1, 99])
    axes[0].imshow(img_v, cmap='gray', vmin=p1v, vmax=p99v); axes[0].set_title("Image"); axes[0].axis('off')
    show_mask(axes[1], mask_v, "prediction")
    show_mask(axes[2], gt_v, "ground truth")
    plt.tight_layout(); plt.show()""")

    b.code("""# Score every built-in model on every synthetic image (the only ones with truth) -- benchmark table
synth_imgs = {n: (img, gt) for n, (img, gt, _) in TEST_IMAGES.items() if gt is not None}

rows = []
for img_name, (im, tr) in synth_imgs.items():
    for mid in BUILTIN_MODELS:
        try:
            mask_b, meta_b = run_model(mid, im)
            iou_b, dice_b = iou_dice_binary(mask_b, tr)
            inst_b = match_instances(mask_b, tr, iou_threshold=0.5)
            bio_b = biological_summary(mask_b, tr)
            rows.append({
                "image": img_name.split(" — ")[1],
                "model": mid.split(" (")[0] + (" " + mid.split(" (")[1].rstrip(")") if " (" in mid else ""),
                "IoU": round(iou_b, 3),
                "Dice": round(dice_b, 3),
                "prec": round(inst_b['precision'], 2),
                "rec": round(inst_b['recall'], 2),
                "count_err": bio_b['count_error'],
                "elapsed_s": round(meta_b['elapsed_s'], 1),
            })
        except Exception as e:
            rows.append({"image": img_name.split(" — ")[1], "model": mid.split(" (")[0],
                         "IoU": None, "Dice": None, "prec": None, "rec": None, "count_err": None,
                         "elapsed_s": None})

df = pd.DataFrame(rows)
print("Benchmark — every built-in model × every synthetic image:")
print()
print(df.to_string(index=False))""")

    b.md("""**Reading the benchmark.** No model wins on every metric. A model that gets the count right may have lower pixel IoU; a model that gets boundaries tight may miss objects entirely. **The right model depends on the question you're asking** — count cells, measure size, characterize morphology — and that's a biology question, not a model question.

This is exactly the metrics-vs-biology distinction Lab 2 made hands-on. Here we make it across-models.""")


# ---------------------------------------------------------------------------
# Inter-model agreement (validation without truth)
# ---------------------------------------------------------------------------
def section_agreement(b):
    b.md("""## Inter-model agreement (validation without truth)

The harder validation problem: real images, no ground truth. What do you do?

One useful signal: run several models, look at where they agree. **Pixels that all models call foreground are high-confidence; pixels that only one model calls foreground are low-confidence.** This is not a substitute for ground truth, but it gives you something to look at when you don't have any.""")

    b.code("""# @title Inter-model agreement on a real image { run: \"auto\" }
test_image_a = "skimage — human mitosis (nuclei)"  # @param ["skimage — single cell (cell)", "skimage — human mitosis (nuclei)", "skimage — cells3d mid-Z (nuclei)"]

img_a, _, _ = TEST_IMAGES[test_image_a]
print(f"Image: {test_image_a}\\n")

# Run all models on this image
masks_a = {}
for mid in BUILTIN_MODELS:
    try:
        m, meta = run_model(mid, img_a)
        masks_a[mid] = (m > 0).astype(np.uint8)  # binarize for agreement
    except Exception as e:
        print(f"  {mid}: skipped ({e})")

# Agreement map: how many models call each pixel foreground
agree = np.sum(list(masks_a.values()), axis=0)
print(f"Per-pixel agreement levels:")
for level in range(len(masks_a) + 1):
    n = (agree == level).sum()
    pct = 100 * n / agree.size
    print(f"  {level} model(s) agree: {n:>7} px ({pct:5.1f}%)")

# Pairwise IoU between model outputs
mids = list(masks_a.keys())
pairwise = np.zeros((len(mids), len(mids)))
for i, mi in enumerate(mids):
    for j, mj in enumerate(mids):
        a = masks_a[mi].astype(bool)
        b_ = masks_a[mj].astype(bool)
        inter = (a & b_).sum()
        union = (a | b_).sum()
        pairwise[i, j] = inter / union if union > 0 else 0.0

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
p1a, p99a = np.percentile(img_a, [1, 99])
axes[0].imshow(img_a, cmap='gray', vmin=p1a, vmax=p99a); axes[0].set_title("Image"); axes[0].axis('off')

# Agreement heatmap
im_h = axes[1].imshow(agree, cmap='viridis', vmin=0, vmax=len(masks_a))
axes[1].set_title(f"Agreement (0 = nobody, {len(masks_a)} = consensus)"); axes[1].axis('off')
plt.colorbar(im_h, ax=axes[1], fraction=0.045)

# Pairwise IoU heatmap
im_p = axes[2].imshow(pairwise, cmap='magma', vmin=0, vmax=1)
axes[2].set_xticks(range(len(mids)));
axes[2].set_xticklabels([m.split(" (")[0] for m in mids], rotation=45, ha='right', fontsize=7)
axes[2].set_yticks(range(len(mids)));
axes[2].set_yticklabels([m.split(" (")[0] for m in mids], fontsize=7)
axes[2].set_title("Pairwise IoU between models")
for i in range(len(mids)):
    for j in range(len(mids)):
        axes[2].text(j, i, f"{pairwise[i, j]:.2f}", ha='center', va='center',
                     color='white' if pairwise[i, j] < 0.5 else 'black', fontsize=7)
plt.colorbar(im_p, ax=axes[2], fraction=0.045)
plt.tight_layout(); plt.show()""")

    b.md("""**Reading the agreement map and pairwise table.**

- *Bright (high-agreement) pixels* — most models call this foreground. Those are likely real.
- *Dim (low-agreement) pixels* — only one or two models flag this. Those are uncertain regions; flag for human review or for ground-truth annotation if you have the budget.
- *Pairwise IoU near 1.0* — those two models behave nearly identically on this data. (Often happens for two Cellpose variants on a clean fluorescence image.)
- *Pairwise IoU near 0* — those two models are disagreeing fundamentally. Either one is wrong, or the image breaks an assumption built into one of them.

**This is not a substitute for validation.** It's a triage tool. It tells you *where* to spend annotation effort — on the disagreement regions, not on the consensus regions. That's a lot more useful than nothing, which is what you'd otherwise have on real un-annotated data.""")


# ---------------------------------------------------------------------------
# Ad-hoc BiMZ loader
# ---------------------------------------------------------------------------
def section_bimz(b):
    b.md("""## Ad-hoc loader — any model from the BioImage Model Zoo

The five built-in models above are a curated starter set. The BioImage Model Zoo registry has *hundreds* of pretrained models, indexed by task and modality. The cell below loads any model by its registry ID and runs it on the selected test image.

Use this when:

- You're prototyping and don't know which model fits your data.
- A pretrained model exists for your specific sample type (the BiMZ filter cell in Notebook 04 helps you find it).
- You want to compare a registry model against the built-in candidates above.""")

    b.code("""# @title BiMZ ad-hoc model loader { run: \"auto\" }
bimz_model_id = "10.5281/zenodo.5864646"  # @param {type: \"string\"}
test_image_bz = "skimage — human mitosis (nuclei)"  # @param ["synthetic — in-distribution (12 cells)", "synthetic — OOD (18 irregular cells)", "skimage — single cell (cell)", "skimage — human mitosis (nuclei)", "skimage — cells3d mid-Z (nuclei)"]

img_bz, gt_bz, _ = TEST_IMAGES[test_image_bz]

try:
    from bioimageio.core import load_resource_description, predict_with_padding
    from bioimageio.core.prediction_pipeline import create_prediction_pipeline

    print(f"Loading BiMZ model: {bimz_model_id}")
    rd = load_resource_description(bimz_model_id)
    pipeline = create_prediction_pipeline(bioimageio_model=rd)
    print(f"Loaded: {rd.name}")
    print(f"  Inputs : {[ax.name for ax in rd.inputs[0].axes]}")
    print(f"  Outputs: {[ax.name for ax in rd.outputs[0].axes]}")

    # Best-effort reshape: most BiMZ segmentation models expect (B, C, H, W) float32
    arr = img_bz.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, None]  # (1, 1, H, W)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = arr.transpose(2, 0, 1)[None]  # HWC -> (1, C, H, W)

    print(f"\\nRunning inference on {arr.shape}...")
    result = predict_with_padding(pipeline, arr)
    out = np.asarray(result[0]).squeeze()
    print(f"Output shape: {out.shape}")

    # Best-effort visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    p1bz, p99bz = np.percentile(img_bz, [1, 99])
    axes[0].imshow(img_bz, cmap='gray', vmin=p1bz, vmax=p99bz); axes[0].set_title("Image"); axes[0].axis('off')
    if out.ndim == 2:
        axes[1].imshow(out, cmap='viridis'); axes[1].set_title(f"{rd.name} output"); axes[1].axis('off')
    elif out.ndim == 3:
        axes[1].imshow(out[0], cmap='viridis'); axes[1].set_title(f"{rd.name} output (channel 0)"); axes[1].axis('off')
    plt.tight_layout(); plt.show()

except Exception as e:
    print(f"BiMZ loader failed: {e}")
    print()
    print("This is expected if:")
    print("  - The model ID has changed in the BiMZ registry")
    print("  - The model expects a specific input shape we didn't reshape correctly")
    print("  - bioimageio.core can't reach the registry")
    print()
    print("Browse https://bioimage.io for current model IDs. The Notebook 04 BiMZ section")
    print("has a filter cell that lists models with their input/output specs.")""")

    b.md("""**Note on shape compatibility.** BiMZ models declare their expected input shape and channel order in their `rdf.yaml` metadata. The cell above does a best-effort reshape from `(H, W)` to `(1, 1, H, W)` which works for many but not all models. If the inference fails with a shape error, look at the model's `rd.inputs[0].axes` (printed at the top of the cell) and reshape manually.

For a stricter version that respects each model's declared shape, see Notebook 04 §2.""")


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------
def section_closing(b):
    b.md("""## Closing reflection

Validation in the wild is a workflow, not a number. The pieces this lab demonstrated:

1. **A picker over candidate models.** Five built-in + ad-hoc BiMZ loader. The point is not that the menu is exhaustive — it's that *making model choice explicit* turns validation into an answerable question.
2. **A curated test set.** Easy + OOD synthetic + real images. Some have truth, most don't.
3. **Automated metrics where truth exists.** IoU, Dice, instance precision/recall, count error — applied uniformly across models so the comparison is fair.
4. **Inter-model agreement where truth doesn't exist.** Pixel-level consensus + pairwise IoU. Triage, not validation; useful all the same.
5. **Easy reach into the BiMZ registry** when none of the curated models is right for your data.

**The pattern generalizes.** Drop in your own image into `TEST_IMAGES`, add it to the dropdown, and the rest of the lab applies. Drop in a new model adapter (e.g., μSAM, Mesmer, your fine-tuned Cellpose) and `run_model` dispatches uniformly. Validation becomes scalable.

**Where to go next:**

- Lab 2 ([Notebook 02](02_validation_quantification)) — the metric definitions and the metrics-vs-biology gap, made visceral.
- Notebook 04 — the broader catalog of methods you might want in the picker.
- Cellpose 2.0+ training ([Notebook 09](09_cellpose_finetune)) — when no pretrained model fits, train your own.
- BioImage Model Zoo ([bioimage.io](https://bioimage.io)) — the live registry; the picker pattern above is exactly what BiMZ enables at scale.

If a particular model wins consistently for your sample type, the next step is fine-tuning it on a small set of your labeled images. The picker stays useful — it tells you *which model to start fine-tuning from*.""")


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def main():
    b = CellBuilder("nb13")
    section_title(b)
    section_setup(b)
    section_helpers(b)
    section_test_images(b)
    section_adapter(b)
    section_pick(b)
    section_compare_all(b)
    section_validate_truth(b)
    section_agreement(b)
    section_bimz(b)
    section_closing(b)
    build_notebook(b.cells, "13_validation_case_study")


if __name__ == "__main__":
    main()
