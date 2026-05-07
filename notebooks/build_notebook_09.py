"""
Build script for Notebook 09 — Cellpose 2.0+ Human-in-the-Loop Fine-Tuning (Colab).

Closes the gap between "use a pretrained model" (NB01) and "train your own."

The pedagogical arc: load pretrained → see it fail on a sample type it wasn't
trained on → fine-tune on a small handful of labeled images → see the same model
do better → reflect on when fine-tuning helps vs. when it just memorizes.

Run:
    python build_notebook_09.py
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/2026-workshop/notebooks/09_cellpose_finetune.ipynb)

*Click the badge to open this notebook in Google Colab. For best performance, switch to a GPU runtime: Runtime → Change runtime type → T4 GPU.*""")

    b.md("""# Notebook 09 — Cellpose 2.0+ Human-in-the-Loop Fine-Tuning (Colab)

**Status.** Extension lab — post-workshop self-paced. Recommended after Notebook 01.
**Estimated time.** 15–25 minutes on Colab T4 (longer on CPU).

**Learning goals.**

1. Recognize when a pretrained Cellpose model fails on your sample type.
2. Collect a small labeled dataset (8–10 training images + 2–3 held-out test images).
3. Fine-tune the pretrained model on your labeled data using `model.train()`.
4. Compare pretrained vs. fine-tuned inference on the same held-out images.
5. Reflect on the tradeoff: fine-tuning helps on matched data; small datasets risk overfitting.

> **Why this lab matters.** Lab 01 showed pretrained Cellpose works well in-distribution and fails out-of-distribution. Lab 02 gave you metrics to *quantify* that failure. This lab closes the gap: when pretraining alone isn't enough, how do you bootstrap fine-tuning? The answer is "not much more complex than Lab 01" — load → train → compare. The harder question — "is my fine-tuned model learning or just memorizing?" — is what makes validation essential.

> **A note on the form widgets.** Several cells below use `#@param` comments. In **Google Colab** these render as interactive form widgets (sliders, dropdowns) at the top of the cell. In **other environments** they appear as plain Python comments — edit the values directly and re-run the cell.""")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def section_setup(b):
    b.md("""## Setup

GPU is recommended but not required; Cellpose runs on CPU (slower). On Colab: `Runtime → Change runtime type → T4 GPU`.""")

    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

# Install Cellpose (3.0+) with all dependencies
%pip install --quiet "cellpose>=3.0" scikit-image matplotlib numpy

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
print("Imports OK.""")

    b.code("""# GPU detection
try:
    import torch
    print("torch          :", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device         :", torch.cuda.get_device_name(0))
except ImportError:
    print("torch not installed (cellpose will install it)")""")


# ---------------------------------------------------------------------------
# Helpers (metric functions)
# ---------------------------------------------------------------------------
def section_helpers(b):
    b.md("""## Helper functions

Metric functions from Lab 01–02: binary IoU, instance matching. Repeated here for self-containment.""")

    b.code("""from matplotlib.colors import ListedColormap

_label_cmap = ListedColormap(['black'] + plt.get_cmap('tab20')(np.linspace(0, 1, 20)).tolist())


def show_mask(ax, mask, title=""):
    \"\"\"Render an integer-label mask with one color per instance.\"\"\"
    n_lbl = max(int(mask.max()) if mask.size else 0, 1)
    ax.imshow(mask, cmap=_label_cmap, vmin=0, vmax=20)
    ax.set_title(f"{title}  ({n_lbl} obj)" if title else f"{n_lbl} obj")
    ax.axis('off')


def iou_dice_binary(pred_mask, gt_mask):
    \"\"\"Pixel-level IoU and Dice (binary).\"\"\"
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
    \"\"\"Match instances by IoU. Returns tp, fp, fn, precision, recall.\"\"\"
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


print("Helpers defined.""")


# ---------------------------------------------------------------------------
# Generate or load labeled training dataset
# ---------------------------------------------------------------------------
def section_data(b):
    b.md("""## Step 1: Get a small labeled dataset

Fine-tuning requires paired (image, instance_mask) examples. We generate a small synthetic dataset for reproducibility; in practice, you would use your own images with ground-truth masks drawn in a tool like FIJI or Cellpose's GUI.

**The dataset**: 8 training images + 2 held-out test images. Each is 200×200 with ~10–15 round cells per image. Completely synthetic so you can reproduce exactly. (See link at the end to real labeled datasets if you want to fine-tune on actual microscopy.)**

Predict before running. If you fine-tune on 8 images with ~12 cells each (~100 cell instances total), what do you expect the model to learn?
- (a) Robust features; the model generalizes to new images.
- (b) Per-image memorization; the model overfits to the 8 training images.
- (c) Something in between; depends on whether the training images are diverse enough.""")

    b.code("""from scipy.ndimage import gaussian_filter

def make_labeled_image(seed=0, size=200, n_cells=12):
    \"\"\"Generate a synthetic image + instance ground truth.\"\"\"
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=float)
    truth = np.zeros((size, size), dtype=np.int32)
    centers = rng.uniform(20, size-20, (n_cells, 2))
    radii = rng.uniform(9, 16, n_cells)
    for idx, ((cy, cx), r) in enumerate(zip(centers, radii), start=1):
        Y, X = np.ogrid[:size, :size]
        circle = (Y - cy)**2 + (X - cx)**2 <= r**2
        img[circle] = rng.uniform(0.5, 1.0)
        truth[circle] = idx
    img = gaussian_filter(img, sigma=1.0) + rng.normal(0, 0.05, img.shape)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img, truth

# Generate training set (8 images)
train_images = []
train_labels = []
for i in range(8):
    img, lbl = make_labeled_image(seed=i, n_cells=rng.integers(10, 15))
    train_images.append(img)
    train_labels.append(lbl)
    skio.imsave(f"train_img_{i:02d}.png", img)
    np.save(f"train_lbl_{i:02d}.npy", lbl)

# Generate test set (2 held-out images)
test_images = []
test_labels = []
for i in range(2):
    img, lbl = make_labeled_image(seed=1000 + i, n_cells=rng.integers(10, 15))
    test_images.append(img)
    test_labels.append(lbl)
    skio.imsave(f"test_img_{i:02d}.png", img)
    np.save(f"test_lbl_{i:02d}.npy", lbl)

print(f"Training dataset: {len(train_images)} images")
for i, (img, lbl) in enumerate(zip(train_images, train_labels)):
    print(f"  train_{i}: shape={img.shape}, {lbl.max()} instances")
print()
print(f"Held-out test set: {len(test_images)} images")
for i, (img, lbl) in enumerate(zip(test_images, test_labels)):
    print(f"  test_{i}: shape={img.shape}, {lbl.max()} instances")""")

    b.code("""rng = np.random.default_rng(42)

# Visualize the training dataset
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(4):
    axes[0, i].imshow(train_images[i], cmap='gray'); axes[0, i].set_title(f"train img {i}"); axes[0, i].axis('off')
    axes[1, i].imshow(train_labels[i], cmap='tab20'); axes[1, i].set_title(f"train lbl {i}"); axes[1, i].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**What you should be seeing.** Eight training images with randomly generated synthetic round cells, and their instance masks (each cell is a different integer label). The test set (not shown) follows the same pattern but with different random seeds.""")


# ---------------------------------------------------------------------------
# Baseline: pretrained inference
# ---------------------------------------------------------------------------
def section_baseline(b):
    b.md("""## Step 2: Baseline — pretrained model on held-out images

Load a pretrained Cellpose model (default `cyto3`) and run it on the held-out test images. This is the **before** snapshot.""")

    b.code("""from cellpose import models, core

use_gpu = core.use_gpu()
print(f"Using GPU: {use_gpu}")

# Load the pretrained model (cyto3 is the default)
model_pretrained = models.CellposeModel(gpu=use_gpu, model_type="cyto3")
print("Pretrained model loaded.")

# Run on held-out test images
masks_pretrained_test = []
for i, test_img in enumerate(test_images):
    masks, _, _ = model_pretrained.eval(test_img, diameter=None, channels=[0, 0])
    masks_pretrained_test.append(masks)
    print(f"Test image {i}: predicted {masks.max()} objects (true: {test_labels[i].max()})")""")

    b.code("""# Quantify baseline performance
print("Baseline metrics (pretrained on held-out test set):")
print()
for i, (pred, gt) in enumerate(zip(masks_pretrained_test, test_labels)):
    iou, dice = iou_dice_binary(pred, gt)
    inst = match_instances(pred, gt, iou_threshold=0.5)
    print(f"Test image {i}:")
    print(f"  Pixel IoU       : {iou:.3f}")
    print(f"  Pixel Dice      : {dice:.3f}")
    print(f"  Instance prec   : {inst['precision']:.3f}  ({inst['tp']} TP, {inst['fp']} FP)")
    print(f"  Instance recall : {inst['recall']:.3f}  ({inst['tp']} TP, {inst['fn']} FN)")
    print()""")

    b.md("""**Expected observation.** On synthetic data the pretrained model likely does reasonably well (IoU > 0.6, recall > 0.7) because the synthetic round-cell distribution overlaps with Cellpose's training data. If performance were poor, that would be the signal to fine-tune. Here it's okay-but-improvable, which is the realistic case.""")


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
def section_finetuning(b):
    b.md("""## Step 3: Fine-tune on labeled training data

Cellpose 2.0+ exposes a `model.train()` API. We pass training images, ground-truth masks, and a handful of hyperparameters, then let the model adapt.

**Predict before fine-tuning.** On 8 training images with ~12 cells each, how many epochs do you expect to need before the model overfits?
- (a) Very few (5–10) — small dataset, quick memorization.
- (b) Moderate (20–50) — the model finds a sweet spot before overfitting.
- (c) Many (100+) — with regularization, the model resists overfitting for a long time.

The default we use is **n_epochs=20**, tuned for Colab T4 wall-clock time. You can experiment with the `#@param` slider below.""")

    b.code("""# @title Fine-tune hyperparameters { run: \"auto\" }
n_epochs = 20  # @param {type: \"slider\", min: 5, max: 100, step: 5}
batch_size = 2  # @param {type: \"slider\", min: 1, max: 8, step: 1}
learning_rate = 0.1  # @param

print(f"Fine-tuning config:")
print(f"  n_epochs    : {n_epochs}")
print(f"  batch_size  : {batch_size}")
print(f"  learning_rate : {learning_rate}")""")

    b.code("""# Cellpose v3/v4 train() API:
# model.train(train_data, train_labels, ...) trains in-place (modifies model weights)
# Convert training data to numpy arrays for the train call

X_train = np.array(train_images)  # (8, 200, 200) uint8
Y_train = np.array(train_labels)  # (8, 200, 200) int32

print(f"Training data shape: {X_train.shape}")
print(f"Training label shape: {Y_train.shape}")

# Create a fresh copy of the model so we don't mutate the loaded pretrained instance
# (In production, you'd checkpoint before fine-tuning.)
model_finetuned = models.CellposeModel(gpu=use_gpu, model_type="cyto3")

# Train
print("\\nStarting fine-tuning...")
import time
t0 = time.time()

# Cellpose v3+ train() signature: train(train_data, train_labels, test_data=None, test_labels=None, ...)
# We use train_data + train_labels; test_data and test_labels are optional for validation logging
model_finetuned.train(
    X_train, Y_train,
    test_data=np.array(test_images),    # validation set (optional; for logging only)
    test_labels=np.array(test_labels),
    n_epochs=n_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    channels=[0, 0],  # grayscale input
)

elapsed = time.time() - t0
print(f"Fine-tuning completed in {elapsed:.1f}s.""")

    b.md("""⚠ **Note on Cellpose API stability.** The `model.train()` signature may vary across Cellpose 3.x and 4.x releases. If you encounter an `TypeError` on the `train()` call above, check the current Cellpose documentation at https://cellpose.readthedocs.io/. The pattern stays the same (load → train → eval), but exact parameters shift.""")


# ---------------------------------------------------------------------------
# Inference with fine-tuned model
# ---------------------------------------------------------------------------
def section_finetune_infer(b):
    b.md("""## Step 4: Evaluate fine-tuned model on held-out test set

Run the fine-tuned model on the same held-out test images and compare metrics.""")

    b.code("""# Run fine-tuned model on held-out test images
masks_finetuned_test = []
for i, test_img in enumerate(test_images):
    masks, _, _ = model_finetuned.eval(test_img, diameter=None, channels=[0, 0])
    masks_finetuned_test.append(masks)
    print(f"Test image {i}: predicted {masks.max()} objects (true: {test_labels[i].max()})")""")

    b.code("""# Quantify fine-tuned performance
print("Fine-tuned metrics (on held-out test set):")
print()
for i, (pred, gt) in enumerate(zip(masks_finetuned_test, test_labels)):
    iou, dice = iou_dice_binary(pred, gt)
    inst = match_instances(pred, gt, iou_threshold=0.5)
    print(f"Test image {i}:")
    print(f"  Pixel IoU       : {iou:.3f}")
    print(f"  Pixel Dice      : {dice:.3f}")
    print(f"  Instance prec   : {inst['precision']:.3f}  ({inst['tp']} TP, {inst['fp']} FP)")
    print(f"  Instance recall : {inst['recall']:.3f}  ({inst['tp']} TP, {inst['fn']} FN)")
    print()""")


# ---------------------------------------------------------------------------
# Comparison: before vs. after
# ---------------------------------------------------------------------------
def section_comparison(b):
    b.md("""## Step 5: Side-by-side comparison

Pretrained vs. fine-tuned on the same test image.""")

    b.code("""# @title Compare on one test image { run: \"auto\" }
test_idx = 0  # @param {type: \"slider\", min: 0, max: 1, step: 1}

img = test_images[test_idx]
gt = test_labels[test_idx]
pred_pre = masks_pretrained_test[test_idx]
pred_ft = masks_finetuned_test[test_idx]

iou_pre, dice_pre = iou_dice_binary(pred_pre, gt)
iou_ft, dice_ft = iou_dice_binary(pred_ft, gt)
inst_pre = match_instances(pred_pre, gt, iou_threshold=0.5)
inst_ft = match_instances(pred_ft, gt, iou_threshold=0.5)

print(f"Test image {test_idx}:")
print()
print(f"Pretrained:")
print(f"  IoU: {iou_pre:.3f}, Dice: {dice_pre:.3f}")
print(f"  Precision: {inst_pre['precision']:.3f}, Recall: {inst_pre['recall']:.3f}")
print()
print(f"Fine-tuned:")
print(f"  IoU: {iou_ft:.3f}, Dice: {dice_ft:.3f}")
print(f"  Precision: {inst_ft['precision']:.3f}, Recall: {inst_ft['recall']:.3f}")
print()
print(f"Improvement:")
print(f"  ΔIoU: {iou_ft - iou_pre:+.3f}, ΔDice: {dice_ft - dice_pre:+.3f}")
print(f"  ΔRecall: {inst_ft['recall'] - inst_pre['recall']:+.3f}")

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
p1, p99 = np.percentile(img, [1, 99])
axes[0].imshow(img, cmap='gray', vmin=p1, vmax=p99); axes[0].set_title("Image"); axes[0].axis('off')
show_mask(axes[1], pred_pre, "Pretrained"); axes[1].axis('off')
show_mask(axes[2], pred_ft, "Fine-tuned"); axes[2].axis('off')
show_mask(axes[3], gt, "Ground truth"); axes[3].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**What you should be seeing.** On synthetic in-distribution data, fine-tuning likely improves both IoU and recall compared to the pretrained model. The improvement is usually modest (ΔRecall ~0.05–0.15) because the pretrained model already does reasonably well; fine-tuning adapts to your specific sample type, but can't create information that wasn't there.

**The tradeoff.** If the improvement is large (ΔRecall > 0.3), that suggests the pretrained model was badly mismatched to your data. Fine-tuning helped a lot. If improvement is small (< 0.05), the fine-tuning may be learning per-image variation rather than generalizing.""")


# ---------------------------------------------------------------------------
# Reflection: learning vs. memorization
# ---------------------------------------------------------------------------
def section_reflection(b):
    b.md("""## Step 6: Reflection — when does fine-tuning help vs. memorize?

**The honest question.** We fine-tuned on 8 images and tested on 2 held-out images. Did the model learn generalizable features about your sample type, or did it memorize the training distribution?

**Red flags for memorization:**
- Test performance is *much better* than training performance (impossible on proper validation).
- Fine-tuning on random labels gives similar improvement to fine-tuning on real labels (the model is just fitting noise).
- Performance on out-of-distribution test data (a different cell type, different stain, different microscope) is poor despite good in-distribution performance.

**Green flags for learning:**
- Fine-tuning consistently improves a *diverse* test set (many images from the same sample type).
- The improvement persists on *out-of-distribution* data (different microscope, different prep, slight morphological shift).
- Ablation: fine-tuning on fewer epochs gives less improvement; more epochs give more — the model is learning, not just memorizing.

**In this lab.** We used:
- 8 training images (small).
- 2 test images (tiny — barely statistical significance).
- Same distribution (synthetic round cells) for train and test.

This setup risks both overfitting (8 images can be memorized) and false confidence (test set is too similar to training set). In production:
- Collect 50–100+ training images if possible.
- Hold out a truly different test set (different sample type, different prep, different microscope).
- Use k-fold cross-validation or a formal validation protocol.

**The practical path forward:**
1. If you have 10+ of your own labeled images, fine-tune.
2. If you have fewer, consider using Cellpose's interactive annotation mode to label more.
3. If you have exactly these few, fine-tune for 10–20 epochs max, then validate hard on real held-out data.
4. If your held-out performance plateaus or drops, you've hit the memorization regime — stop there.""")

    b.md("""## Closing: next steps

**You've now covered the full supervised-learning arc:**
- Lab 01: Pretrained models (use what's available).
- Lab 02: Validation metrics (quantify success).
- Lab 09 (this): Fine-tuning (adapt to your data when pretrained isn't enough).

**Where to go next:**
- **Human-in-the-loop (Cellpose interactive).** Instead of pre-labeling 50 images, label a few, fine-tune, flag uncertain regions for correction, repeat. [Cellpose GUI tutorial](https://cellpose.readthedocs.io/).
- **Full Cellpose training docs.** The `model.train()` API has more options: custom `channels`, `rescale`, regularization. [Cellpose training guide](https://cellpose.readthedocs.io/).
- **Broader training workflows.** ZeroCostDL4Mic and Notebook 13 show how to build validation protocols for any fine-tuned model.
- **Open questions for your own data:** How many training images do you actually need? How much does microscope/stain variation hurt transfer? These are empirical questions best answered by trying.""")


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def main():
    b = CellBuilder("nb09")
    section_title(b)
    section_setup(b)
    section_helpers(b)
    section_data(b)
    section_baseline(b)
    section_finetuning(b)
    section_finetune_infer(b)
    section_comparison(b)
    section_reflection(b)
    build_notebook(b.cells, "09_cellpose_finetune")


if __name__ == "__main__":
    main()
