"""
Build script for notebook.ipynb (Lab 1: Pretrained Segmentation with Cellpose-SAM).

This script is the source-of-truth for the Lab 1 notebook. Edit this script and
re-run it to regenerate the .ipynb. Direct edits to the .ipynb JSON should be
avoided — they will be overwritten on the next build.

Usage:
    python build_notebook.py

Outputs:
    notebook.ipynb — Jupyter notebook for the 90-minute hands-on lab

Design notes:
    - Targets free-tier Google Colab (T4 GPU). Falls back to CPU with a
      visible warning rather than failing.
    - Uses skimage.data for both the "easy" and "hard" cases — zero downloads,
      deterministic, never moves. The hard case is a noise-degraded variant of
      the same dataset to surface Cellpose-SAM's failure modes.
    - Pins Cellpose to >=4.0,<5.0 (the SAM-integrated v4 release line).
    - Cellpose v4 API: `cellpose.models.Cellpose` was removed. Use only
      `cellpose.models.CellposeModel`. Default model is 'cpsam'. The eval()
      method returns 3 values (masks, flows, styles) — the v3 `diams` is gone.
    - Saves outputs to a workshop-relative path so Lab 2 can pick them up.
"""

import json
import uuid
from pathlib import Path

OUT = Path(__file__).parent / "notebook.ipynb"


def _id() -> str:
    """Stable-per-build cell ID. nbformat 5.1.4+ requires every cell to have one."""
    return uuid.uuid4().hex[:12]


def md(source: str) -> dict:
    """Markdown cell."""
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": source.lstrip("\n").rstrip() + "\n",
    }


def code(source: str) -> dict:
    """Code cell."""
    return {
        "cell_type": "code",
        "id": _id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.lstrip("\n").rstrip() + "\n",
    }


CELLS = []


# -----------------------------------------------------------------------------
# 1. Title and goals
# -----------------------------------------------------------------------------
CELLS.append(md("""
# Lab 1 — Pretrained Segmentation with Cellpose-SAM

**Lab time:** 90 minutes (workshop slot 13:00–14:30)
**Prerequisites:** Notebook 00 completed; GPU runtime enabled in Colab
**Tool version:** Cellpose v4 (Cellpose-SAM, 2025+)

## Goals

By the end of this lab, you should be able to:

1. Run pretrained Cellpose-SAM on a microscopy image and obtain instance-level segmentation masks.
2. Visualize segmentation overlays on the original image.
3. Identify at least three distinct types of segmentation error in the outputs and articulate why each likely occurred.
4. Compute simple per-object features (count, area, equivalent diameter) and summarize their distributions.
5. Decide whether the segmentation is fit for a downstream quantitative purpose.

## Structure

This lab walks the same image-processing workflow twice — once on a "clean" case where Cellpose-SAM is expected to perform well, then on a deliberately harder case where it is expected to struggle. The point is to make the failure modes as visible as the successes. We then quantify per-object features and save the outputs to disk for Lab 2 (validation).

> **Pedagogical note for instructors:** Resist the urge to skip past the harder case quickly. The single most common failure mode in workshop labs is "the easy case worked, so the model works." Drawing attention to the harder case explicitly — and pausing for the predict-then-reveal exercise — is what distinguishes critical use of AI tools from button-pushing.
"""))


# -----------------------------------------------------------------------------
# 2. Setup and imports
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Setup

The next cell installs and imports the dependencies. On a fresh Colab runtime, the install takes about 1–2 minutes (mostly downloading the Cellpose-SAM model weights).

If you see a warning about no GPU available, stop and switch the runtime to GPU before continuing: **Runtime → Change runtime type → T4 GPU**. Cellpose will run on CPU but each segmentation will take 1–2 minutes instead of a few seconds.
"""))

CELLS.append(code("""
# Setup cell — installs and verifies the runtime.
# Safe to run multiple times.
import sys, subprocess, importlib

IN_COLAB = "google.colab" in sys.modules

# Install cellpose v4 (Cellpose-SAM). Pinned to the v4 release line.
# Other dependencies (numpy, scikit-image, matplotlib, pandas, tifffile)
# are pre-installed on Colab; we re-check them below.
def _ensure(pkg, install_name=None, version_spec=""):
    install_name = install_name or pkg
    try:
        importlib.import_module(pkg)
    except ImportError:
        cmd = [sys.executable, "-m", "pip", "install", "--quiet", install_name + version_spec]
        subprocess.check_call(cmd)

if IN_COLAB:
    # cellpose v4 may not be pre-installed; install with version pin.
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "cellpose>=4.0,<5.0"]
    )

_ensure("numpy")
_ensure("skimage", install_name="scikit-image")
_ensure("matplotlib")
_ensure("pandas")
_ensure("tifffile")

# Imports
import numpy as np
import matplotlib.pyplot as plt
from skimage import data as skdata
from skimage.measure import regionprops_table
from skimage.color import label2rgb
import pandas as pd
import tifffile
from pathlib import Path

# Cellpose imports — v4 API. Note: cellpose.models.Cellpose was REMOVED in v4.
# Only CellposeModel is available; default pretrained_model='cpsam'.
from cellpose import models, io
import cellpose

print(f"cellpose version: {cellpose.__version__}")

# GPU check
try:
    import torch
    has_gpu = torch.cuda.is_available()
    print(f"GPU available: {has_gpu}")
    if has_gpu:
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: No GPU detected. Cellpose will run on CPU and be slow.")
        print("  In Colab: Runtime → Change runtime type → T4 GPU, then restart.")
except ImportError:
    has_gpu = False
    print("torch not available — cellpose will run on CPU.")

# Reproducibility — seed for any synthetic noise we generate below.
np.random.seed(42)
"""))


# -----------------------------------------------------------------------------
# 3. Choose the dataset
# -----------------------------------------------------------------------------
CELLS.append(md("""
## A note on the dataset

We use `skimage.data.human_mitosis()` for both cases:

- **Easy case:** the original image — clean DAPI-stained nuclei from a fluorescence microscopy dataset. Cellpose-SAM is expected to handle this well.
- **Hard case:** the same image, with strong synthetic noise added. This simulates a low-SNR acquisition (short exposure, low photon budget, deeper tissue) and surfaces failure modes that you will encounter in real low-SNR data.

Using one canonical dataset means the lab is fully self-contained — no downloads, no broken URLs, no waiting. The pedagogical content (where does the model succeed, where does it fail, why) is identical to what you would observe on real data.

> If you brought your own data, the **Extensions** section at the bottom of this notebook walks through swapping it in.
"""))

CELLS.append(code("""
# Easy case: clean DAPI nuclei from skimage.
img_easy = skdata.human_mitosis()
print(f"img_easy shape: {img_easy.shape}, dtype: {img_easy.dtype}")
print(f"  intensity range: [{img_easy.min()}, {img_easy.max()}]")
"""))


# -----------------------------------------------------------------------------
# 4. Display the easy case
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Display the first image

Always look at your data before running a model on it. Take 30 seconds to read the image: how many objects do you see? Are they well-separated? Are any touching or overlapping? Roughly how big are they in pixels?
"""))

CELLS.append(code("""
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_easy, cmap="gray")
ax.set_title(f"Easy case — clean nuclei ({img_easy.shape[0]}×{img_easy.shape[1]}, dtype={img_easy.dtype})")
ax.axis("off")
plt.tight_layout()
plt.show()
"""))


# -----------------------------------------------------------------------------
# 5. Run Cellpose-SAM on the easy case
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Run Cellpose-SAM on the easy case

We use the default `'cpsam'` model — Cellpose's SAM-based pretrained model. In v4, you no longer need to choose between `'cyto'` and `'nuclei'` for most images; the SAM backbone generalizes across stain types.

The first call downloads the model weights (~400 MB) and may take 30–60 seconds. Subsequent calls reuse the cached weights.
"""))

CELLS.append(code("""
# Initialize the Cellpose-SAM model. In v4, the default pretrained_model
# is 'cpsam' — no need to pass model_type.
model = models.CellposeModel(gpu=has_gpu)

# Run inference. The eval() signature in v4 returns 3 values: masks, flows, styles.
# (The v3 'diams' return is gone — v4 is robust to diameter and does not estimate it.)
# `diameter=None` lets the model use its built-in size handling.
masks_easy, flows_easy, styles_easy = model.eval(img_easy, diameter=None)

n_objects_easy = int(masks_easy.max())
print(f"Detected {n_objects_easy} objects in the easy case.")
"""))


# -----------------------------------------------------------------------------
# 6. Visualize the segmentation
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Visualize the segmentation

Three views, side by side:

1. **Original** — the raw image.
2. **Mask** — instance labels (each detected object gets a unique integer; background is 0).
3. **Overlay** — the original image with the segmentation rendered as colored translucent regions.

The overlay is the most useful view for spotting errors — under-segmentation (two cells merged into one), over-segmentation (one cell split into multiple), and missed objects all appear as discrepancies between the image content and the colored regions.
"""))

CELLS.append(code("""
def show_segmentation(img, masks, title_prefix=""):
    \"\"\"Display image, mask, and overlay side by side.\"\"\"
    overlay = label2rgb(masks, image=img, bg_label=0, alpha=0.4, image_alpha=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title(f"{title_prefix}original"); axes[0].axis("off")
    axes[1].imshow(masks, cmap="nipy_spectral"); axes[1].set_title(f"{title_prefix}mask ({int(masks.max())} objects)"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title(f"{title_prefix}overlay"); axes[2].axis("off")
    plt.tight_layout()
    plt.show()

show_segmentation(img_easy, masks_easy, title_prefix="EASY: ")
"""))


# -----------------------------------------------------------------------------
# 7. Discussion: easy case
# -----------------------------------------------------------------------------
CELLS.append(md("""
## ✋ Pause and look

Before moving on, take 2 minutes with the overlay above:

- **Where did Cellpose succeed?** Most of the well-separated nuclei should be cleanly captured.
- **Are there any merged objects?** Look at clusters of touching nuclei — did the model split them correctly?
- **Are there any missed objects?** Look at faint or partial nuclei near the image edges.
- **Are there any false positives?** Bright background spots that got labeled as objects.

Capture your observations in your head or in a margin note. We will compare these to the harder case below.
"""))


# -----------------------------------------------------------------------------
# 8. Quantify per-object features
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Quantify per-object features

A segmentation is just an intermediate result. The biological question is usually quantitative — *how many cells*, *what size*, *what intensity*. We extract per-object features using `skimage.measure.regionprops_table`.
"""))

CELLS.append(code("""
def quantify(img, masks):
    \"\"\"Return a per-object feature DataFrame.\"\"\"
    props = regionprops_table(
        masks,
        intensity_image=img,
        properties=("label", "area", "equivalent_diameter_area",
                    "eccentricity", "mean_intensity"),
    )
    df = pd.DataFrame(props)
    return df

df_easy = quantify(img_easy, masks_easy)
print(f"Per-object feature table (easy case): {len(df_easy)} rows")
df_easy.head(10)
"""))


# -----------------------------------------------------------------------------
# 9. Plot feature distributions
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Summarize and plot

Before looking at the histograms, **predict** what the area and equivalent-diameter distributions should look like. For a clean nuclei image with no biological gradient, you would expect roughly unimodal, mildly right-skewed distributions. Major deviations (multiple peaks, very long tails) often signal segmentation errors.
"""))

CELLS.append(code("""
def plot_distributions(df, title_prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].hist(df["area"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("area (pixels)"); axes[0].set_ylabel("count")
    axes[0].set_title(f"{title_prefix}area")
    axes[1].hist(df["equivalent_diameter_area"], bins=30, color="seagreen", edgecolor="white")
    axes[1].set_xlabel("equivalent diameter (pixels)")
    axes[1].set_title(f"{title_prefix}equivalent diameter")
    axes[2].hist(df["eccentricity"], bins=30, color="indianred", edgecolor="white")
    axes[2].set_xlabel("eccentricity (0=circle, 1=line)")
    axes[2].set_title(f"{title_prefix}eccentricity")
    plt.tight_layout()
    plt.show()

plot_distributions(df_easy, title_prefix="EASY: ")
print(df_easy[["area", "equivalent_diameter_area", "eccentricity", "mean_intensity"]].describe())
"""))


# -----------------------------------------------------------------------------
# 10. Reflection on the easy case
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Brief reflection

When Cellpose-SAM works well, the workflow is straightforward: load → run → quantify. There is a real risk in stopping here — if the only test case in your workshop or your paper is an easy one, you have not actually tested the model. The next cells move to the harder case to make this point concrete.
"""))


# -----------------------------------------------------------------------------
# 11. Build and run the harder image
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Now: a harder case

We construct the harder case by adding strong Gaussian and Poisson noise to the same image, simulating a low-SNR acquisition. The objects are still there — a human annotator could still find them — but Cellpose-SAM will struggle in characteristic ways.
"""))

CELLS.append(code("""
def degrade(img, gauss_sigma=25, poisson_scale=0.3, seed=42):
    \"\"\"Simulate low-SNR acquisition: scale down (Poisson stand-in) and add Gaussian noise.\"\"\"
    rng = np.random.default_rng(seed)
    base = img.astype(np.float32)
    # Reduce signal (simulates short exposure / low photon budget).
    scaled = base * poisson_scale
    # Add additive Gaussian read-noise.
    noisy = scaled + rng.normal(0, gauss_sigma, size=base.shape)
    # Clip to valid range and cast back to uint8.
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

img_hard = degrade(img_easy)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(img_easy, cmap="gray"); axes[0].set_title("Easy (original)"); axes[0].axis("off")
axes[1].imshow(img_hard, cmap="gray"); axes[1].set_title("Hard (degraded: low signal + noise)"); axes[1].axis("off")
plt.tight_layout()
plt.show()
"""))


# -----------------------------------------------------------------------------
# 12. Predict-then-reveal
# -----------------------------------------------------------------------------
CELLS.append(md("""
## ✋ Predict before you run

Before running Cellpose-SAM on the hard case, predict:

- Will it detect *fewer* objects, *more* objects, or about the same?
- Will the per-object **area** distribution be similar, larger, or smaller? Why?
- Where do you expect the model to fail most — on the bright/well-separated nuclei, or the faint clustered ones?

Write down your predictions, then run the next cell.
"""))

CELLS.append(code("""
masks_hard, flows_hard, styles_hard = model.eval(img_hard, diameter=None)
n_objects_hard = int(masks_hard.max())
print(f"Easy case detected {n_objects_easy} objects.")
print(f"Hard case detected {n_objects_hard} objects.")
print(f"Difference: {n_objects_hard - n_objects_easy:+d}")

show_segmentation(img_hard, masks_hard, title_prefix="HARD: ")
"""))


# -----------------------------------------------------------------------------
# 13. Quantify on the hard image
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Quantify the hard case

Run the same feature-extraction pipeline on the hard segmentation and compare distributions.
"""))

CELLS.append(code("""
df_hard = quantify(img_hard, masks_hard)
plot_distributions(df_hard, title_prefix="HARD: ")
print(df_hard[["area", "equivalent_diameter_area", "eccentricity", "mean_intensity"]].describe())
"""))


# -----------------------------------------------------------------------------
# 14. Side-by-side comparison
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Side by side: easy vs. hard

The single most useful comparison: object count, mean area, and area distribution shape.

> **Discussion prompt:** if you reported these counts in a paper — say, "we detected N nuclei per field" — would you trust the hard-case number? What would you do differently?
"""))

CELLS.append(code("""
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].imshow(label2rgb(masks_easy, image=img_easy, bg_label=0, alpha=0.4, image_alpha=1.0))
axes[0, 0].set_title(f"EASY: {n_objects_easy} objects"); axes[0, 0].axis("off")
axes[0, 1].imshow(label2rgb(masks_hard, image=img_hard, bg_label=0, alpha=0.4, image_alpha=1.0))
axes[0, 1].set_title(f"HARD: {n_objects_hard} objects"); axes[0, 1].axis("off")
axes[1, 0].hist(df_easy["area"], bins=30, alpha=0.6, label="easy", color="steelblue")
axes[1, 0].hist(df_hard["area"], bins=30, alpha=0.6, label="hard", color="indianred")
axes[1, 0].set_xlabel("area (pixels)"); axes[1, 0].set_ylabel("count"); axes[1, 0].legend()
axes[1, 0].set_title("area distribution")
axes[1, 1].hist(df_easy["mean_intensity"], bins=30, alpha=0.6, label="easy", color="steelblue")
axes[1, 1].hist(df_hard["mean_intensity"], bins=30, alpha=0.6, label="hard", color="indianred")
axes[1, 1].set_xlabel("mean intensity"); axes[1, 1].legend()
axes[1, 1].set_title("mean-intensity distribution")
plt.tight_layout()
plt.show()
"""))


# -----------------------------------------------------------------------------
# 15. Try a different pretrained model
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Try a different pretrained model

Cellpose v4 ships several pretrained variants alongside `'cpsam'`. The earlier `'nuclei'` and `'cyto3'` (v3-era) models are still available and were trained with different sample mixes. They may behave better or worse than the SAM model on a given image — the only way to find out is to try.

This cell swaps to the `'nuclei'` model and re-runs on the hard image. **Predict before you run:** will it do better, worse, or about the same?
"""))

CELLS.append(code("""
# Load an alternative pretrained model.
# Note: the exact set of available pretrained_model strings can shift between
# minor cellpose versions. If 'nuclei' is not recognized, try 'cyto3' or check
# `print(models.MODEL_NAMES)` for the current valid set.
try:
    model_alt = models.CellposeModel(gpu=has_gpu, pretrained_model="nuclei")
    masks_alt, _, _ = model_alt.eval(img_hard, diameter=None)
    n_alt = int(masks_alt.max())
    print(f"'nuclei' model detected {n_alt} objects on the hard case.")
    print(f"  (compared to {n_objects_hard} from the default 'cpsam' model)")
    show_segmentation(img_hard, masks_alt, title_prefix="HARD/'nuclei': ")
except Exception as e:
    print(f"Could not load alternative model: {e}")
    print("Try one of: ", getattr(models, 'MODEL_NAMES', '(MODEL_NAMES not exposed)'))
"""))


# -----------------------------------------------------------------------------
# 16. Adjust segmentation parameters
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Adjust segmentation parameters

Two parameters that meaningfully affect the output:

- `flow_threshold` — how strictly the model enforces flow consistency. **Lower** = more permissive, more candidate masks (more false positives). **Higher** = stricter, fewer masks.
- `cellprob_threshold` — the probability cutoff for a pixel to count as foreground. **Lower** = more pixels classified as cells. **Higher** = fewer pixels, smaller masks.

We sweep three combinations and look at the effect on the hard case.
"""))

CELLS.append(code("""
sweeps = [
    {"flow_threshold": 0.4, "cellprob_threshold": 0.0},   # cellpose default
    {"flow_threshold": 0.6, "cellprob_threshold": -1.0},  # more permissive
    {"flow_threshold": 0.2, "cellprob_threshold": 1.0},   # stricter
]

fig, axes = plt.subplots(1, len(sweeps), figsize=(6 * len(sweeps), 6))
for ax, params in zip(axes, sweeps):
    masks_p, _, _ = model.eval(img_hard, diameter=None, **params)
    n = int(masks_p.max())
    overlay = label2rgb(masks_p, image=img_hard, bg_label=0, alpha=0.4, image_alpha=1.0)
    ax.imshow(overlay)
    ax.set_title(f"flow={params['flow_threshold']}, cellprob={params['cellprob_threshold']}\\n→ {n} objects")
    ax.axis("off")
plt.tight_layout()
plt.show()
"""))


# -----------------------------------------------------------------------------
# 17. When does parameter tuning help?
# -----------------------------------------------------------------------------
CELLS.append(md("""
## ✋ When does parameter tuning help, and when does it just produce different mistakes?

The sweep above will probably show three different segmentations, each with its own object count. It is tempting to keep tuning until the segmentation "looks right" — but without ground truth, you cannot tell whether you are *fitting parameters* or *fitting biological reality*. Lab 2 introduces ground truth and validation metrics; the parameters that produce the *visually most plausible* segmentation are not necessarily the parameters that produce the *biologically most accurate* segmentation. Hold this thought until Lab 2.
"""))


# -----------------------------------------------------------------------------
# 18. Save outputs for Lab 2
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Save outputs for Lab 2

Lab 2 will pick up these outputs and validate them against ground truth. We save the easy and hard masks plus the per-object feature tables.
"""))

CELLS.append(code("""
out_dir = Path("./lab1_outputs")
out_dir.mkdir(exist_ok=True)

# Save masks as TIFFs (preserves integer instance labels).
tifffile.imwrite(out_dir / "img_easy.tif", img_easy)
tifffile.imwrite(out_dir / "masks_easy.tif", masks_easy.astype(np.uint16))
tifffile.imwrite(out_dir / "img_hard.tif", img_hard)
tifffile.imwrite(out_dir / "masks_hard.tif", masks_hard.astype(np.uint16))

# Save feature tables as CSV.
df_easy.to_csv(out_dir / "features_easy.csv", index=False)
df_hard.to_csv(out_dir / "features_hard.csv", index=False)

print(f"Saved Lab 1 outputs to: {out_dir.resolve()}")
for p in sorted(out_dir.iterdir()):
    print(f"  {p.name} ({p.stat().st_size / 1024:.1f} KB)")
"""))


# -----------------------------------------------------------------------------
# 19. Closing reflection and bridge to Lab 2
# -----------------------------------------------------------------------------
CELLS.append(md("""
## Closing reflection

What this lab demonstrated:

1. Pretrained Cellpose-SAM works well on clean, in-distribution data — the workflow is genuinely *load → run → quantify*.
2. On out-of-distribution data (here: low-SNR), the same workflow produces results that **look plausible but are systematically wrong**.
3. Different pretrained variants and different parameter settings produce different "wrong" answers. Without ground truth, the choice between them is guesswork.

**Bridge to Lab 2.** Lab 2 introduces ground-truth annotations and the metrics (IoU, Dice, count error, biological-feature error) that let you tell *which* wrong answer is *less* wrong — and whether the difference matters for the biological question you are asking.

## Extensions (optional, if you finish early)

- **Try Cellpose's diameter estimation explicitly.** Pass a fixed `diameter=` value (e.g., 30 pixels) and observe how the segmentation changes.
- **Run on your own image.** Replace `img_easy` with a single channel from your own data: `from skimage.io import imread; img_mine = imread('/path/to/your.tif')`. Note that color images need to be reduced to a single channel first (e.g., `img_mine[..., 0]` for the red channel).
- **Apply a per-object filter.** Drop objects below an area threshold (`df_filtered = df_easy[df_easy['area'] > 50]`) and observe how the distribution shifts.

## What this lab did *not* do

- Fine-tune Cellpose on custom data (deferred to a later workshop).
- Compare Cellpose to alternatives (Stardist, Mesmer, segment-anything) — Lab 3 option B touches segment-anything.
- Process 3D or time-series data.
"""))


# -----------------------------------------------------------------------------
# Notebook envelope
# -----------------------------------------------------------------------------
NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
        "colab": {
            "provenance": [],
            "name": "Lab 1 — Cellpose-SAM segmentation",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT.write_text(json.dumps(NOTEBOOK, indent=1))
    n_md = sum(1 for c in CELLS if c["cell_type"] == "markdown")
    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"Wrote {OUT} ({len(CELLS)} cells: {n_md} markdown, {n_code} code)")


if __name__ == "__main__":
    main()
