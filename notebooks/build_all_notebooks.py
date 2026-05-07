"""
Build script for the workshop's lab notebooks.

Generates five Jupyter notebooks:
  00_setup_self_check.ipynb     — pre-workshop verification
  01_cellpose_segmentation.ipynb — Lab 1
  02_validation_quantification.ipynb — Lab 2
  03a_denoising_n2v.ipynb       — Lab 3 option A
  03b_foundation_model_segmentation.ipynb — Lab 3 option B

Run:
    python build_all_notebooks.py
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------
class CellBuilder:
    """Counts cells for unique IDs per notebook."""

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
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
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
# Notebook 00 — Setup and Self-Check
# ---------------------------------------------------------------------------
def build_notebook_00():
    b = CellBuilder("n00")
    b.md("""# Notebook 00 — Setup and Self-Check

**Purpose.** Verify that your Colab (or local Jupyter) environment is ready for the workshop, and exercise the basic Python and image-data skills the labs assume. Run this notebook before the workshop day. Re-run on the morning of the workshop to confirm nothing has drifted.

**Estimated time.** 20–30 minutes.

By the end of this notebook you should know whether you are ready for the workshop or whether to attend the optional ramp-up evening session.""")

    b.md("## Step 1 — Detect your environment")
    b.code("""import sys
import platform

IN_COLAB = "google.colab" in sys.modules
print("Python    :", sys.version.split()[0])
print("Platform  :", platform.platform())
print("Runtime   :", "Google Colab" if IN_COLAB else "Local Jupyter")""")

    b.md("""## Step 2 — Install the dependencies

This cell installs only what the self-check needs (numpy, matplotlib, scikit-image, requests). The lab notebooks each install their own additional dependencies.""")
    b.code("""# Idempotent install. If you already have these, pip will skip them.
%pip install --quiet numpy matplotlib scikit-image requests tifffile
print("Dependencies installed.")""")

    b.md("""## Step 3 — Download a canonical sample image

We'll use one of the Cellpose example images. If your network blocks the download, the next cell falls back to a synthetic image so the rest of the notebook still runs.""")
    b.code("""import os
import requests
import numpy as np

SAMPLE_URL = "http://www.cellpose.org/static/data/img02.png"
SAMPLE_PATH = "sample.png"

try:
    r = requests.get(SAMPLE_URL, timeout=20)
    r.raise_for_status()
    with open(SAMPLE_PATH, "wb") as f:
        f.write(r.content)
    print(f"Downloaded {SAMPLE_PATH} ({len(r.content)} bytes)")
    download_ok = True
except Exception as e:
    print(f"Download failed: {e}")
    print("Falling back to a synthetic image so the notebook still runs.")
    download_ok = False""")

    b.md("## Step 4 — Load and inspect the image")
    b.code("""from skimage import io as skio

if download_ok:
    img = skio.imread(SAMPLE_PATH)
else:
    # Synthetic fallback: a small image with a few bright blobs
    rng = np.random.default_rng(42)
    img = np.zeros((200, 200), dtype=np.uint8)
    for _ in range(8):
        cy, cx = rng.integers(20, 180, size=2)
        r = rng.integers(8, 16)
        Y, X = np.ogrid[:200, :200]
        mask = (Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2
        img[mask] = 255
    img = img + rng.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)

print("shape :", img.shape)
print("dtype :", img.dtype)
print("min   :", int(img.min()))
print("max   :", int(img.max()))
print("mean  :", float(img.mean().round(2)))""")

    b.md("""## Step 5 — Predict before you display

Before you run the next cell, look at the shape, dtype, and stats above and form a mental picture. What do you expect to see?

Then run the cell.""")
    b.code("""import matplotlib.pyplot as plt

# Pick a sensible display contrast
p1, p99 = np.percentile(img, [1, 99])

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(img, cmap="gray", vmin=p1, vmax=p99)
ax.set_title(f"Sample image  ({img.shape[0]}x{img.shape[1]})")
ax.axis("off")
plt.show()""")

    b.md("""## Step 6 — Numpy slicing exercise

A short hands-on check. The cell below extracts a subregion and computes its mean, then compares to the whole-image mean. Skim the code, predict whether the subregion mean will be higher or lower, then run.""")
    b.code("""# Take a 60x60 subregion from the center of the image
h, w = img.shape[:2]
cy, cx = h // 2, w // 2
sub = img[cy - 30:cy + 30, cx - 30:cx + 30]

sub_mean = sub.mean()
img_mean = img.mean()
print(f"Whole-image mean : {img_mean:.2f}")
print(f"Subregion mean   : {sub_mean:.2f}")
print(f"Difference       : {sub_mean - img_mean:+.2f}")""")

    b.md("""## Step 7 — Self-grading and recommendation

This cell summarises the self-check and tells you whether to attend the optional ramp-up evening session.""")
    b.code("""env_ok = True
load_ok = "img" in dir() and img.size > 0
slice_ok = "sub" in dir() and sub.size > 0

print("=" * 50)
print("Self-check summary")
print("=" * 50)
print(f"  Environment ready : {'OK' if env_ok else 'MISSING'}")
print(f"  Image loaded      : {'OK' if load_ok else 'MISSING'}")
print(f"  Slicing exercise  : {'OK' if slice_ok else 'MISSING'}")
print()

if env_ok and load_ok and slice_ok:
    print("Recommendation: you are ready for the workshop.")
elif env_ok and load_ok:
    print("Recommendation: you are likely ready. Consider the ramp-up if you")
    print("found the slicing exercise unfamiliar.")
else:
    print("Recommendation: please attend the ramp-up evening session.")
    print("Email the instructors if you get stuck before the workshop.")""")

    b.md("""## What this notebook did *not* do

- Train any model
- Perform any segmentation
- Touch any GPU
- Use any of the workshop's lab tools (Cellpose, Noise2Void, segment-anything)

Those are deliberately reserved for the workshop day. The self-check is just a runtime sanity check.

If you want to push further before the workshop, look at the optional cells in the ramp-up evening notebooks (`ramp-up/notebooks/`).""")

    build_notebook(b.cells, "00_setup_self_check")


# ---------------------------------------------------------------------------
# Notebook 01 — Cellpose-SAM Segmentation (Lab 1)
# ---------------------------------------------------------------------------
def build_notebook_01():
    b = CellBuilder("n01")
    b.md("""# Notebook 01 — Pretrained Segmentation with Cellpose-SAM (Lab 1)

**Lab time.** 90 minutes.
**Tool.** Cellpose-SAM (Cellpose v4 — Pachitariu, Rariden, Stringer, 2025).

**Learning goals.**

1. Run pretrained Cellpose on canonical microscopy data.
2. Inspect the segmentation output critically.
3. Identify cases where the model succeeds and where it fails.
4. Quantify simple morphological features from the segmentation.
5. Save outputs for the Lab 2 validation notebook.

We will deliberately try the model on both an easy case (where it shines) and a harder case (where it struggles). The failure analysis is half the lesson.""")

    b.md("""## Setup

GPU is recommended. On Colab: `Runtime → Change runtime type → GPU` (T4 free tier is fine). Cellpose runs on CPU too; just slower.""")
    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

# Install Cellpose. The latest release pulls in cellpose-SAM by default.
%pip install --quiet "cellpose>=3.0" matplotlib scikit-image tifffile pandas

import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.measure import regionprops_table
import pandas as pd
print("Imports OK.")""")

    b.code("""# Check GPU availability (informational; Cellpose handles fallback automatically)
try:
    import torch
    print("torch    :", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device         :", torch.cuda.get_device_name(0))
except ImportError:
    print("torch not installed (cellpose will install it)")""")

    b.md("""## Generate the working dataset

Two synthetic images we control fully: an **easy** case (round, well-separated cells, similar to Cellpose's training distribution) and a **harder** case (irregular shapes, denser packing — engineered to fall outside the easy distribution). Generating them inline keeps the notebook self-contained — no external URLs to rot.

Real canonical data comes in a later cell — once we've established the basic pattern, we apply Cellpose-SAM to three publicly available scikit-image research datasets.""")
    b.code("""from scipy.ndimage import gaussian_filter

def make_easy_image(seed=0, size=200, n_cells=12):
    \"\"\"Easy case: round, well-separated cells. Cellpose-friendly.\"\"\"
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=float)
    centers = rng.uniform(20, size-20, (n_cells, 2))
    radii = rng.uniform(10, 18, n_cells)
    for (cy, cx), r in zip(centers, radii):
        Y, X = np.ogrid[:size, :size]
        img[(Y - cy)**2 + (X - cx)**2 <= r**2] = rng.uniform(0.6, 1.0)
    img = gaussian_filter(img, sigma=1.0) + rng.normal(0, 0.05, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def make_hard_image(seed=2, size=200, n_cells=18):
    \"\"\"Harder case: irregular, dense, varying intensity. Out of distribution.\"\"\"
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=float)
    centers = rng.uniform(15, size-15, (n_cells, 2))
    for cy, cx in centers:
        Y, X = np.ogrid[:size, :size]
        ry = rng.uniform(6, 14)
        rx = rng.uniform(6, 14) * rng.uniform(0.7, 1.4)
        ang = rng.uniform(0, np.pi)
        Yr = (Y - cy) * np.cos(ang) + (X - cx) * np.sin(ang)
        Xr = -(Y - cy) * np.sin(ang) + (X - cx) * np.cos(ang)
        img[(Yr / ry)**2 + (Xr / rx)**2 <= 1] = rng.uniform(0.3, 0.9)
    img = gaussian_filter(img, sigma=0.8) + rng.normal(0, 0.08, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

# Save as PNG so the rest of the notebook can read them as files
img_easy_synth = make_easy_image()
img_hard_synth = make_hard_image()
skio.imsave("sample_easy.png", img_easy_synth)
skio.imsave("sample_hard.png", img_hard_synth)
print(f"sample_easy.png: shape={img_easy_synth.shape} dtype={img_easy_synth.dtype}")
print(f"sample_hard.png: shape={img_hard_synth.shape} dtype={img_hard_synth.dtype}")""")

    b.md("""## Load Cellpose-SAM

We initialize the default Cellpose model. In Cellpose v4 (Cellpose-SAM) the default model uses the SAM transformer backbone for improved generalization.""")
    b.code("""from cellpose import models, core

# core.use_gpu() picks the best available device automatically
use_gpu = core.use_gpu()
print(f"Using GPU: {use_gpu}")

# CellposeModel is the v3+/v4 entry point
model = models.CellposeModel(gpu=use_gpu)
print("Model loaded.")""")

    b.md("""## Run on the easy image""")
    b.code("""img_easy = skio.imread("sample_easy.png")
# If the image is RGB, Cellpose expects channels in a specific format
print("Easy image shape:", img_easy.shape, "dtype:", img_easy.dtype)

# Cellpose API: model.eval returns (masks, flows, styles)
# diameter=None lets the model auto-estimate cell size
masks_easy, flows_easy, styles_easy = model.eval(
    img_easy,
    diameter=None,
    channels=[0, 0],  # grayscale (or use [2, 1] for RGB cyto + nuclei)
)
print(f"Found {masks_easy.max()} objects in the easy image.")""")

    b.md("## Visualize the easy result")
    b.code("""from matplotlib.colors import ListedColormap

# Build a colormap for instance labels (background black, then qualitative)
n_lbl = max(masks_easy.max(), 1)
cmap = ListedColormap(['black'] + plt.get_cmap('tab20')(np.linspace(0, 1, 20)).tolist())

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(img_easy, cmap='gray'); axes[0].set_title("Image")
axes[1].imshow(masks_easy, cmap=cmap, vmin=0, vmax=20); axes[1].set_title(f"Masks ({n_lbl} objects)")
axes[2].imshow(img_easy, cmap='gray')
axes[2].imshow(np.where(masks_easy > 0, masks_easy, np.nan), cmap=cmap, alpha=0.5)
axes[2].set_title("Overlay")
for a in axes:
    a.axis("off")
plt.tight_layout(); plt.show()""")

    b.md("""**Discussion.** Where did Cellpose succeed? Are there any cells it missed, merged, or split? Make notes mentally before moving on — comparing predictions to your own observations is the core habit Lab 2 will exercise.""")

    b.md("## Now try the harder image")
    b.code("""img_hard = skio.imread("sample_hard.png")
print("Hard image shape:", img_hard.shape)

masks_hard, flows_hard, styles_hard = model.eval(
    img_hard,
    diameter=None,
    channels=[0, 0],
)
print(f"Found {masks_hard.max()} objects in the harder image.")""")

    b.code("""n_lbl = max(masks_hard.max(), 1)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(img_hard, cmap='gray'); axes[0].set_title("Image (harder)")
axes[1].imshow(masks_hard, cmap=cmap, vmin=0, vmax=20); axes[1].set_title(f"Masks ({n_lbl} objects)")
axes[2].imshow(img_hard, cmap='gray')
axes[2].imshow(np.where(masks_hard > 0, masks_hard, np.nan), cmap=cmap, alpha=0.5)
axes[2].set_title("Overlay")
for a in axes:
    a.axis("off")
plt.tight_layout(); plt.show()""")

    b.md("""**Predict before you analyze.** Before reading the next cell, what do you expect about cell counts and feature distributions on the harder image relative to the easy one? Higher? Lower? Different shapes?""")

    b.md("## Quantify per-object features")
    b.code("""def feature_table(masks, name):
    if masks.max() == 0:
        return pd.DataFrame()
    props = regionprops_table(
        masks,
        properties=['label', 'area', 'equivalent_diameter', 'eccentricity', 'centroid'],
    )
    df = pd.DataFrame(props)
    df['image'] = name
    return df

df_easy = feature_table(masks_easy, "easy")
df_hard = feature_table(masks_hard, "hard")
df_all = pd.concat([df_easy, df_hard], ignore_index=True)

print("Per-image summary:")
print(df_all.groupby('image').agg(
    n_objects=('label', 'count'),
    mean_area=('area', 'mean'),
    mean_diam=('equivalent_diameter', 'mean'),
).round(2))""")

    b.md("## Compare feature distributions")
    b.code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for img_name, color in [("easy", "#4C72B0"), ("hard", "#C44E52")]:
    sub = df_all[df_all['image'] == img_name]
    axes[0].hist(sub['area'], bins=20, alpha=0.6, label=img_name, color=color)
    axes[1].hist(sub['equivalent_diameter'], bins=20, alpha=0.6, label=img_name, color=color)
axes[0].set_xlabel("Area (pixels)"); axes[0].set_ylabel("Count"); axes[0].legend(); axes[0].set_title("Cell area distribution")
axes[1].set_xlabel("Equivalent diameter (pixels)"); axes[1].set_ylabel("Count"); axes[1].legend(); axes[1].set_title("Cell diameter distribution")
plt.tight_layout(); plt.show()""")

    b.md("""**Reflection.** Are the distributions plausible for what you saw in the images? If the harder image's distribution looks *very* different from the easy one's, ask: is that biology, or is that segmentation failure?

This is the question Lab 2 will let you answer quantitatively against ground truth.""")

    # =========================================================================
    # Real canonical microscopy data (3 publicly available datasets)
    # =========================================================================
    b.md("""## Real canonical microscopy data

Now that the basic pattern works on synthetic images, let's apply Cellpose-SAM to **real** research data. We use three datasets bundled with scikit-image — they are real fluorescence-style microscopy images, included in the package, so no external download is needed.""")

    b.code("""from skimage import data

img_cell    = data.cell()                   # single fluorescence cell (2D grayscale)
img_mitosis = data.human_mitosis()          # real mitosis fluorescence (2D grayscale)
img_3d      = data.cells3d()                # 3D fluorescence stack: (z, channel, y, x)
img_3d_slice = img_3d[30, 1]                # mid-Z slice, nuclei channel

real_data = {
    "data.cell()":            img_cell,
    "data.human_mitosis()":   img_mitosis,
    "data.cells3d() (slice)": img_3d_slice,
}

for name, img in real_data.items():
    print(f"{name:<26}: shape={img.shape}  dtype={img.dtype}  range=[{img.min()}, {img.max()}]")""")

    b.md("""**Try this — predict before you reveal.** Looking at the dimensions and value ranges above, which image do you expect Cellpose-SAM to handle best? Worst? Why?""")

    b.code("""# Display the three real images
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, img) in zip(axes, real_data.items()):
    p1, p99 = np.percentile(img, [1, 99])
    ax.imshow(img, cmap='gray', vmin=p1, vmax=p99)
    ax.set_title(name); ax.axis('off')
plt.tight_layout(); plt.show()""")

    b.code("""# Run Cellpose-SAM on each, with auto-diameter
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['black'] + plt.get_cmap('tab20')(np.linspace(0, 1, 20)).tolist())

real_masks = {}
for name, img in real_data.items():
    masks, _, _ = model.eval(img, diameter=None, channels=[0, 0])
    real_masks[name] = masks
    print(f"{name:<26}: {masks.max()} objects detected")

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for col, (name, img) in enumerate(real_data.items()):
    masks = real_masks[name]
    p1, p99 = np.percentile(img, [1, 99])
    axes[0, col].imshow(img, cmap='gray', vmin=p1, vmax=p99)
    axes[0, col].set_title(name); axes[0, col].axis('off')
    axes[1, col].imshow(img, cmap='gray', vmin=p1, vmax=p99)
    axes[1, col].imshow(np.where(masks > 0, masks, np.nan), cmap=cmap, alpha=0.5, vmin=0, vmax=20)
    axes[1, col].set_title(f"{masks.max()} objects"); axes[1, col].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Discussion.** Compare your predictions to what you saw. Where did Cellpose-SAM perform well? Where did it struggle? The auto-diameter estimator is reasonable but not perfect — for any specific image, you may know the true diameter and providing it can help.

This brings us to the customization workshop.""")

    # =========================================================================
    # Customization workshop — four parameters
    # =========================================================================
    b.md("""## Customization workshop

Cellpose-SAM has knobs. Most users leave them on default and live with the result. The next four cells walk through the four most consequential parameters and show how output changes when you turn each knob. The lesson is mechanical: build intuition for *which knob does what*. The harder lesson — *parameters don't fix fundamentally OOD data* — is at the end.

For each knob: predict before you run, observe how the count changes, ask whether the change made the segmentation more or less *biologically* correct.""")

    # Knob 1: diameter
    b.md("""### Knob 1 — `diameter`

The most important parameter. Cellpose needs to know cell size in pixels. Default `diameter=None` lets the model auto-estimate. If you guess wrong by a factor of 2× either way, you get nonsense.

**Try this** — predict, then run:
- Auto (`None`) — should land near the true diameter (~22 px for our easy synthetic case).
- Very small (`5 px`) — predict: more objects, fewer, or different shape?
- Very large (`50 px`) — predict the same.""")

    b.code("""img = img_easy_synth  # synthetic easy case (12 cells, true diameter ~22 px)

diameter_settings = [("auto (None)", None), ("small (5 px)", 5), ("large (50 px)", 50)]
diameter_results = {}
for label, dia in diameter_settings:
    masks, _, _ = model.eval(img, diameter=dia, channels=[0, 0])
    diameter_results[label] = (masks, masks.max())
    print(f"diameter={label:<14}: {masks.max()} objects")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, (label, (masks, n)) in zip(axes, diameter_results.items()):
    ax.imshow(img, cmap='gray')
    ax.imshow(np.where(masks > 0, masks, np.nan), cmap=cmap, alpha=0.5, vmin=0, vmax=20)
    ax.set_title(f"diameter = {label}\\n{n} objects"); ax.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Lesson.** Diameter is the most important knob. Wrong diameter → wrong segmentation, even on a clean image. Auto-estimation is reasonable; explicit values help when you know the truth. Guessing badly produces confidently-wrong output.""")

    # Knob 2: model variants
    b.md("""### Knob 2 — model variant

Cellpose ships several specialized models. The default `cyto3` is for cytoplasm-stained cells. The `nuclei` model is trained on nuclei specifically. Same image, different model, different inductive bias.

**Try this** — apply both `cyto3` and `nuclei` to `img_mitosis` (which IS nuclei). Predict which model performs better before running.""")

    b.code("""from cellpose import models as cp_models

# Load both model variants
model_cyto3   = cp_models.CellposeModel(gpu=use_gpu, model_type="cyto3")
model_nuclei  = cp_models.CellposeModel(gpu=use_gpu, model_type="nuclei")

img_for_test = img_mitosis  # nuclei image; should favor 'nuclei' model

masks_cyto3,  _, _ = model_cyto3.eval(img_for_test,  diameter=None, channels=[0, 0])
masks_nuclei, _, _ = model_nuclei.eval(img_for_test, diameter=None, channels=[0, 0])

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
p1, p99 = np.percentile(img_for_test, [1, 99])
axes[0].imshow(img_for_test, cmap='gray', vmin=p1, vmax=p99); axes[0].set_title("Image (mitosis)")
axes[1].imshow(img_for_test, cmap='gray', vmin=p1, vmax=p99)
axes[1].imshow(np.where(masks_cyto3 > 0, masks_cyto3, np.nan), cmap=cmap, alpha=0.5, vmin=0, vmax=20)
axes[1].set_title(f"cyto3 model: {masks_cyto3.max()} objects")
axes[2].imshow(img_for_test, cmap='gray', vmin=p1, vmax=p99)
axes[2].imshow(np.where(masks_nuclei > 0, masks_nuclei, np.nan), cmap=cmap, alpha=0.5, vmin=0, vmax=20)
axes[2].set_title(f"nuclei model: {masks_nuclei.max()} objects")
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Lesson.** Model choice matters. Use the model trained on data closest to yours — `nuclei` for nuclei, `cyto`/`cyto3` for cytoplasm. The model-type registry is a real research signal: if no specialized model exists for your sample type, your task is harder.""")

    # Knob 3: flow_threshold
    b.md("""### Knob 3 — `flow_threshold`

Cellpose internally predicts gradient flows; cells are recovered from the flow. `flow_threshold` gates how strict the quality check on each candidate cell is. Default `0.4`. Lower = stricter (rejects more borderline cells). Higher = lenient (accepts more, including likely mistakes).

**Try this** — apply three values to the easy case. Predict the count direction.""")

    b.code("""img = img_easy_synth
flow_results = {}
for ft in [0.0, 0.4, 0.9]:
    masks, _, _ = model.eval(img, diameter=None, channels=[0, 0], flow_threshold=ft)
    flow_results[ft] = masks
    print(f"flow_threshold={ft}: {masks.max()} objects")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, (ft, masks) in zip(axes, flow_results.items()):
    ax.imshow(img, cmap='gray')
    ax.imshow(np.where(masks > 0, masks, np.nan), cmap=cmap, alpha=0.5, vmin=0, vmax=20)
    ax.set_title(f"flow_threshold = {ft}\\n{masks.max()} objects"); ax.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Lesson.** `flow_threshold` trades off completeness vs. purity. Strict (low value) drops borderline cells; lenient (high value) catches more cells but also more spurious detections.""")

    # Knob 4: cellprob_threshold
    b.md("""### Knob 4 — `cellprob_threshold`

Per-pixel cell-probability threshold. Default `0`. Higher = more confident (smaller masks, fewer cells). Lower = less confident (larger masks, more cells, potentially merging neighbors).

**Try this** — apply three values to the easy case. Watch how mask *boundaries* change, not just object counts.""")

    b.code("""img = img_easy_synth
cprob_results = {}
for cpt in [-2, 0, 2]:
    masks, _, _ = model.eval(img, diameter=None, channels=[0, 0], cellprob_threshold=cpt)
    cprob_results[cpt] = masks
    print(f"cellprob_threshold={cpt}: {masks.max()} objects")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, (cpt, masks) in zip(axes, cprob_results.items()):
    ax.imshow(img, cmap='gray')
    ax.imshow(np.where(masks > 0, masks, np.nan), cmap=cmap, alpha=0.5, vmin=0, vmax=20)
    ax.set_title(f"cellprob_threshold = {cpt}\\n{masks.max()} objects"); ax.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Lesson.** `cellprob_threshold` controls how confident the model has to be that a pixel is foreground. Lower thresholds expand masks (sometimes merging adjacent cells); higher thresholds shrink them (sometimes losing parts of cells).""")

    # =========================================================================
    # Failure-mode finale: tune everything on the OOD case
    # =========================================================================
    b.md("""## Failure-mode finale — can we rescue the hard case with parameters?

The hard synthetic image (`img_hard_synth`) is fundamentally out-of-distribution: irregular shapes, dense packing, anisotropic morphology. It does not look like Cellpose's training distribution.

Now that you've seen each knob, the obvious question: can you turn the knobs cleverly enough to rescue the hard case?

**Try this** — predict before running. If parameter tuning can fix OOD data, then "the model failed" is just a tuning problem. If it can't, then "the model failed" means something stronger.""")

    b.code("""img = img_hard_synth

# Three plausible parameter combinations to try
configs = [
    {"label": "auto everything",          "diameter": None, "flow_threshold": 0.4, "cellprob_threshold": 0},
    {"label": "small + lenient",           "diameter": 8,    "flow_threshold": 0.7, "cellprob_threshold": -1},
    {"label": "moderate + strict",         "diameter": 14,   "flow_threshold": 0.2, "cellprob_threshold": 1},
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, cfg in zip(axes, configs):
    masks, _, _ = model.eval(
        img, channels=[0, 0],
        diameter=cfg["diameter"],
        flow_threshold=cfg["flow_threshold"],
        cellprob_threshold=cfg["cellprob_threshold"],
    )
    ax.imshow(img, cmap='gray')
    ax.imshow(np.where(masks > 0, masks, np.nan), cmap=cmap, alpha=0.5, vmin=0, vmax=20)
    ax.set_title(f"{cfg['label']}\\n{masks.max()} objects"); ax.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Lesson.** Parameter tuning shifts output but cannot rescue fundamentally OOD data. When the model has never seen anything like this sample type, no combination of knob settings will produce correct segmentation. Different output ≠ correct output.

The right move when you hit an OOD case is *not* harder parameter tuning. It is one of:

- *Use a different model* (cyto3, nuclei, livecell, custom) trained closer to your sample type.
- *Fine-tune* a model on a small set of your own labeled images (Cellpose 2.0+).
- *Use a foundation model with prompts* (segment-anything, μSAM) where the user provides per-image guidance — Lab 3b.
- *Accept the limit* and use a different segmentation paradigm entirely (classical, semi-automatic, etc.).

Lab 2 will quantify what "correct" actually means against ground truth — turning this qualitative judgment into validation metrics.""")

    b.md("""## Save outputs for Lab 2

Lab 2 picks up these files. Don't skip this step.""")
    b.code("""np.save("masks_easy.npy", masks_easy)
np.save("masks_hard.npy", masks_hard)
df_all.to_csv("features.csv", index=False)
print("Saved: masks_easy.npy, masks_hard.npy, features.csv")""")

    b.md("""## Going broader — community alternatives to Cellpose

Cellpose is one tool among many. Notebook 04 catalogs the full ZeroCostDL4Mic / DL4MicEverywhere / BioImage Model Zoo ecosystem. Methods worth comparing against Cellpose:

- **StarDist** — instance segmentation with star-convex shape priors. Often better for densely packed nuclei. *Notebook 04, catalog.*
- **Mesmer / DeepCell** — whole-cell (membrane + nucleus) segmentation; strong on tissue. *Notebook 04, catalog.*
- **U-Net (semantic)** — when you don't need instance separation. Lighter, faster. *Notebook 04, catalog.*
- **Browse the BioImage Model Zoo live** — the API in Notebook 04 lets you search the registry by task and load any model. If a pretrained segmentation model exists for your specific sample type, you save weeks.

Picking the right tool is half the work; the catalog makes that explicit.""")

    b.md("""## Closing reflection

Lab 1 demonstrated the full pretrained-model workflow: load, run, visualize, quantify. The harder case probably surfaced confident-but-wrong predictions — exactly the failure mode the morning lecture flagged.

Lab 2 will give you the validation tools to put numbers on those failures. See you there.

**Where to go next on your own:**
- Try Cellpose-SAM on one of your own images. Swap the file path in the load cell.
- Try `model.eval(img, channels=[2, 1])` on RGB images for cytoplasm + nuclei.
- Visit the [Cellpose GitHub repository](https://github.com/MouseLand/cellpose) for the model zoo and human-in-the-loop training.""")

    build_notebook(b.cells, "01_cellpose_segmentation")


# ---------------------------------------------------------------------------
# Notebook 02 — Validation and Quantification (Lab 2)
# ---------------------------------------------------------------------------
def build_notebook_02():
    b = CellBuilder("n02")
    b.md("""# Notebook 02 — Validation and Quantification (Lab 2)

**Lab time.** 75 minutes.
**Prerequisites.** Notebook 01 completed; segmentation outputs saved.

**Learning goals.**

1. Compute pixel-level (IoU, Dice) and instance-level (precision, recall) metrics.
2. Compare predicted segmentations to ground truth.
3. Demonstrate the **metrics versus biology** gap: a model can have high IoU and still produce wrong biological measurements.
4. Choose validation metrics that match the biological question.

The lab's central learning moment is in Step 5 — pay attention there.""")

    b.md("## Setup and load Lab 1 outputs")
    b.code("""%pip install --quiet numpy scikit-image matplotlib pandas seaborn scipy
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load Lab 1 outputs. If they don't exist, fall back to synthetic data so
# the lab can still be completed.
have_lab1 = all(os.path.exists(p) for p in ["masks_easy.npy", "masks_hard.npy"])
if have_lab1:
    masks_easy = np.load("masks_easy.npy")
    masks_hard = np.load("masks_hard.npy")
    print("Loaded Lab 1 outputs.")
else:
    print("Lab 1 outputs not found. Generating synthetic data for the lab.")
    # Synthetic fallback: a 'predicted' segmentation we can score
    rng = np.random.default_rng(42)
    masks_easy = np.zeros((200, 200), dtype=np.int32)
    for i in range(8):
        cy, cx = rng.integers(20, 180, size=2)
        r = rng.integers(10, 18)
        Y, X = np.ogrid[:200, :200]
        masks_easy[(Y - cy)**2 + (X - cx)**2 <= r**2] = i + 1
    masks_hard = masks_easy.copy()
    masks_hard[masks_hard == 3] = 2  # merge two cells
    masks_hard[masks_hard == 6] = 0  # delete a cell

print(f"masks_easy: {masks_easy.shape}, {int(masks_easy.max())} objects")
print(f"masks_hard: {masks_hard.shape}, {int(masks_hard.max())} objects")""")

    b.md("""## Generate ground truth

Real ground truth is hard. For this lab we construct a synthetic ground truth that we *know* is correct, so we can measure exactly how the predictions deviate. In practice the ground truth would come from expert annotation.""")
    b.code("""# Use the easy mask as the 'truth' for both images, simulating a case where
# the model overfits the easy distribution and degrades on the harder one.
gt_easy = masks_easy.copy()

# For the hard ground truth, we'll perturb less aggressively — pretending the
# real biology has more cells than the model found.
gt_hard = masks_easy.copy()  # in this synthetic setup, ground truth is the same as easy
print(f"gt_easy: {int(gt_easy.max())} objects")
print(f"gt_hard: {int(gt_hard.max())} objects")""")

    b.md("## Compute pixel-level metrics (IoU, Dice)")
    b.code("""def iou_dice_binary(pred_mask, gt_mask):
    pred_bin = pred_mask > 0
    gt_bin = gt_mask > 0
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    return iou, dice

iou_e, dice_e = iou_dice_binary(masks_easy, gt_easy)
iou_h, dice_h = iou_dice_binary(masks_hard, gt_hard)

print(f"Easy image  : IoU = {iou_e:.3f}, Dice = {dice_e:.3f}")
print(f"Hard image  : IoU = {iou_h:.3f}, Dice = {dice_h:.3f}")""")

    b.md("""**Note the gap.** The easy image probably scores near 1.0; the hard image is a bit lower but still high. Pixel metrics make problems look *less* severe than they are. Watch what instance-level metrics show.""")

    b.md("## Compute instance-level metrics")
    b.code("""def match_instances(pred_mask, gt_mask, iou_threshold=0.5):
    \"\"\"Match each predicted instance to a GT instance by IoU >= threshold.

    Returns matched count, false positives, false negatives.
    \"\"\"
    pred_ids = [i for i in np.unique(pred_mask) if i != 0]
    gt_ids   = [i for i in np.unique(gt_mask) if i != 0]
    matched_pred = set()
    matched_gt = set()

    for p in pred_ids:
        pred_bin = (pred_mask == p)
        best_iou, best_g = 0.0, None
        for g in gt_ids:
            if g in matched_gt: continue
            gt_bin = (gt_mask == g)
            inter = (pred_bin & gt_bin).sum()
            union = (pred_bin | gt_bin).sum()
            if union == 0: continue
            iou_val = inter / union
            if iou_val > best_iou:
                best_iou, best_g = iou_val, g
        if best_iou >= iou_threshold:
            matched_pred.add(p)
            matched_gt.add(best_g)

    tp = len(matched_pred)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}

m_easy = match_instances(masks_easy, gt_easy)
m_hard = match_instances(masks_hard, gt_hard)
print("Easy:", m_easy)
print("Hard:", m_hard)""")

    b.md("""**Compare.** Pixel IoU might be 0.95+ on the harder case while instance recall is much lower. The metric you choose changes how bad the problem looks.""")

    b.md("## Compute biological measurements")
    b.code("""def biological_summary(pred, gt, label):
    pred_count = int(np.unique(pred[pred > 0]).size)
    gt_count = int(np.unique(gt[gt > 0]).size)
    return {
        "image": label,
        "predicted_count": pred_count,
        "true_count": gt_count,
        "count_error": pred_count - gt_count,
    }

bio_easy = biological_summary(masks_easy, gt_easy, "easy")
bio_hard = biological_summary(masks_hard, gt_hard, "hard")
df_bio = pd.DataFrame([bio_easy, bio_hard])
print(df_bio)""")

    b.md("## The metric–biology plot")
    b.code("""# Visualize the gap: high IoU does not mean a correct count.
import seaborn as sns

# Build a small set of perturbed predictions to plot a curve
rng = np.random.default_rng(0)
n_trials = 30
perturbations = np.linspace(0.0, 0.8, n_trials)

ious_curve = []
count_errors_curve = []
for p_strength in perturbations:
    test = gt_easy.copy()
    # Perturb: randomly delete a fraction of cells, randomly merge a few
    cell_ids = [i for i in np.unique(test) if i != 0]
    n_delete = int(p_strength * len(cell_ids) * 0.5)
    for cell in rng.choice(cell_ids, size=min(n_delete, len(cell_ids)), replace=False):
        test[test == cell] = 0
    iou_val, _ = iou_dice_binary(test, gt_easy)
    bio = biological_summary(test, gt_easy, "perturbed")
    ious_curve.append(iou_val)
    count_errors_curve.append(abs(bio["count_error"]))

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(ious_curve, count_errors_curve, c=perturbations, cmap='viridis', s=60, edgecolor='k', linewidth=0.5)
ax.set_xlabel("IoU (pixel overlap)")
ax.set_ylabel("Absolute count error")
ax.set_title("High IoU does not guarantee a correct cell count")
ax.invert_xaxis()
cbar = plt.colorbar(ax.collections[0], ax=ax, label="Perturbation strength")
plt.tight_layout(); plt.show()""")

    b.md("""## The key insight

A model can be 'good' by IoU and still produce systematically wrong biological measurements. **Choose validation metrics that match the biological question:**

- *How many cells?* → count error
- *What is the mean cell size?* → mean-area error
- *Is the size distribution different between conditions?* → distribution-comparison test (KS, Earth Mover's, etc.)
- *Is the model spatially biased?* → per-region precision/recall

Lab 2's central message: don't take the IoU number from a paper at face value. Ask whether it's measuring the question you actually care about.""")

    b.md("""## Closing reflection

Lab 2 makes the metrics-versus-biology gap visible. The next lab (Lab 3) extends the responsible-use thread to a different challenge: how do you validate a model output when, by definition, you don't have a clean version to compare against?

That challenge is what AI restoration (Lab 3a) and foundation-model segmentation (Lab 3b) both face. See you there.""")

    build_notebook(b.cells, "02_validation_quantification")


# ---------------------------------------------------------------------------
# Notebook 03a — Noise2Void Denoising (Lab 3 option A)
# ---------------------------------------------------------------------------
def build_notebook_03a():
    b = CellBuilder("n03a")
    b.md("""# Notebook 03a — AI Denoising with Noise2Void (Lab 3, Option A)

**Lab time.** 60 minutes.
**Tool.** Noise2Void (Krull, Buchholz, Jug, CVPR 2019).

**Learning goals.**

1. Apply self-supervised denoising to a noisy microscopy image.
2. Compare the denoised result to a clean reference and to the noisy input.
3. Identify cases where the model restores real signal vs invents features.
4. Articulate integrity-reporting expectations for AI-restored images.

**Note on scope.** A full Noise2Void training run takes hours on CPU. For the workshop we'll use a tiny synthetic example so the lab finishes in 60 minutes; the patterns we observe still hold. To do this with real data, follow the [n2v GitHub examples](https://github.com/juglab/n2v).""")

    b.md("## Setup and synthetic noisy data")
    b.code("""%pip install --quiet numpy matplotlib scikit-image scipy
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from scipy.ndimage import gaussian_filter

print("Imports OK.")""")

    b.md("""We'll use a synthetic 'clean' image (a few bright blobs on a dark background) and add realistic Poisson + Gaussian noise to simulate low-light fluorescence. Both clean and noisy versions are available for evaluation — though Noise2Void itself only trains on the noisy data.""")
    b.code("""rng = np.random.default_rng(0)

def make_clean_image(size=128, n_objects=10):
    img = np.zeros((size, size), dtype=float)
    for _ in range(n_objects):
        cy, cx = rng.integers(15, size-15, size=2)
        r = rng.integers(6, 12)
        Y, X = np.ogrid[:size, :size]
        img[(Y-cy)**2 + (X-cx)**2 <= r**2] = rng.uniform(0.5, 1.0)
    return gaussian_filter(img, sigma=1.0)

clean = make_clean_image()

def add_noise(img, photons=20):
    # Poisson + Gaussian read noise; simulates low-light fluorescence
    scaled = img * photons
    noisy = rng.poisson(scaled).astype(float) / photons
    noisy = noisy + rng.normal(0, 0.05, noisy.shape)
    return np.clip(noisy, 0, None)

noisy = add_noise(clean)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(clean, cmap='gray', vmin=0, vmax=1); axes[0].set_title("Clean reference (for evaluation only)")
axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1); axes[1].set_title("Noisy input (what Noise2Void sees)")
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**The Noise2Void principle.** The model learns to predict each pixel's value from its neighbors *without ever seeing a clean reference*. This works because true signal has spatial correlation; pixel-independent noise does not.

We'll use a tiny stand-in here — a Gaussian-filter denoiser that gives a comparable visual effect — so we can finish in 60 minutes without GPU training. The lessons about validation and hallucination are the same.""")
    b.code("""# Simple stand-in denoiser. In real Noise2Void, this would be a trained CNN.
denoised = gaussian_filter(noisy, sigma=1.5)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(noisy,    cmap='gray', vmin=0, vmax=1); axes[0].set_title("Noisy input")
axes[1].imshow(denoised, cmap='gray', vmin=0, vmax=1); axes[1].set_title("Denoised (stand-in)")
axes[2].imshow(clean,    cmap='gray', vmin=0, vmax=1); axes[2].set_title("Clean reference")
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("## Quantify the denoising")
    b.code("""def psnr(reference, prediction, data_range=1.0):
    mse = np.mean((reference - prediction) ** 2)
    if mse == 0: return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))

from skimage.metrics import structural_similarity as ssim

baseline_psnr = psnr(clean, noisy)
denoise_psnr  = psnr(clean, denoised)
baseline_ssim = ssim(clean, noisy, data_range=1.0)
denoise_ssim  = ssim(clean, denoised, data_range=1.0)

print(f"Baseline (noisy vs clean)    : PSNR={baseline_psnr:.2f} dB, SSIM={baseline_ssim:.3f}")
print(f"Denoised vs clean             : PSNR={denoise_psnr:.2f} dB, SSIM={denoise_ssim:.3f}")
print()
print(f"PSNR improvement              : +{denoise_psnr - baseline_psnr:.2f} dB")""")

    b.md("""**The metrics look great.** PSNR and SSIM both improved substantially. So the denoising worked, right?

Now look at the difference image — this is where the hallucination question becomes visible.""")

    b.md("## The hallucination check")
    b.code("""# Difference between denoised and clean: residual error
diff = denoised - clean

# Threshold to find structural disagreements (not just noise smoothing)
# Anywhere |diff| is large in the denoised result indicates either: noise that
# wasn't smoothed away, OR features the denoiser introduced/distorted.
threshold = 0.1
anomaly_mask = np.abs(diff) > threshold

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(denoised, cmap='gray', vmin=0, vmax=1); axes[0].set_title("Denoised")
axes[1].imshow(diff, cmap='RdBu_r', vmin=-0.3, vmax=0.3); axes[1].set_title("Diff (denoised − clean)")
axes[2].imshow(denoised, cmap='gray', vmin=0, vmax=1)
axes[2].imshow(np.where(anomaly_mask, 1, np.nan), cmap='autumn', alpha=0.5)
axes[2].set_title(f"Anomalies (|diff| > {threshold})")
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()

n_anomaly_px = anomaly_mask.sum()
total_px = anomaly_mask.size
print(f"Anomaly pixels: {n_anomaly_px} / {total_px} ({100*n_anomaly_px/total_px:.1f}%)")""")

    b.md("""**Where the model hallucinated.** Anomaly pixels mark locations where the denoised image differs from the clean reference in ways that aren't explained by noise smoothing alone. In a real workflow with no clean reference, you'd never see this — but the disagreement is real.

This is the central tension of AI restoration: it works, *and* it costs you something. The cost is integrity.""")

    b.md("""## Apply the model with no clean reference

In real experiments you don't have a clean reference. Here's what your workflow looks like in that case.""")
    b.code("""# Apply the same denoiser to a *new* noisy image (no clean reference)
new_clean = make_clean_image()
new_noisy = add_noise(new_clean)
new_denoised = gaussian_filter(new_noisy, sigma=1.5)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(new_noisy,   cmap='gray', vmin=0, vmax=1); axes[0].set_title("Noisy input (production)")
axes[1].imshow(new_denoised, cmap='gray', vmin=0, vmax=1); axes[1].set_title("Denoised (you only see this)")
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()

print("In production, you have only the right image. The left image is the only")
print("thing distinguishing 'real' from 'hallucinated' — and you don't have it.")""")

    b.md("""## Integrity reporting walkthrough

A short methods/figure-caption template you can use in publications when AI restoration appears in a figure:""")
    b.code("""template = '''
Methods:
  Image restoration was performed using [METHOD] (model version [V],
  trained on [DATA] for [N_STEPS] steps). The displayed images in
  Figures [X, Y, Z] show restored data; quantitative analyses
  reported in the text were performed on the original raw data.

Figure caption (where AI-restored images appear):
  "Image displayed has been restored with [METHOD vX]. Restoration
  may introduce features that were not present in the original
  measurement. Quantitative measurements shown were performed on
  the raw, unrestored data."
'''
print(template)""")

    b.md("""## Going broader — community alternatives to Noise2Void

Noise2Void is *self-supervised* — no clean reference required. The complement is *supervised* denoising, which produces higher-quality results when paired data is available. Notebook 04 catalogs the alternatives:

- **CARE (Content-Aware Image Restoration)** — supervised denoising. Higher ceiling than N2V when you have paired clean/noisy data. *Notebook 04, inline demo.*
- **DecoNoising (DL4ME)** — joint deconvolution and denoising. Useful when blur and noise are both present.
- **3D-RCAN** — 3D super-resolution that doubles as denoising for many use cases.
- **Browse the BioImage Model Zoo** — the API in Notebook 04 lets you find pretrained denoising models for specific microscopy modalities.

When deciding between N2V and CARE: do you have paired data? If yes, CARE. If no, N2V.""")

    b.md("""## Closing reflection

Lab 3a demonstrated:

1. AI denoising works — visually and by PSNR/SSIM.
2. AI denoising introduces structural changes — the model invents features in places.
3. Without a clean reference, you cannot detect these changes from the output alone.
4. Disclosure in publications is therefore not optional.

**Where to go next:**

- The [Noise2Void GitHub repo](https://github.com/juglab/n2v) for full training procedures on real data.
- [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) for ready-made Colab notebooks running CARE, N2V, and other restoration methods.
- The [DL4MicEverywhere](https://github.com/HenriquesLab/DL4MicEverywhere) container if you want to run this locally with full GPU.""")

    build_notebook(b.cells, "03a_denoising_n2v")


# ---------------------------------------------------------------------------
# Notebook 03b — Foundation-Model Segmentation (Lab 3 option B)
# ---------------------------------------------------------------------------
def build_notebook_03b():
    b = CellBuilder("n03b")
    b.md("""# Notebook 03b — Foundation-Model Segmentation (Lab 3, Option B)

**Lab time.** 60 minutes.
**Tool.** Segment Anything (SAM) and the microscopy-tuned variant (μSAM).

**Learning goals.**

1. Describe the foundation-model paradigm and how it differs from task-specific models.
2. Apply SAM with point and box prompts to segment objects.
3. Compare foundation-model output to a Cellpose baseline.
4. Identify validation challenges specific to foundation-model output.
5. Decide when foundation-model approaches are preferable to task-specific models.

**Setup note.** SAM model weights are large (~360 MB for vit_b). On Colab T4 this downloads in ~30 seconds; on CPU expect slower inference. For runtime safety, this notebook can also run with `simulate_sam=True` to demonstrate the workflow without a real SAM download.""")

    b.md("## Setup")
    b.code("""# SAM and helpers. micro-sam is large; we install it optionally.
%pip install --quiet "git+https://github.com/facebookresearch/segment-anything.git" matplotlib scikit-image numpy
import sys, os, urllib.request
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio

IN_COLAB = "google.colab" in sys.modules
print("In Colab:", IN_COLAB)""")

    b.code("""# We'll demo with simulated SAM behavior if downloading the real model fails.
# Toggle this to True to skip the model download.
SIMULATE_SAM = False
SAM_CKPT = "sam_vit_b_01ec64.pth"
SAM_URL  = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if not SIMULATE_SAM and not os.path.exists(SAM_CKPT):
    try:
        print(f"Downloading SAM model (~360 MB)...")
        urllib.request.urlretrieve(SAM_URL, SAM_CKPT)
        print(f"  Got {os.path.getsize(SAM_CKPT)//1024//1024} MB")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Falling back to simulated SAM behavior so the lab still runs.")
        SIMULATE_SAM = True""")

    b.md("""## Load a non-canonical microscopy image

We use a synthetic image deliberately designed to fall outside Cellpose's training distribution: irregular shapes that look more like tissue than cultured cells.""")
    b.code("""rng = np.random.default_rng(7)
size = 256

img = np.zeros((size, size), dtype=float)
# Add irregular blob-like structures
for _ in range(5):
    cy, cx = rng.integers(40, size-40, size=2)
    rr = rng.integers(15, 30)
    Y, X = np.ogrid[:size, :size]
    blob = ((Y-cy)/(rr*1.4))**2 + ((X-cx)/(rr*0.7))**2 <= 1
    img[blob] = rng.uniform(0.6, 1.0)
img = img + rng.normal(0, 0.05, img.shape)
img = np.clip(img, 0, 1)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(img, cmap='gray'); ax.set_title("Non-canonical image"); ax.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("## Set up the SAM predictor")
    b.code("""if not SIMULATE_SAM:
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
    predictor = SamPredictor(sam)
    # SAM expects a 3-channel uint8 image
    img_rgb = np.stack([img] * 3, axis=-1)
    img_rgb = (img_rgb * 255).astype(np.uint8)
    predictor.set_image(img_rgb)
    print("SAM ready.")
else:
    print("SIMULATE mode: skipping real SAM. We'll mock the prediction API.")""")

    b.md("""## Apply SAM with a point prompt

Click on one of the blobs (we'll just pick coordinates manually here). SAM treats the point as a foreground hint and segments the object containing it.""")
    b.code("""# Pick a point inside one of the blobs by inspecting the image
# (In real use, you'd click on it interactively in napari or similar.)
nonzero = np.argwhere(img > 0.5)
if len(nonzero) > 0:
    point = nonzero[len(nonzero) // 2]  # roughly middle of the brightest region
else:
    point = np.array([size // 2, size // 2])

input_point = np.array([[point[1], point[0]]])  # SAM uses (x, y) order
input_label = np.array([1])  # 1 = foreground

if not SIMULATE_SAM:
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    sam_mask = masks[best]
    print(f"SAM returned {len(masks)} candidate masks; best score = {scores[best]:.3f}")
else:
    # Simulated: threshold around the prompt point
    Y, X = np.ogrid[:size, :size]
    sam_mask = (Y - point[0])**2 + (X - point[1])**2 <= 25**2

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img, cmap='gray')
axes[0].plot(input_point[0, 0], input_point[0, 1], 'r*', markersize=18)
axes[0].set_title("Image + point prompt"); axes[0].axis('off')
axes[1].imshow(img, cmap='gray')
axes[1].imshow(np.where(sam_mask, 1, np.nan), cmap='autumn', alpha=0.5)
axes[1].set_title("SAM segmentation"); axes[1].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("## Apply SAM with a bounding box prompt")
    b.code("""# Bounding box around (roughly) the same blob
y0, x0 = max(point[0] - 30, 0), max(point[1] - 30, 0)
y1, x1 = min(point[0] + 30, size), min(point[1] + 30, size)
input_box = np.array([x0, y0, x1, y1])

if not SIMULATE_SAM:
    masks_box, scores_box, _ = predictor.predict(
        box=input_box[None, :],
        multimask_output=False,
    )
    sam_box_mask = masks_box[0]
else:
    Y, X = np.ogrid[:size, :size]
    sam_box_mask = (Y >= y0) & (Y < y1) & (X >= x0) & (X < x1) & (img > 0.4)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img, cmap='gray')
import matplotlib.patches as mpatches
rect = mpatches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='red', facecolor='none')
axes[0].add_patch(rect); axes[0].set_title("Image + box prompt"); axes[0].axis('off')
axes[1].imshow(img, cmap='gray')
axes[1].imshow(np.where(sam_box_mask, 1, np.nan), cmap='autumn', alpha=0.5)
axes[1].set_title("SAM segmentation (box)"); axes[1].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Compare the two outputs.** Did the point and box prompts produce similar masks? In our simple example they should; on real data they often diverge in informative ways.""")

    b.md("## Prompt sensitivity test")
    b.code("""# Move the point around and see how the mask changes
shifts = [(-20, 0), (0, 0), (20, 0)]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for i, (dy, dx) in enumerate(shifts):
    test_point = point + np.array([dy, dx])
    test_input = np.array([[test_point[1], test_point[0]]])

    if not SIMULATE_SAM:
        masks_t, scores_t, _ = predictor.predict(
            point_coords=test_input,
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best_t = int(np.argmax(scores_t))
        sam_mask_t = masks_t[best_t]
    else:
        Y, X = np.ogrid[:size, :size]
        sam_mask_t = (Y - test_point[0])**2 + (X - test_point[1])**2 <= 25**2

    axes[i].imshow(img, cmap='gray')
    axes[i].imshow(np.where(sam_mask_t, 1, np.nan), cmap='autumn', alpha=0.5)
    axes[i].plot(test_input[0, 0], test_input[0, 1], 'r*', markersize=15)
    axes[i].set_title(f"Point shifted by ({dy}, {dx})")
    axes[i].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**The validation challenge with foundation models.** Output depends on the prompt. There is no fixed 'training distribution' to validate against the way you would for a task-specific model.

For Lab 3b's setting, the right validation evidence asks:

- *Reproducibility* — does the same prompt always give the same output? (Yes, deterministic.)
- *Prompt sensitivity* — how much does the output change as the prompt moves? (Often: surprisingly little, but sometimes: a lot.)
- *Coverage* — for an automatic-mask-generation pass, does it produce all the objects you expect? Are there false positives?""")

    b.md("## When to use foundation models vs task-specific models")
    b.code("""decision_framework = '''
Use task-specific pretrained models (Cellpose, Stardist, Mesmer) when:
  - The model's training distribution covers your data
  - The model has been validated on cases similar to yours
  - You need consistent, automated batch processing

Use foundation models with prompts (SAM, μSAM) when:
  - No fitting task-specific model exists
  - You have a small number of images and can prompt per-image
  - You're doing exploratory analysis on novel sample types
  - You want flexible, prompt-driven segmentation

Train a custom model (fine-tune Cellpose, fine-tune a foundation model) when:
  - You have labeled data
  - The volume of work justifies the investment
  - The project will run for months or years
'''
print(decision_framework)""")

    b.md("""## Going broader — beyond SAM and μSAM

Notebook 04 catalogs the broader foundation-model and community-platform ecosystem:

- **BioImage Model Zoo browser** — live API to list, filter, and load *hundreds* of pretrained models. The "choose on the fly" pattern: at workshop time, browse what's available for your task and load it inline. *Notebook 04, Section 2.*
- **fnet (label-free prediction)** — predicts fluorescence from brightfield without staining. Demonstrates the generation-task pattern. *Notebook 04, inline demo.*
- **pix2pix (image-to-image translation)** — supervised translation between paired image domains. Common bioimage uses: virtual staining, modality translation. *Notebook 04, inline demo.*
- **CycleGAN** — unpaired translation, no paired data needed. *Notebook 04, catalog.*

Foundation models extend reach; the community catalog extends the menu of alternatives. Start with BiMZ when prototyping; if a pretrained model exists for your task, you save the training cost.""")

    b.md("""## Closing reflection

Foundation-model segmentation extends reach into novel sample types at the cost of greater prompt-engineering and validation work. The choice between task-specific and foundation-model approaches is a deliberate trade-off, not a default.

**Where to go next:**

- The [μSAM (micro-sam) GitHub repository](https://github.com/computational-cell-analytics/micro-sam) for the microscopy-specific SAM variant.
- [napari plugins for SAM](https://www.napari-hub.org/?search=sam) for interactive prompting.
- The [Cellpose-SAM](https://github.com/MouseLand/cellpose) v4 release combines task-specific Cellpose with the SAM transformer backbone — worth comparing.""")

    build_notebook(b.cells, "03b_foundation_model_segmentation")


# ---------------------------------------------------------------------------
# Build all notebooks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_notebook_00()
    build_notebook_01()
    build_notebook_02()
    build_notebook_03a()
    build_notebook_03b()
    print("\nAll notebooks built.")
