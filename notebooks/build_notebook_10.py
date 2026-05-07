"""
Build script for Notebook 10 — 3D Segmentation: Cellpose 3D + StarDist 3D.

Companion to Notebook 01 (2D Cellpose), extending to true volumetric segmentation.
The pedagogical arc: 2D on slices loses connectivity → run Cellpose 3D and StarDist 3D
on the full volume → compare inductive biases → discuss when 3D is worth the cost.

Run:
    python build_notebook_10.py
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/2026-workshop/notebooks/10_3d_segmentation.ipynb)

*Click the badge to open this notebook in Google Colab. For best performance, switch to a GPU runtime: Runtime → Change runtime type → T4 GPU.*""")

    b.md("""# Notebook 10 — 3D Segmentation: Cellpose 3D + StarDist 3D

**Status.** Post-workshop self-paced extension. Recommended after Notebook 01 (2D Cellpose).
**Estimated time.** 30–45 minutes on Colab T4.
**Prerequisites.** Notebook 01 (basic Cellpose workflow), comfort with volumetric microscopy data.

**Learning goals.**

1. Understand the difference between 2D-per-slice and true 3D segmentation: 2D stacking loses through-plane connectivity; 3D enforces it.
2. Run Cellpose 3D and StarDist 3D on the same 3D volume; observe how their inductive biases differ.
3. Quantify 3D outputs: per-object volume, equivalent diameter, count — compare models statistically.
4. Use `#@param` dropdowns to choose segmentation method and re-run the same analysis.
5. Decide when true 3D segmentation is worth the computational cost versus slice-by-slice 2D or classical approaches.

> **The 3D segmentation challenge.** In 2D, instance separation is a 2D problem: find connected components in a 2D label image. In 3D, it's harder: you need consistency across Z. Cellpose 3D and StarDist 3D tackle this differently — one via recurrent convolutions over Z, the other via star-convex shape priors in 3D. Both are expensive; we measure whether that cost is justified on a realistic volume.

> **A note on the form widgets.** Several cells below use `#@param` comments. In **Google Colab** these render as interactive form widgets (dropdowns) at the top of the cell. In **other environments** they appear as plain Python comments — edit the values directly and re-run the cell.""")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def section_setup(b):
    b.md("""## Setup

GPU is recommended. On Colab: `Runtime → Change runtime type → T4 GPU`. Both Cellpose 3D and StarDist 3D run on T4 memory without issue on the full `cells3d()` volume (60 × 256 × 256).""")

    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

# Install 3D-capable libraries
%pip install --quiet "cellpose>=3.0" "stardist>=0.8" csbdeep scikit-image matplotlib numpy scipy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from skimage import data as skdata
from skimage.measure import regionprops_table
import pandas as pd
import time

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
# Load the volume
# ---------------------------------------------------------------------------
def section_load_volume(b):
    b.md("""## Load the 3D volume

We use scikit-image's canonical `cells3d()` dataset: shape (z=60, c=2, y=256, x=256).
- Channel 0: membrane stain
- Channel 1: DAPI (nuclei)

We will segment the nuclei channel (channel 1) because it's cleaner and more interpretable.
The 60 Z-slices span the full depth of a confocal stack, so connectivity through Z is real and matters.""")

    b.code("""# Load the reference 3D dataset
cells_volume = skdata.cells3d()  # shape (z=60, c=2, y=256, x=256)
print(f"cells3d() shape: {cells_volume.shape}  dtype={cells_volume.dtype}")
print(f"  Channel 0 (membrane): range=[{cells_volume[:, 0].min()}, {cells_volume[:, 0].max()}]")
print(f"  Channel 1 (DAPI):     range=[{cells_volume[:, 1].min()}, {cells_volume[:, 1].max()}]")

# Extract the nuclei channel (channel 1) for segmentation
volume_zyx = cells_volume[:, 1]  # shape (z=60, y=256, x=256)
print(f"\\nNuclei volume (channel 1): shape={volume_zyx.shape}  dtype={volume_zyx.dtype}")""")

    b.md("""## Visualize three Z-slices through the volume

Show the geometry: what does the 3D data look like at different depths? This is the context for all downstream analysis.""")

    b.code("""# Show three slices from the volume
z_slices = [15, 30, 45]  # three representative Z positions
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, z in zip(axes, z_slices):
    p1, p99 = np.percentile(volume_zyx[z], [1, 99])
    ax.imshow(volume_zyx[z], cmap='Blues_r', vmin=p1, vmax=p99)
    ax.set_title(f"Z = {z}")
    ax.axis('off')
plt.tight_layout(); plt.show()""")


# ---------------------------------------------------------------------------
# 2D baseline
# ---------------------------------------------------------------------------
def section_2d_baseline(b):
    b.md("""## 2D-per-slice baseline: apply Cellpose 2D to each Z independently

This is the naive approach: run 2D segmentation on each Z-slice, stack the results. The problem is that instance labels don't carry meaning across Z — each slice gets fresh label IDs, even when the same cell spans multiple slices.

**Predict before you run.** If you manually traced nuclei through the 60 Z-slices and wrote down instance IDs that matched across slices (e.g., nucleus #3 spans Z=20–28), then stacked 2D results, would you expect the same instance IDs? Why or why not?""")

    b.code("""from cellpose import models, core

# Load Cellpose 2D model
use_gpu = core.use_gpu()
model_2d = models.CellposeModel(gpu=use_gpu, model_type='nuclei')
print(f"Loaded Cellpose 2D model (GPU: {use_gpu})")

# Apply 2D segmentation to each Z independently, stack into a fake 3D label volume
masks_2d_stack = []
print("Running Cellpose 2D on each Z slice...")
for z in range(volume_zyx.shape[0]):
    img_z = volume_zyx[z]
    masks_z, _, _ = model_2d.eval(img_z, diameter=None, channels=[0, 0])
    masks_2d_stack.append(masks_z)
    if (z + 1) % 20 == 0:
        print(f"  Z={z+1}/{volume_zyx.shape[0]}")

# Stack into (z, y, x) array
masks_2d_stacked = np.stack(masks_2d_stack, axis=0)
print(f"\\n2D-stacked masks shape: {masks_2d_stacked.shape}")
print(f"Unique labels across volume: {len(np.unique(masks_2d_stacked)) - 1} objects")
print(f"  (Note: same cell may have different IDs in different slices)")""")

    b.md("""**What you should be seeing.** The cell count is high because each Z-slice resets the label ID counter. If we truly stacked 3D results, we'd expect one label per object across all Z; here we get many more labels because the same cell gets a different ID in each slice it appears in.

This inefficiency motivates true 3D segmentation.""")


# ---------------------------------------------------------------------------
# Cellpose 3D path
# ---------------------------------------------------------------------------
def section_cellpose_3d(b):
    b.md("""## Cellpose 3D: 3D-aware segmentation

Cellpose 3D extends the 2D model with recurrent connections over Z, enforcing consistency across slices. Input shape: (z, y, x) or (z, c, y, x) if you have multiple channels.

**Predict before you run.** Cellpose 3D should return *fewer* labels than the 2D stack, because it enforces that the same cell has the same ID across Z. Should the count match the true number of nuclei? (Spoiler: not exactly — model errors still happen — but it should be more stable.)**""")

    b.code("""from cellpose.models import CellposeModel

# Load Cellpose 3D model
model_3d = CellposeModel(gpu=use_gpu, model_type='nuclei', ndim=3)
print("Loaded Cellpose 3D model")

# Run inference on the full 3D volume
# Input: (z, y, x) array
print("Running Cellpose 3D on the full volume...")
t0 = time.time()
masks_cellpose_3d, _, _ = model_3d.eval(
    volume_zyx,
    diameter=None,
    channels=[0, 0],
    do_3D=True,
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
print(f"Cellpose 3D masks shape: {masks_cellpose_3d.shape}")
print(f"Cellpose 3D found: {masks_cellpose_3d.max()} objects")""")

    b.md("""## Visualize Cellpose 3D output: three orthogonal mid-slices""")

    b.code("""# Show three orthogonal slices through the 3D label volume
z_mid = masks_cellpose_3d.shape[0] // 2
y_mid = masks_cellpose_3d.shape[1] // 2
x_mid = masks_cellpose_3d.shape[2] // 2

# Colormap for labels
cmap = ListedColormap(['black'] + plt.get_cmap('tab20')(np.linspace(0, 1, 20)).tolist())

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
# XY (Z-mid)
axes[0].imshow(masks_cellpose_3d[z_mid], cmap=cmap, vmin=0, vmax=20)
axes[0].set_title(f"XY plane (Z={z_mid})")
axes[0].axis('off')
# XZ (Y-mid)
axes[1].imshow(masks_cellpose_3d[:, y_mid, :], cmap=cmap, vmin=0, vmax=20)
axes[1].set_title(f"XZ plane (Y={y_mid})")
axes[1].axis('off')
# YZ (X-mid)
axes[2].imshow(masks_cellpose_3d[:, :, x_mid], cmap=cmap, vmin=0, vmax=20)
axes[2].set_title(f"YZ plane (X={x_mid})")
axes[2].axis('off')
plt.tight_layout(); plt.show()""")


# ---------------------------------------------------------------------------
# StarDist 3D path
# ---------------------------------------------------------------------------
def section_stardist_3d(b):
    b.md("""## StarDist 3D: star-convex shape priors in 3D

StarDist predicts star-convex distance maps in 3D: from each nucleus center, measure the distance to the boundary in many radial directions (e.g., 32 rays). Instance segmentation is then a watershed post-processing step. The advantage: the shape prior is built into the loss, so StarDist often handles dense nuclei better. The drawback: it assumes star-convex morphology (not always true for irregular cells).

**Note on pretrained models.** StarDist 3D comes with several pretrained models. We try `'3D_demo'` first; if that fails, we list alternatives. StarDist model names have changed in recent releases, so defensive error handling is important.""")

    b.code("""from stardist.models import StarDist3D

# Try to load the default 3D model
try:
    print("Loading StarDist 3D pretrained model...")
    stardist_model = StarDist3D.from_pretrained('3D_demo')
    print(f"Loaded StarDist 3D model: 3D_demo")
except Exception as e:
    print(f"Could not load '3D_demo': {e}")
    print("\\nAvailable StarDist 3D models:")
    try:
        available_models = StarDist3D.list_pretrained()
        for model_name in available_models:
            if '3d' in model_name.lower() or '3D' in model_name:
                print(f"  {model_name}")
    except:
        print("  (could not list models; check StarDist GitHub for current names)")
    print("\\nFalling back to brief explanation — see StarDist docs for model registry.")
    stardist_model = None""")

    b.code("""if stardist_model is not None:
    # Run inference
    # Input: (z, y, x) array; normalize via csbdeep
    from csbdeep.utils import normalize

    print("\\nRunning StarDist 3D inference...")
    t0 = time.time()

    # Normalize for csbdeep (1st and 99.8th percentile)
    volume_norm = normalize(volume_zyx, 1, 99.8)

    # Predict: returns (labels, distances) where labels is the instance mask
    labels, _ = stardist_model.predict_instances(volume_norm)
    elapsed = time.time() - t0

    masks_stardist_3d = labels.astype(np.int32)
    print(f"Done in {elapsed:.1f}s")
    print(f"StarDist 3D masks shape: {masks_stardist_3d.shape}")
    print(f"StarDist 3D found: {masks_stardist_3d.max()} objects")
else:
    print("\\nSkipping StarDist 3D inference because model load failed.")
    masks_stardist_3d = None""")

    b.md("""## Visualize StarDist 3D output: three orthogonal mid-slices

(Only if the model loaded successfully)""")

    b.code("""if masks_stardist_3d is not None:
    z_mid = masks_stardist_3d.shape[0] // 2
    y_mid = masks_stardist_3d.shape[1] // 2
    x_mid = masks_stardist_3d.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    # XY (Z-mid)
    axes[0].imshow(masks_stardist_3d[z_mid], cmap=cmap, vmin=0, vmax=20)
    axes[0].set_title(f"XY plane (Z={z_mid})")
    axes[0].axis('off')
    # XZ (Y-mid)
    axes[1].imshow(masks_stardist_3d[:, y_mid, :], cmap=cmap, vmin=0, vmax=20)
    axes[1].set_title(f"XZ plane (Y={y_mid})")
    axes[1].axis('off')
    # YZ (X-mid)
    axes[2].imshow(masks_stardist_3d[:, :, x_mid], cmap=cmap, vmin=0, vmax=20)
    axes[2].set_title(f"YZ plane (X={x_mid})")
    axes[2].axis('off')
    plt.tight_layout(); plt.show()
else:
    print("StarDist 3D model did not load; skipping visualization.")""")


# ---------------------------------------------------------------------------
# Choose your own
# ---------------------------------------------------------------------------
def section_choose_method(b):
    b.md("""## Choose a method and analyze

**Predict before you choose.** Look at the Cellpose 3D and StarDist 3D counts above. If they differ by a lot, which one do you expect to be more accurate? Why?""")

    b.code("""# @title Pick segmentation method { run: "auto" }
method = "Cellpose 3D"  # @param ["Cellpose 3D", "StarDist 3D", "Both side-by-side (2D-stacked, Cellpose, StarDist)"]

if method == "Cellpose 3D":
    chosen_mask = masks_cellpose_3d
    method_name = "Cellpose 3D"
    print(f"Selected: {method_name}")
    print(f"  Objects: {chosen_mask.max()}")

elif method == "StarDist 3D":
    if masks_stardist_3d is not None:
        chosen_mask = masks_stardist_3d
        method_name = "StarDist 3D"
        print(f"Selected: {method_name}")
        print(f"  Objects: {chosen_mask.max()}")
    else:
        print("StarDist 3D model did not load; falling back to Cellpose 3D")
        chosen_mask = masks_cellpose_3d
        method_name = "Cellpose 3D"

else:  # Both
    print("Comparing all three methods:")
    print(f"  2D-stacked (fake 3D): {masks_2d_stacked.max()} unique labels (but labels repeat across Z)")
    print(f"  Cellpose 3D:          {masks_cellpose_3d.max()} objects")
    if masks_stardist_3d is not None:
        print(f"  StarDist 3D:          {masks_stardist_3d.max()} objects")
    else:
        print(f"  StarDist 3D:          (model did not load)")
    chosen_mask = masks_cellpose_3d
    method_name = "Cellpose 3D (showing first; see next cell for comparison table)"
    print(f"\\nShowing Cellpose 3D for detailed analysis below.")""")


# ---------------------------------------------------------------------------
# Per-object statistics
# ---------------------------------------------------------------------------
def section_stats(b):
    b.md("""## Quantify per-object statistics

Use scikit-image's `regionprops_table` to compute volume, equivalent diameter, and other morphological features from the 3D label mask.

**What these metrics mean:**
- **Volume (voxels)**: total number of voxels in each object. Beware: volume in voxels depends on voxel size (isotropic here, but not always).
- **Equivalent diameter (voxels)**: diameter of a sphere with the same volume. Useful for comparing object sizes.
- **Count**: total number of objects found.""")

    b.code("""from skimage.measure import regionprops_table

def analyze_3d_mask(mask_3d, method_label):
    \"\"\"Analyze a 3D label mask and return a DataFrame of per-object stats.\"\"\"
    if mask_3d.max() == 0:
        return pd.DataFrame()

    props = regionprops_table(
        mask_3d,
        properties=['label', 'volume', 'equivalent_diameter_3d'],
    )
    df = pd.DataFrame(props)
    df['method'] = method_label
    return df

# Analyze the chosen method
df_chosen = analyze_3d_mask(chosen_mask, method_name)
print(f"\\n{method_name} statistics:")
print(df_chosen[['label', 'volume', 'equivalent_diameter_3d']].describe().round(2))
print(f"\\nTotal objects: {len(df_chosen)}")
print(f"Mean volume per object: {df_chosen['volume'].mean():.1f} voxels")
print(f"Mean equivalent diameter: {df_chosen['equivalent_diameter_3d'].mean():.1f} voxels")""")


# ---------------------------------------------------------------------------
# Compare all methods
# ---------------------------------------------------------------------------
def section_compare_methods(b):
    b.md("""## Compare all available methods side-by-side

Tabulate per-object stats from each method. Compare object counts, mean volumes, and distributions.

**Key question:** Do the methods agree on how many nuclei there are? If they disagree by a lot (e.g., Cellpose 3D says 50 but StarDist 3D says 80), that's a red flag: at least one of them is wrong in a way the other isn't. That's the failure mode we need to catch in validation.""")

    b.code("""# Compare counts and mean statistics across methods
results_summary = []

# 2D-stacked
results_summary.append({
    'method': '2D-stacked',
    'n_objects': masks_2d_stacked.max(),
    'mean_volume': None,  # not computed; labels repeat across Z
    'note': 'labels repeat across Z; counts unreliable',
})

# Cellpose 3D
df_cp3d = analyze_3d_mask(masks_cellpose_3d, 'Cellpose 3D')
results_summary.append({
    'method': 'Cellpose 3D',
    'n_objects': len(df_cp3d),
    'mean_volume': df_cp3d['volume'].mean() if len(df_cp3d) > 0 else None,
    'note': '',
})

# StarDist 3D (if available)
if masks_stardist_3d is not None:
    df_sd3d = analyze_3d_mask(masks_stardist_3d, 'StarDist 3D')
    results_summary.append({
        'method': 'StarDist 3D',
        'n_objects': len(df_sd3d),
        'mean_volume': df_sd3d['volume'].mean() if len(df_sd3d) > 0 else None,
        'note': '',
    })

df_compare = pd.DataFrame(results_summary)
print("Comparison of methods:")
print(df_compare.to_string(index=False))""")


# ---------------------------------------------------------------------------
# 3D visualization
# ---------------------------------------------------------------------------
def section_viz_3d(b):
    b.md("""## Visualize the 3D segmentation: max-projection + orthogonal slices

A max-projection collapses the 3D volume to 2D along the Z-axis, showing all nuclei in one image. Orthogonal slices show internal structure.""")

    b.code("""# Max-projection along Z (collapse 3D -> 2D)
max_proj = chosen_mask.max(axis=0)  # max label along Z

z_mid = chosen_mask.shape[0] // 2
y_mid = chosen_mask.shape[1] // 2
x_mid = chosen_mask.shape[2] // 2

fig, axes = plt.subplots(2, 2, figsize=(12, 11))

# Max-projection (top-left)
axes[0, 0].imshow(max_proj, cmap=cmap, vmin=0, vmax=20)
axes[0, 0].set_title(f"Max-projection (Z-collapse)\\n{chosen_mask.max()} objects")
axes[0, 0].axis('off')

# XY mid-plane (top-right)
axes[0, 1].imshow(chosen_mask[z_mid], cmap=cmap, vmin=0, vmax=20)
axes[0, 1].set_title(f"XY plane (Z={z_mid})")
axes[0, 1].axis('off')

# XZ mid-plane (bottom-left)
axes[1, 0].imshow(chosen_mask[:, y_mid, :], cmap=cmap, vmin=0, vmax=20)
axes[1, 0].set_title(f"XZ plane (Y={y_mid})")
axes[1, 0].axis('off')

# YZ mid-plane (bottom-right)
axes[1, 1].imshow(chosen_mask[:, :, x_mid], cmap=cmap, vmin=0, vmax=20)
axes[1, 1].set_title(f"YZ plane (X={x_mid})")
axes[1, 1].axis('off')

plt.tight_layout(); plt.show()""")


# ---------------------------------------------------------------------------
# Discussion
# ---------------------------------------------------------------------------
def section_discussion(b):
    b.md("""## When is 3D segmentation worth it?

We've now run three approaches on the same volume:

1. **2D-stacked:** Apply 2D segmentation to each Z independently, stack results. Fast, but loses through-plane connectivity. Instance labels don't carry meaning across Z.

2. **Cellpose 3D:** Uses recurrent 3D convolutions to enforce consistency across Z. More expensive (20–100× slower than 2D per-slice), but returns true 3D instances.

3. **StarDist 3D:** Uses star-convex distance maps in 3D. Also expensive, but often better on dense nuclei because the shape prior is enforced in the loss.

**Trade-offs:**

- **Speed:** 2D-stacked << Cellpose 3D ≈ StarDist 3D (both heavy).
- **Connectivity:** 2D-stacked has none; Cellpose 3D and StarDist 3D both enforce it.
- **Accuracy:** Depends on morphology. StarDist often wins on *densely packed* nuclei (shapes are star-convex). Cellpose 3D is more flexible for irregular morphologies.
- **Robustness:** When in doubt, use the method closest to your sample type. If the two disagree, that's a red flag.

**When to use each:**

- **2D-stacked (2D per-slice, no 3D tracking):** Quick prototyping, large volumes where memory is tight, or when you don't care about through-plane connectivity (e.g., if you're counting nuclei per slice, not tracing them through the volume).

- **True 3D (Cellpose 3D or StarDist 3D):** When connectivity matters (e.g., measuring nuclear volume, tracing cell morphology through depth, time-lapse tracking), or when dense packing requires shape awareness. Accept the compute cost.

- **Classical 3D segmentation (watershed, level-sets):** When the data is very far from the training distribution of neural models, and you have domain knowledge about the morphology (e.g., intensity-based watershed on high-contrast images). Not covered in this lab.

**A note on validation:** The methods may disagree on the count. **Disagreement is informative — it tells you the count depends on the model choice.** If you have ground truth (manually traced nuclei), validate both and pick the winner. If you don't have ground truth, the disagreement itself is a signal to do manual QA before publishing.

This is exactly the validation pattern Notebook 02 (and later, Notebook 13) makes systematic.""")


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------
def section_closing(b):
    b.md("""## Closing reflection

This notebook demonstrated the full 3D segmentation workflow:

1. Load a real 3D volume (cells3d() nuclei channel).
2. Run a naive baseline (2D per-slice) to show why connectivity matters.
3. Run two 3D methods (Cellpose 3D, StarDist 3D) with different inductive biases.
4. Quantify outputs: per-object volume, equivalent diameter, count.
5. Compare methods and discuss trade-offs.

**The big question left unanswered here:** Which model is *correct* for this data? We don't have ground truth, so we can't answer that yet. In a real project, you would:

- Use inter-model agreement as a triage signal (regions all methods agree on are high-confidence).
- Annotate a small subset of slices as ground truth (20–40 nuclei across 3–5 slices is often enough).
- Apply Notebook 02's validation metrics to both models.
- Pick the winner, or fine-tune the loser.

That validation workflow is the core of Notebook 02 (for 2D) and would extend naturally to 3D.

**Where to go next on your own:**

- **Notebook 01** — review the 2D Cellpose workflow and parameter tuning if 3D feels abstract.
- **Notebook 02** (Validation quantification) — the metrics that turn "which model is better?" from a visual judgment into a number. Essential if you're publishing 3D segmentation results.
- **Notebook 09** (Cellpose fine-tuning) — if the 3D pretrained model doesn't fit your data, train your own on a small set of your images.
- **StarDist GitHub** ([stardist.net](https://stardist.net)) — training and advanced usage of StarDist 3D.
- **μSAM for 3D** (Notebook 03b extended) — segment-anything adapted to 3D; useful when models don't exist for your sample type.

The 3D-to-validation pipeline is young; methods and best practices are still evolving. The pattern in this lab — pick a method, measure it, pick a winner — is timeless.""")


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def main():
    b = CellBuilder("nb10")
    section_title(b)
    section_setup(b)
    section_load_volume(b)
    section_2d_baseline(b)
    section_cellpose_3d(b)
    section_stardist_3d(b)
    section_choose_method(b)
    section_stats(b)
    section_compare_methods(b)
    section_viz_3d(b)
    section_discussion(b)
    section_closing(b)
    build_notebook(b.cells, "10_3d_segmentation")


if __name__ == "__main__":
    main()
