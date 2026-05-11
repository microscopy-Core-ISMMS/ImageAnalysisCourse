"""
Build script for Notebook 14 — Spot Detection (FISH / single-molecule / particles).

The pedagogical arc: spot detection is its own task class → classical Laplacian-of-Gaussian
works well in clean low-density data → DL heatmap-CNN approach scales to harder cases →
choose-your-own dropdown lets users pick.

Run:
    python build_notebook_14.py
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/notebooks/14_spot_detection.ipynb)

*Click the badge to open this notebook in Google Colab. For best performance, switch to a GPU runtime: Runtime → Change runtime type → T4 GPU.*""")

    b.md("""# Notebook 14 — Spot Detection (FISH / single-molecule / particles)

**Status.** Extension lab — post-workshop self-paced. Recommended after Notebooks 01, 04.
**Estimated time.** 8–12 minutes on Colab T4.

**Learning goals.**

1. Understand spot detection as a task distinct from segmentation: localize individual point sources, not region boundaries.
2. Run classical Laplacian-of-Gaussian (LoG) on synthetic FISH-like data and measure precision/recall.
3. Train a small heatmap-based CNN detector on the same data and compare performance.
4. Use density stress-testing to reveal where each method fails.
5. Recognize trade-offs: LoG excels without labeled data; DL heatmap-CNN scales to crowded scenes.

> **Why this lab matters.** Single-molecule localization (FISH, STORM, confocal spots, vesicles) drives quantitative biology — spot count, localization precision, and clustering statistics. Getting the spots right is the foundation. Classical LoG is fast and works well in sparse data; when density increases or signal conditions degrade, a trained DL detector often outperforms hard-coded thresholds.

> **A note on the form widgets.** Several cells below use `#@param` comments. In **Google Colab** these render as interactive form widgets (sliders, dropdowns) at the top of the cell. In **other environments** they appear as plain Python comments — edit the values directly and re-run the cell.""")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def section_setup(b):
    b.md("""## Setup

Install the required libraries.""")

    b.code("""%pip install --quiet torch torchvision scikit-image matplotlib numpy scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage.feature import blob_log
import torch
import torch.nn as nn

print("Imports OK.")""")

    b.code("""# GPU detection — informational only; each library handles fallback automatically
try:
    import torch
    print("torch          :", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device         :", torch.cuda.get_device_name(0))
except ImportError:
    print("torch not installed")""")


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
def section_data_gen(b):
    b.md("""## Synthetic FISH-like spot data generator

We generate synthetic microscopy images with randomly placed point sources (spots) at known centers. This gives us ground truth for validation. We'll vary: number of spots, density, noise level, and optional PSF blur.

The generator produces ~64 training images and ~16 held-out test images.""")

    b.code("""def generate_spot_data(n_images=64, image_size=128, seed=0,
                         n_spots_range=(8, 16), noise_std=0.08, psf_sigma=1.0):
    \"\"\"Generate synthetic FISH-like spot images with ground-truth centers.

    Returns:
        images: (n_images, H, W) float32 array, range [0, 1]
        centers: list of n_images lists of (y, x) center tuples
    \"\"\"
    rng = np.random.default_rng(seed)
    images = []
    centers_list = []

    for img_idx in range(n_images):
        img = np.zeros((image_size, image_size), dtype=np.float32)
        centers = []

        # Random number of spots in this image
        n_spots = rng.integers(n_spots_range[0], n_spots_range[1] + 1)

        for _ in range(n_spots):
            # Random center location
            cy = rng.uniform(8, image_size - 8)
            cx = rng.uniform(8, image_size - 8)
            centers.append((cy, cx))

            # Add a Gaussian spot at this location
            Y, X = np.mgrid[:image_size, :image_size].astype(np.float32)
            spot = 0.8 * np.exp(-((Y - cy)**2 + (X - cx)**2) / (2 * psf_sigma**2))
            img += spot

        # Add noise
        img = img + rng.normal(0, noise_std, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1)

        # Optional PSF blur (makes spots slightly wider)
        if psf_sigma > 0:
            img = gaussian_filter(img, sigma=0.5)

        images.append(img)
        centers_list.append(centers)

    return np.array(images), centers_list


# Generate training and test data
print("Generating synthetic FISH-like data...")
train_images, train_centers = generate_spot_data(n_images=64, seed=0)
test_images, test_centers = generate_spot_data(n_images=16, seed=1)

print(f"Training: {train_images.shape}  ({len(train_centers)} images)")
print(f"Test:     {test_images.shape}  ({len(test_centers)} images)")""")


# ---------------------------------------------------------------------------
# Show one example
# ---------------------------------------------------------------------------
def section_show_example(b):
    b.md("""## Show one example

2-panel view: image + ground-truth spots as red markers overlaid.""")

    b.code("""idx = 0  # Show first training image

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: raw image
axes[0].imshow(train_images[idx], cmap='gray')
axes[0].set_title(f"Image (spots visible as bright peaks)")
axes[0].axis('off')

# Right: image with ground-truth centers marked
axes[1].imshow(train_images[idx], cmap='gray')
for cy, cx in train_centers[idx]:
    axes[1].plot(cx, cy, 'r+', markersize=12, markeredgewidth=1.5)
axes[1].set_title(f"Ground truth: {len(train_centers[idx])} spots")
axes[1].axis('off')

plt.tight_layout(); plt.show()""")


# ---------------------------------------------------------------------------
# Predict-before-run quiz
# ---------------------------------------------------------------------------
def section_quiz(b):
    b.md("""## Predict-before-run quiz

Before we run any detector, make predictions:

**Q1: Which method should work better on sparse, well-separated spots (e.g., isolated FISH spots in clean background)?**
- (a) Laplacian-of-Gaussian (LoG) — classical, no training data needed
- (b) DL heatmap CNN — requires training but learns data-specific noise patterns
- (c) They should perform equally

**Q2: What happens when spots start overlapping (density increases)?**
- (a) LoG: precision drops because Gaussians blur together. CNN: still okay if trained on dense data.
- (b) CNN: drops; LoG: stays the same.
- (c) Both fail equally.

**Q3: What happens if you train a CNN on sparse data and then test on dense data?**
- (a) CNN generalizes fine — networks are good at that.
- (b) CNN fails outside its training distribution. Precision/recall drop sharply.
- (c) LoG wins in both cases anyway.

Expected answers: **(a) for Q1, (a) for Q2, (b) for Q3.** Let's test these hypotheses.""")


# ---------------------------------------------------------------------------
# Method 1: LoG classical
# ---------------------------------------------------------------------------
def section_log_method(b):
    b.md("""## Method 1 — Laplacian-of-Gaussian (classical)

scikit-image provides `blob_log`: efficient, parameter-free localization for isolated spots. Returns (y, x, sigma) for each detected blob.

Key tuning parameters:
- `min_sigma` / `max_sigma`: expected blob width (in pixels)
- `threshold`: detection threshold (higher = fewer, more confident detections)

We'll run on test images and compute precision/recall by matching predicted spots to ground truth (within a 3-pixel tolerance).""")

    b.code("""def match_spots(predicted, ground_truth, tolerance_px=3.0):
    \"\"\"Greedy nearest-neighbor matching of predicted spots to ground-truth centers.

    Each predicted spot matches at most one ground-truth spot.
    Returns: (true_positives, false_positives, false_negatives, precision, recall)
    \"\"\"
    matched_gt = set()
    matched_pred = set()

    for p_idx, (py, px) in enumerate(predicted):
        best_dist = np.inf
        best_gt_idx = None
        for g_idx, (gy, gx) in enumerate(ground_truth):
            if g_idx in matched_gt:
                continue
            dist = np.sqrt((py - gy)**2 + (px - gx)**2)
            if dist < best_dist:
                best_dist, best_gt_idx = dist, g_idx
        if best_dist <= tolerance_px and best_gt_idx is not None:
            matched_gt.add(best_gt_idx)
            matched_pred.add(p_idx)

    tp = len(matched_pred)
    fp = len(predicted) - tp
    fn = len(ground_truth) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return tp, fp, fn, precision, recall


def run_log_detector(images, centers, min_sigma=1.5, max_sigma=4.0, threshold=0.05):
    \"\"\"Run LoG on a batch of test images. Return detections + metrics.\"\"\"
    detections = []
    metrics = []

    for img, gt_centers in zip(images, centers):
        # blob_log returns (row, col, sigma)
        blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma,
                         threshold=threshold, overlap=0.5)

        # Extract (y, x) from the first two columns
        predicted = [(b[0], b[1]) for b in blobs]

        # Match and compute metrics
        tp, fp, fn, prec, rec = match_spots(predicted, gt_centers)

        detections.append(predicted)
        metrics.append({'tp': tp, 'fp': fp, 'fn': fn, 'precision': prec, 'recall': rec})

    return detections, metrics


# Run LoG detector with interactive parameters
# @param controls these values when run in Colab
min_sigma = 1.5  # @param {type: "slider", min: 0.5, max: 3.0, step: 0.2}
max_sigma = 4.0  # @param {type: "slider", min: 2.0, max: 6.0, step: 0.5}
threshold = 0.05  # @param {type: "slider", min: 0.01, max: 0.20, step: 0.01}

print(f"Running LoG detector on {len(test_images)} test images...")
print(f"  min_sigma={min_sigma}, max_sigma={max_sigma}, threshold={threshold}")

log_dets, log_metrics = run_log_detector(test_images, test_centers,
                                          min_sigma, max_sigma, threshold)

# Aggregate metrics
log_all_tp = sum(m['tp'] for m in log_metrics)
log_all_fp = sum(m['fp'] for m in log_metrics)
log_all_fn = sum(m['fn'] for m in log_metrics)
log_all_prec = log_all_tp / (log_all_tp + log_all_fp) if (log_all_tp + log_all_fp) > 0 else 0.0
log_all_rec = log_all_tp / (log_all_tp + log_all_fn) if (log_all_tp + log_all_fn) > 0 else 0.0

print(f"\\nResults:")
print(f"  TP={log_all_tp}, FP={log_all_fp}, FN={log_all_fn}")
print(f"  Precision={log_all_prec:.3f}, Recall={log_all_rec:.3f}")""")


# ---------------------------------------------------------------------------
# Method 2: DL heatmap detector
# ---------------------------------------------------------------------------
def section_dl_method(b):
    b.md("""## Method 2 — DL heatmap detector (CNN)

We train a small CNN that maps images → heatmaps of spot centers. This approach:
- Learns the noise statistics from the training set.
- Scales naturally to dense spots (as long as they're in the training distribution).
- Requires labeled training data (ground-truth spot centers).

Architecture: 4-layer CNN with ReLU activations. Input: (1, H, W) grayscale image. Output: (1, H, W) heatmap with values near 1.0 at spot centers.""")

    b.code("""class SpotDetector(nn.Module):
    \"\"\"Small CNN for spot-detection heatmap prediction.\"\"\"
    def __init__(self, base=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 5, padding=2), nn.ReLU(),
            nn.Conv2d(base, base, 5, padding=2), nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def make_heatmap_target(centers, image_size=128, sigma=1.5):
    \"\"\"Create a Gaussian heatmap with peaks at spot centers.\"\"\"
    heatmap = np.zeros((image_size, image_size), dtype=np.float32)
    for cy, cx in centers:
        Y, X = np.mgrid[:image_size, :image_size].astype(np.float32)
        heatmap += np.exp(-((Y - cy)**2 + (X - cx)**2) / (2 * sigma**2))
    return np.clip(heatmap, 0, 1)


def train_dl_detector(images, centers, epochs=60, batch_size=16, lr=1e-3):
    \"\"\"Train the CNN detector on synthetic FISH data.\"\"\"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare data
    X = torch.tensor(images[:, None]).to(device).float()  # (N, 1, H, W)
    heatmaps = np.array([make_heatmap_target(c) for c in centers])
    Y = torch.tensor(heatmaps[:, None]).to(device).float()

    net = SpotDetector().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    n_samples = X.shape[0]

    print(f"Training on {n_samples} images, {epochs} epochs, batch_size={batch_size}")
    for epoch in range(epochs):
        idx = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_steps = 0

        for step in range(0, n_samples, batch_size):
            batch_idx = idx[step:step+batch_size]
            pred_heatmap = net(X[batch_idx])
            loss = loss_fn(pred_heatmap, Y[batch_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

        avg_loss = epoch_loss / n_steps
        losses.append(avg_loss)

        if (epoch + 1) % 15 == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss {avg_loss:.4f}")

    return net, losses


print("Training CNN detector...")
device = "cuda" if torch.cuda.is_available() else "cpu"
cnn_net, cnn_losses = train_dl_detector(train_images, train_centers,
                                         epochs=60, batch_size=16)
print(f"Training complete on {device}.")""")

    b.code("""def detect_from_heatmap(heatmap, threshold=0.3, min_distance=3):
    \"\"\"Extract spot coordinates from predicted heatmap via local maxima.\"\"\"
    # Find local maxima
    peak_mask = (heatmap == maximum_filter(heatmap, size=2*min_distance+1))
    peak_mask = peak_mask & (heatmap > threshold)

    # Extract coordinates
    coords = np.argwhere(peak_mask)
    return [(c[0], c[1]) for c in coords]


def run_dl_detector(images, centers, net, threshold=0.3):
    \"\"\"Run CNN detector on a batch of test images.\"\"\"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detections = []
    metrics = []

    net.eval()
    with torch.no_grad():
        for img, gt_centers in zip(images, centers):
            # Forward pass
            img_t = torch.tensor(img[None, None], dtype=torch.float32).to(device)
            pred_heatmap = net(img_t).squeeze().cpu().numpy()

            # Extract spots from heatmap
            predicted = detect_from_heatmap(pred_heatmap, threshold=threshold)

            # Match and compute metrics
            tp, fp, fn, prec, rec = match_spots(predicted, gt_centers)

            detections.append(predicted)
            metrics.append({'tp': tp, 'fp': fp, 'fn': fn, 'precision': prec, 'recall': rec})

    return detections, metrics


print(f"Running CNN detector on {len(test_images)} test images...")
cnn_dets, cnn_metrics = run_dl_detector(test_images, test_centers, cnn_net, threshold=0.3)

# Aggregate metrics
cnn_all_tp = sum(m['tp'] for m in cnn_metrics)
cnn_all_fp = sum(m['fp'] for m in cnn_metrics)
cnn_all_fn = sum(m['fn'] for m in cnn_metrics)
cnn_all_prec = cnn_all_tp / (cnn_all_tp + cnn_all_fp) if (cnn_all_tp + cnn_all_fp) > 0 else 0.0
cnn_all_rec = cnn_all_tp / (cnn_all_tp + cnn_all_fn) if (cnn_all_tp + cnn_all_fn) > 0 else 0.0

print(f"\\nResults:")
print(f"  TP={cnn_all_tp}, FP={cnn_all_fp}, FN={cnn_all_fn}")
print(f"  Precision={cnn_all_prec:.3f}, Recall={cnn_all_rec:.3f}")""")


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------
def section_comparison(b):
    b.md("""## Side-by-side comparison

For a single held-out test image, visualize: raw image + ground truth + LoG detections + CNN detections. Show precision/recall metrics.""")

    b.code("""# @title Pick a test image to compare { run: "auto" }
test_idx = 0  # @param {type: "slider", min: 0, max: 15, step: 1}

if test_idx >= len(test_images):
    test_idx = 0

img = test_images[test_idx]
gt = test_centers[test_idx]
log_pred = log_dets[test_idx]
cnn_pred = cnn_dets[test_idx]

# Compute metrics for this image
log_tp, log_fp, log_fn, log_p, log_r = match_spots(log_pred, gt)
cnn_tp, cnn_fp, cnn_fn, cnn_p, cnn_r = match_spots(cnn_pred, gt)

fig, axes = plt.subplots(2, 2, figsize=(12, 11))

# Row 0: image + ground truth
axes[0, 0].imshow(img, cmap='gray')
for cy, cx in gt:
    axes[0, 0].plot(cx, cy, 'r+', markersize=10, markeredgewidth=1.5)
axes[0, 0].set_title(f"Ground truth ({len(gt)} spots)")
axes[0, 0].axis('off')

# Row 0: image + LoG
axes[0, 1].imshow(img, cmap='gray')
for cy, cx in log_pred:
    axes[0, 1].plot(cx, cy, 'g^', markersize=8, fillstyle='none', markeredgewidth=1.5)
axes[0, 1].set_title(f"LoG detected ({len(log_pred)} spots)\\nP={log_p:.2f} R={log_r:.2f}")
axes[0, 1].axis('off')

# Row 1: image + CNN
axes[1, 0].imshow(img, cmap='gray')
for cy, cx in cnn_pred:
    axes[1, 0].plot(cx, cy, 'bs', markersize=8, fillstyle='none', markeredgewidth=1.5)
axes[1, 0].set_title(f"CNN detected ({len(cnn_pred)} spots)\\nP={cnn_p:.2f} R={cnn_r:.2f}")
axes[1, 0].axis('off')

# Row 1: metrics table
axes[1, 1].axis('off')
table_data = [
    ['Metric', 'LoG', 'CNN'],
    ['TP', str(log_tp), str(cnn_tp)],
    ['FP', str(log_fp), str(cnn_fp)],
    ['FN', str(log_fn), str(cnn_fn)],
    ['Precision', f'{log_p:.3f}', f'{cnn_p:.3f}'],
    ['Recall', f'{log_r:.3f}', f'{cnn_r:.3f}'],
]
table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
axes[1, 1].set_title("Comparison metrics", fontsize=11, fontweight='bold')

plt.tight_layout(); plt.show()

print(f"\\n** What you should be seeing **")
print(f"On sparse, well-separated spots: LoG and CNN should both perform well.")
print(f"On this specific image, the method with higher recall caught more true spots;")
print(f"the method with higher precision made fewer false detections.")""")


# ---------------------------------------------------------------------------
# Choose-your-own
# ---------------------------------------------------------------------------
def section_choose_own(b):
    b.md("""## Choose-your-own: pick method and test image

Run both methods on any held-out test image and see side-by-side results. Adjust parameters and re-run.""")

    b.code("""# @title Pick method and test image { run: "auto" }
method = "LoG classical"  # @param ["LoG classical", "DL heatmap CNN", "both side-by-side"]
test_image_idx = 5  # @param {type: "slider", min: 0, max: 15, step: 1}

if test_image_idx >= len(test_images):
    test_image_idx = 0

img = test_images[test_image_idx]
gt = test_centers[test_image_idx]

if method in ["LoG classical", "both side-by-side"]:
    log_pred = log_dets[test_image_idx]
    log_tp, log_fp, log_fn, log_p, log_r = match_spots(log_pred, gt)

if method in ["DL heatmap CNN", "both side-by-side"]:
    cnn_pred = cnn_dets[test_image_idx]
    cnn_tp, cnn_fp, cnn_fn, cnn_p, cnn_r = match_spots(cnn_pred, gt)

if method == "LoG classical":
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(img, cmap='gray')
    for cy, cx in gt:
        axes[0].plot(cx, cy, 'r+', markersize=10, markeredgewidth=1.5)
    axes[0].set_title(f"Ground truth ({len(gt)} spots)")
    axes[0].axis('off')

    axes[1].imshow(img, cmap='gray')
    for cy, cx in log_pred:
        axes[1].plot(cx, cy, 'g^', markersize=8, fillstyle='none', markeredgewidth=1.5)
    axes[1].set_title(f"LoG (P={log_p:.2f}, R={log_r:.2f}, {len(log_pred)} detected)")
    axes[1].axis('off')

elif method == "DL heatmap CNN":
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(img, cmap='gray')
    for cy, cx in gt:
        axes[0].plot(cx, cy, 'r+', markersize=10, markeredgewidth=1.5)
    axes[0].set_title(f"Ground truth ({len(gt)} spots)")
    axes[0].axis('off')

    axes[1].imshow(img, cmap='gray')
    for cy, cx in cnn_pred:
        axes[1].plot(cx, cy, 'bs', markersize=8, fillstyle='none', markeredgewidth=1.5)
    axes[1].set_title(f"CNN (P={cnn_p:.2f}, R={cnn_r:.2f}, {len(cnn_pred)} detected)")
    axes[1].axis('off')

else:  # both side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(img, cmap='gray')
    for cy, cx in gt:
        axes[0].plot(cx, cy, 'r+', markersize=10, markeredgewidth=1.5)
    axes[0].set_title(f"Ground truth ({len(gt)} spots)")
    axes[0].axis('off')

    axes[1].imshow(img, cmap='gray')
    for cy, cx in log_pred:
        axes[1].plot(cx, cy, 'g^', markersize=8, fillstyle='none', markeredgewidth=1.5)
    axes[1].set_title(f"LoG (P={log_p:.2f}, R={log_r:.2f})")
    axes[1].axis('off')

    axes[2].imshow(img, cmap='gray')
    for cy, cx in cnn_pred:
        axes[2].plot(cx, cy, 'bs', markersize=8, fillstyle='none', markeredgewidth=1.5)
    axes[2].set_title(f"CNN (P={cnn_p:.2f}, R={cnn_r:.2f})")
    axes[2].axis('off')

plt.tight_layout(); plt.show()""")


# ---------------------------------------------------------------------------
# Density stress test
# ---------------------------------------------------------------------------
def section_density_test(b):
    b.md("""## Density stress test

Generate synthetic data at progressively higher spot densities (sparse → dense) and measure how precision/recall degrade for each method. This reveals the breaking points.""")

    b.code("""print("Generating synthetic data at varying densities...")

# Density levels: n_spots ranges from 4 (sparse) to 32 (very dense)
density_levels = [4, 8, 12, 16, 20, 24, 28, 32]
log_precisions = []
log_recalls = []
cnn_precisions = []
cnn_recalls = []

for n_spots in density_levels:
    # Generate test images with fixed density
    images_dens, centers_dens = generate_spot_data(
        n_images=8, image_size=128, seed=100+n_spots,
        n_spots_range=(n_spots, n_spots), noise_std=0.08
    )

    # Run both detectors
    log_dets_dens, log_mets_dens = run_log_detector(images_dens, centers_dens,
                                                     min_sigma=1.5, max_sigma=4.0,
                                                     threshold=0.05)
    cnn_dets_dens, cnn_mets_dens = run_dl_detector(images_dens, centers_dens, cnn_net)

    # Aggregate
    log_prec = sum(m['tp'] for m in log_mets_dens) / max(1, sum(m['tp'] + m['fp'] for m in log_mets_dens))
    log_rec = sum(m['tp'] for m in log_mets_dens) / max(1, sum(m['tp'] + m['fn'] for m in log_mets_dens))
    cnn_prec = sum(m['tp'] for m in cnn_mets_dens) / max(1, sum(m['tp'] + m['fp'] for m in cnn_mets_dens))
    cnn_rec = sum(m['tp'] for m in cnn_mets_dens) / max(1, sum(m['tp'] + m['fn'] for m in cnn_mets_dens))

    log_precisions.append(log_prec)
    log_recalls.append(log_rec)
    cnn_precisions.append(cnn_prec)
    cnn_recalls.append(cnn_rec)

    print(f"n_spots={n_spots:2d}: LoG P={log_prec:.3f} R={log_rec:.3f} | CNN P={cnn_prec:.3f} R={cnn_rec:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(density_levels, log_precisions, 'g-o', label='LoG', linewidth=2, markersize=6)
axes[0].plot(density_levels, cnn_precisions, 'b-s', label='CNN', linewidth=2, markersize=6)
axes[0].set_xlabel('Spot density (n_spots per image)', fontsize=11)
axes[0].set_ylabel('Precision', fontsize=11)
axes[0].set_ylim([0, 1.05])
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)
axes[0].set_title('Precision vs spot density')

axes[1].plot(density_levels, log_recalls, 'g-o', label='LoG', linewidth=2, markersize=6)
axes[1].plot(density_levels, cnn_recalls, 'b-s', label='CNN', linewidth=2, markersize=6)
axes[1].set_xlabel('Spot density (n_spots per image)', fontsize=11)
axes[1].set_ylabel('Recall', fontsize=11)
axes[1].set_ylim([0, 1.05])
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)
axes[1].set_title('Recall vs spot density')

plt.tight_layout(); plt.show()

print("\\n** What you should be seeing **")
print("At low density: both methods perform well.")
print("At high density: LoG precision drops (spots blur together);")
print("                 CNN holds up better IF density is in training distribution.")
print("If you train CNN on sparse data and test on dense: CNN fails outside its distribution.")""")


# ---------------------------------------------------------------------------
# TrackPy baseline pointer
# ---------------------------------------------------------------------------
def section_trackpy_pointer(b):
    b.md("""## TrackPy baseline pointer

For production single-molecule work (FISH spots, particle tracking), `trackpy.locate` is the gold standard. It implements the Crocker-Grier algorithm with subpixel refinement and sophisticated peak detection.

**Why not use it here?** TrackPy has extra dependencies and is optimized for *very* sparse data (the case where LoG excels). For this notebook, we kept imports minimal.

**When to switch to TrackPy:**
- Your spots are well-isolated and subpixel accuracy matters.
- You have >1000 frames and need speed.
- You need published peer-reviewed algorithms.

[TrackPy documentation](https://soft-matter.github.io/trackpy/) — start here for production work.""")


# ---------------------------------------------------------------------------
# Closing reflection
# ---------------------------------------------------------------------------
def section_closing(b):
    b.md("""## Closing reflection

**Key takeaways:**

1. **Spot detection is distinct from segmentation.** It localizes point sources, not region boundaries. Classical LoG works well in sparse, clean data.

2. **LoG vs CNN trade-off:**
   - LoG: no labeled data needed, works instantly, but struggles when spots overlap.
   - CNN: needs labeled training data, but generalizes to harder conditions (dense, noisy).

3. **Density matters.** Both methods fail outside their assumed operating point. LoG fails when spots touch; CNN fails when test density exceeds training density.

4. **Choose deliberately:**
   - Sparse FISH with few spots? LoG.
   - Crowded confocal slices or high-density particles? CNN (after labeling a training set).
   - Production single-molecule work? TrackPy.

**Where to go next:**

- **[Notebook 04 §3e](04_community_platforms.ipynb#3e.-inline-demo)** has a working YOLO-style bounding-box detector. For bounding boxes instead of point sources, that's the pattern.
- **[deepBlink](https://github.com/BBQuercus/deepBlink)** — production DL FISH spot detector. Start here if you have real FISH data and want a pre-trained model.
- **[TrackPy](https://soft-matter.github.io/trackpy/)** — the classical Crocker-Grier particle tracking library. Use for isolated spots with subpixel accuracy requirements.
- **[Cellpose](https://cellpose.readthedocs.io/)** — when your "spots" are actually small cells (e.g., bacteria), segmentation is a better framing than detection.

**The broader lesson:** task framing matters. Don't ask "which method is best?" Ask "what is the task?" Is it localization (detect centers), segmentation (detect regions), or tracking (detect + link)? Different tasks, different methods.""")


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def main():
    b = CellBuilder("nb14")
    section_title(b)
    section_setup(b)
    section_data_gen(b)
    section_show_example(b)
    section_quiz(b)
    section_log_method(b)
    section_dl_method(b)
    section_comparison(b)
    section_choose_own(b)
    section_density_test(b)
    section_trackpy_pointer(b)
    section_closing(b)
    build_notebook(b.cells, "14_spot_detection")


if __name__ == "__main__":
    main()
