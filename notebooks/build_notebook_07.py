"""
Build script for Notebook 07 — AI Super-Resolution from Widefield (Multi-Method, Choose-Your-Own).

The central pedagogy: take a low-res/blurry widefield input and learn a model that produces
a higher-resolution-looking output. Multiple methods (bicubic baseline, CARE-style U-Net,
DFCAN-style attention SR) are taught via a choose-your-own dropdown. The hallucination check
surfaces the central risk: SR models invent fine structure that wasn't in the input.

Run:
    python build_notebook_07.py
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/2026-workshop/notebooks/07_widefield_superres.ipynb)

*Click the badge to open this notebook in Google Colab. For best performance, switch to a GPU runtime: Runtime → Change runtime type → T4 GPU.*""")

    b.md("""# Notebook 07 — AI Super-Resolution from Widefield (Multi-Method, Choose-Your-Own)

**Status.** Extension lab — post-workshop self-paced.
**Estimated time.** 30–45 minutes on Colab T4 (3–5 min for all method training + evaluation).
**Prerequisites.** Notebook 03a (denoising + hallucination check), Notebook 06 (virtual staining validation pattern), Notebook 12 (deconvolution restoration framework).

**Learning goals.**

1. Understand super-resolution (SR) as a restoration task: take low-resolution (LR) input and learn a model that outputs higher-resolution-like images.
2. Compare multiple SR methods (bicubic baseline, CARE-style U-Net, DFCAN-style attention) on the same test set.
3. Recognize that SR inherently invents fine structure; distinguish between recovered signal and hallucinated features.
4. Apply the **hallucination check** from Lab 3a to SR outputs to quantify the invention rate per method.
5. Use cross-validation and held-out tests to decide when SR genuinely recovers information vs. when it hides noise.

> **Why this lab matters.** SR is powerful but dangerous. A model that outputs a "beautiful" high-frequency image might be inventing cellular features that were never in the input. This lab teaches you to validate SR outputs and report them honestly. The choose-your-own dropdown lets you explore how different architectures and losses trade off fidelity vs. hallucination.

> **A note on the form widgets.** Several cells below use `#@param` comments. In **Google Colab** these render as interactive form widgets (sliders, dropdowns). In **other environments** they appear as plain Python comments — edit the values directly and re-run.""")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def section_setup(b):
    b.md("""## Setup

Install core deep-learning + metrics libraries. CARE training is brief (3–5 epochs, ~30s on T4); DFCAN variant is comparable.""")

    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

%pip install --quiet csbdeep tensorflow torch torchvision scikit-image matplotlib numpy scipy

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

print("Imports OK.")
print(f"NumPy: {np.__version__}")""")

    b.code("""# GPU detection
try:
    import torch
    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("torch not installed; CSBDeep will manage it")

try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed")""")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def section_data(b):
    b.md("""## Synthetic widefield/SR paired-data generator

We build a synthetic high-resolution (HR) dataset of round cells with sharp boundaries, then degrade each to a low-resolution (LR) widefield version:
1. Generate HR at 256×256 with bright cells.
2. Apply Gaussian blur (simulating widefield optics).
3. Downsample 2× to 128×128 (LR).
4. Add Poisson + Gaussian noise (low-light model).
5. Upsample LR back to 256×256 for training (the model learns to denoise + sharpen).

This is realistic: widefield microscopy degrades resolution by a factor of 2–3, and SR networks learn to invert that within the limits of information content.""")

    b.code("""rng = np.random.default_rng(42)

def make_hr_ground_truth(size=256, n_cells=16, seed=None):
    \"\"\"High-resolution ground truth: round bright cells on dark background.\"\"\"
    if seed is not None:
        rng_local = np.random.default_rng(seed)
    else:
        rng_local = rng
    img = np.zeros((size, size), dtype=float)
    centers = rng_local.uniform(30, size - 30, (n_cells, 2))
    radii = rng_local.uniform(12, 22, n_cells)
    for (cy, cx), r in zip(centers, radii):
        Y, X = np.ogrid[:size, :size]
        mask = (Y - cy)**2 + (X - cx)**2 <= r**2
        img[mask] = rng_local.uniform(0.8, 1.0)
    # Apply very slight smoothing to simulate real cell edges
    img = gaussian_filter(img, sigma=0.5)
    return np.clip(img, 0, 1)

def degrade_to_widefield(hr_img, blur_sigma=1.5, downsample=2, noise_photons=20):
    \"\"\"Degrade HR to LR widefield with noise.\"\"\"
    # Blur (widefield PSF)
    blurred = gaussian_filter(hr_img.astype(float), sigma=blur_sigma)
    # Downsample
    lr = zoom(blurred, 1.0 / downsample, order=1)
    # Add Poisson + Gaussian noise (low-light)
    scaled = lr * noise_photons
    noisy_lr = rng.poisson(scaled).astype(float) / noise_photons
    noisy_lr = noisy_lr + rng.normal(0, 0.04, noisy_lr.shape)
    return np.clip(noisy_lr, 0, 1)

# Generate 64 training pairs
n_train = 64
lr_size = 128
hr_size = 256

print("Generating synthetic paired data...")
hr_train = np.array([make_hr_ground_truth(size=hr_size, n_cells=16, seed=i) for i in range(n_train)])
lr_train_small = np.array([degrade_to_widefield(img, blur_sigma=1.5, downsample=2, noise_photons=20) for img in hr_train])
# Upsample LR back to HR size for training (models will learn to denoise + sharpen)
lr_train = np.array([zoom(img, 2.0, order=1) for img in lr_train_small])

# Held-out test set: 8 images
n_test = 8
hr_test = np.array([make_hr_ground_truth(size=hr_size, n_cells=16, seed=1000 + i) for i in range(n_test)])
lr_test_small = np.array([degrade_to_widefield(img, blur_sigma=1.5, downsample=2, noise_photons=20) for img in hr_test])
lr_test = np.array([zoom(img, 2.0, order=1) for img in lr_test_small])

print(f"Training set:  HR {hr_train.shape}, LR {lr_train.shape}")
print(f"Test set:      HR {hr_test.shape}, LR {lr_test.shape}")""")

    b.md("""**What the degradation does:** The Gaussian blur + downsample + noise is a realistic model of widefield microscopy. By upsampling LR back to HR resolution before training, the model has a clear task: denoise and sharpen to match HR. In practice, the network learns to recover some real structure and invents some plausible high-frequency detail. Our hallucination check will measure how much is invention.""")


# ---------------------------------------------------------------------------
# Show one example
# ---------------------------------------------------------------------------
def section_show_example(b):
    b.md("""## One paired example: HR ground truth, LR widefield input, bicubic baseline

Three panels to anchor intuition:""")

    b.code("""# Show the first training pair with bicubic upsampling baseline
idx = 0
hr = hr_train[idx]
lr = lr_train[idx]
bicubic_up = zoom(lr_train_small[idx], 2.0, order=3)  # bicubic order=3

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(hr, cmap='viridis'); axes[0].set_title("HR ground truth (256×256)"); axes[0].axis('off')
axes[1].imshow(lr, cmap='viridis'); axes[1].set_title("LR widefield input (256×256, upsampled)"); axes[1].axis('off')
axes[2].imshow(bicubic_up, cmap='viridis'); axes[2].set_title("Bicubic baseline (order=3)"); axes[2].axis('off')
plt.tight_layout(); plt.show()

print("The bicubic baseline is the simplest non-AI method: upsampling with spline interpolation.")
print("It can never invent — only smooth. DL methods will be sharper; the question is whether")
print("that sharpness is recovery or hallucination.")""")


# ---------------------------------------------------------------------------
# Predict-before-run quiz
# ---------------------------------------------------------------------------
def section_predict_quiz(b):
    b.md("""## Predict-before-run: What will the methods recover?

Look at the LR widefield image above. The HR ground truth has sharp round cell boundaries. Some of that sharpness is recoverable from the blurry LR input (the cell location is still visible); some is genuinely lost (the exact edge shape required a higher-resolution input).

**Prediction quiz** (pick one and think it through before running the methods below):

**(a)** All three methods will produce nearly identical outputs — they'll all converge on the same recovered HR.

**(b)** Bicubic will be smooth; CARE-style U-Net will be sharper; DFCAN-style will be the sharpest. All three will match the ground truth about equally well on metrics.

**(c)** Bicubic will match ground truth best because it's conservative. CARE and DFCAN will invent plausible high-frequency detail that looks good but doesn't match the truth.

**(d)** CARE will match ground truth best because U-Nets are good at restoration. DFCAN will overfit and invent artifacts.

**Why this matters:** The "correct" answer depends on what information the LR input actually contains. If the LR is blurry enough, the ground truth is partially lost forever — and SR models will invent something sensible but wrong. That's the hallucination risk.""")


# ---------------------------------------------------------------------------
# Method 1: Bicubic baseline
# ---------------------------------------------------------------------------
def section_bicubic(b):
    b.md("""## Method 1 — Bicubic upsampling (the non-AI baseline)

Bicubic interpolation (order=3 spline) is the simplest non-AI method. It cannot invent features — it can only smooth and interpolate. This makes it a useful sanity check: if a DL method doesn't beat bicubic on evaluation metrics, something is wrong with the DL method.""")

    b.code("""def bicubic_sr(lr_img):
    \"\"\"Bicubic upsampling as baseline.\"\"\"
    return zoom(lr_img, 1.0, order=3)  # Already at target size; just recompute for consistency

# Apply to held-out test set
pred_bicubic = np.array([zoom(lr_test_small[i], 2.0, order=3) for i in range(n_test)])

# Compute PSNR and SSIM against ground truth
psnr_bicubic = np.array([psnr_metric(hr_test[i], pred_bicubic[i], data_range=1.0) for i in range(n_test)])
ssim_bicubic = np.array([ssim_metric(hr_test[i], pred_bicubic[i], data_range=1.0) for i in range(n_test)])

print("Bicubic upsampling (2× on LR):")
print(f"  Mean PSNR: {psnr_bicubic.mean():.2f} dB")
print(f"  Mean SSIM: {ssim_bicubic.mean():.3f}")
print(f"  Per-image PSNR: {psnr_bicubic}")
print(f"  Per-image SSIM: {ssim_bicubic}")""")

    b.md("""**What you're seeing.** Bicubic produces smooth, blurry output. It typically scores low on PSNR/SSIM because it lacks the sharp features in the ground truth. But it never invents structure — every pixel is a weighted average of the input. That conservative behavior makes it an excellent baseline to compare DL methods against.""")


# ---------------------------------------------------------------------------
# Method 2: CARE-style U-Net
# ---------------------------------------------------------------------------
def section_care(b):
    b.md("""## Method 2 — CARE-style deep-learning super-resolution

CARE (Weigert et al., *Nature Methods* 2018) uses a small U-Net trained with L2 loss (MSE) on paired LR/HR images. This is the most common DL approach to restoration: minimize per-pixel squared error. The network learns to denoise + sharpen, but because MSE loss doesn't penalize hallucinated high-frequency detail as long as it's "close," the output can invent plausible structure.

We train a tiny U-Net to keep the notebook fast (3–5 epochs, ~30 seconds on Colab T4).""")

    b.code("""import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class TinyUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u = self.up(e2)
        return self.dec(torch.cat([u, e1], dim=1))


def train_care_unet(X_lr, Y_hr, epochs=5, batch=8, lr=1e-3):
    \"\"\"Train CARE-style U-Net on paired LR/HR.\"\"\"
    net = TinyUNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xhat = torch.tensor(X_lr[:, None], dtype=torch.float32).to(device)  # (N, 1, H, W)
    Yhat = torch.tensor(Y_hr[:, None], dtype=torch.float32).to(device)
    losses = []
    n = X_lr.shape[0]
    print("Training CARE-style U-Net (MSE loss)...")
    for ep in range(epochs):
        idx = torch.randperm(n)
        ep_loss = 0.0
        n_steps = 0
        for k in range(0, n, batch):
            b_idx = idx[k:k+batch]
            pred = net(Xhat[b_idx])
            loss = loss_fn(pred, Yhat[b_idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_steps += 1
        losses.append(ep_loss / n_steps)
        print(f"  epoch {ep+1:2d}/{epochs}  loss {losses[-1]:.4f}")
    return net, losses

t0 = time.time()
net_care, losses_care = train_care_unet(lr_train, hr_train, epochs=5)
print(f"Done in {time.time() - t0:.1f}s.")""")

    b.code("""# Inference on held-out test set
net_care.eval()
with torch.no_grad():
    Xhat_test = torch.tensor(lr_test[:, None], dtype=torch.float32).to(device)
    pred_care = net_care(Xhat_test).cpu().numpy().squeeze(1)

# Compute metrics
psnr_care = np.array([psnr_metric(hr_test[i], pred_care[i], data_range=1.0) for i in range(n_test)])
ssim_care = np.array([ssim_metric(hr_test[i], pred_care[i], data_range=1.0) for i in range(n_test)])

print("CARE-style U-Net super-resolution:")
print(f"  Mean PSNR: {psnr_care.mean():.2f} dB")
print(f"  Mean SSIM: {ssim_care.mean():.3f}")
print(f"  Per-image PSNR: {psnr_care}")
print(f"  Per-image SSIM: {ssim_care}")""")

    b.md("""**What you should be seeing.** CARE typically beats bicubic on PSNR/SSIM because it learns to sharpen and denoise. But the hallucination check (coming next) will show that some of that improvement comes from inventing high-frequency detail.""")


# ---------------------------------------------------------------------------
# Method 3: DFCAN-style with Squeeze-and-Excitation attention
# ---------------------------------------------------------------------------
def section_dfcan(b):
    b.md("""## Method 3 — DFCAN-style attention super-resolution (simplified)

DFCAN (Qiao et al., 2021) augments SR networks with **Squeeze-and-Excitation (SE) attention blocks**. The idea: learn to recalibrate feature-map channels based on their importance for the reconstruction. This helps the network focus restoration effort where it matters.

We implement a *simplified DFCAN-style architecture*: a small U-Net with one SE attention block in the bottleneck. This is a teaching approximation, not the production DFCAN model (which has more elaborate attention patterns). The point is to show how attention changes the SR behavior.

**Note:** The production DFCAN from `qc17-thu/DL-SR` (linked in the closing) is more sophisticated. We teach the architectural idea here inline.""")

    b.code("""class SqueezeExcitation(nn.Module):
    \"\"\"Squeeze-and-Excitation attention block.

    Learns to recalibrate feature channels: reweight based on global context.
    \"\"\"
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels),
            nn.Sigmoid())

    def forward(self, x):
        # x shape: (B, C, H, W)
        # Squeeze: (B, C, H, W) -> (B, C)
        squeeze = x.mean(dim=(2, 3))
        # Excitation: (B, C) -> (B, C)
        excitation = self.fc(squeeze)
        # Reweight: (B, C) -> (B, C, 1, 1), then broadcast
        return x * excitation.view(x.size(0), -1, 1, 1)


class DFCANStyleUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        # SE attention in the bottleneck
        self.se = SqueezeExcitation(base*2)
        self.up = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e2_att = self.se(e2)  # Apply SE attention
        u = self.up(e2_att)
        return self.dec(torch.cat([u, e1], dim=1))


def train_dfcan_style(X_lr, Y_hr, epochs=5, batch=8, lr=1e-3):
    \"\"\"Train DFCAN-style U-Net with SE attention.\"\"\"
    net = DFCANStyleUNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xhat = torch.tensor(X_lr[:, None], dtype=torch.float32).to(device)
    Yhat = torch.tensor(Y_hr[:, None], dtype=torch.float32).to(device)
    losses = []
    n = X_lr.shape[0]
    print("Training DFCAN-style U-Net with SE attention (MSE loss)...")
    for ep in range(epochs):
        idx = torch.randperm(n)
        ep_loss = 0.0
        n_steps = 0
        for k in range(0, n, batch):
            b_idx = idx[k:k+batch]
            pred = net(Xhat[b_idx])
            loss = loss_fn(pred, Yhat[b_idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_steps += 1
        losses.append(ep_loss / n_steps)
        print(f"  epoch {ep+1:2d}/{epochs}  loss {losses[-1]:.4f}")
    return net, losses

t0 = time.time()
net_dfcan, losses_dfcan = train_dfcan_style(lr_train, hr_train, epochs=5)
print(f"Done in {time.time() - t0:.1f}s.")""")

    b.code("""# Inference on held-out test set
net_dfcan.eval()
with torch.no_grad():
    Xhat_test = torch.tensor(lr_test[:, None], dtype=torch.float32).to(device)
    pred_dfcan = net_dfcan(Xhat_test).cpu().numpy().squeeze(1)

# Compute metrics
psnr_dfcan = np.array([psnr_metric(hr_test[i], pred_dfcan[i], data_range=1.0) for i in range(n_test)])
ssim_dfcan = np.array([ssim_metric(hr_test[i], pred_dfcan[i], data_range=1.0) for i in range(n_test)])

print("DFCAN-style U-Net with SE attention:")
print(f"  Mean PSNR: {psnr_dfcan.mean():.2f} dB")
print(f"  Mean SSIM: {ssim_dfcan.mean():.3f}")
print(f"  Per-image PSNR: {psnr_dfcan}")
print(f"  Per-image SSIM: {ssim_dfcan}")

print()
print("Comparison summary:")
print(f"  Bicubic mean PSNR : {psnr_bicubic.mean():.2f} dB")
print(f"  CARE mean PSNR    : {psnr_care.mean():.2f} dB")
print(f"  DFCAN mean PSNR   : {psnr_dfcan.mean():.2f} dB")""")

    b.md("""**What you should be seeing.** Attention mechanisms (SE blocks) often improve PSNR/SSIM slightly by focusing feature-refinement effort. However, the real test is the hallucination check — does the attention help with *recovery* or with *plausible invention*?""")


# ---------------------------------------------------------------------------
# Choose-your-own dropdown for side-by-side
# ---------------------------------------------------------------------------
def section_dropdown(b):
    b.md("""## Choose-your-own: pick a method and test image

Use the dropdown to select a method (or "all side-by-side") and a held-out test image, then run the cell. The visualization shows: HR ground truth, LR input, and the selected method's output, with PSNR/SSIM reported.""")

    b.code("""# @title Pick method and test image { run: "auto" }
method = "all side-by-side"  # @param ["bicubic baseline", "CARE-style U-Net", "DFCAN-style with attention", "all side-by-side"]
test_idx = 0  # @param {type: "slider", min: 0, max: 7, step: 1}

if test_idx >= n_test:
    test_idx = 0

hr_ref = hr_test[test_idx]
lr_input = lr_test[test_idx]

if method == "all side-by-side":
    preds = {
        "HR ground truth": hr_ref,
        "LR input": lr_input,
        "Bicubic": pred_bicubic[test_idx],
        "CARE U-Net": pred_care[test_idx],
        "DFCAN+SE": pred_dfcan[test_idx],
    }
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 5))
    for ax, (name, img) in zip(axes, preds.items()):
        ax.imshow(img, cmap='viridis')
        ax.set_title(name)
        ax.axis('off')
    plt.tight_layout(); plt.show()

    print("Side-by-side comparison for test image", test_idx)
    print()
    print(f"{'Method':<20} {'PSNR (dB)':<15} {'SSIM':<10}")
    print("-" * 45)
    print(f"{'Bicubic':<20} {psnr_bicubic[test_idx]:<15.2f} {ssim_bicubic[test_idx]:<10.3f}")
    print(f"{'CARE U-Net':<20} {psnr_care[test_idx]:<15.2f} {ssim_care[test_idx]:<10.3f}")
    print(f"{'DFCAN+SE':<20} {psnr_dfcan[test_idx]:<15.2f} {ssim_dfcan[test_idx]:<10.3f}")

else:
    if method == "bicubic baseline":
        pred = pred_bicubic[test_idx]
        psnr_val = psnr_bicubic[test_idx]
        ssim_val = ssim_bicubic[test_idx]
    elif method == "CARE-style U-Net":
        pred = pred_care[test_idx]
        psnr_val = psnr_care[test_idx]
        ssim_val = ssim_care[test_idx]
    else:  # DFCAN
        pred = pred_dfcan[test_idx]
        psnr_val = psnr_dfcan[test_idx]
        ssim_val = ssim_dfcan[test_idx]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(lr_input, cmap='viridis'); axes[0].set_title("LR input"); axes[0].axis('off')
    axes[1].imshow(pred, cmap='viridis'); axes[1].set_title(f"{method}\\nPSNR {psnr_val:.1f} dB"); axes[1].axis('off')
    axes[2].imshow(hr_ref, cmap='viridis'); axes[2].set_title("HR ground truth"); axes[2].axis('off')
    plt.tight_layout(); plt.show()

    print(f"Test image {test_idx} — {method}")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.3f}")""")


# ---------------------------------------------------------------------------
# Hallucination check
# ---------------------------------------------------------------------------
def section_hallucination(b):
    b.md("""## The hallucination check (the core validation for SR)

Super-resolution is more dangerous than denoising. Denoising can only smooth; SR can invent. A difference map between the SR output and ground truth reveals where invention happens.

We compute:
1. **Difference map** (pred − ground truth) to see signed errors.
2. **Anomaly threshold** |diff| > 0.15 to find structural disagreements.
3. **Anomaly percentage** across each method to quantify the invention rate.

The warning: a method with high PSNR but high anomaly_pct is producing plausible hallucinations, not recovery.""")

    b.code("""thresh_anom = 0.15  # Threshold for structural disagreement (in normalized [0,1] units)

# Compute difference maps and anomalies for all methods
diff_bicubic = pred_bicubic - hr_test
anomaly_bicubic = np.abs(diff_bicubic) > thresh_anom

diff_care = pred_care - hr_test
anomaly_care = np.abs(diff_care) > thresh_anom

diff_dfcan = pred_dfcan - hr_test
anomaly_dfcan = np.abs(diff_dfcan) > thresh_anom

print("Anomaly pixels (structural disagreements, |diff| > {:.2f}):".format(thresh_anom))
print(f"  Bicubic    : {anomaly_bicubic.sum():>8} px / {anomaly_bicubic.size:>8}  ({100*anomaly_bicubic.mean():.1f}%)")
print(f"  CARE U-Net : {anomaly_care.sum():>8} px / {anomaly_care.size:>8}  ({100*anomaly_care.mean():.1f}%)")
print(f"  DFCAN+SE   : {anomaly_dfcan.sum():>8} px / {anomaly_dfcan.size:>8}  ({100*anomaly_dfcan.mean():.1f}%)")""")

    b.code("""# Visualize hallucination for one test image
idx_viz = 2  # Middle test image
fig, axes = plt.subplots(3, 3, figsize=(14, 12))

# Row 1: Bicubic
axes[0, 0].imshow(pred_bicubic[idx_viz], cmap='viridis'); axes[0, 0].set_title("Bicubic prediction"); axes[0, 0].axis('off')
axes[0, 1].imshow(diff_bicubic[idx_viz], cmap='RdBu_r', vmin=-0.5, vmax=0.5); axes[0, 1].set_title("Bicubic diff (pred - gt)"); axes[0, 1].axis('off')
axes[0, 2].imshow(pred_bicubic[idx_viz], cmap='gray')
axes[0, 2].imshow(np.where(anomaly_bicubic[idx_viz], 1, np.nan), cmap='autumn', alpha=0.6)
axes[0, 2].set_title(f"Bicubic anomalies ({100*anomaly_bicubic[idx_viz].mean():.1f}%)"); axes[0, 2].axis('off')

# Row 2: CARE
axes[1, 0].imshow(pred_care[idx_viz], cmap='viridis'); axes[1, 0].set_title("CARE prediction"); axes[1, 0].axis('off')
axes[1, 1].imshow(diff_care[idx_viz], cmap='RdBu_r', vmin=-0.5, vmax=0.5); axes[1, 1].set_title("CARE diff (pred - gt)"); axes[1, 1].axis('off')
axes[1, 2].imshow(pred_care[idx_viz], cmap='gray')
axes[1, 2].imshow(np.where(anomaly_care[idx_viz], 1, np.nan), cmap='autumn', alpha=0.6)
axes[1, 2].set_title(f"CARE anomalies ({100*anomaly_care[idx_viz].mean():.1f}%)"); axes[1, 2].axis('off')

# Row 3: DFCAN
axes[2, 0].imshow(pred_dfcan[idx_viz], cmap='viridis'); axes[2, 0].set_title("DFCAN prediction"); axes[2, 0].axis('off')
axes[2, 1].imshow(diff_dfcan[idx_viz], cmap='RdBu_r', vmin=-0.5, vmax=0.5); axes[2, 1].set_title("DFCAN diff (pred - gt)"); axes[2, 1].axis('off')
axes[2, 2].imshow(pred_dfcan[idx_viz], cmap='gray')
axes[2, 2].imshow(np.where(anomaly_dfcan[idx_viz], 1, np.nan), cmap='autumn', alpha=0.6)
axes[2, 2].set_title(f"DFCAN anomalies ({100*anomaly_dfcan[idx_viz].mean():.1f}%)"); axes[2, 2].axis('off')

plt.tight_layout(); plt.show()

print(f"Hallucination visualization for test image {idx_viz}:")
print(f"  Bicubic anomaly: {100*anomaly_bicubic[idx_viz].mean():.1f}%")
print(f"  CARE anomaly:    {100*anomaly_care[idx_viz].mean():.1f}%")
print(f"  DFCAN anomaly:   {100*anomaly_dfcan[idx_viz].mean():.1f}%")""")

    b.md("""**What you should be seeing:**
- **Bicubic** has low anomaly % because it can only smooth — it can't invent structure, just blur what's there.
- **CARE and DFCAN** typically have higher anomaly % because they learn to invent plausible high-frequency detail. Whether that invention is "good" (recovering real structure) or "bad" (hallucinating) depends on how much information the LR actually contains.

The orange-colored anomaly regions show where the prediction disagreed with ground truth. If these regions are at cell boundaries (where some information was lost in the LR), that's expected invention. If they're in the cell interior (where the LR is smooth and featureless), the invention is less justified.""")


# ---------------------------------------------------------------------------
# Per-image metrics table
# ---------------------------------------------------------------------------
def section_metrics_table(b):
    b.md("""## Per-method anomaly fraction and metrics across the test set

Tabulate PSNR, SSIM, and anomaly_pct for all three methods across all 8 held-out images. This is the honest quality signal: a method that scores high on PSNR but high on anomaly_pct is producing plausible hallucinations.""")

    b.code("""import pandas as pd

rows = []
for i in range(n_test):
    for name, psnr_arr, ssim_arr, anom_arr in [
        ("bicubic", psnr_bicubic, ssim_bicubic, anomaly_bicubic),
        ("CARE", psnr_care, ssim_care, anomaly_care),
        ("DFCAN", psnr_dfcan, ssim_dfcan, anomaly_dfcan),
    ]:
        rows.append({
            "test_idx": i,
            "method": name,
            "PSNR_dB": round(psnr_arr[i], 2),
            "SSIM": round(ssim_arr[i], 3),
            "anomaly_%": round(100 * anom_arr[i].mean(), 1),
        })

df = pd.DataFrame(rows)
print("Per-image metrics on held-out test set (8 images × 3 methods):")
print()
print(df.to_string(index=False))
print()
print("Summary statistics:")
print(df.groupby("method")[["PSNR_dB", "SSIM", "anomaly_%"]].mean().round(2).to_string())""")

    b.md("""**Reading the table:**
- **PSNR/SSIM** reward outputs that match ground truth pixel-by-pixel. Higher is "better," but only if you trust the reference.
- **anomaly_%** is the fraction of pixels where the prediction diverges from ground truth by more than the threshold. This measures "structural wrongness" — places where the model invented vs. recovered.

A well-balanced method has good PSNR/SSIM but not-too-high anomaly_pct. **A method with high PSNR and high anomaly_pct is a red flag — it's producing visually plausible but biologically incorrect output, exactly the failure mode that gets papers retracted.**""")


# ---------------------------------------------------------------------------
# When SR helps vs. hides noise
# ---------------------------------------------------------------------------
def section_when_sr_helps(b):
    b.md("""## When SR genuinely helps vs. when it hides noise

SR is a powerful tool, but it has fundamental limits.

**SR genuinely helps when:**
- The LR input contains enough information to resolve the structure (the structure is *encoded* in the LR, just at lower resolution).
- Example: A widefield microscopy image with cells that are 50 pixels in diameter; upsampling by 2× is plausible because the cell interior is still resolvable.

**SR hides noise when:**
- The LR is degraded by noise more than by blur. An SR network "learns" to smooth the noise by inventing a plausible-looking structure.
- Example: If the input is dominated by photon noise, SR models will invent cellular structure that approximates the noise statistics.

**The honest test:** Cross-validate on a held-out experimental condition (e.g., a new sample or light level) where you know the answer. If the SR model's predictions on the new condition don't match reality, you've caught hallucination.

In this notebook, we can't run a true cross-validation because we only have synthetic data. In practice: **always validate on real data that you didn't use to train the model.**""")


# ---------------------------------------------------------------------------
# Closing reflection
# ---------------------------------------------------------------------------
def section_closing(b):
    b.md("""## Closing reflection

This lab demonstrated:

1. **Three SR methods** spanning classical (bicubic), DL standard (CARE/U-Net), and DL modern (DFCAN with attention).
2. **Paired synthetic data** generated from a realistic degradation model (blur + downsample + noise).
3. **Side-by-side evaluation** using a choose-your-own dropdown for interactive exploration.
4. **The hallucination check** — computing difference maps and anomaly fractions to quantify where SR invents vs. recovers.
5. **Honest metrics** — PSNR/SSIM alone are insufficient; anomaly_pct measures structural wrongness.

**Where to go next:**

- **Notebook 03a** — denoising and the hallucination check; the foundational validation pattern.
- **Notebook 06** — virtual staining; another generation task that needs hallucination validation.
- **Notebook 12** — deconvolution; a related restoration problem using classical (Richardson-Lucy) and DL (CARE) approaches.
- **Notebook 08** — SRRF (Super-Resolution Radial Fluctuations), a completely different SR paradigm using temporal stacks instead of training.
- **Cellpose 3** and related — joint restoration + segmentation; if your SR output is going to be segmented, train those together.

**Production SR codes and models:**

- **CSBDeep/CARE** — the production U-Net restoration framework: https://github.com/CSBDeep/CSBDeep
- **DFCAN (Qiao et al., 2021)** — the full production code with elaborate attention: https://github.com/qc17-thu/DL-SR
- **3D-RCAN (Aivia)** — 3D super-resolution for volumetric data: https://github.com/AiviaCommunity/3D-RCAN
- **BioImage Model Zoo** — registry of pretrained SR models: https://bioimage.io

**The philosophical point:** Super-resolution is powerful because it leverages learned priors. But those priors are learned from your training data — if your training data has a bias, your SR will amplify it. **Always validate SR outputs on held-out real data, and always report the hallucination check metrics (anomaly fraction) in your methods.** That's the barrier to papers that make claims about images no microscope ever measured.""")


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def main():
    b = CellBuilder("nb07")
    section_title(b)
    section_setup(b)
    section_data(b)
    section_show_example(b)
    section_predict_quiz(b)
    section_bicubic(b)
    section_care(b)
    section_dfcan(b)
    section_dropdown(b)
    section_hallucination(b)
    section_metrics_table(b)
    section_when_sr_helps(b)
    section_closing(b)
    build_notebook(b.cells, "07_widefield_superres")


if __name__ == "__main__":
    main()
