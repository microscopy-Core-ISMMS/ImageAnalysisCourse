"""
Build script for Notebook 12 — Deconvolution (DL + classical baseline).

Pairs CARE-deconv (DL) with Richardson-Lucy (classical). Demonstrates:
  1. PSF blur + noise model
  2. Classical Richardson-Lucy as analytical baseline
  3. DL deconvolution (CARE-style) trained on paired data
  4. Side-by-side comparison and hallucination check
  5. When to use each method

Run:
    python build_notebook_12.py
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/2026-workshop/notebooks/12_deconvolution.ipynb)

*Click the badge to open this notebook in Google Colab. For best performance, switch to a GPU runtime: Runtime → Change runtime type → T4 GPU.*""")

    b.md("""# Notebook 12 — Deconvolution (DL + classical baseline)

**Status.** Extension lab — post-workshop self-paced.
**Estimated time.** 30–40 minutes on Colab T4.
**Prerequisites.** Notebook 03a (denoising + hallucination check), Notebook 06 (virtual staining validation pattern).

**Learning goals.**

1. Understand the PSF (point-spread function) blur model and why real microscopy is fundamentally blurry.
2. Apply Richardson-Lucy (RL) deconvolution as a classical analytical baseline. Observe how RL trades off restoration vs noise amplification across iterations.
3. Train a CARE-style DL deconvolution model on paired blurred/clean data. See how DL learns to invert the blur without ground-truth iteration counts.
4. Compare RL and CARE-deconv side-by-side on held-out test images using PSNR/SSIM metrics.
5. Apply the hallucination check from Lab 3a to deconvolution. Show that both methods can introduce features; RL does it predictably (noise creep), CARE does it subtly (learned artifacts).
6. Decide when to use each method: RL when you trust your PSF estimate and want predictable behavior; CARE when you have paired training data and need higher quality output.

> **Why this lab matters.** Microscopy images are blurred by the optics; AI restoration is not optional. The choice is between knowing what classical RL does (noise accumulation with iteration) and hoping CARE learned something sensible. This lab makes the trade-off explicit.

> **A note on the form widgets.** Several cells below use `#@param` comments. In **Google Colab** these render as interactive form widgets. In **other environments** they appear as plain Python comments — edit the values directly and re-run.""")


def section_setup(b):
    b.md("""## Setup

Install core restoration + metrics libraries. CARE training will be brief — intentionally tiny network so Colab T4 completes in ~1 minute.""")

    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

%pip install --quiet csbdeep tensorflow scikit-image matplotlib numpy scipy

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

print("Imports OK.")""")

    b.code("""# GPU detection
try:
    import torch
    print(f"torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("torch not installed (csbdeep will manage it)")

try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed")""")


def section_data(b):
    b.md("""## Generate synthetic blurred + noisy paired data

Real microscopy is blurred by the point-spread function (PSF) and corrupted by Poisson + Gaussian noise.
We'll build a synthetic dataset: clean round-blob objects → apply Gaussian PSF blur → add noise → use as training pairs.""")

    b.code("""rng = np.random.default_rng(42)

def make_clean_image(size=64, n_objects=8):
    \"\"\"Bright circular blobs on dark background.\"\"\"
    img = np.zeros((size, size), dtype=float)
    centers = rng.uniform(12, size - 12, (n_objects, 2))
    radii = rng.uniform(4, 8, n_objects)
    for (cy, cx), r in zip(centers, radii):
        Y, X = np.ogrid[:size, :size]
        img[(Y - cy)**2 + (X - cx)**2 <= r**2] = rng.uniform(0.7, 1.0)
    img = gaussian_filter(img, sigma=0.5)
    return np.clip(img, 0, 1)

def make_psf(size=11, sigma=2.5):
    \"\"\"Gaussian PSF.\"\"\"
    kernel = np.zeros((size, size))
    kernel[size // 2, size // 2] = 1.0
    return gaussian_filter(kernel, sigma=sigma)

def blur_image(img, psf):
    \"\"\"Convolve with PSF.\"\"\"
    from scipy.signal import fftconvolve
    blurred = fftconvolve(img, psf, mode='same')
    return np.clip(blurred, 0, 1)

def add_noise(img, photons=30):
    \"\"\"Poisson + Gaussian noise (low-light fluorescence model).\"\"\"
    scaled = img * photons
    noisy = rng.poisson(scaled).astype(float) / photons
    noisy = noisy + rng.normal(0, 0.03, noisy.shape)
    return np.clip(noisy, 0, 1)

# Create PSF once
psf = make_psf()
psf_norm = psf / psf.sum()

# Generate training pairs: 64 pairs
n_train = 64
X_train_clean = np.array([make_clean_image() for _ in range(n_train)])
X_train_blurred = np.array([blur_image(img, psf_norm) for img in X_train_clean])
X_train_blurred_noisy = np.array([add_noise(img) for img in X_train_blurred])

# Held-out test set: 4 pairs
n_test = 4
X_test_clean = np.array([make_clean_image() for _ in range(n_test)])
X_test_blurred = np.array([blur_image(img, psf_norm) for img in X_test_clean])
X_test_blurred_noisy = np.array([add_noise(img) for img in X_test_blurred])

print(f"Training pairs (clean):         {X_train_clean.shape}")
print(f"Training pairs (blurred):       {X_train_blurred.shape}")
print(f"Training pairs (blurred+noisy): {X_train_blurred_noisy.shape}")
print(f"Held-out test (clean):          {X_test_clean.shape}")
print(f"Held-out test (blurred+noisy):  {X_test_blurred_noisy.shape}")
print(f"PSF shape (normalized):         {psf_norm.shape}, sum={psf_norm.sum():.3f}")""")

    b.md("""**Note on the PSF.** Richardson-Lucy *requires* the PSF. We've assumed a Gaussian with sigma=2.5 (realistic for widefield microscopy with moderate defocus). In real workflows, you'd either calibrate the PSF experimentally or estimate it from the image itself.""")


def section_show_example(b):
    b.md("""## One paired example: clean → blurred → noisy

This is what deconvolution has to deal with: signal buried in blur and noise.""")

    b.code("""idx = 0
clean_ex = X_train_clean[idx]
blurred_ex = X_train_blurred[idx]
noisy_ex = X_train_blurred_noisy[idx]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].imshow(clean_ex, cmap='gray'); axes[0].set_title("Clean (ground truth)")
axes[1].imshow(blurred_ex, cmap='gray'); axes[1].set_title("Blurred (PSF applied)")
axes[2].imshow(noisy_ex, cmap='gray'); axes[2].set_title("Blurred + noisy (input)")
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()

print("The task: given the right image, recover the left image."
      "\\n  - The blur is deterministic (PSF) but severe (σ=2.5 pixels)."
      "\\n  - The noise is stochastic Poisson + Gaussian."
      "\\n  - Both methods below try to invert this. Neither is perfect.")""")


def section_quiz(b):
    b.md("""## Predict before you run

**Question:** Which statement is more true of Richardson-Lucy vs CARE-deconv?

(a) RL is learned from data; CARE uses math.
(b) RL has formal convergence guarantees; CARE is a learned blackbox.
(c) RL amplifies noise with each iteration; CARE never amplifies.
(d) CARE learns from paired data so it won't hallucinate; RL always produces real signal.

**Answer:** (b) is closest. RL converges (you can prove it); CARE fits parameters (you hope it learned well). Neither (c) nor (d) is true — both amplify noise / both hallucinate in different ways.""")


def section_rl(b):
    b.md("""## Method 1 — Richardson-Lucy (classical, iterative)

RL is an *expectation-maximization* algorithm. Intuition: at each iteration, divide by the blurred estimate of the previous estimate, convolve back by the PSF transpose. This converges but the late iterations amplify noise. Your job: find the iteration count that balances restoration vs noise.""")

    b.code("""# @title Richardson-Lucy deconvolution { run: \"auto\" }
num_iter = 30  # @param {type: \"slider\", min: 5, max: 50, step: 5}

def run_richardson_lucy(img, psf, num_iter):
    \"\"\"Run Richardson-Lucy on a single image.\"\"\"
    return richardson_lucy(img, psf, num_iter=num_iter, clip=True)

# Apply RL to all test images
rl_outputs = []
for i in range(n_test):
    img_in = X_test_blurred_noisy[i]
    pred_rl = run_richardson_lucy(img_in, psf_norm, num_iter)
    rl_outputs.append(pred_rl)

# Show one example
idx_show = 0
fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
axes[0].imshow(X_test_blurred_noisy[idx_show], cmap='gray'); axes[0].set_title("Input (blurred+noisy)")
axes[1].imshow(rl_outputs[idx_show], cmap='gray'); axes[1].set_title(f"RL output ({num_iter} iter)")
axes[2].imshow(X_test_clean[idx_show], cmap='gray'); axes[2].set_title("Ground truth")
diff_rl = rl_outputs[idx_show] - X_test_clean[idx_show]
axes[3].imshow(diff_rl, cmap='RdBu_r', vmin=-0.3, vmax=0.3); axes[3].set_title("RL diff (pred − gt)")
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()

# Metrics
psnr_rl = psnr_metric(X_test_clean[idx_show], rl_outputs[idx_show], data_range=1.0)
ssim_rl = ssim_metric(X_test_clean[idx_show], rl_outputs[idx_show], data_range=1.0)
print(f"Richardson-Lucy ({num_iter} iterations):")
print(f"  PSNR: {psnr_rl:.2f} dB")
print(f"  SSIM: {ssim_rl:.3f}")
print(f"\\nNote: Try different iteration counts. More iterations → sharper but noisier."
      "\\nLess iterations → smoother but blurrier. The sweet spot depends on your SNR.")""")


def section_care(b):
    b.md("""## Method 2 — CARE-style DL deconvolution (learned)

CARE (Content-Aware Image Restoration) trains a U-Net on paired blurred/clean images. Instead of iterations, the network learns a direct blur-inversion function. We'll use a tiny 2-layer U-Net so the lab finishes in ~1 minute. The pattern scales to full 3D medical imaging.""")

    b.code("""# Try csbdeep.models.CARE; fall back to simple PyTorch U-Net if it fails
try:
    from csbdeep.models import CARE, Config
    print("csbdeep CARE available")
    use_csbdeep_care = True
except ImportError as e:
    print(f"csbdeep CARE import failed ({e}), using fallback U-Net")
    use_csbdeep_care = False
    import torch
    import torch.nn as nn

if use_csbdeep_care:
    # CARE training hyperparameters belong in Config, not model.train().
    # The csbdeep API accepts train_epochs / train_steps_per_epoch / train_batch_size
    # as Config kwargs. model.train() takes only the data + optional validation_data.
    config = Config(
        axes='YXC',
        n_channel_in=1,
        n_channel_out=1,
        unet_n_depth=2,
        unet_kern_size=3,
        train_epochs=3,
        train_steps_per_epoch=16,
        train_batch_size=4,
    )
    model = CARE(config, name='care_deconv')
    X_train = X_train_blurred_noisy[:, :, :, None]
    Y_train = X_train_clean[:, :, :, None]
    X_test = X_test_blurred_noisy[:, :, :, None]
    Y_test_gt = X_test_clean[:, :, :, None]
    print(f"Training CARE with {X_train.shape[0]} pairs")
    model.train(X_train, Y_train, validation_data=(X_test, Y_test_gt))
    # CARE.predict() takes one image at a time and requires the `axes` string
    # describing that single image's dimensions. Loop and stack.
    care_outputs = np.stack([
        model.predict(X_test[i], axes='YXC').squeeze(-1)
        for i in range(X_test.shape[0])
    ])
    print("CARE training complete.")
else:
    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
            )
        def forward(self, x):
            enc = self.enc(x)
            return self.dec(enc)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = TinyUNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X_train_blurred_noisy[:, None], dtype=torch.float32).to(device)
    Y_t = torch.tensor(X_train_clean[:, None], dtype=torch.float32).to(device)
    print("Training CARE-style U-Net fallback...")
    for ep in range(3):
        idx = torch.randperm(X_t.shape[0])
        ep_loss = 0.0
        for k in range(0, X_t.shape[0], 4):
            b_idx = idx[k:k+4]
            pred = net(X_t[b_idx])
            loss = loss_fn(pred, Y_t[b_idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        print(f"  Epoch {ep+1}: loss {ep_loss / (k//4 + 1):.4f}")
    net.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_blurred_noisy[:, None], dtype=torch.float32).to(device)
        Y_pred = net(X_test_t).cpu().numpy().squeeze(1)
    care_outputs = Y_pred
    print("U-Net training complete (fallback).")""")

    b.md("""**What you should be seeing.** CARE training completes in ~1 minute on Colab T4. The loss should decrease. If csbdeep fails to load, the fallback U-Net gives a comparable visual result.""")


def section_compare(b):
    b.md("""## Side-by-side comparison: RL vs CARE vs ground truth

Both methods on the same held-out image. PSNR and SSIM tell you pixel-level fidelity; visual inspection tells you about artifacts.""")

    b.code("""# @title Pick a held-out test image { run: \"auto\" }
test_idx = 0  # @param {type: \"slider\", min: 0, max: 3, step: 1}

inp = X_test_blurred_noisy[test_idx]
gt = X_test_clean[test_idx]
pred_rl = rl_outputs[test_idx]
pred_care = care_outputs[test_idx]

psnr_input = psnr_metric(gt, inp, data_range=1.0)
psnr_rl_val = psnr_metric(gt, pred_rl, data_range=1.0)
psnr_care_val = psnr_metric(gt, pred_care, data_range=1.0)

ssim_input = ssim_metric(gt, inp, data_range=1.0)
ssim_rl_val = ssim_metric(gt, pred_rl, data_range=1.0)
ssim_care_val = ssim_metric(gt, pred_care, data_range=1.0)

print(f"Test image {test_idx}:")
print(f"  Input (noisy)    : PSNR {psnr_input:.2f} dB,  SSIM {ssim_input:.3f}")
print(f"  Richardson-Lucy  : PSNR {psnr_rl_val:.2f} dB,  SSIM {ssim_rl_val:.3f}")
print(f"  CARE-deconv      : PSNR {psnr_care_val:.2f} dB,  SSIM {ssim_care_val:.3f}")

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
axes[0].imshow(inp, cmap='gray'); axes[0].set_title("Input (blurred+noisy)"); axes[0].axis('off')
axes[1].imshow(pred_rl, cmap='gray'); axes[1].set_title(f"RL (PSNR {psnr_rl_val:.1f})"); axes[1].axis('off')
axes[2].imshow(pred_care, cmap='gray'); axes[2].set_title(f"CARE (PSNR {psnr_care_val:.1f})"); axes[2].axis('off')
axes[3].imshow(gt, cmap='gray'); axes[3].set_title("Ground truth"); axes[3].axis('off')
plt.tight_layout(); plt.show()""")


def section_hallucination(b):
    b.md("""## Hallucination check (the key validation)

Both RL and CARE can introduce features that aren't in the ground truth. RL does it predictably (noise creep with iteration); CARE does it subtly (learned artifacts). Difference maps + anomaly thresholds reveal where each method went wrong.

**This is identical to the pattern from Lab 3a (denoising).** The principle generalizes across all inverse problems.""")

    b.code("""# Difference maps
diff_rl = pred_rl - gt
diff_care = pred_care - gt

# Threshold for anomalies
thresh = 0.15
anomaly_rl = np.abs(diff_rl) > thresh
anomaly_care = np.abs(diff_care) > thresh

print(f"Anomaly pixels (|diff| > {thresh}) on test image {test_idx}:")
print(f"  RL   : {anomaly_rl.sum():>5} / {anomaly_rl.size} ({100*anomaly_rl.mean():.1f}%)")
print(f"  CARE : {anomaly_care.sum():>5} / {anomaly_care.size} ({100*anomaly_care.mean():.1f}%)")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# RL row
axes[0, 0].imshow(pred_rl, cmap='gray'); axes[0, 0].set_title("RL prediction"); axes[0, 0].axis('off')
axes[0, 1].imshow(diff_rl, cmap='RdBu_r', vmin=-0.3, vmax=0.3); axes[0, 1].set_title("RL diff (pred − gt)"); axes[0, 1].axis('off')
axes[0, 2].imshow(pred_rl, cmap='gray')
axes[0, 2].imshow(np.where(anomaly_rl, 1, np.nan), cmap='autumn', alpha=0.6)
axes[0, 2].set_title(f"RL anomalies (|diff|>{thresh})"); axes[0, 2].axis('off')

# CARE row
axes[1, 0].imshow(pred_care, cmap='gray'); axes[1, 0].set_title("CARE prediction"); axes[1, 0].axis('off')
axes[1, 1].imshow(diff_care, cmap='RdBu_r', vmin=-0.3, vmax=0.3); axes[1, 1].set_title("CARE diff (pred − gt)"); axes[1, 1].axis('off')
axes[1, 2].imshow(pred_care, cmap='gray')
axes[1, 2].imshow(np.where(anomaly_care, 1, np.nan), cmap='autumn', alpha=0.6)
axes[1, 2].set_title(f"CARE anomalies (|diff|>{thresh})"); axes[1, 2].axis('off')

plt.tight_layout(); plt.show()

print("\\nInterpretation:")
print("  - RL: Anomalies clustered in noisy regions; structured noise amplification.")
print("  - CARE: Anomalies may be subtler or sharper depending on what the network learned.")
print("  - Higher anomaly % does NOT mean the method is worse — it means structural changes.")""")


def section_quantify(b):
    b.md("""## Quantitative summary across all held-out images

Tabulate PSNR, SSIM, and anomaly fraction for both methods across all 4 held-out test images.""")

    b.code("""import pandas as pd

rows = []
for i in range(n_test):
    inp_i = X_test_blurred_noisy[i]
    gt_i = X_test_clean[i]

    for name, pred_arr in [("RL", rl_outputs), ("CARE", care_outputs)]:
        pred_i = pred_arr[i]
        psnr_i = psnr_metric(gt_i, pred_i, data_range=1.0)
        ssim_i = ssim_metric(gt_i, pred_i, data_range=1.0)
        anom_pct = 100 * (np.abs(pred_i - gt_i) > thresh).mean()
        rows.append({
            "image": i,
            "method": name,
            "PSNR": round(psnr_i, 2),
            "SSIM": round(ssim_i, 3),
            "anomaly_%": round(anom_pct, 1),
        })

df = pd.DataFrame(rows)
print("Held-out test set metrics:")
print(df.to_string(index=False))
print()
print("Summary by method:")
summary = df.groupby("method")[["PSNR", "SSIM", "anomaly_%"]].mean().round(2)
print(summary.to_string())""")

    b.md("""**Key observations:**

- **PSNR/SSIM** reward outputs close to the reference. High values look good but don't guarantee biological correctness.
- **anomaly_%** is the fraction of pixels that disagree with ground truth beyond the threshold. A higher percentage means more structural change, which may be good (removing noise) or bad (adding artifact).
- **Trade-off:** RL is predictable but may be noisier at low iterations or too smooth at high ones. CARE is often sharper but may learn unintended patterns.

The "best" method depends on your downstream analysis. If you're counting objects, anomalies matter more than PSNR. If you're measuring intensity, PSNR matters more.""")


def section_method_selection(b):
    b.md("""## When to use each method

**Richardson-Lucy:**
- ✓ You have a good estimate of the PSF (experimental or from literature).
- ✓ You want predictable, mathematically-guaranteed behavior.
- ✓ You're okay tuning iteration count empirically (may require trial).
- ✓ Your images have moderate SNR (RL amplifies noise, so clean input helps).
- ✗ No paired training data needed, but you must know the PSF exactly.

**CARE-deconv:**
- ✓ You have 50+ paired blurred/clean image pairs (or can generate them synthetically).
- ✓ You want a single inference pass with no hyperparameter tuning.
- ✓ Your images have low SNR (CARE learns to denoise + deconvolve jointly).
- ✗ Requires GPU training (even on Colab T4, ~1-5 minutes for real-scale nets).
- ✗ May hallucinate features if training data is unrepresentative.

**Practical workflow:**
1. Start with RL if you have a PSF estimate. It's fast (no training) and interpretable.
2. If RL looks noisy or requires excessive tuning, consider CARE.
3. Always validate with a hallucination check (this lab's difference-map approach) before publishing.""")


def section_closing(b):
    b.md("""## Closing reflection

Deconvolution — whether classical or learned — is not optional in microscopy. The choice is between understanding what you're doing (RL) and hoping the learned model is sensible (CARE).

This lab demonstrated:

1. **The PSF blur model** — why real microscopy is fundamentally blurry.
2. **Richardson-Lucy** — an analytical solution with known trade-offs (noise vs blur).
3. **CARE-style DL** — a learned alternative that can achieve higher quality at the cost of explainability.
4. **Side-by-side comparison** using PSNR/SSIM.
5. **The hallucination check** — the same validation pattern as Lab 3a, applied to deconvolution.
6. **Method selection** — decision tree for choosing RL vs CARE for your data.

**Relationship to other labs:**
- **Notebook 03a** (denoising + hallucination) — same validation pattern applied to noise removal.
- **Notebook 06** (virtual staining) — hallucination check applied to cross-channel prediction.
- **Notebook 07** (super-resolution) — another inverse problem; similar classical vs learned choice.

**Where to go next:**
- The [scikit-image deconvolution docs](https://scikit-image.org/docs/stable/api/skimage.restoration.html) for RL variants (non-negative, total-variation).
- The [CARE paper](https://www.nature.com/articles/s41592-018-0216-7) (Weigert et al., *Nature Methods* 2018) and the [csbdeep documentation](https://csbdeep.bioimagecomputing.com/) for full-scale DL training.
- [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) for ready-made CARE Colabs on real microscopy data.
- Notebook 07 (super-resolution) — the next level of inverse problems in computational microscopy.""")


def main():
    b = CellBuilder("nb12")
    section_title(b)
    section_setup(b)
    section_data(b)
    section_show_example(b)
    section_quiz(b)
    section_rl(b)
    section_care(b)
    section_compare(b)
    section_hallucination(b)
    section_quantify(b)
    section_method_selection(b)
    section_closing(b)
    build_notebook(b.cells, "12_deconvolution")


if __name__ == "__main__":
    main()
