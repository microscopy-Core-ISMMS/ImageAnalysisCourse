"""
Build script for Notebook 06 — Virtual Staining and Label-Free Prediction.

Goes deeper than Notebook 04's fnet/pix2pix mini-workflows by using *real*
microscopy data (skimage cells3d() DAPI -> membrane cross-channel prediction)
and dedicating the back half of the lab to the hallucination check and
integrity reporting.

Run:
    python build_notebook_06.py
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/notebooks/06_virtual_staining.ipynb)

*Click the badge to open this notebook in Google Colab. For best performance, switch to a GPU runtime: Runtime → Change runtime type → T4 GPU.*""")

    b.md("""# Notebook 06 — Virtual Staining and Label-Free Prediction

**Status.** Extension lab — post-workshop self-paced.
**Estimated time.** 25–40 minutes on Colab T4 (longer on CPU).
**Prerequisites.** Notebook 03a (denoising / hallucination), Notebook 04 (fnet inline mini-workflow).

**Learning goals.**

1. Train two virtual-staining models — **fnet-style** (paired U-Net) and **pix2pix-style** (paired GAN) — on real cross-channel microscopy data.
2. Compare the two outputs against the ground-truth fluorescence channel.
3. Apply the **hallucination check** from Lab 3a to virtual-staining outputs specifically. Identify regions where the model invented features rather than predicting them from real signal.
4. Decide whether a virtual-staining output is fit for a stated experimental purpose.
5. Write an integrity-reporting paragraph that satisfies contemporary journal image-integrity expectations for AI-generated figures.

> **Why this lab matters.** Virtual staining (also called *in silico labeling* or *label-free prediction*) lets you predict fluorescence-style readouts from images that lack the relevant stain — saving phototoxicity, cost, and acquisition time. The trade-off is that the predicted image is a model output, not a measurement. **Hiding the AI provenance of a stained-looking image violates most journals' image-integrity policies.** This lab makes the trade-off concrete.

> **Relationship to Notebook 04.** Notebook 04 has fnet and pix2pix as standalone mini-workflows on synthetic data. This lab uses *real* microscopy data and goes further on validation, hallucination, and reporting. If you've worked through Notebook 04, the modeling code here will look familiar — the new content is in the validation half.

> **A note on the form widgets.** Several cells below use `#@param` comments. In **Google Colab** these render as interactive form widgets at the top of the cell. In **other environments** they appear as plain Python comments — edit the values directly and re-run.""")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def section_setup(b):
    b.md("""## Setup""")

    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

%pip install --quiet torch torchvision scikit-image matplotlib numpy

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skimage import data as skdata
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch  : {torch.__version__}")
print(f"device : {device}")""")


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------
def section_data(b):
    b.md("""## Real cross-channel data — `cells3d()` DAPI → membrane

scikit-image bundles a 3D confocal stack with two channels: DAPI (nuclei) and a membrane stain. We use this as a virtual-staining test bed: the input is the DAPI channel; the target is the membrane channel.

The challenge is real — DAPI and membrane signals don't perfectly correlate, so the model has to learn anatomy-aware translation. This is the same problem Christiansen et al. (2018, *Cell*) and Ounkomol et al. (2018, *Nature Methods*) tackled at full scale.""")

    b.code("""# Load and slice cells3d() into 2D image pairs
cells = skdata.cells3d()  # shape: (z=60, c=2, y=256, x=256)
print(f"cells3d: {cells.shape}, dtype={cells.dtype}")
print(f"  channel 0 = membrane,  channel 1 = nuclei (DAPI)")

# We'll predict membrane (input) from nuclei (so DAPI -> membrane = nuclei -> membrane)
# Use most slices for training, hold out a few for evaluation
n_z = cells.shape[0]
train_z = list(range(0, 50))
test_z = list(range(50, n_z))   # 10 held-out slices for evaluation

def normalize_slice(arr):
    arr = arr.astype(np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    # np.percentile returns float64 — cast back to float32 so downstream torch tensors stay float32
    out = np.clip((arr - np.float32(p1)) / np.float32(max(p99 - p1, 1e-8)), 0, 1)
    return out.astype(np.float32)

# Channel 1 = DAPI nuclei = INPUT;  Channel 0 = membrane = TARGET
X_train = np.stack([normalize_slice(cells[z, 1]) for z in train_z])
Y_train = np.stack([normalize_slice(cells[z, 0]) for z in train_z])
X_test  = np.stack([normalize_slice(cells[z, 1]) for z in test_z])
Y_test  = np.stack([normalize_slice(cells[z, 0]) for z in test_z])

print(f"\\nTraining pairs: {X_train.shape}")
print(f"Held-out pairs: {X_test.shape}")

# Show the first training pair
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(X_train[0], cmap='Blues_r'); axes[0].set_title("DAPI nuclei (input)"); axes[0].axis('off')
axes[1].imshow(Y_train[0], cmap='magma'); axes[1].set_title("Membrane stain (target)"); axes[1].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Predict before you train.** Look at the input/target pair above. How accurately do you expect a network to predict membrane from DAPI?

- (a) Near-perfect — the network should pick up enough cell-shape signal from nuclei position to draw the membrane.
- (b) Reasonable — gross outlines correct, fine structure off.
- (c) Poor — DAPI doesn't actually have enough information about membrane location.
- (d) Mixed — okay where membrane wraps tightly around nuclei, bad where it doesn't.

The right answer is closest to (d). DAPI tells you *where the cells are* but not *where the membrane is exactly*. This is the central virtual-staining challenge: the input modality usually has correlated but incomplete information about the target modality.""")


# ---------------------------------------------------------------------------
# fnet-style U-Net
# ---------------------------------------------------------------------------
def section_fnet(b):
    b.md("""## Method 1 — fnet-style U-Net (paired, regression)

fnet (Ounkomol 2018) is a 3D U-Net trained with mean-squared-error loss on paired brightfield/fluorescence stacks. We use a small 2D variant here so the lab finishes in minutes; the *pattern* is the same.""")

    b.code("""class TinyUNet(nn.Module):
    def __init__(self, base=24):
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


def train_fnet(X, Y, epochs=8, batch=4, lr=1e-3):
    net = TinyUNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X[:, None], dtype=torch.float32).to(device)  # (N, 1, H, W)
    Yt = torch.tensor(Y[:, None], dtype=torch.float32).to(device)
    losses = []
    n = X.shape[0]
    print("Training fnet-style U-Net...")
    for ep in range(epochs):
        idx = torch.randperm(n)
        ep_loss = 0.0
        n_steps = 0
        for k in range(0, n, batch):
            b_idx = idx[k:k+batch]
            pred = net(Xt[b_idx])
            loss = loss_fn(pred, Yt[b_idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_steps += 1
        losses.append(ep_loss / n_steps)
        print(f"  epoch {ep+1:2d}/{epochs}  loss {losses[-1]:.4f}")
    return net, losses


t0 = time.time()
net_fnet, losses_fnet = train_fnet(X_train, Y_train, epochs=8)
print(f"Done in {time.time() - t0:.1f}s.")""")

    b.code("""# Inference on held-out slices
net_fnet.eval()
with torch.no_grad():
    Xt_test = torch.tensor(X_test[:, None], dtype=torch.float32).to(device)
    Y_pred_fnet = net_fnet(Xt_test).cpu().numpy().squeeze(1)

# Show a few held-out predictions
n_show = min(3, X_test.shape[0])
fig, axes = plt.subplots(n_show, 3, figsize=(11, 3.5 * n_show))
if n_show == 1:
    axes = axes.reshape(1, -1)
for i in range(n_show):
    axes[i, 0].imshow(X_test[i], cmap='Blues_r'); axes[i, 0].set_title("DAPI input")
    axes[i, 1].imshow(Y_pred_fnet[i], cmap='magma'); axes[i, 1].set_title("fnet prediction")
    axes[i, 2].imshow(Y_test[i], cmap='magma'); axes[i, 2].set_title("Ground truth")
    for ax in axes[i]: ax.axis('off')
plt.tight_layout(); plt.show()""")


# ---------------------------------------------------------------------------
# pix2pix-style
# ---------------------------------------------------------------------------
def section_pix2pix(b):
    b.md("""## Method 2 — pix2pix-style conditional GAN (paired, adversarial)

pix2pix (Isola et al. 2017) trains the same paired image-to-image translation but with a discriminator alongside the generator. The adversarial loss pushes the generator toward outputs that *look like* real images of the target modality, not just outputs that minimize per-pixel error. This often produces sharper outputs at the cost of more variance — and more hallucination.""")

    b.code("""class P2PGenerator(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(base*2, base*4, 4, 2, 1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, inplace=True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(base*4, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(base*4, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(base*2, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        d1 = self.dec1(e3)
        d2 = self.dec2(torch.cat([d1, e2], 1))
        return self.dec3(torch.cat([d2, e1], 1))


class P2PDiscriminator(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*4, 1, 4, 1, 1))

    def forward(self, src, tgt):
        return self.net(torch.cat([src, tgt], 1))


def train_pix2pix(X, Y, epochs=8, batch=4, lr=2e-4, lam=100.0):
    G = P2PGenerator().to(device)
    D = P2PDiscriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    Xt = torch.tensor(X[:, None], dtype=torch.float32).to(device)
    Yt = torch.tensor(Y[:, None], dtype=torch.float32).to(device)
    n = X.shape[0]
    print("Training pix2pix-style GAN...")
    for ep in range(epochs):
        idx = torch.randperm(n)
        d_total, g_total = 0.0, 0.0
        n_steps = 0
        for k in range(0, n, batch):
            b_idx = idx[k:k+batch]
            real_src, real_tgt = Xt[b_idx], Yt[b_idx]
            fake_tgt = G(real_src)
            # Discriminator
            d_real = D(real_src, real_tgt)
            d_fake = D(real_src, fake_tgt.detach())
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()
            # Generator
            d_fake_g = D(real_src, fake_tgt)
            g_loss = bce(d_fake_g, torch.ones_like(d_fake_g)) + lam * l1(fake_tgt, real_tgt)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            d_total += d_loss.item(); g_total += g_loss.item(); n_steps += 1
        print(f"  epoch {ep+1:2d}/{epochs}  D {d_total/n_steps:.3f}  G {g_total/n_steps:.3f}")
    return G, D


t0 = time.time()
G_p2p, _ = train_pix2pix(X_train, Y_train, epochs=8)
print(f"Done in {time.time() - t0:.1f}s.")""")

    b.code("""# Inference
G_p2p.eval()
with torch.no_grad():
    Y_pred_p2p = G_p2p(Xt_test).cpu().numpy().squeeze(1)

fig, axes = plt.subplots(n_show, 3, figsize=(11, 3.5 * n_show))
if n_show == 1:
    axes = axes.reshape(1, -1)
for i in range(n_show):
    axes[i, 0].imshow(X_test[i], cmap='Blues_r'); axes[i, 0].set_title("DAPI input")
    axes[i, 1].imshow(Y_pred_p2p[i], cmap='magma'); axes[i, 1].set_title("pix2pix prediction")
    axes[i, 2].imshow(Y_test[i], cmap='magma'); axes[i, 2].set_title("Ground truth")
    for ax in axes[i]: ax.axis('off')
plt.tight_layout(); plt.show()""")


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------
def section_compare(b):
    b.md("""## Side-by-side comparison

Both methods on the same held-out slices. Pixel similarity (PSNR, SSIM) and visual inspection.""")

    b.code("""# @title Pick a held-out slice to compare { run: \"auto\" }
slice_idx = 0  # @param {type: \"slider\", min: 0, max: 9, step: 1}

if slice_idx >= X_test.shape[0]:
    slice_idx = 0

inp = X_test[slice_idx]
gt = Y_test[slice_idx]
pred_fnet = Y_pred_fnet[slice_idx]
pred_p2p = Y_pred_p2p[slice_idx]

psnr_fnet = psnr_metric(gt, pred_fnet, data_range=1.0)
psnr_p2p = psnr_metric(gt, pred_p2p, data_range=1.0)
ssim_fnet = ssim_metric(gt, pred_fnet, data_range=1.0)
ssim_p2p = ssim_metric(gt, pred_p2p, data_range=1.0)

print(f"Held-out slice {slice_idx}:")
print(f"  fnet     : PSNR {psnr_fnet:5.2f} dB,  SSIM {ssim_fnet:.3f}")
print(f"  pix2pix  : PSNR {psnr_p2p:5.2f} dB,  SSIM {ssim_p2p:.3f}")

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
axes[0].imshow(inp, cmap='Blues_r'); axes[0].set_title("Input (DAPI)"); axes[0].axis('off')
axes[1].imshow(pred_fnet, cmap='magma'); axes[1].set_title(f"fnet  (PSNR {psnr_fnet:.1f}dB)"); axes[1].axis('off')
axes[2].imshow(pred_p2p, cmap='magma'); axes[2].set_title(f"pix2pix  (PSNR {psnr_p2p:.1f}dB)"); axes[2].axis('off')
axes[3].imshow(gt, cmap='magma'); axes[3].set_title("Ground truth membrane"); axes[3].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Reading the comparison.**

- *fnet* tends to give a smoother, blurrier output. Optimizing MSE pushes the model toward the conditional mean — it hedges where the answer is uncertain.
- *pix2pix* tends to give sharper, more *plausible-looking* output. The discriminator forces the generator to emit images with the right texture statistics, even where the underlying signal doesn't tell you exactly what should be there.

**That sharpness is a warning sign**, not a quality signal. The next section makes the point concrete.""")


# ---------------------------------------------------------------------------
# Hallucination check
# ---------------------------------------------------------------------------
def section_hallucination(b):
    b.md("""## The hallucination check (the central pedagogy of this lab)

Lab 3a applied this check to denoising. The same pattern applies to virtual staining — and lands harder, because virtual staining inherently invents pixels (the source image doesn't directly contain the target signal).

We compute a *difference map* between each model's prediction and the ground-truth membrane image. Bright regions in the difference map are places the model got wrong. Among those, some are noise smoothing (forgivable); others are *structural disagreements* — features the model invented that aren't in the ground truth, or features it missed that are.""")

    b.code("""# Difference maps
diff_fnet = pred_fnet - gt
diff_p2p = pred_p2p - gt

# Threshold to find structural disagreements
thresh = 0.20  # tune to taste; 0.20 in normalized [0, 1] units is a fairly loud disagreement
anomaly_fnet = np.abs(diff_fnet) > thresh
anomaly_p2p = np.abs(diff_p2p) > thresh

print(f"Anomaly pixels (|diff| > {thresh}):")
print(f"  fnet     : {anomaly_fnet.sum():>6} / {anomaly_fnet.size} ({100*anomaly_fnet.mean():.1f}%)")
print(f"  pix2pix  : {anomaly_p2p.sum():>6} / {anomaly_p2p.size} ({100*anomaly_p2p.mean():.1f}%)")

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes[0, 0].imshow(pred_fnet, cmap='magma'); axes[0, 0].set_title("fnet prediction"); axes[0, 0].axis('off')
axes[0, 1].imshow(diff_fnet, cmap='RdBu_r', vmin=-0.5, vmax=0.5); axes[0, 1].set_title("fnet diff (pred − gt)"); axes[0, 1].axis('off')
axes[0, 2].imshow(pred_fnet, cmap='gray')
axes[0, 2].imshow(np.where(anomaly_fnet, 1, np.nan), cmap='autumn', alpha=0.6)
axes[0, 2].set_title(f"fnet anomalies (|diff|>{thresh})"); axes[0, 2].axis('off')

axes[1, 0].imshow(pred_p2p, cmap='magma'); axes[1, 0].set_title("pix2pix prediction"); axes[1, 0].axis('off')
axes[1, 1].imshow(diff_p2p, cmap='RdBu_r', vmin=-0.5, vmax=0.5); axes[1, 1].set_title("pix2pix diff (pred − gt)"); axes[1, 1].axis('off')
axes[1, 2].imshow(pred_p2p, cmap='gray')
axes[1, 2].imshow(np.where(anomaly_p2p, 1, np.nan), cmap='autumn', alpha=0.6)
axes[1, 2].set_title(f"pix2pix anomalies (|diff|>{thresh})"); axes[1, 2].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**What you should be seeing.** pix2pix typically produces *more* anomaly pixels than fnet despite producing more visually appealing predictions. The adversarial loss pushed the generator to invent membrane structure that doesn't actually exist in the ground truth.

**The painful truth about virtual staining.** A model with high PSNR/SSIM and a "beautiful" output can still be wrong in ways that matter biologically. If you used the pix2pix prediction to *count* membrane-tagged structures, your count would be wrong. If you used it as a *figure* in a paper without disclosure, you'd be making claims about pixel locations that no microscope ever measured.

**Try this.** Slide the `slice_idx` parameter back through the held-out slices and watch where the anomalies land. Are they in the same regions across slices (suggesting a systematic failure mode of the model), or in different regions (suggesting per-image variability)?""")


# ---------------------------------------------------------------------------
# Per-region quantification
# ---------------------------------------------------------------------------
def section_quantify(b):
    b.md("""## Quantitative metrics across the held-out set

Per-image PSNR/SSIM are useful but not sufficient. We tabulate them across all held-out slices to see how stable each model's performance is, and we add an explicit **anomaly fraction** — what fraction of pixels disagree with ground truth above the threshold. The anomaly fraction is the *honest* quality signal for virtual staining.""")

    b.code("""rows = []
for i in range(X_test.shape[0]):
    gt_i = Y_test[i]
    for name, pred_arr in [("fnet", Y_pred_fnet), ("pix2pix", Y_pred_p2p)]:
        pred_i = pred_arr[i]
        rows.append({
            "slice": test_z[i],
            "model": name,
            "psnr": round(psnr_metric(gt_i, pred_i, data_range=1.0), 2),
            "ssim": round(ssim_metric(gt_i, pred_i, data_range=1.0), 3),
            "anomaly_pct": round(100 * (np.abs(pred_i - gt_i) > thresh).mean(), 1),
        })

import pandas as pd
df = pd.DataFrame(rows)
print("Per-slice metrics on held-out data:")
print(df.to_string(index=False))
print()
print("Mean across slices:")
print(df.groupby("model")[["psnr", "ssim", "anomaly_pct"]].mean().round(2).to_string())""")

    b.md("""**Key columns:**

- `psnr` and `ssim` reward outputs that look close to the reference. Higher is "better" by these metrics, but only if you trust the reference and only if the reference is what your biology actually depends on.
- `anomaly_pct` is the fraction of pixels where the model's prediction differs from ground truth by more than `thresh`. This is closer to a "biological wrongness" measure than the global similarity scores, because it says *how much of the image is wrong* rather than the average wrongness.

A good model has all three favorable. **A model that scores well on PSNR/SSIM but high on anomaly_pct is producing visually plausible but biologically incorrect output.** That's the failure mode that has retracted papers.""")


# ---------------------------------------------------------------------------
# Integrity reporting walkthrough
# ---------------------------------------------------------------------------
def section_integrity(b):
    b.md("""## Integrity reporting walkthrough

Most journal image-integrity policies (and most reviewers) will treat an undisclosed AI-staining figure as a manipulation. The minimum disclosure: tell the reader the displayed image is AI-generated, name the method and version, name the input modality, and report quantitative analyses on the *raw* unstained data — not on the model output.

The cell below renders a methods + figure-caption template you can adapt for your own paper.""")

    b.code("""template = '''
==============================
Methods (image-analysis subsection)
==============================

Virtual staining was performed using a [METHOD: fnet-style U-Net | pix2pix-style
conditional GAN] (architecture: [ARCH], training: [N_PAIRS] paired images of
[INPUT_MODALITY] / [TARGET_MODALITY] from [SOURCE], optimizer: [OPT], epochs:
[N_EPOCHS], framework: PyTorch [VERSION]). Held-out evaluation reported PSNR
[VALUE] dB, SSIM [VALUE], and anomaly fraction [VALUE]% (defined as the
fraction of pixels with |prediction − ground truth| > [THRESH] in normalized
intensity units). All quantitative measurements reported in the text were
performed on the original [INPUT_MODALITY] data; figure panels marked "*"
display the virtual-staining output.

==============================
Figure caption (where AI-stained images appear)
==============================

Figure [N]. [Description of the experiment]. * [Channel name] was generated by
[METHOD] virtual staining from the [INPUT_MODALITY] channel. The displayed
image is a model prediction, not a fluorescence measurement. Quantitative
analyses were performed on the original raw data. Source training data,
trained model weights, and inference scripts are available at [REPO]
under [LICENSE].

==============================
What this disclosure does NOT cover
==============================

- It does not justify the choice of method — that goes in Methods proper.
- It does not say "the virtual-staining output is correct" — it says "the
  reader can tell which image is real and which is generated."
- It does not absolve you of the validation step. The PSNR/SSIM/anomaly_pct
  numbers are what tell readers how much to trust the displayed image.
'''

print(template)""")

    b.md("""**The minimum bar.** If your paper has a virtual-staining figure, the disclosure above is the floor — not the ceiling. The ceiling is also publishing your training data, your trained weights, and your inference code so a reader can reproduce or audit your prediction.

If that bar feels high, consider whether the virtual-staining output is necessary for the paper or whether the same biological claim can be made on the raw data. Often it can be.""")


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------
def section_closing(b):
    b.md("""## Closing reflection

This lab demonstrated:

1. Two virtual-staining methods (fnet / pix2pix) trained on real cross-channel microscopy data.
2. Visual + quantitative comparison against ground truth on held-out slices.
3. The hallucination check — finding regions where the model invented features.
4. The per-slice anomaly fraction as a more honest quality signal than PSNR/SSIM alone.
5. A reporting template that satisfies image-integrity expectations.

**Where to go next on your own:**

- The **production-quality fnet Colab** in [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic). 3D variant, longer training, real data pipelines. Linked from Notebook 04.
- **CycleGAN** for *unpaired* virtual staining when paired data isn't available — covered in Notebook 04 as a catalog entry.
- The **ANNA-PALM** and **DeepImageJ** entries in the [Resources page](../resources) for downstream applications of virtual-staining-like methods.
- The original **fnet paper** (Ounkomol et al., *Nature Methods* 2018) and **in-silico labeling paper** (Christiansen et al., *Cell* 2018) for the canonical references.

**What the workshop will *not* tell you:** whether virtual staining is appropriate for your specific experiment. That's a biology question, not a method question. The method is real and useful in some contexts and wrong to use in others. The validation in this lab gives you the tools to make that call deliberately.""")


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def main():
    b = CellBuilder("nb06")
    section_title(b)
    section_setup(b)
    section_data(b)
    section_fnet(b)
    section_pix2pix(b)
    section_compare(b)
    section_hallucination(b)
    section_quantify(b)
    section_integrity(b)
    section_closing(b)
    build_notebook(b.cells, "06_virtual_staining")


if __name__ == "__main__":
    main()
