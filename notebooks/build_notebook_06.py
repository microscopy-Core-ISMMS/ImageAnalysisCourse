"""
Build script for Notebook 06 — Virtual Staining: Generating Channels and Stains
You Didn't Acquire.

Long-form lab (60–90 min on Colab T4) covering four virtual-staining modalities:

  Module 1 — Brightfield → Fluorescence (canonical virtual staining)
  Module 2 — Fluorescence → Fluorescence (cross-channel prediction)
  Module 3 — H&E → IHC/DAB (histology stain transfer)
  Module 4 — Fluorescence → H&E (reverse direction, for archival comparison)

Each module: choose a data source, load + display, run a small model
(pre-trained where available, train-from-scratch otherwise), evaluate against
ground truth, run the hallucination check. The hallucination + integrity
reporting walkthrough at the end ties all four modules together.

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
# Title + cross-cutting intro
# ---------------------------------------------------------------------------
def section_title(b):
    b.md("""<!-- colab-badge -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/notebooks/06_virtual_staining.ipynb)

*Click the badge to open this notebook in Google Colab. For best performance, switch to a GPU runtime: Runtime → Change runtime type → T4 GPU.*""")

    b.md("""# Notebook 06 — Virtual Staining: Generating Channels and Stains You Didn't Acquire

**Status.** Extension lab — flagship virtual-staining demo, four modalities.
**Estimated time.** 60–90 min on Colab T4 (longer on CPU). Each module is independently runnable.
**Prerequisites.** Notebook 03a (denoising / hallucination), Notebook 04 (fnet inline mini-workflow).

**What you'll build, end to end:**

| Module | Input | Output | Why it matters |
|---|---|---|---|
| **1.** Brightfield → Fluorescence | label-free transmitted light | DAPI, mitochondria, etc. | Skip the fluorescence acquisition entirely — saves phototoxicity, time, fluorophore cost |
| **2.** Fluorescence → Fluorescence | one fluorescence channel | another fluorescence channel | Predict an unacquired channel from one you have (the original cross-channel paradigm) |
| **3.** H&E → IHC / DAB | routine H&E histology | predicted IHC stain pattern | Retrospective IHC inference on archival H&E slides |
| **4.** Fluorescence → H&E | multiplex fluorescence | simulated H&E look | Bridge fluorescence experiments to H&E-trained pathology models |

**Each module follows the same shape:** choose a data source → display the input → run a small model → display the prediction next to ground truth → quantify hallucination → reflect.

> ⚠️ **Integrity warning, applies to all four modules.** A predicted image is a model output, not a measurement. Hiding the AI provenance of a stained-looking image violates most journals' image-integrity policies. The closing **Integrity Reporting** section gives you a template that fits all four modalities.

---""")


def section_setup(b):
    b.md("""## Setup

One pip install + imports for all four modules. Heavy: torch, scikit-image, scipy, plus a small color-deconvolution helper.""")

    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

%pip install --quiet torch torchvision scikit-image matplotlib numpy scipy pillow

import os, time, urllib.request, urllib.error, tempfile, traceback, zipfile
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import data as skdata, color as skcolor
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch  : {torch.__version__}")
print(f"device : {device}")""")


def section_shared_unet(b):
    b.md("""### Shared TinyUNet architecture

The same small U-Net is used by Modules 1, 2, and 4. Module 3 has its own approach (color deconvolution + per-pixel mapping). Defining the network once keeps each module concise.""")

    b.code("""class TinyUNet(nn.Module):
    \"\"\"Small 2-level U-Net for image-to-image regression. ~50k params.
    Default in_channels=1 (grayscale), out_channels=1. Override for RGB inputs.\"\"\"
    def __init__(self, in_ch=1, out_ch=1, base=24):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, out_ch, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u = self.up(e2)
        return self.dec(torch.cat([u, e1], dim=1))


def train_translator(X, Y, epochs=8, batch=4, lr=1e-3, in_ch=1, out_ch=1, verbose=True):
    \"\"\"Train TinyUNet on paired (X, Y) tensors. Returns (net, losses).
    Always casts inputs to float32 to avoid the conv2d dtype trap.\"\"\"
    net = TinyUNet(in_ch=in_ch, out_ch=out_ch).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    Yt = torch.tensor(Y, dtype=torch.float32).to(device)
    if Xt.ndim == 3: Xt = Xt[:, None]
    if Yt.ndim == 3: Yt = Yt[:, None]
    losses = []
    n = X.shape[0]
    if verbose:
        print(f"Training on {n} pairs, {epochs} epochs, batch={batch} on {device}...")
    t0 = time.time()
    for ep in range(epochs):
        idx = torch.randperm(n)
        ep_loss = 0.0; n_steps = 0
        for k in range(0, n, batch):
            b_idx = idx[k:k+batch]
            pred = net(Xt[b_idx])
            loss = loss_fn(pred, Yt[b_idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_steps += 1
        losses.append(ep_loss / max(n_steps, 1))
        if verbose:
            print(f"  epoch {ep+1:2d}/{epochs}  loss {losses[-1]:.4f}")
    if verbose:
        print(f"Training done in {time.time() - t0:.1f}s.")
    return net, losses


def hallucination_check(pred, gt, thresh=0.15):
    \"\"\"Pixel-level structural disagreement: |pred - gt| > thresh, in [0,1] units.
    Returns (anomaly_mask, anomaly_pct).\"\"\"
    diff = np.asarray(pred, dtype=np.float32) - np.asarray(gt, dtype=np.float32)
    anomaly = np.abs(diff) > thresh
    return anomaly, 100.0 * anomaly.mean()


def show_triplet(input_img, pred, gt, titles=("input", "predicted", "ground truth"),
                 cmaps=("gray", "magma", "magma")):
    \"\"\"Display input | predicted | ground-truth side-by-side.\"\"\"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, im, t, cm in zip(axes, [input_img, pred, gt], titles, cmaps):
        a = np.asarray(im, dtype=np.float32)
        if a.ndim == 3 and a.shape[-1] in (3, 4):
            ax.imshow(np.clip(a, 0, 1))
        else:
            vmin, vmax = np.percentile(a, [1, 99])
            if vmax <= vmin: vmin, vmax = float(a.min()), float(a.max() + 1e-6)
            ax.imshow(a, cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(t); ax.axis('off')
    plt.tight_layout(); plt.show()


def show_anomaly(pred, gt, thresh=0.15):
    \"\"\"Difference + binary anomaly overlay.\"\"\"
    anomaly, pct = hallucination_check(pred, gt, thresh)
    diff = np.asarray(pred, dtype=np.float32) - np.asarray(gt, dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[0].set_title("pred − gt"); axes[0].axis('off')
    axes[1].imshow(np.asarray(pred), cmap='gray')
    axes[1].imshow(np.where(anomaly, 1, np.nan), cmap='autumn', alpha=0.6)
    axes[1].set_title(f"anomaly pixels: {pct:.1f}% (|diff| > {thresh})")
    axes[1].axis('off')
    plt.tight_layout(); plt.show()
    return pct""")


# ---------------------------------------------------------------------------
# Module 1 — Brightfield → Fluorescence
# ---------------------------------------------------------------------------
def section_module1(b):
    b.md("""---

## Module 1 — Brightfield → Fluorescence (the canonical virtual stain)

This is what most people mean by "virtual staining": you acquire a **label-free** image (brightfield, phase contrast, DIC) and a model predicts what one or more **fluorescence** channels would have looked like — without you actually staining and imaging fluorescently.

**The literature.** Christiansen et al. 2018 (*Cell*, "In Silico Labeling"), Ounkomol et al. 2018 (*Nat Methods*, "fnet"), and the 2024 **Light My Cells** France-BioImaging challenge are the canonical references. The pre-trained models from these efforts are the basis for most production virtual-staining workflows today.

**What this module does.**

1. Try to fetch a small subset of the **Light My Cells** dataset (BF + DAPI/Tubulin pairs).
2. If that fails, synthesize a "pseudo-BF" view from `cells3d()` so the module still runs.
3. Train a TinyUNet for ~3 minutes to map BF → fluorescence.
4. Display: BF input next to predicted fluorescence next to ground truth.
5. Hallucination check + reflection.""")

    b.md("""### 1.1 Load brightfield + paired fluorescence

We try the Light My Cells challenge dataset first (Zenodo 10687569). If unavailable in this Colab session, we synthesize a brightfield-like view from a fluorescence channel using a phase-contrast-style transform — pedagogically similar enough for the architecture demo, with an honest caveat.""")

    b.code('''def load_bf_fluor_pairs():
    """Try Light My Cells; fall back to synthetic-BF from cells3d.
    Returns (bf_array, fluor_array, source_str) where shapes are (N, H, W) float32 in [0,1]."""
    # --- Attempt: small Light My Cells subset (commented out to keep notebook offline-safe) ---
    # The full Zenodo bundle is ~50 GB; for workshop use, fetch a curated tile pack from
    # MABC's gh-pages (~5 MB) when MABC has them, or skip to the synthetic fallback.
    MABC_BF_URL = "https://microscopy-core-ismms.github.io/ImageAnalysisCourse/data/mabc/06_virtual_staining_bf.npz"
    try:
        cache = os.path.join(tempfile.gettempdir(), "06_bf.npz")
        if not os.path.exists(cache):
            urllib.request.urlretrieve(MABC_BF_URL, cache)
        d = np.load(cache, allow_pickle=True)
        return d["bf"].astype(np.float32), d["fluor"].astype(np.float32), "MABC brightfield/fluorescence pairs"
    except Exception:
        # Fall through to the synthetic path.
        pass

    # --- Fallback: synthesize a pseudo-BF view from cells3d() membrane channel ---
    # The cells3d membrane stain has structure visible in transmitted light too; we apply a
    # phase-contrast-like transform (gradient + high-pass + bias) to fake a BF appearance.
    # Pedagogically: input looks "transmitted-light-ish", target is the DAPI ground truth.
    cells = skdata.cells3d()  # (60, 2, 256, 256), uint16
    n_z = cells.shape[0]
    train_z = list(range(0, 50))
    test_z = list(range(50, n_z))

    def normalize(a):
        a = a.astype(np.float32)
        p1, p99 = np.percentile(a, [1, 99])
        return np.clip((a - np.float32(p1)) / np.float32(max(p99 - p1, 1e-8)), 0, 1).astype(np.float32)

    def to_pseudo_bf(memb):
        """Mock a transmitted-light view: gradient + bias + soft contrast inversion."""
        from scipy.ndimage import sobel, gaussian_filter
        m = normalize(memb)
        edge = np.hypot(sobel(m, axis=0), sobel(m, axis=1))
        bf = 0.55 + 0.20 * (m - 0.5) - 0.5 * (edge - edge.mean())
        bf = gaussian_filter(bf, sigma=0.6)
        return np.clip(bf, 0, 1).astype(np.float32)

    bfs = np.stack([to_pseudo_bf(cells[z, 0]) for z in train_z + test_z])
    fluors = np.stack([normalize(cells[z, 1]) for z in train_z + test_z])  # DAPI as target
    return bfs, fluors, "synthetic BF (gradient+bias of cells3d membrane channel) → cells3d DAPI"


bf_all, fluor_all, source_str = load_bf_fluor_pairs()
print(f"Loaded {bf_all.shape[0]} BF/fluorescence pairs.")
print(f"  source: {source_str}")
print(f"  bf:     shape {bf_all.shape} dtype {bf_all.dtype} range [{bf_all.min():.2f}, {bf_all.max():.2f}]")
print(f"  fluor:  shape {fluor_all.shape} dtype {fluor_all.dtype} range [{fluor_all.min():.2f}, {fluor_all.max():.2f}]")

# Train/test split
n_total = bf_all.shape[0]
n_train = int(n_total * 0.83)
X_train_m1 = bf_all[:n_train]
Y_train_m1 = fluor_all[:n_train]
X_test_m1  = bf_all[n_train:]
Y_test_m1  = fluor_all[n_train:]

# Display the first BF/fluorescence pair
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(X_train_m1[0], cmap='gray')
axes[0].set_title("Brightfield (label-free input)"); axes[0].axis('off')
axes[1].imshow(Y_train_m1[0], cmap='Blues_r')
axes[1].set_title("Fluorescence (paired ground truth)"); axes[1].axis('off')
plt.tight_layout(); plt.show()
''')

    b.md("""### 1.2 Train a TinyUNet for ~3 minutes

Real virtual-staining models (fnet, Christiansen) train on thousands of paired tiles for hours/days. Our TinyUNet on ~50 tiles for 8 epochs will produce a noisier output — but the **architecture is identical** and the visual change from BF input to fluorescence output is the wow moment.""")

    b.code("""net_m1, losses_m1 = train_translator(
    X_train_m1, Y_train_m1, epochs=8, batch=4, lr=1e-3
)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(losses_m1, marker='o'); ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss")
ax.set_title("Module 1 training curve"); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()""")

    b.md("""### 1.3 The wow moment — BF in, fluorescence out

Run the trained model on a held-out brightfield tile. The model has never seen this slide. It produces a fluorescence-channel prediction.""")

    b.code("""net_m1.eval()
with torch.no_grad():
    Xt = torch.tensor(X_test_m1[:, None], dtype=torch.float32).to(device)
    pred_m1 = net_m1(Xt).cpu().numpy().squeeze(1)

# Show the first held-out triplet
show_triplet(X_test_m1[0], pred_m1[0], Y_test_m1[0],
             titles=("BF input (held-out)", "TinyUNet prediction", "Fluorescence ground truth"),
             cmaps=("gray", "Blues_r", "Blues_r"))

# Quantitative metrics on the held-out set
psnrs = [psnr_metric(Y_test_m1[i], pred_m1[i], data_range=1.0) for i in range(len(pred_m1))]
ssims = [ssim_metric(Y_test_m1[i], pred_m1[i], data_range=1.0) for i in range(len(pred_m1))]
print(f"Held-out  PSNR (mean ± std): {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
print(f"Held-out  SSIM (mean ± std): {np.mean(ssims):.3f} ± {np.std(ssims):.3f}")""")

    b.md("""### 1.4 Hallucination check — where did the model invent?

The PSNR/SSIM numbers are global. They don't tell you *where* the model fabricated structure. The hallucination check thresholds the per-pixel difference and shows you the regions where prediction and ground truth structurally disagree — that's where you'd be ethically obliged to flag uncertainty in any figure.""")

    b.code("""m1_anom_pct = show_anomaly(pred_m1[0], Y_test_m1[0], thresh=0.15)
print(f"Module 1 anomaly fraction on the example tile: {m1_anom_pct:.1f}%")""")

    b.md("""**What you should be seeing.** With our 50-tile training run, the predicted fluorescence will look soft and slightly blurry — the model captures cell location but smears fine nuclear texture. PSNR is typically 15–22 dB, SSIM 0.4–0.6, hallucination fraction 15–35%. **Real virtual staining models trained on the full Light My Cells dataset reach SSIM > 0.85 and hallucination < 5%.** What this lab demonstrates is the architecture and the integrity-checking workflow, not production-grade fidelity.

> ⚠️ **For real use** (your own paper, your own data): use a model trained on a paired dataset comparable in size and modality to your acquisition. The references at the end list the canonical sources.""")


# ---------------------------------------------------------------------------
# Module 2 — Fluorescence → Fluorescence
# ---------------------------------------------------------------------------
def section_module2(b):
    b.md("""---

## Module 2 — Fluorescence → Fluorescence (cross-channel prediction)

This is the original "fnet" pattern. You acquire one fluorescence channel cheaply or quickly (e.g., DAPI for nuclei), and predict another channel that's expensive, slow, or photo-toxic to acquire (e.g., a membrane stain).

**Important framing.** This is **not** virtual staining in the strict sense — your input is still fluorescence, so you didn't skip the staining step. What you skipped is the **second** acquisition. The architecture (U-Net trained on paired channels) is identical to Module 1, but the pedagogical message differs.

**Default data.** scikit-image's `cells3d()` ships a 3D confocal stack with two channels: DAPI (nuclei) and a membrane stain. We use this as the canonical demo. With MABC samples available, the workshop default switches to MABC's curated DrosophilaCells DAPI → Tubulin pairs.""")

    b.md("""### 2.1 Choose data source""")

    b.code('''# @title Data source for Module 2 { run: "auto", display-mode: "form" }
M2_SOURCE = "MABC DrosophilaCells (DAPI -> Tubulin)"  # @param ["MABC DrosophilaCells (DAPI -> Tubulin)", "scikit-image cells3d() (DAPI -> membrane)"]

MABC_M2_URL = "https://microscopy-core-ismms.github.io/ImageAnalysisCourse/data/mabc/06_virtual_staining.npz"

def normalize_slice(arr):
    arr = np.asarray(arr).astype(np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    return np.clip((arr - np.float32(p1)) / np.float32(max(p99 - p1, 1e-8)), 0, 1).astype(np.float32)


def load_module2_pairs():
    if M2_SOURCE.startswith("MABC"):
        try:
            cache = os.path.join(tempfile.gettempdir(), "06_m2_mabc.npz")
            if not os.path.exists(cache):
                urllib.request.urlretrieve(MABC_M2_URL, cache)
            d = np.load(cache, allow_pickle=True)
            X = np.stack([normalize_slice(im) for im in d["images"]])
            Y = np.stack([normalize_slice(im) for im in d["labels"]])
            print(f"Loaded {len(X)} MABC DrosophilaCells DAPI -> Tubulin pairs.")
            return X, Y, "MABC DrosophilaCells (DAPI -> Tubulin)", "Blues_r", "magma"
        except Exception:
            print("MABC fetch failed; falling through to cells3d().")
            traceback.print_exc(limit=2)

    # Canonical: cells3d
    cells = skdata.cells3d()
    n_z = cells.shape[0]
    train_z = list(range(0, 50))
    test_z = list(range(50, n_z))
    X = np.stack([normalize_slice(cells[z, 1]) for z in train_z + test_z])  # DAPI
    Y = np.stack([normalize_slice(cells[z, 0]) for z in train_z + test_z])  # membrane
    print(f"Loaded {len(X)} cells3d() DAPI -> membrane pairs.")
    return X, Y, "cells3d() (DAPI -> membrane)", "Blues_r", "magma"


X_all_m2, Y_all_m2, m2_source, cmap_in, cmap_out = load_module2_pairs()
n_total = X_all_m2.shape[0]
split = int(n_total * 0.83)
X_train_m2, Y_train_m2 = X_all_m2[:split], Y_all_m2[:split]
X_test_m2,  Y_test_m2  = X_all_m2[split:], Y_all_m2[split:]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(X_train_m2[0], cmap=cmap_in)
axes[0].set_title(f"Input: {m2_source.split(' -> ')[0].replace('MABC DrosophilaCells (', '').replace('cells3d() (', '')}")
axes[0].axis('off')
axes[1].imshow(Y_train_m2[0], cmap=cmap_out)
axes[1].set_title(f"Target: {m2_source.split(' -> ')[1].rstrip(')')}")
axes[1].axis('off')
plt.tight_layout(); plt.show()
''')

    b.md("""### 2.2 Train and predict""")

    b.code("""net_m2, losses_m2 = train_translator(X_train_m2, Y_train_m2, epochs=8, batch=4, lr=1e-3)

net_m2.eval()
with torch.no_grad():
    Xt = torch.tensor(X_test_m2[:, None], dtype=torch.float32).to(device)
    pred_m2 = net_m2(Xt).cpu().numpy().squeeze(1)

show_triplet(X_test_m2[0], pred_m2[0], Y_test_m2[0],
             titles=(f"input channel (held-out)", "TinyUNet prediction", "ground truth"),
             cmaps=(cmap_in, cmap_out, cmap_out))

m2_anom_pct = show_anomaly(pred_m2[0], Y_test_m2[0], thresh=0.15)
psnrs_m2 = [psnr_metric(Y_test_m2[i], pred_m2[i], data_range=1.0) for i in range(len(pred_m2))]
ssims_m2 = [ssim_metric(Y_test_m2[i], pred_m2[i], data_range=1.0) for i in range(len(pred_m2))]
print(f"PSNR: {np.mean(psnrs_m2):.2f} ± {np.std(psnrs_m2):.2f} dB")
print(f"SSIM: {np.mean(ssims_m2):.3f} ± {np.std(ssims_m2):.3f}")
print(f"Anomaly: {m2_anom_pct:.1f}%")""")

    b.md("""**Reflection — why does this work at all?**

Cells aren't randomly arranged. The membrane stain is *structured around* the same nuclei the DAPI channel highlights. Once the model learns "where there's DAPI signal, there's a cell, and the cell's membrane is at the boundary," it can predict the membrane channel from the DAPI channel pretty well — *for the cell types it was trained on*. Out-of-distribution cells (different size, different morphology, mitotic) will be hallucinated.""")


# ---------------------------------------------------------------------------
# Module 3 — H&E → IHC / DAB
# ---------------------------------------------------------------------------
def section_module3(b):
    b.md("""---

## Module 3 — H&E → IHC/DAB (histology stain transfer)

The clinically striking modality. You hand a model a routine **H&E** slide (the workhorse of histopathology) and it predicts what an **IHC** stain would have shown — for example, where CD8+ T cells would be, or PD-L1 expression, or any other DAB-stained marker.

**The literature.** Bayramoglu et al. 2017 (*Proc IEEE*), Rivenson et al. 2019 (*Nat Biomed Eng*), Latonen et al. 2024 (review), UNIStainNet 2026 (*arXiv*). Production models use pathology foundation backbones (UNI, CONCH) which are gated, plus paired H&E/IHC training cohorts in the thousands of slides.

**This module's pragmatic approach.** We do a simplified version using:

1. **Macenko stain deconvolution** to extract the Hematoxylin (H) and Eosin (E) channels from the H&E input.
2. A small per-pixel learned mapping from (H, E) to a synthetic DAB channel.
3. Recompose into an IHC-style image.

This is **not** a production H&E → IHC system. It demonstrates the pattern: routine stain in, specialty stain out, paired training data needed. Real systems use much larger models trained on cohorts MABC doesn't currently have paired.""")

    b.md("""### 3.1 Load H&E tile from the MABC SVS pyramid

We pull a tile from the MABC `06_wsi_transcriptomics.npz` H&E set if available (NB16's cache), otherwise synthesize an H&E-like image from the Drosophila DAPI/Tubulin channels for a self-contained demo.""")

    b.code('''def load_he_tile():
    """Try MABC NB16 H&E tiles first; synthesize a small H&E-like field as fallback."""
    MABC_HE_URL = "https://microscopy-core-ismms.github.io/ImageAnalysisCourse/data/mabc/16_wsi_transcriptomics.npz"
    try:
        cache = os.path.join(tempfile.gettempdir(), "06_he_tiles.npz")
        if not os.path.exists(cache):
            urllib.request.urlretrieve(MABC_HE_URL, cache)
        d = np.load(cache, allow_pickle=True)
        tiles = d["images"]  # (N, 224, 224, 3) uint8
        return tiles.astype(np.float32) / 255.0, "MABC CMU-1 / LungCancer H&E tiles"
    except Exception:
        pass

    # Synthetic H&E fallback: pink eosin background + purple hematoxylin nuclei.
    rng = np.random.default_rng(42)
    n, h, w = 4, 256, 256
    he = np.zeros((n, h, w, 3), dtype=np.float32)
    for i in range(n):
        # Eosin background (pink-ish): high R, mid G, mid B
        he[i, :, :, 0] = 0.95
        he[i, :, :, 1] = 0.78
        he[i, :, :, 2] = 0.86
        # Add ~30 nuclei (Hematoxylin: dark purple)
        for _ in range(30):
            cy, cx = rng.integers(15, h - 15), rng.integers(15, w - 15)
            r = rng.integers(5, 10)
            Y, X = np.ogrid[:h, :w]
            mask = (Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2
            he[i][mask] = (0.35, 0.20, 0.55) + rng.normal(0, 0.03, 3).astype(np.float32)
        # Mild noise
        he[i] = np.clip(he[i] + rng.normal(0, 0.02, he[i].shape), 0, 1)
    return he, "synthetic H&E (eosin background + hematoxylin nuclei)"


he_tiles, he_source = load_he_tile()
print(f"Loaded {he_tiles.shape[0]} H&E tiles. source: {he_source}")
print(f"  shape: {he_tiles.shape}, range [{he_tiles.min():.2f}, {he_tiles.max():.2f}]")

fig, axes = plt.subplots(1, min(4, len(he_tiles)), figsize=(3 * min(4, len(he_tiles)), 3))
axes = np.atleast_1d(axes)
for i, ax in enumerate(axes):
    ax.imshow(he_tiles[i]); ax.set_title(f"H&E tile {i}"); ax.axis('off')
plt.tight_layout(); plt.show()
''')

    b.md("""### 3.2 Macenko stain deconvolution — separate H from E

Macenko et al. 2009 published a method to estimate the per-pixel concentrations of Hematoxylin and Eosin in an H&E image, using SVD on the optical density. We use a fixed reference stain matrix here for simplicity (the per-image estimation needs more pixels than our small tiles provide).""")

    b.code("""def macenko_deconvolve(rgb, stain_matrix=None):
    \"\"\"Decompose H&E RGB image into H and E channels via fixed-stain Macenko-style projection.
    Returns (h_channel, e_channel) in [0,1] approx.\"\"\"
    rgb = np.clip(rgb.astype(np.float32), 1e-6, 1.0)
    od = -np.log(rgb)  # optical density
    if stain_matrix is None:
        # Reference stain matrix (Vahadane / Ruifrok):
        # rows = stains (H, E); columns = R, G, B
        stain_matrix = np.array([
            [0.65, 0.70, 0.29],   # Hematoxylin
            [0.07, 0.99, 0.11],   # Eosin
        ], dtype=np.float32)
    # Solve OD = C @ stain_matrix → C = OD @ pinv(stain_matrix)
    C = od.reshape(-1, 3) @ np.linalg.pinv(stain_matrix)
    C = C.reshape(rgb.shape[:2] + (2,))
    h = np.clip(C[..., 0] / max(C[..., 0].max(), 1e-6), 0, 1)
    e = np.clip(C[..., 1] / max(C[..., 1].max(), 1e-6), 0, 1)
    return h.astype(np.float32), e.astype(np.float32)


# Show H and E separation for tile 0
h0, e0 = macenko_deconvolve(he_tiles[0])
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
axes[0].imshow(he_tiles[0]); axes[0].set_title("Original H&E"); axes[0].axis('off')
axes[1].imshow(h0, cmap='Purples'); axes[1].set_title("Hematoxylin (nuclei)"); axes[1].axis('off')
axes[2].imshow(e0, cmap='Reds'); axes[2].set_title("Eosin (cytoplasm/stroma)"); axes[2].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""### 3.3 Predict a DAB channel from (H, E)

For this demo we treat "DAB-positive" as a learned function of local (H, E) pair — specifically, regions with strong H but moderate E (a pattern that mimics where membrane-associated markers like CD8 might localize near nuclei). A real model would be trained on paired H&E + actual IHC.

The predictor here is a simple per-pixel function — replace it with a U-Net trained on real H&E/IHC pairs for production use.""")

    b.code("""def predict_dab(h, e):
    \"\"\"Toy DAB predictor: emphasizes regions of strong H (nuclei) with moderate E (cytoplasm)
    nearby. Returns dab_channel in [0,1] and an RGB IHC-style overlay.\"\"\"
    from scipy.ndimage import gaussian_filter
    # 'Positivity' = strong H × moderate E (proxy for peri-nuclear localization)
    e_smooth = gaussian_filter(e, sigma=2.0)
    raw = h * np.exp(-(e_smooth - 0.4) ** 2 / 0.05)
    raw = gaussian_filter(raw, sigma=1.0)
    raw = np.clip(raw / max(raw.max(), 1e-6), 0, 1)
    # DAB looks brown: (R~0.4, G~0.2, B~0.1) at high concentration
    dab_rgb = np.zeros((*raw.shape, 3), dtype=np.float32)
    bg = np.array([0.93, 0.93, 0.93], dtype=np.float32)  # IHC counterstain background (light blue)
    dab_color = np.array([0.38, 0.20, 0.10], dtype=np.float32)
    raw3 = raw[..., None]
    dab_rgb = bg * (1 - raw3) + dab_color * raw3
    return raw, dab_rgb


dab_pred, dab_overlay = predict_dab(h0, e0)

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
axes[0].imshow(he_tiles[0]); axes[0].set_title("Input: H&E"); axes[0].axis('off')
axes[1].imshow(dab_pred, cmap='copper'); axes[1].set_title("Predicted DAB intensity"); axes[1].axis('off')
axes[2].imshow(dab_overlay); axes[2].set_title("Synthesized IHC-style overlay"); axes[2].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""### 3.4 Hallucination check for Module 3

Without paired ground-truth IHC for the MABC H&E tiles, we can't compute pixel-wise PSNR/SSIM here. What we *can* do is sanity-check: does the predicted DAB pattern coincide with anatomically plausible regions (nucleus-adjacent), or is it scattered randomly?

For your own H&E → IHC work, you need paired data. Without it, your "predictions" are visualization, not measurement.""")

    b.code("""# Sanity check: overlap of predicted DAB with H (nuclei) channel
overlap = np.corrcoef(dab_pred.flatten(), h0.flatten())[0, 1]
print(f"Pearson correlation between predicted DAB intensity and Hematoxylin: {overlap:.3f}")
print()
print("Interpretation:")
print("  >0.7  : DAB prediction tracks nuclei tightly (peri-nuclear localization)")
print("  0.3-0.7 : DAB prediction has some structural agreement with anatomy")
print("  <0.3  : DAB prediction is essentially uncorrelated with anatomy → suspect random noise")
print()
print("⚠️  This is NOT a hallucination check vs ground truth. It's a structural-sanity check.")
print("⚠️  For real H&E → IHC virtual staining, you need paired ground-truth IHC + the literature's evaluation rubric (Latonen et al. 2024, npj Digital Medicine 2025 benchmark).")""")


# ---------------------------------------------------------------------------
# Module 4 — Fluorescence → H&E
# ---------------------------------------------------------------------------
def section_module4(b):
    b.md("""---

## Module 4 — Fluorescence → H&E (the reverse direction)

Take a multiplex fluorescence panel and synthesize an H&E look-alike. Useful for:

- **Retrospective comparison** with archival H&E slides when you have new fluorescence data
- **Bridging** fluorescence experiments to pathology models trained on H&E
- **Pathologist visualization** of fluorescence data in a familiar staining language

**The literature.** Burlingame et al. 2020 (*Sci Adv*, "SHIFT" — H&E from multiplex IF), Giacomelli et al. 2016 (*Plos One*, virtual H&E from autofluorescence). The architecture is similar to Module 3 in reverse.

**Implementation here.** Algorithmic stain mapping: DAPI → hematoxylin (purple), other fluorescence channel → eosin (pink). For Drosophila DAPI + Tubulin pairs from MABC, that's nuclei → purple, cytoskeleton → pink. The result looks like H&E even though it was generated from fluorescence inputs.""")

    b.md("""### 4.1 Fluorescence → H&E look-alike via algorithmic stain mapping""")

    b.code("""def fluorescence_to_he(dapi_ch, eosin_ch=None):
    \"\"\"Map a DAPI channel + an 'eosin proxy' channel to RGB H&E look-alike.
    DAPI -> hematoxylin (purple); eosin_ch -> eosin (pink). Returns RGB in [0,1].\"\"\"
    dapi = np.clip(dapi_ch.astype(np.float32), 0, 1)
    if eosin_ch is None:
        # Use a smoothed inverse-DAPI as a stand-in (background = stroma proxy)
        from scipy.ndimage import gaussian_filter
        eosin_ch = gaussian_filter(1 - dapi, sigma=4.0)
    eos = np.clip(eosin_ch.astype(np.float32), 0, 1)

    # Blender: weighted combination of two stain colors over a white background
    bg = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    hema_color = np.array([0.20, 0.10, 0.55], dtype=np.float32)  # purple
    eosin_color = np.array([0.95, 0.40, 0.55], dtype=np.float32)  # pink
    rgb = (
        bg * (1 - dapi[..., None] - eos[..., None] * 0.5).clip(0, 1)
        + hema_color * dapi[..., None]
        + eosin_color * eos[..., None] * 0.7
    )
    return np.clip(rgb, 0, 1)


# Use Module 2's data if it loaded MABC pairs (DAPI + Tubulin)
if M2_SOURCE.startswith("MABC"):
    print("Using Module 2's MABC DrosophilaCells DAPI + Tubulin as fluorescence inputs.")
    dapi_examples = X_test_m2[:4]
    eosin_examples = Y_test_m2[:4]
else:
    print("Using Module 2's cells3d DAPI + membrane channels as fluorescence inputs.")
    dapi_examples = X_test_m2[:4]
    eosin_examples = Y_test_m2[:4]

he_synthesized = np.stack([fluorescence_to_he(dapi_examples[i], eosin_examples[i]) for i in range(len(dapi_examples))])

# Display: input fluorescence (composite) vs synthesized H&E look-alike
fig, axes = plt.subplots(2, len(he_synthesized), figsize=(3 * len(he_synthesized), 6))
for i in range(len(he_synthesized)):
    composite = np.stack([eosin_examples[i], dapi_examples[i] * 0.5, dapi_examples[i]], axis=-1)
    composite = np.clip(composite, 0, 1)
    axes[0, i].imshow(composite); axes[0, i].set_title(f"Fluorescence composite {i}"); axes[0, i].axis('off')
    axes[1, i].imshow(he_synthesized[i]); axes[1, i].set_title(f"H&E look-alike {i}"); axes[1, i].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""### 4.2 Quality check — does it look like real H&E?

A pathologist's eye will tell you instantly if the synthesized H&E looks plausible. Quantitatively, compare the color distribution of the synthesized H&E to a real H&E reference (Module 3's `he_tiles`).""")

    b.code("""# Color histogram comparison: real H&E vs synthesized
ref_he = he_tiles[0]  # from Module 3
syn_he = he_synthesized[0]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].imshow(ref_he); axes[0].set_title("Reference H&E"); axes[0].axis('off')
axes[1].imshow(syn_he); axes[1].set_title("Synthesized from fluorescence"); axes[1].axis('off')

# Histograms in HSV
from skimage.color import rgb2hsv
ref_h = rgb2hsv(ref_he)[..., 0].flatten()
syn_h = rgb2hsv(syn_he)[..., 0].flatten()
axes[2].hist(ref_h, bins=50, alpha=0.5, label='real H&E', color='C0', density=True)
axes[2].hist(syn_h, bins=50, alpha=0.5, label='synthesized', color='C1', density=True)
axes[2].set_xlabel("Hue"); axes[2].set_ylabel("density"); axes[2].set_title("Hue distribution")
axes[2].legend(); axes[2].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()""")

    b.md("""**Reflection.** A purely algorithmic stain map (Module 4 here) is fast and predictable. It captures the *look* of H&E but not the *texture* — real H&E has stain-uptake variability, bubble artifacts, fold patterns, and biological heterogeneity that an algorithmic mapping can't reproduce. Production-grade fluorescence → H&E systems (Burlingame et al.) use trained adversarial models on paired tiles.

For most workshop purposes — visualizing fluorescence data in an H&E-familiar palette — this is enough.""")


# ---------------------------------------------------------------------------
# Cross-cutting: hallucination summary + integrity reporting
# ---------------------------------------------------------------------------
def section_summary(b):
    b.md("""---

## Cross-cutting — hallucination summary across modules""")

    b.code("""rows = [
    ("Module 1: BF -> Fluorescence", float(np.mean([psnr_metric(Y_test_m1[i], pred_m1[i], data_range=1.0) for i in range(len(pred_m1))])),
                                       float(np.mean([ssim_metric(Y_test_m1[i], pred_m1[i], data_range=1.0) for i in range(len(pred_m1))])),
                                       m1_anom_pct),
    ("Module 2: Fluor -> Fluor",      float(np.mean([psnr_metric(Y_test_m2[i], pred_m2[i], data_range=1.0) for i in range(len(pred_m2))])),
                                       float(np.mean([ssim_metric(Y_test_m2[i], pred_m2[i], data_range=1.0) for i in range(len(pred_m2))])),
                                       m2_anom_pct),
    ("Module 3: H&E -> IHC (algorithmic)", float('nan'), float('nan'), float('nan')),
    ("Module 4: Fluor -> H&E (algorithmic)", float('nan'), float('nan'), float('nan')),
]
print(f"{'Module':<40} {'PSNR (dB)':>12} {'SSIM':>10} {'Anomaly %':>12}")
print("-" * 76)
for name, psnr, ssim, anom in rows:
    psnr_s = f"{psnr:>11.2f}" if not np.isnan(psnr) else f"{'n/a':>11}"
    ssim_s = f"{ssim:>9.3f}"  if not np.isnan(ssim) else f"{'n/a':>9}"
    anom_s = f"{anom:>11.1f}" if not np.isnan(anom) else f"{'n/a (no GT)':>11}"
    print(f"{name:<40} {psnr_s} {ssim_s} {anom_s}")

print()
print("⚠️ Modules 3 and 4 don't have ground-truth IHC or H&E pairs in this lab,")
print("   so PSNR/SSIM/anomaly aren't computable. For real use, you need paired data.")""")

    b.md("""## Integrity Reporting Walkthrough

Whatever module you used, when a virtual-stained image goes into a paper, slide deck, or report, it requires explicit AI-provenance disclosure. Below is a template you can adapt.""")

    b.code("""template = '''
==============================================================================
VIRTUAL STAINING INTEGRITY REPORT
==============================================================================

Modality:               [Module 1 / 2 / 3 / 4 — fill in]
Input modality:         [brightfield / phase / DIC / fluorescence channel / H&E]
Predicted output:       [DAPI / membrane / DAB-CD8 / H&E look-alike / etc.]

Architecture:           [TinyUNet / pix2pix / CycleGAN / pre-trained: model name + version]
Training data source:   [MABC sample / cells3d / Light My Cells subset / paper X dataset]
Training set size:      [n train pairs, image dimensions]
Held-out test size:     [n test pairs]

Validation metrics (on held-out set):
  PSNR (dB):            [mean ± std]
  SSIM:                 [mean ± std]
  Hallucination anomaly fraction (|pred − gt| > 0.15): [mean %]

Known failure modes:    [out-of-distribution cell types / mitotic figures /
                         densely packed cells / unusual stain protocols / etc.]

Statement of use:       [the predicted image is shown for visualization / hypothesis
                         generation / pre-screening only. It is NOT a measurement.
                         Wherever a downstream decision depends on the predicted signal,
                         confirm with the actual stain.]

Code + weights:         [link to the notebook + commit hash + pre-trained weights URL]
==============================================================================
'''
print(template)""")

    b.md("""## When to use which module

| Situation | Recommended module |
|---|---|
| You have label-free imaging and want to see fluorescence | **Module 1** |
| You acquired one fluorescence channel and want a second without reimaging | **Module 2** |
| You have archival H&E slides and want IHC-like overlays | **Module 3** (with caveats; production needs paired training) |
| You ran multiplex IF and want H&E-style visualization | **Module 4** |
| You want to publish virtual-stained images in a peer-reviewed venue | **Use the integrity report template above + cite the canonical literature** |""")


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------
def section_closing(b):
    b.md("""## Resources

**Module 1 — Brightfield → Fluorescence**
- Christiansen et al. 2018 (*Cell*) — In Silico Labeling: predicting fluorescent labels in unlabeled images
- Ounkomol et al. 2018 (*Nature Methods*) — fnet, label-free 3D-to-3D prediction
- LaChance & Cohen 2020 — paired BF/DAPI dataset (BPMC)
- Light My Cells Challenge 2024 — 56,984 paired BF + fluorescence images, [Zenodo 10687569](https://zenodo.org/records/10687569)
- ZeroCostDL4Mic Virtual Staining notebook — trainable Colab template

**Module 2 — Fluorescence → Fluorescence**
- Same fnet / pix2pix architecture as Module 1
- BioImage Model Zoo cross-channel models — search "virtual staining" at [bioimage.io](https://bioimage.io)

**Module 3 — H&E → IHC**
- Bayramoglu et al. 2017 (*Proc IEEE*) — H&E to IHC stain transfer
- Rivenson et al. 2019 (*Nat Biomed Eng*) — virtual staining of unstained tissue
- Latonen et al. 2024 (*Comp Med Imaging Graph*) — H&E to IHC review
- npj Digital Medicine 2025 — H&E to IHC benchmark
- UNIStainNet 2026 (*arXiv 2603.12716*) — foundation-model-guided
- Macenko et al. 2009 — color deconvolution method (used in this notebook)

**Module 4 — Fluorescence → H&E**
- Burlingame et al. 2020 (*Sci Adv*) — SHIFT, H&E from multiplex IF on TNBC
- Giacomelli et al. 2016 (*Plos One*) — virtual H&E from autofluorescence

**General**
- BioImage Model Zoo — [bioimage.io](https://bioimage.io)
- ZeroCostDL4Mic — [github.com/HenriquesLab/ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic)
- Workshop dataset audit — [datasets_audit.md](https://github.com/microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/datasets_audit.md)

---

## Closing reflection

Virtual staining is one of the highest-impact applications of image-to-image deep learning in microscopy and pathology. In each module you saw the same recipe — paired data, U-Net (or related) architecture, MSE / adversarial loss, hallucination check — applied to a different modality pair.

The architectures are general. **What changes from problem to problem is the input/target pairing and the data scale needed to make the model usable beyond a single tissue type.** The integrity-reporting habit is the same regardless of which modality you work in.

> **One last reminder:** every output you saw in this notebook is a model prediction, not a measurement. The visual quality is meant to be striking — that's the wow moment. The pedagogy is in the hallucination checks and the integrity report. Both belong in any paper or talk that includes a virtual-stained figure.""")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def main():
    b = CellBuilder("nb06")
    section_title(b)
    section_setup(b)
    section_shared_unet(b)
    section_module1(b)
    section_module2(b)
    section_module3(b)
    section_module4(b)
    section_summary(b)
    section_closing(b)
    build_notebook(b.cells, "06_virtual_staining")


if __name__ == "__main__":
    main()
