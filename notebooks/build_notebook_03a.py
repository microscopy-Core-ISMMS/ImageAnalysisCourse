"""
Build script for Notebook 03a — AI Denoising: CARE, Noise2Void, and pre-trained
models on real microscopy.

Layout (per Nikos's confirmation 2026-05-11):
  §1  Title + Colab badge
  §2  Setup (pip + imports)
  §3  Data decision (@param DATA_SOURCE: MABC / Canonical / Synthetic)
  §4  Load data + preview (with @param Z_INDEX slider over the full stack)
  §5  Method definitions (Gaussian, CARE-scratch, N2V-scratch, BiMZ pretrained)
  §6  Method execution — STEPWISE individual cells (run one by one by default)
      §6a Gaussian baseline (instant)
      §6b CARE from scratch (~3 min on T4)
      §6c N2V from scratch (~3 min on T4)
      §6d Pre-trained CARE from BioImage Model Zoo (~30 sec)
  §7  Optional: Run all four methods at once
  §8  Quantitative comparison — 4-row table + side-by-side grid
  §9  Hallucination check — per-method @param picker
  §10 Apply to held-out tile (production scenario, no clean reference)
  §11 Integrity reporting walkthrough
  §12 Going broader — community alternatives (markdown)
  §13 Closing reflection

Run:
    python build_notebook_03a.py
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
# §1 Title + Colab badge
# ---------------------------------------------------------------------------
def section_title(b):
    b.md("""<!-- colab-badge -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/notebooks/03a_denoising_n2v.ipynb)

*Click the badge to open this notebook in Google Colab. For best performance, switch to a GPU runtime: Runtime → Change runtime type → T4 GPU.*""")

    b.md("""# Notebook 03a — AI Denoising on Real Microscopy

**Status.** Lab 3, Option A — denoising flagship.
**Estimated time.** 30–60 min on Colab T4 (longer if you train both CARE and N2V from scratch).

**What you'll do, end to end:**

1. Load **real noisy fluorescence microscopy** (PAM mice E0771 Z-stack from the Mt Sinai Microscopy Core) — not synthetic noise on top of clean images.
2. Compute an **axial-averaged "clean reference"** from adjacent Z-slices (denoised proxy, not strict ground truth).
3. Try **four denoising methods on the same noisy input** and compare:
   - Gaussian baseline (classical, deterministic)
   - **CARE** trained from scratch (supervised: needs the clean reference)
   - **Noise2Void (N2V)** trained from scratch (self-supervised: no clean reference needed)
   - **Pre-trained CARE from the BioImage Model Zoo** (production weights via `bioimageio.core`)
4. Quantify each: PSNR, SSIM, hallucination map.
5. Apply the trained models to a held-out Z-slice and inspect the **production scenario** (no clean reference available — only the hallucination check tells you what to trust).
6. Walk the **integrity reporting** template.

**Why three AI methods instead of one.** They sit at three points on the data-requirement spectrum:

| Method | Needs clean reference? | Production scenario |
|---|---|---|
| CARE | **Yes** — paired noisy/clean training | You acquired both noisy and clean (e.g. averaged) versions |
| N2V | No — self-supervised on the noisy image | You only have noisy data, no clean ground truth |
| BiMZ pre-trained | No — uses published weights | Someone has already trained on similar fluorescence |

> ⚠️ **Integrity warning, applies to all four methods.** A "denoised" image is a model output, not a measurement. Hallucinated structure can look more real than real structure. The hallucination check + integrity template at the end gives you the workflow for honest reporting.

---""")


# ---------------------------------------------------------------------------
# §2 Setup
# ---------------------------------------------------------------------------
def section_setup(b):
    b.md("""## §2 Setup

One pip install + imports. The bioimageio.core install is optional (only Method 6d uses it); the cell will still complete if BiMZ install fails.""")

    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

%pip install --quiet torch torchvision scikit-image matplotlib numpy scipy pillow
# bioimageio.core is optional — only needed for the pre-trained BiMZ method.
# Failure here is non-fatal; Method 6d will fall back to N2V-from-scratch.
%pip install --quiet bioimageio.core 2> /dev/null || echo "bioimageio.core install failed; Method 6d will fall back."

import os, time, urllib.request, urllib.error, tempfile, traceback
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import data as skdata
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from scipy.ndimage import gaussian_filter

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch  : {torch.__version__}")
print(f"device : {device}")

# Shared results store — methods append here; comparison section reads from it.
results = {}
""")


# ---------------------------------------------------------------------------
# §3 Data decision
# ---------------------------------------------------------------------------
def section_data_decision(b):
    b.md("""---

## §3 Choose data source

Pick the tier in the form below. **MABC hosted** is the recommended default — real PAM mice E0771 fluorescence with axial-averaged clean reference. Fall-through chain on failure: MABC → Canonical → Synthetic.

| Tier | What it is | Clean reference |
|---|---|---|
| **MABC hosted (recommended)** | Real PAM mice E0771 Z-stack, baked from Mt Sinai imaging. 5 training pairs (256×256), 1 held-out test slice, plus the full 73-slice Z-stack for browsing. | Axial average of ±5 adjacent Z-slices (denoised proxy, not strict ground truth — adjacent Z shares some noise) |
| **Canonical** | scikit-image `cells3d()` confocal stack. Built into skimage, zero download. | Same axial-averaging trick on the cells3d stack. |
| **Synthetic** | Synthetic blobs + Poisson + Gaussian noise. Always works, even offline. | The original noise-free blobs (true ground truth, since we generated both). |""")

    b.code('''# @title Data source { run: "auto", display-mode: "form" }
DATA_SOURCE = "MABC hosted (recommended)"  # @param ["MABC hosted (recommended)", "Canonical (scikit-image cells3d)", "Synthetic (blobs + Poisson + Gaussian)"]
print("DATA_SOURCE:", DATA_SOURCE)
''')


# ---------------------------------------------------------------------------
# §4 Load + preview
# ---------------------------------------------------------------------------
def section_load_and_preview(b):
    b.md("""---

## §4 Load data + preview

The loader honors **DATA_SOURCE** and falls through automatically on failure. It produces:

- `noisy_train`, `clean_train` — paired training set (shape `(N, H, W)`)
- `test_noisy`, `test_clean_ref` — single held-out pair for the production-scenario test in §10
- `full_stack` — full Z-stack (3D) for browsing via the Z_INDEX slider below""")

    b.code('''MABC_03A_URL = "https://microscopy-core-ismms.github.io/ImageAnalysisCourse/data/mabc/03a_denoising_n2v.npz"

def _normalize_slice(a):
    a = np.asarray(a).astype(np.float32)
    p1, p99 = np.percentile(a, [1, 99])
    return np.clip((a - np.float32(p1)) / np.float32(max(p99 - p1, 1e-8)), 0, 1).astype(np.float32)


def _axial_pair_from_stack(stack_3d, train_z, test_z, window=5):
    """Given a 3D Z-stack (Z, H, W) of normalized floats in [0,1], pick training noisy slices
    + axial-averaged clean refs, plus a held-out test slice."""
    n_z = stack_3d.shape[0]
    def _avg(z):
        lo, hi = max(0, z - window), min(n_z, z + window + 1)
        return stack_3d[lo:hi].mean(axis=0).astype(np.float32)
    noisy_train_arr = np.stack([stack_3d[z].astype(np.float32) for z in train_z])
    clean_train_arr = np.stack([_avg(z) for z in train_z])
    return noisy_train_arr, clean_train_arr, stack_3d[test_z].astype(np.float32), _avg(test_z)


def load_data():
    """Returns dict with keys: noisy_train, clean_train, test_noisy, test_clean_ref,
    full_stack, source_str."""
    tier = globals().get("DATA_SOURCE", "MABC hosted (recommended)")

    if tier.startswith("MABC"):
        try:
            cache = os.path.join(tempfile.gettempdir(), "03a_pam.npz")
            if not os.path.exists(cache):
                print(f"Fetching MABC: {MABC_03A_URL}")
                urllib.request.urlretrieve(MABC_03A_URL, cache)
            d = np.load(cache, allow_pickle=True)
            # MABC npz contract: images, labels, test_noisy, test_clean_ref, full_stack (all uint8)
            return {
                "noisy_train":    np.stack([_normalize_slice(im) for im in d["images"]]),
                "clean_train":    np.stack([_normalize_slice(im) for im in d["labels"]]),
                "test_noisy":     _normalize_slice(d["test_noisy"]),
                "test_clean_ref": _normalize_slice(d["test_clean_ref"]),
                "full_stack":     np.stack([_normalize_slice(s) for s in d["full_stack"]]),
                "source_str":     "MABC: PAM mice E0771 Z-stack (real fluorescence, axial-averaged clean_ref)",
            }
        except (urllib.error.HTTPError, urllib.error.URLError, FileNotFoundError):
            print("MABC pack unavailable; falling through to canonical (cells3d).")
            tier = "Canonical"
        except Exception:
            print("MABC read failed unexpectedly; falling through to canonical.")
            traceback.print_exc(limit=1)
            tier = "Canonical"

    if tier.startswith("Canonical"):
        cells = skdata.cells3d()  # (60, 2, 256, 256) uint16, channels = (membrane, DAPI)
        ch_stack = np.stack([_normalize_slice(s) for s in cells[:, 1]])  # DAPI = noisier near edges
        n_train, c_train, t_noisy, t_clean = _axial_pair_from_stack(
            ch_stack, train_z=[5, 8, 11, 14, 17], test_z=25, window=3
        )
        return {
            "noisy_train": n_train, "clean_train": c_train,
            "test_noisy": t_noisy,  "test_clean_ref": t_clean,
            "full_stack": ch_stack,
            "source_str": "Canonical: cells3d DAPI channel, axial-averaged clean_ref",
        }

    # Synthetic last resort — true ground truth available.
    rng = np.random.default_rng(0)
    def make_clean(size=256, n_objects=15):
        img = np.zeros((size, size), dtype=np.float32)
        for _ in range(n_objects):
            cy, cx = rng.integers(20, size - 20, size=2)
            r = int(rng.integers(8, 16))
            YY, XX = np.ogrid[:size, :size]
            img[(YY - cy) ** 2 + (XX - cx) ** 2 <= r ** 2] = float(rng.uniform(0.5, 1.0))
        return gaussian_filter(img, sigma=1.0).astype(np.float32)
    def add_noise(img, photons=20):
        scaled = img * photons
        n = rng.poisson(scaled).astype(np.float32) / photons
        n = n + rng.normal(0, 0.05, n.shape).astype(np.float32)
        return np.clip(n, 0, 1).astype(np.float32)
    cleans = np.stack([make_clean() for _ in range(5)])
    noisys = np.stack([add_noise(c) for c in cleans])
    test_c = make_clean()
    test_n = add_noise(test_c)
    full = np.stack([add_noise(test_c) for _ in range(8)])  # synthetic "z-stack" = repeated noisy
    return {
        "noisy_train": noisys, "clean_train": cleans,
        "test_noisy": test_n,  "test_clean_ref": test_c,
        "full_stack": full,
        "source_str": "Synthetic: blobs + Poisson + Gaussian noise",
    }


data = load_data()
noisy_train    = data["noisy_train"]
clean_train    = data["clean_train"]
test_noisy     = data["test_noisy"]
test_clean_ref = data["test_clean_ref"]
full_stack     = data["full_stack"]

print(f"\\nLoaded data from: {data['source_str']}")
print(f"  noisy_train:    {noisy_train.shape} dtype={noisy_train.dtype} range=[{noisy_train.min():.2f}, {noisy_train.max():.2f}]")
print(f"  clean_train:    {clean_train.shape}")
print(f"  test_noisy:     {test_noisy.shape}")
print(f"  test_clean_ref: {test_clean_ref.shape}")
print(f"  full_stack:     {full_stack.shape} (Z-slices for browsing)")
''')

    b.md("""### §4.1 Preview the loaded training pairs

You're looking at one row per training pair: noisy input on the left, axial-averaged clean reference on the right. The two should be the same field of view; the difference is the smoothing from averaging over adjacent Z-slices.""")

    b.code("""n_pairs = noisy_train.shape[0]
fig, axes = plt.subplots(n_pairs, 2, figsize=(7, 2.7 * n_pairs))
if n_pairs == 1:
    axes = axes[None, :]
for i in range(n_pairs):
    axes[i, 0].imshow(noisy_train[i], cmap='gray', vmin=0, vmax=1)
    axes[i, 0].set_title(f"pair {i}: noisy"); axes[i, 0].axis('off')
    axes[i, 1].imshow(clean_train[i], cmap='gray', vmin=0, vmax=1)
    axes[i, 1].set_title(f"pair {i}: clean_ref (axial avg)"); axes[i, 1].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""### §4.2 Browse the full Z-stack

The training pairs came from specific Z-slices in a 3D acquisition. Use the slider below to scrub through the full stack and see how signal and noise vary with focus. Out-of-focus slices have lower SNR — that's why the training pairs were picked from the mid-stack.""")

    b.code('''# @title Browse Z-stack { run: "auto", display-mode: "form" }
Z_INDEX = 30  # @param {type:"slider", min:0, max:72, step:1}

z = min(Z_INDEX, full_stack.shape[0] - 1)
slc = full_stack[z]
mean_v, std_v = float(slc.mean()), float(slc.std())

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(slc, cmap='gray', vmin=0, vmax=1)
ax.set_title(f"Z = {z}/{full_stack.shape[0] - 1}\\nmean={mean_v:.2f}  std={std_v:.2f}")
ax.axis('off')
plt.tight_layout(); plt.show()
''')


# ---------------------------------------------------------------------------
# §5 Method definitions
# ---------------------------------------------------------------------------
def section_method_definitions(b):
    b.md("""---

## §5 Method definitions

Four denoisers, defined here once. The §6 cells below call them. The §7 "Run all" cell also calls them.

- `predict_gaussian(noisy)` — classical, instant
- `train_and_apply_care(noisy_train, clean_train, test_noisy)` — supervised TinyUNet
- `train_and_apply_n2v(noisy_train, test_noisy)` — self-supervised TinyUNet (center-pixel masking)
- `load_and_apply_bimz(test_noisy)` — `bioimageio.core` pre-trained model""")

    b.code('''class TinyUNet(nn.Module):
    """Same TinyUNet pattern as NB06. ~50k params."""
    def __init__(self, in_ch=1, out_ch=1, base=24):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, out_ch, 1))
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u = self.up(e2)
        return self.dec(torch.cat([u, e1], dim=1))


def predict_gaussian(noisy_2d, sigma=1.5):
    """Classical Gaussian-filter denoiser — fixed sigma."""
    return gaussian_filter(noisy_2d, sigma=sigma).astype(np.float32)


def train_and_apply_care(noisy_train, clean_train, test_noisy,
                         epochs=40, batch=2, lr=1e-3, verbose=True):
    """CARE: supervised TinyUNet trained on (noisy, clean) pairs.
    Returns (prediction_on_test_noisy, training_loss_curve)."""
    net = TinyUNet(in_ch=1, out_ch=1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(noisy_train, dtype=torch.float32).to(device)[:, None]
    Yt = torch.tensor(clean_train, dtype=torch.float32).to(device)[:, None]
    losses = []
    n = Xt.shape[0]
    if verbose:
        print(f"CARE: training {n} pairs, {epochs} epochs on {device}...")
    t0 = time.time()
    for ep in range(epochs):
        idx = torch.randperm(n)
        ep_loss = 0.0; steps = 0
        for k in range(0, n, batch):
            b_idx = idx[k:k + batch]
            pred = net(Xt[b_idx])
            loss = loss_fn(pred, Yt[b_idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); steps += 1
        losses.append(ep_loss / max(steps, 1))
        if verbose and (ep + 1) % 5 == 0:
            print(f"  epoch {ep + 1:3d}/{epochs}  loss {losses[-1]:.5f}")
    if verbose:
        print(f"CARE done in {time.time() - t0:.1f}s.")
    net.eval()
    with torch.no_grad():
        Xn = torch.tensor(test_noisy[None, None], dtype=torch.float32).to(device)
        pred = net(Xn).cpu().numpy().squeeze()
    return np.clip(pred, 0, 1).astype(np.float32), losses


def train_and_apply_n2v(noisy_train, test_noisy,
                        epochs=40, batch=2, lr=1e-3, mask_frac=0.02, verbose=True):
    """Noise2Void: self-supervised TinyUNet. Randomly mask center pixels with
    a neighbor's value; the network must predict the original. Trained only on noisy data."""
    net = TinyUNet(in_ch=1, out_ch=1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(noisy_train, dtype=torch.float32).to(device)[:, None]
    losses = []
    n, _, H, W = Xt.shape
    if verbose:
        print(f"N2V: training {n} noisy images (self-supervised), {epochs} epochs on {device}...")
    t0 = time.time()
    for ep in range(epochs):
        idx = torch.randperm(n)
        ep_loss = 0.0; steps = 0
        for k in range(0, n, batch):
            b_idx = idx[k:k + batch]
            x = Xt[b_idx]
            # Random "blind-spot" mask: pick mask_frac of pixels per image; replace with random neighbor.
            bs = x.shape[0]
            mask = torch.zeros_like(x, dtype=torch.bool)
            x_masked = x.clone()
            n_mask = max(1, int(mask_frac * H * W))
            for bi in range(bs):
                ys = torch.randint(1, H - 1, (n_mask,))
                xs = torch.randint(1, W - 1, (n_mask,))
                # Neighbor swap: pick a random neighbor in 3x3 (excluding center)
                dy = torch.randint(-1, 2, (n_mask,))
                dx = torch.randint(-1, 2, (n_mask,))
                dy[(dy == 0) & (dx == 0)] = 1
                ny = (ys + dy).clamp(0, H - 1)
                nx = (xs + dx).clamp(0, W - 1)
                x_masked[bi, 0, ys, xs] = x[bi, 0, ny, nx]
                mask[bi, 0, ys, xs] = True
            pred = net(x_masked)
            # Loss only on the masked positions: predict the ORIGINAL noisy pixel value from neighbors.
            loss = loss_fn(pred[mask], x[mask])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); steps += 1
        losses.append(ep_loss / max(steps, 1))
        if verbose and (ep + 1) % 5 == 0:
            print(f"  epoch {ep + 1:3d}/{epochs}  loss {losses[-1]:.5f}")
    if verbose:
        print(f"N2V done in {time.time() - t0:.1f}s.")
    net.eval()
    with torch.no_grad():
        Xn = torch.tensor(test_noisy[None, None], dtype=torch.float32).to(device)
        pred = net(Xn).cpu().numpy().squeeze()
    return np.clip(pred, 0, 1).astype(np.float32), losses


# Pinned BiMZ model ID. If lookup fails, this method falls back to N2V-from-scratch
# with a clear print message so the comparison table still gets a "BiMZ" row.
BIMZ_MODEL_ID = "affable-shark"  # Noise2Void model on bioimage.io


def load_and_apply_bimz(test_noisy, model_id=BIMZ_MODEL_ID, verbose=True):
    """Pre-trained denoising via bioimageio.core. Returns (prediction, source_str).
    Domain caveat: BiMZ N2V/CARE models are trained on specific modalities (e.g. DAPI);
    running on a different modality reduces denoising quality. Honest disclaimer below."""
    try:
        from bioimageio.core import load_description, predict
        if verbose:
            print(f"BiMZ: loading model '{model_id}' from bioimage.io ...")
        t0 = time.time()
        desc = load_description(model_id)
        # Convert (H, W) → (1, 1, H, W) for prediction; BiMZ accepts numpy or tensors.
        arr = test_noisy[None, None]  # batch, channel
        out = predict(model=desc, inputs=[arr.astype(np.float32)])
        if isinstance(out, (list, tuple)):
            out = out[0]
        # Coax various return types into a numpy array
        pred = np.asarray(out)
        # Squeeze to 2D
        while pred.ndim > 2:
            pred = pred[0]
        if verbose:
            print(f"BiMZ done in {time.time() - t0:.1f}s.")
        return np.clip(pred, 0, 1).astype(np.float32), f"BiMZ: pre-trained '{model_id}'"
    except Exception as e:
        print(f"BiMZ lookup or inference failed ({type(e).__name__}: {e}).")
        print("Falling back to N2V-from-scratch as a stand-in so the comparison still has a 4th row.")
        traceback.print_exc(limit=1)
        pred, _ = train_and_apply_n2v(noisy_train, test_noisy, verbose=False)
        return pred, "BiMZ unavailable → N2V-from-scratch fallback"


# Shared helpers for §6/§8/§9.
def compute_metrics(pred, clean_ref):
    p = np.asarray(pred, dtype=np.float32)
    c = np.asarray(clean_ref, dtype=np.float32)
    return {
        "psnr":        float(psnr_metric(c, p, data_range=1.0)),
        "ssim":        float(ssim_metric(c, p, data_range=1.0)),
        "anomaly_pct": 100.0 * float((np.abs(p - c) > 0.15).mean()),
    }


def show_triplet(noisy, pred, clean_ref, method_name):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(noisy,     cmap='gray', vmin=0, vmax=1); axes[0].set_title("noisy input")
    axes[1].imshow(pred,      cmap='gray', vmin=0, vmax=1); axes[1].set_title(f"denoised ({method_name})")
    axes[2].imshow(clean_ref, cmap='gray', vmin=0, vmax=1); axes[2].set_title("clean reference")
    for a in axes: a.axis('off')
    plt.tight_layout(); plt.show()


def show_loss(losses, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses, marker='o'); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
''')


# ---------------------------------------------------------------------------
# §6 Stepwise method execution (4 cells)
# ---------------------------------------------------------------------------
def section_method_6a_gaussian(b):
    b.md("""---

## §6a Gaussian baseline

Classical filter. Instant. No learning. Provides the non-AI reference number for the comparison table.""")

    b.code("""pred_gaussian = predict_gaussian(test_noisy, sigma=1.5)
metrics_g = compute_metrics(pred_gaussian, test_clean_ref)
results["Gaussian"] = {"pred": pred_gaussian, **metrics_g}

show_triplet(test_noisy, pred_gaussian, test_clean_ref, "Gaussian σ=1.5")
print(f"Gaussian:  PSNR={metrics_g['psnr']:.2f} dB  SSIM={metrics_g['ssim']:.3f}  anomaly={metrics_g['anomaly_pct']:.1f}%")""")


def section_method_6b_care(b):
    b.md("""---

## §6b CARE — trained from scratch (supervised, needs clean reference)

CARE = Content-Aware Restoration (Weigert et al. 2018, *Nat Methods*). Supervised: trained on paired (noisy → clean) examples. Uses the axial-averaged clean_ref as the regression target.

**~3 minutes on Colab T4** (CPU works but slower). Watch the loss curve drop — that's the network learning to reproduce the clean reference from the noisy input.""")

    b.code("""pred_care, loss_care = train_and_apply_care(noisy_train, clean_train, test_noisy)
metrics_c = compute_metrics(pred_care, test_clean_ref)
results["CARE (scratch)"] = {"pred": pred_care, **metrics_c}

show_loss(loss_care, "CARE training curve")
show_triplet(test_noisy, pred_care, test_clean_ref, "CARE (scratch)")
print(f"CARE:      PSNR={metrics_c['psnr']:.2f} dB  SSIM={metrics_c['ssim']:.3f}  anomaly={metrics_c['anomaly_pct']:.1f}%")""")


def section_method_6c_n2v(b):
    b.md("""---

## §6c Noise2Void — trained from scratch (self-supervised, no clean reference needed)

N2V (Krull et al. 2019, *CVPR*). Self-supervised: trained on the noisy image only, using the trick that noise is pixel-independent but signal is spatially correlated. The network masks center pixels with a random neighbor and learns to predict the original from context — effectively forced to denoise.

**~3 minutes on Colab T4.** Notably: this run **does not look at the clean reference at all** during training. The clean_ref is only used to score the result after the fact.""")

    b.code("""pred_n2v, loss_n2v = train_and_apply_n2v(noisy_train, test_noisy)
metrics_n = compute_metrics(pred_n2v, test_clean_ref)
results["N2V (scratch)"] = {"pred": pred_n2v, **metrics_n}

show_loss(loss_n2v, "N2V training curve (self-supervised loss on masked pixels)")
show_triplet(test_noisy, pred_n2v, test_clean_ref, "N2V (scratch)")
print(f"N2V:       PSNR={metrics_n['psnr']:.2f} dB  SSIM={metrics_n['ssim']:.3f}  anomaly={metrics_n['anomaly_pct']:.1f}%")""")


def section_method_6d_bimz(b):
    b.md("""---

## §6d Pre-trained CARE from the BioImage Model Zoo

Uses `bioimageio.core` to load a published model and run inference. **~30 seconds** including download. No training needed — someone else already did that.

> ⚠️ **Domain caveat.** BiMZ N2V/CARE models are trained on specific modalities (a particular cell line, a particular fluorophore). If the model wasn't trained on PAM-mice-like data, the denoising quality drops. The pinned model here (`affable-shark`) is a Noise2Void model from bioimage.io. If the lookup fails (BiMZ API churn, network), this cell falls back to N2V-from-scratch.""")

    b.code("""pred_bimz, bimz_source = load_and_apply_bimz(test_noisy)
metrics_b = compute_metrics(pred_bimz, test_clean_ref)
results["BiMZ pretrained"] = {"pred": pred_bimz, **metrics_b, "source": bimz_source}

show_triplet(test_noisy, pred_bimz, test_clean_ref, bimz_source)
print(f"BiMZ:      PSNR={metrics_b['psnr']:.2f} dB  SSIM={metrics_b['ssim']:.3f}  anomaly={metrics_b['anomaly_pct']:.1f}%")
print(f"  source: {bimz_source}")""")


# ---------------------------------------------------------------------------
# §7 Optional: run all methods at once
# ---------------------------------------------------------------------------
def section_run_all(b):
    b.md("""---

## §7 *Optional:* run all four methods at once

If you'd rather skip running each §6 cell individually, this one cell runs all four sequentially. Takes ~7 minutes on T4 (CARE + N2V dominate). Idempotent — re-running just overwrites the `results` dict.""")

    b.code("""# Run all four methods. Each populates `results[method_name]`.

# Method 1 — Gaussian
p = predict_gaussian(test_noisy, sigma=1.5)
results["Gaussian"] = {"pred": p, **compute_metrics(p, test_clean_ref)}
print(f"Gaussian:  PSNR={results['Gaussian']['psnr']:.2f} SSIM={results['Gaussian']['ssim']:.3f}")
show_triplet(test_noisy, p, test_clean_ref, "Gaussian σ=1.5")

# Method 2 — CARE
p, _ = train_and_apply_care(noisy_train, clean_train, test_noisy, verbose=False)
results["CARE (scratch)"] = {"pred": p, **compute_metrics(p, test_clean_ref)}
print(f"CARE:      PSNR={results['CARE (scratch)']['psnr']:.2f} SSIM={results['CARE (scratch)']['ssim']:.3f}")
show_triplet(test_noisy, p, test_clean_ref, "CARE (scratch)")

# Method 3 — N2V
p, _ = train_and_apply_n2v(noisy_train, test_noisy, verbose=False)
results["N2V (scratch)"] = {"pred": p, **compute_metrics(p, test_clean_ref)}
print(f"N2V:       PSNR={results['N2V (scratch)']['psnr']:.2f} SSIM={results['N2V (scratch)']['ssim']:.3f}")
show_triplet(test_noisy, p, test_clean_ref, "N2V (scratch)")

# Method 4 — BiMZ pretrained (with fallback)
p, src = load_and_apply_bimz(test_noisy, verbose=False)
results["BiMZ pretrained"] = {"pred": p, **compute_metrics(p, test_clean_ref), "source": src}
print(f"BiMZ:      PSNR={results['BiMZ pretrained']['psnr']:.2f} SSIM={results['BiMZ pretrained']['ssim']:.3f}")
show_triplet(test_noisy, p, test_clean_ref, src)
""")


# ---------------------------------------------------------------------------
# §8 Quantitative comparison
# ---------------------------------------------------------------------------
def section_comparison(b):
    b.md("""---

## §8 Quantitative comparison

The table below reads `results`. Methods you haven't run yet are skipped. Run §6a-d individually (or §7 in one shot) to populate all four rows.

**How to read the table.** PSNR higher is better; SSIM closer to 1 is better; anomaly % is the fraction of pixels where the denoised result differs from the clean reference by more than 0.15 in normalized intensity — *lower is better only if the clean reference is trustworthy*. For our axial-averaged proxy, take anomaly % as "structural disagreement", not "model failure".""")

    b.code("""# Build the 4-row comparison table
if not results:
    print("No methods run yet. Run §6a-d (or §7) first.")
else:
    print(f"\\n{'Method':<22} | {'PSNR (dB)':>10} | {'SSIM':>6} | {'Anomaly %':>10}")
    print('-' * 60)
    for name, r in results.items():
        print(f"{name:<22} | {r['psnr']:>10.2f} | {r['ssim']:>6.3f} | {r['anomaly_pct']:>9.1f}%")
    print()

    # Side-by-side grid: noisy / each method's prediction / clean reference
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods + 2, figsize=(3 * (n_methods + 2), 3.5))
    axes[0].imshow(test_noisy, cmap='gray', vmin=0, vmax=1); axes[0].set_title("noisy"); axes[0].axis('off')
    for i, (name, r) in enumerate(results.items()):
        axes[i + 1].imshow(r["pred"], cmap='gray', vmin=0, vmax=1)
        axes[i + 1].set_title(f"{name}\\nPSNR={r['psnr']:.1f}")
        axes[i + 1].axis('off')
    axes[-1].imshow(test_clean_ref, cmap='gray', vmin=0, vmax=1); axes[-1].set_title("clean ref"); axes[-1].axis('off')
    plt.tight_layout(); plt.show()
""")


# ---------------------------------------------------------------------------
# §9 Hallucination check
# ---------------------------------------------------------------------------
def section_hallucination(b):
    b.md("""---

## §9 Hallucination check — pick a method to inspect

PSNR and SSIM are global numbers. The hallucination check below shows you **where** the model invented or distorted structure. For each method, the residual `|pred − clean_ref|` thresholded at 0.15 gives a binary "anomaly" map.""")

    b.code('''# @title Pick a method to inspect { run: "auto", display-mode: "form" }
HALLUC_METHOD = "CARE (scratch)"  # @param ["Gaussian", "CARE (scratch)", "N2V (scratch)", "BiMZ pretrained"]

if HALLUC_METHOD not in results:
    print(f"'{HALLUC_METHOD}' has not been run yet. Run the corresponding §6 cell first.")
else:
    pred = results[HALLUC_METHOD]["pred"]
    diff = pred.astype(np.float32) - test_clean_ref.astype(np.float32)
    anomaly = np.abs(diff) > 0.15
    anomaly_pct = 100.0 * anomaly.mean()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(pred, cmap='gray', vmin=0, vmax=1); axes[0].set_title(f"{HALLUC_METHOD}: denoised")
    axes[0].axis('off')
    axes[1].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5); axes[1].set_title("pred − clean_ref")
    axes[1].axis('off')
    axes[2].imshow(pred, cmap='gray', vmin=0, vmax=1)
    axes[2].imshow(np.where(anomaly, 1, np.nan), cmap='autumn', alpha=0.6)
    axes[2].set_title(f"anomaly: {anomaly_pct:.1f}% (|diff|>0.15)")
    axes[2].axis('off')
    plt.tight_layout(); plt.show()
''')

    b.md("""**Reading the anomaly map.** Where the denoised image disagrees structurally with the clean reference, you're either:

- Over-smoothing (real fine structure lost) → systematic anomalies on edges
- Under-smoothing (noise survived) → speckle-like anomalies
- **Hallucinating** (model invented structure) → anomalies inside otherwise-uniform regions

For the axial-averaged clean reference: anomalies inside cells often reflect axial-averaging artifacts more than model errors. Take this measurement as advisory.""")


# ---------------------------------------------------------------------------
# §10 Held-out / production scenario
# ---------------------------------------------------------------------------
def section_held_out(b):
    b.md("""---

## §10 Production scenario — held-out tile, no clean reference

Until now you've been scoring against `test_clean_ref`. **In production you don't have a clean reference** — you only have new noisy acquisitions. The trained models can be applied to any new tile from the same modality, and the only feedback you get is the hallucination check against… nothing.

Below, we apply the CARE and N2V models you trained in §6b/c to a different held-out slice from the same Z-stack. There's no clean reference for this slice — that's the realistic case. You're left looking at the result and deciding whether to trust it.""")

    b.code("""# Use a second Z-slice from the full stack (not the test slice from §6).
held_out_z = min(50, full_stack.shape[0] - 1)
held_out = full_stack[held_out_z]

# Apply the two trained models. Gaussian and BiMZ are deterministic functions of input, so re-apply.
held_results = {}
held_results["Noisy"] = held_out
held_results["Gaussian σ=1.5"] = predict_gaussian(held_out, sigma=1.5)

# CARE / N2V — reuse the trained networks from §6b/c if they're in `results`.
# (For simplicity, re-train quickly if needed; in a real workflow you'd cache the weights.)
if "CARE (scratch)" in results:
    held_results["CARE (re-applied)"], _ = train_and_apply_care(noisy_train, clean_train, held_out, verbose=False)
if "N2V (scratch)" in results:
    held_results["N2V (re-applied)"], _ = train_and_apply_n2v(noisy_train, held_out, verbose=False)
if "BiMZ pretrained" in results:
    held_results["BiMZ (re-applied)"], _ = load_and_apply_bimz(held_out, verbose=False)

n = len(held_results)
fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
for ax, (name, im) in zip(axes, held_results.items()):
    ax.imshow(im, cmap='gray', vmin=0, vmax=1); ax.set_title(name); ax.axis('off')
plt.tight_layout(); plt.show()
print(f"Held-out slice = Z={held_out_z} from full_stack.")
print("No clean reference exists for this slice — you're looking at four candidate denoisings with no truth panel.")""")


# ---------------------------------------------------------------------------
# §11 Integrity reporting
# ---------------------------------------------------------------------------
def section_integrity_reporting(b):
    b.md("""---

## §11 Integrity reporting walkthrough

If you put a denoised figure in a paper, journals (Nature 2024, Cell 2025, eLife 2024) increasingly require disclosure of:

- **Model identity** (training data, architecture, version)
- **Hallucination metrics** (anomaly % vs clean reference if available; visual checks otherwise)
- **Domain match** (was the model trained on similar modality?)
- **Reproducibility** (training seed, hyperparameters, software versions)

The template below is a fill-in for that disclosure paragraph.""")

    b.code("""template = '''
AI Denoising Integrity Report
==============================

Method:            {method}
Source (training): {training_source}
Model architecture: {arch}
Pre-trained?       {pretrained}
Training data:     {training_data}

Quantitative metrics on clean reference:
  PSNR:           {psnr:.2f} dB
  SSIM:           {ssim:.3f}
  Anomaly %:      {anomaly_pct:.1f}%   (residual > 0.15 normalized intensity)

Clean reference origin: {clean_ref_origin}
  ⚠ Note: if 'axial-averaged neighbors', this is a denoised proxy, not strict ground truth.

Inference on new data (production):
  Held-out slice:  Z={held_out_z}
  Clean reference available?  No (production scenario)
  Reviewer must inspect visually; no automated hallucination score possible.

Hyperparameters / software:
  PyTorch: {torch_version}
  Device:  {device}
  Seeds:   numpy={np_seed}, torch={torch_seed}
'''

# Fill the template using whatever's in `results`. Run §6 cells first, then this cell.
if "CARE (scratch)" in results:
    r = results["CARE (scratch)"]
    print(template.format(
        method="CARE (scratch)",
        training_source="this notebook",
        arch="TinyUNet (~50k params)",
        pretrained="No",
        training_data="5 PAM mice E0771 Z-slices paired with axial-averaged neighbors",
        psnr=r['psnr'], ssim=r['ssim'], anomaly_pct=r['anomaly_pct'],
        clean_ref_origin="axial-averaged neighbors (Z ±5)",
        held_out_z=min(50, full_stack.shape[0] - 1),
        torch_version=torch.__version__,
        device=device,
        np_seed=0,
        torch_seed='default (no explicit torch.manual_seed)',
    ))
else:
    print("Run §6b first to populate the CARE row of `results`; the template fills from there.")""")


# ---------------------------------------------------------------------------
# §12 Going broader
# ---------------------------------------------------------------------------
def section_broader(b):
    b.md("""---

## §12 Going broader — community alternatives

Beyond the four methods above:

- **BM3D / NLM (classical, non-AI)** — `scikit-image.restoration.denoise_nl_means`. Excellent baselines, no learning required.
- **DnCNN, DRUNet** — classical CNN denoisers (Zhang et al. 2017, 2021). Generic, work on any grayscale modality.
- **Probabilistic N2V (PN2V)** — extends N2V with explicit pixel noise model. Better when noise model is known.
- **DivNoising** — variational approach giving multiple denoised hypotheses per input. Useful when uncertainty matters.
- **Self-supervised CARE** — train CARE without paired data via a teacher-student trick.
- **Diffusion-based denoising** — Baikal (Schmid 2024 preprint), unpaired diffusion denoising of fluorescence.

**Where to look for pre-trained models:**

- [BioImage Model Zoo](https://bioimage.io/) — collated CARE/N2V/StarDist/Cellpose models with bioimage.io DOIs
- [CSBDeep examples](https://csbdeep.bioimagecomputing.com/doc/index.html) — CARE/N2V Python framework with example datasets
- [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) — Colab notebooks pairing model + training pipeline for many denoising methods

**Where to look for real noisy-clean paired datasets:**

- [Hagen et al. GigaDB 100888](http://gigadb.org/dataset/100888) — paired fluorescence noisy + clean (averaged), gold-standard benchmark (~50 GB, too large for inline Colab)
- [BioImage Archive](https://www.ebi.ac.uk/bioimage-archive/) — DOI-stable studies, search "denoising"
- This notebook's PAM mice E0771 acquisition is a Mt Sinai dataset — for similar Mt Sinai data, contact the MABC.""")


# ---------------------------------------------------------------------------
# §13 Closing reflection
# ---------------------------------------------------------------------------
def section_closing(b):
    b.md("""---

## §13 Closing reflection

You ran four denoising methods on the same real noisy fluorescence input and compared them quantitatively.

**Key takeaways:**

1. **Gaussian baseline is hard to beat for small-σ noise.** It's also the only method with zero hallucination risk — it can only over- or under-smooth.

2. **CARE wins when you have paired clean references.** The supervised loss tells the network exactly what to produce.

3. **N2V wins when you don't.** Surprisingly close to CARE in many cases, while requiring nothing but the noisy data itself.

4. **Pre-trained BiMZ models are convenient when the modality matches.** Otherwise they're not magic — domain mismatch produces bad denoising fast.

5. **The hallucination check matters most for the production scenario** (§10) where you have no clean reference. The clean-reference comparison in §8/§9 is your only sanity check for whether the trained model is denoising or distorting.

6. **Reporting integrity is non-optional.** Whatever method you ship in a figure, the §11 template gets you most of the way to a disclosure that a referee won't bounce.

---

**Next:** Lab 4 (NB04 — community platforms) shows you how to *find* models like the ones in §6d for your own modality. Lab 6 (NB06 — virtual staining) takes the same architecture and uses it for prediction across modalities instead of restoration within one.""")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def main():
    b = CellBuilder("nb03a")
    section_title(b)
    section_setup(b)
    section_data_decision(b)
    section_load_and_preview(b)
    section_method_definitions(b)
    section_method_6a_gaussian(b)
    section_method_6b_care(b)
    section_method_6c_n2v(b)
    section_method_6d_bimz(b)
    section_run_all(b)
    section_comparison(b)
    section_hallucination(b)
    section_held_out(b)
    section_integrity_reporting(b)
    section_broader(b)
    section_closing(b)
    build_notebook(b.cells, "03a_denoising_n2v")


if __name__ == "__main__":
    main()
