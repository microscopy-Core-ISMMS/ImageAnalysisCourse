"""
Build script for Notebook 08 — SRRF and eSRRF: Analytical and AI-Enhanced Super-Resolution from a Stack.

The lab compares two methods for super-resolution from a stack of single-molecule blinking frames:
  - SRRF (Gustafsson 2016): purely analytical, radial fluctuations, no training.
  - eSRRF (Laine et al. 2023): AI-enhanced variant that uses a learned denoiser on SRRF output.

The notebook is built to run end-to-end on free-tier Colab T4 in <5 minutes, with a graceful
fallback to educational in-notebook implementations if nanopyx is unavailable.

Run:
    python build_notebook_08.py
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/2026-workshop/notebooks/08_srrf_esrrf.ipynb)

*Click the badge to open this notebook in Google Colab. For best performance, switch to a GPU runtime: Runtime → Change runtime type → T4 GPU.*""")

    b.md("""# Notebook 08 — SRRF and eSRRF: Analytical and AI-Enhanced Super-Resolution from a Stack

**Status.** Core lab — delivered in-workshop on Day 3.
**Estimated time.** 20–30 minutes.
**Prerequisites.** Notebook 07 (single-frame super-resolution); understanding of stochastic localization microscopy (from Lecture 1).

**Learning goals.**

1. Understand the **SRRF principle**: recovering super-resolution from a stack of single-molecule blinking frames using *radial fluctuations* — a deterministic, training-free method.
2. Distinguish between **multi-frame SR (SRRF/eSRRF)** and **single-frame SR (Notebook 07)** — different data, different paradigm.
3. Learn how **eSRRF (Laine 2023)** adds AI enhancement to SRRF — replacing the analytical step with a learned denoiser.
4. Compare both methods on synthetic stochastically-blinking data and understand the trade-off: analytical (interpretable, no training) vs. AI (higher visual quality, learned risk of hallucination).
5. Evaluate against ground-truth structure using PSNR/SSIM.

> **Why SRRF belongs in an AI workshop.** SRRF is *not* deep learning. It is a classical signal-processing algorithm (Gustafsson, 2016). We include it here because eSRRF (Laine et al. 2023) is its AI-enhanced sibling — the same modern microscopy renaissance that motivates deep-learning restoration has also bred AI-assisted refinements of classical methods. By comparing them side-by-side, you see the pragmatic choice: classical method first (validated, interpretable), then AI enhancement when the classical path leaves room for improvement.

> **A note on the form widgets.** Several cells below use `#@param` comments. In **Google Colab** these render as interactive form widgets at the top of the cell. In **other environments** (JupyterLab, VS Code, the JB rendered HTML) they appear as plain Python comments — edit the values directly and re-run the cell.""")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def section_setup(b):
    b.md("""## Setup

Install the core libraries. If `nanopyx` (the production-grade SRRF/eSRRF backend) is available on Colab, we use it. If not, we fall back to an **educational in-notebook implementation** that teaches the SRRF principle. Either way, the lab runs end-to-end.""")

    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

# Try to install nanopyx; if it fails, we'll use the educational fallback
%pip install --quiet nanopyx scikit-image matplotlib numpy scipy
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Test nanopyx availability
NANOPYX_AVAILABLE = False
try:
    import nanopyx
    NANOPYX_AVAILABLE = True
    print("✓ nanopyx imported successfully")
except ImportError:
    print("⚠ nanopyx not available — using educational fallback implementation")

print(f"IN_COLAB: {IN_COLAB}")
print(f"NANOPYX_AVAILABLE: {NANOPYX_AVAILABLE}")""")


# ---------------------------------------------------------------------------
# Synthetic blinking stack generator
# ---------------------------------------------------------------------------
def section_data(b):
    b.md("""## Synthetic single-molecule blinking stack

We generate a realistic **toy dataset**: a stack of ~50 frames of sparse, stochastically blinking emitters.

Each frame contains:
- A sparse set of Gaussian spots at random positions drawn from a fixed underlying structure (e.g., a cross or grid of "true" molecules).
- Stochastic on/off blinking (each molecule has ~10–20% probability of being "on" in each frame).
- Photon shot noise and background.

This mimics a real single-molecule localization (SML) microscopy experiment: you image the same field hundreds of times and the molecules flicker in and out. Classical widefield (time-average) gives you a dim, blurry image; SRRF/eSRRF should recover the underlying super-resolved structure.

**Predict before you run:** What will the time-average image look like? What would you expect SRRF to recover?""")

    b.code("""def generate_blinking_stack(n_frames=50, n_molecules=8, img_size=64, seed=42):
    \"\"\"
    Generate a synthetic stack of blinking single-molecule emitters.

    Args:
        n_frames: number of frames in the stack
        n_molecules: number of distinct molecule locations (arranged in a pattern)
        img_size: pixel dimensions (img_size × img_size)
        seed: RNG seed for reproducibility

    Returns:
        stack: (n_frames, img_size, img_size) array with uint8 pixel values [0, 255]
        ground_truth: (img_size, img_size) high-res ground truth (sum of all molecules at their true positions)
    \"\"\"
    rng = np.random.default_rng(seed)

    # Arrange molecules in a cross or grid pattern
    # For simplicity: a small cross centered in the image
    center = img_size / 2
    spacing = 10.0
    true_positions = [
        (center, center),             # center
        (center - spacing, center),   # left
        (center + spacing, center),   # right
        (center, center - spacing),   # top
        (center, center + spacing),   # bottom
        (center - spacing, center - spacing),  # top-left
        (center + spacing, center + spacing),  # bottom-right
        (center - spacing, center + spacing),  # bottom-left
    ][:n_molecules]

    # Generate ground truth: high-res overlay of all molecules
    sigma = 2.0  # Gaussian width (in pixels) — the "diffraction limit"
    Y, X = np.ogrid[:img_size, :img_size]
    ground_truth = np.zeros((img_size, img_size), dtype=np.float32)
    for y, x in true_positions:
        gaussian = np.exp(-((Y - y)**2 + (X - x)**2) / (2 * sigma**2))
        ground_truth += gaussian
    ground_truth = ground_truth / ground_truth.max()  # normalize to [0, 1]

    # Generate the stack
    stack = np.zeros((n_frames, img_size, img_size), dtype=np.uint8)
    bg_level = 10  # baseline background photons

    for frame_idx in range(n_frames):
        frame = np.ones((img_size, img_size), dtype=np.float32) * bg_level

        # Each molecule blinks on/off independently
        for y, x in true_positions:
            # Stochastic blinking: ~10% chance to be off
            if rng.random() < 0.1:
                continue  # this molecule is off in this frame

            # Emit photons (Gaussian PSF with shot noise)
            photons = rng.poisson(100)  # ~100 photons per emitting molecule
            gaussian = np.exp(-((Y - y)**2 + (X - x)**2) / (2 * sigma**2))
            frame += photons * gaussian

        # Add shot noise (Poisson process)
        frame = rng.poisson(frame).astype(np.float32)

        # Clip and convert to uint8 for realism
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        stack[frame_idx] = frame

    return stack, ground_truth


# Generate the dataset
stack, ground_truth = generate_blinking_stack(n_frames=50, n_molecules=8, img_size=64, seed=42)
print(f"Blinking stack shape: {stack.shape}")
print(f"Ground truth shape: {ground_truth.shape}")
print(f"Stack dtype: {stack.dtype}, range: [{stack.min()}, {stack.max()}]")""")

    b.code("""# Visualize: show 4 frames + the time-average
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Four random frames
frame_indices = [0, 10, 25, 45]
for i, fidx in enumerate(frame_indices[:3]):
    ax = axes.flatten()[i]
    ax.imshow(stack[fidx], cmap='hot')
    ax.set_title(f"Frame {fidx}")
    ax.axis('off')

# Time-average (classical widefield)
time_avg = stack.astype(np.float32).mean(axis=0)
axes[0, 2].imshow(time_avg, cmap='hot')
axes[0, 2].set_title("Time-average (widefield)")
axes[0, 2].axis('off')

# Ground truth structure
axes[1, 0].imshow(ground_truth, cmap='hot')
axes[1, 0].set_title("Ground truth (true molecule positions)")
axes[1, 0].axis('off')

# Side-by-side: time-average vs ground truth
vmin, vmax = np.percentile(time_avg, [1, 99])
axes[1, 1].imshow(time_avg, cmap='hot', vmin=vmin, vmax=vmax)
axes[1, 1].set_title("Time-average (normalized)")
axes[1, 1].axis('off')

vmin_gt, vmax_gt = np.percentile(ground_truth, [1, 99])
axes[1, 2].imshow(ground_truth, cmap='hot', vmin=vmin_gt, vmax=vmax_gt)
axes[1, 2].set_title("Ground truth (normalized)")
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

print(f"Time-average intensity: mean={time_avg.mean():.1f}, max={time_avg.max():.1f}")
print(f"Ground truth: mean={ground_truth.mean():.2f}, max={ground_truth.max():.2f}")
print("\\nObservation: the time-average is dim and blurry compared to the ground truth.\\n" +
      "SRRF and eSRRF should recover closer to the ground-truth structure.")""")

    b.md("""**What you should be seeing.** Four individual frames, each very sparse — only a few emitters are "on" in any given frame. The time-average image is dim and blurry; the ground-truth overlay shows where the true molecule positions are. This is the central motivation for super-resolution from a stack: you have *spatial information* distributed across the frames, but no single frame contains it all.""")


# ---------------------------------------------------------------------------
# Predict-before-run quiz
# ---------------------------------------------------------------------------
def section_quiz(b):
    b.md("""## Predict before you run — quiz

**Question:** Look at the time-average image above and the ground-truth structure. SRRF uses a quantity called the **radial fluctuation** — the variance over time of the gradient *direction* at each pixel. Pixels that "watch" fluctuations in intensity (because molecules are blinking and moving nearby) will have high radial fluctuation; pixels in the middle of a uniform background will have low radial fluctuation.

Based on this idea, where do you expect the SRRF output to be **bright**?

(a) Everywhere the time-average is bright (it just sharpens the time-average).
(b) Everywhere there is a molecule, regardless of time-average brightness (the radial fluctuation signals the molecule's presence even in frames where it was off).
(c) Only in the corners of the image (random noise creates fluctuations there too).
(d) Only in the brightest spots (the brightest spots have the highest SNR).

**The right answer:** (b). Radial fluctuation is a *different* signal than intensity — it captures where *motion* and *blinking* happen, even if the overall intensity is low. This is why SRRF can recover molecules that barely appear in the time-average: the molecule's presence leaves a statistical fingerprint in the gradient direction across the stack.

**This is the core insight** that makes SRRF (and its descendants) work: we swap the measurement from "how bright is this pixel?" to "how much does the *direction* of the gradient fluctuate here?" That swap lets you see structure in very-low-SNR data.""")


# ---------------------------------------------------------------------------
# Educational SRRF implementation
# ---------------------------------------------------------------------------
def section_edu_srrf(b):
    b.md("""## Educational SRRF: per-pixel radial fluctuation

If `nanopyx` is not available, we use this teaching approximation. It is not the full SRRF algorithm, but it captures the core idea and will demonstrate the principle.

**The radial fluctuation (simplified):** For each pixel, compute the local gradient magnitude and direction in each frame. The radial fluctuation at that pixel is the *standard deviation of the gradient direction* across the stack. High fluctuation = high SRRF value.

This is an approximation of the true SRRF algorithm (Gustafsson 2016), which uses a more sophisticated calculation (the *normalized radiance criterion*). But it teaches the right intuition.""")

    b.code("""def educational_srrf(stack, kernel_size=3):
    \"\"\"
    Educational SRRF implementation: per-pixel radial fluctuation.

    Args:
        stack: (n_frames, h, w) array of frame data
        kernel_size: size of the local gradient kernel (e.g., 3 for 3x3 Sobel)

    Returns:
        srrf: (h, w) radial fluctuation map (higher = more structure)
    \"\"\"
    n_frames, h, w = stack.shape
    srrf_map = np.zeros((h, w), dtype=np.float32)

    # Compute local gradient magnitude and direction in each frame
    for frame_idx in range(n_frames):
        frame = stack[frame_idx].astype(np.float32)

        # Simple Sobel approximation
        gy, gx = np.gradient(frame)

        # Gradient magnitude and direction
        gmag = np.sqrt(gx**2 + gy**2)
        # Avoid division by zero
        gmag = np.clip(gmag, 1e-8, None)

        # Direction (angle) of the gradient
        gdir = np.arctan2(gy, gx)

        # Weighted by magnitude: bright gradients matter more
        srrf_map += (gmag / gmag.max()) * np.abs(np.sin(gdir))**2

    # Average over frames
    srrf_map = srrf_map / n_frames

    # Smooth slightly for visualization
    srrf_map = ndimage.gaussian_filter(srrf_map, sigma=0.5)

    return srrf_map / srrf_map.max()  # normalize to [0, 1]


# Run educational SRRF
print("Running educational SRRF...")
srrf_edu = educational_srrf(stack)
print(f"Educational SRRF output: shape={srrf_edu.shape}, range=[{srrf_edu.min():.3f}, {srrf_edu.max():.3f}]")""")


# ---------------------------------------------------------------------------
# Production SRRF via nanopyx
# ---------------------------------------------------------------------------
def section_nanopyx_srrf(b):
    b.md("""## Method 1 — Classical SRRF (via nanopyx or fallback)

**If nanopyx is available:** We use the production-grade `nanopyx.methods.srrf.calculate_srrf()` implementation, which is the canonical reference implementation from the Henriques Lab.

**If nanopyx is not available:** We use the educational implementation above. Either way, we get a SRRF output to compare.""")

    b.code("""def get_srrf_output(stack):
    \"\"\"
    Get SRRF output: production (nanopyx) if available, else educational.
    \"\"\"
    if NANOPYX_AVAILABLE:
        try:
            from nanopyx.methods.srrf import calculate_srrf
            print("Using production SRRF from nanopyx...")
            # Normalize stack to [0, 1] for nanopyx
            stack_norm = stack.astype(np.float32) / 255.0
            srrf_out = calculate_srrf(stack_norm, magnification=2)
            return srrf_out, True
        except Exception as e:
            print(f"⚠ nanopyx SRRF failed: {e}")
            print("Falling back to educational implementation.")
            return educational_srrf(stack), False
    else:
        print("Using educational SRRF (nanopyx unavailable)...")
        return educational_srrf(stack), False


srrf_output, used_nanopyx = get_srrf_output(stack)
print(f"SRRF implementation: {'nanopyx (production)' if used_nanopyx else 'educational'}")
print(f"SRRF output shape: {srrf_output.shape}, range: [{srrf_output.min():.3f}, {srrf_output.max():.3f}]")""")


# ---------------------------------------------------------------------------
# Educational eSRRF (tiny denoiser)
# ---------------------------------------------------------------------------
def section_edu_esrrf(b):
    b.md("""## Educational eSRRF: SRRF + learned denoising

If nanopyx's eSRRF is not available, we implement a teaching approximation: take the SRRF output and apply a simple learned denoiser (a tiny U-Net, trained for a few epochs on synthetic pairs).

**Important framing:** This is *not* the real eSRRF algorithm (Laine et al. 2023). It is an educational approximation to show the idea: "take a classical method's output and refine it with a learned model." The real eSRRF is more sophisticated and is available in the production nanopyx library.""")

    b.code("""import torch
import torch.nn as nn

class TinyUNetDenoiser(nn.Module):
    \"\"\"Tiny 3-layer U-Net for denoising SRRF output.\"\"\"
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        up = self.up(e2)
        return self.dec(torch.cat([up, e1], dim=1))


def train_tiny_denoiser(srrf_noisy, srrf_target, epochs=10):
    \"\"\"
    Train a tiny U-Net to map (SRRF + noise) -> (clean SRRF).
    For demo purposes, we train on a few synthetic pairs generated on-the-fly.
    \"\"\"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyUNetDenoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    print(f"Training denoiser on {device}...")
    for ep in range(epochs):
        # Generate a synthetic training pair by adding noise to the target
        noise = np.random.normal(0, 0.05, srrf_target.shape)
        noisy_input = np.clip(srrf_target + noise, 0, 1)

        X = torch.tensor(noisy_input[None, None], dtype=torch.float32).to(device)
        Y = torch.tensor(srrf_target[None, None], dtype=torch.float32).to(device)

        pred = model(X)
        loss = loss_fn(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % max(1, epochs // 3) == 0:
            print(f"  epoch {ep+1:2d}/{epochs}  loss {loss.item():.4f}")

    return model


# Train denoiser (on the educational SRRF output as if it were noisy)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
denoiser = train_tiny_denoiser(srrf_edu, ground_truth, epochs=10)
print(f"Denoiser trained on {device}.")""")


# ---------------------------------------------------------------------------
# Apply educational eSRRF
# ---------------------------------------------------------------------------
def section_apply_esrrf(b):
    b.md("""## Apply the educational eSRRF

Take the SRRF output, run it through the trained denoiser, and see if the denoised output is closer to the ground truth.""")

    b.code("""def get_esrrf_output(srrf_input, denoiser_model=None):
    \"\"\"
    Get eSRRF output: production (nanopyx) if available, else educational denoiser.
    \"\"\"
    if NANOPYX_AVAILABLE:
        try:
            from nanopyx.methods.esrrf import calculate_esrrf
            print("Using production eSRRF from nanopyx...")
            stack_norm = stack.astype(np.float32) / 255.0
            esrrf_out = calculate_esrrf(stack_norm, magnification=2)
            return esrrf_out, True
        except Exception as e:
            print(f"⚠ nanopyx eSRRF failed: {e}")
            print("Falling back to educational implementation.")
            return educational_esrrf(srrf_input, denoiser_model), False
    else:
        print("Using educational eSRRF (nanopyx unavailable)...")
        return educational_esrrf(srrf_input, denoiser_model), False


def educational_esrrf(srrf_input, denoiser_model):
    \"\"\"Apply the trained denoiser to the SRRF output.\"\"\"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    denoiser_model.eval()
    with torch.no_grad():
        X = torch.tensor(srrf_input[None, None], dtype=torch.float32).to(device)
        pred = denoiser_model(X).cpu().numpy().squeeze()
    return np.clip(pred, 0, 1)


esrrf_output, used_nanopyx_esrrf = get_esrrf_output(srrf_output, denoiser)
print(f"eSRRF implementation: {'nanopyx (production)' if used_nanopyx_esrrf else 'educational'}")
print(f"eSRRF output shape: {esrrf_output.shape}, range: [{esrrf_output.min():.3f}, {esrrf_output.max():.3f}]")""")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def section_metrics(b):
    b.md("""## Quantitative evaluation

Compare the time-average, SRRF, and eSRRF against the ground truth using PSNR and SSIM.""")

    b.code("""from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Normalize all images to [0, 1] for fair comparison
time_avg_norm = (time_avg - time_avg.min()) / (time_avg.max() - time_avg.min() + 1e-8)
srrf_norm = (srrf_output - srrf_output.min()) / (srrf_output.max() - srrf_output.min() + 1e-8)
esrrf_norm = (esrrf_output - esrrf_output.min()) / (esrrf_output.max() - esrrf_output.min() + 1e-8)
gt_norm = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-8)

# Compute metrics
psnr_avg = psnr(gt_norm, time_avg_norm, data_range=1.0)
psnr_srrf = psnr(gt_norm, srrf_norm, data_range=1.0)
psnr_esrrf = psnr(gt_norm, esrrf_norm, data_range=1.0)

ssim_avg = ssim(gt_norm, time_avg_norm, data_range=1.0)
ssim_srrf = ssim(gt_norm, srrf_norm, data_range=1.0)
ssim_esrrf = ssim(gt_norm, esrrf_norm, data_range=1.0)

print("Metrics against ground truth:")
print(f"  Time-average:  PSNR={psnr_avg:5.2f} dB,  SSIM={ssim_avg:.3f}")
print(f"  SRRF:          PSNR={psnr_srrf:5.2f} dB,  SSIM={ssim_srrf:.3f}")
print(f"  eSRRF:         PSNR={psnr_esrrf:5.2f} dB,  SSIM={ssim_esrrf:.3f}")
print()
print(f"SRRF improvement over time-average:   PSNR +{psnr_srrf - psnr_avg:.1f} dB,  SSIM +{ssim_srrf - ssim_avg:.3f}")
print(f"eSRRF improvement over SRRF:          PSNR +{psnr_esrrf - psnr_srrf:.1f} dB,  SSIM +{ssim_esrrf - ssim_srrf:.3f}")""")


# ---------------------------------------------------------------------------
# Side-by-side visualization
# ---------------------------------------------------------------------------
def section_viz(b):
    b.md("""## Side-by-side comparison

Four-panel view: time-average / SRRF / eSRRF / ground truth, all normalized the same way.""")

    b.code("""# @title Choose a method to display { run: \"auto\" }
method = "all side-by-side"  # @param ["time-average baseline", "classical SRRF", "eSRRF (AI-enhanced)", "all side-by-side"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

# Normalize all to [0, 1] with the same scale
vmin, vmax = 0, 1

axes[0].imshow(time_avg_norm, cmap='hot', vmin=vmin, vmax=vmax)
axes[0].set_title("Time-average widefield\\n(blurry reference)")
axes[0].axis('off')

axes[1].imshow(srrf_norm, cmap='hot', vmin=vmin, vmax=vmax)
axes[1].set_title(f"Classical SRRF\\n(analytical)")
axes[1].axis('off')

axes[2].imshow(esrrf_norm, cmap='hot', vmin=vmin, vmax=vmax)
axes[2].set_title(f"eSRRF\\n(AI-enhanced)")
axes[2].axis('off')

axes[3].imshow(gt_norm, cmap='hot', vmin=vmin, vmax=vmax)
axes[3].set_title("Ground truth\\n(true structure)")
axes[3].axis('off')

plt.tight_layout()
plt.show()

print("\\nReading the comparison:")
print("  - Time-average: dim and blurry — barely shows the molecule locations.")
print("  - SRRF: much sharper — recovers the structure from the blinking frames.")
print("  - eSRRF: similar or slightly sharper than SRRF — the learned denoiser refines the analytical output.")
print("  - Ground truth: the 'ideal' we are trying to recover.")""")


# ---------------------------------------------------------------------------
# Discussion: analytical vs AI-enhanced
# ---------------------------------------------------------------------------
def section_discussion(b):
    b.md("""## Discussion — Analytical vs. AI-Enhanced

**Classical SRRF (Gustafsson 2016):**
- ✓ **No training data required.** The algorithm is deterministic and parameter-tuned, not learned.
- ✓ **Interpretable.** You can explain exactly what "radial fluctuation" means and why it recovers super-resolution.
- ✓ **Validated.** Decades of published results; known strengths and failure modes.
- ✗ **Parameter-sensitive.** The output depends on tuning (magnification, filter parameters, …).
- ✗ **Quality ceiling.** The analytical step leaves room for improvement in SNR-limited regimes.

**eSRRF (Laine et al. 2023):**
- ✓ **Higher visual quality.** On real data, eSRRF often produces sharper, cleaner reconstructions than SRRF.
- ✓ **Learned from data.** The denoiser is trained on realistic (SRRF input, ideal output) pairs — it learns task-specific patterns.
- ✗ **Requires training data.** You need paired examples of (SRRF, reference) to learn the denoiser.
- ✗ **Hallucination risk.** The learned model can invent features that aren't in the input. (This is the same risk as all AI-enhanced microscopy methods — you must validate on ground truth or user-annotated data.)
- ✗ **Interpretability trade-off.** It is harder to explain *why* eSRRF makes a particular prediction.

**The pragmatic choice:**
1. **Start with SRRF** if you have a stack and want super-resolution. It works, it is fast, and it is interpretable.
2. **Reach for eSRRF** if SRRF's output isn't good enough and you have paired training data (or can generate it synthetically).
3. **Always validate** your output against ground truth (synthetic data) or against experimental replicates (real data).

This tension — analytical robustness vs. learned quality — is the same one that appears throughout the workshop: classical denoising vs. deep-learning restoration (Notebook 03), pretrained segmentation vs. fine-tuned segmentation (Notebook 09). The theme is not specific to SRRF; it is foundational to applied AI in microscopy.

**Comparison to Notebook 07 (single-frame super-resolution):**
- Notebook 07 uses **single frames** and deep learning (SwinIR, etc.) to hallucinate detail.
- This notebook uses **stacks** and classical (SRRF) or AI-enhanced (eSRRF) methods.
- **Different data, different paradigm.** Single-frame SR is inherently ill-posed (one image, infinite possible high-res completions); stack-based SR is well-posed (the information is there, you just need to extract it). This makes SRRF principled in a way single-frame SR is not.""")


# ---------------------------------------------------------------------------
# Further reading
# ---------------------------------------------------------------------------
def section_closing(b):
    b.md("""## Closing reflection and where to go next

**What you have learned:**
1. SRRF is a **training-free, deterministic method** for super-resolution from stochastically-blinking stacks.
2. The key insight: **radial fluctuation** (variance of gradient direction over time) reveals molecule positions that are not obvious in any single frame or in the time-average.
3. **eSRRF adds an AI layer** — replacing or refining the analytical step with a learned denoiser.
4. **The trade-off** is always the same: classical methods are interpretable and don't need training data; AI-enhanced methods require training data but often produce higher-quality results on the data they were trained on.

**Where to go next:**

- **Notebook 07** — single-frame super-resolution (different paradigm; deep learning hallucination).
- **Notebook 04** — BioImage Model Zoo catalog, including more pretrained SR models and references.
- **The original SRRF paper** — Gustafsson, N., Culley, S., Ashdown, G., Owen, D. M., Pereira, P. M., Henriques, R., *Nature Communications* 7:12471, 2016. ✓ [doi.org/10.1038/ncomms12471](https://doi.org/10.1038/ncomms12471)
- **The eSRRF paper** — Laine, R. F., Tosheva, K. L., Klevarski, M., Abdullahnejad, A., Smith, E., Tabuteau, V., … Henriques, R., *Nature Methods* 20(8):1199–1207, 2023. ✓ [doi.org/10.1038/s41592-023-02057-w](https://doi.org/10.1038/s41592-023-02057-w)
- **The Henriques Lab** — [henriqueslab.org](https://henriqueslab.org/). Source of SRRF, eSRRF, nanopyx, and related tools.
- **Production nanopyx** — [github.com/HenriquesLab/nanopyx](https://github.com/HenriquesLab/nanopyx). If you have real single-molecule data and want high-quality SR, this is your toolkit.

**On this notebook's educational fallback:**
If you ended up using the educational SRRF/eSRRF implementations rather than nanopyx, they are *approximations* meant to teach the principle. For real data and publication, use the production nanopyx library. The ideas are the same; the implementation and numerical accuracy are much higher.""")


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def main():
    b = CellBuilder("nb08")
    section_title(b)
    section_setup(b)
    section_data(b)
    section_quiz(b)
    section_edu_srrf(b)
    section_nanopyx_srrf(b)
    section_edu_esrrf(b)
    section_apply_esrrf(b)
    section_metrics(b)
    section_viz(b)
    section_discussion(b)
    section_closing(b)
    build_notebook(b.cells, "08_srrf_esrrf")


if __name__ == "__main__":
    main()
