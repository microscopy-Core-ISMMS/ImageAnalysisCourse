"""
Build script for Notebook 15 — Diffusion Models in Bioimage (Frontier Demo).

Frontier-status notebook covering the diffusion model concept, forward/reverse
processes, and two pedagogical demonstrations:
  1. Forward diffusion visualization (analytical noise process)
  2. Reverse diffusion with a small pretrained model from HuggingFace
     (tries google/ddpm-cifar10-32, falls back to tiny in-line training if download fails)

This is orientation for a fast-moving research area, not a "use today" guide.

Run:
    python build_notebook_15.py
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/2026-workshop/notebooks/15_diffusion_models.ipynb)

*Click the badge to open this notebook in Google Colab. For best performance, switch to a GPU runtime: Runtime → Change runtime type → T4 GPU.*""")

    b.md("""# Notebook 15 — Diffusion Models in Bioimage (Frontier Demo)

**Status.** Frontier exploration — cutting-edge research, tools and methods change rapidly.
**Estimated time.** 10–15 minutes on Colab T4.
**Prerequisites.** Notebook 03a (denoising / hallucination awareness), Notebook 06 (integrity reporting for generated images).

> **⚠ Frontier-status notice.** This notebook covers a fast-moving area. Tools, pretrained models, and best practices in diffusion-based bioimage analysis change yearly. **Treat this notebook as orientation, not as a how-to guide for a paper you're writing today.** Read it to understand what's coming; for applied work, check the latest literature (see **Resources** at the end).

**Learning goals.**

1. Understand the **diffusion model concept**: forward process (gradual noise addition), reverse process (learned denoising), and where diffusion fits in the AI taxonomy (replacing GANs in many image generation tasks).
2. See a **forward diffusion process** in action — a microscopy image gradually becomes pure noise.
3. Run **inference with a small pretrained diffusion model** to generate samples (demonstrating the reverse process).
4. Identify **bioimage applications** where diffusion is emerging (restoration, synthetic data augmentation, latent diffusion).
5. Understand **integrity obligations** when a diffusion model generates a "microscopy-like" image (it is fully synthetic, not a measurement).
6. Know where to learn more as the field develops.

---""")


# ---------------------------------------------------------------------------
# Frontier-status banner
# ---------------------------------------------------------------------------
def section_banner(b):
    b.md("""## ⚠ Frontier-status banner

Diffusion models are the current frontier for generative AI in imaging. This notebook introduces the concept and shows one or two small examples. **Three things to know:**

1. **This is 2025–2026 research.** Most bioimage applications of diffusion are 1–2 years old (as of 2025). The tools, libraries, and best practices are still settling.
2. **Papers in this space cite each other rapidly.** Check arxiv and recent bioimage conferences (ISBI, NeurIPS) for the latest work.
3. **Your paper's reviewers may not yet expect to see diffusion outputs.** Conversely, in 5 years they may demand it. Track the field.

If you're reading this in 2026 or later, the references below may be dated — that's the frontier for you.""")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def section_setup(b):
    b.md("""## Setup

Install the libraries we'll use. The `diffusers` library from HuggingFace is the main interface for pretrained diffusion models; `transformers` provides the text encoders some models need.""")

    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

# Install main stack
%pip install --quiet diffusers transformers torch torchvision scikit-image matplotlib numpy

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import data as skdata
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# Check GPU
try:
    import torch
    print(f"torch       : {torch.__version__}")
    print(f"CUDA avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device  : {torch.cuda.get_device_name(0)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    print("torch not installed (diffusers will install it)")
    device = "cpu"

print(f"Using device: {device}")
print("Imports OK.")""")


# ---------------------------------------------------------------------------
# The diffusion idea, conceptually
# ---------------------------------------------------------------------------
def section_concept(b):
    b.md("""## The diffusion idea, conceptually

**Forward process (noising).** Start with a clean image. Over T timesteps (typically 50–1000), gradually add Gaussian noise until the image is indistinguishable from pure random noise. This is *purely analytical* — no learning involved.

**Reverse process (denoising).** A neural network learns to predict and remove the noise added at each step. Given an image at timestep t, the network predicts the noise and subtracts it, recovering an image closer to timestep t-1. By iterating this reverse process from pure noise down to timestep 0, the network generates a new sample from the data distribution it was trained on.

**Why this works.** The forward process is deterministic and analytically tractable (it's just adding Gaussian noise). The reverse process learns the score function — the gradient of the log-probability density. This is elegant: instead of learning to generate directly (like a GAN), the network learns to "denoise" at each step, which is a much more stable optimization problem.

**Key paper.** Ho, Jain, & Abbeel (2020) — *Denoising Diffusion Probabilistic Models* (DDPM). This is the canonical reference for the concept and the math.

**Visual intuition:**
- **Forward**: clean image → t=10 steps of noise → t=20 → t=50 → pure noise
- **Reverse**: pure noise → network predicts noise at t=50 → subtract → slightly cleaner image → iterate → t=0 → generated sample (not a measurement, but statistically similar to training data)

The generated sample is **never** a real measurement — it is a sample from the learned distribution. This is crucial for integrity reporting (see §11 below).""")


# ---------------------------------------------------------------------------
# Demo 1: Forward diffusion process visualization
# ---------------------------------------------------------------------------
def section_demo_forward(b):
    b.md("""## Demo 1 — Forward diffusion process visualization

We'll take a real microscopy image (from scikit-image) and progressively add Gaussian noise to show what the forward process looks like. This is purely analytical — no training.

**Predict before you run:** Look at the image. Guess what it looks like after 25 steps of noise. After 50 steps?""")

    b.code("""# Load a microscopy image
img = skdata.cell()  # grayscale, uint8, shape ~ (512, 512)
print(f"Loaded: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")

# Normalize to [0, 1]
x0 = img.astype(np.float32) / 255.0
print(f"Normalized: range=[{x0.min():.3f}, {x0.max():.3f}]")

# Define the forward diffusion schedule
# Linear schedule from beta_start to beta_end over num_steps
num_steps = 50
beta_start = 0.0001
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, num_steps)

# Precompute alpha values (alpha_t = product of (1 - beta_i) for i=1..t)
alphas = np.cumprod(1.0 - betas)
sqrt_alphas = np.sqrt(alphas)
sqrt_one_minus_alphas = np.sqrt(1.0 - alphas)

print(f"Forward schedule: {num_steps} timesteps, beta in [{beta_start}, {beta_end}]")
print(f"  sqrt_alpha_0 = {sqrt_alphas[0]:.4f}, sqrt_1-alpha_0 = {sqrt_one_minus_alphas[0]:.4f}")
print(f"  sqrt_alpha_T = {sqrt_alphas[-1]:.4f}, sqrt_1-alpha_T = {sqrt_one_minus_alphas[-1]:.4f}")""")

    b.code("""# Render the forward process at 6 timepoints
fig, axes = plt.subplots(1, 6, figsize=(15, 2.5))
timepoints = [0, 10, 20, 30, 40, 49]

for i, t in enumerate(timepoints):
    # Forward diffusion formula: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
    # where epsilon is standard normal noise
    noise = np.random.randn(*x0.shape)
    x_t = sqrt_alphas[t] * x0 + sqrt_one_minus_alphas[t] * noise

    # Clip to [0, 1] for visualization
    x_t_viz = np.clip(x_t, 0, 1)

    axes[i].imshow(x_t_viz, cmap='gray')
    axes[i].set_title(f"t={t}/{num_steps-1}")
    axes[i].axis('off')

plt.tight_layout()
plt.suptitle("Forward diffusion process: clean → noise", y=1.02, fontsize=12)
plt.show()

print(f"\\nShowing 6 frames from the forward process:")
print(f"  t=0   : clean image (signal strong, noise weak)")
print(f"  t=10  : beginning of corruption")
print(f"  t=20  : noticeable degradation")
print(f"  t=30  : mostly noise")
print(f"  t=40  : nearly pure noise")
print(f"  t=49  : nearly indistinguishable from pure Gaussian noise")""")

    b.md("""**What you should be seeing.** The image gradually becomes noisier. By t=40–50, it's nearly indistinguishable from a sample of Gaussian noise. This process is deterministic and invertible in principle — if we knew the noise added at each step, we could reconstruct the original image perfectly. But we don't. The reverse process learns to *predict* that noise, step by step.""")


# ---------------------------------------------------------------------------
# Demo 2: Reverse diffusion (denoising) with a small pretrained model
# ---------------------------------------------------------------------------
def section_demo_reverse(b):
    b.md("""## Demo 2 — Reverse diffusion with a small pretrained model

We'll load a small pretrained diffusion model from HuggingFace and run it for a few steps to show the reverse process — going from noise back toward data-like samples.

**Important caveat:** The pretrained model was trained on natural images (CIFAR-10 32×32 images), not microscopy. The samples it generates will look like simple colored noise-patterns, not like real cells. But the *pattern* — that we start from pure noise and the network denoise iteratively — is exactly what happens with a microscopy-trained diffusion model.""")

    b.code("""# Try to load a small pretrained DDPM from HuggingFace
# Using google/ddpm-cifar10-32 which is small and public
try:
    from diffusers import DDPMPipeline
    import torch

    print("Attempting to load pretrained diffusion model...")
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32", torch_dtype=torch.float32)
    pipe = pipe.to(device)

    print(f"Model loaded: google/ddpm-cifar10-32")
    print(f"  Input size  : 32x32 (CIFAR-10)")
    print(f"  Training set: natural images (cars, cats, dogs, ...)")
    print(f"  Note: NOT trained on microscopy — output is for demonstration only")

    MODEL_LOADED = True
except Exception as e:
    print(f"Could not load HuggingFace model: {e}")
    print("Will train a tiny DDPM inline instead (Option B).")
    MODEL_LOADED = False

print(f"\\nMODEL_LOADED = {MODEL_LOADED}")""")

    b.code("""if MODEL_LOADED:
    # Option A: Use the pretrained model
    print("\\n=== Option A: Pretrained model inference ===\\n")

    # Generate a batch of 4 samples
    n_samples = 4
    print(f"Generating {n_samples} samples from pure noise...")

    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(42)
        images = pipe(
            batch_size=n_samples,
            generator=generator,
            num_inference_steps=50
        ).images

    # images is a list of PIL images, size 32x32 each
    print(f"Generated {len(images)} samples, each 32x32")

    # Visualize
    fig, axes = plt.subplots(1, n_samples, figsize=(10, 2.5))
    for i, img_pil in enumerate(images):
        arr = np.array(img_pil)
        axes[i].imshow(arr)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.suptitle("Samples from pretrained DDPM (CIFAR-10 natural images)", y=1.02)
    plt.show()

    print("\\nObservations:")
    print("  - The model generates small 32x32 color images")
    print("  - Trained on natural images (CIFAR-10), so outputs resemble cars, animals, etc.")
    print("  - NOT trained on microscopy, so they don't look like cells")
    print("  - But the diffusion *process* (start from noise → iterative denoising → sample) is identical")

else:
    print("\\nSkipping Option A (model download failed). See code cell below for Option B.")""")

    b.code("""if not MODEL_LOADED:
    # Option B: Train a tiny DDPM inline
    print("\\n=== Option B: Train a tiny DDPM on synthetic round-cell data ===\\n")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Generate synthetic training data: simple round blobs on black background
    print("Generating synthetic training data...")
    n_train = 64
    img_size = 32
    X_train = []
    for i in range(n_train):
        rng = np.random.default_rng(i)
        img = np.zeros((img_size, img_size), dtype=np.float32)
        n_cells = rng.integers(2, 5)
        for _ in range(n_cells):
            cy, cx = rng.uniform(5, img_size - 5, 2)
            r = rng.uniform(3, 7)
            Y, X = np.ogrid[:img_size, :img_size]
            circle = (Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2
            img[circle] = rng.uniform(0.7, 1.0)
        # Add a tiny bit of Gaussian blur
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=0.5)
        X_train.append(img)
    X_train = np.array(X_train, dtype=np.float32)
    print(f"Generated {n_train} training images, shape {X_train.shape}")

    # Convert to torch
    X_train_torch = torch.from_numpy(X_train[:, None]).to(device)  # (N, 1, 32, 32)

    # Define diffusion schedule for training
    num_steps_train = 50
    betas_train = np.linspace(0.0001, 0.02, num_steps_train)
    alphas_train = np.cumprod(1.0 - betas_train)
    sqrt_alphas_train = np.sqrt(alphas_train)
    sqrt_one_minus_alphas_train = np.sqrt(1.0 - alphas_train)

    # Simple U-Net for noise prediction
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32*32, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 32*32)

        def forward(self, x):
            # x: (B, 1, 32, 32)
            x = x.view(x.shape[0], -1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x.view(-1, 1, 32, 32)

    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Model: SimpleUNet with {sum(p.numel() for p in model.parameters())} parameters")

    # Training loop
    print("\\nTraining diffusion model...")
    n_epochs = 100
    batch_size = 16
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        for b_idx in range(0, len(X_train_torch), batch_size):
            batch = X_train_torch[b_idx : b_idx + batch_size]

            # Sample random timesteps
            t_idx = np.random.randint(0, num_steps_train, size=batch.shape[0])
            t_tensor = torch.from_numpy(t_idx).to(device)

            # Corrupt the batch
            eps = torch.randn_like(batch)
            alpha_t = torch.from_numpy(sqrt_alphas_train[t_idx]).to(device).view(-1, 1, 1, 1).float()
            one_minus_alpha_t = torch.from_numpy(sqrt_one_minus_alphas_train[t_idx]).to(device).view(-1, 1, 1, 1).float()
            x_t = alpha_t * batch + one_minus_alpha_t * eps

            # Predict noise
            eps_pred = model(x_t)
            loss = nn.functional.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= (len(X_train_torch) // batch_size)
        losses.append(epoch_loss)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{n_epochs}: loss = {epoch_loss:.4f}")

    print("Training complete.")

    # Plot training loss
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(losses, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training loss: model learning to predict noise")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Generate samples by iterating the reverse process
    print("\\nGenerating samples by reverse diffusion...")
    n_gen = 4
    with torch.no_grad():
        # Start from pure noise
        x = torch.randn(n_gen, 1, 32, 32).to(device)

        # Iterate from t=49 down to t=0
        for t_idx in range(num_steps_train - 1, -1, -1):
            eps_pred = model(x)

            # Reverse step formula
            alpha_t = sqrt_alphas_train[t_idx]
            one_minus_alpha_t = sqrt_one_minus_alphas_train[t_idx]

            # x_{t-1} = (x_t - sqrt(1-alpha_t) * eps_pred) / sqrt(alpha_t)
            x = (x - one_minus_alpha_t * eps_pred) / alpha_t

            # Add small noise except at final step
            if t_idx > 0:
                sigma = np.sqrt(one_minus_alpha_t ** 2 / alpha_t ** 2 * (1.0 - alphas_train[t_idx - 1] / alphas_train[t_idx]))
                z = torch.randn_like(x)
                x = x + sigma * z

    samples = x.cpu().numpy().squeeze()
    print(f"Generated {n_gen} samples")

    # Visualize
    fig, axes = plt.subplots(1, n_gen, figsize=(10, 2.5))
    for i in range(n_gen):
        sample_viz = np.clip(samples[i], 0, 1)
        axes[i].imshow(sample_viz, cmap='gray')
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.suptitle("Samples from trained tiny DDPM (synthetic round cells)", y=1.02)
    plt.show()

    print("\\nObservations:")
    print("  - Model trained on synthetic round-cell data")
    print("  - Generated samples show noisy round-blob patterns")
    print("  - Loss decreased during training (model learned the distribution)")
    print("  - Samples are fully generated, not measurements")""")

    b.md("""**What you should be seeing.**

**Option A (pretrained model):** Small 32×32 color images that resemble CIFAR-10 objects (cars, animals, etc.). The network generates these by starting from noise and iteratively denoising.

**Option B (tiny trained model):** Small 32×32 grayscale images showing noisy round blobs (the training distribution). The loss decreased during training, showing the model learned to predict noise. Samples are fully generated by reversing the diffusion process.

**Key insight:** Both options show the same workflow — start from pure noise, run the network T times, get a generated sample. The difference is the training data. A microscopy-trained diffusion model would generate microscopy-like patterns; this one generates the pattern it saw in training.""")


# ---------------------------------------------------------------------------
# Predict-before-run quiz
# ---------------------------------------------------------------------------
def section_quiz(b):
    b.md("""## Predict-before-run quiz

**Multiple choice.** Before scrolling down, commit to an answer.

**Question 1.** You generate a diffusion-model output that looks like a fluorescence microscopy image of cells. What is the most important thing to know about those cells?

A) They are a blurry average of the training data.
B) They are statistically sampled from the learned distribution, not measurements.
C) They are perfect if the loss was low.
D) They are only valid for counting.

**Answer:** B. The diffusion model generates samples from its learned distribution. They are *not measurements* — no photons were actually emitted from them. This is the integrity reporting point: you must disclose that the image is synthetic.

---

**Question 2.** You compare a diffusion-generated "cell" image against a real fluorescence image. What can go wrong?

A) The diffusion model invents features not in the training set.
B) The diffusion model blurs fine details.
C) The diffusion model can hallucinate, just like the GANs in NB06.
D) All of the above.

**Answer:** D. Diffusion models have the same hallucination risk as any generative model. A diffusion output can look convincing but be anatomically impossible. Use the same hallucination check from NB03a: does the generated image have plausible correlations with real data?""")


# ---------------------------------------------------------------------------
# Bioimage applications survey
# ---------------------------------------------------------------------------
def section_applications(b):
    b.md("""## Bioimage applications — the emerging frontier

Diffusion models have started replacing GANs in several bioimage contexts. This is a rapidly moving area (2023–2025). The papers below are representative; **check arxiv for 2025-2026 work:**

### Diffusion for image restoration / denoising

Chung et al. (2022, ICCV) and Saharia et al. (2022, NeurIPS) showed that diffusion models can serve as powerful priors for inverse problems — denoising, inpainting, super-resolution. The key insight: instead of training a network to directly denoise, you train the reverse process to reverse the forward process (which includes noise). This is more stable than a U-Net-to-noise direct prediction.

⚠ **Frontier flag:** Papers applying this to bioimage are emerging (2024–2025). Check if the restoration output has been validated against real (non-synthetic) benchmarks.

### Diffusion for synthetic data augmentation

If you have labeled microscopy data but not enough of it, a diffusion model can generate realistic synthetic samples. This is beginning to appear in segmentation papers: train the model on diffusion-generated + real data, often improving robustness.

⚠ **Frontier flag:** Real bioimage benchmarks for synthetic augmentation are sparse. Most papers in 2024 show proof-of-concept; large-scale validation is ongoing.

### Latent diffusion

Stable Diffusion and similar text-to-image models work in a learned latent space, not in pixel space. This is much faster. Some bioimage groups are adapting latent diffusion to microscopy (2024–2025).

⚠ **Frontier flag:** Latent diffusion for microscopy is very recent. The papers are likely on arxiv; check the latest conference proceedings.

### Score-based generation for super-resolution

Song et al. (2020, NeurIPS) introduced score-based generative modeling (mathematically equivalent to diffusion). Some bioimage SR work uses score-based priors.

⚠ **Frontier flag:** This is largely in the 2024–2025 literature. Stable reference benchmarks are still being established.

---

**References:**
- Awesome-Diffusion-Models-in-Medical-Imaging: [github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging](https://github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging) — comprehensive meta-list (tracks papers through 2024–2025).
- Original DDPM: Ho, Jain, & Abbeel (2020) [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- HuggingFace Diffusers: [huggingface.co/docs/diffusers/](https://huggingface.co/docs/diffusers/)""")


# ---------------------------------------------------------------------------
# Integrity reporting for diffusion-generated bioimage
# ---------------------------------------------------------------------------
def section_integrity(b):
    b.md("""## Integrity reporting for diffusion-generated bioimage

**Core principle:** A diffusion-generated image is *fully synthetic*. Every pixel is a model output. This is no different from a virtually stained image (NB06) or a GAN output — it must be disclosed.

### What to do

1. **Label as synthetic in the figure legend.** Example:
   > *"Images in panel C are generated by a diffusion model trained on [dataset]. They are not experimental measurements."*

2. **Include a hallucination check.** Did you run your generated image through a validation step (e.g., does it correlate with segmentation ground truth on real images)? Disclose that.

3. **Report the training data.** What was the diffusion model trained on? Public models (DDPM-CIFAR10) are easier to justify than proprietary data.

4. **Be honest about the use case.**
   - **Acceptable:** *"We used diffusion to augment training data for a segmentation model, then validated on real images."*
   - **Questionable:** *"We used diffusion to generate data that looks more convincing than reality to illustrate a morphology we never measured."*

### Example paragraph

> *"Figure 3C shows synthetic cell morphologies generated by a diffusion model fine-tuned on [reference segmentation dataset]. These are not experimental measurements; they are samples from the learned distribution intended to illustrate plausible morphological diversity. The diffusion model was trained with [loss function], and generated samples were verified to maintain biological plausibility by [method]. All experimental data in Figure 3A–B are from [real data source]."*

### Cross-reference

See NB03a (denoising hallucination check) and NB06 (virtual staining integrity) for the same principle applied to other generative methods. **The rule is consistent:** Generated != Measured. Disclose.""")


# ---------------------------------------------------------------------------
# Closing reflection
# ---------------------------------------------------------------------------
def section_closing(b):
    b.md("""## Closing reflection

Diffusion models are the current frontier for image generation and restoration. Over 2024–2026, expect:

- More bioimage applications (restoration, SR, augmentation, synthetic data).
- Faster inference methods (latent diffusion, score distillation).
- Better public benchmarks (to move from "proof-of-concept" to "validated").
- Clearer integrity standards for diffusion outputs in journals.

**This notebook is orientation.** It shows you *what diffusion is*, not *how to deploy it in your paper today*. As the field settles (2026–2027), best practices will crystallize. For now:

- **Read actively:** Check arxiv weekly for new bioimage+diffusion papers.
- **Validate strictly:** If you use a diffusion output, run it through the same hallucination checks as NB03a and NB06.
- **Disclose honestly:** Treat generated images like any other AI output — full disclosure of the method and validation.

---

### Resources and next steps

**Learning more about diffusion:**
- Original DDPM paper (Ho, Jain, Abbeel 2020): [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- HuggingFace Diffusers documentation: [huggingface.co/docs/diffusers/](https://huggingface.co/docs/diffusers/)
- Luo (2022) — Diffusion Models Explained: [arxiv.org/abs/2208.11970](https://arxiv.org/abs/2208.11970)

**Bioimage-specific applications:**
- Awesome-Diffusion-Models-in-Medical-Imaging meta-list: [github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging](https://github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging)
- Check Notebook 04 (Resources page) for curated stable links to bioimage-AI papers.

**Related notebooks in this workshop:**
- **Notebook 03a** — Denoising and the hallucination check (same validation principle).
- **Notebook 06** — Virtual staining and integrity reporting (similar disclosure obligations).
- **Notebook 04** — Broader AI taxonomy (where diffusion fits).

---

**Final thought.** The fact that diffusion models are now frontier suggests that 2–3 years ago (2022–2023), they were bleeding-edge research. The pace of change in generative AI is fast. As you read this in 2026 or later, what's in this notebook may already feel outdated. That's OK — **it means the field is advancing.** The core principle stays the same: *generated ≠ measured*. Disclose, validate, and deploy with integrity.""")


# ---------------------------------------------------------------------------
# Assemble the notebook
# ---------------------------------------------------------------------------
def main():
    b = CellBuilder("nb15")
    section_title(b)
    section_banner(b)
    section_setup(b)
    section_concept(b)
    section_demo_forward(b)
    section_demo_reverse(b)
    section_quiz(b)
    section_applications(b)
    section_integrity(b)
    section_closing(b)
    build_notebook(b.cells, "15_diffusion_models")


if __name__ == "__main__":
    main()
