"""
Build script for Notebook 04 — Community Platforms.

Catalog + live BioImage Model Zoo browser + 5 inline mini-workflows
(CARE, fnet, pix2pix, Deep-STORM, YOLOv2-style detection).

The notebook is designed as a *menu* — workshop deliverers pick which sections
to demo live based on audience interest. Running every cell takes 30+ minutes
even with short training.

Run:
    python build_notebook_04.py
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
# Section: Title and how to use
# ---------------------------------------------------------------------------
def section_title(b):
    b.md("""# Notebook 04 — Community Platforms

**ZeroCostDL4Mic · DL4MicEverywhere · BioImage Model Zoo**

A structured menu of community DL-microscopy platforms and methods. Designed for *broad coverage with on-the-fly selection*: workshop deliverers pick what's relevant; attendees use the catalog as a reference for self-directed learning afterward.

**This notebook is not meant to be run top-to-bottom.** Each section stands alone. Pick the platform or method that matches the audience's interest.

---

## How to use this notebook

The notebook has four parts:

1. **Catalog** — every method in ZeroCostDL4Mic and DL4MicEverywhere with one-line descriptions and direct Colab links. Use this as a reference; click out to run any specific method on its own Colab.

2. **BioImage Model Zoo** — live API integration. Browse the model registry from inside the notebook, filter by task, load a model, run inference. Demonstrates the "choose on the fly" pattern.

3. **Five inline mini-workflows** — full demos of methods *not* covered in Labs 1, 3a, or 3b: CARE (supervised denoising), fnet (label-free prediction), pix2pix (image translation), Deep-STORM (super-resolution localization), YOLOv2-style detection. Each shows the workflow end-to-end with synthetic data and short training.

4. **Pointers** to where each method continues — full ZCD4M Colabs, ZCD4M GitHub, BiMZ entries.

**Note on runtime.** The five mini-workflows train models on tiny synthetic datasets. They'll run in 1–3 minutes each on Colab T4 — enough for the *pattern* to land, not enough to produce production-quality models. Each linked ZCD4M notebook is the production version.""")


# ---------------------------------------------------------------------------
# Section: Catalog
# ---------------------------------------------------------------------------
def section_catalog(b):
    b.md("""## 1. Catalog — ZeroCostDL4Mic and DL4MicEverywhere

Below: every method that ships with ZeroCostDL4Mic, plus DL4MicEverywhere additions. Each entry has a direct Colab link. Click the link to launch the dedicated Colab for that method.

**Key:** `[ZCD4M]` = in ZeroCostDL4Mic. `[DL4ME]` = added by DL4MicEverywhere. `[both]` = in both.

### Segmentation

- **U-Net (semantic segmentation)** `[both]` — pixel-wise multi-class segmentation. The canonical reference architecture (Ronneberger et al., 2015, MICCAI).
  [ZCD4M U-Net Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/U-Net-(2D))
- **StarDist (instance segmentation, star-convex)** `[both]` — instance segmentation for densely packed nuclei. Schmidt et al., 2018, MICCAI.
  [ZCD4M StarDist Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/Stardist-(2D))
- **Cellpose** — generalist cell segmentation. Stringer et al., 2021, Nature Methods. *Covered in detail in Lab 1.*

### Detection

- **YOLOv2 (object detection)** `[ZCD4M]` — bounding-box detection of objects (e.g., FISH spots, vesicles).
  [ZCD4M YOLOv2 Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/YOLOv2)

### Restoration / denoising

- **CARE (Content-Aware Image Restoration)** `[both]` — supervised denoising and restoration. Weigert et al., 2018, Nature Methods.
  [ZCD4M CARE Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/CARE-(2D))
- **Noise2Void** `[both]` — self-supervised denoising. Krull et al., 2019. *Covered in detail in Lab 3a.*
- **DecoNoising** `[DL4ME]` — joint deconvolution + denoising. Newer addition.

### Super-resolution

- **Deep-STORM** `[both]` — single-molecule localization microscopy reconstruction. Nehme et al., 2018, Optica.
  [ZCD4M Deep-STORM Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/Deep-STORM)
- **3D-RCAN** `[both]` — 3D super-resolution. Chen et al., 2021.
- **DFCAN/DFGAN** `[DL4ME]` — fluorescence super-resolution with attention.

### Image-to-image / generation / translation

- **fnet (in silico labeling)** `[ZCD4M]` — predict fluorescence labels from brightfield. Ounkomol et al., 2018, Nature Methods.
  [ZCD4M fnet Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/Label-free-prediction-(fnet))
- **pix2pix** `[ZCD4M]` — supervised image-to-image translation. Isola et al., 2017.
  [ZCD4M pix2pix Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/pix2pix)
- **CycleGAN** `[ZCD4M]` — unsupervised image-to-image translation (no paired data needed).
  [ZCD4M CycleGAN Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/CycleGAN)

### Tracking

- **Cell tracking** — typically via Mastodon, btrack, or Cellpose tracking. Not in ZCD4M as a standalone notebook; integrated in DL4ME containers.

### Registration

- **Image registration** `[DL4ME]` — added in DL4MicEverywhere. Not in original ZCD4M.

### Foundation models

- **Segment Anything (SAM) / μSAM** — covered in Lab 3b. The foundation-model paradigm is increasingly available via DL4ME containers.

---

### How to interpret the catalog

ZeroCostDL4Mic is the *canonical* free toolbox: a Colab notebook per method. **Use ZCD4M when** you want to try a method without installing anything.

DL4MicEverywhere is the *production-flexible* successor: same methods plus more, packaged as Docker containers so they run on Colab, local, HPC, or cloud. **Use DL4ME when** you want to scale beyond Colab or run reproducibly on shared infrastructure.

Both share the same notebook format, so once you know one, you know the other.""")


# ---------------------------------------------------------------------------
# Section: BioImage Model Zoo
# ---------------------------------------------------------------------------
def section_biomodel_zoo(b):
    b.md("""## 2. BioImage Model Zoo — live browse and load

The **BioImage Model Zoo** ([bioimage.io](https://bioimage.io)) is a community repository of pretrained DL models in a standardized format. Hundreds of models for segmentation, denoising, classification, and more — each with metadata, sample inputs/outputs, and weights.

The Python package `bioimageio.core` lets you browse the registry and load any model from inside a notebook. This is the **"choose on the fly"** pattern: scan available models, pick one that fits, run it.""")

    b.code("""# Install the BiMZ Python interface
%pip install --quiet bioimageio.core matplotlib numpy
print("Installed.")""")

    b.md("""### Browse available models""")

    b.code("""# Live query of the BiMZ registry. Falls back to a curated list if offline.
import json, urllib.request

REGISTRY_URL = "https://bioimage-io.github.io/collection-bioimage-io/collection.json"

try:
    with urllib.request.urlopen(REGISTRY_URL, timeout=15) as r:
        registry = json.load(r)
    models = [c for c in registry.get("collection", []) if c.get("type") == "model"]
    print(f"Live registry: {len(models)} models available.")
    online = True
except Exception as e:
    print(f"Live registry unreachable ({e}). Falling back to curated list.")
    models = [
        {"id": "10.5281/zenodo.5764892",  "name": "EnhancerMitochondriaEM2D",
         "tags": ["3d", "electron microscopy", "denoising"]},
        {"id": "10.5281/zenodo.5749843",  "name": "PlatynereisEMnucleiSegmentationBoundaryModel",
         "tags": ["2d", "electron microscopy", "segmentation"]},
        {"id": "10.5281/zenodo.5864646",  "name": "CovidIfCellSegmentationBoundary",
         "tags": ["2d", "fluorescence microscopy", "segmentation"]},
    ]
    online = False

# Print the first 10
print("\\nFirst 10 models:")
for m in models[:10]:
    name = m.get("name", "—")
    mid = m.get("id", "—")
    tags = ", ".join(m.get("tags", [])[:5])
    print(f"  {name[:50]:<52} [{mid[:30]}]  ({tags})")""")

    b.md("""### Filter by task or modality

Each model carries `tags` (e.g., 'segmentation', 'denoising', 'fluorescence microscopy', 'electron microscopy'). You can filter the registry to match what you need.""")

    b.code("""# Pick a task
target_tag = "segmentation"  # try 'denoising', 'classification', etc.

matches = [m for m in models if target_tag in m.get("tags", [])]
print(f"Models tagged '{target_tag}': {len(matches)}")
for m in matches[:8]:
    print(f"  - {m.get('name', '—')[:55]:<57}  ({', '.join(m.get('tags', [])[:4])})")""")

    b.md("""### Load and run a model

Once you've picked a model ID, `bioimageio.core` loads it (downloading weights as needed) and runs inference. The example below demonstrates the API on a small synthetic input. For a real workflow, replace the synthetic image with your own.""")

    b.code("""# Inference walkthrough (heavily commented). Wrapped in try/except because
# any specific model ID can fall out of the registry over time.
import numpy as np
import matplotlib.pyplot as plt

try:
    from bioimageio.core import load_resource_description, predict_with_padding
    from bioimageio.core.prediction_pipeline import create_prediction_pipeline

    # Example: a 2D fluorescence segmentation model from the curated list.
    # Substitute any model ID from your filtered list above.
    MODEL_ID = "10.5281/zenodo.5864646"  # CovidIfCellSegmentationBoundary

    print(f"Loading model: {MODEL_ID}")
    rd = load_resource_description(MODEL_ID)
    pipeline = create_prediction_pipeline(bioimageio_model=rd)
    print(f"Model loaded: {rd.name}")
    print(f"  Inputs : {[ax.name for ax in rd.inputs[0].axes]}")
    print(f"  Outputs: {[ax.name for ax in rd.outputs[0].axes]}")

    # Build a synthetic input matching the model's expected shape
    # (real workflow: load your own image and reshape as needed)
    synth = np.random.rand(1, 1, 256, 256).astype(np.float32)  # [B, C, H, W]

    print("Running inference...")
    result = predict_with_padding(pipeline, synth)
    print(f"Output shape: {result[0].shape}")

    # Visualize input and output
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(synth[0, 0], cmap='gray'); axes[0].set_title("Synthetic input"); axes[0].axis('off')
    axes[1].imshow(result[0][0, 0], cmap='viridis'); axes[1].set_title(f"{rd.name} output"); axes[1].axis('off')
    plt.tight_layout(); plt.show()

except Exception as e:
    print(f"Inference walkthrough failed: {e}")
    print("This is expected if the model ID has changed or BiMZ is unreachable.")
    print("Browse https://bioimage.io for current model IDs.")""")

    b.md("""**Try this — browse and pick.** Change `target_tag` in the filter cell above to match your interest (`denoising`, `super-resolution`, `classification`, etc.). Pick a model ID that looks promising. Substitute it for `MODEL_ID` in the inference cell. Run.

The pattern: BiMZ is a *registry*, not a fixed library. Use the API to discover what's available at workshop time, not what was available when this notebook was written.""")


# ---------------------------------------------------------------------------
# Section: Inline demo — CARE (Content-Aware Image Restoration)
# ---------------------------------------------------------------------------
def demo_care(b):
    b.md("""## 3a. Inline demo — CARE (Content-Aware Image Restoration)

**What:** supervised denoising / restoration. Trained with paired noisy/clean images.
**Vs Lab 3a Noise2Void:** N2V is *self-supervised* (no clean reference); CARE is *supervised* (paired data). When you have clean references, CARE typically wins. When you don't, N2V is the only option.
**Citation:** Weigert et al., 2018, Nature Methods.
**Production version:** [ZCD4M CARE Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/CARE-(2D))

The mini-workflow below shows the CARE pattern: paired data → train → infer → evaluate.""")

    b.code("""# Install CSBDeep (the package CARE ships in)
%pip install --quiet csbdeep tensorflow numpy matplotlib scipy
print("Installed.")""")

    b.code("""# Generate paired clean + noisy synthetic data
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(0)

def make_clean_pair(size=128, n_objects=10):
    img = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_objects):
        cy, cx = rng.integers(15, size - 15, size=2)
        r = rng.integers(6, 12)
        Y, X = np.ogrid[:size, :size]
        img[(Y - cy)**2 + (X - cx)**2 <= r**2] = rng.uniform(0.6, 1.0)
    return gaussian_filter(img, sigma=1.0)

def add_noise(img, photons=20):
    scaled = img * photons
    noisy = rng.poisson(np.clip(scaled, 0, None)).astype(np.float32) / photons
    noisy = noisy + rng.normal(0, 0.05, noisy.shape).astype(np.float32)
    return np.clip(noisy, 0, None)

# 64 paired training samples
clean_train = np.stack([make_clean_pair() for _ in range(64)])
noisy_train = np.stack([add_noise(c) for c in clean_train])

print(f"clean_train: {clean_train.shape}, noisy_train: {noisy_train.shape}")

# Show one pair
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(noisy_train[0], cmap='gray'); axes[0].set_title("Noisy (input)"); axes[0].axis('off')
axes[1].imshow(clean_train[0], cmap='gray'); axes[1].set_title("Clean (target)"); axes[1].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Try this — predict before training.** With 64 paired examples and a short training run, do you expect CARE to (a) match clean target perfectly, (b) approximate it but with residual error, or (c) fail because data is too small? Run the next cell.""")

    b.code("""# Train a tiny CARE model. CSBDeep handles all the network + training plumbing.
from csbdeep.models import Config, CARE

# Reshape to (N, H, W, C) — CSBDeep expects channel-last
X = noisy_train[..., np.newaxis]
Y = clean_train[..., np.newaxis]

config = Config(axes='YXC',
                n_channel_in=1, n_channel_out=1,
                unet_n_depth=2, unet_kern_size=3,
                train_steps_per_epoch=10,  # tiny for demo
                train_epochs=5,            # tiny for demo
                train_batch_size=8)
print(config)

model_care = CARE(config, name='care_demo', basedir='care_outputs')
history = model_care.train(X, Y, validation_data=(X[:8], Y[:8]))
print("Training complete.")""")

    b.code("""# Inference on a held-out noisy image
test_clean = make_clean_pair()
test_noisy = add_noise(test_clean)

restored = model_care.predict(test_noisy[..., np.newaxis], axes='YXC').squeeze()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(test_noisy,  cmap='gray'); axes[0].set_title("Noisy input")
axes[1].imshow(restored,    cmap='gray'); axes[1].set_title("CARE restored")
axes[2].imshow(test_clean,  cmap='gray'); axes[2].set_title("Clean reference")
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Lesson.** With paired clean/noisy data and 5 short training epochs, CARE produces visibly improved images. Real workflows would train for 100+ epochs on much more data; the production [ZCD4M CARE Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/CARE-(2D)) walks through proper training.

CARE vs Noise2Void (Lab 3a):
- *CARE* needs paired data. Higher quality when you have the pairs.
- *N2V* needs only the noisy data. Lower ceiling but works when clean references don't exist.""")


# ---------------------------------------------------------------------------
# Section: Inline demo — fnet (label-free prediction / in silico labeling)
# ---------------------------------------------------------------------------
def demo_fnet(b):
    b.md("""## 3b. Inline demo — fnet (label-free prediction)

**What:** predict fluorescence-style images from brightfield. Trained on paired brightfield/fluorescence stacks.
**Why it matters:** label-free workflows skip fluorescent staining (no phototoxicity, no spectral overlap concerns) but still get fluorescence-style readouts.
**Citation:** Ounkomol et al., 2018, Nature Methods.
**Production version:** [ZCD4M fnet Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/Label-free-prediction-(fnet))

The mini-workflow shows the *pattern* — paired brightfield/fluorescence simulation → train a small U-Net → predict on held-out brightfield.""")

    b.code("""# Setup. PyTorch is the backbone for fnet-style models.
%pip install --quiet torch torchvision matplotlib numpy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")""")

    b.code("""# Synthetic paired data: 'brightfield' = low-contrast, 'fluorescence' = high-contrast pattern
rng = np.random.default_rng(42)

def make_pair(size=64):
    fluor = np.zeros((size, size), dtype=np.float32)
    for _ in range(rng.integers(4, 10)):
        cy, cx = rng.integers(8, size - 8, size=2)
        r = rng.integers(3, 7)
        Y, X = np.ogrid[:size, :size]
        fluor[(Y - cy)**2 + (X - cx)**2 <= r**2] = rng.uniform(0.6, 1.0)
    # 'Brightfield' is low-contrast inverse with more noise
    bf = 0.5 + 0.2 * (fluor - fluor.mean()) + rng.normal(0, 0.04, fluor.shape).astype(np.float32)
    bf = np.clip(bf, 0, 1)
    return bf, fluor

# 128 training pairs
n_train = 128
bfs = np.stack([make_pair()[0] for _ in range(n_train)])
flrs = np.stack([make_pair()[1] for _ in range(n_train)])
print(f"Brightfield: {bfs.shape}, Fluorescence: {flrs.shape}")

# Show one pair
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(bfs[0], cmap='gray'); axes[0].set_title("Brightfield (input)"); axes[0].axis('off')
axes[1].imshow(flrs[0], cmap='magma'); axes[1].set_title("Fluorescence (target)"); axes[1].axis('off')
plt.tight_layout(); plt.show()""")

    b.code("""# Tiny U-Net for the demo. Real fnet uses a 3D U-Net with much deeper architecture.
class TinyUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, base, 3, padding=1), nn.ReLU(), nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(), nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU())
        self.up   = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec  = nn.Sequential(nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(), nn.Conv2d(base, 1, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u  = self.up(e2)
        return self.dec(torch.cat([u, e1], dim=1))

device = "cuda" if torch.cuda.is_available() else "cpu"
net = TinyUNet().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X = torch.tensor(bfs[:, None]).to(device)   # (N, 1, H, W)
Y = torch.tensor(flrs[:, None]).to(device)

# Short training: 50 steps, batch size 16
print("Training...")
for step in range(50):
    idx = torch.randperm(n_train)[:16]
    pred = net(X[idx])
    loss = loss_fn(pred, Y[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0:
        print(f"  step {step:3d}  loss {loss.item():.4f}")
print("Done.")""")

    b.code("""# Inference on held-out brightfield
test_bf, test_flr = make_pair()
with torch.no_grad():
    pred_flr = net(torch.tensor(test_bf[None, None], dtype=torch.float32).to(device)).squeeze().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(test_bf,  cmap='gray');  axes[0].set_title("Brightfield input")
axes[1].imshow(pred_flr, cmap='magma'); axes[1].set_title("fnet predicted fluorescence")
axes[2].imshow(test_flr, cmap='magma'); axes[2].set_title("Ground truth fluorescence")
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Lesson.** With 128 synthetic pairs and 50 training steps, fnet-style prediction shows the *pattern* — but quality is limited by data and training. Real fnet workflows train on hundreds of brightfield/fluorescence pairs in 3D; the production version is in the [ZCD4M fnet Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/Label-free-prediction-(fnet)).

**Image-integrity reminder:** fnet-predicted fluorescence is *generated*, not measured. It must be disclosed in figure captions. Same rule as for restoration outputs (Lab 3a).""")


# ---------------------------------------------------------------------------
# Section: Inline demo — pix2pix (image-to-image translation)
# ---------------------------------------------------------------------------
def demo_pix2pix(b):
    b.md("""## 3c. Inline demo — pix2pix (image-to-image translation)

**What:** supervised image-to-image translation. Maps any input image domain to any output domain (e.g., simulated → realistic, brightfield → stained, mask → fluorescence).
**Pattern:** conditional GAN. Generator + discriminator trained against each other.
**Citation:** Isola et al., 2017, CVPR.
**Production version:** [ZCD4M pix2pix Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/pix2pix)

Demo task: translate a binary segmentation mask back to a fluorescence-like image. Inverse of segmentation; demonstrates the I2I pattern.""")

    b.code("""# Same PyTorch setup as fnet
import torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

def make_pair_segmask(size=64):
    \"\"\"Pair: (binary mask) -> (fluorescence-like image)\"\"\"
    fluor = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.float32)
    for _ in range(rng.integers(3, 8)):
        cy, cx = rng.integers(8, size-8, size=2)
        r = rng.integers(3, 7)
        Y, X = np.ogrid[:size, :size]
        circle = (Y-cy)**2 + (X-cx)**2 <= r**2
        mask[circle] = 1.0
        fluor[circle] = rng.uniform(0.5, 1.0)
    fluor = fluor + rng.normal(0, 0.04, fluor.shape).astype(np.float32)
    return mask, np.clip(fluor, 0, 1)

# Training pairs
n = 128
masks = np.stack([make_pair_segmask()[0] for _ in range(n)])
fluors = np.stack([make_pair_segmask()[1] for _ in range(n)])

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(masks[0], cmap='gray');  axes[0].set_title("Mask (input)"); axes[0].axis('off')
axes[1].imshow(fluors[0], cmap='magma'); axes[1].set_title("Fluorescence (target)"); axes[1].axis('off')
plt.tight_layout(); plt.show()""")

    b.code("""# Generator (small U-Net) + discriminator (small CNN). Highly simplified pix2pix.
class Generator(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, base, 4, 2, 1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(base*2, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(base*2, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1)
        d1 = self.dec1(e2); d1 = torch.cat([d1, e1], 1)
        return self.dec2(d1)

class Discriminator(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, base, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2),
            nn.Conv2d(base*2, 1, 4, 2, 1)
        )

    def forward(self, mask, img):
        return self.net(torch.cat([mask, img], 1))

device = "cuda" if torch.cuda.is_available() else "cpu"
G = Generator().to(device)
D = Discriminator().to(device)

opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
l1  = nn.L1Loss()

X = torch.tensor(masks[:, None]).to(device)
Y = torch.tensor(fluors[:, None]).to(device)

print("Training pix2pix-style GAN...")
for step in range(80):
    idx = torch.randperm(n)[:16]
    real_mask, real_img = X[idx], Y[idx]
    fake_img = G(real_mask)
    # Discriminator
    d_real = D(real_mask, real_img)
    d_fake = D(real_mask, fake_img.detach())
    d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
    opt_D.zero_grad(); d_loss.backward(); opt_D.step()
    # Generator
    d_fake_for_g = D(real_mask, fake_img)
    g_loss = bce(d_fake_for_g, torch.ones_like(d_fake_for_g)) + 100 * l1(fake_img, real_img)
    opt_G.zero_grad(); g_loss.backward(); opt_G.step()
    if step % 20 == 0:
        print(f"  step {step:3d}  D {d_loss.item():.3f}  G {g_loss.item():.3f}")
print("Done.")""")

    b.code("""# Inference on a held-out mask
test_mask, test_real = make_pair_segmask()
with torch.no_grad():
    test_pred = G(torch.tensor(test_mask[None, None], dtype=torch.float32).to(device)).squeeze().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(test_mask, cmap='gray');   axes[0].set_title("Mask input")
axes[1].imshow(test_pred, cmap='magma');  axes[1].set_title("pix2pix predicted")
axes[2].imshow(test_real, cmap='magma');  axes[2].set_title("Ground truth")
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Lesson.** pix2pix learns a mapping between paired image domains. Common bioimage uses: virtual staining (label-free → stained), modality translation, simulation-to-real. The integrity caveat applies: pix2pix outputs are *generated* and must be disclosed.

For the production version with proper training settings and architecture, see the [ZCD4M pix2pix Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/pix2pix). For *unpaired* translation (no paired data), see CycleGAN in the catalog above.""")


# ---------------------------------------------------------------------------
# Section: Inline demo — Deep-STORM (super-resolution localization microscopy)
# ---------------------------------------------------------------------------
def demo_deepstorm(b):
    b.md("""## 3d. Inline demo — Deep-STORM (super-resolution from sparse localizations)

**What:** reconstruct super-resolution images from sparse single-molecule PALM/STORM data using deep learning, *much* faster than classical localization.
**Citation:** Nehme et al., 2018, Optica.
**Production version:** [ZCD4M Deep-STORM Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/Deep-STORM)

Demo: a CNN learns to map sparse-emitter input frames to dense super-resolution maps.""")

    b.code("""import torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)

def make_storm_pair(size=64, n_emitters=20):
    \"\"\"Sparse emitters (input) + dense super-res image (target)\"\"\"
    super_res = np.zeros((size, size), dtype=np.float32)
    sparse = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_emitters):
        y = rng.uniform(2, size - 2)
        x = rng.uniform(2, size - 2)
        # super-res: a tight Gaussian per emitter
        Y, X = np.mgrid[:size, :size].astype(np.float32)
        super_res += np.exp(-((Y - y)**2 + (X - x)**2) / 0.5)
        # sparse: a single bright pixel per emitter (with subpixel jitter)
        sparse[int(y), int(x)] = 1.0
    sparse += rng.normal(0, 0.05, sparse.shape).astype(np.float32)
    super_res = np.clip(super_res, 0, 1)
    return sparse, super_res

n = 128
inputs  = np.stack([make_storm_pair()[0] for _ in range(n)])
targets = np.stack([make_storm_pair()[1] for _ in range(n)])

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(inputs[0],  cmap='gray');    axes[0].set_title("Sparse emitters (input)"); axes[0].axis('off')
axes[1].imshow(targets[0], cmap='hot');     axes[1].set_title("Super-res target"); axes[1].axis('off')
plt.tight_layout(); plt.show()""")

    b.code("""# Small CNN for STORM-style super-resolution
class StormCNN(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 5, padding=2), nn.ReLU(),
            nn.Conv2d(base, base, 5, padding=2), nn.ReLU(),
            nn.Conv2d(base, base, 5, padding=2), nn.ReLU(),
            nn.Conv2d(base, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
net = StormCNN().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X = torch.tensor(inputs[:, None]).to(device)
Y = torch.tensor(targets[:, None]).to(device)

print("Training STORM-style network...")
for step in range(60):
    idx = torch.randperm(n)[:16]
    pred = net(X[idx])
    loss = loss_fn(pred, Y[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 15 == 0:
        print(f"  step {step:3d}  loss {loss.item():.4f}")
print("Done.")""")

    b.code("""# Inference on a held-out sparse input
test_in, test_tgt = make_storm_pair()
with torch.no_grad():
    pred = net(torch.tensor(test_in[None, None], dtype=torch.float32).to(device)).squeeze().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(test_in,  cmap='gray'); axes[0].set_title("Sparse input")
axes[1].imshow(pred,     cmap='hot');  axes[1].set_title("Predicted super-res")
axes[2].imshow(test_tgt, cmap='hot');  axes[2].set_title("Ground truth super-res")
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Lesson.** STORM-style networks learn to densify sparse single-molecule input frames into super-resolved images. Real Deep-STORM workflows train on tens of thousands of paired frames; the [production Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/Deep-STORM) walks through proper data preparation.

This is *not* a substitute for classical PALM/STORM localization in cases where uncertainty quantification matters — Deep-STORM trades interpretability for speed.""")


# ---------------------------------------------------------------------------
# Section: Inline demo — YOLOv2-style detection
# ---------------------------------------------------------------------------
def demo_yolo(b):
    b.md("""## 3e. Inline demo — YOLOv2-style detection (bounding-box object detection)

**What:** detect objects with bounding boxes. Common bioimage use: spot detection (FISH spots, vesicles, particles), nuclei detection without instance masks.
**Citation:** Redmon and Farhadi, 2017 (YOLOv2). The YOLO family has many newer versions; YOLOv2 is the variant in ZCD4M.
**Production version:** [ZCD4M YOLOv2 Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/YOLOv2)

Demo: a simplified detector that predicts a heatmap of object centers from synthetic spot data. (Full YOLOv2 with anchors and multi-scale predictions is beyond a 5-cell demo; the pattern is the same.)""")

    b.code("""import torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(13)

def make_spots(size=64, n_spots=8):
    img = np.zeros((size, size), dtype=np.float32)
    centers = []
    for _ in range(rng.integers(4, 12)):
        cy, cx = rng.integers(4, size - 4, size=2)
        Y, X = np.ogrid[:size, :size]
        img += 0.8 * np.exp(-((Y - cy)**2 + (X - cx)**2) / 4)
        centers.append((cy, cx))
    img = img + rng.normal(0, 0.04, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    # Target: Gaussian heatmap at each center
    heatmap = np.zeros((size, size), dtype=np.float32)
    for cy, cx in centers:
        Y, X = np.ogrid[:size, :size]
        heatmap += np.exp(-((Y - cy)**2 + (X - cx)**2) / 1.0)
    return img, np.clip(heatmap, 0, 1), centers

n = 128
imgs   = np.stack([make_spots()[0] for _ in range(n)])
maps   = np.stack([make_spots()[1] for _ in range(n)])

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(imgs[0], cmap='gray'); axes[0].set_title("Image with spots"); axes[0].axis('off')
axes[1].imshow(maps[0], cmap='hot');  axes[1].set_title("Center-heatmap target"); axes[1].axis('off')
plt.tight_layout(); plt.show()""")

    b.code("""# Small CNN: image -> heatmap of spot centers
class SpotDetector(nn.Module):
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

device = "cuda" if torch.cuda.is_available() else "cpu"
net = SpotDetector().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X = torch.tensor(imgs[:, None]).to(device)
Y = torch.tensor(maps[:, None]).to(device)

print("Training spot detector...")
for step in range(60):
    idx = torch.randperm(n)[:16]
    pred = net(X[idx])
    loss = loss_fn(pred, Y[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 15 == 0:
        print(f"  step {step:3d}  loss {loss.item():.4f}")
print("Done.")""")

    b.code("""# Inference + extracting spot coordinates from the predicted heatmap
from scipy.ndimage import maximum_filter

test_img, test_map, true_centers = make_spots()
with torch.no_grad():
    pred_map = net(torch.tensor(test_img[None, None], dtype=torch.float32).to(device)).squeeze().cpu().numpy()

# Find local maxima in the predicted heatmap (poor man's NMS)
threshold = 0.3
peak_mask = (pred_map == maximum_filter(pred_map, size=5)) & (pred_map > threshold)
detected = np.argwhere(peak_mask)

print(f"True centers   : {len(true_centers)}")
print(f"Detected spots : {len(detected)}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(test_img, cmap='gray'); axes[0].set_title("Image"); axes[0].axis('off')
axes[1].imshow(pred_map, cmap='hot');  axes[1].set_title("Predicted heatmap"); axes[1].axis('off')
axes[2].imshow(test_img, cmap='gray')
for cy, cx in detected:
    axes[2].plot(cx, cy, 'rs', markersize=8, fillstyle='none', markeredgewidth=1.5)
axes[2].set_title(f"Detected spots ({len(detected)})"); axes[2].axis('off')
plt.tight_layout(); plt.show()""")

    b.md("""**Lesson.** Detection-as-heatmap is one of several ways to phrase object detection for biology. Full YOLOv2 uses anchor boxes and predicts (x, y, w, h, confidence) per cell of a coarse grid — heavier but more flexible. The [ZCD4M YOLOv2 Colab](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki/YOLOv2) walks through the production version.

For most bioimage spot-detection cases, simpler heatmap-based detectors like the one above (or classical Laplacian-of-Gaussian) are sufficient and easier to train.""")


# ---------------------------------------------------------------------------
# Section: Closing
# ---------------------------------------------------------------------------
def section_closing(b):
    b.md("""## Closing — how to combine these in your work

Three patterns emerge from the catalog and the inline demos:

1. **Pick the right tool for the task.** Each method specializes. Cellpose for general cell segmentation; CARE for paired denoising; N2V for self-supervised denoising; fnet/pix2pix for translation; Deep-STORM for super-resolution localization; YOLOv2 for bounding-box detection. The catalog above is the menu.

2. **Use community platforms instead of writing custom code.** ZCD4M and DL4ME exist so you don't reinvent training pipelines. Their Colab notebooks handle data loading, augmentation, training schedules, evaluation, and export. Use them.

3. **Browse the BioImage Model Zoo first.** Before training your own model, check whether someone already trained one for your problem. The BiMZ API (Section 2) lets you list, filter, and load models live. If a model exists, you save weeks.

**Where to go next:**

- **For a specific method**, click through to its ZCD4M Colab in the Catalog (Section 1). Run the production version on your own data.
- **For browsing**, use the BiMZ filter cells in Section 2 to find pretrained models matching your task.
- **For research-level work**, DL4MicEverywhere offers Docker containers that run the same methods on local hardware, HPC, or cloud — production-grade reproducibility.

Workshop attendees: copy this notebook URL and bookmark it. The catalog continues being useful long after the workshop.""")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_notebook_04():
    b = CellBuilder("n04")
    section_title(b)
    section_catalog(b)
    section_biomodel_zoo(b)
    demo_care(b)
    demo_fnet(b)
    demo_pix2pix(b)
    demo_deepstorm(b)
    demo_yolo(b)
    section_closing(b)
    build_notebook(b.cells, "04_community_platforms")


if __name__ == "__main__":
    build_notebook_04()
