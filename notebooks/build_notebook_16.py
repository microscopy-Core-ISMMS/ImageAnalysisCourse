"""
Build script for Notebook 16 — WSI → Transcriptomics (Frontier Demo).

Frontier-status notebook covering the prediction of gene expression and gene
signatures from H&E whole-slide imagery. Architecture follows the SEQUOIA /
DeepPT family: foundation-model embedding extraction → linear/MLP head per
signature.

Foundation model: UNI (Mahmood Lab, gated on HuggingFace) primary; DINOv2
(Meta, ungated) fallback. Tiles: synthetic H&E-like patches by default; option
to load STimage-1K4M samples if the user has them locally.

Run:
    python build_notebook_16.py
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
# Title + orientation
# ---------------------------------------------------------------------------
def section_title(b):
    b.md("""<!-- colab-badge -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/notebooks/16_wsi_transcriptomics.ipynb)

*Click the badge to open this notebook in Google Colab. CPU is fine for the DINOv2 fallback path; T4 GPU recommended if loading UNI.*""")

    b.md("""# Notebook 16 — WSI → Transcriptomics (Frontier Demo)

**Status.** Frontier exploration. The papers behind this notebook were published in 2024–2025; the field is moving fast.
**Estimated time.** 15–20 minutes on Colab CPU; 8–12 minutes on T4 GPU.
**Prerequisites.** Notebook 03b (foundation models for segmentation), Notebook 06 (cross-modality prediction patterns), Notebook 13 (validation case study mindset).

> **⚠ Frontier-status notice.** Predicting gene expression from H&E histology is one of the most striking developments in computational pathology. The architecture is now well-established (foundation-model embedding + lightweight prediction head), but **clinical deployment is research-grade only**. Read this notebook to understand the recipe and what's now possible — but treat any prediction here as a teaching demo, not a clinical readout.

**Learning goals.**

1. Understand the **WSI → transcriptomics architecture**: tile a slide → foundation-model embedding per tile → aggregate / regress to a gene signature or per-gene expression.
2. Run the pipeline end-to-end on H&E-like tiles: load tiles, extract embeddings, predict a small panel of gene signatures, visualize predictions vs. (simulated) ground truth.
3. Compare a **pathology-specialized** foundation model (UNI) with a **general-vision** foundation model (DINOv2) — in this NB the latter is a fallback, but the comparison itself is the lesson.
4. See where the architecture is **honest** (interpretable, generalizable) and where it's **risky** (out-of-distribution slides, ethically loaded predictions, no causal claims).
5. Know where to read more (SEQUOIA, DeepPT, DeepSpot, UNI, CONCH, Prov-GigaPath, STimage-1K4M).

---""")


# ---------------------------------------------------------------------------
# Frontier banner + the recipe
# ---------------------------------------------------------------------------
def section_banner(b):
    b.md("""## ⚠ Frontier-status banner — this is a 2024–2026 capability

Three things to know before reading the rest:

1. **The recipe is now standard.** Foundation model trained on millions of tiles → frozen as feature extractor → small head learns per-gene or per-signature prediction. Used in SEQUOIA (Nat Comm 2024), DeepPT/ENLIGHT-DeepPT (Nat Cancer 2024), DeepSpot (2025 preprint), and aMIL-style PathAI work (2025).
2. **Foundation models are gated.** UNI, CONCH, and Prov-GigaPath all live on HuggingFace behind license-acceptance. This notebook tries UNI first; if the gate fails, it falls back to DINOv2 (ungated, but trained on natural images, not pathology). The architecture is the same — only the embedding quality differs.
3. **Ground truth here is simulated** so the notebook is reproducible without downloading STimage-1K4M (~2 GB) or TCGA WSIs (~10 GB each). The pipeline is real; the targets are made up. To run on real paired data, see the references at the end.""")

    b.md("""## The recipe in one paragraph

A whole-slide image (WSI) of an H&E-stained tissue section is huge — 50,000 × 50,000 pixels, gigabytes per slide — but only a tiny fraction of those pixels carry useful tissue signal. We **tile** the slide into 224 × 224 patches, embed each tile through a pre-trained vision foundation model into a ~1024-dim feature vector, then **aggregate** those tile-features (mean pool, attention pool, or learned aggregator) into one slide-level vector. A small linear or MLP head maps the slide-level vector to gene-expression targets — either bulk RNA-seq (one number per gene) or gene signatures (one number per pre-defined gene set, like "immune infiltration"). The foundation model stays frozen. Only the aggregator + head are trained on paired (slide, expression) data.

This NB does the same thing on **8 demo tiles** instead of a full WSI, and predicts **5 named signatures** instead of all 20,000 genes — but the code shape is identical to a SEQUOIA-style pipeline.""")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def section_setup(b):
    b.md("""## Setup""")

    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

# Core scientific stack
%pip install --quiet numpy matplotlib scikit-image scikit-learn scipy

# Vision foundation models. transformers + huggingface_hub for both UNI and DINOv2.
%pip install --quiet "transformers>=4.40" "huggingface_hub>=0.23" "timm>=0.9"

# UMAP for embedding visualization
%pip install --quiet "umap-learn>=0.5"

import os, time, math, traceback
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

print("Setup done.")""")

    b.code("""# GPU detection — informational only; both UNI and DINOv2 run on CPU (slowly).
try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        device = "cpu"
        print("Running on CPU. UNI/DINOv2 inference on 8 tiles takes ~15-30 s here.")
except ImportError:
    device = "cpu"
    print("torch not installed; will use CPU paths only.")
""")


# ---------------------------------------------------------------------------
# Get H&E tiles — synthetic by default, opt-in to real
# ---------------------------------------------------------------------------
def section_tiles(b):
    b.md("""## Step 1 — Get H&E tiles

Real workflow: tile a `.svs` whole-slide image with `openslide-python` or `pyvips`, extract patches at the right magnification (typically 20× or 40×, 224 × 224 px). Filter out background tiles (mostly-white tiles).

This NB skips that step and gives you 8 H&E-like tiles directly:
- **Default:** synthetic tiles generated below (pink/purple eosin-and-hematoxylin appearance, with cell-like nuclei). Reproducible, no download. Works offline.
- **Opt-in:** load real tiles from a local folder (set `HE_TILES_DIR` to a path with `*.png` or `*.tif` files). Useful if you have STimage-1K4M or TCGA-derived patches handy.""")

    b.code("""# Synthetic H&E generator — eosin (pink) background with hematoxylin (purple/blue) nuclei.
# This is a teaching analogue, not a real H&E section. A real one has finer texture, more
# variability in stain intensity, and many more nuclei per field.

rng = np.random.default_rng(42)

def make_he_tile(seed=0, size=224, n_nuclei=30):
    rng = np.random.default_rng(seed)
    # Eosin background — pink, slightly mottled
    bg_h = 0.92 + rng.normal(0, 0.04, (size, size))
    bg_h = np.clip(bg_h, 0.6, 1.0)
    # RGB: pink = (high R, mid G, mid B)
    img = np.stack([0.95 * bg_h, 0.75 * bg_h, 0.85 * bg_h], axis=-1)
    # Add hematoxylin-stained nuclei — purple/dark blue ovals
    centers = rng.uniform(15, size - 15, (n_nuclei, 2))
    radii = rng.uniform(4, 9, n_nuclei)
    for (cy, cx), r in zip(centers, radii):
        Y, X = np.ogrid[:size, :size]
        nucleus = ((Y - cy) ** 2 + (X - cx) ** 2) <= r ** 2
        # Hematoxylin nuclei: dark, slightly blue/purple
        for ch, val in [(0, 0.35), (1, 0.20), (2, 0.55)]:
            img[..., ch][nucleus] = val + rng.normal(0, 0.05)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


HE_TILES_DIR = ""  # set to "/content/my_he_tiles" to load real PNG/TIFF tiles

he_tiles = None
if HE_TILES_DIR and os.path.isdir(HE_TILES_DIR):
    try:
        from PIL import Image
        files = sorted(
            f for f in os.listdir(HE_TILES_DIR)
            if f.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg"))
        )[:8]
        he_tiles = [np.array(Image.open(os.path.join(HE_TILES_DIR, f)).convert("RGB"))
                    for f in files]
        print(f"Loaded {len(he_tiles)} real tiles from {HE_TILES_DIR}.")
    except Exception:
        print(f"Could not load from {HE_TILES_DIR}; falling back to synthetic.")
        traceback.print_exc(limit=2)
        he_tiles = None

if he_tiles is None:
    # Vary the seed and nucleus count so tiles look different
    he_tiles = [make_he_tile(seed=i, n_nuclei=25 + i * 4) for i in range(8)]
    print(f"Using {len(he_tiles)} synthetic H&E tiles.")
""")

    b.code("""# Visualize the tiles
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(he_tiles[i])
    ax.set_title(f"Tile {i}", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()""")


# ---------------------------------------------------------------------------
# Foundation embedding — UNI primary, DINOv2 fallback
# ---------------------------------------------------------------------------
def section_embedding(b):
    b.md("""## Step 2 — Foundation-model embedding

**Two paths, same architecture.**

- **Primary: UNI (Mahmood Lab, Nature Medicine 2024).** Vision Transformer (ViT-L/16) trained on 100,000+ pathology slides. Gated on HuggingFace as `MahmoodLab/UNI` — you must accept the license at <https://huggingface.co/MahmoodLab/UNI> and provide a HuggingFace access token below.
- **Fallback: DINOv2 (Meta, 2023).** ViT-Base trained on 142M natural images via self-supervised learning. Ungated as `facebook/dinov2-base`. Not pathology-specialized, but produces useful general-purpose visual features.

Both return a single feature vector per tile. Architectures of SEQUOIA and DeepPT use UNI-class embeddings; the rest of the pipeline is otherwise identical.""")

    b.code("""# Try UNI first; on any failure (gate, missing token, network), fall back to DINOv2.

import torch
import torch.nn.functional as F

HF_TOKEN = ""  # @param {type:"string"}  paste your HuggingFace token here for UNI access
USE_UNI = bool(HF_TOKEN)

embedding_model = None
embedding_name = None
embedding_dim = None
preprocess = None

if USE_UNI:
    try:
        # UNI loading recipe from the Mahmood Lab repo
        from huggingface_hub import login
        import timm
        from torchvision import transforms

        login(token=HF_TOKEN, add_to_git_credential=False)
        embedding_model = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        embedding_model.eval().to(device)
        embedding_dim = embedding_model.num_features  # 1024 for UNI ViT-L/16
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        embedding_name = "UNI (MahmoodLab/UNI, ViT-L/16, pathology-specialized)"
        print(f"Loaded {embedding_name}, dim={embedding_dim}.")
    except Exception:
        print("UNI failed (gate, token, or network). Falling back to DINOv2.")
        traceback.print_exc(limit=3)
        embedding_model = None

if embedding_model is None:
    # DINOv2 fallback path. Ungated, ~340 MB download.
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    embedding_model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)
    embedding_dim = embedding_model.config.hidden_size  # 768 for ViT-B
    embedding_name = "DINOv2 (facebook/dinov2-base, ViT-B/14, general vision)"

    def _preprocess(img_np):
        # Returns a tensor of shape (3, 224, 224)
        out = proc(images=img_np, return_tensors="pt")
        return out["pixel_values"][0]

    preprocess = _preprocess
    print(f"Loaded {embedding_name}, dim={embedding_dim}.")
""")

    b.code("""# Run inference on the 8 tiles -> embeddings shape (8, embedding_dim)
import torch

embeddings = []
t0 = time.time()
with torch.no_grad():
    for tile in he_tiles:
        x = preprocess(tile).unsqueeze(0).to(device)
        if "DINOv2" in embedding_name:
            out = embedding_model(pixel_values=x)
            # Use CLS token (first hidden state)
            emb = out.last_hidden_state[:, 0, :]
        else:  # UNI
            emb = embedding_model(x)
        embeddings.append(emb.cpu().numpy().squeeze(0))
embeddings = np.stack(embeddings)
print(f"Embedded {len(he_tiles)} tiles in {time.time() - t0:.1f}s. Shape: {embeddings.shape}")
print(f"Per-tile feature norm range: [{np.linalg.norm(embeddings, axis=1).min():.2f}, {np.linalg.norm(embeddings, axis=1).max():.2f}]")
""")


# ---------------------------------------------------------------------------
# Visualize embedding space
# ---------------------------------------------------------------------------
def section_viz_embedding(b):
    b.md("""## Step 3 — Visualize the embedding space

A useful sanity check: project the high-dim embeddings to 2D with UMAP (or PCA as a fallback). Tiles that look similar to the foundation model land near each other.

With 8 tiles this is a tiny demonstration; on real WSI workflows you'd embed thousands of tiles per slide and the UMAP becomes informative.""")

    b.code("""# 2D projection. Use PCA (deterministic, works for any N) — UMAP needs more samples.
n_show = embeddings.shape[0]
if n_show >= 4:
    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(proj[:, 0], proj[:, 1], s=80, c=range(n_show), cmap="viridis", edgecolor="k")
    for i, (x, y) in enumerate(proj):
        ax.annotate(f"tile {i}", (x, y), fontsize=10,
                    xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)")
    ax.set_title(f"Tile embeddings — {embedding_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"PC1+PC2 explain {sum(var_explained)*100:.1f}% of variance across {n_show} tiles.")
else:
    print("Need at least 4 tiles for a meaningful PCA projection.")
""")


# ---------------------------------------------------------------------------
# Gene-signature prediction — train a tiny linear head
# ---------------------------------------------------------------------------
def section_signatures(b):
    b.md("""## Step 4 — Gene-signature prediction

A **gene signature** is a named gene set summarized into one number — for example, "immune infiltration" pools dozens of immune-cell-marker genes into a single per-sample score. SEQUOIA-style models predict either individual genes (ridge regression per gene) or a small panel of curated signatures (multi-output regression).

We define **5 illustrative signatures** and **simulate** their values from each tile. Real workflow: you'd load paired (tile, expression) measurements from STimage-1K4M, the SEQUOIA training data, or an in-house cohort.

> **⚠ The ground truth in this NB is fake.** The signatures are real concepts; the values are computed from each tile's pixel statistics so the linear head has *something* to learn. This is purely to demonstrate the architecture. Do **not** read clinical meaning into any predicted number below.""")

    b.code("""SIGNATURES = [
    "immune_infiltration",
    "hypoxia",
    "EMT",
    "proliferation",
    "stromal_content",
]


def simulate_targets(tiles):
    \"\"\"Compute simulated 'ground-truth' signature values from tile pixel statistics.

    Each signature is a deterministic function of features we can read from the
    tile. Real targets would come from paired RNA-seq.
    \"\"\"
    n = len(tiles)
    targets = np.zeros((n, len(SIGNATURES)))
    for i, t in enumerate(tiles):
        gray = t.astype(float).mean(axis=-1) / 255.0
        # immune_infiltration: more dark spots (nuclei) -> higher
        targets[i, 0] = (gray < 0.4).mean()
        # hypoxia: darker mean intensity -> higher
        targets[i, 1] = 1.0 - gray.mean()
        # EMT: more local variance (texture) -> higher
        targets[i, 2] = float(np.std(gray))
        # proliferation: small dark spots count
        from scipy.ndimage import label
        labels, _ = label(gray < 0.45)
        targets[i, 3] = labels.max() / 50.0
        # stromal_content: pinker (more red than blue) -> higher
        r, b = t[..., 0].mean(), t[..., 2].mean()
        targets[i, 4] = max(0, (r - b) / 255.0)
    # Normalize each column to [0, 1] for plotting
    mins = targets.min(axis=0, keepdims=True)
    maxs = targets.max(axis=0, keepdims=True)
    rng_ = maxs - mins
    rng_[rng_ == 0] = 1
    targets = (targets - mins) / rng_
    # Add a little noise so it's not perfectly fittable
    targets += 0.05 * np.random.default_rng(0).standard_normal(targets.shape)
    return np.clip(targets, 0, 1)


targets = simulate_targets(he_tiles)
print(f"Simulated targets shape: {targets.shape}  (n_tiles, n_signatures)")
print(f"Signatures: {SIGNATURES}")
print(f"Per-signature range: {[(s, f'[{targets[:, i].min():.2f}, {targets[:, i].max():.2f}]') for i, s in enumerate(SIGNATURES)]}")
""")

    b.md("""**Predict before you run.** We have 8 tile embeddings (1024-d for UNI / 768-d for DINOv2) and 8 simulated 5-d targets. We'll train a Ridge regression with leave-one-out cross-validation and report per-signature R².

What do you expect?
- (a) R² > 0.9 across all signatures (the simulated targets ARE deterministic functions of the tiles, so a good embedding should crack this).
- (b) R² near zero (8 samples is too few to fit 1024 features without massive overfitting; LOOCV will catch it).
- (c) Mixed — texture-based signatures fit well, others don't.""")

    b.code("""from sklearn.model_selection import LeaveOneOut

# Multi-output Ridge with strong regularization (we have 8 samples << 1024 features)
loo = LeaveOneOut()
preds_loocv = np.zeros_like(targets)

for train_idx, test_idx in loo.split(embeddings):
    X_tr, X_te = embeddings[train_idx], embeddings[test_idx]
    y_tr = targets[train_idx]
    model = Ridge(alpha=10.0)  # heavy regularization for tiny n
    model.fit(X_tr, y_tr)
    preds_loocv[test_idx] = model.predict(X_te)

# Per-signature R²
print(f"Leave-one-out R² scores ({embedding_name}):")
for i, sig in enumerate(SIGNATURES):
    r2 = r2_score(targets[:, i], preds_loocv[:, i])
    print(f"  {sig:25s}  R² = {r2:+.3f}")
print()
print("Negative R² means the embedding + Ridge predicts WORSE than the mean. Expected with 8 samples.")
""")


# ---------------------------------------------------------------------------
# Predictions vs ground truth
# ---------------------------------------------------------------------------
def section_predictions(b):
    b.md("""## Step 5 — Predictions vs (simulated) ground truth

Per-signature scatter: each dot is a tile. Diagonal = perfect prediction.""")

    b.code("""fig, axes = plt.subplots(1, len(SIGNATURES), figsize=(4 * len(SIGNATURES), 4))
for i, (ax, sig) in enumerate(zip(axes, SIGNATURES)):
    ax.scatter(targets[:, i], preds_loocv[:, i], s=60, edgecolor="k")
    lim_lo = min(targets[:, i].min(), preds_loocv[:, i].min()) - 0.05
    lim_hi = max(targets[:, i].max(), preds_loocv[:, i].max()) + 0.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.4)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    r2 = r2_score(targets[:, i], preds_loocv[:, i])
    ax.set_title(f"{sig}\\nR² = {r2:+.2f}", fontsize=10)
    ax.set_xlabel("True (simulated)")
    if i == 0:
        ax.set_ylabel("LOOCV prediction")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""")

    b.md("""**What to read into this.** With only 8 tiles and 1024-dim embeddings, LOOCV R² is going to be noisy or even negative for some signatures — that's expected, not a bug. The point is the **architecture works**: embedding → linear head → per-signature score. SEQUOIA trains its head on **thousands of paired slide-expression examples**; that's where the real generalization comes from.

The take-home from this small demo: the embedding either *contains* information about the target (R² goes up) or *doesn't* (R² stays near zero). Pathology-specialized models like UNI tend to contain more of the right pathology signal than DINOv2 — which is the whole point of the gated, license-restricted pre-training.""")


# ---------------------------------------------------------------------------
# Saliency / interpretability
# ---------------------------------------------------------------------------
def section_saliency(b):
    b.md("""## Step 6 — What part of the tile drove the prediction?

A common interpretability check: compute the **input gradient** of the predicted signature score with respect to the input image. Pixels with large gradient magnitude contributed most to the prediction. Heatmap overlay shows which tile regions the foundation model is "looking at" for a given signature.

This is not the same as a causal explanation, but it's the standard sanity check (e.g., "does the model focus on nuclei when predicting an immune signature, or on stroma when predicting EMT?").""")

    b.code("""# Pick a signature and a tile to inspect
target_sig_idx = 0  # @param {type: "slider", min: 0, max: 4, step: 1}
target_tile_idx = 0  # @param {type: "slider", min: 0, max: 7, step: 1}

print(f"Saliency for signature='{SIGNATURES[target_sig_idx]}' on tile {target_tile_idx}.")

# Re-fit a Ridge head on all 8 tiles (we'll use it to score the chosen tile)
ridge_full = Ridge(alpha=10.0).fit(embeddings, targets)
# Coefficient vector for the chosen signature
w = ridge_full.coef_[target_sig_idx]  # shape (embedding_dim,)
b_ = ridge_full.intercept_[target_sig_idx]

# Compute gradient of (w @ embedding(tile)) w.r.t. input pixels via torch autograd
tile = he_tiles[target_tile_idx]
x = preprocess(tile).unsqueeze(0).to(device).requires_grad_(True)

if "DINOv2" in embedding_name:
    out = embedding_model(pixel_values=x)
    emb_t = out.last_hidden_state[:, 0, :]
else:
    emb_t = embedding_model(x)

score = (emb_t * torch.tensor(w, dtype=emb_t.dtype, device=device)).sum() + float(b_)
score.backward()

# Saliency: aggregate channel-wise abs gradient
grad = x.grad.detach().cpu().numpy().squeeze(0)  # (3, 224, 224)
saliency = np.abs(grad).max(axis=0)  # max across channels
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-9)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(tile)
axes[0].set_title(f"Tile {target_tile_idx} (input)")
axes[0].axis("off")
axes[1].imshow(tile)
axes[1].imshow(saliency, cmap="hot", alpha=0.5)
axes[1].set_title(f"Saliency for '{SIGNATURES[target_sig_idx]}' (predicted score = {score.item():.3f})")
axes[1].axis("off")
plt.tight_layout()
plt.show()
""")

    b.md("""**Read the heatmap with care.** Bright pixels are where small changes in the input would change the prediction the most. They are *not* a guarantee that the model "understood" those pixels biologically — only that they were leveraged by the linear head. For real interpretability claims, pair this with attention rollout (transformer-specific), feature attribution methods like Integrated Gradients, or pathologist review of the highlighted regions.""")


# ---------------------------------------------------------------------------
# Discussion + ethical caveats
# ---------------------------------------------------------------------------
def section_discussion(b):
    b.md("""## Step 7 — Discussion: what works, what doesn't

**What this architecture actually delivers, end-to-end (per the literature):**

| Strength | Source |
|---|---|
| Bulk RNA-seq prediction across 16 TCGA cancer types | SEQUOIA (Nat Comm 2024) |
| Treatment-response prediction from imputed transcriptomics | DeepPT / ENLIGHT-DeepPT (Nat Cancer 2024) |
| Spatial gene-expression maps from H&E (per-tile) | DeepSpot (2025 preprint) |
| Interpretable gene-signature heatmaps | aMIL PathAI 2025 |

**What the architecture does NOT deliver:**

- **No causal claims.** Predicting expression from histology does not mean histology *causes* expression — both reflect the same underlying biology. The model learns a correlation in a training cohort.
- **No clinical use without validation.** Every paper above flags the gap between research-grade prediction and clinical deployment. Predictions on out-of-distribution slides (different scanner, different stain protocol, different population) degrade unpredictably.
- **No generalization without diverse training.** A model trained on one cancer cohort doesn't transfer to a different cancer. Foundation models help (UNI was trained on diverse pathology) but the prediction head is cohort-specific.
- **No replacement for sequencing.** When the molecular readout matters clinically, sequence the sample. The image-based prediction is a screening tool or a way to recover signal from archival H&E where RNA was never collected.

**Ethical notes when reporting WSI-transcriptomics work:**

- Always state that predictions are **research-grade**.
- Report **out-of-distribution performance** (different cohort, different scanner) — not just held-out test set from the training cohort.
- Be careful with predictions that intersect with sensitive attributes (genetic ancestry, sex, ethnicity).
- If publishing a model: make the **training data composition** explicit, including site/scanner distribution.""")


# ---------------------------------------------------------------------------
# References + closing reflection
# ---------------------------------------------------------------------------
def section_references(b):
    b.md("""## Resources

**Bulk RNA-seq from H&E**
- SEQUOIA (Lim et al., *Nat Communications* 2024): <https://www.nature.com/articles/s41467-024-54017-3>
- DeepPT / ENLIGHT-DeepPT (Hoang et al., *Nat Cancer* 2024)
- Schmauch et al. 2020 (predecessor work) — earlier WSI→RNA mapping
- tRNAsformer 2023 — transformer-based WSI→RNA

**Spatial transcriptomics from H&E**
- DeepSpot (2025 preprint) — predicts spatial transcriptomics with foundation-model embeddings + deep-set + multilevel context

**Pathology foundation models** (the embedding backbones)
- UNI (Chen et al., *Nat Medicine* 2024) — `MahmoodLab/UNI` (gated)
- CONCH (Lu et al., *Nat Medicine* 2024) — vision-language pathology FM
- Prov-GigaPath (Xu et al., *Nature* 2024) — Microsoft/Providence FM

**Gene-signature prediction**
- aMIL PathAI-style approach (2025) — bulk LRRC15+ TGFβ-CAF signatures with attention-MIL aggregation

**Paired image-expression resources**
- STimage-1K4M (2024) — large image–gene-expression resource for benchmarking

**Workshop dataset audit:** [`datasets_audit.md`](https://github.com/microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/datasets_audit.md) covers BBBC, BIA, IDR, Allen Cell, BSCCM, Cellpose, GigaDB, CIL — none specifically histology, but the same evaluation pattern applies if you adopt one of the WSI sources above.""")

    b.md("""## Closing reflection

This notebook ran end-to-end on 8 tiles. Real WSI→transcriptomics work scales the same architecture to thousands of tiles per slide and thousands of slides per study. The foundation-model + linear-head recipe is now the default; the open questions are about **training data composition**, **generalization across cohorts**, and **clinical actionability**.

If you're starting work in this area: clone SEQUOIA, request UNI access, pick one of the published cohorts (TCGA, CPTAC, Tempus), and replicate one of the cancer-type prediction results before extending to your own data. The architecture is not the bottleneck; the data and validation are.

> **Reminder:** the predictions in this notebook used **simulated ground-truth values** computed from each tile's pixel statistics. The pipeline is real; the targets are not. Do not interpret the predicted "immune infiltration" or "hypoxia" scores as biology.""")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def main():
    b = CellBuilder("nb16")
    section_title(b)
    section_banner(b)
    section_setup(b)
    section_tiles(b)
    section_embedding(b)
    section_viz_embedding(b)
    section_signatures(b)
    section_predictions(b)
    section_saliency(b)
    section_discussion(b)
    section_references(b)
    build_notebook(b.cells, "16_wsi_transcriptomics")


if __name__ == "__main__":
    main()
