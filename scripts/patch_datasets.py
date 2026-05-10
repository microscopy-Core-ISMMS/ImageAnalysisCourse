#!/usr/bin/env python3
"""Apply dataset audit patches to all 17 notebooks.

Two patch tiers:
  - T1 (executable): inserts a markdown + code cell PAIR before the existing
    synthetic-data section. The code tries to fetch a canonical real dataset
    over the network and falls back cleanly to the existing synthetic path on
    any failure.
  - T2 (reference): appends a single markdown cell at the end of the NB
    listing canonical published datasets that fit the topic.

Idempotent: each inserted cell carries a sentinel (`<!-- DATASET-AUDIT-PATCH -->`
in markdown, `# DATASET-AUDIT-PATCH` in code) so re-running the script replaces
the prior patch instead of stacking duplicates.

Run from the repo root:
    python3 scripts/patch_datasets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "notebooks"

# Sentinels used to detect prior patch cells so the script is re-runnable.
# Old (pre-2026-05-08): 4 cells per T1 NB.
MD_SENTINEL = "<!-- DATASET-AUDIT-PATCH -->"
CO_SENTINEL = "# DATASET-AUDIT-PATCH"
SWAP_MD_SENTINEL = "<!-- DATASET-AUDIT-PATCH-SWAP -->"
SWAP_CO_SENTINEL = "# DATASET-AUDIT-PATCH-SWAP"
# New (2026-05-08+): 2 cells per T1 NB — one decision block instead of separate
# download + swap. Old sentinels are still caught by _strip_prior_patch so a
# notebook with the old 4-cell pattern migrates cleanly to the new 2-cell pattern.
DECISION_MD_SENTINEL = "<!-- DATA-DECISION -->"
DECISION_CO_SENTINEL = "# DATA-DECISION"

# MABC samples are hosted on the gh-pages branch alongside the JB build.
# URL pattern: <gh-pages root>/data/mabc/<nb_id>.npz
MABC_URL_BASE = "https://microscopy-core-ismms.github.io/ImageAnalysisCourse/data/mabc"

# T0 reference notebook URL (for the "My own data" tier).
T0_URL = "https://microscopy-core-ismms.github.io/ImageAnalysisCourse/notebooks/00_data_sources.html"

# ---------------------------------------------------------------------------
# Tier 1 specs — executable real-data cells
# ---------------------------------------------------------------------------

T1_SPECS = {
    "01_cellpose_segmentation.ipynb": {
        "anchor_text": "## Generate the working dataset",
        "dataset_name": "BBBC020 — Murine bone-marrow derived macrophages",
        "license_note": "CC0",
        "citation": "Ljosa et al., Nature Methods, 2012 — BBBC020",
        "source_url": "https://bbbc.broadinstitute.org/BBBC020",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip",
        "what_it_is": "20 fields of murine bone-marrow derived macrophages (DAPI + CD11b + F-actin); paired ground-truth outlines available.",
        "swap_vars": ["img_easy", "img_hard", "img_easy_synth", "img_hard_synth", "img"],
        "swap_code": (
            "if real_imgs and len(real_imgs) >= 2:\n"
            "    img_easy = real_imgs[0]\n"
            "    img_hard = real_imgs[1]\n"
            "    img_easy_synth = real_imgs[0]\n"
            "    img_hard_synth = real_imgs[1]\n"
            "    img = real_imgs[0]\n"
            "    print('img_easy / img_hard / *_synth now bound to BBBC020 real images (real_imgs[0] and real_imgs[1]).')\n"
            "    print(\"NOTE: 'What you should be seeing' callouts were written for synthetic; counts and shapes will differ.\")\n"
            "else:\n"
            "    print('real_imgs is None or has <2 images; staying with synthetic.')\n"
        ),
    },
    "02_validation_quantification.ipynb": {
        "anchor_text": "## Generate ground truth",
        "dataset_name": "BBBC005 v1 ground truth — synthetic cells with paired binary masks",
        "license_note": "CC0 (Anne Carpenter waiver)",
        "citation": "Lehmussola et al., IEEE T. Med. Imaging, 2007; Bray et al., J. Biomol. Screen, 2011",
        "source_url": "https://bbbc.broadinstitute.org/BBBC005",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip",
        "what_it_is": "1,200 in-focus synthetic cell images with paired binary foreground/background masks. Ideal for IoU/Dice validation.",
        "swap_vars": ["gt_easy", "gt_hard"],
        "swap_code": (
            "if real_imgs and len(real_imgs) >= 2:\n"
            "    import numpy as _np\n"
            "    gt_easy = (_np.asarray(real_imgs[0]) > 0).astype(_np.uint8)\n"
            "    gt_hard = (_np.asarray(real_imgs[-1]) > 0).astype(_np.uint8)\n"
            "    print('gt_easy / gt_hard now from BBBC005 real binary masks.')\n"
            "    print(\"NOTE: 'pred_*' is still synthetic from Lab 1; rerun Lab 1 with real data first if you want a real-vs-real comparison. IoU/Dice numbers will differ from the callouts.\")\n"
            "else:\n"
            "    print('real_imgs is None or insufficient; staying with synthetic.')\n"
        ),
    },
    "03a_denoising_n2v.ipynb": {
        "anchor_text": "**The Noise2Void principle.**",
        "dataset_name": "BBBC020 — Murine bone-marrow derived macrophages (real fluorescence, used as clean reference)",
        "license_note": "CC0",
        "citation": "Ljosa et al., Nature Methods, 2012 — BBBC020",
        "source_url": "https://bbbc.broadinstitute.org/BBBC020",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip",
        "what_it_is": "Real fluorescence as the 'clean' reference. We add controlled synthetic noise on top so we can still measure PSNR/SSIM against a known truth.",
        "swap_vars": ["clean", "noisy"],
        "swap_code": (
            "if real_imgs:\n"
            "    import numpy as _np\n"
            "    _src = _np.asarray(real_imgs[0]).astype(float)\n"
            "    if _src.ndim == 3:\n"
            "        _src = _src.mean(axis=-1) if _src.shape[-1] in (3, 4) else _src[_src.shape[0]//2]\n"
            "    _src = (_src - _src.min()) / (_src.max() - _src.min() + 1e-9)\n"
            "    clean = _src\n"
            "    _rng_real = _np.random.default_rng(0)\n"
            "    noisy = clean + 0.10 * _rng_real.standard_normal(clean.shape)\n"
            "    print('clean / noisy now derived from BBBC020 real fluorescence + simulated Gaussian noise (sigma=0.10).')\n"
            "    print(\"NOTE: synthetic Gaussian noise is a *teaching analogue*. Real microscopy noise has Poisson + read components; for true denoising benchmarks see GigaDB 100888.\")\n"
            "else:\n"
            "    print('real_imgs is None; staying with synthetic.')\n"
        ),
    },
    "03b_foundation_model_segmentation.ipynb": {
        "anchor_text": "## Load a non-canonical microscopy image",
        "dataset_name": "BBBC020 — Murine bone-marrow derived macrophages",
        "license_note": "CC0",
        "citation": "Ljosa et al., Nature Methods, 2012 — BBBC020",
        "source_url": "https://bbbc.broadinstitute.org/BBBC020",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip",
        "what_it_is": "Multi-channel macrophage fluorescence — exactly the 'non-canonical microscopy' content SAM/μSAM are tested against.",
        "swap_vars": ["img"],
        "swap_code": (
            "if real_imgs:\n"
            "    import numpy as _np\n"
            "    _src = _np.asarray(real_imgs[0]).astype(float)\n"
            "    if _src.ndim == 3:\n"
            "        _src = _src.mean(axis=-1) if _src.shape[-1] in (3, 4) else _src[_src.shape[0]//2]\n"
            "    img = (_src - _src.min()) / (_src.max() - _src.min() + 1e-9)\n"
            "    print('img is now from real data (BBBC020). SAM cells below will run on real microscopy.')\n"
            "    print(\"NOTE: 'What you should be seeing' callouts were written for the synthetic image; SAM mask shapes will differ.\")\n"
            "else:\n"
            "    print('real_imgs is None; staying with synthetic.')\n"
        ),
    },
    "06_virtual_staining.ipynb": {
        "anchor_text": "## Method 1 — fnet-style U-Net (paired, regression)",
        "dataset_name": "BBBC020 — Murine bone-marrow derived macrophages (multi-channel fluorescence)",
        "license_note": "CC0",
        "citation": "Ljosa et al., Nature Methods, 2012 — BBBC020",
        "source_url": "https://bbbc.broadinstitute.org/BBBC020",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip",
        "what_it_is": "Real multi-channel fluorescence used as a *teaching analogue* for cross-channel prediction (input channel → target channel). True virtual-staining datasets have brightfield/phase as input — see Allen Cell Imaging Collections in the audit for that.",
        "swap_vars": ["X_train", "Y_train", "X_test", "Y_test"],
        "swap_code": (
            "# MABC NB06 ships paired DAPI (real_imgs) + Tubulin (real_labels) sub-tiles\n"
            "# from DrosophilaCells. 12 paired tiles total (3 fly samples sub-tiled 4 ways).\n"
            "if real_imgs and len(real_imgs) >= 4:\n"
            "    import numpy as _np\n"
            "    def _to2d(im):\n"
            "        a = _np.asarray(im).astype(_np.float32)\n"
            "        if a.ndim == 3:\n"
            "            a = a.mean(axis=-1) if a.shape[-1] in (3,4) else a[a.shape[0]//2]\n"
            "        rng = a.max() - a.min()\n"
            "        if rng < 1e-9:\n"
            "            return _np.zeros_like(a, dtype=_np.float32)\n"
            "        return ((a - a.min()) / rng).astype(_np.float32)\n"
            "    _inputs = [_to2d(im) for im in real_imgs]\n"
            "    # Use Tubulin labels as targets when MABC provided them; otherwise fall back\n"
            "    # to halving the inputs (canonical-tier path).\n"
            "    if globals().get('real_labels') is not None and len(real_labels) >= len(_inputs):\n"
            "        _targets = [_to2d(t) for t in real_labels[:len(_inputs)]]\n"
            "    else:\n"
            "        _half = len(_inputs) // 2\n"
            "        _targets = _inputs[_half:_half*2] + _inputs[:_half]\n"
            "    _n = min(len(_inputs), len(_targets))\n"
            "    _split = max(1, int(_n * 0.75))\n"
            "    X_train = _np.stack(_inputs[:_split])\n"
            "    Y_train = _np.stack(_targets[:_split])\n"
            "    X_test = _np.stack(_inputs[_split:]) if _n - _split > 0 else X_train[-1:]\n"
            "    Y_test = _np.stack(_targets[_split:]) if _n - _split > 0 else Y_train[-1:]\n"
            "    print(f'X_train (DAPI) {X_train.shape} {X_train.dtype} / Y_train (Tubulin) {Y_train.shape} from MABC DrosophilaCells.')\n"
            "    print(f'X_test {X_test.shape} / Y_test {Y_test.shape}.')\n"
            "    print('Real cross-channel pairs (DAPI->Tubulin). TinyUNet/pix2pix will train from scratch on this small set.')\n"
            "else:\n"
            "    print('real_imgs is None or has <4 images; staying with cells3d() pairs above.')\n"
        ),
    },
    "07_widefield_superres.ipynb": {
        "anchor_text": "## One paired example: HR ground truth, LR widefield input, bicubic baseline",
        "dataset_name": "BBBC020 — Murine bone-marrow derived macrophages (used as HR; we synthesize LR by Gaussian blur + downsample)",
        "license_note": "CC0",
        "citation": "Ljosa et al., Nature Methods, 2012 — BBBC020",
        "source_url": "https://bbbc.broadinstitute.org/BBBC020",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip",
        "what_it_is": "Real fluorescence (DAPI + CD11b + F-actin macrophages) used as HR ground truth; LR is synthesized by Gaussian blur + downsample (the classic SR degradation model). For true paired widefield/SIM data, see CSBDeep CARE and ZeroCostDL4Mic.",
        "swap_vars": ["hr_train", "lr_train", "lr_train_small", "hr_test", "lr_test", "lr_test_small", "n_train", "n_test"],
        "swap_code": (
            "if real_imgs and len(real_imgs) >= 4:\n"
            "    import numpy as _np\n"
            "    from scipy.ndimage import gaussian_filter as _gf, zoom as _zoom\n"
            "    def _to_hr(im, target=128):\n"
            "        a = _np.asarray(im).astype(_np.float32)\n"
            "        if a.ndim == 3:\n"
            "            a = a.mean(axis=-1) if a.shape[-1] in (3,4) else a[a.shape[0]//2]\n"
            "        a = (a - a.min()) / (a.max() - a.min() + 1e-9)\n"
            "        s = min(a.shape)\n"
            "        a = a[:s, :s]\n"
            "        a = _zoom(a, target / s, order=1)\n"
            "        return a\n"
            "    # Split: 75% train, 25% test (with 8 real images = 6 train + 2 test).\n"
            "    _split = max(2, int(len(real_imgs) * 0.75))\n"
            "    hr_train = _np.stack([_to_hr(im) for im in real_imgs[:_split]])\n"
            "    hr_test = _np.stack([_to_hr(im) for im in real_imgs[_split:]])\n"
            "    lr_train_small = _np.stack([_zoom(_gf(h, sigma=2.0), 0.5, order=1) for h in hr_train])\n"
            "    lr_train = _np.stack([_zoom(s, 2.0, order=3) for s in lr_train_small])\n"
            "    lr_test_small = _np.stack([_zoom(_gf(h, sigma=2.0), 0.5, order=1) for h in hr_test])\n"
            "    lr_test = _np.stack([_zoom(s, 2.0, order=3) for s in lr_test_small])\n"
            "    n_train = len(hr_train); n_test = len(hr_test)\n"
            "    print(f'Bound HR/LR train ({n_train}) + test ({n_test}) from BBBC005 real fluorescence (LR via blur+downsample).')\n"
            "    print('NOTE: synthesized LR is a teaching analogue, not true widefield optics. With n_test=2 the metrics are noisy; the architecture is what is shown end-to-end.')\n"
            "    # Display all train + test images so you can see what loaded\n"
            "    import matplotlib.pyplot as _plt\n"
            "    _n_show = min(8, n_train + n_test)\n"
            "    _imgs = list(hr_train[:6]) + list(hr_test[:2])\n"
            "    _titles = [f'train {i}' for i in range(min(6, n_train))] + [f'test {i}' for i in range(min(2, n_test))]\n"
            "    _ncols = 4; _nrows = (_n_show + _ncols - 1) // _ncols\n"
            "    _fig, _axes = _plt.subplots(_nrows, _ncols, figsize=(3*_ncols, 3*_nrows))\n"
            "    for _ax, _im, _t in zip(_axes.flat, _imgs[:_n_show], _titles[:_n_show]):\n"
            "        _ax.imshow(_im, cmap='viridis'); _ax.set_title(_t, fontsize=9); _ax.axis('off')\n"
            "    for _ax in _axes.flat[_n_show:]:\n"
            "        _ax.axis('off')\n"
            "    _plt.tight_layout(); _plt.show()\n"
            "else:\n"
            "    print('real_imgs is None or insufficient (<4); will fall through to synthetic SR pairs.')\n"
        ),
    },
    "09_cellpose_finetune.ipynb": {
        "anchor_text": "## Step 1: Get a small labeled dataset",
        "dataset_name": "BBBC038v1 — 2018 Data Science Bowl nuclei (training subset)",
        "license_note": "CC0 (BBBC mirror of the Kaggle 2018 Data Science Bowl)",
        "citation": "Caicedo et al., Nature Methods, 2019 — BBBC038",
        "source_url": "https://bbbc.broadinstitute.org/BBBC038",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC038v1/BBBC038v1_train.zip",
        "what_it_is": "Diverse nuclei across modalities — designed for segmentation model training. Training subset only (smaller download).",
        "swap_vars": ["train_images", "train_labels", "test_images", "test_labels", "img"],
        "swap_code": (
            "if real_imgs and len(real_imgs) >= 6:\n"
            "    import numpy as _np\n"
            "    train_images = list(real_imgs[:6])\n"
            "    test_images = list(real_imgs[6:8]) if len(real_imgs) >= 8 else list(real_imgs[6:])\n"
            "    # BBBC038 train zip ships paired masks in mask/ subfolders, but the simple zip\n"
            "    # walker doesn't separate them. Initialize labels to zeros so the schema works;\n"
            "    # for true fine-tuning, run the baseline cells below first and use the predicted\n"
            "    # masks as starting labels.\n"
            "    train_labels = [_np.zeros(_np.asarray(im).shape[:2], dtype=_np.int32) for im in train_images]\n"
            "    test_labels = [_np.zeros(_np.asarray(im).shape[:2], dtype=_np.int32) for im in test_images]\n"
            "    img = _np.asarray(real_imgs[0])\n"
            "    print(f'train_images ({len(train_images)}) / test_images ({len(test_images)}) now from BBBC038 real nuclei.')\n"
            "    print(\"NOTE: train_labels/test_labels are zero-initialized. Run the baseline first to get predicted masks before fine-tuning.\")\n"
            "else:\n"
            "    print('real_imgs is None or has <6 images; staying with synthetic.')\n"
        ),
    },
    "12_deconvolution.ipynb": {
        "anchor_text": "## One paired example: clean → blurred → noisy",
        "dataset_name": "BBBC005 v1 ground truth — in-focus images (used as clean) + synthetic blur",
        "license_note": "CC0",
        "citation": "Lehmussola et al., IEEE T. Med. Imaging, 2007",
        "source_url": "https://bbbc.broadinstitute.org/BBBC005",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip",
        "what_it_is": "Real in-focus fluorescence as the clean ground truth; we apply a Gaussian PSF + Poisson/Gaussian noise to simulate the wide-field forward model.",
        "swap_vars": ["X_train_clean", "X_train_blurred", "X_train_blurred_noisy", "X_test_clean", "X_test_blurred", "X_test_blurred_noisy", "n_train", "n_test"],
        "swap_code": (
            "if real_imgs and len(real_imgs) >= 4:\n"
            "    import numpy as _np\n"
            "    from scipy.ndimage import gaussian_filter as _gf, zoom as _zoom\n"
            "    def _to_clean(im, target=64):\n"
            "        a = _np.asarray(im).astype(_np.float32)\n"
            "        if a.ndim == 3:\n"
            "            a = a.mean(axis=-1) if a.shape[-1] in (3,4) else a[a.shape[0]//2]\n"
            "        a = (a - a.min()) / (a.max() - a.min() + 1e-9)\n"
            "        s = min(a.shape)\n"
            "        a = a[:s, :s]\n"
            "        return _zoom(a, target / s, order=1)\n"
            "    # Split: 75% train, 25% test (with 8 real images = 6 train + 2 test).\n"
            "    _split = max(2, int(len(real_imgs) * 0.75))\n"
            "    X_train_clean = _np.stack([_to_clean(im) for im in real_imgs[:_split]])\n"
            "    X_test_clean = _np.stack([_to_clean(im) for im in real_imgs[_split:]])\n"
            "    X_train_blurred = _np.stack([_gf(c, sigma=2.0) for c in X_train_clean])\n"
            "    X_test_blurred = _np.stack([_gf(c, sigma=2.0) for c in X_test_clean])\n"
            "    _rng_d = _np.random.default_rng(7)\n"
            "    X_train_blurred_noisy = X_train_blurred + 0.05 * _rng_d.standard_normal(X_train_blurred.shape)\n"
            "    X_test_blurred_noisy = X_test_blurred + 0.05 * _rng_d.standard_normal(X_test_blurred.shape)\n"
            "    n_train = len(X_train_clean); n_test = len(X_test_clean)\n"
            "    print(f'Bound clean/blurred/blurred_noisy train ({n_train}) + test ({n_test}) from BBBC005 + synthetic Gaussian PSF.')\n"
            "    print('NOTE: PSF is approximate (Gaussian sigma=2.0). With n_test=2 the metrics are noisy.')\n"
            "    # Display loaded train+test\n"
            "    import matplotlib.pyplot as _plt\n"
            "    _n_show = min(8, n_train + n_test)\n"
            "    _imgs = list(X_train_clean[:6]) + list(X_test_clean[:2])\n"
            "    _titles = [f'train clean {i}' for i in range(min(6, n_train))] + [f'test clean {i}' for i in range(min(2, n_test))]\n"
            "    _ncols = 4; _nrows = (_n_show + _ncols - 1) // _ncols\n"
            "    _fig, _axes = _plt.subplots(_nrows, _ncols, figsize=(3*_ncols, 3*_nrows))\n"
            "    for _ax, _im, _t in zip(_axes.flat, _imgs[:_n_show], _titles[:_n_show]):\n"
            "        _ax.imshow(_im, cmap='gray'); _ax.set_title(_t, fontsize=9); _ax.axis('off')\n"
            "    for _ax in _axes.flat[_n_show:]:\n"
            "        _ax.axis('off')\n"
            "    _plt.tight_layout(); _plt.show()\n"
            "else:\n"
            "    print('real_imgs is None or insufficient (<4); will fall through to synthetic.')\n"
        ),
    },
    "13_validation_case_study.ipynb": {
        "anchor_text": "## Test image registry",
        "dataset_name": "BBBC020 — Murine bone-marrow derived macrophages",
        "license_note": "CC0",
        "citation": "Ljosa et al., Nature Methods, 2012 — BBBC020",
        "source_url": "https://bbbc.broadinstitute.org/BBBC020",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip",
        "what_it_is": "Pairs naturally with NB01 — same dataset, multiple validation models compared on it.",
        "swap_vars": ["img", "TEST_IMAGES"],
        "swap_code": (
            "if real_imgs:\n"
            "    img = real_imgs[0]\n"
            "    # Pre-create TEST_IMAGES with real entries. The synthetic creation cell\n"
            "    # below is gated to skip when real_imgs is loaded, so this dict survives\n"
            "    # into the downstream model picker.\n"
            "    # Each entry: name -> (image, ground_truth_or_None).\n"
            "    TEST_IMAGES = {\n"
            "        f'real_bbbc020_{i:02d}': (real_imgs[i], None)\n"
            "        for i in range(min(4, len(real_imgs)))\n"
            "    }\n"
            "    print(f'img + TEST_IMAGES now from BBBC020 real data ({len(TEST_IMAGES)} registry entries; ground-truth masks not loaded for BBBC020).')\n"
            "else:\n"
            "    print('real_imgs is None; staying with synthetic.')\n"
        ),
    },
    "14_spot_detection.ipynb": {
        "anchor_text": "## Show one example",
        "dataset_name": "BBBC020 — Murine bone-marrow derived macrophages (spot-like fluorescence intensities)",
        "license_note": "CC0",
        "citation": "Ljosa et al., Nature Methods, 2012 — BBBC020",
        "source_url": "https://bbbc.broadinstitute.org/BBBC020",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip",
        "what_it_is": "Real fluorescence imagery as a *teaching analogue* for spot detection. True FISH / single-molecule benchmarks live in deepBlink and BIA — see the audit.",
        "swap_vars": ["train_images", "train_centers", "test_images", "test_centers"],
        "swap_code": (
            "if real_imgs and len(real_imgs) >= 4:\n"
            "    import numpy as _np\n"
            "    def _to2d(im, target=128):\n"
            "        a = _np.asarray(im).astype(_np.float32)\n"
            "        if a.ndim == 3:\n"
            "            a = a.mean(axis=-1) if a.shape[-1] in (3,4) else a[a.shape[0]//2]\n"
            "        a = (a - a.min()) / (a.max() - a.min() + 1e-9)\n"
            "        s = min(a.shape)\n"
            "        a = a[:s, :s]\n"
            "        from scipy.ndimage import zoom as _zoom\n"
            "        return _zoom(a, target / s, order=1)\n"
            "    # Split: 75% train, 25% test (with 8 real images = 6 train + 2 test).\n"
            "    _split = max(2, int(len(real_imgs) * 0.75))\n"
            "    train_images = _np.stack([_to2d(im) for im in real_imgs[:_split]])\n"
            "    test_images = _np.stack([_to2d(im) for im in real_imgs[_split:]])\n"
            "    train_centers = [None] * len(train_images)\n"
            "    test_centers = [None] * len(test_images)\n"
            "    print(f'Bound train_images ({len(train_images)}) + test_images ({len(test_images)}) from BBBC020. centers = None (no ground-truth coords for this dataset).')\n"
            "    print('NOTE: BBBC020 is a teaching analogue for spot detection. For true FISH benchmarks with annotated coords, see deepBlink.')\n"
            "    # Display loaded\n"
            "    import matplotlib.pyplot as _plt\n"
            "    _n_show = min(8, len(train_images) + len(test_images))\n"
            "    _imgs = list(train_images[:6]) + list(test_images[:2])\n"
            "    _titles = [f'train {i}' for i in range(min(6, len(train_images)))] + [f'test {i}' for i in range(min(2, len(test_images)))]\n"
            "    _ncols = 4; _nrows = (_n_show + _ncols - 1) // _ncols\n"
            "    _fig, _axes = _plt.subplots(_nrows, _ncols, figsize=(3*_ncols, 3*_nrows))\n"
            "    for _ax, _im, _t in zip(_axes.flat, _imgs[:_n_show], _titles[:_n_show]):\n"
            "        _ax.imshow(_im, cmap='gray'); _ax.set_title(_t, fontsize=9); _ax.axis('off')\n"
            "    for _ax in _axes.flat[_n_show:]:\n"
            "        _ax.axis('off')\n"
            "    _plt.tight_layout(); _plt.show()\n"
            "else:\n"
            "    print('real_imgs is None or insufficient (<4); will fall through to synthetic.')\n"
        ),
    },
}


def t1_markdown(spec: dict) -> str:
    return f"""{MD_SENTINEL}
## Try it on canonical real data

This optional block fetches a canonical published microscopy dataset over the
network and prepares it for the rest of the notebook. **If the download fails**
(firewall, missing optional dependency, expired URL), the notebook continues
with the synthetic data below — this cell is safe to skip and re-runnable.

- **Dataset:** {spec['dataset_name']}
- **What it is:** {spec['what_it_is']}
- **Source:** [{spec['source_url']}]({spec['source_url']})
- **License:** {spec['license_note']}
- **Citation:** {spec['citation']}

After this cell runs, `real_imgs` is either a list of NumPy arrays from the real
dataset, or `None` if the download failed. The next cell binds working variables
from `real_imgs` if available; if it is `None`, the synthetic-generation cell
below runs as a fallback so the notebook still works end-to-end.
"""


def t1_code(spec: dict) -> str:
    return f"""{CO_SENTINEL}
# Real-data option from the dataset audit (2026-05-07).
# Tries to fetch a canonical published dataset; falls back to synthetic on any failure.

import os, sys, traceback, tempfile, urllib.request, zipfile

REAL_DATA_URL = {spec['zip_url']!r}
REAL_DATA_NAME = {spec['dataset_name']!r}
USE_REAL_DATA = True  # set False to skip the download entirely

real_imgs = None
real_filenames = None

if USE_REAL_DATA:
    try:
        cache_zip = os.path.join(tempfile.gettempdir(), os.path.basename(REAL_DATA_URL))
        cache_dir = cache_zip + "_extracted"

        if not os.path.exists(cache_zip):
            print(f"Downloading {{REAL_DATA_NAME}} (this can take 10-60 s)...")
            urllib.request.urlretrieve(REAL_DATA_URL, cache_zip)
            print(f"  → cached at {{cache_zip}} ({{os.path.getsize(cache_zip)/1e6:.1f}} MB)")

        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            with zipfile.ZipFile(cache_zip) as zf:
                zf.extractall(cache_dir)

        # Walk the extracted tree for image files. tifffile preferred; PIL fallback.
        try:
            import tifffile
            _read = lambda p: tifffile.imread(p)
        except Exception:
            from PIL import Image
            import numpy as np
            _read = lambda p: np.array(Image.open(p))

        exts = (".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG")
        all_paths = []
        for root, _, files in os.walk(cache_dir):
            for fn in files:
                if fn.endswith(exts):
                    all_paths.append(os.path.join(root, fn))
        all_paths.sort()

        # Take only the first few so the cell runs quickly. Attendees can index further.
        sample_paths = all_paths[:8]
        real_imgs = [_read(p) for p in sample_paths]
        real_filenames = [os.path.relpath(p, cache_dir) for p in sample_paths]

        print(f"Loaded {{len(real_imgs)}} real images from {{REAL_DATA_NAME}}.")
        print(f"  first image: shape={{real_imgs[0].shape}}, dtype={{real_imgs[0].dtype}}")

        # Visual confirmation — display the first image inline so you can see the load worked.
        try:
            import numpy as _np
            import matplotlib.pyplot as _plt
            _sample = real_imgs[0]
            # Pick a 2D plane to display: handle (H,W), (H,W,C), or (C,H,W) / (Z,H,W).
            if _sample.ndim == 2:
                _display = _sample
                _cmap = "gray"
            elif _sample.ndim == 3 and _sample.shape[-1] in (3, 4):
                _display = _sample
                _cmap = None
            elif _sample.ndim == 3:
                # Multi-channel or Z-stack — pick the largest plane.
                _axis = int(_np.argmin(_sample.shape))  # smallest = channel/Z axis
                _display = _np.take(_sample, _sample.shape[_axis] // 2, axis=_axis)
                _cmap = "gray"
            else:
                _display = _sample.reshape(_sample.shape[-2:]) if _sample.size else _sample
                _cmap = "gray"
            # Robust contrast for 16-bit / float images. Fall back to min/max
            # when the percentile stretch collapses (e.g. sparse binary masks
            # where >99% of pixels are background — vmin == vmax = 0 → all-black).
            _vmin, _vmax = _np.percentile(_display, [1, 99])
            if _vmax <= _vmin:
                _vmin, _vmax = float(_np.min(_display)), float(_np.max(_display))
                if _vmax <= _vmin:
                    _vmax = _vmin + 1.0
            _fig, _ax = _plt.subplots(figsize=(6, 6))
            _ax.imshow(_display, cmap=_cmap, vmin=_vmin, vmax=_vmax)
            _ax.set_title(
                f"{{REAL_DATA_NAME}}\\n"
                f"{{real_filenames[0]}} (shape={{_sample.shape}}, dtype={{_sample.dtype}})",
                fontsize=10,
            )
            _ax.axis("off")
            _plt.tight_layout()
            _plt.show()
        except Exception:
            print("Could not render preview (matplotlib issue?); data is still in real_imgs.")
            traceback.print_exc(limit=2)
    except Exception:
        print(f"Real-data download failed; continuing with synthetic.")
        traceback.print_exc(limit=2)
        real_imgs = None
        real_filenames = None
else:
    print("USE_REAL_DATA = False; skipping download. Synthetic flow runs below.")
"""


# ---------------------------------------------------------------------------
# Tier 2 specs — reference appendix
# ---------------------------------------------------------------------------

T2_SPECS = {
    "00_setup_self_check.ipynb": {
        "intro": (
            "Once your environment passes the self-check, you are ready to work with "
            "real microscopy data. The full audit is in `datasets_audit.md`; the most "
            "common starting points across the workshop are:"
        ),
        "items": [
            ("BBBC (Broad Bioimage Benchmark Collection)", "https://bbbc.broadinstitute.org/image_sets",
             "60+ image sets, mostly CC0; ground truth available; canonical for segmentation benchmarks."),
            ("BioImage Archive (EMBL-EBI)", "https://www.ebi.ac.uk/bioimage-archive/",
             "DOI-stable; per-study licenses (commonly CC variants); REMBI metadata."),
            ("Image Data Resource (IDR)", "https://idr.openmicroscopy.org/",
             "Public OMERO server; CC-BY 4.0 site-level; programmatic API."),
        ],
    },
    "04_community_platforms.ipynb": {
        "intro": "BiMZ live calls cover the model side; for paired *image* datasets:",
        "items": [
            ("BBBC", "https://bbbc.broadinstitute.org/image_sets", "CC0 segmentation benchmarks."),
            ("BioImage Archive", "https://www.ebi.ac.uk/bioimage-archive/", "DOI-stable; per-study CC."),
            ("IDR", "https://idr.openmicroscopy.org/", "CC-BY 4.0 site; OMERO API."),
            ("Allen Cell Imaging Collections", "https://www.allencell.org/data-downloading.html",
             "Allen Terms of Use — noncommercial; mandatory citation."),
        ],
    },
    "05_desktop_gui_complements.ipynb": {
        "intro": (
            "The Fiji / QuPath workflows in this notebook expect a real image to load. "
            "Recommended sources for desktop hands-on:"
        ),
        "items": [
            ("BBBC005 ground truth", "https://bbbc.broadinstitute.org/BBBC005",
             "12 MB ZIP — pairs of in-focus images + binary masks. Ideal for trying TWS / Pixel Classifier."),
            ("BBBC020", "https://bbbc.broadinstitute.org/BBBC020",
             "Multi-channel macrophages — good StarDist target."),
            ("Allen Cell Imaging Collections", "https://www.allencell.org/data-downloading.html",
             "OME-TIFF hiPSC volumes — strong fit for deepImageJ / 3D Fiji."),
            ("Cell Image Library (per-image license)", "https://www.cellimagelibrary.org/pages/datasets",
             "12,000+ datasets at UCSD CRBS; filter by Public Domain or CC-BY before redistributing."),
            ("IDR", "https://idr.openmicroscopy.org/", "OMERO sample images for QuPath WSI workflows."),
        ],
    },
    "08_srrf_esrrf.ipynb": {
        "intro": "Real blinking stacks for SRRF / eSRRF beyond the synthetic demo above:",
        "items": [
            ("Henriques lab NanoJ-SRRF demo data", "https://github.com/HenriquesLab/NanoJ-SRRF",
             "Reference TIRF stacks bundled with the SRRF distribution."),
            ("BioImage Archive — search 'SRRF' or 'eSRRF'", "https://www.ebi.ac.uk/bioimage-archive/",
             "Recent eSRRF studies (Laine 2023) with raw frames."),
            ("ZeroCostDL4Mic eSRRF notebook", "https://github.com/HenriquesLab/ZeroCostDL4Mic",
             "Includes a small blinking dataset for the Colab tutorial."),
        ],
    },
    "10_3d_segmentation.ipynb": {
        "intro": (
            "`cells3d()` is the always-works real volume in this NB. For larger 3D demos "
            "(too big for in-NB download) try:"
        ),
        "items": [
            ("BSCCM (Berkeley Single Cell Computational Microscopy)", "https://github.com/Waller-Lab/BSCCM",
             "BSD-3-Clause. Six size tiers, BSCCM-tiny is 0.6 GB. Python package handles download."),
            ("Allen Cell Imaging Collections", "https://www.allencell.org/data-downloading.html",
             "OME-TIFF hiPSC volumes. Allen TOU — noncommercial; mandatory citation."),
            ("BBBC024 (3D HL60 synthetic)", "https://bbbc.broadinstitute.org/BBBC024",
             "Synthetic 3D HL60 — CC0; useful for 3D method validation."),
            ("BBBC027 (3D Colon Tissue)", "https://bbbc.broadinstitute.org/BBBC027",
             "Real 3D colon tissue — CC0."),
        ],
    },
    "11_dl_tracking.ipynb": {
        "intro": "Canonical tracking benchmarks beyond the synthetic time-lapse:",
        "items": [
            ("Cell Tracking Challenge", "http://celltrackingchallenge.net/",
             "Standard benchmark for TRA/SEG metrics; per-dataset license, all permit research use."),
            ("BBBC019 (Collective cell migration)", "https://bbbc.broadinstitute.org/BBBC019",
             "Real time-lapse with annotated tracks — CC0."),
            ("IDR studies tagged 'time-lapse'", "https://idr.openmicroscopy.org/", "OMERO API for batch fetch."),
        ],
    },
    "15_diffusion_models.ipynb": {
        "intro": (
            "The HF model hub already provides demo weights. For applying / fine-tuning "
            "diffusion models on bioimage data:"
        ),
        "items": [
            ("BioImage Archive", "https://www.ebi.ac.uk/bioimage-archive/", "DOI-stable training corpora."),
            ("BBBC", "https://bbbc.broadinstitute.org/image_sets", "CC0 — clean for redistribution / fine-tuning runs."),
            ("Hugging Face microscopy datasets", "https://huggingface.co/datasets?other=microscopy",
             "Curated user-uploaded sets; check per-dataset license."),
        ],
    },
}


def t2_markdown(spec: dict) -> str:
    lines = [
        MD_SENTINEL,
        "---",
        "## Real microscopy datasets to explore next",
        "",
        spec["intro"],
        "",
    ]
    for name, url, note in spec["items"]:
        lines.append(f"- **{name}** — [{url}]({url}) — {note}")
    lines += [
        "",
        "Full audit and notebook ↔ dataset mapping in "
        "[`datasets_audit.md`](https://github.com/microscopy-Core-ISMMS/ImageAnalysisCourse/"
        "blob/2026-workshop/datasets_audit.md).",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def _md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) if text else [],
    }


def _code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True) if text else [],
    }


def _strip_prior_patch(cells: List[dict]) -> List[dict]:
    """Remove any cell carrying any patch sentinel — old 4-cell pattern OR new
    2-cell decision pattern. Lets a notebook with the old layout migrate cleanly."""
    sentinels = (
        MD_SENTINEL, CO_SENTINEL, SWAP_MD_SENTINEL, SWAP_CO_SENTINEL,
        DECISION_MD_SENTINEL, DECISION_CO_SENTINEL,
    )
    out = []
    for c in cells:
        src = "".join(c.get("source", []))
        if any(s in src for s in sentinels):
            continue
        out.append(c)
    return out


def t1_swap_markdown(spec: dict) -> str:
    vars_listed = ", ".join(f"`{v}`" for v in spec.get("swap_vars", []))
    return f"""{SWAP_MD_SENTINEL}
### Bind working variables — real if available, synthetic as fallback

**Real data is the default.** If the download above succeeded, this cell binds
the working variables ({vars_listed}) directly from `real_imgs`. The
synthetic-generation cell below is gated to skip in that case.

If the download failed (`real_imgs is None`), this cell prints a notice and
the synthetic-generation cell runs as a fallback so the notebook still works
end-to-end.

⚠️  The "What you should be seeing" callouts further down were written against
the synthetic data — when running on real data, your output will differ in
counts, shapes, and metric values. That's expected and is itself a useful
teaching moment.
"""


def t1_swap_code(spec: dict) -> str:
    body = _indent(spec["swap_code"], 4)
    # The bind block (`spec["swap_code"]`) is itself a `if real_imgs and ...:`
    # block per the per-NB specs. We just route execution into it; no flag.
    return f"""{SWAP_CO_SENTINEL}
# Bind working variables. Real is the default; if the download above failed
# (real_imgs is None), this is a no-op and the gated synthetic cell below
# generates the working variables instead.

if globals().get('real_imgs') is None:
    print('Real data not loaded; the synthetic-generation cell below will run as a fallback.')
else:
{body}"""


def _indent(block: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join((pad + line) if line else line for line in block.splitlines()) + ("\n" if block.endswith("\n") else "")


def _find_anchor_index(cells: List[dict], anchor_text: str) -> int:
    """Find the cell whose source starts with `anchor_text` (after stripping)."""
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if src.lstrip().startswith(anchor_text):
            return i
    return -1


# ---------------------------------------------------------------------------
# Patch operations
# ---------------------------------------------------------------------------

def t1_decision_markdown(spec: dict) -> str:
    """The intro markdown cell that explains the four-tier choice."""
    return f"""{DECISION_MD_SENTINEL}
## Choose your data source

This notebook can run on four kinds of data — pick one in the cell below.

- **MABC hosted** *(default)* — curated samples produced by the Mt Sinai Microscopy and Advanced Bioimaging Core, sized and formatted for this notebook. Fast, reproducible, license-clean.
- **Canonical** — fetch the published reference dataset ({spec['dataset_name']}) from its upstream source. Slower but pedagogically the same.
- **Synthetic** — generate the toy data the notebook was originally written against. Always works, even offline. The "what you should be seeing" callouts further down were written for this path.
- **My own data → see T0** — opens the [data sources reference notebook]({T0_URL}) with copy-pasteable blocks (local files, Google Drive, public URL, etc.).

If the chosen tier fails (network down, file missing), the loader falls through automatically: MABC → canonical → synthetic. Every cell prints which tier won.

- **Source:** [{spec['source_url']}]({spec['source_url']})
- **License:** {spec['license_note']}
- **Citation:** {spec['citation']}
"""


def t1_decision_code(spec: dict, nb_id: str) -> str:
    """The single decision-block code cell that does tier resolution + bind + display."""
    bind_body = _indent(spec["swap_code"], 4)
    return f"""{DECISION_CO_SENTINEL}
# @title Choose data source {{ run: "auto", display-mode: "form" }}
DATA_SOURCE = "MABC hosted"  # @param ["MABC hosted", "Canonical (BBBC etc.)", "Synthetic", "My own data → see T0 notebook"]

import os, sys, traceback, tempfile, urllib.request, urllib.error, zipfile
import numpy as _np

NB_ID = {nb_id!r}
MABC_URL = f"{MABC_URL_BASE}/{{NB_ID}}.npz"
CANONICAL_URL = {spec['zip_url']!r}
CANONICAL_NAME = {spec['dataset_name']!r}

real_imgs = None
real_filenames = None
real_metadata = None
real_labels = None
loaded_tier = None


def _try_mabc():
    \"\"\"Fetch the MABC sample npz from gh-pages. Returns (imgs, filenames, metadata, labels).
    `labels` is None unless the npz includes a `labels.npy` (paired masks, target channels, etc.).\"\"\"
    cache = os.path.join(tempfile.gettempdir(), os.path.basename(MABC_URL))
    if not os.path.exists(cache):
        print(f"Fetching MABC sample: {{MABC_URL}}")
        urllib.request.urlretrieve(MABC_URL, cache)
    data = _np.load(cache, allow_pickle=True)
    imgs = list(data['images'])
    fnames = list(data['filenames']) if 'filenames' in data.files else [f"mabc_{{i}}" for i in range(len(imgs))]
    try:
        meta = data['metadata'].item() if 'metadata' in data.files else {{}}
    except Exception:
        meta = {{}}
    labels = list(data['labels']) if 'labels' in data.files else None
    return imgs, fnames, meta, labels


def _try_canonical():
    \"\"\"Existing zip-based fetch from BBBC / GigaDB. Same logic as the prior architecture.\"\"\"
    cache_zip = os.path.join(tempfile.gettempdir(), os.path.basename(CANONICAL_URL))
    cache_dir = cache_zip + "_extracted"
    if not os.path.exists(cache_zip):
        print(f"Fetching canonical: {{CANONICAL_NAME}} (this can take 10-60 s)...")
        urllib.request.urlretrieve(CANONICAL_URL, cache_zip)
        print(f"  cached at {{cache_zip}} ({{os.path.getsize(cache_zip)/1e6:.1f}} MB)")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        with zipfile.ZipFile(cache_zip) as zf:
            zf.extractall(cache_dir)
    try:
        import tifffile
        _read = lambda p: tifffile.imread(p)
    except ImportError:
        from PIL import Image
        _read = lambda p: _np.array(Image.open(p))
    exts = ('.tif', '.tiff', '.TIF', '.TIFF', '.png', '.PNG')
    paths = []
    for root, _, files in os.walk(cache_dir):
        for fn in files:
            if fn.endswith(exts):
                paths.append(os.path.join(root, fn))
    paths.sort()
    paths = paths[:8]
    imgs = [_read(p) for p in paths]
    fnames = [os.path.relpath(p, cache_dir) for p in paths]
    meta = {{"source": CANONICAL_NAME, "url": CANONICAL_URL, "tier": "canonical"}}
    return imgs, fnames, meta


# Tier resolution
if DATA_SOURCE == "My own data → see T0 notebook":
    print("Open the T0 notebook for copy-paste data-loading blocks:")
    print(f"  {T0_URL}")
    print("Once your images are loaded into a list called `real_imgs`, re-run the rest of this notebook.")

elif DATA_SOURCE == "Synthetic":
    print("Synthetic-only mode: skipping all real-data tiers; the synthetic generation cell below will run.")

else:
    if DATA_SOURCE == "MABC hosted":
        try:
            real_imgs, real_filenames, real_metadata, real_labels = _try_mabc()
            loaded_tier = "MABC"
        except (urllib.error.HTTPError, urllib.error.URLError, FileNotFoundError):
            print("MABC sample not yet available; falling through to canonical.")
        except Exception:
            print("MABC fetch raised an unexpected error; falling through to canonical.")
            traceback.print_exc(limit=2)

    if real_imgs is None:
        try:
            real_imgs, real_filenames, real_metadata = _try_canonical()
            real_labels = None
            loaded_tier = "Canonical"
        except Exception:
            print("Canonical fetch failed; the synthetic-generation cell below will run as the final fallback.")
            traceback.print_exc(limit=2)


# Bind working variables and display the loaded grid (only if a real tier won).
if real_imgs is not None:
    print(f"\\nLoaded {{len(real_imgs)}} images from tier: {{loaded_tier}}.")
    if real_metadata:
        print(f"  source: {{real_metadata.get('source', '(unknown)')}}")
        print(f"  license: {{real_metadata.get('license', 'see source')}}")
        if 'citation' in real_metadata:
            print(f"  cite: {{real_metadata['citation']}}")

    # ---- Per-NB binding (lifted from the prior architecture's swap_code) ----
{bind_body}

    # ---- Universal display grid ----
    try:
        import matplotlib.pyplot as _plt
        _n_show = min(8, len(real_imgs))
        _ncols = 4
        _nrows = (_n_show + _ncols - 1) // _ncols
        _fig, _axes = _plt.subplots(_nrows, _ncols, figsize=(3 * _ncols, 3 * _nrows))
        _ax_iter = list(_axes.flat) if hasattr(_axes, 'flat') else [_axes]
        for _i, _ax in enumerate(_ax_iter[:_n_show]):
            _disp = _np.asarray(real_imgs[_i]).astype(float)
            if _disp.ndim == 3:
                if _disp.shape[-1] in (3, 4):
                    pass  # RGB(A)
                else:
                    _disp = _disp.mean(axis=-1) if _disp.shape[-1] < min(_disp.shape[:2]) else _disp[_disp.shape[0]//2]
            _vmin, _vmax = _np.percentile(_disp, [1, 99])
            if _vmax <= _vmin:
                _vmin, _vmax = float(_disp.min()), float(_disp.max())
                if _vmax <= _vmin:
                    _vmax = _vmin + 1.0
            _cmap = None if (_disp.ndim == 3 and _disp.shape[-1] in (3, 4)) else 'gray'
            _ax.imshow(_disp, cmap=_cmap, vmin=_vmin, vmax=_vmax)
            _fn = (real_filenames[_i] if real_filenames and _i < len(real_filenames) else f'img {{_i}}')
            _ax.set_title(f"{{loaded_tier}}: {{str(_fn)[:32]}}", fontsize=8)
            _ax.axis('off')
        for _ax in _ax_iter[_n_show:]:
            _ax.axis('off')
        _plt.tight_layout(); _plt.show()
    except Exception:
        print("Could not render preview grid; data is still in real_imgs.")
        traceback.print_exc(limit=2)
else:
    print("Real-data tiers did not produce data. The synthetic-generation cell below will run.")
"""


def patch_t1(nb: dict, fn: str, spec: dict) -> Tuple[bool, str]:
    cells = _strip_prior_patch(nb["cells"])
    idx = _find_anchor_index(cells, spec["anchor_text"])
    if idx < 0:
        return False, f"  T1 anchor not found in {fn}: {spec['anchor_text']!r}"
    nb_id = fn.replace(".ipynb", "")
    new_md = _md_cell(t1_decision_markdown(spec))
    new_co = _code_cell(t1_decision_code(spec, nb_id))
    cells[idx:idx] = [new_md, new_co]
    nb["cells"] = cells
    return True, f"  T1 inserted at cell {idx} in {fn} (2 cells: decision-block intro + tiered loader)"


def patch_t2(nb: dict, fn: str, spec: dict) -> Tuple[bool, str]:
    cells = _strip_prior_patch(nb["cells"])
    cells.append(_md_cell(t2_markdown(spec)))
    nb["cells"] = cells
    return True, f"  T2 appended in {fn}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate everything but do not write notebooks.")
    args = parser.parse_args()

    nb_files = sorted(NB_DIR.glob("*.ipynb"))
    if not nb_files:
        print(f"No notebooks found at {NB_DIR}", file=sys.stderr)
        return 2

    n_t1 = n_t2 = n_skip = n_err = 0

    for nb_path in nb_files:
        fn = nb_path.name
        try:
            with open(nb_path) as f:
                nb = json.load(f)
        except Exception as e:
            print(f"[ERR ] {fn}: failed to parse JSON: {e}")
            n_err += 1
            continue

        if fn in T1_SPECS:
            ok, msg = patch_t1(nb, fn, T1_SPECS[fn])
            tier = "T1"
        elif fn in T2_SPECS:
            ok, msg = patch_t2(nb, fn, T2_SPECS[fn])
            tier = "T2"
        else:
            print(f"[SKIP] {fn}: no spec")
            n_skip += 1
            continue

        if not ok:
            print(f"[ERR ] {fn}: {msg}")
            n_err += 1
            continue

        if args.dry_run:
            print(f"[DRY ] {fn}: would apply {tier}")
        else:
            with open(nb_path, "w") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
                f.write("\n")
            print(f"[ {tier} ] {msg}")

        if tier == "T1":
            n_t1 += 1
        else:
            n_t2 += 1

    print()
    print(f"Done. T1={n_t1}, T2={n_t2}, skipped={n_skip}, errors={n_err}.")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
