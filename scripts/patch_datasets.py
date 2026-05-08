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
MD_SENTINEL = "<!-- DATASET-AUDIT-PATCH -->"
CO_SENTINEL = "# DATASET-AUDIT-PATCH"

# ---------------------------------------------------------------------------
# Tier 1 specs — executable real-data cells
# ---------------------------------------------------------------------------

T1_SPECS = {
    "01_cellpose_segmentation.ipynb": {
        "anchor_text": "## Generate the working dataset",  # markdown header text
        "dataset_name": "BBBC020 — Murine bone-marrow derived macrophages",
        "license_note": "CC0",
        "citation": "Ljosa et al., Nature Methods, 2012 — BBBC020",
        "source_url": "https://bbbc.broadinstitute.org/BBBC020",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip",
        "what_it_is": "20 fields of murine bone-marrow derived macrophages (DAPI + CD11b + F-actin); paired ground-truth outlines available.",
        "loader_kind": "bbbc_zip_tif",
    },
    "02_validation_quantification.ipynb": {
        "anchor_text": "## Generate ground truth",
        "dataset_name": "BBBC005 v1 ground truth — synthetic cells with paired binary masks",
        "license_note": "CC0 (Anne Carpenter waiver)",
        "citation": "Lehmussola et al., IEEE T. Med. Imaging, 2007; Bray et al., J. Biomol. Screen, 2011",
        "source_url": "https://bbbc.broadinstitute.org/BBBC005",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip",
        "what_it_is": "1,200 in-focus synthetic cell images with paired binary foreground/background masks. Ideal for IoU/Dice validation.",
        "loader_kind": "bbbc_zip_tif",
    },
    "03b_foundation_model_segmentation.ipynb": {
        "anchor_text": "## Load a non-canonical microscopy image",
        "dataset_name": "BBBC020 — Murine bone-marrow derived macrophages",
        "license_note": "CC0",
        "citation": "Ljosa et al., Nature Methods, 2012 — BBBC020",
        "source_url": "https://bbbc.broadinstitute.org/BBBC020",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip",
        "what_it_is": "Multi-channel macrophage fluorescence — exactly the 'non-canonical microscopy' content SAM/μSAM are tested against.",
        "loader_kind": "bbbc_zip_tif",
    },
    "09_cellpose_finetune.ipynb": {
        "anchor_text": "## Step 1: Get a small labeled dataset",
        "dataset_name": "BBBC038v1 — 2018 Data Science Bowl nuclei (training subset)",
        "license_note": "CC0 (BBBC mirror of the Kaggle 2018 Data Science Bowl)",
        "citation": "Caicedo et al., Nature Methods, 2019 — BBBC038",
        "source_url": "https://bbbc.broadinstitute.org/BBBC038",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC038v1/BBBC038v1_train.zip",
        "what_it_is": "Diverse nuclei across modalities — designed for segmentation model training. Training subset only (smaller download).",
        "loader_kind": "bbbc_zip_tif",
    },
    "13_validation_case_study.ipynb": {
        "anchor_text": "## Test image registry",
        "dataset_name": "BBBC020 — Murine bone-marrow derived macrophages",
        "license_note": "CC0",
        "citation": "Ljosa et al., Nature Methods, 2012 — BBBC020",
        "source_url": "https://bbbc.broadinstitute.org/BBBC020",
        "zip_url": "https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip",
        "what_it_is": "Pairs naturally with NB01 — same dataset, multiple validation models compared on it.",
        "loader_kind": "bbbc_zip_tif",
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
dataset, or `None` if the download was skipped or failed. Downstream synthetic
cells run regardless.
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
        print(f"Try: real_imgs[0].shape, real_imgs[0].dtype")
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
    "03a_denoising_n2v.ipynb": {
        "intro": "Real-world denoising benchmarks for the Noise2Void / CARE family:",
        "items": [
            ("GigaDB 100888 (Hagen et al. 2021)", "https://gigadb.org/dataset/100888",
             "Built specifically for training denoising DNNs. CC0 with citation request."),
            ("CSBDeep CARE example data", "http://csbdeep.bioimagecomputing.com/",
             "Paired clean/noisy fluorescence — the canonical CARE training set."),
            ("BioImage Archive (search 'denoising')", "https://www.ebi.ac.uk/bioimage-archive/",
             "Many published denoising studies with full raw + restored stacks."),
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
    "06_virtual_staining.ipynb": {
        "intro": "Canonical paired training data for virtual-staining / label-free prediction:",
        "items": [
            ("Allen Cell Imaging Collections (AICS-25 etc.)", "https://www.allencell.org/data-downloading.html",
             "16-bit OME-TIFF fields + 8-bit segmentation masks. Allen TOU: noncommercial; cite per Allen Citation Policy."),
            ("CSBDeep example data (fnet, CARE)", "http://csbdeep.bioimagecomputing.com/",
             "Paired transmitted-light → fluorescence; the original fnet pairs."),
            ("BioImage Model Zoo bioimage-applications track", "https://bioimage.io/",
             "Models with linked sample image triplets."),
        ],
    },
    "07_widefield_superres.ipynb": {
        "intro": "Paired low-res / high-res microscopy for SR training:",
        "items": [
            ("CSBDeep CARE example data", "http://csbdeep.bioimagecomputing.com/",
             "Paired widefield → confocal / SIM examples."),
            ("ZeroCostDL4Mic (DFCAN, DFGAN, RCAN tutorials)", "https://github.com/HenriquesLab/ZeroCostDL4Mic",
             "Per-method demo data with Colab notebooks."),
            ("BioImage Archive — search 'super-resolution'", "https://www.ebi.ac.uk/bioimage-archive/",
             "DOI-stable raw frames with paired ground truth."),
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
    "12_deconvolution.ipynb": {
        "intro": "Paired blurred/deconvolved data for benchmarking classical RL vs DL deconv:",
        "items": [
            ("CSBDeep CARE deconvolution demo", "http://csbdeep.bioimagecomputing.com/",
             "Paired noisy/clean confocal — the original Care-deconv example."),
            ("BioImage Archive — search 'deconvolution'", "https://www.ebi.ac.uk/bioimage-archive/",
             "Several published studies with raw + processed stacks."),
            ("BBBC005 (focus blur)", "https://bbbc.broadinstitute.org/BBBC005",
             "Synthetic in-focus / out-of-focus pairs — useful as a focus-restoration analogue."),
        ],
    },
    "14_spot_detection.ipynb": {
        "intro": "Real spot-detection / single-molecule data:",
        "items": [
            ("deepBlink example data", "https://github.com/BBQuercus/deepBlink",
             "Heatmap-based detector with bundled FISH-like demo set."),
            ("BioImage Archive — search 'FISH' or 'smFISH'", "https://www.ebi.ac.uk/bioimage-archive/",
             "Many published FISH studies with annotated spots."),
            ("IDR FISH studies", "https://idr.openmicroscopy.org/", "OMERO API; per-study annotations."),
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
    """Remove any cell that carries the patch sentinel."""
    out = []
    for c in cells:
        src = "".join(c.get("source", []))
        if MD_SENTINEL in src or CO_SENTINEL in src:
            continue
        out.append(c)
    return out


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

def patch_t1(nb: dict, fn: str, spec: dict) -> Tuple[bool, str]:
    cells = _strip_prior_patch(nb["cells"])
    idx = _find_anchor_index(cells, spec["anchor_text"])
    if idx < 0:
        return False, f"  T1 anchor not found in {fn}: {spec['anchor_text']!r}"
    new_md = _md_cell(t1_markdown(spec))
    new_co = _code_cell(t1_code(spec))
    cells[idx:idx] = [new_md, new_co]
    nb["cells"] = cells
    return True, f"  T1 inserted at cell {idx} in {fn}"


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
