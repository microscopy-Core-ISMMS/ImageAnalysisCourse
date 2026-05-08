# Real-data patch plan (per dataset audit, 2026-05-07)

**Pattern:** every notebook receives one of two patches.

- **Tier 1 (executable):** insert one markdown + one code cell **before** the existing synthetic-data section. Code does `try: download_real_data()` → cache to `/tmp` → load via `tifffile`/`imread` → assign to `real_*` variable. On any exception (timeout, 4xx, missing optional dep) it logs the failure and falls through to the existing synthetic path. Downstream cells continue to reference the existing variable names so the rest of the notebook keeps working unchanged.
- **Tier 2 (reference-only):** append a markdown cell at the bottom of the notebook listing the canonical real datasets that fit the topic, with download URLs and license notes. No executable code — these notebooks already have stable demo data (cells3d(), BiMZ live calls, HF model hub, screenshot walkthroughs, etc.) and don't need a runtime fetch.

Both tiers carry attribution. Tier 1's anchor cell is chosen to land *before* the synthetic generation so the real-vs-synthetic comparison is teachable.

## Per-notebook mapping

| # | NB | Tier | Real dataset | URL | Anchor (insert before) | License |
|---|---|---|---|---|---|---|
| 00 | setup self-check | T2 | n/a — reference appendix | — | end of NB | — |
| 01 | Cellpose segmentation | **T1** | BBBC020 (mouse macrophages, 2-ch + masks) + BBBC006 try-this | data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip | cell 5 (synthetic data section) | CC0 |
| 02 | Validation/QC | **T1** | BBBC005 ground truth (1200 in-focus images + binary masks, 12 MB) | data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_ground_truth.zip | cell 6 (`Generate ground truth`) | CC0 |
| 03a | Denoising N2V | T2 | GigaDB 100888 + BIA reference | gigadb.org/dataset/100888 | end of NB | CC0 |
| 03b | Foundation seg (SAM/μSAM) | **T1** | BBBC020 (non-canonical microscopy fits the SAM-on-microscopy theme) | data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip | cell 5 (`Load a non-canonical microscopy image`) | CC0 |
| 04 | Community catalog | T2 | BiMZ live + audit list | — | end of NB | — |
| 05 | Desktop GUI complements | T2 | BBBC + BIA + IDR + Allen + CIL pointers (these are the natural Fiji/QuPath datasets) | — | end of NB | mixed |
| 06 | Virtual staining | T2 | Allen AICS + CSBDeep CARE pointers | downloads.allencell.org + csbdeep.bioimagecomputing.com | end of NB | Allen TOU + BSD |
| 07 | AI super-resolution | T2 | CSBDeep CARE samples + ZeroCostDL4Mic | csbdeep.bioimagecomputing.com | end of NB | BSD |
| 08 | SRRF / eSRRF | T2 | Henriques lab demo data + BIA | github.com/HenriquesLab/NanoJ-SRRF | end of NB | mixed |
| 09 | Cellpose fine-tuning | **T1** | BBBC038 (Kaggle DSB 2018 nuclei — designed for segmentation training) | data.broadinstitute.org/bbbc/BBBC038v1/BBBC038v1_train.zip | cell 7 (`Get a small labeled dataset`) | CC0 (BBBC mirror) |
| 10 | 3D segmentation | T2 | cells3d() already real; BSCCM + AICS pointers (datasets too large for runtime download) | — | end of NB | BSD-3 / Allen TOU |
| 11 | DL tracking | T2 | Cell Tracking Challenge | celltrackingchallenge.net | end of NB | per-CTC dataset |
| 12 | Deconvolution | T2 | CSBDeep deconv samples + BIA | csbdeep.bioimagecomputing.com | end of NB | BSD |
| 13 | Validation case study | **T1** | BBBC020 (matches NB01 chain) | data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip | cell 7 (`Test image registry`) | CC0 |
| 14 | Spot detection | T2 | deepBlink samples + BIA FISH studies | github.com/BBQuercus/deepBlink | end of NB | per-source |
| 15 | Diffusion models | T2 | HF model hub already there + BIA | huggingface.co/google/ddpm-cifar10-32 | end of NB | various |

**Tier 1 count: 10 NBs (01, 02, 03a, 03b, 06, 07, 09, 12, 13, 14).** Each gets four cells: download-md → download-code (with inline imshow viz) → swap-md → swap-code (NB-specific variable rebind).

**Tier 2 count: 7 NBs (00, 04, 05, 08, 10, 11, 15).** These either don't have a working image variable (00 setup, 05 walkthrough, 15 frontier), already use real data (04 BiMZ live, 10 cells3d), or need a deliberate dataset selection that hasn't been done yet (08 SRRF blinking stacks, 11 tracking time series).

**Promoted in second round (2026-05-07):** NB03a, NB06, NB07, NB12, NB14 — using verified BBBC005 / BBBC020 URLs. NB06/07/14 carry an explicit "pedagogical analogue" caveat in the markdown because the real dataset is not a perfect topical match (e.g. BBBC005 in/out-of-focus pairs are used as the SR low-res / high-res analogue in NB07).

## Tier 1 cell template

Markdown cell (above):
```
## Try it on canonical real data
This optional block fetches a canonical published dataset over the network and
runs the same pipeline on it. If the download fails (firewall, missing optional
dependency, expired URL), the notebook continues with the synthetic data below
— so this cell is safe to skip.

**Dataset:** {dataset_name} (license: {license})
**Source:** {url}
**Citation:** {citation}
```

Code cell (below):
```python
# Real-data option — try BBBCxxx, fall back to synthetic if anything fails.
import os, urllib.request, zipfile, tempfile, traceback

REAL_DATA_URL = "{URL}"
REAL_DATA_NAME = "{name}"
USE_REAL_DATA = True  # set False to skip the download entirely

real_imgs = None  # always defined; downstream cells can check `if real_imgs:`
if USE_REAL_DATA:
    try:
        cache = os.path.join(tempfile.gettempdir(), "{cache_filename}")
        if not os.path.exists(cache):
            print(f"Downloading {REAL_DATA_NAME}...")
            urllib.request.urlretrieve(REAL_DATA_URL, cache)
        # ...load images into real_imgs list...
        print(f"Loaded {len(real_imgs)} images from {REAL_DATA_NAME}")
    except Exception:
        print(f"Real-data download failed; continuing with synthetic.")
        traceback.print_exc()
        real_imgs = None
```

Downstream synthetic cells stay untouched. Attendees who completed the synthetic flow can repeat with `real_imgs[i]` to see the real-data behavior.

## Tier 2 cell template

Single markdown cell at the end of the notebook:
```
---
## Real microscopy datasets to explore next

Beyond the demo data above, here are canonical published datasets that fit
this notebook's topic. License notes from the workshop dataset audit:

- **{dataset 1}** — {URL} — {license note}
- **{dataset 2}** — {URL} — {license note}
- ...

Full audit and notebook ↔ dataset mapping in `Claude_Workspace/datasets_audit.md`.
```

## Risk register

- BBBC mirror URLs are stable (audited verbatim 2026-05-07) but the `data.broadinstitute.org` endpoint is occasionally slow. Tier 1 cells have a 60-second timeout and clean fallback.
- BBBC038 mirror at `data.broadinstitute.org/bbbc/BBBC038v1/` — confirm exact zip name before commit. **Flagged for confirmation in code (treat as risk, fall back gracefully if not found).**
- BBBC020 zip is ~25 MB → ~5–10 s on a fast link, ~30–60 s on conference Wi-Fi. Acceptable.
- BBBC005 ground truth zip is 12 MB → fast.
- All Tier 1 NBs preserve the original synthetic flow; turning off USE_REAL_DATA returns the NB to its pre-patch behavior.
