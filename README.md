# AI for Microscopy Image Analysis

A hands-on workshop on AI for microscopy image analysis, developed by the Microscopy and Advanced Bioimaging Core (MABC) at the Icahn School of Medicine at Mount Sinai.

This repository hosts the workshop's Jupyter Book companion: a lecture, lab notebooks, desktop GUI walkthroughs, and a curated resources catalog. The material is designed for biologists with technical fluency — comfortable in Python notebooks and Colab, no machine-learning background required — who want to add modern AI methods to their image-analysis toolkit responsibly.

The companion is delivered alongside the live afternoon block of Day 3 of the 2026 MABC Image Analysis Workshop and outlives the workshop day for self-paced study.

## What's inside

- **`lectures/`** — slide-based lectures.
  - `01_intro_ai_imaging/` — *AI for Scientific Image Analysis: What Works, What Doesn't, and Why* (90 min). Source notebook + reveal.js HTML export, speaker cues, verification log.
  - `02_walking_the_day/` — *Walking the Day*, a bridge talk between morning concepts and afternoon labs (45 min).
- **`notebooks/`** — Colab-ready labs.
  - `00_setup_self_check.ipynb` — environment + Python/numpy sanity check (run before workshop day).
  - `01_cellpose_segmentation.ipynb` — Lab 1: pretrained Cellpose-SAM segmentation, customization workshop, failure-mode finale.
  - `02_validation_quantification.ipynb` — Lab 2: IoU, Dice, instance metrics, and the metrics-versus-biology gap.
  - `03a_denoising_n2v.ipynb` — Lab 3 option A: AI denoising and the hallucination question.
  - `03b_foundation_model_segmentation.ipynb` — Lab 3 option B: foundation-model segmentation with prompts.
  - `04_community_platforms.ipynb` — catalog of ZeroCostDL4Mic / DL4MicEverywhere methods plus a live BioImage Model Zoo browser and five inline mini-workflows.
  - `05_desktop_gui_complements.ipynb` — desktop-GUI walkthroughs (Trainable Weka, QuPath Pixel Classifier, deepImageJ, StarDist Fiji, StarDist QuPath).
- **`ramp-up/notebooks/`** — optional pre-workshop refresh on Python and image data fundamentals.
- **`resources.md`** — curated catalog of community projects, tools, and datasets.
- **`handout.md`** — post-workshop take-home reference with every link cited.
- **`schedule.md`** — full timing for the workshop day.
- **`prerequisites.md`** — what to know and have set up before arriving.
- **`acknowledgments.md`** — institutional credit and per-tool attribution.
- **`env/`** — conda environment for local use after the workshop, plus a lecture-environment self-check.
- **`data/`** — dataset references (most labs use `skimage.data` built-ins or generate synthetic data inline; no large bundled assets).

## Two ways to use the labs

**Option A — Google Colab (recommended for the workshop day).**
Each notebook has a Colab-launch badge at the top. Click it; the notebook opens in Colab with the runtime ready. The setup cells pip-install dependencies on first run.

**Option B — Local Jupyter (after the workshop).**
Create the conda environment from `env/environment.yml`:

```bash
conda env create -f env/environment.yml
conda activate ai-microscopy-workshop
jupyter lab
```

See `env/README.md` for the three-tier environment guide (view static slides only / rebuild the lecture / present live with executable cells).

## Building the Jupyter Book

```bash
# From the repo root
pip install jupyter-book
jupyter-book build .
# Output lands in _build/html/
```

## License

- **Code** — MIT License (`LICENSE`)
- **Content** (notebook markdown, slides, handout, resources, prose) — Creative Commons Attribution 4.0 International (`CONTENT_LICENSE.md`)
- Mount Sinai brand assets used per institutional brand-usage guidelines, not relicensed.
- Cited tools (Cellpose, StarDist, SAM, μSAM, Noise2Void, CARE, ZeroCostDL4Mic, etc.) retain their own licenses.

## Citation

If you use these materials in your own teaching or research, please cite:

> Microscopy and Advanced Bioimaging Core (MABC), Icahn School of Medicine at Mount Sinai. *AI for Microscopy Image Analysis* workshop materials. 2026. Licensed CC-BY-4.0.

## Status and contributions

This repository is staged locally during the workshop build. It will be pushed to the MABC GitHub organization once the core notebooks have passed end-to-end Colab tests.

Issues, corrections, and contributions: please open an issue on the public repository once it is published.

## Acknowledgments

This workshop draws on community resources from the bioimage-analysis ecosystem. Specific tools and methods are cited inline in each notebook and lecture; full attributions are in `acknowledgments.md` and `resources.md`.
