---
layout: default
title: Environment
nav_order: 6
---

# Environment Setup

The workshop runs entirely on Google Colab during the workshop day, so most attendees do not need to install anything locally. After the workshop, attendees who want to continue with the materials on their own machines can use the conda environment in this folder.

The lecture has its own three-tier prerequisite ladder, separate from the lab environment — see [the next section](#lecture-environment-three-tiers) below if you are *delivering* Lecture 1 rather than running labs.

## Lecture environment — three tiers

The lecture deck has been authored as a self-contained reveal.js export plus an editable Jupyter notebook. There are three ways to use it, each with progressively more dependencies.

### Tier 0 — view the pre-built deck

**No install needed.** The static HTML deck at `lectures/01_intro_ai_imaging/lecture_slides.slides.html` is fully self-contained: reveal.js bundled, all 14 figures base64-embedded, citations baked in. Just open it in any modern browser.

```bash
# Optional: serve over HTTP so the speaker-view popup works without same-origin issues
cd lectures/01_intro_ai_imaging
python3 -m http.server 8001
# open http://localhost:8001/lecture_slides.slides.html
```

Press `S` for speaker view, `?` for keyboard shortcuts.

### Tier 1 — rebuild the deck from source

Lets you re-run `python build_lecture_notebook.py` after editing the slide content. Needs:

- `python>=3.10`
- `jupyter` (provides `nbconvert`)
- `nbconvert>=7,<8`
- `numpy`, `scipy`, `matplotlib` (for `build_figures.py`)

All present in the workshop conda env (`environment.yml` in this folder).

### Tier 2 — present LIVE with executable code cells (RISE)

Lets you open `lecture.ipynb` in JupyterLab, click the RISE icon, and present the lecture as a slideshow where the code cells stay live — you can re-run them mid-presentation in front of the audience. Needs Tier 1 plus:

- `jupyterlab_rise>=0.40,<1.0` (for JupyterLab 4.x or Notebook 7) — what most current users need
- *or* `rise` (for legacy classic Jupyter Notebook 6.x)

Both are on conda-forge.

### Verify your tier — Tier 0 prerequisites check

Two parallel ways to verify, same checks under each:

```bash
# Command-line script
conda activate ai-microscopy-workshop
python env/check_lecture_env.py
```

Or open the parallel notebook in Jupyter:

```
lectures/01_intro_ai_imaging/00_lecture_prereqs_check.ipynb
```

Either path imports each dependency, prints status (`✓` / `○` for optional / `✗` for required-missing), explains *why* the lecture needs it, and prints the exact `conda install` or `pip install` command to fix any failure. The command-line script returns exit code 0 if all *required* checks pass.

## Workshop day: Google Colab

Each notebook has a setup cell at the top that detects whether it is running on Colab and pip-installs the dependencies it needs. You do not need to set anything up before the workshop beyond:

1. A working Google account with Colab access
2. Confirmation that `notebooks/00_setup_self_check.ipynb` runs end-to-end (per the [prerequisites]({{ site.baseurl }}/prerequisites))

Colab free tier is sufficient for the workshop, with one caveat: enabling a GPU runtime ("Runtime → Change runtime type → GPU") makes Cellpose, Noise2Void, and SAM noticeably faster. The setup cell prints a warning if no GPU is enabled.

## After the workshop: local conda environment

The file `env/environment.yml` (in this folder) defines a conda environment that mirrors the dependencies used in the Colab notebooks. To set it up locally:

```bash
# Create the environment
conda env create -f env/environment.yml

# Activate it
conda activate ai-microscopy-workshop

# Install Jupyter and start it
jupyter lab
```

The environment uses Python 3.11 and pins all major dependencies. Specific versions in `environment.yml` should be reviewed against current state of the toolchain before any external use, as the bioimage AI ecosystem moves quickly.

## What the environment includes

At a high level, the environment provides:

- **Core scientific Python**: numpy, scipy, scikit-image, pandas, matplotlib, seaborn
- **Notebook runtime**: jupyterlab, ipython
- **Image I/O**: tifffile, imageio, openslide-python (for future pathology work)
- **AI tools used in labs**: cellpose, n2v (Noise2Void), segment-anything, optional micro-sam
- **PyTorch**: with CUDA support if your machine has a compatible GPU
- **Validation utilities**: scikit-learn, statsmodels

The exact list lives in `environment.yml`. We deliberately keep the environment compact — adding more tools is easy; removing them once attendees have built habits around them is hard.

## GPU considerations for local use

The labs are GPU-friendly but not strictly GPU-required. Specifically:

- *Lab 1 (Cellpose):* runs on CPU; GPU helps on larger images
- *Lab 2 (Validation):* CPU only; no GPU benefit
- *Lab 3 option A (Noise2Void):* GPU strongly recommended for training; CPU is workable for inference on small images
- *Lab 3 option B (SAM):* GPU strongly recommended; CPU inference is slow but possible on small images

If you are running locally without a GPU, expect the training step in Lab 3 option A to take noticeably longer than it does on Colab's free-tier T4.

## Updating and maintaining the environment

The environment file is pinned to specific minor versions to keep the workshop reproducible. When the underlying tools release new major versions:

1. Test the workshop notebooks against the new version on Colab
2. Update `environment.yml` to match
3. Update the workshop's tested-on date in this README
4. Re-run all notebooks end-to-end as a verification

For long-term maintenance, the workshop materials should be re-tested on Colab annually at minimum, and after any major version bump of Cellpose, segment-anything, or PyTorch.

## Reporting issues

Issues with the environment or with running the notebooks should be reported on the repository's GitHub issues page. Please include the environment manager (conda or pip), the Python version, and the operating system in the issue.
