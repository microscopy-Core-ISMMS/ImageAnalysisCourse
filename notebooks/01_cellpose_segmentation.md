---
layout: default
title: "01 — Cellpose segmentation"
parent: Notebooks
nav_order: 2
---

# Notebook 01: Pretrained Segmentation with Cellpose-SAM

**Status:** Outline only. Working notebook to be authored in a later build step.
**Lab time:** 90 minutes (workshop slot 13:00–14:30).
**Prerequisites:** Notebook 00 completed.
**Curriculum mapping:** T3M.3 (segmentation and quantification in microscopy), T1.2 (ML fundamentals applied), T2A.1 / T2B.2 (pretrained model use).
**Tool version:** Cellpose-SAM (Cellpose v4, 2025). The Cellpose tool has gone through four major versions; the current Cellpose-SAM adapts the SAM transformer backbone to the Cellpose framework. Use the latest stable release.

## Purpose

Apply pretrained Cellpose-SAM (the current v4 release of the Cellpose tool) to a canonical microscopy dataset, observe its output, identify cases where the model succeeds and where it fails, and quantify simple morphological features from the segmentation. The lab is the workshop's first hands-on encounter with a real AI tool. It is structured to make the failure modes as visible as the successes. The Cellpose-SAM choice is deliberate: it represents the current state of the tool ecosystem (combining the original Cellpose framework with the SAM transformer backbone) and ensures attendees learn the version they will use in practice.

## Learning objectives

By the end of this notebook, attendees should be able to:

1. Run pretrained Cellpose on a microscopy image and obtain instance-level segmentation masks
2. Visualize segmentation overlays on the original image
3. Identify at least three distinct types of segmentation error in the outputs and articulate why each likely occurred
4. Compute simple per-object features (count, area, equivalent diameter) and summarize their distributions
5. Decide whether the segmentation is fit for a downstream quantitative purpose

## Cell-by-cell outline

### Markdown cell — Title and goals

Restates the four learning objectives. Briefly recaps what was covered in the morning lecture about segmentation and pretrained models. Notes the dataset that will be used.

### Code cell — Setup and imports

Detects Colab; pip-installs Cellpose and lab-specific dependencies (cellpose, numpy, scikit-image, matplotlib, tifffile). Imports the relevant modules. Confirms GPU availability and picks the appropriate Cellpose backend.

*Pseudocode:* conditional install of `cellpose`; `from cellpose import models, io, utils`; check GPU with `torch.cuda.is_available()`.

### Code cell — Download the canonical dataset

Pulls a small canonical Cellpose-friendly dataset to the Colab runtime. The dataset includes images that Cellpose handles well and images where it struggles, deliberately.

*Pseudocode:* download from a stable URL or use Cellpose's bundled example data; list the resulting files.

### Markdown cell — A note on the dataset

Brief description of what the dataset is, why it was chosen, and what mix of "easy" and "hard" examples it contains. Important: attendees should know the lab is set up to surface failures, not just successes.

### Code cell — Load and display the first image

Loads the first image, prints its shape, displays it inline with appropriate contrast. The image is a "clean" example where Cellpose is expected to perform well.

### Code cell — Run Cellpose on the first image

Initializes the Cellpose model (using a pretrained `cyto` or `nuclei` model depending on the image). Runs inference. Returns the segmentation mask.

*Pseudocode:* `model = models.Cellpose(model_type='cyto')`; `masks, flows, styles, diams = model.eval(img, ...)`.

### Code cell — Visualize the segmentation

Displays the image with the segmentation overlaid as colored instance labels. Side-by-side comparison: original image, mask, overlay.

### Markdown cell — Inspect the result

Discussion prompt: where did Cellpose succeed? Are there any objects it missed, merged, or split? Capture observations before moving on.

### Code cell — Quantify per-object features

Uses scikit-image's `regionprops` to extract per-object features: count, area, equivalent diameter, eccentricity. Stores them in a pandas DataFrame.

*Pseudocode:* `from skimage.measure import regionprops_table`; `props = regionprops_table(masks, properties=[...])`; `df = pd.DataFrame(props)`.

### Code cell — Summarize and plot feature distributions

Plots histograms or violin plots of the per-object features. Prints summary statistics. Asks the attendee to predict what the distributions should look like before showing them.

### Markdown cell — Reflection on the easy case

Brief reflection: when Cellpose works well, the workflow is straightforward — load, run, quantify. The next cells move to the harder case.

### Code cell — Load and run on a harder image

Loads a second image from the dataset, deliberately chosen to challenge Cellpose: out-of-distribution sample type, unusual stain pattern, dense or overlapping objects. Runs Cellpose with the same model. Visualizes the result.

### Markdown cell — Predict the failure modes

Discussion prompt before revealing the harder case's output. Where do attendees expect Cellpose to fail? After revealing, compare predictions to actual failures.

### Code cell — Quantify on the harder image

Same quantification pipeline as before. Compute the same features. Compare distributions to the easy case.

### Markdown cell — Side-by-side comparison

Displays both images, both segmentations, both feature distributions side by side. Discussion prompt: if you reported these counts in a paper, would you trust them? What would you do differently?

### Code cell — Try a different pretrained model

Switch from `cyto` to `nuclei` (or appropriate alternative). Re-run on the harder image. Observe whether a different pretrained variant performs better. This introduces the idea that pretrained models are not interchangeable.

### Code cell — Adjust segmentation parameters

Walk through `flow_threshold`, `cellprob_threshold`, and `diameter`. For each, demonstrate the effect of changing the value on the output. The attendee runs three or four parameter combinations and observes the effect.

### Markdown cell — When does parameter tuning help, and when does it just produce different mistakes?

A reflection prompt addressing the temptation to keep tuning until the segmentation looks right. The intent is to draw out the difference between *fitting parameters* and *fitting biological reality*.

### Code cell — Save outputs for Lab 2

Writes the segmentation masks and per-object feature tables to disk so Lab 2 can pick them up. Confirms the files are saved.

### Markdown cell — Closing reflection and bridge to Lab 2

Summarizes what the lab demonstrated: pretrained Cellpose works well in some cases, fails in others, and the failures are not always obvious without ground truth. Lab 2 will introduce ground truth and validation methodology.

## Common pitfalls

- *GPU runtime not enabled.* If the attendee did not enable a GPU runtime in Colab, Cellpose will run on CPU and be slow. The setup cell prints a clear message if this happens.
- *Out-of-memory errors.* Free-tier Colab can run out of GPU memory on larger images. The lab uses small enough images that this is unlikely; if it happens, the attendee can downsample the image first.
- *Confusing model variants.* Cellpose has multiple pretrained models. The notebook's first run is on `cyto`; the second deliberately tries `nuclei`. The lab does not get into the full model zoo to keep focus.
- *Beautiful-output bias.* Attendees may be impressed by the easy-case output and skip the critical evaluation. Instructors should call attention to the harder-case failures explicitly during the room walkthrough.

## Extensions (not assigned)

For attendees who finish early or want to push further:

- Try Cellpose's diameter estimation on an image where you don't know the diameter; compare the estimate to a manually-measured value
- Run Cellpose on one of your own images (if you brought one); discuss in the wrap-up
- Apply per-object filtering (size threshold, shape threshold) and observe how the feature distributions change

## What this lab does *not* do

- Fine-tune Cellpose on custom data (deferred to T2B.3 in the broader curriculum)
- Compare Cellpose to alternatives (Stardist, Mesmer, segment-anything) — Lab 3 option B touches segment-anything
- Process 3D or time-series data (deferred to T3M.5 in the broader curriculum)

## Implementation notes for the build-out

When this notebook is authored as a working `.ipynb`:

- Pin Cellpose to a specific minor version (the API has stabilized but tutorials drift)
- Pre-stage the dataset so download is fast and reliable
- Include a clean instructor-solution copy in `solutions/`
- Add expected runtime annotations on each major cell
- Test on free-tier Colab before any external delivery
