---
layout: default
title: "02 — Validation and quantification"
parent: Notebooks
nav_order: 3
---

# Notebook 02: Validation and Quantification

**Status:** Outline only. Working notebook to be authored in a later build step.
**Lab time:** 75 minutes (workshop slot 14:45–16:00).
**Prerequisites:** Notebook 01 completed; segmentation outputs saved.
**Curriculum mapping:** T1.4 (validation principles), T2B.5 (advanced validation), T3M.3 (quantification in microscopy).

## Purpose

Take the Lab 1 segmentation outputs, validate them against ground truth, and demonstrate the difference between metric correctness and biological correctness. The lab puts the morning lecture's validation message to direct hands-on test: a model can have a high IoU and still produce systematically wrong biological measurements.

## Learning objectives

By the end of this notebook, attendees should be able to:

1. Load ground-truth segmentation masks and align them with model predictions
2. Compute pixel-level metrics (IoU, Dice) and instance-level metrics (precision, recall) appropriate to instance segmentation
3. Compute per-object biological measurements (count per condition, mean cell area) from both predicted and ground-truth segmentations
4. Identify systematic errors that affect biological measurements but do not show up clearly in pixel-level metrics
5. Articulate to a non-AI colleague whether the segmentation pipeline is fit for a stated biological question

## Cell-by-cell outline

### Markdown cell — Title and goals

Restates the five learning objectives. Recaps the Lab 1 outputs that this lab builds on. Frames the lab's central question: how do we know our segmentation is good enough for the biological question we are asking?

### Code cell — Setup and imports

Detects Colab; pip-installs validation-related dependencies (scikit-image, scikit-learn, pandas, seaborn). Imports modules. Loads the segmentation masks and feature tables saved at the end of Lab 1.

### Code cell — Load ground-truth annotations

Loads ground-truth segmentation masks for the same images used in Lab 1. The dataset includes ground truth deliberately so this comparison is possible. Brief discussion (in the following markdown cell) about how ground truth was generated for this dataset and the inter-rater agreement that informs its limits.

### Markdown cell — A note on what "ground truth" means here

Important framing: ground truth in image segmentation is rarely the absolute biological truth. It is typically expert annotation, with disagreement between expert annotators that bounds how well any model can possibly do. This cell discusses the dataset's annotation provenance, measured inter-rater agreement, and what that implies for the metrics we are about to compute.

### Code cell — Visualize prediction versus ground truth side by side

Displays for one image: original, prediction, ground truth, and a difference / overlay map highlighting where they disagree. Use a clear color scheme (green for true positive, red for false positive, blue for false negative).

### Code cell — Compute pixel-level metrics

Compute IoU, Dice, pixel-level precision and recall. Print results.

*Pseudocode:* convert instance masks to binary; `iou = intersect / union`; `dice = 2 * intersect / (sum_pred + sum_gt)`; precision and recall from confusion matrix.

### Code cell — Compute instance-level metrics

For instance segmentation, pixel-level metrics are insufficient. Compute matched-instance precision and recall at IoU thresholds (e.g., 0.5, 0.7). Use the standard matching algorithm (Hungarian matching or greedy at the threshold).

*Pseudocode:* match each predicted instance to its highest-IoU ground-truth instance above a threshold; count matched / unmatched pairs; compute precision and recall per threshold.

### Markdown cell — Why instance metrics differ from pixel metrics

Brief explanation: pixel IoU can be high even when instance counts are wrong (one merged blob covers two cells with high pixel agreement but is wrong as an instance). This is a key concept the lab is designed to surface.

### Code cell — Quantify biological measurements from prediction and ground truth

Compute per-image: total cell count, mean cell area, area distribution. Compute these from both prediction and ground truth. Compare.

### Code cell — Plot prediction-versus-ground-truth biological measurements

Scatter plot: ground-truth count on x, predicted count on y, one point per image. Add y=x reference line. Repeat for area. The intent is to make systematic bias visible.

### Markdown cell — Reading the comparison plot

Discussion prompts: is the prediction systematically biased (e.g., always undercounts)? Is the bias the same across all conditions, or is it worse in some? If you reported the predicted counts, what would your error bars look like? What if you reported the predicted area distributions instead?

### Code cell — Compare metrics versus biological correctness

Tabulate: image-by-image IoU, Dice, count error, area error. Compute correlations between IoU and count error. The expected finding is that IoU does not strongly predict count error — high IoU images can still have noticeable count error, and vice versa.

### Markdown cell — The key insight

This cell is the lab's central learning moment. The metric–biology gap is real. A model can be "good" by IoU and still produce systematically wrong biological measurements. The implication: choose validation metrics that match the biological question.

### Code cell — Choose a validation metric appropriate to a biological question

Walk through three example biological questions and the appropriate validation metric for each:

- "How many cells are in this image?" → count error
- "What is the mean cell size in this condition?" → mean-area error
- "Is the size distribution different between conditions?" → distribution-comparison metric (KS, Earth Mover's, etc.)

Run each on the data and report.

### Code cell — Sensitivity to threshold and parameters

Re-run the validation metrics with different segmentation parameter settings (carrying through from Lab 1's parameter exploration). Show how each metric responds. The goal is to make the dependence between segmentation parameters and downstream biological measurements visible.

### Markdown cell — Closing reflection and bridge to Lab 3

Summary of what the lab demonstrated: segmentation outputs need validation against ground truth, and the right validation metric depends on the biological question. Lab 3 extends the responsible-use thread to a different challenge: how do you validate a model output when, by definition, you don't have a clean version to compare against?

## Common pitfalls

- *Mismatched mask formats.* Lab 1 saves instance masks as integer-labeled arrays; ground truth may be stored differently. The setup cell normalizes both to a common format with comments explaining the conversion.
- *Off-by-one errors in instance matching.* Hungarian matching can fail subtly when zero-labeled background is treated as an instance. The matching code includes explicit background handling and a comment about this trap.
- *Drawing strong conclusions from one image.* Encourage attendees to compute metrics across the full set of images, not just the one they happen to be looking at.

## Extensions (not assigned)

- Compute the metrics under deliberate perturbations (slightly shrunk masks, slightly merged masks) to develop intuition for what each metric is sensitive to
- Try alternative instance-matching strategies and observe sensitivity
- Apply the validation pipeline to one of the parameter settings you tried in Lab 1 to identify the "best" parameter setting for a specific biological question

## What this lab does *not* do

- Train a custom segmentation model (deferred to T2B.3)
- Address validation when no ground truth is available (this is the central challenge of Lab 3 option A)
- Address external validation across institutions or scanners (deferred to T2B.5)

## Implementation notes for the build-out

- Choose a dataset with clear ground truth and quantified inter-rater agreement; the lab depends on this
- Pre-stage all data so the lab can run end-to-end in 75 minutes including discussion
- Provide a clean solution notebook in `solutions/`
- Test on free-tier Colab; the validation computation is light and should complete in seconds per image
