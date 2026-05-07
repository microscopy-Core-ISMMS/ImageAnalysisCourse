---
layout: default
title: "03b — Foundation-model segmentation (option B)"
parent: Notebooks
nav_order: 5
---

# Notebook 03b: Foundation-Model Segmentation (Lab 3, Option B)

**Status:** Outline only. Working notebook to be authored in a later build step.
**Lab time:** 60 minutes (workshop slot 16:15–17:15).
**Prerequisites:** Notebook 00 completed; Notebooks 01 and 02 recommended for context.
**Curriculum mapping:** T3M.8 (microscopy foundation models and self-supervised learning), T4.1 (foundation models in biomedical imaging).

## When to choose this option

Run option B (this notebook) if the audience is interested in the methodological frontier or works with non-canonical sample types where fitting pretrained models do not exist. Run [option A]({{ site.baseurl }}/notebooks/03a_denoising_n2v) if the audience works substantially with low-SNR or live-cell imaging.

## Purpose

Apply a microscopy-tuned segment-anything variant (μSAM-style, or SAM with prompts) to a segmentation problem that does not have a fitting Cellpose-style pretrained model. The lab demonstrates the foundation-model paradigm: instead of training a task-specific model, you adapt a generally-trained model with prompts or light fine-tuning.

## Learning objectives

By the end of this notebook, attendees should be able to:

1. Describe the foundation-model paradigm and how it differs from task-specific pretrained models like Cellpose
2. Apply a microscopy-tuned segment-anything variant (or SAM with prompts) to segment objects in a microscopy image
3. Compare the foundation-model output to a Cellpose output on the same image
4. Identify the validation challenges specific to foundation-model output (no fixed training distribution, prompt-dependent behavior, drift across prompts)
5. Decide when a foundation-model approach is preferable to a task-specific pretrained model and what trade-offs that decision implies

## Cell-by-cell outline

### Markdown cell — Title and goals

Restates the five learning objectives. Recaps the morning lecture's content on foundation models. Frames the lab's central question: when you have a sample type that no pretrained model fits, what do you do?

### Code cell — Setup and imports

Detects Colab; pip-installs SAM (Segment Anything) and the chosen microscopy-tuned variant (μSAM or comparable). Pip-installs the visualization stack. Imports modules. Confirms GPU availability — SAM is GPU-friendly though small inferences run on CPU.

*Pseudocode:* conditional install of `segment-anything`, `micro-sam` (or equivalent); check GPU.

### Code cell — Load the SAM model

Initialize a SAM model variant. Load weights. Confirm the model is ready to run.

*Pseudocode:* `from segment_anything import SamPredictor, sam_model_registry`; load checkpoint; create predictor.

### Code cell — Load a non-canonical microscopy image

Load an image from the workshop dataset that is deliberately different from the Cellpose training distribution: a different sample type (e.g., tissue rather than cultured cells), an unusual stain, or a structure that Cellpose tends to miss.

### Markdown cell — Why this image is interesting

Brief discussion: this is the kind of image where Cellpose's pretrained model produces a poor result, and where training a custom Cellpose model would require labeled data the attendee does not have. Foundation models offer a different path.

### Code cell — Cellpose baseline on the same image

Run pretrained Cellpose on this image as a baseline. Visualize the result. The result is expected to be unsatisfactory — that is the lab's setup.

### Code cell — Apply SAM with point prompts

Run the SAM predictor on the same image, providing point prompts (one point per object). Visualize the resulting masks.

*Pseudocode:* `predictor.set_image(img)`; for each prompt point, `masks, scores, logits = predictor.predict(point_coords=..., point_labels=..., multimask_output=True)`; pick best mask.

### Markdown cell — Reading the SAM output

Discussion prompt: how does the SAM output compare to the Cellpose output on this image? Where does SAM win? Where does it struggle?

### Code cell — Apply SAM with bounding-box prompts

Re-run SAM with bounding-box prompts instead of points. Compare to the point-prompted result. Discuss when each prompt type is preferable.

### Code cell — Apply SAM in "everything" mode

Run SAM's automatic mask-generation mode that segments everything in the image without explicit prompts. Visualize. This is the foundation-model paradigm at its most general — no task-specific input, just the image.

### Code cell — Apply a microscopy-tuned variant

Run a microscopy-tuned segment-anything variant (μSAM or comparable) on the same image. Compare to vanilla SAM. The expectation: the microscopy-tuned variant performs better on microscopy images because it has been fine-tuned on microscopy-specific data.

### Markdown cell — The validation question for foundation models

Foundation models pose a validation challenge that task-specific models don't. They have no fixed training distribution to anchor expectations. Their behavior depends on prompts, which the user provides. Their generality cuts both ways: they can address novel sample types but they can also fail in surprising ways on familiar ones.

This cell discusses what validation evidence an attendee should require before trusting a foundation-model output for a downstream biological purpose.

### Code cell — Validate foundation-model output against ground truth

For one of the prompted segmentations, compute IoU and instance-level metrics against ground truth (where available). Compare the metrics to the Cellpose baseline. Compare to the metrics the workshop attendee saw in Lab 2.

### Code cell — Test prompt sensitivity

Run SAM with several different prompt configurations on the same object — different point locations, different bounding-box sizes. Observe how the output changes. The intent is to make prompt sensitivity visible.

### Markdown cell — When to use foundation models versus task-specific models

A short decision-framework discussion:

- *Use task-specific pretrained models (Cellpose, Stardist, Mesmer) when* the model's training distribution covers your data and the model has been validated on cases similar to yours
- *Use foundation models with prompts (SAM, μSAM) when* no fitting task-specific model exists, you have a small number of images, you are willing to provide per-image prompts, or you are doing exploratory analysis
- *Train a custom model (fine-tune Cellpose, fine-tune a foundation model) when* you have labeled data and the volume of work justifies the investment

### Markdown cell — Closing reflection

Summary of what the lab demonstrated: foundation models extend reach into novel sample types and unfamiliar problems, at the cost of greater prompt-engineering and validation work. The choice between task-specific and foundation-model approaches is a deliberate trade-off, not a default.

## Common pitfalls

- *Memory pressure.* SAM and its variants are larger than Cellpose. Free-tier Colab may struggle on larger images. The lab uses small enough images that this is unlikely; if it happens, downsample.
- *Prompt-engineering rabbit hole.* Tuning prompts to perfection is satisfying but distracts from the lab's actual goal. Instructors should keep attendees moving through the cells.
- *"It just works" syndrome.* The "everything" mode often produces beautiful-looking outputs that include many false positives. Attendees may take these outputs at face value. The validation cell exists to push back on this; instructors should call attention to it during the room walkthrough.

## Extensions (not assigned)

- Try fine-tuning a microscopy-tuned variant on a small set of labeled images and observe how performance changes
- Apply SAM to a 3D stack one slice at a time and compare to a 3D-aware approach
- Apply SAM's mask-generation pipeline to one of your own images (if you brought one)

## What this lab does *not* do

- Pretrain a foundation model from scratch (orders of magnitude beyond the scope)
- Use other foundation-model paradigms (e.g., contrastive, masked autoencoder) explicitly — those would belong in a longer T3M.8 module
- Address quantitative downstream analysis from foundation-model segmentations in the same depth as Lab 2 did for Cellpose

## Implementation notes for the build-out

- Choose the microscopy-tuned SAM variant carefully; the field is fast-moving and tooling shifts
- Provide a fallback to vanilla SAM if the variant is unavailable on the workshop date
- Pre-stage model checkpoints so download is reliable
- Test on free-tier Colab; SAM inference is GPU-heavy but workable
- Pin all package versions
