---
layout: default
title: "03a — AI denoising (option A)"
parent: Notebooks
nav_order: 4
---

# Notebook 03a: AI Denoising with Noise2Void (Lab 3, Option A)

**Status:** Outline only. Working notebook to be authored in a later build step.
**Lab time:** 60 minutes (workshop slot 16:15–17:15).
**Prerequisites:** Notebook 00 completed; the morning lecture's restoration and hallucination content covered.
**Curriculum mapping:** T3M.6 (AI-based image restoration: denoising, deconvolution, super-resolution), T1.6 (uncertainty), T1.8 (responsible use, image integrity).

## When to choose this option

Lab 3 has two options. Run option A (this notebook) if the audience works substantially with low-SNR or live-cell imaging, where noise is a routine experimental constraint. Run [option B]({{ site.baseurl }}/notebooks/03b_foundation_model_segmentation) if the audience is more interested in the methodological frontier or works with non-canonical sample types.

## Purpose

Apply self-supervised AI denoising to a low-SNR live-cell sequence and reason about hallucination risk. The lab demonstrates that AI restoration is a real and useful capability, and that it carries an integrity cost: the denoised image contains content the model invented. Workshops on responsible AI use that skip this point produce attendees who undervalue uncertainty.

## Learning objectives

By the end of this notebook, attendees should be able to:

1. Apply Noise2Void to a noisy microscopy image and produce a denoised result
2. Compare the denoised result to a clean reference (where available) and to the noisy input
3. Identify cases where Noise2Void restores real signal and cases where it invents features
4. Articulate to a non-AI colleague the integrity reporting expectations for AI-restored images
5. Decide whether AI denoising is appropriate for a stated experimental purpose and what reporting to attach if it is

## Cell-by-cell outline

### Markdown cell — Title and goals

Restates the five learning objectives. Recaps the morning lecture's content on restoration and hallucination. Notes the dataset that will be used and why it was chosen.

### Code cell — Setup and imports

Detects Colab; pip-installs Noise2Void / CARE / appropriate denoising package, plus visualization libraries. Imports modules. Confirms GPU availability — denoising training is GPU-friendly though not strictly required for inference on small images.

### Code cell — Download the noisy dataset

Pulls a canonical low-SNR dataset to the Colab runtime. The dataset is structured with paired noisy and clean reference images so attendees can compare against ground truth where appropriate.

### Markdown cell — A note on the dataset

Brief description: what the data are, why they are noisy (live-cell, photobleaching budget, fast imaging), and the fact that paired clean-noisy pairs are available *for evaluation only*. The point: in real experiments, attendees will *not* have a clean reference, which is the entire reason self-supervised denoising matters.

### Code cell — Inspect a noisy image and a clean reference

Display a noisy image and its paired clean reference side by side. Discuss the noise characteristics (Poisson + Gaussian for fluorescence). Set up the framing for what Noise2Void will attempt to do.

### Code cell — Train Noise2Void on the noisy data

Apply Noise2Void's self-supervised training procedure on the noisy data. Use a short training run (a few hundred steps) to keep the lab within the 60-minute window. Save the trained model.

*Pseudocode:* configure Noise2Void model and trainer; fit on noisy images; track loss; save checkpoint.

### Markdown cell — What just happened conceptually

Brief explanation of the Noise2Void principle: the model learned to predict each pixel's clean value from its neighbors, without ever seeing a clean reference. This is a critical insight that distinguishes self-supervised denoising from supervised methods that require paired data.

### Code cell — Run inference and visualize

Apply the trained model to a held-out noisy image. Display the noisy input, the denoised output, and the clean reference side by side. Compute the per-pixel difference (denoised minus clean) and visualize it as a heatmap.

### Code cell — Quantify the denoising

Compute PSNR and SSIM between denoised and clean reference. Print results. Compare to the PSNR/SSIM between noisy input and clean reference (the no-denoising baseline).

### Markdown cell — A discussion of metrics

Brief discussion: PSNR and SSIM are standard image-quality metrics. They reward outputs that look close to the reference. They do not directly reward biological correctness. This recapitulates Lab 2's metric–biology gap in a different setting.

### Code cell — The hallucination check

Look closely at the difference image (denoised minus clean). Identify regions where the denoised image differs from the clean reference in ways that are *not* due to denoising the noise — i.e., features that were invented by the model rather than restored from the data.

This is the lab's central learning moment.

*Pseudocode:* compute structural difference between denoised and clean; threshold at a level that excludes simple noise variation; visualize the regions where the model changed structural content.

### Markdown cell — When does denoising hallucinate?

Discussion prompts: where in this image did the model invent or alter features? Are those alterations small enough to ignore for quantitative analysis? What about for qualitative figure presentation? What if you reported a denoised image in a paper without disclosure — would that be acceptable under image-integrity policies?

### Code cell — Apply on a "no clean reference" example

Take a noisy image for which no clean reference exists. Apply the trained model. Visualize the result. Without a reference, the attendee has only the model's word that the output is correct. Discuss what would establish trust in this output: independent biological controls? Expert review? External experiments?

### Code cell — Integrity reporting walkthrough

Walk through what the attendee would need to write in a paper or report to comply with contemporary image-integrity standards if they used this denoising. Generate a draft methods paragraph in markdown.

*Pseudocode:* render markdown describing the Noise2Void model version, training data, training steps, inference parameters, and disclosure that the displayed images are AI-restored.

### Markdown cell — Closing reflection

Summary of what the lab demonstrated: self-supervised denoising is real and useful, and it costs you something. The cost is integrity. Reporting AI-denoised images without disclosure violates contemporary image-integrity expectations. The decision to use AI denoising should be made deliberately and reported transparently.

## Common pitfalls

- *Training time.* If the training step takes too long on free-tier Colab, the lab runs over time. Pin the training-step count to a value that completes within 10 minutes on free-tier GPU.
- *Confusing self-supervised with supervised.* Some attendees will assume the model is trained on the clean reference because the lab shows clean references for comparison. Emphasize that the references are for *evaluation only*, not training.
- *Beautiful-output bias, again.* Denoised images can look stunning. Attendees may not notice the hallucinated regions unless explicitly asked to look. The hallucination-check cell exists for exactly this reason; instructors should walk the room during it.

## Extensions (not assigned)

- Try training Noise2Void with different mask schemes and compare results
- Apply the trained model to a different noise level (lower or higher) and observe how it generalizes
- Try a supervised denoising approach (CARE) on the same data with paired references and compare PSNR/SSIM and hallucination patterns

## What this lab does *not* do

- Train CARE or other supervised denoising methods (would require paired data and longer training time)
- Address deconvolution as a separate restoration topic (deferred to a longer T3M.6 module)
- Address super-resolution from widefield (deferred; conceptually similar but methodologically distinct)
- Quantify hallucination at scale or systematically (research-grade question)

## Implementation notes for the build-out

- Use a dataset with established noise characteristics so the denoising model has a fair chance to converge in a short training run
- Pre-train an instructor reference checkpoint and ship it so attendees who run out of time can still see the inference and hallucination steps
- Pin all package versions; self-supervised denoising packages have evolved
- The hallucination-check cell deserves particular care: choose images where the hallucination is visible to the naked eye, not subtle
