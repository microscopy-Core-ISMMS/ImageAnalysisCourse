---
layout: default
title: "Lecture 2: Walking the Day"
parent: Lectures
nav_order: 2
---

# Lecture 2: Walking the Day

**Time:** 45 minutes
**Format:** Short talk with notebook walkthroughs (no live coding yet)
**Audience:** Same as morning lecture
**Slides:** `lectures/slides/02_walking_the_day.pdf` (to be authored)

## Why this block exists

The afternoon labs work poorly if attendees arrive at them cold. Without context, attendees fall into the click-and-watch failure mode: they run the cells, watch outputs appear, and leave with no mental model of what just happened. This bridge talk pre-loads the conceptual structure of each lab so that, when they hit the cells, they know what they are looking for and what success looks like.

## Learning objectives

By the end of this block, attendees should be able to:

1. State the goal of each afternoon lab in one sentence
2. Predict what success will look like for Lab 1 and Lab 2
3. Anticipate at least one failure mode they expect to see in Lab 1
4. Choose between Lab 3 options A and B for their own work and explain the choice

## Section outline

### Section 1 — Lab 1 walkthrough (12 min)

Walk through the [Lab 1 notebook]({{ site.baseurl }}/notebooks/01_cellpose_segmentation) at the slide level, not running code. Cover:

- Goal: apply pretrained Cellpose to canonical data, look critically at the output, quantify simple features.
- Why Cellpose: representative of the modern bioimage-analysis stack, well-documented, runs reliably on Colab, many of you will use it.
- The dataset we'll use and why (canonical, not domain-specific, to keep the focus on method rather than data).
- The prediction we want them to make before running: where will Cellpose work well? Where will it fail?
- What we want them to do *with* the output: not just look at it, but quantify it and compare across conditions.

### Section 2 — Lab 2 walkthrough (10 min)

Walk through the [Lab 2 notebook]({{ site.baseurl }}/notebooks/02_validation_quantification) at the slide level. Cover:

- Goal: take Lab 1 outputs, validate them against ground truth, quantify systematic errors.
- The distinction between *metric correctness* (IoU is high) and *biological correctness* (the count is right). These are not the same; we will demonstrate this directly.
- Why ground truth is hard. Where the ground truth in this lab comes from. What its limitations are.
- The connection to the morning lecture: this is where Section 5 (validation) becomes hands-on.

### Section 3 — Lab 3 walkthrough (15 min)

Cover *both* Lab 3 options at the slide level so attendees understand the choice point even though they will only run one.

**Option A — AI denoising with Noise2Void.** [Notebook]({{ site.baseurl }}/notebooks/03a_denoising_n2v).

- Goal: apply self-supervised denoising to a low-SNR live-cell sequence.
- Why Noise2Void: representative self-supervised method; trains directly on the noisy data without paired ground truth; runs on Colab.
- The hallucination question: when does denoising restore real signal, and when does it invent features that were not there?
- The validation challenge: how do you trust a restored image when, by definition, you don't have a clean version to compare against?

**Option B — Foundation-model segmentation.** [Notebook]({{ site.baseurl }}/notebooks/03b_foundation_model_segmentation).

- Goal: apply a microscopy-tuned segment-anything variant or SAM with prompts to a segmentation problem that lacks a fitting pretrained model.
- Why this matters: the foundation-model paradigm changes how we think about pretrained models — the model is general, you adapt with prompts or light fine-tuning rather than training from scratch.
- The promise and the limit: foundation models extend reach into novel sample types but require new validation discipline because they have no fixed training distribution to anchor expectations.

For workshop instances that run only one Lab 3 option, the other is described briefly and pointed at as a follow-up activity attendees can run on their own.

### Section 4 — How to engage with the labs (8 min)

Practical guidance for the afternoon:

- Run cells deliberately, not on autopilot. Read what each cell is doing before you run it.
- Predict what the next cell's output will look like before running it. If the prediction is wrong, that is the most important moment to pause and figure out why.
- Pair with someone. The strongest learning happens when one person is explaining and the other is questioning.
- Ask questions early and often. The lab block has time for group debugging.
- Notebooks are saved to your own Colab account; you keep them.

## Discussion prompts

- "Before we start Lab 1, what are you most curious to see Cellpose do well at? What are you most skeptical it will get right?"
- "For Lab 3, A or B is most relevant to your own work? Why?"

## Notes for instructors

- This block is high-value but easy to under-deliver. Resist the temptation to give a 5-minute version and start labs early. The pre-loading is what makes the labs land.
- If the morning lecture ran long, prefer to compress Section 4 (engagement guidance) rather than Sections 1–3 (lab walkthroughs).
- Capture audience predictions about Lab 1 outputs on the whiteboard. Returning to them at the end of Lab 1 (during the break or wrap-up) is a powerful reflection moment.
