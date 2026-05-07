---
layout: default
title: "Lecture 1: Intro to AI for Imaging"
parent: Lectures
nav_order: 1
---

# Lecture 1: AI for Scientific Image Analysis — What Works, What Doesn't, and Why

**Time:** 90 minutes (≈ 70 min content + ≈ 20 min Q&A)
**Format:** Slide-based lecture with four demo videos / screen recordings interspersed
**Audience:** Biologists with technical fluency; assumes biology and microscopy fluency, no ML background
**Slides:** `lectures/slides/01_intro_ai_imaging.pdf` (to be authored)
**Curriculum mapping:** Compresses T1.1 (landscape and history), T1.2 (ML fundamentals for image data), T1.3 (artifacts and preprocessing), T1.4 (validation principles), T1.6 (uncertainty), T1.8 (responsible use, briefly), and introduces T1.9 (image registration) at task-taxonomy level

## Learning objectives

By the end of the lecture, attendees should be able to:

1. Place a given image-analysis problem into the correct AI task category (classification, detection, segmentation, restoration, generation, registration, or tracking)
2. Describe at a conceptual level what training, validation, and generalization mean for an image model
3. Identify three to five common failure modes of AI in microscopy and explain why each occurs
4. Articulate the difference between a model that works well in a paper and a model that works well on their data
5. Describe what validation evidence they should ask for before trusting an AI tool's output

## Section outline

### Section 1 — Why now? (10 min)

The convergence of compute, data, and methods that brought AI to scientific imaging in the last decade. Three drivers, briefly:

- *Compute.* GPU availability and the scale shift from single-machine workflows
- *Data.* Public datasets, image-sharing infrastructure (image.sc, BioImage Archive, IDR), reusable training corpora
- *Methods.* CNN era, vision transformers, foundation models — but with the framing that *methods don't matter without the other two*

Frame the rest of the lecture as: AI is here because it now works for some problems; the day is about understanding which problems and how to use it responsibly.

### Section 2 — The image-analysis tasks AI addresses (15 min)

Walk through the major task types, each with a one-slide microscopy example:

- **Classification.** Image-level or patch-level labels. Example: cell-cycle phase, phenotype, mitosis-or-not.
- **Detection.** Bounding boxes or points. Example: spot detection (FISH spots, vesicles).
- **Segmentation.** Pixel-level labels (semantic) or instance-level (each cell is its own object). Example: Cellpose nuclei segmentation.
- **Restoration.** Image-to-image transformation: denoising, deconvolution, super-resolution. Example: CARE-restored low-light fluorescence.
- **Generation.** Producing new images from input or noise: virtual staining, in silico labeling. Example: brightfield-to-fluorescence prediction.
- **Registration.** Aligning images to a common spatial reference: stitching multiple fields of view, drift correction in time-lapse, multi-channel alignment, cross-modality alignment, atlas-based normalization. Example: AI-driven stitching in microscopy, deep-learning image registration (VoxelMorph and similar) in clinical imaging, serial-section alignment in pathology.
- **Tracking.** Connecting objects across time. Example: cell-lineage tracking in time-lapse.

For each, name a representative tool the audience may have encountered and indicate where it sits relative to the task taxonomy.

### Section 3 — How models actually learn (15 min)

Conceptual scaffolding without architecture details. Cover:

- *Training, validation, and test data.* What each is for, why they need to be different, and what data leakage looks like.
- *Loss and gradient descent at the conceptual level.* The "learning rule" intuition; one slide, no math.
- *Generalization.* Why a model that does well on training data may do poorly on yours. The relationship between training-data diversity and model robustness.
- *Pretrained models, fine-tuning, and zero-shot inference.* The three operating modes the rest of the day will use, defined plainly.

The key idea: a model is a parametric function fit to data; everything that can go wrong with that fit can go wrong with the model.

### Section 4 — When AI works well, and when it doesn't (15 min)

This is the section attendees will remember most. For each of the task types in Section 2:

- *Where it currently works well.* Concrete examples of robust, well-validated applications.
- *Where it doesn't.* Concrete examples of common failure modes, with screenshots: out-of-distribution input, scanner / staining variation, rare biological presentations, edge artifacts, hallucinated features in restoration outputs.

A short tour of failure-mode patterns:

- *Domain shift.* Tools trained on one organism, modality, or stain often degrade on another.
- *Sample-preparation effects.* Fixation, mounting, and staining choices drive failure modes that the model has never seen.
- *Confident wrongness.* Models often produce outputs that look plausible to a non-expert but are wrong; this is the failure mode biologists most need to recognize.
- *Hallucination in generative methods.* Restoration and virtual staining can invent features that were never in the input.
- *Misregistration in alignment tasks.* When source and target images differ enough — different sample types, large or non-rigid deformations, missing or extra structures — registration models can align some structures correctly while distorting others, corrupting downstream measurements without leaving an obvious signature.

Tie to the day's labs: Lab 1 will demonstrate confident wrongness; Lab 3 will demonstrate hallucination if option A is chosen.

### Section 5 — Validation: how to know your output is right (10 min)

The lecture's responsible-use spine. Cover:

- *What validation is for.* Establishing trust in a specific use case, not generic accuracy.
- *Internal versus external validation.* Why both matter.
- *Metrics versus biological correctness, and metrics versus the right metric.* IoU and Dice are not the same as "the count is right." Different tasks also require different metrics — segmentation uses overlap-based metrics, restoration uses PSNR or SSIM, registration uses landmark distance or structure preservation, tracking uses ID consistency. We will demonstrate the segmentation case concretely in Lab 2; the principle generalizes to every task in Section 2.
- *Reading the literature critically.* What kinds of evidence to look for in published AI methods, and what to be skeptical of.

A short list of red-flag patterns when reading a paper or a vendor's marketing material:

- Reported metrics on a single dataset and no external validation
- No discussion of failure modes
- Black-box claims without access to the model or training data
- Generic accuracy figures without per-class or per-condition breakdown

### Section 6 — Reproducibility and responsible use (5 min)

A compressed slice of T1.7 and T1.8. Cover:

- Versioning models and code; documenting parameters
- Reporting standards (CLAIM and similar) at the level of "these exist; learn them when you publish"
- Bias sources and population effects
- Image integrity: AI restoration and generation methods can introduce changes that violate journal image-integrity policies if not reported

### Section 7 — Closing (5–10 min) and Q&A

The closing slide situates the day: AI is a tool, not a magic wand; the workshop will give attendees their first hands-on experience using it critically. Pivot directly into the [walking-the-day talk]({{ site.baseurl }}/lectures/02_walking_the_day) after the break.

## Demo videos / screen recordings to embed

Four short clips (≤ 1 minute each) interspersed in Sections 2 and 4:

1. *Cellpose running on canonical data.* Visual proof that segmentation works well in a clean case.
2. *Cellpose failing on out-of-distribution data.* Same model, different sample: confident-but-wrong output.
3. *A virtual-staining or restoration result with an obvious hallucination.* Used in Section 4 to make the integrity point visceral.
4. *A registration alignment example, successful and failed.* A microscopy stitching or time-lapse alignment that succeeds, paired with one that misregisters subtly — used in Section 2 to make the registration task concrete and again in Section 4 to illustrate the misregistration failure mode.

## Discussion prompts (Q&A or interspersed)

- "Has anyone here trusted an AI output that turned out to be wrong? What did the failure look like?"
- "Where in your work do you think AI would help most? Where would you be most worried about deploying it?"
- "What evidence would convince you a vendor's tool actually works on your data?"

## Materials needed

- Slide deck (`lectures/slides/01_intro_ai_imaging.pdf`)
- Four demo videos (`lectures/demos/`)
- Backup static screenshots in case AV fails
- Whiteboard or shared document for capturing audience questions and follow-ups

## Notes for instructors

- Resist the temptation to teach architectures. Attendees do not need to know what a U-Net is; they need to know what it does and where it fails.
- Tie every concept to the afternoon labs explicitly. The lecture should set up Lab 1 by the end of Section 4 and Lab 2 by the end of Section 5.
- The hardest section to deliver is Section 3. Watch the room; if eyes glaze, drop into more concrete examples and skip the gradient-descent slide entirely.
