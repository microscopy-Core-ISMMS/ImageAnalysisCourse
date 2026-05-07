---
layout: default
title: "00 — Setup and self-check"
parent: Notebooks
nav_order: 1
---

# Notebook 00: Setup and Self-Check

**Status:** Outline only. Working notebook to be authored in a later build step.
**Run before:** the workshop day. Re-run on the morning of the workshop to confirm nothing has drifted.
**Estimated time:** 20–30 minutes.
**Colab badge target:** to be added when the notebook is authored.

## Purpose

This notebook does three things:

1. Verifies the attendee's Google Colab environment opens, mounts external resources, and runs cells reliably
2. Walks through a short sequence of Python and numpy operations on an image, exercising the core skills the workshop assumes
3. Self-grades the attendee's readiness and recommends whether to attend the optional ramp-up evening session

The self-check is non-judgmental. Falling short on the check is exactly the signal the [ramp-up evening]({{ site.baseurl }}/ramp-up/rampup_evening) is designed to address.

## Learning objectives

By the end of this notebook, attendees should be able to:

- Open a Colab notebook from a public GitHub URL and run it end-to-end
- Install workshop dependencies into the Colab runtime via the standard pip workflow
- Load a sample microscopy image and inspect its shape, dtype, and basic statistics
- Display an image inline using matplotlib
- Recognize whether they are ready for the workshop's hands-on sections, or whether to attend the ramp-up evening

## Cell-by-cell outline

The notebook is structured as alternating markdown explanation cells and code cells. Code cells are described here in prose and pseudocode — actual Python implementations are deferred to the implementation build step.

### Markdown cell — Title and orientation

Welcome message. Brief restatement of the three purposes above. Indication that the notebook self-grades at the end and that nothing about the result is held against the attendee.

### Code cell — Environment detection

Detects whether the notebook is running on Colab or locally. Prints the Python version and confirms the runtime. Sets a flag used in later cells to choose installation method.

*Pseudocode:* check `IN_COLAB = 'google.colab' in sys.modules`; print `sys.version`; print runtime info.

### Code cell — Dependency install

If running on Colab, pip-installs the workshop dependencies (numpy, scikit-image, matplotlib, tifffile, requests). If running locally, prints a reminder to use the conda env in `env/environment.yml`. Imports verified at the end.

*Pseudocode:* conditional pip install; import check; print versions of installed packages.

### Markdown cell — A note on what the install does

Short explanation of the difference between `pip install` (per-runtime, transient on Colab) and a managed conda environment (persistent locally). Used to ground attendees who have never thought about environment management.

### Code cell — Download a sample image

Downloads a small canonical microscopy image (a Cellpose example or BBBC sample) to the Colab working directory. Confirms the file exists. This exercises the network and file-system skills the workshop will rely on.

*Pseudocode:* `requests.get(SAMPLE_URL)` → write to disk → `os.path.exists` check.

### Code cell — Load and inspect the image

Opens the downloaded image with tifffile or skimage. Prints the array shape, dtype, min, max, and mean. Asks the attendee (in a markdown cell that follows) to predict what the dimensions correspond to before they look at the image.

*Pseudocode:* `img = tifffile.imread(path)`; `print(img.shape, img.dtype, img.min(), img.max(), img.mean())`.

### Markdown cell — Predict before you display

A short prompt asking the attendee to write down (in a comment cell, or just mentally) what they expect the image to look like based on its shape and statistics. This is the first time the workshop's "predict before run" habit appears.

### Code cell — Display the image

Renders the image inline with matplotlib. Includes axis labels, a colorbar, and a sensible vmin/vmax. Confirms the image displays.

*Pseudocode:* `plt.imshow(img, cmap='gray', vmin=p1, vmax=p99)`; `plt.colorbar()`; `plt.show()`.

### Markdown cell — Reflection prompt

Asks: did the displayed image match your prediction? If not, what surprised you? This is a deliberate moment of metacognition; it is also the moment at which biology-first attendees often realize they have an intuitive grasp of what they are looking at — useful for confidence early in the workshop.

### Code cell — Basic image manipulation exercise

A short exercise: extract a subregion of the image, compute its mean intensity, compare to the whole-image mean. The code is partially written; the attendee fills in three lines. Self-graded by comparing the computed values to expected values printed at the end.

*Pseudocode:* `subregion = img[y1:y2, x1:x2]`; `subregion_mean = subregion.mean()`; assert against expected.

### Markdown cell — What this exercise tested

Explains in prose what each step of the exercise demonstrated about the attendee's array-handling fluency. Names the skills explicitly (slicing, mean computation, comparison) so attendees know which gaps they have.

### Code cell — Self-grading and recommendation

Aggregates the success / failure flags from earlier cells. Prints a structured summary:

- Environment OK / Not OK
- Image loading OK / Not OK
- Image manipulation exercise OK / Not OK
- Recommendation: Ready for workshop / Recommend ramp-up evening / Suggest contacting instructors

*Pseudocode:* boolean aggregation; conditional `print` of recommendation; suggestion to attend the ramp-up evening if any of the three checks failed.

### Markdown cell — Next steps

If the recommendation is "ready," the cell points at the workshop schedule. If it is "ramp-up recommended," the cell points at the [ramp-up evening session]({{ site.baseurl }}/ramp-up/rampup_evening). If it is "contact instructors," the cell points at the support channels listed in [prerequisites]({{ site.baseurl }}/prerequisites).

## Common pitfalls

- *Colab session timeouts.* If the notebook has been idle for more than 90 minutes, Colab disconnects and the runtime is reset; the attendee will need to re-run from the beginning. Mention this explicitly in the orientation cell.
- *Free-tier resource limits.* The self-check is light; it will run on free-tier Colab. Heavier labs may benefit from Pro but do not require it.
- *Browser tab permissions.* Some institutional networks block downloads. Provide a fallback path (mounting Google Drive with a shared sample image) for affected attendees.

## Extensions (not assigned)

Optional cells at the bottom of the notebook for attendees with extra time or who want to push further:

- Display the image with a different colormap and explain why colormap choice matters for quantitative interpretation
- Compute a histogram and identify the bit-depth from the histogram
- Apply a simple thresholding operation and visualize the result

## What this notebook does *not* do

- Train any model
- Perform any segmentation
- Touch any GPU
- Use any tool covered in the labs (Cellpose, Noise2Void, segment-anything)

Those are deliberately reserved for the workshop day, where they sit on a foundation that this self-check has confirmed.

## Implementation notes for the build-out

When this notebook is authored as a working `.ipynb`:

- Pin dependency versions in the install cell so the self-check is reproducible across attendees and over time
- Use a single canonical sample image referenced by a stable URL (Cellpose's example data or a BBBC URL)
- Keep total runtime under 5 minutes on free-tier Colab to avoid frustration
- Include a fallback path for attendees who are blocked from downloading by institutional firewalls
