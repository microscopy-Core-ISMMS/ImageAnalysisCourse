"""
Build script for Lecture 2: Walking the Day.

Short bridge talk between the morning lecture and the afternoon labs.
Reuses the same reveal.js slideshow conventions as Lecture 1.

Run:
    python build_lecture2_notebook.py
    jupyter-nbconvert --to slides lecture.ipynb --SlidesExporter.reveal_theme=white --output lecture_slides
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "lecture.ipynb"


def md(slide_type, source):
    return {
        "cell_type": "markdown",
        "metadata": {"slideshow": {"slide_type": slide_type}},
        "source": source.lstrip("\n"),  # already a string; preserved with newlines
    }


def notes(source):
    return md("notes", source)


CELLS = []

CELLS.append(md("slide", """
# Lecture 2 — Walking the Day

### *A 45-minute bridge between the morning lecture and the afternoon labs*

The morning lecture set up the conceptual scaffold. The labs run on top of that scaffold.
This block is what makes the labs land instead of feeling like clicking through a tutorial.
"""))

CELLS.append(notes("""
This block is high-value but easy to under-deliver. Resist the temptation to give a
5-minute version and start labs early. The pre-loading is what makes the labs land.
"""))

CELLS.append(md("slide", """
## Why this block exists

Without context, attendees fall into the click-and-watch failure mode: they run
the cells, watch outputs appear, and leave with no mental model of what just happened.

This block pre-loads the conceptual structure of each lab so that, when they
hit the cells, they know what they are looking for and what success looks like.
"""))

CELLS.append(md("slide", """
## What we will cover

For each afternoon lab:

1. **Goal** — what we want attendees to be able to do
2. **What success looks like** — a prediction to make before running cells
3. **Where it will break** — failure modes to watch for
4. **Why it matters** — the broader workshop arc this lab sits in
"""))

CELLS.append(md("slide", """
## Lab 1 — Pretrained segmentation with Cellpose-SAM

**Goal.** Apply pretrained Cellpose-SAM (Cellpose v4) to canonical microscopy data,
look critically at the output, quantify simple features.

**Why Cellpose-SAM.** Representative of the modern bioimage-analysis stack.
Well-documented. Runs reliably on Colab. Many of you will use it in your own work.

**Dataset.** Cellpose example images — canonical, not domain-specific. The focus
is on the method, not on the data.
"""))

CELLS.append(md("subslide", """
### Lab 1 — what success looks like

Before running anything, predict:

- *Where will Cellpose-SAM work well?* Cells similar to its training distribution.
- *Where will it fail?* Out-of-distribution sample types, dense overlapping cells,
  unusual stain patterns.

Lab 1 deliberately includes both cases. The success of the easy case will feel
satisfying. The failure of the harder case is where the real lesson lives.

**The thing to do with the output:** not just look at it. Quantify it. Compare
across conditions. The morning lecture's "metrics versus biology" theme starts
playing here.
"""))

CELLS.append(md("slide", """
## Lab 2 — Validation and quantification

**Goal.** Take Lab 1 outputs, validate against ground truth, quantify systematic
errors, demonstrate the metrics-versus-biology gap directly.

**The distinction we want to make visible.** A model can have high IoU and still
produce systematically wrong biological measurements. *We will show this concretely.*

**Where ground truth comes from.** In Lab 2 it is constructed for the lesson.
In your own work it comes from expert annotation, with documented inter-rater
agreement. Real ground truth is rarely the absolute biological truth.
"""))

CELLS.append(md("subslide", """
### Lab 2 — the connection to the morning

Lab 2 is where the morning lecture's Section 5 (validation) becomes hands-on.

You will compute IoU and Dice — the metrics every published AI paper reports.
You will then compute count error and area-distribution differences — the things
that actually matter for biological conclusions.

The gap will surprise you.
"""))

CELLS.append(md("slide", """
## Lab 3 — Choose option A or B

Both options are fully described and runnable. You will run **one** during the
workshop; the other is yours to explore on your own.

**Option A — AI denoising with Noise2Void.** Self-supervised denoising on
low-SNR data. Demonstrates restoration *and* hallucination risk. Most relevant
if you work with live-cell or low-photon-budget imaging.

**Option B — Foundation-model segmentation.** SAM (and microscopy-tuned
variants) on a sample type that lacks a fitting pretrained model. Demonstrates
the foundation-model paradigm and prompt-based segmentation. Most relevant
if you work with non-canonical sample types or want exposure to the methodological
frontier.
"""))

CELLS.append(md("subslide", """
### Lab 3a — AI denoising with Noise2Void

**Goal.** Apply self-supervised denoising. Compare to a clean reference. Reason
about hallucination.

**Why it matters.** Self-supervised denoising is real and useful — *and* it
costs you something. The cost is integrity. AI restoration changes pixels;
the displayed image is no longer a measurement.

**Validation challenge.** When you have no clean reference (the real production
case), how do you trust the denoised output? Lab 3a confronts this directly.
"""))

CELLS.append(md("subslide", """
### Lab 3b — Foundation-model segmentation

**Goal.** Use Segment Anything (SAM) and microscopy-tuned variants for a
segmentation problem that lacks a fitting pretrained model.

**Why it matters.** Foundation models change how we think about pretrained
models. The model is general; you adapt with prompts or light fine-tuning
rather than training from scratch.

**Validation challenge.** Foundation-model output depends on the prompt.
There's no fixed training distribution to anchor expectations. This requires
a different validation discipline.
"""))

CELLS.append(md("slide", """
## How to engage with the labs

Practical guidance for the afternoon:

- **Run cells deliberately, not on autopilot.** Read what each cell is doing
  before you run it.
- **Predict outputs before running.** If your prediction is wrong, that's the
  most important moment to pause and figure out why.
- **Pair up.** The strongest learning happens when one person explains and
  the other questions.
- **Ask questions early and often.** The lab block has time for group debugging.
- **Notebooks are saved to your own Colab account.** You keep them.
"""))

CELLS.append(md("slide", """
## A word about the resources page

The [Resources page](../../resources) and [Handout](../../handout) are your
post-workshop reference. Every tool, every dataset, every paper we mention
today is there.

Pick *one* URL to bookmark tonight: the [image.sc forum](https://forum.image.sc).
That's where the bioimage analysis community lives.
"""))

CELLS.append(md("slide", """
## Discussion before the labs

A few prompts to set up the afternoon:

- Before we start Lab 1, what are you most curious to see Cellpose-SAM do well at?
  What are you most skeptical it will get right?
- For Lab 3, A or B is most relevant to your own work? Why?
- What questions about AI in your imaging work brought you here today?
"""))

CELLS.append(notes("""
This is the last block before lunch and labs. Use the discussion prompts to
gauge the room's energy level — if engagement is high, take more questions
before lunch. If energy is flagging, end on time and let lunch reset the room.

Capture audience predictions about Lab 1 outputs on the whiteboard. Returning
to them at the end of Lab 1 (during the break or wrap-up) is a powerful
reflection moment.
"""))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
        "celltoolbar": "Slideshow",
        "rise": {"theme": "white", "transition": "slide", "controls": True, "progress": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    for i, cell in enumerate(CELLS):
        cell["id"] = f"cell-{i:03d}"
    OUT.write_text(json.dumps(NOTEBOOK, indent=1))
    n_slides = sum(1 for c in CELLS if c.get("metadata", {}).get("slideshow", {}).get("slide_type") in ("slide", "subslide"))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes, {len(CELLS)} cells, {n_slides} slides+subslides)")
