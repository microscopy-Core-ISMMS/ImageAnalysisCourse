# Lecture 1 — Interactive Notebook and HTML Slides

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/lectures/01_intro_ai_imaging/lecture.ipynb)

Click the badge above to open `lecture.ipynb` in Google Colab and run the four interactive demo cells (numpy + matplotlib, no GPU needed). The slideshow does **not** paginate in Colab — for the presentation experience, view `lecture_slides.html` in a browser instead.

This folder contains the interactive Jupyter notebook and exported HTML slide deck for **Lecture 1: AI for Scientific Image Analysis — What Works, What Doesn't, and Why**. The notebook is the canonical source; the HTML is exported from it.

## Contents

| File | Purpose |
|---|---|
| `lecture.ipynb` | Source notebook with reveal.js slideshow metadata. Open in Jupyter Lab / Colab. |
| `lecture_slides.html` | Self-contained HTML slide deck. Open in any browser. |
| `build_lecture_notebook.py` | Build script that generates `lecture.ipynb` from Python. |
| `references.bib` | BibTeX bibliography for all cited works. |
| `verification_log.md` | Per-reference verification status and audit trail. |
| `colab_setup.py` | Minimal Colab runtime setup helper. |

## How to view, present, and edit

### View the slides

Open `lecture_slides.html` in any modern browser (Chrome, Firefox, Safari, Edge). Slides advance with arrow keys (right/down for next, left/up for previous). Press `f` for fullscreen, `s` for speaker notes view.

The HTML is fully self-contained except for one CDN dependency (reveal.js itself, loaded from unpkg.com). Works offline if you have a previously-rendered local cache; if you need a strictly air-gapped version, see the "offline export" note below.

### Present live from the notebook

Two options:

1. **From the exported HTML.** Open `lecture_slides.html` in a browser. This is the simplest path and works everywhere.
2. **From the notebook with RISE.** In Jupyter Lab or Notebook with the `jupyterlab-rise` (Lab) or `RISE` (classic Notebook) extension installed, open `lecture.ipynb` and click the slideshow toggle. Live code cells become runnable during the presentation. Note that **RISE does not currently work in Google Colab**; for live presentation use a local Jupyter install.

### Run interactive code cells

Open `lecture.ipynb` in Jupyter Lab or Colab — click the **Open in Colab** badge at the top of this README — and run cells normally. The notebook contains four runnable demonstration cells (setup check, train/val/test split, Cellpose-style result viewer, metrics-vs-biology scatter); the rest are markdown slides that render as plain markdown when viewed in Colab.

### Edit and rebuild

The lecture is authored in Python via `build_lecture_notebook.py`. Edit the script, then re-run it:

```bash
python build_lecture_notebook.py
```

This regenerates `lecture.ipynb`. To re-export the HTML slides:

```bash
jupyter nbconvert --to slides lecture.ipynb --SlidesExporter.reveal_theme=white --output lecture_slides
```

You can edit `lecture.ipynb` directly in Jupyter Lab if you prefer; the build script and the notebook both produce equivalent output, but the script is easier to diff and review.

## Verification, citations, and hallucination flags

Every cited work in the lecture carries an inline verification flag using one of four symbols:

- **✓** verified via web search this session
- **✏** corrected — initial training-knowledge claim was wrong; entry reflects correction
- **⚠** uncertain — drawn from training, not directly verified; sanity-check before publication
- **❓** synthesis — interpretation or framing, not a specific citation

Flags appear inline next to citation chips on each slide and in the bibliography slide at the end. Full audit trail is in `verification_log.md`.

**Important caveat:** Phase 1 verification used WebSearch results, which surface metadata from canonical sources but do not retrieve full text. Two references (Cellpose, μSAM) were spot-checked directly. All references should still be verified against published source articles before any external delivery — search-result metadata can lag, mislabel, or contain errors.

## Build status — v1.0 complete

This lecture has gone through all three planned build phases.

**What is in place:**

- 53 slides + sub-slides covering all seven sections plus the Resources Appendix
- ~30 verified citations across `references.bib` (round 1 + round 2 verifications)
- Verification log with audit trail (`verification_log.md`)
- Four working interactive code cells (numpy + matplotlib only, runs anywhere)
- Resources Appendix (10 slides) navigable from any slide
- Auto-generated handout (`../../handout.md`) and full resources page (`../../resources.md`)
- Speaker cue sheet (`speaker_cues.md`) covering audience signals, drill-down opening cues, and time-pressure cues
- HTML export tested and rendering with reveal.js
- Pinned dependency versions in `colab_setup.py`

**Items deliberately deferred to real-world authoring (out of scope per project agreement):**

- *Demo videos.* The lecture references four demo clips (Cellpose working, Cellpose failing, restoration hallucination, registration success/failure). v1 has placeholder text; an authoring pass should source verified embeddable videos from paper supplements (Stringer 2021, Krull 2019, Weigert 2018, Hörl 2019) or render animated stills from the interactive demo cells.
- *End-to-end Colab test.* The interactive cells use only numpy + matplotlib and should run anywhere, but a free-tier T4 walkthrough is recommended before external delivery.
- *Verification depth.* Round-1 and round-2 verifications confirmed citations from search-result metadata. Cross-checking against source GitHub READMEs and full paper PDFs is recommended before publication.

## Per-phase summary (for reference)

**Phase 1 — research and skeleton:** verification log, BibTeX, slide skeleton with section headings and citation chips. Reviewable HTML.

**Phase 2A — resource library:** opening sweep across Allen Institute, CZI, Janelia, EMBL, Blue Brain, European bioimage networks. ~50 entries with verification flags. `resources.md` at the workshop root.

**Phase 2B — architectural restructure:** drill-down sub-slides for Section 2 tasks, in-slide hyperlinks to GitHub/Colab, Resources Appendix navigable from any slide, speaker notes for key slides. Auto-generated `handout.md`.

**Phase 2C — content fill:** full slide content for Sections 3 (how models learn), 4 (failure modes), 5 (validation), 6 (responsible use). Working interactive code cells. Round-2 verification corrections applied.

**Phase 3 — polish:** bibliography updates with all verified entries, pinned Colab setup, speaker cue sheet, README finalization.

## Offline / air-gapped use

The exported HTML loads reveal.js from `unpkg.com`. To produce a strictly air-gapped HTML for presentation in environments without internet access, use:

```bash
jupyter nbconvert --to slides lecture.ipynb --SlidesExporter.reveal_theme=white --output lecture_slides_offline --reveal-prefix=reveal.js
```

then download reveal.js into a local `reveal.js/` folder next to the HTML. This is deferred to Phase 3.

## Acknowledgments and license

The lecture draws on community resources from the bioimage-analysis ecosystem. Specific tools and methods are cited inline; full bibliography is in `references.bib` and rendered on the bibliography slide at the end of the deck. Workshop materials are intended for open educational use; specific licensing will be specified at the workshop-repository root.
