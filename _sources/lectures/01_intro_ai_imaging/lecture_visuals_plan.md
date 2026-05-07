# Lecture 1 — Visuals plan

This file is the master checklist for every visual in Lecture 1 (`build_lecture_notebook.py`).

The lecture currently uses two helpers:

- **`embed_img(name, alt, width)`** — embeds a PNG from `figures/` as a base64-encoded `<img>` tag. Produces a self-contained slide.
- **`visual_placeholder(viz_id, description, source)`** — renders a visible dashed-border box (brand magenta) that names the missing visual, describes what it should show, and notes how to source/build it. The box is *visible* in dry-run review so the gap is obvious.
- **`embed_iframe(url, width, height, title)`** — embeds a live iframe (TF Playground, YouTube, etc.). Caveat: requires internet during delivery.

To fill a placeholder:

1. Build or source the visual.
2. Save as `figures/<placeholder_filename>.png` if it's a PNG.
3. Replace the `visual_placeholder(...)` call in the build script with `embed_img(...)` (or `embed_iframe(...)` for live embeds).
4. Mark this row's status as `built` (or `verified` after a Colab/projector test).
5. Re-run `python build_lecture_notebook.py` and `jupyter nbconvert` to refresh the deck.

---

## Status legend

- `needed` — placeholder in place; visual not yet authored.
- `sourcing` — Mount Sinai or external source identified; awaiting transfer.
- `built` — `embed_img` or `embed_iframe` is now in the slide; not yet projector-tested.
- `verified` — passed a dry-run on a projector / Colab T4 / second display.

## Already-built visuals (no placeholder needed)

These slides already carry good visuals. Keep them as-is unless you are doing a polish pass.

| Slide | Visual | Filename in figures/ |
|---|---|---|
| Title slide | Core logo (Microscopy and Advanced Bioimaging Core) | `brand_core_logo.png` |
| Section 1 — Where the data and tools live | Layered ecosystem diagram | `ecosystem_layers.png` |
| Section 2 — Task: Classification | Classification example | `task_classification.png` |
| Section 2 — Task: Detection | Detection with bounding boxes | `task_detection.png` |
| Section 2 — Task: Segmentation | Segmentation: image, masks, overlay | `task_segmentation.png` |
| Section 2 — Task: Restoration | Restoration: noisy in, denoised out | `task_restoration.png` |
| Section 2 — Task: Generation | Generation: brightfield → fluorescence-like | `task_generation.png` |
| Section 2 — Task: Registration | Registration: two images and overlay | `task_registration.png` |
| Section 2 — Task: Tracking | Tracking: trajectories over time | `task_tracking.png` |
| Section 4 — Training, validation, and test data | Train/val/test split diagram | `train_val_test.png` |
| Section 5 — Failure mode: domain shift | Same model, in-distribution vs OOD | `failure_domain_shift.png` |
| Section 5 — Failure mode: hallucination | Restored image with invented feature | `failure_hallucination.png` |
| Section 5 — Failure mode: misregistration | Successful vs failed registration overlay | `failure_misregistration.png` |
| Section 6 — What validation is for | Validation gradient: research → clinical | `validation_gradient.png` |
| Section 6 — Metrics versus biology | High IoU does not guarantee correct count | `metrics_vs_biology.png` |

---

## Placeholders

Each row corresponds to a `visual_placeholder(...)` call in the build script. The order matches lecture flow.

### Section 0 — Roadmap

#### `VIZ-ROADMAP-1` · status: needed
- **Slide:** Roadmap for the next 90 minutes
- **Visual:** Eight-section flow diagram. Horizontal track of 8 connected nodes (one per section), with subtle group labels: nodes 1-3 = "foundations", 4 = "mechanics", 5-7 = "judgment", 8 = "wrap". Color the "judgment" group in brand magenta to signal pedagogy emphasis. Approx 800x180px.
- **Source / how to obtain:** Build with matplotlib (graphviz-style boxes-and-arrows). Reference style: 3b1b's "arc of a series" map at the start of each video.

### Section 1 — Why now?

#### `VIZ-S1-OVR-1` · status: needed
- **Slide:** Section 1 — Why now? (overview)
- **Visual:** Three converging trends. Timeline (1980-2025) with three rising curves: Compute (sharp post-2010 GPU), Data (steady from public datasets), Methods (CNN 2012, transformer 2017, foundation 2022). Curves converge in the recent decade.
- **Source:** Build with matplotlib (3-line trend chart, log y-axis if needed). Anchors: AlexNet 2012, ImageNet 2009, BBBC ~2012, ZeroCostDL4Mic 2021.

#### `VIZ-S1-DRIVERS-1` · status: needed
- **Slide:** Three drivers, all at once
- **Visual:** Three icons or stylized cards: Compute / Data / Methods, with one-line milestone tags under each. Three-column layout. Approx 900x220px.
- **Source:** Build with matplotlib (3-panel) OR substitute emoji directly in the slide markdown.

### Section 2 — Tasks

#### `VIZ-S2-OVR-1` · status: needed
- **Slide:** Section 2 — The image-analysis tasks AI addresses (overview)
- **Visual:** 4+3 grid of small thumbnails — one per task category — using the existing 7 task figures. Caption: "Seven categories that cover most AI imaging."
- **Source:** Compose with matplotlib using `task_classification.png` ... `task_tracking.png` shrunk into a grid.

#### `VIZ-S2-SEG-WORKS-1` · status: needed
- **Slide:** Drill-down — when Cellpose-SAM works
- **Visual:** Successful Cellpose-SAM segmentation on a real fluorescence image. Two-panel side-by-side (raw + clean overlay).
- **Source:** **REQUEST FROM USER**: Mount Sinai canonical fluorescence image with clean cell morphology. **Alt:** build with matplotlib using `skimage.data.cells3d()` middle slice + Cellpose-SAM run output (already produced by Notebook 01).

#### `VIZ-S2-SEG-FAILS-1` · status: needed
- **Slide:** Drill-down — when Cellpose-SAM fails
- **Visual:** Failure example: tissue cross-section or dense-overlap image where the model produces wrong masks. Annotated arrows pointing to merge/miss/hallucinated cell.
- **Source:** **REQUEST FROM USER**: Mount Sinai tissue cross-section, dense IHC, EM tile, or atypical morphology image with known Cellpose-SAM failure. **Alt:** Notebook 01's `img_hard_synth` or BBBC tissue dataset. Tied to outstanding **task #48** (real OOD data).

### Section 3 — Common model families

#### `VIZ-S3-OVR-1` · status: needed
- **Slide:** Section 3 overview
- **Visual:** 4-card 2×2 grid (CNN / Transformer / GAN / Other). Each card has a tiny stylized architecture sketch + 1-line description + workshop-lab pointer.
- **Source:** Build with matplotlib (4 small subplots). Reference: Stanford CS231n architecture overview.

#### `VIZ-S3-CNN-1` · status: needed
- **Slide:** CNNs — Convolutional Neural Networks
- **Visual:** U-Net architecture diagram: encoder path → bottleneck → decoder path with horizontal skip connections. Approx 900x320px.
- **Source:** Build with matplotlib OR use Ronneberger 2015 Fig 1 directly under CC BY 4.0. Reference: Olaf Ronneberger's GitHub repo has clean SVG renderings.

#### `VIZ-S3-TRANS-1` · status: needed
- **Slide:** Transformers — global context via self-attention
- **Visual:** Two-panel: (left) ViT patch tokenization on a 4×4 grid; (right) self-attention heatmap for one selected token, showing attention weights to all other tokens.
- **Source:** Build with matplotlib (left = imshow with grid; right = imshow heatmap of synthetic attention weights). **Alt:** still from 3Blue1Brown's GPT video (~ attention visualization), fair-use with attribution.

#### `VIZ-S3-GAN-1` · status: needed
- **Slide:** GANs — adversarial generation
- **Visual:** Adversarial loop: Real images, Generator (with noise input) producing Fake images, both feeding a Discriminator that outputs "real or fake?". Gradient flow shown in magenta.
- **Source:** Build with matplotlib (boxes + arrows). Reference: Goodfellow 2014 Fig 1 OR Lilian Weng's blog diagrams (CC BY-NC-SA).

#### `VIZ-S3-OTHER-1` · status: needed
- **Slide:** Other architectures worth recognizing
- **Visual:** 2×2 grid: U-Net / Autoencoder / pix2pix gen / fnet, each as a small flowchart with the shared encoder-decoder backbone bracketed in soft gray.
- **Source:** Build with matplotlib (4 small subplots). Reference: Ronneberger 2015, fnet (Ounkomol 2018), pix2pix (Isola 2017).

### Section 4 — How models learn

#### `VIZ-S4-OVR-1` · status: needed
- **Slide:** Section 4 — How models actually learn (overview)
- **Visual:** Feedback loop: Data → Model → Predictions → Loss → Gradient → Updated Model. Approx 800x300px.
- **Source:** Build with matplotlib (boxes + arrows in a loop). Reference: 3Blue1Brown's "But what is a neural network?" video at ~0:30 has a comparable visual.

#### `VIZ-S4-LOSS-1` · status: needed (interactive embed)
- **Slide:** Loss and learning, conceptually
- **Visual:** **TensorFlow Playground iframe** (or screenshot fallback). Two-spirals dataset, 2 hidden layers, ReLU activation. Audience watches decision boundary form on Run.
- **Source:** Iframe URL (preset configuration): `https://playground.tensorflow.org/#activation=relu&dataset=spiral&regularizationRate=0&learningRate=0.03&noise=0&networkShape=4,2&seed=0.18553&showTestData=false&discretize=false&percTrainData=50&problem=classification`
- **Caveat:** requires internet during delivery; **also capture a screenshot fallback** for offline-safe presentation.

#### `VIZ-S4-LOSS-2` · status: needed
- **Slide:** Loss and learning, conceptually (companion to VIZ-S4-LOSS-1)
- **Visual:** Loss landscape with gradient-descent path. 2D contour plot of a synthetic loss function with a marble's trail walking downhill toward a minimum.
- **Source:** Build with matplotlib (contour + scatter trail). **Alt:** link to 3Blue1Brown "Gradient descent, how neural networks learn" at ≈ 4:30, fair-use still with attribution.

#### `VIZ-S4-GEN-1` · status: needed
- **Slide:** Generalization
- **Visual:** Three loss-curve regimes on one chart: underfit / just-right / overfit. Mark sweet-spot epoch with a dashed vertical line.
- **Source:** Build with matplotlib (3 paired curves on one axes). Reference: any introductory ML overfitting figure (Stanford CS229, Andrew Ng).

#### `VIZ-S4-MODES-1` · status: needed
- **Slide:** Three modes of use today
- **Visual:** Three-column comparison table: Pretrained / Fine-tune / Zero-shot. Rows: icon, description, data needed, skill needed, workshop lab.
- **Source:** Build with matplotlib (3-column comparison) OR author as a markdown table directly in the slide.

### Section 5 — When AI works and when it doesn't

#### `VIZ-S5-OVR-1` · status: needed
- **Slide:** Section 5 — When AI works, and when it doesn't (overview)
- **Visual:** Two-column hero: "Works" (green-tinted, 3-4 success thumbnails) vs "Fails" (magenta-tinted, 3-4 failure thumbnails).
- **Source:** Compose with matplotlib using `task_*` and `failure_*` figures already in `figures/`.

#### `VIZ-S5-WORKS-1` · status: needed
- **Slide:** Where it currently works well
- **Visual:** 5 thumbnail-success cards in a row: cell segmentation, denoised fluorescence, classified phenotypes, stitched lightsheet, μSAM prompted output. Each card with tool name and metric.
- **Source:** Compose with matplotlib using existing task-figure stills + 1-2 new images. Could pull screenshots from BBBC benchmark dashboards (CC BY).

#### `VIZ-S5-FAILS-1` · status: needed
- **Slide:** Where it fails
- **Visual:** 5 thumbnail-failure cards mapping to the bullet list: OOD sample with confidently-wrong mask, rare-category miss, sample-prep drift, edge-effect failure, hallucinated feature in restoration.
- **Source:** **REQUEST FROM USER** for at least 1-2 real Mount Sinai examples. **Alt:** existing `failure_*` figures + 2 new matplotlib mock-ups (rare-category, edge-effect).

#### `VIZ-S5-PATTERNS-1` · status: needed
- **Slide:** Failure-mode patterns to recognize
- **Visual:** Failure-mode taxonomy diagram: 5 named failure modes in radial or 5-row layout, each with 2-3 sub-bullet examples.
- **Source:** Build with matplotlib. Reference: any technical-report failure-mode taxonomy figure.

### Section 6 — Validation

#### `VIZ-S6-LIT-1` · status: needed
- **Slide:** Reading the literature critically
- **Visual:** Two columns: "Red flags" checklist (4 items, red X icons) and "Reporting standards" (CLAIM, STARD-AI, CONSORT-AI, SPIRIT-AI, MI-CLAIM as 5 chips).
- **Source:** Build with matplotlib (two-column text + shapes) OR replace with a markdown table in the slide.

### Section 7 — Reproducibility

#### `VIZ-S7-HABITS-1` · status: needed
- **Slide:** Three habits that make AI-assisted analyses reproducible
- **Visual:** Three icon cards: version-everything, document-decisions, reporting-standards. One-line caption per card.
- **Source:** Build with matplotlib (3-column with simple shapes) OR substitute emoji.

#### `VIZ-S7-INTEG-1` · status: needed
- **Slide:** Image integrity: a class of risk worth knowing
- **Visual:** Side-by-side: raw fluorescence (caption "measured pixels") vs AI-restored version with a magenta-highlighted invented feature (caption "AI restoration — disclose"). Below: example acceptable disclosure text.
- **Source:** Build with matplotlib using a real CARE/Noise2Void example from Notebook 03a or 04. Invented feature should be subtle but present.

#### `VIZ-S7-BIAS-1` · status: needed
- **Slide:** Bias, fairness, and population effects
- **Visual:** 3-panel: Data bias (pie chart with missing slice) / Label bias (two annotators disagreeing) / Deployment bias (performance bar chart by cohort).
- **Source:** Build with matplotlib. Reference: any FAccT or fairness-in-ML overview.

### Section 8 — Closing

#### `VIZ-S8-TAKE-1` · status: needed
- **Slide:** What to take from this lecture
- **Visual:** 5 numbered chips for the durable lessons: AI is a tool, failure modes are predictable, metrics ≠ biology, validation is for trust, integrity reporting is not optional.
- **Source:** Build with matplotlib (5 stacked chips) OR markdown numbered list with brand-color emphasis.

---

## Sourcing requests for the user

**Highest leverage from your time** — items I cannot build from generic sources:

1. **`VIZ-S2-SEG-WORKS-1`** — Mount Sinai canonical fluorescence image where Cellpose-SAM works cleanly.
2. **`VIZ-S2-SEG-FAILS-1`** — Mount Sinai image where Cellpose-SAM is known to fail (tissue, IHC, EM, atypical morphology). Also resolves outstanding task #48.
3. **`VIZ-S5-FAILS-1`** — at least 1-2 real Mount Sinai failure examples to mix with the synthetic ones.
4. **`VIZ-S5-WORKS-1`** — optional: any past-workshop screenshots showing your group's actual workflow, for grounding.
5. **3Blue1Brown timestamps you specifically remember as effective** — better than my general read of the channel.
6. **TensorFlow Playground configurations** — if a particular dataset/architecture lands the lesson better than the spirals default I picked, send the URL state.

---

## Phase plan summary

- **Phase A (this build)** — ✅ done. 25 placeholders in place across all sections; deck renders with visible placeholder boxes.
- **Phase B (next)** — fill the matplotlib-buildable visuals: VIZ-ROADMAP-1, VIZ-S1-OVR-1, VIZ-S1-DRIVERS-1, VIZ-S3-CNN-1, VIZ-S3-GAN-1, VIZ-S3-OTHER-1, VIZ-S4-OVR-1, VIZ-S4-LOSS-2, VIZ-S4-GEN-1, VIZ-S4-MODES-1, VIZ-S5-PATTERNS-1, VIZ-S6-LIT-1, VIZ-S7-HABITS-1, VIZ-S7-BIAS-1, VIZ-S8-TAKE-1. Estimated ~4 hours.
- **Phase C** — fill the iframe + 3b1b-leaning visuals: VIZ-S3-TRANS-1, VIZ-S4-LOSS-1 (TF Playground), VIZ-S2-OVR-1, VIZ-S5-OVR-1. Estimated ~2 hours.
- **Phase D** — fill the user-sourced visuals once Mount Sinai images arrive: VIZ-S2-SEG-WORKS-1, VIZ-S2-SEG-FAILS-1, VIZ-S5-WORKS-1, VIZ-S5-FAILS-1, VIZ-S7-INTEG-1. Estimated ~2 hours after sourcing.

Total to fully fill: ~8-10 hours of authoring across phases B-D.
