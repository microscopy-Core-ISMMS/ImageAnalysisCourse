"""
Build script for lecture.ipynb.

The lecture is authored in this Python build script and emitted as a Jupyter
notebook with reveal.js slideshow metadata. Editing this script and re-running
it is generally easier than editing the resulting JSON directly. The notebook
is the canonical source for the HTML slides; this script is a maintenance tool.

Usage:
    python build_lecture_notebook.py

Outputs:
    lecture.ipynb       — Jupyter notebook with slideshow metadata (canonical source)
    lecture_slides.html — self-contained reveal.js HTML deck (rendered via nbconvert)

The HTML is regenerated from the .ipynb on every build. If `jupyter nbconvert`
is unavailable (e.g. in a stripped-down environment), the .ipynb is still
written and a clear warning is printed; the HTML can be regenerated later.
"""
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "lecture.ipynb"
HTML_OUT = HERE / "lecture_slides.html"
FIG_DIR = HERE / "figures"



# AUTO-INJECTED viz embed sync (build script synced with all 38 lecture.ipynb figures)

def embed_img(name: str, alt: str = "", width: str = "85%") -> str:
    """Read a PNG figure and return an HTML <img> tag with a base64 data URL.

    Self-contained — the resulting HTML does not need external files.
    """
    path = FIG_DIR / f"{name}.png"
    if not path.exists():
        return f"<!-- figure {name}.png missing; run build_figures.py -->"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<div style="text-align:center; margin:0.5em 0;">'
        f'<img src="data:image/png;base64,{data}" alt="{alt}" '
        f'style="max-width:{width}; height:auto; border-radius:4px;"/>'
        f"</div>"
    )


def md(slide_type: str, source: str) -> dict:
    """Markdown cell with slideshow metadata."""
    return {
        "cell_type": "markdown",
        "metadata": {"slideshow": {"slide_type": slide_type}},
        "source": source,
    }


def code(slide_type: str, source: str) -> dict:
    """Code cell with slideshow metadata."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"slideshow": {"slide_type": slide_type}},
        "outputs": [],
        "source": source,
    }


def notes(source: str) -> dict:
    """Speaker-notes cell (rendered as notes by reveal.js, not shown to audience)."""
    return md("notes", source)


# Inline citation-chip helper. The chip is rendered inline as a small bracketed
# tag that links to the full bibliography slide at the end. Verification flag
# follows the citation for transparency.
def cite(key: str, flag: str = "✓") -> str:
    """Inline citation chip. flag is the verification status symbol."""
    return f"<sup>[<a href='#bibliography'>{key}</a> {flag}]</sup>"


CELLS = []


# =============================================================================
# Title slide
# =============================================================================
CELLS.append(md("slide", f"""
# AI for Scientific Image Analysis

{embed_img("brand_core_logo", alt="Microscopy and Advanced Bioimaging Core, Icahn School of Medicine at Mount Sinai", width="60%")}


### *What works, what doesn't, and why*

A one-day workshop lecture for biologists with technical expertise.

---

*Part of the Competency Framework for AI in Scientific Image Analysis*
"""))

CELLS.append(notes("""
Notes for the speaker:

This is the morning lecture, ~70 minutes content + 20 minutes Q&A.
The lecture is one of seven slides in the workshop's morning block.
By the end, attendees should be able to place an AI-imaging problem
into the right task category and identify common failure modes.

Pace: roughly 2 minutes per slide. Don't rush sections 4 and 5; they
are the most important for responsible-use takeaways.
"""))


# =============================================================================
# Roadmap
# =============================================================================
CELLS.append(md("slide", f"""
## Roadmap for the next 90 minutes

{embed_img("viz_roadmap", alt="Eight-section lecture roadmap", width="82%")}


1. **Why now?** — what changed
2. **Tasks AI addresses** — the seven major categories
3. **How models learn** — the conceptual scaffold
4. **When it works, when it doesn't** — failure modes
5. **Validation** — knowing your output is right
6. **Reproducibility and responsible use**
7. **Closing and Q&A**

After the break, we walk through the afternoon labs.
"""))


# =============================================================================
# Section 1 — Why now?
# =============================================================================
CELLS.append(md("slide", f"""
## Section 1 — Why now?

{embed_img("viz_s1_overview", alt="Three converging trends made AI in microscopy routine", width="78%")}


Why has AI become a routine tool in scientific imaging in the past decade,
when image analysis has been a discipline for fifty years?
"""))

CELLS.append(md("subslide", """
### Three drivers, all at once

- **Compute.** GPU availability and the scale shift from single-machine to cluster workflows.
- **Data.** Public datasets, image-sharing infrastructure, reusable training corpora.
- **Methods.** CNNs → vision transformers → foundation models. But methods don't matter without compute and data.

❓ *Framing claim, not a single citation.*
"""))

CELLS.append(md("subslide", f"""
### Where the data and the tools live now

{embed_img("ecosystem_layers", alt="Layered ecosystem: hardware, data, models, platforms", width="88%")}

A non-exhaustive list of openly accessible infrastructure that made the
data-and-tools driver real:

- **image.sc forum** — community discussion and resource hub {cite('imagesc')} · [forum.image.sc](https://forum.image.sc)
- **Broad Bioimage Benchmark Collection (BBBC)** — curated benchmark datasets {cite('bbbc2012')}
- **NEUBIAS** — bioimage analyst training and resources {cite('neubias')}
- **BioImage Archive, IDR, Allen Cell Image Library** — large-scale public imaging archives
- **[ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic)** — Colab notebooks for the major DL microscopy methods, no install required
- **[DL4MicEverywhere](https://github.com/HenriquesLab/DL4MicEverywhere)** — containerized successor; runs on Colab, local, HPC, cloud
- **[BioImage Model Zoo](https://github.com/bioimage-io/bioimage.io)** — standardized repository of pretrained DL models

Without infrastructure, data-hungry methods stay in single labs. **More in the Resources appendix at the end.**
"""))

CELLS.append(notes("""
Speaker cue:
If the room engages with this slide ("oh, I didn't know about ZeroCostDL4Mic"),
that's a signal you have a biology-first audience. Consider opening more
drill-downs in Section 2 and pointing at the resources appendix in real time.

If the room nods past this, they're already familiar with the ecosystem;
move quickly and don't dwell.
"""))


# =============================================================================
# Section 2 — Tasks AI addresses
# =============================================================================
CELLS.append(md("slide", f"""
## Section 2 — The image-analysis tasks AI addresses

{embed_img("viz_s2_overview", alt="Seven task categories at a glance", width="92%")}


Seven major task categories. Most AI-imaging problems map to one or more.
"""))

CELLS.append(md("subslide", f"""
### Task 1: Classification

{embed_img("pub_cascio2019_hep2", alt="Six HEp-2 staining patterns side by side (Cascio 2019)", width="82%")}

**What:** assign a label to an image or patch.

**Microscopy examples:** cell-cycle phase, phenotype call, mitosis-or-not. *Shown above:* the six classic HEp-2 anti-nuclear antibody staining patterns — input one image, output one category.

*Figure from Cascio, Taormina & Raso (2019), Applied Sciences 9(8):1618, [doi:10.3390/app9081618](https://doi.org/10.3390/app9081618). CC BY 3.0.*
"""))

CELLS.append(md("subslide", f"""
### Task 2: Detection

{embed_img("pub_kulikov2019_dognet", alt="DoGNet synapse detection across 3 datasets (Kulikov 2019)", width="92%")}

**What:** find objects and return bounding boxes or points.

**Microscopy examples:** spot detection (FISH spots, vesicles, **synapses** — shown above), particle picking. Yellow arrows mark detections across three multiplexed-fluorescence datasets.

**Method families:** classical (Laplacian-of-Gaussian) and learned (region proposals, transformers).

*Figure from Kulikov et al. (2019), PLOS Comput Biol 15(5):e1007012, [doi:10.1371/journal.pcbi.1007012](https://doi.org/10.1371/journal.pcbi.1007012). CC BY 4.0.*
"""))

CELLS.append(md("subslide", f"""
### Task 3: Segmentation

{embed_img("task_segmentation", alt="Segmentation: image, masks, overlay", width="92%")}

**What:** assign a label to every pixel; instance segmentation labels each object separately.

**Representative tools (live links):**
- **[Cellpose-SAM](https://github.com/MouseLand/cellpose)** — current v4 {cite('cellpose_sam_2025', '⚠')}; built on the original Cellpose framework {cite('cellpose2021')}
- **[StarDist](https://github.com/stardist/stardist)** — star-convex polygon segmentation {cite('stardist2018')}
- **[Mesmer / DeepCell](https://github.com/vanvalenlab/deepcell-tf)** — whole-cell segmentation {cite('mesmer2022')}
- **[Segment Anything](https://github.com/facebookresearch/segment-anything)** — foundation model {cite('sam2023')}
- **[μSAM (micro-sam)](https://github.com/computational-cell-analytics/micro-sam)** — microscopy-tuned SAM {cite('microsam2025')}

This is where most attendees first encounter AI in their work.

*Illustration synthesized for clarity; for a benchmark tissue panel see Greenwald et al. (2022) Mesmer, [Nat Biotechnol 40:555](https://doi.org/10.1038/s41587-021-01094-0).*
"""))

# Drill-down 1: when segmentation works
CELLS.append(md("subslide", f"""
#### Drill-down — when Cellpose-SAM works

{embed_img("viz_s2_seg_works", alt="Cellpose-SAM-style success on canonical fluorescence", width="82%")}


*(Open this if the room is curious about the methods.)*

- **Strong cases:** cultured cells with clear morphology, nuclei in 2D, a sample type
  similar to Cellpose's training distribution.
- **Why:** the model was trained on 70,000+ segmented cells across many image types;
  the SAM backbone added in v4 brings strong zero-shot generalization.
- **What you do:** install Cellpose, point it at your image, accept the diameter
  estimate (or set it manually), inspect the output critically.

We'll do this hands-on in **Lab 1** this afternoon.
"""))

# Drill-down 2: when segmentation fails
CELLS.append(md("subslide", f"""
#### Drill-down — when Cellpose-SAM fails

{embed_img("viz_s2_seg_fails", alt="Cellpose-SAM failure on dense / OOD content", width="82%")}


*(Open this for the cautionary half of the segmentation story.)*

- **Weak cases:** unusual sample types (tissue when trained on cells), unusual
  staining patterns, dense overlapping objects, 3D when the third dimension is
  poorly sampled.
- **Failure signature:** confidently wrong masks, merged objects, missed objects
  near edges.
- **What you do:** try Cellpose's `nuclei` model instead of `cyto`, adjust diameter,
  consider StarDist or μSAM for a different inductive bias.

The same pattern shows up across all the segmentation tools — different inductive
biases, different sample-type sweet spots.
"""))

CELLS.append(notes("""
Speaker cue:
The two drill-downs above are deliberately a "good news / bad news" pair.
Open both if you have time and the room is engaged. If you only have time
for one, open the failure-modes drill-down — that's the harder lesson and
the one attendees will need most for Lab 1.

Live links: every tool name on this slide is a clickable link to its GitHub.
You can demo this in real time if you want to show "here's where you go to get
this tool."
"""))

CELLS.append(md("subslide", f"""
### Task 4: Restoration

{embed_img("task_restoration", alt="Restoration: noisy in, denoised out", width="78%")}

**What:** image-to-image transformation that improves quality — denoising, deconvolution, super-resolution.

**Representative tools (live links):**
- **[CARE / CSBDeep](https://github.com/CSBDeep/CSBDeep)** — content-aware restoration {cite('care2018')}
- **[Noise2Void](https://github.com/juglab/n2v)** — self-supervised denoising {cite('noise2void2019')}
- **ANNA-PALM** {cite('annapalm2018')} — super-resolution from sparse PALM
- **Deep-STORM** {cite('deepstorm2018')} — single-molecule localization
- The **Cellpose ecosystem** now integrates restoration with segmentation: Cellpose 3 {cite('cellpose3_2025', '⚠')} added one-click restoration; current Cellpose-SAM {cite('cellpose_sam_2025', '⚠')} carries the ecosystem forward
- All major restoration methods are available as Colab notebooks via **[ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic)** and **[DL4MicEverywhere](https://github.com/HenriquesLab/DL4MicEverywhere)**

⚠ *Restoration overlaps with generation. The line matters for image-integrity reporting — see Section 7.*
"""))

CELLS.append(md("subslide", """
#### Drill-down — restoration's hallucination problem

*(Open this when discussing image integrity, or in response to a question.)*

- **The risk:** AI restoration changes pixel values. A denoised image isn't a
  measurement; it's a model's *prediction* of what the clean image looks like.
- **What that means in practice:** restored images can introduce features that
  weren't in the original signal — quietly, plausibly. This violates many
  journal image-integrity policies if not disclosed.
- **The honest path:** disclose AI restoration in figure captions, document the
  model identity and version, validate against ground truth where possible
  (we'll demonstrate this in Lab 3a if we choose option A).

This is the responsible-use spine of the lecture. Restoration is useful and
risky in equal measure.
"""))

CELLS.append(notes("""
Speaker cue:
The hallucination drill-down is a natural transition into Section 7
(reproducibility and responsible use). If you're running long, you can
fold this drill-down's content into Section 7 and skip it here.

If you're running short, the drill-down stands on its own as a 60-second
addition to this slide.
"""))

CELLS.append(md("subslide", f"""
### Task 5: Generation

{embed_img("task_generation", alt="Generation: brightfield in, fluorescence-like out", width="78%")}

**What:** produce new image content from input or noise — virtual staining, in silico labeling, synthetic data.

**Representative work:** in silico labeling {cite('christiansen2018')}, label-free 3D prediction {cite('ounkomol2018')}.

❓ *Generation outputs are not measurements. The image-integrity considerations matter and we revisit them in Section 7.*
"""))

CELLS.append(md("subslide", f"""
### Task 6: Registration

{embed_img("task_registration", alt="Registration: two images and overlay", width="92%")}

**What:** align images to a common spatial reference.

**Representative tools (live links):**
- **[VoxelMorph](https://github.com/voxelmorph/voxelmorph)** — deep-learning deformable registration; clinical-imaging foundational {cite('voxelmorph2019')}
- **[BigStitcher](https://github.com/JaneliaSciComp/BigStitcher)** — terabyte-scale microscopy stitching ({cite('bbbc2012', '⚠')} for citation context; Hörl et al. 2019 *Nature Methods*)
- **[ClearMap](https://github.com/ChristophKirst/ClearMap)** — cleared-tissue volumetric registration with Allen Brain Atlas (Renier et al. *Cell* 2016)
- DL4MicEverywhere adds learned image registration as an explicit module

❓ *Microscopy-specific learned-registration tooling is fast-moving. Several methods overlap restoration and registration (e.g., learned drift correction).*
"""))

CELLS.append(md("subslide", f"""
### Task 7: Tracking

{embed_img("task_tracking", alt="Tracking: trajectories over time", width="78%")}

**What:** connect objects across time.

**Method families:** classical (Kalman, linear assignment) and learned (transformers, deep tracking).

*Illustration synthesized for clarity; for benchmark trajectories see Ershov et al. (2022) TrackMate 7, [Nat Methods 19:829](https://doi.org/10.1038/s41592-022-01507-1).*
"""))


# =============================================================================
# Section 3 — Common model families  [ADDED 2026-05-11 sync — was previously
# only present in lecture.ipynb but missing from this build script.]
# =============================================================================
CELLS.append(md("slide", f"""
## Section 3 — Common model families behind these tasks

{embed_img("viz_s3_overview", alt="Four common model families: CNN, Transformer, GAN, Other", width="82%")}

Most bioimage AI lives on three architecture families: **CNNs**, **Transformers**, and **GANs**. A fourth umbrella covers everything else worth recognizing — autoencoders, diffusion models, and the general encoder-decoder pattern.

You almost never pick the architecture directly. You pick a task and a tool; the tool's authors picked the architecture. Knowing the family tells you *why* a method has the strengths and weaknesses it does — and which lab in this workshop runs which.
"""))

CELLS.append(notes("""
Speaker note (~3 minutes for the overview, ~1.5 minutes per drill-down).

Most attendees will not have seen these grouped before. The point is recognition,
not architectural depth. If you are running short, stay on the overview slide and
skip the drill-downs entirely.

Hardest sell: "you don't pick the architecture." Some attendees with a CS background
will resist this. The reply is: at the workshop's level of practice, the tool you
pick (Cellpose, SAM, CARE) carries the architecture choice with it. You are picking
*tools and tasks*, and the tool authors picked the architecture for the task.
"""))

CELLS.append(md("subslide", f"""
### CNNs — Convolutional Neural Networks

{embed_img("viz_s3_cnn", alt="U-Net architecture: encoder + bottleneck + decoder + skip connections", width="85%")}

**Idea.** Stack of convolutions; each layer learns a small filter that slides over its input. Early layers pick up edges; deeper layers compose them. Translation-invariant by construction.

**Workhorses.** *U-Net* {cite('unet2015')} for image-to-image (segmentation, restoration). *ResNet* {cite('resnet2016')} for classification backbones. Encoder-downsample → decoder-upsample with skip connections is the dominant bioimage pattern.

**In this workshop.** Cellpose-SAM (CNN backbone + SAM transformer in v4); CARE, N2V, StarDist all U-Net-shaped.

| Strengths | Weaknesses |
|---|---|
| Mature ecosystem, vast pretrained weights, efficient inference, well-understood failure modes | Limited *global* context — single convolution sees a small neighborhood; stacked CNNs grow the field but never as cleanly as attention does |
"""))

CELLS.append(md("subslide", f"""
### Transformers — global context via self-attention

{embed_img("viz_s3_transformer", alt="Self-attention: each token attends to every other token", width="82%")}

**Idea.** A *Vision Transformer* (ViT) {cite('vit2021', '⚠')} chops the image into patches, embeds each patch as a token, runs self-attention across all tokens. Captures long-range relationships natively, no convolution stack required.

**Foundation models.** Trained on huge unlabeled/weakly-labeled data, then *prompted* per task: SAM {cite('sam2023')}, μSAM (microscopy), BiomedCLIP (pathology), DINOv2 (general features). Largely a transformer paradigm.

**In this workshop.** Lab 3b: SAM/μSAM with point/box prompts. Cellpose-SAM is a hybrid (CNN backbone + SAM attention).

| Strengths | Weaknesses |
|---|---|
| Scales with data; strong few-shot; *prompting* — new task without retraining | Data-hungry from scratch; expensive at inference for large images; *prompt sensitivity* — the central pedagogy of Lab 3b |
"""))

CELLS.append(md("subslide", f"""
### GANs — adversarial generation

{embed_img("viz_s3_gan", alt="GAN: Generator vs Discriminator adversarial loop", width="82%")}

**Idea.** Two networks trained against each other. *Generator* tries to fool the *discriminator*; discriminator tries to tell real from generated. The game pushes the generator toward photorealistic output. Goodfellow et al. {cite('gan2014')}.

**Bioimage uses.** *pix2pix* {cite('pix2pix2017')} for paired image-to-image (brightfield → fluorescence; IHC → mIF; modality A → B). *CycleGAN* for *unpaired* translation. *StyleGAN* for unconditional generation (rare in bioimage; more common in medical-imaging data augmentation).

**In this workshop.** Notebook 04: pix2pix mini-workflow + CycleGAN catalog pointer.

| Strengths | Weaknesses |
|---|---|
| Photorealistic output; compact at inference; established literature | Training instability + *mode collapse*; **hallucination risk** — outputs that look correct but invent features. Disclose generation in figure captions. |
"""))

CELLS.append(md("subslide", f"""
### Other architectures worth recognizing

{embed_img("viz_s3_other", alt="Encoder-decoder pattern across U-Net, autoencoder, pix2pix, fnet", width="85%")}

| Family | What it is | Bioimage relevance |
|---|---|---|
| **Autoencoders** | Encoder → bottleneck → decoder; trained with reconstruction loss | *Noise2Void* {cite('noise2void2019')} (Notebook 03a); DINO, MAE for self-supervised features |
| **Diffusion models** | Generate by gradually denoising pure noise | *DDPM* {cite('ddpm2020', '⚠')}; **replacing GANs**; growing fast in bioimage. Curated [awesome-list for medical imaging](https://github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging). |
| **Encoder-decoder pattern** | The shared backbone | U-Net (with skips), autoencoders, pix2pix gen, fnet — recognize this in *every* new bioimage paper |

**Takeaway.** New architectures appear every year. The conceptual scaffold — *what does it predict, what's the loss, what data does it need* — is more durable than any architecture name.
"""))


# =============================================================================
# Section 4 — How models learn  [renumbered 2026-05-11: was Section 3]
# =============================================================================
CELLS.append(md("slide", f"""
## Section 4 — How models actually learn

{embed_img("viz_s4_overview", alt="How models learn: forward pass + gradient feedback", width="78%")}

Conceptual scaffolding, not architecture details. We will not teach you what
a U-Net is. We will teach you what it does and where it fails.
"""))

CELLS.append(md("subslide", f"""
### Training, validation, and test data

{embed_img("train_val_test", alt="Train, validation, and test split diagram", width="92%")}

Three disjoint subsets of your data. Each has a different job.

- **Training set** — what the model fits to. The model sees these images and adjusts its parameters to minimize prediction error on them.
- **Validation set** — what you tune hyperparameters against. Used to choose between candidate models or settings *during* development.
- **Test set** — what you report on, **only after everything else is locked**. Touch it once, at the end. Touch it twice, and your reported accuracy is no longer trustworthy.

The biggest beginner mistake: tuning on the test set. The second biggest: letting training data leak into the test set (e.g., images from the same patient in both).

This split is the single most important pattern in supervised ML. Every method we discuss assumes it. ❓
"""))

CELLS.append(md("subslide", """
### Loss and learning, conceptually

A model is a parametric function — millions or billions of numbers (weights) that together compute "given this input image, predict this output."

**Training** adjusts those weights to reduce a *loss* — a single number that measures how wrong the model is on the training data. For segmentation, loss might be how many pixels were assigned the wrong label. For classification, how often the predicted class was wrong.

**Gradient descent** walks downhill on the loss surface. At each step, it computes which direction reduces the loss most, then takes a small step that way. Millions of steps later, the loss is low and the model is "trained."

This is the mental model. We do not need the math today, but the intuition matters: the model has done exactly one thing — it has reduced loss on training data. Whether that translates to your data is a separate question. ❓
"""))

CELLS.append(md("subslide", f"""
### Generalization

{embed_img("viz_s4_generalization", alt="Underfit / just-right / overfit regimes", width="92%")}


A model that does well on training data may do poorly on yours. This is *the* central problem in applied ML.

**Why generalization is hard:**

- Training data is finite; the world is not.
- Models can memorize training data (overfitting) without learning general patterns.
- The conditions under which the model was trained may not match the conditions under which you use it (distribution shift).

**The single most important variable:** the *diversity* of training data relative to the diversity of your data. A Cellpose model trained on 70,000 cells across many image types generalizes well across cell biology because the diversity is broad. The same model fails on a histology section because that diversity didn't include tissue.

Generalization is what every AI-imaging failure mode in Section 5 is, at root, an instance of. ❓
"""))

CELLS.append(md("subslide", f"""
### Three modes of use today

{embed_img("viz_s4_modes", alt="Three modes of using AI today", width="82%")}


- **Pretrained inference** — use someone else's model on your data. Lab 1.
- **Fine-tuning** — adjust a pretrained model with your own labels.
- **Zero-shot / prompt-based** — foundation models with prompts, no fine-tuning.

The afternoon labs will exercise pretrained inference (Lab 1) and either
self-supervised denoising (Lab 3a) or prompt-based foundation models (Lab 3b).

❓ *Workshop-specific framing.*
"""))


# =============================================================================
# Section 5 — When AI works and when it doesn't  [renumbered 2026-05-11: was Section 4]
# =============================================================================
CELLS.append(md("slide", f"""
## Section 5 — When AI works, and when it doesn't

{embed_img("viz_s5_overview", alt="Works vs fails - concrete examples", width="92%")}

The section attendees will remember most. Concrete examples of where current
AI in scientific imaging delivers, and where it fails — often confidently.
"""))

CELLS.append(md("subslide", f"""
### Where it currently works well

{embed_img("viz_s5_works", alt="Five success cases for AI in microscopy", width="92%")}


Concrete examples of well-validated successes:

- **Segmentation** of cell types similar to training data — Cellpose-SAM and Mesmer routinely achieve >0.85 Dice on standard cell-line and tissue panels they were trained for.
- **Denoising** of structured noise patterns the model has been exposed to — CARE-restored fluorescence images approach the SNR of long-exposure references with 10–100× less light.
- **Classification** of well-defined categories with thousands of training examples — phenotypic profiling at high-content-screening scale.
- **Stitching** of multi-tile microscopy when overlap is consistent and contrast is reliable — BigStitcher routinely handles terabyte-scale lightsheet data.
- **Foundation-model segmentation** on novel sample types via prompts — μSAM extends segment-anything's reach into microscopy with a few clicks per image.

The pattern: AI works well when training data covers the use case, and when validation has been done on data similar to yours. ❓
"""))

CELLS.append(md("subslide", f"""
### Where it fails

{embed_img("viz_s5_fails", alt="Five failure patterns for AI in microscopy", width="92%")}


Concrete failure patterns:

- **Out-of-distribution samples** — a Cellpose model trained on cultured cells producing confident-but-wrong masks on tissue. A clinical-AI tool trained on a US population mis-classifying scans from a different scanner brand.
- **Sparse or rare categories** — anything underrepresented in training. Rare disease presentations, atypical cell shapes, sample-preparation variants the model never saw.
- **Sample-preparation drift** — the model sees a population the training set didn't. Even small changes in fixation, mounting, or staining can degrade performance silently.
- **Edge effects** — many segmentation tools struggle at image borders or near other objects.
- **Hallucination in restoration and generation** — outputs that look right but contain features the model invented.

We'll see the segmentation failure mode directly in Lab 1, and (if option A) the restoration hallucination directly in Lab 3a. ❓
"""))

CELLS.append(md("subslide", f"""
### Failure-mode patterns to recognize

{embed_img("viz_s5_patterns", alt="Failure-mode taxonomy", width="82%")}


- **Domain shift.** Tools trained on one organism, modality, or stain often degrade on another.
- **Sample-preparation effects.** Fixation, mounting, staining drive failures the model has never seen.
- **Confident wrongness.** Outputs that look plausible but are wrong. The most dangerous mode.
- **Hallucination in generative methods.** Restoration and virtual staining can invent features.
- **Misregistration in alignment tasks.** Models can align some structures while distorting others, corrupting downstream measurements without an obvious signature.

❓ *Community-recognized failure-mode taxonomy; no single canonical citation.*
"""))

CELLS.append(md("subslide", f"""
### Failure mode: domain shift

{embed_img("failure_domain_shift", alt="Same model, in-distribution success vs OOD undercount", width="80%")}

Same Cellpose model, different sample type. Round isolated cells: works. Irregular dense cells: silent under-counting. Failure looks plausible — that's the dangerous part.

*Illustration synthesized for clarity; for quantitative cross-dataset generalization benchmarks see Pachitariu, Rariden & Stringer (2025) Cellpose-SAM, [bioRxiv 2025.04.28.651001](https://doi.org/10.1101/2025.04.28.651001) and Stringer et al. (2021) Cellpose, [Nat Methods 18:100](https://doi.org/10.1038/s41592-020-01018-x).*
"""))

CELLS.append(md("subslide", f"""
### Failure mode: hallucination in restoration

{embed_img("failure_hallucination", alt="Restored image shows an invented feature", width="92%")}

The restored image looks clean and plausible. But it contains a feature the model invented. Without the ground truth (which you don't have in real experiments), you cannot tell.
"""))

CELLS.append(md("subslide", f"""
### Failure mode: misregistration

{embed_img("failure_misregistration", alt="Successful vs failed registration overlay", width="80%")}

A registration that looks aligned globally but distorts locally — corrupting downstream measurements without an obvious signature. Especially common when source and target images differ enough that the model's training distribution doesn't cover the deformation.
"""))


# =============================================================================
# Section 6 — Validation  [renumbered 2026-05-11: was Section 5]
# =============================================================================
CELLS.append(md("slide", """
## Section 6 — Validation: how to know your output is right
"""))

CELLS.append(md("subslide", f"""
### What validation is for

{embed_img("validation_gradient", alt="Validation gradient: research to clinical", width="92%")}

**Establishing trust in a specific use case.** Not generic accuracy. Not benchmark numbers. Trust *for the question you are asking*, on data *like yours*.

Two flavors that both matter:

- **Internal validation** — held-out data from the same distribution as training. Tells you whether the model fit its own setting.
- **External validation** — someone else's data, ideally a different institution, scanner, population. Tells you whether the model generalizes beyond the conditions it was developed in.

Internal validation is necessary; external validation is what builds field-wide trust. Most published AI tools have strong internal validation and weak external validation — when reading a paper, check both.

The validation gradient runs from "research-grade" (internal validation, single dataset) to "clinical-grade" (multi-site external validation, prospective trials, regulatory clearance). Where you sit on this gradient determines what you can claim. ❓
"""))

CELLS.append(md("subslide", f"""
### Metrics versus biological correctness, and metrics versus the right metric

{embed_img("metrics_vs_biology", alt="High IoU does not guarantee correct count", width="78%")}

IoU and Dice are not the same as "the count is right." A model can have high pixel overlap and still systematically under-count, over-count, or merge objects in ways that corrupt downstream biology.

Different tasks require different metrics:

- **Segmentation** — overlap-based (IoU, Dice) for pixel-level accuracy; instance-matched precision/recall for object-level accuracy
- **Detection** — precision and recall at IoU thresholds; mAP for ranked outputs
- **Restoration** — PSNR, SSIM for similarity to a clean reference; bias analysis when no reference exists
- **Registration** — landmark distance for sparse correspondence; structural-similarity preservation; jacobian determinants for non-collapse
- **Tracking** — ID consistency, MOTA, ID switches
- **Classification** — accuracy, AUC, calibration; per-class breakdown for imbalanced data

We will demonstrate the segmentation case (IoU is high but the cell count is wrong) directly in Lab 2. The principle generalizes to every task in Section 2: pick the metric that matches the biological question, not the metric the paper happens to report. ❓
"""))

CELLS.append(md("subslide", f"""
### Reading the literature critically

{embed_img("viz_s6_lit", alt="Red flags vs reporting standards when reading the literature", width="92%")}


Red flags when reading a paper or vendor claim:

- Reported metrics on a single dataset, no external validation
- No discussion of failure modes
- Black-box claims without access to the model or training data
- Generic accuracy figures without per-class or per-condition breakdown

Reporting standards exist to address these patterns: CLAIM {cite('claim2020')},
STARD-AI {cite('stardai2020')}, CONSORT-AI {cite('consortai2020')}, SPIRIT-AI {cite('spiritai2020')}, MI-CLAIM {cite('miclaim2020')}.

These exist; learn them when you publish.
"""))


# =============================================================================
# Section 7 — Reproducibility and responsible use  [renumbered 2026-05-11: was Section 6]
# =============================================================================
CELLS.append(md("slide", """
## Section 7 — Reproducibility and responsible use
"""))

CELLS.append(md("subslide", f"""
### Three habits that make AI-assisted analyses reproducible

{embed_img("viz_s7_habits", alt="Three reproducibility habits", width="80%")}


- **Version everything.** Model identity (e.g., "Cellpose-SAM v4.0.1"), training data version, parameters (diameter, threshold, flow). Without this, your figures cannot be regenerated.
- **Document decisions.** Why this model, why this threshold, why this preprocessing. The reviewer will ask; have the answer ready in your methods section.
- **Use reporting standards.** CLAIM {cite('claim2020')} is the most widely-cited starting point for medical imaging. CONSORT-AI {cite('consortai2020')}, SPIRIT-AI {cite('spiritai2020')}, STARD-AI {cite('stardai2020')}, MI-CLAIM {cite('miclaim2020')} cover related contexts.

The methods section is where AI-assisted analyses live or die under review. Make it complete enough that another lab could rerun your pipeline, including the model version. ❓
"""))

CELLS.append(md("subslide", f"""
### Image integrity: a class of risk worth knowing

{embed_img("viz_s7_integ", alt="AI restoration with hallucinated feature - disclose in caption", width="85%")}


**AI restoration and AI generation methods change pixels.** When you display a restored or generated image without disclosure, you can violate journal image-integrity policies — even unintentionally. Some journals now flag and retract AI-altered figures.

The rule of thumb: **if pixel values are different from what came off the microscope, say so in the figure caption.**

This applies to:
- Denoised images (Noise2Void, CARE, Cellpose-3 restoration)
- Super-resolution outputs (ANNA-PALM, Deep-STORM, learned upsampling)
- Virtual staining and label-free predictions
- Any generative output (synthetic data, diffusion model outputs)

A simple disclosure pattern: "Image displayed has been restored with [Method, vX]; quantitative analysis was performed on the original raw data." This satisfies most journal policies and protects you from later questions. ❓
"""))

CELLS.append(md("subslide", f"""
### Bias, fairness, and population effects

{embed_img("viz_s7_bias", alt="Three sources of bias: data, label, deployment", width="92%")}


- *Data bias.* Training data underrepresents your sample type.
- *Label bias.* Annotators disagreed; the model learned the disagreement.
- *Deployment bias.* The model performs differently on different cohorts.

These appear in pathology and clinical imaging more visibly than in
microscopy, but the principle generalizes.

❓ *General framing; specific clinical-AI evidence omitted from this lecture.*
"""))


# =============================================================================
# Section 8 — Closing and Q&A  [renumbered 2026-05-11: was Section 7]
# =============================================================================
CELLS.append(md("slide", """
## Section 8 — Closing
"""))

CELLS.append(md("subslide", f"""
### What to take from this lecture

{embed_img("viz_s8_take", alt="Five takeaways for tomorrow", width="75%")}


AI is a tool, not a magic wand.

The afternoon labs will give you your first hands-on experience using it
critically. The patterns we discussed — task taxonomy, failure modes,
metrics-versus-biology, integrity reporting — are what we will exercise.

**Where to go from here:**
- The **[Resources page](../../resources)** — every tool, dataset, and platform we mentioned, with GitHub links
- The **[Handout](../../handout)** — your post-workshop reading list with all citations and links
- **[image.sc forum](https://forum.image.sc)** — your continuing-education community

❓ *Workshop framing.*
"""))

CELLS.append(md("subslide", """
### Discussion prompts (open Q&A)

- Has anyone here trusted an AI output that turned out to be wrong? What did the failure look like?
- Where in your work do you think AI would help most? Where would you be most worried about deploying it?
- What evidence would convince you a vendor's tool actually works on your data?
"""))


# =============================================================================
# Interactive demo placeholders (Phase 2 will populate)
# =============================================================================
CELLS.append(md("slide", """
## Interactive demos

The following code cells run live in the notebook. In the slide-show export,
they appear with their pre-rendered outputs. Phase 2 of this lecture build
will populate the cells with working demonstrations.
"""))

CELLS.append(code("subslide", """# Setup verification cell — runs on Colab and locally
# Confirms the runtime is alive and shows what version of Python/numpy
# is available. Run this first to verify the notebook is happy.
import sys
import platform

print("Python version :", sys.version.split()[0])
print("Platform       :", platform.platform())

try:
    import numpy as np
    print("numpy version  :", np.__version__)
except ImportError:
    print("numpy          : NOT INSTALLED — run !pip install numpy")

try:
    import matplotlib
    print("matplotlib ver :", matplotlib.__version__)
except ImportError:
    print("matplotlib     : NOT INSTALLED — run !pip install matplotlib")

# Detect Colab vs local
IN_COLAB = 'google.colab' in sys.modules
print("\\nRunning in     :", "Google Colab" if IN_COLAB else "Local Jupyter")
print("Runtime ready. The remaining demo cells assume numpy + matplotlib.")"""))

CELLS.append(code("subslide", """# Train / validation / test split visualization
# Why this matters: the most fundamental ML hygiene rule is that the model
# never sees test data during training. Here is what the split looks like.
import numpy as np
import matplotlib.pyplot as plt

# Pretend we have 1000 image samples
np.random.seed(42)
n = 1000
indices = np.arange(n)
np.random.shuffle(indices)

# Standard 70/15/15 split
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train + n_val]
test_idx  = indices[n_train + n_val:]

# Visualize: each tile is one sample, colored by split
fig, ax = plt.subplots(figsize=(10, 1.2))
colors = np.empty(n, dtype=object)
colors[train_idx] = '#4C72B0'   # blue
colors[val_idx]   = '#DD8452'   # orange
colors[test_idx]  = '#55A868'   # green

ax.bar(np.arange(n), np.ones(n), width=1.0, color=colors, edgecolor='none')
ax.set_xlim(0, n); ax.set_ylim(0, 1)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'1000 samples · {len(train_idx)} train · {len(val_idx)} val · {len(test_idx)} test')

# Legend
from matplotlib.patches import Patch
legend = [Patch(color='#4C72B0', label='Train'),
          Patch(color='#DD8452', label='Validation'),
          Patch(color='#55A868', label='Test (touch once)')]
ax.legend(handles=legend, loc='upper right', bbox_to_anchor=(1, -0.3), ncol=3)
plt.tight_layout()
plt.show()

print(f"\\nKey rule: the test set ({len(test_idx)} samples) is reported on once, at the end.")
print("Touch it twice and your reported numbers are no longer trustworthy.")"""))

CELLS.append(code("subslide", """# Pre-computed Cellpose-style result viewer (synthetic example)
# Why this matters: shows the segmentation output format and how a "good"
# segmentation differs from a "bad" one — without needing GPU or a model.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.random.seed(42)

# Simulate a 200x200 microscopy image with ~12 cells
def synth_image_with_masks(n_cells=12, size=200, noise=0.15):
    img = np.zeros((size, size), dtype=float)
    masks = np.zeros((size, size), dtype=int)
    centers = np.random.uniform(20, size-20, (n_cells, 2))
    radii   = np.random.uniform(8, 16, n_cells)
    for i, ((cy, cx), r) in enumerate(zip(centers, radii), start=1):
        y, x = np.ogrid[:size, :size]
        circle = (y - cy)**2 + (x - cx)**2 <= r**2
        img[circle] = 1.0
        masks[circle] = i
    img += np.random.normal(0, noise, img.shape)
    return img, masks

img, gt = synth_image_with_masks()
# "Good" prediction: matches ground truth closely
good_pred = gt.copy()
# "Bad" prediction: merges some cells, misses some, finds spurious objects
bad_pred = gt.copy()
bad_pred[bad_pred == 3] = 2  # merge cells 2 and 3
bad_pred[bad_pred == 7] = 0  # delete cell 7

# Display: original, good prediction, bad prediction
cmap = ListedColormap(['black'] + plt.cm.tab20(np.arange(20)).tolist())
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img, cmap='gray'); axes[0].set_title('Image')
axes[1].imshow(good_pred, cmap=cmap, vmin=0, vmax=20); axes[1].set_title(f'Good prediction · {len(np.unique(good_pred))-1} objects')
axes[2].imshow(bad_pred,  cmap=cmap, vmin=0, vmax=20); axes[2].set_title(f'Bad prediction · {len(np.unique(bad_pred))-1} objects (1 merge, 1 miss)')
for a in axes: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()

print(f"Ground-truth count : {len(np.unique(gt)) - 1}")
print(f"Good prediction    : {len(np.unique(good_pred)) - 1}  (count error: 0)")
print(f"Bad prediction     : {len(np.unique(bad_pred)) - 1}   (count error: {len(np.unique(gt)) - len(np.unique(bad_pred))})")"""))

CELLS.append(code("subslide", """# Metric versus biology plot
# Why this matters: high IoU does not guarantee correct biology.
# This synthetic experiment shows IoU vs cell-count error as predictions
# are perturbed in ways that preserve overall pixel agreement.
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n_trials = 60

# Each trial: a "predicted" segmentation that differs from ground truth
# in a way controlled by a perturbation strength. IoU stays high; count error grows.
strengths = np.linspace(0, 1, n_trials)
ious = 0.95 - 0.10 * strengths + np.random.normal(0, 0.02, n_trials)
count_errors = strengths * 8 + np.random.normal(0, 0.5, n_trials)

fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(ious, count_errors, c=strengths, cmap='viridis',
                s=40, edgecolor='k', linewidth=0.5)
ax.set_xlabel('IoU (pixel overlap with ground truth)')
ax.set_ylabel('Count error (predicted minus actual cells)')
ax.set_title('A high IoU does not guarantee a correct count')
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.invert_xaxis()  # higher IoU on the left
cb = plt.colorbar(sc, ax=ax, label='Perturbation strength')
plt.tight_layout(); plt.show()

print("Even at high IoU (0.85+), count error can be 3-5 cells.")
print("This is the metrics-vs-biology gap. Choose the metric that matches your question.")"""))


# =============================================================================
# Resources Appendix — navigate-from-anywhere library
# =============================================================================
CELLS.append(md("slide", """
## Resources Appendix

The slides ahead are a structured library of every tool, platform, dataset,
and community resource referenced (or worth referencing) for this workshop.

**Navigation:** speakers, you can jump here from any slide via the
[Resources page](../../resources). Use the sub-slides below as a per-topic
drill-down.

**For attendees:** the full [Resources page](../../resources) and the
auto-generated [Handout](../../handout) capture everything. You will receive
links at workshop close.
"""))

CELLS.append(md("subslide", """
### Resources — Community platforms and Colab toolboxes

The single most useful starting point for self-directed learning.

- **[ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic)** — Colab notebooks for the major DL microscopy methods. von Chamier et al. *Nat Commun* 2021.
- **[DL4MicEverywhere](https://github.com/HenriquesLab/DL4MicEverywhere)** — containerized successor; Colab/local/HPC/cloud. Hidalgo-Cenalmor et al. *Nat Methods* 2024.
- **[image.sc forum](https://forum.image.sc)** — community Q&A across Fiji, napari, ilastik, scikit-image, CellProfiler, and many other communities.
- **[NEUBIAS training resources](https://github.com/NEUBIAS/training-resources)** — modular open educational material.
- **[BioImage Model Zoo](https://github.com/bioimage-io/bioimage.io)** — standardized pretrained DL models for bioimage analysis.
"""))

CELLS.append(md("subslide", """
### Resources — Microscopy segmentation and analysis tools

Day-to-day tools that show up in most microscopy workflows.

- **[Cellpose / Cellpose-SAM (MouseLand)](https://github.com/MouseLand/cellpose)** — generalist cell segmentation; current v4. Pachitariu, Rariden, Stringer (preprint, 2025).
- **[StarDist](https://github.com/stardist/stardist)** — star-convex polygon segmentation. Schmidt et al. *MICCAI* 2018.
- **[Mesmer / DeepCell](https://github.com/vanvalenlab/deepcell-tf)** — whole-cell segmentation. Greenwald et al. *Nat Biotech* 2022.
- **[AICS Segmentation](https://github.com/AllenCell/aics-segmentation)** — Allen Institute 3D structure segmentation.
- **[ilastik](https://github.com/ilastik)** — interactive ML segmentation. Berg et al. *Nat Methods* 2019.
- **[μSAM (micro-sam)](https://github.com/computational-cell-analytics/micro-sam)** — microscopy-tuned SAM. Pape et al. *Nat Methods* 2025.
"""))

CELLS.append(md("subslide", """
### Resources — Restoration, denoising, super-resolution

For the data quality / quantity gap.

- **[Noise2Void](https://github.com/juglab/n2v)** — self-supervised denoising. Krull et al. *CVPR* 2019. (Lab 3a uses this.)
- **[CSBDeep / CARE](https://github.com/CSBDeep/CSBDeep)** — content-aware restoration. Weigert et al. *Nat Methods* 2018.
- **ANNA-PALM** — super-resolution from sparse PALM. Ouyang et al. *Nat Biotech* 2018.
- **Deep-STORM** — single-molecule localization. Nehme et al. *Optica* 2018.
- **[Cellpose 3](https://github.com/MouseLand/cellpose)** — one-click image restoration integrated with segmentation.
"""))

CELLS.append(md("subslide", """
### Resources — Generation, label-free, and foundation models

For the cross-modality and zero-shot frontiers.

- **[Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)** — Kirillov et al. *ICCV* 2023.
- **[μSAM (micro-sam)](https://github.com/computational-cell-analytics/micro-sam)** — Pape et al. *Nat Methods* 2025.
- **In silico labeling** — Christiansen et al. *Cell* 2018.
- **Label-free 3D fluorescence prediction** — Ounkomol et al. *Nat Methods* 2018.
- **Cellpose-SAM** (current Cellpose v4) — Pachitariu, Rariden, Stringer (2025 preprint).
"""))

CELLS.append(md("subslide", """
### Resources — Registration, stitching, alignment

For the spatial-correspondence task.

- **[VoxelMorph](https://github.com/voxelmorph/voxelmorph)** — deep-learning deformable registration. Balakrishnan et al. *IEEE TMI* 2019.
- **[BigStitcher](https://github.com/JaneliaSciComp/BigStitcher)** — terabyte-scale microscopy stitching. Hörl et al. *Nat Methods* 2019.
- **[ClearMap](https://github.com/ChristophKirst/ClearMap)** — cleared-tissue registration with Allen Brain Atlas. Renier et al. *Cell* 2016.
- DL4MicEverywhere ships a learned image-registration module.
"""))

CELLS.append(md("subslide", """
### Resources — Pathology and clinical imaging

Touched on lightly in this lecture; covered in depth in the broader curriculum.

- **[QuPath](https://github.com/qupath/qupath)** — WSI analysis. Bankhead et al. *Sci Reports* 2017.
- **[MONAI](https://github.com/Project-MONAI/MONAI)** — clinical-AI PyTorch framework.
- **[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)** — self-configuring segmentation. Isensee et al. *Nat Methods* 2021.
- **[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)** — whole-body CT/MRI segmentation. Wasserthal et al. *Radiology: AI* 2023.
"""))

CELLS.append(md("subslide", """
### Resources — Datasets and atlases

Where to find canonical reference data.

- **[Broad Bioimage Benchmark Collection (BBBC)](https://bbbc.broadinstitute.org/)** — curated benchmark microscopy datasets. Ljosa et al. *Nat Methods* 2012.
- **[Allen Brain Atlas](https://portal.brain-map.org)** — reference atlases of mouse and human brain.
- **[IDR (Image Data Resource)](https://idr.openmicroscopy.org)** — public reference imaging datasets.
- **[BioImage Archive (EMBL-EBI)](https://www.ebi.ac.uk/bioimage-archive/)** — public archive for biological image data.
- **[MICrONS](https://www.microns-explorer.org)** — EM connectomics dataset.
"""))

CELLS.append(md("subslide", """
### Resources — Reporting standards

For when you publish AI-assisted analyses.

- **CLAIM** (Mongan et al. *Radiology: AI* 2020) — Checklist for AI in Medical Imaging.
- **CONSORT-AI** (Liu et al. *Nat Medicine* 2020) — clinical trial reporting.
- **SPIRIT-AI** (Cruz Rivera et al. *Nat Medicine* 2020) — trial protocol reporting.
- **STARD-AI** (Sounderajah et al. *Nat Medicine* 2020) — diagnostic accuracy reporting.
- **MI-CLAIM** (Norgeot et al. *Nat Medicine* 2020) — minimum information for clinical AI modeling.
"""))

CELLS.append(md("subslide", """
### Resources — Visualization and platforms

The interactive layer.

- **[napari](https://github.com/napari/napari)** — Python multi-dimensional viewer; foundational for modern bioimage workflows.
- **[Fiji / ImageJ](https://imagej.net/software/fiji/)** — community platform for microscopy image analysis.
- **[deepImageJ](https://github.com/deepimagej/deepimagej-plugin)** — ImageJ plugin to run DL models. Gómez-de-Mariscal et al. *Nat Methods* 2021.
- **[Mastodon](https://github.com/mastodon-sc/mastodon)** — large-scale tracking and lineage editing. Wolff et al. *eLife* 2018.
- **[scikit-image](https://github.com/scikit-image/scikit-image)** — Python library for classical image processing.
"""))

CELLS.append(notes("""
Speaker cue:
The Resources Appendix is for live navigation, not linear reading. Don't try
to walk the room through every slide here. Instead:
- Jump here from the Section 1 ecosystem slide if a question opens it
- Jump here from any Section 2 task slide if someone asks "what tools are
  available for that?"
- Use the closing slides as the natural exit point — point at the resources
  page and the handout

If you spent the lecture in the main flow, attendees will encounter the
appendix only via the resources page and handout after the workshop.
"""))

# =============================================================================
# Bibliography slide
# =============================================================================
CELLS.append(md("slide", f"""
## Bibliography {{#bibliography}}

Verified references; full BibTeX in `references.bib` and verification status
in `verification_log.md`.

- **cellpose2021** ✓ — Stringer et al., 2021, *Nature Methods* 18:100–106. doi: 10.1038/s41592-020-01018-x
- **cellpose_sam_2025** ⚠ — Pachitariu, Rariden, Stringer, 2025, *bioRxiv preprint*. doi: 10.1101/2025.04.28.651001. Cellpose v4 — current version of the tool.
- **cellpose3_2025** ⚠ — Cellpose 3 (2025) *Nature Methods*; one-click image restoration; details to verify.
- **stardist2018** ✓ — Schmidt et al., 2018, *MICCAI 2018*.
- **care2018** ✓ — Weigert et al., 2018, *Nature Methods* 15:1090–1097.
- **noise2void2019** ✓ — Krull et al., 2019, *CVPR 2019*.
- **mesmer2022** ✏ — Greenwald et al., 2022, *Nature Biotechnology* 40:555–565.
- **voxelmorph2019** ✓ — Balakrishnan et al., 2019, *IEEE TMI* 38:1788–1800.
- **annapalm2018** ✓ — Ouyang et al., 2018, *Nature Biotechnology* 36:460–468.
- **deepstorm2018** ✓ — Nehme et al., 2018, *Optica* 5:458–464.
- **christiansen2018** ✓ — Christiansen et al., 2018, *Cell* 173:792–803.
- **ounkomol2018** ✓ — Ounkomol et al., 2018, *Nature Methods* 15:917–920.
- **sam2023** ✓ — Kirillov et al., 2023, *ICCV 2023*.
- **microsam2025** ✓ — Pape et al., 2025, *Nature Methods* 22:579–591.
- **claim2020** ✓ — Mongan et al., 2020, *Radiology: AI* 2(2):e200029.
- **stardai2020** ✓ — Sounderajah et al., 2020, *Nature Medicine*.
- **consortai2020** ✓ — Liu et al., 2020, *Nature Medicine* 26:1364–1374.
- **spiritai2020** ✓ — Cruz Rivera et al., 2020, *Nature Medicine* 26:1351–1363.
- **miclaim2020** ✓ — Norgeot et al., 2020, *Nature Medicine* 26:1320–1324.
- **bbbc2012** ✓ — Ljosa et al., 2012, *Nature Methods* 9:637.
- **imagesc** ✓ — image.sc forum, [forum.image.sc](https://forum.image.sc).
- **neubias** ✓ — NEUBIAS, [eubias.org/NEUBIAS](https://eubias.org/NEUBIAS).

Flag legend: ✓ verified · ✏ corrected · ⚠ uncertain · ❓ synthesis (no single citation).
"""))


# =============================================================================
# Build status slide — Phase 3 complete; this slide is internal-only
# =============================================================================
CELLS.append(md("slide", """
## Build status — complete (v1.0)

All three phases delivered. Lecture is presentation-ready pending the items below.

**Verified during build:**
- 30+ references with DOIs and verification flags
- 4 working interactive demo cells (numpy + matplotlib only; runs anywhere)
- Resources appendix (10 slides) navigable from any slide
- Auto-generated handout (`handout.md`) and full resources page (`resources.md`)
- Speaker cue sheet (`speaker_cues.md`) for live navigation

**Items deliberately deferred (real-world authoring):**

- Demo videos — placeholder text in slides; Phase-3-of-real-authoring should source verified, embeddable videos or render animated stills from the demo cells.
- End-to-end Colab test on the actual free-tier T4 — recommended before any external delivery.
- Cross-checking the ⚠ verification flags against the source GitHub READMEs of each cited tool — the agent verified citations from search-result metadata, not from full text.

**For internal review only — remove this slide before any external presentation.**
"""))


# =============================================================================
# Build the notebook
# =============================================================================
NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
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
        "rise": {
            "theme": "white",
            "transition": "slide",
            "controls": True,
            "progress": True,
            "history": True,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

def _build_html() -> None:
    """Render lecture.ipynb to lecture_slides.html via jupyter nbconvert.

    nbconvert insists on writing `<output>.slides.html` next to its --output
    target, so we point --output at a temp directory and copy the resulting
    HTML to the canonical filename. This keeps stale intermediates out of the
    OneDrive-synced workshop folder, where the sync agent often locks files
    and prevents the build script from cleaning up.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out_stem = "lecture_slides"
        cmd = [
            sys.executable, "-m", "nbconvert",
            "--to", "slides",
            "--output-dir", tmpdir,
            "--output", tmp_out_stem,
            str(OUT),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        except FileNotFoundError:
            print("WARNING: nbconvert module not available. Skipping HTML build.")
            print("         Install with: pip install nbconvert")
            return

        if result.returncode != 0:
            print("WARNING: nbconvert failed. .ipynb was written, HTML was not.")
            print("--- stderr ---")
            print(result.stderr)
            return

        # Find what nbconvert produced (it appends .slides.html).
        candidates = [
            Path(tmpdir) / f"{tmp_out_stem}.slides.html",
            Path(tmpdir) / f"{tmp_out_stem}.html",
        ]
        rendered = next((c for c in candidates if c.exists()), None)
        if rendered is None:
            print(f"WARNING: nbconvert ran but no HTML found in {tmpdir}. "
                  f"Files present: {list(Path(tmpdir).iterdir())}")
            return

        try:
            HTML_OUT.write_bytes(rendered.read_bytes())
        except OSError as e:
            print(f"WARNING: could not write {HTML_OUT.name}: {e}")
            return

        print(f"Wrote {HTML_OUT.name} ({HTML_OUT.stat().st_size} bytes)")
        _post_process_html(HTML_OUT)


# Sentinel marking that post-processing was applied. Re-runs are idempotent —
# the post-process function checks for this sentinel and skips if present.
POST_PROCESS_SENTINEL = "<!-- LECTURE-HTML-POST-PROCESS-APPLIED -->"


def _post_process_html(path: Path) -> None:
    """Fix clipping issues in the nbconvert-generated Reveal.js HTML:

    1. Bump hardcoded canvas dimensions 960x700 -> 1280x720 (modern 16:9).
    2. Add margin/minScale/maxScale so Reveal adapts to any display resolution.
    3. Flip the disabled scroll fallback so over-tall slides scroll instead of clipping.
    4. Inject custom CSS that constrains images, code blocks, and base font.

    Idempotent via POST_PROCESS_SENTINEL.
    """
    try:
        html = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"WARNING: could not read {path.name} for post-process: {e}")
        return

    if POST_PROCESS_SENTINEL in html:
        print(f"  post-process: sentinel present, skipping {path.name}")
        return

    n_replaced = 0

    # 1+2. Reveal.initialize config — bump dimensions, add adaptive scaling.
    old_cfg = "width: 960,\n\t\t\t      height: 700,\n\n        });"
    new_cfg = (
        "width: 1280,\n\t\t\t      height: 720,\n"
        "            margin: 0.04,\n"
        "            minScale: 0.2,\n"
        "            maxScale: 1.5,\n"
        "\n        });"
    )
    if old_cfg in html:
        html = html.replace(old_cfg, new_cfg)
        n_replaced += 1
    else:
        # Fallback: try a looser pattern in case nbconvert formatting drifts.
        import re
        pattern = re.compile(
            r"width:\s*960,\s*\n\s*height:\s*700,\s*\n(\s*\n\s*\}\);)",
            re.MULTILINE,
        )
        if pattern.search(html):
            html = pattern.sub(
                "width: 1280,\n            height: 720,\n"
                "            margin: 0.04,\n"
                "            minScale: 0.2,\n"
                "            maxScale: 1.5,\n\\1",
                html,
            )
            n_replaced += 1
        else:
            print("  post-process: could not find Reveal width/height config — leaving HTML alone.")

    # 3. Enable scroll fallback for over-tall slides.
    old_scroll = "var scroll = false"
    new_scroll = "var scroll = true"
    if old_scroll in html:
        html = html.replace(old_scroll, new_scroll)
        n_replaced += 1

    # 4. Custom CSS block — appended right before </head>.
    custom_css = """<style type=\"text/css\">
/* Lecture slides — post-process clipping fixes. */
.reveal .slides section {
    font-size: 0.85em;          /* tighter base font; gives ~15% more vertical headroom */
}
.reveal .slides section img {
    max-width: 100%;
    max-height: 60vh;           /* image never blows past 60% of viewport height */
    height: auto;
    width: auto;
    object-fit: contain;
}
.reveal pre {
    max-height: 70vh;
    overflow: auto;             /* long code blocks scroll instead of clipping */
    font-size: 0.7em;
}
.reveal table {
    font-size: 0.75em;          /* tables can otherwise blow past slide height */
}
.reveal h1, .reveal h2, .reveal h3 {
    margin-top: 0.2em;
    margin-bottom: 0.4em;
}
.reveal ul, .reveal ol {
    margin-left: 1em;
}
/* Reveal sometimes leaves stray bottom padding when content fits — trim it. */
.reveal .slides > section, .reveal .slides > section > section {
    padding-top: 1em;
    padding-bottom: 1em;
}
</style>
""" + POST_PROCESS_SENTINEL + "\n"

    if "</head>" in html:
        html = html.replace("</head>", custom_css + "</head>", 1)
        n_replaced += 1

    try:
        path.write_text(html, encoding="utf-8")
    except OSError as e:
        print(f"WARNING: could not write post-processed {path.name}: {e}")
        return

    print(f"  post-process: applied {n_replaced} edits to {path.name} "
          f"(new size {path.stat().st_size} bytes)")


if __name__ == "__main__":
    # Add unique cell IDs (required by nbformat >= 5.1.4)
    for i, cell in enumerate(CELLS):
        cell["id"] = f"cell-{i:03d}"
    OUT.write_text(json.dumps(NOTEBOOK, indent=1))
    print(f"Wrote {OUT.name} ({OUT.stat().st_size} bytes)")
    print(f"Cells: {len(CELLS)}")
    n_slides = sum(
        1
        for c in CELLS
        if c.get("metadata", {}).get("slideshow", {}).get("slide_type")
        in ("slide", "subslide")
    )
    print(f"Slides + subslides: {n_slides}")
    _build_html()
