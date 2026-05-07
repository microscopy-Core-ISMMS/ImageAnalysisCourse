# Workshop Schedule

This material is delivered as **Day 3 of the 2026 MABC Image Analysis Workshop** at the Icahn School of Medicine at Mount Sinai.

**Date:** Monday, May 11, 2026
**Venue:** Annenberg 5-212, Icahn School of Medicine at Mount Sinai
**Format:** Two half-day blocks (morning + afternoon), in-person hands-on with attendee laptops

By Day 3 the audience has already been through Days 1 (image-analysis intro + QuPath) and 2 (Fiji hands-on + commercial demos). This day adds modern AI methods on top of that foundation. The afternoon block is the home for the materials in this Jupyter Book; the morning block (Arun Narasimhan's Python/Colab session) is companion content delivered on the Mount Sinai HPC server.

## Day 3 at a glance

| Time | Block | Instructor |
|---|---|---|
| 09:00 – 12:30 | Session 5 — Python / Google Colab (Hands-on) | Arun Narasimhan |
| 12:30 – 13:30 | Lunch | — |
| 13:30 – 17:00 | Session 6 — GUI-Based AI Tools & Applications (Hands-on) | Nikos Tzavaras |

> **Time-slot swap, May 2026.** The originally-published schedule had these in the opposite order. Topics didn't move; only the time slots flipped. Cellpose and μSAM appear twice on Day 3 by design — Arun on the Mount Sinai HPC server in the morning, and again on Google Colab in the afternoon as part of this Jupyter Book companion. Same tools, different surfaces, deliberate reinforcement.

## Morning — Session 5: Python / Google Colab (Arun Narasimhan)

The morning block runs against the Mount Sinai HPC server with Napari, Cellpose, μSAM, and Google Colab pre-installed. Arun's session materials live in the workshop's main folder under `S6_Python_Google Colab (Hands-on)/`.

The two ramp-up notebooks in this repository — `ramp-up/notebooks/python_basics.ipynb` and `image_basics.ipynb` — are available as a companion runway for participants who want a Python/numpy/image-data refresh, either before the workshop or alongside the morning block.

## Afternoon — Session 6: GUI-Based AI Tools & Applications (Nikos Tzavaras)

The afternoon block delivers the AI material in this Jupyter Book. Times below are suggested rather than rigid; the room can adjust by ±15 minutes per block depending on pacing.

### 13:30 – 14:00 — Welcome, setup, and orientation (30 min)

- Brief framing for the afternoon: what we will and won't cover
- Confirm everyone's Google Colab opens and loads this repository
- Run the [setup self-check notebook](notebooks/00_setup_self_check) end-to-end as a group
- Set the participant choice point for the day (live-room paths vs post-workshop self-paced; see "How this material is layered" below)

### 14:00 – 14:45 — Lecture: AI for Scientific Image Analysis (45 min)

A condensed delivery of the [morning lecture](lectures/01_intro_ai_imaging/lecture) — AI task taxonomy, how models learn, where they work and fail, validation, and responsible use. Compressed from 90 minutes since by Day 3 the audience already has imaging fluency from Days 1 and 2.

The lecture deck has drill-down sub-slides; the live presenter opens the ones that fit the room. See [`lectures/01_intro_ai_imaging/speaker_cues.md`](lectures/01_intro_ai_imaging/speaker_cues.md) for guidance.

### 14:45 – 15:00 — Break (15 min)

### 15:00 – 15:30 — Walking the day (30 min)

The [bridge talk](lectures/02_walking_the_day/lecture) previewing each afternoon lab — goal, what success looks like, where we expect things to break. Compressed from 45 minutes for the afternoon-only delivery. This block prevents the click-and-watch failure mode in the labs.

### 15:30 – 16:30 — Lab block 1: Cellpose-SAM segmentation + validation (60 min)

Run together, since Lab 2 picks up Lab 1's outputs:

- [Lab 1 — Cellpose-SAM](notebooks/01_cellpose_segmentation): apply pretrained Cellpose-SAM to canonical and synthetic data; customization workshop with `diameter`, `model_type`, `flow_threshold`, `cellprob_threshold`; failure-mode finale on out-of-distribution data.
- [Lab 2 — Validation & quantification](notebooks/02_validation_quantification): IoU, Dice, instance-level metrics, the metrics-vs-biology gap.

Pacing target: ~35 min Lab 1, ~25 min Lab 2. The lecture and walking-the-day already covered the conceptual setup, so the labs run leaner than in the standalone one-day Demo.

### 16:30 – 16:45 — Break (15 min)

### 16:45 – 17:00 — Wrap-up + path forward (15 min)

- Quick reflection on the day's labs
- The [GUI-AI tools walkthrough](notebooks/05_desktop_gui_complements) (Trainable Weka, QuPath Pixel Classifier, deepImageJ, StarDist Fiji + QuPath) — covered as live demo and as take-home reference.
- Pointers into [Notebook 04](notebooks/04_community_platforms) (community catalog + 5 inline mini-workflows + live BioImage Model Zoo browse) and [Lab 3 options](notebooks/03a_denoising_n2v) (denoising) and [3b](notebooks/03b_foundation_model_segmentation) (foundation models), as self-paced post-workshop work.
- The [Resources page](resources) and [Handout](handout) — bookmark these.
- Q&A; workshop feedback (anonymous form via the workshop organizers).

## How this material is layered

The repository is more comprehensive than fits in any one afternoon. The live-room block above hits the **core spine**; the rest is post-workshop self-paced.

- **Core spine (live-room).** Setup self-check (Notebook 00) · Lecture 1 · Walking the Day · Lab 1 (Cellpose-SAM) · Lab 2 (Validation) · Notebook 05 (GUI-AI tools demo).
- **Extensions (in-room if time, post-workshop otherwise).** Lab 3a (Noise2Void denoising) · Lab 3b (Foundation-model segmentation with SAM) · Notebook 04 (Community platforms catalog + 5 mini-workflows + live BiMZ browse).
- **Reference (post-workshop self-paced).** Resources page · Handout · Prerequisites · Acknowledgments · Ramp-up notebooks (Python and image-data basics) · Lecture references and bibliography.

Participants can run any extension or reference content on their own time; everything is Colab-ready, and the conda environment in `env/environment.yml` reproduces the same setup locally.

## Pre-workshop expectations

Attendees on Day 3 should:

1. Run the [setup self-check notebook](notebooks/00_setup_self_check) on Colab before the workshop day. About 20–30 minutes.
2. Confirm Google account and GitHub access.
3. *(Optional)* Run the [ramp-up notebooks](ramp-up/notebooks/python_basics) the evening before if Python or image-data fluency feels rusty.

See [prerequisites](prerequisites) for full details.

## A note on standalone delivery

The schedule above is calibrated for Day 3 of a multi-day workshop. The materials in this repository can also be delivered standalone as a one-day workshop — the `Microscopy_Workshop_Demo*` template (the seed for this repo) contains a one-day schedule with a longer lecture, a longer walking-the-day, and the full Lab 3 in-room. For standalone delivery, see the speaker cues and the original lecture timing in [`lectures/01_intro_ai_imaging/`](lectures/01_intro_ai_imaging/).
