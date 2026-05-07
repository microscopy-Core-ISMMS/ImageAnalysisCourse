# Ramp-up Evening (Optional T0 Session)

**Format:** 2.5-hour optional session, evening before the workshop day
**Audience:** Attendees whose [self-check notebook]({{ site.baseurl }}/notebooks/00_setup_self_check) recommended additional preparation, plus anyone who self-identifies as needing a runway
**Curriculum mapping:** Compresses T0.1 (Python and scripting), T0.3 (math and ML literacy), and T0.4 (image data fundamentals) into one block focused on what the workshop day will require
**Format constraints:** Designed to be deliverable in person, on Zoom, or self-paced. Nothing depends on real-time interaction.

## Why this exists

The workshop assumes biology fluency and basic Python; some attendees have one but not the other. The biology-first analyst pathway is a real and common entry into bioimage analysis, and rushed Python prep on the morning of the workshop is not a kindness. This ramp-up session gives those attendees a no-pressure runway, with content scoped exactly to what the next-day labs will use and nothing more.

This session is optional. Attending it does not mean an attendee is "behind." Skipping it is fine if the self-check showed readiness.

## Learning objectives

By the end of the session, attendees should be able to:

1. Open, run, and modify a Jupyter / Colab notebook with confidence
2. Manipulate basic numpy arrays representing images: indexing, slicing, computing simple statistics
3. Load a microscopy image and inspect its shape, dtype, and metadata
4. Describe at a conceptual level what training, validation, and a model output mean — enough to follow the morning lecture without confusion
5. Arrive at the workshop the next day ready to focus on the AI methods rather than the tooling

## Session outline (2.5 hours)

### 18:30 – 18:45 — Welcome and orientation (15 min)

- Why this evening: framing as a runway, not a remedial session
- What we will cover, what we will not, and what to expect tomorrow
- Confirm everyone's Colab opens and runs; one-on-one help for anyone who is blocked
- Brief introductions if the group is small

### 18:45 – 19:30 — Block 1: Python and notebooks (45 min)

The minimum Python the workshop requires.

- Variables, types, and the values that come up in image work (integers, floats, arrays)
- Lists, dictionaries, indexing, slicing — concretely, with image-shaped examples
- Function calls and how to read a function's documentation
- numpy's array model: arrays as multi-dimensional grids, shape, dtype, basic operations
- The "predict before run" habit: read the cell, predict the output, then run

This block is hands-on the entire time. Attendees follow along in a notebook tailored for the session (`ramp-up/notebooks/python_basics.ipynb`, to be authored). The notebook intentionally repeats some of the workshop's setup-check material so attendees who attend both feel the continuity.

### 19:30 – 19:45 — Break (15 min)

### 19:45 – 20:30 — Block 2: Image data fundamentals (45 min)

The minimum image-data fluency the workshop requires.

- What a digital image is: pixels, dimensions (height, width, channels, time, Z), bit depth
- Common formats encountered in microscopy (TIFF, OME-TIFF) and how to inspect their metadata
- Loading an image into numpy and reasoning about its shape
- Display: matplotlib basics for showing images with appropriate contrast, colormaps, and scale
- Calibration and pixel size: the difference between pixel coordinates and physical units, briefly

Hands-on the entire time. Attendees follow along in a notebook tailored for the session (`ramp-up/notebooks/image_basics.ipynb`, to be authored).

### 20:30 – 20:45 — Break (15 min)

### 20:45 – 21:15 — Block 3: ML literacy at a glance (30 min)

The minimum conceptual ML fluency the workshop requires. Lecture-style, not hands-on.

- What "machine learning" means at the level the workshop will use the term
- Training, validation, and test data — what each is for and why they need to be different
- Pretrained models, fine-tuning, zero-shot inference: the three operating modes the workshop will use
- What a "model output" actually is: a function applied to data, with all the limitations of any function fit to data
- Where models fail: a quick preview of the morning lecture's failure-mode discussion

This block has no hands-on component. The intent is to give attendees the vocabulary and mental model so the morning lecture can move at its planned pace.

### 21:15 – 21:30 — Wrap-up and Q&A (15 min)

- Preview of the morning: what to expect, where to be, what to bring
- Q&A — open floor
- Reassurance: tomorrow's lectures are designed to land for everyone in this room

## Materials

- `ramp-up/notebooks/python_basics.ipynb` (to be authored)
- `ramp-up/notebooks/image_basics.ipynb` (to be authored)
- `ramp-up/slides/ml_at_a_glance.pdf` (short slide deck for Block 3, to be authored)
- The same `environment.yml` used for the main workshop

## Notes for instructors

- Keep the tone calm and supportive. Attendees who self-identified as needing the ramp-up may be self-conscious about it; the framing matters.
- Resist over-teaching. The session is scoped to what the workshop *requires*, not what would be ideal. Attendees can deepen later.
- Walk the room (or pair attendees in breakout rooms on Zoom) during hands-on blocks. The notebooks are designed to be self-explanatory but human help during the first 10 minutes prevents anyone from getting stuck.
- Time is tight. If a block runs long, skip the optional cells and end on time so attendees get a full night before the workshop.

## Self-paced option

For attendees who cannot attend the live ramp-up session, the same content is available in the `ramp-up/` folder of this repository as self-paced notebooks. Each notebook has the same structure as the live block: 45 minutes (or 30 for ML at a glance), hands-on where appropriate, with an end-of-notebook self-check. Attendees can complete them at their own pace and arrive at the workshop with the same baseline as live-session participants.
