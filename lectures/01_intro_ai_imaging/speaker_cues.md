# Speaker Cues — Lecture 1

A reference for the live presenter. Reading the room is the most important skill in delivering this lecture; the slides have far more material than fits in 70 minutes, and which drill-downs to open depends entirely on who's in front of you. This page captures the patterns that work.

## Reading the room — three audience signals

By the end of Section 1 (the "Why now?" framing) you will have one of three reads:

### Signal A: Biology-first, AI-curious

**You see:** notes being taken on the tools list, questions about which tool to use, comments like "I didn't know about ZeroCostDL4Mic."

**Open these drill-downs:**
- Section 2 Segmentation — both drill-downs (when it works, when it fails)
- Section 2 Restoration — hallucination drill-down
- Resources Appendix — community platforms slide and segmentation-tools slide

**Skip or compress:**
- Section 3 — keep it short; biology-first audiences may glaze on training/loss/gradient descent. Pivot to the generalization slide quickly.

### Signal B: Computational, methods-curious

**You see:** questions about architectures, loss functions, training data sizes; references to specific papers.

**Open these drill-downs:**
- Section 2 Segmentation — failure-mode drill-down (less the success drill-down, which is well-known to this audience)
- Section 4 — failure-mode patterns slide; let the conversation go
- Resources Appendix — datasets and atlases slide; foundation-models slide

**Skip or compress:**
- Section 1 ecosystem slide — this audience already knows the ecosystem
- Section 6 — they already know reporting standards exist; just point at the appendix

### Signal C: Mixed audience (most common)

**You see:** mixed body language; some attendees nodding through Section 3, others looking puzzled.

**Strategy:** stay in the main flow, open one drill-down per Section 2 task that you have time for, end with a clear pointer at the resources page and handout. Mixed rooms benefit most from the resources appendix as the take-home — they can self-direct after the workshop.

## Drill-down opening cues by slide

A per-slide reference for which drill-downs to consider opening:

### Section 1 — Why now?

- *Three drivers slide:* if anyone asks "what made the data layer real?" — open the ecosystem subslide (you're already there, but linger). Otherwise move on.

### Section 2 — Tasks

- *Segmentation slide:* always open the "when Cellpose-SAM works" drill-down (sets up Lab 1). Open the "when it fails" drill-down if time permits — the failure modes are the harder lesson.
- *Restoration slide:* open the hallucination drill-down if anyone asks about virtual staining or label-free imaging, or if you want to bridge into Section 6.
- *Registration slide:* no drill-down currently; if a question opens it, jump to the registration entry in the resources appendix.

### Section 4 — When AI works/fails

- *Failure-mode patterns slide:* this is the lecture's emotional climax. Take time. Don't rush. Use audience experiences if any are offered.

### Section 5 — Validation

- *Metrics slide:* the metrics-vs-biology distinction is what Lab 2 will demonstrate. Make sure attendees know to look for this in Lab 2.

### Section 7 — Closing

- Always show the resources page and handout slide. Always point at image.sc forum as the single most useful URL.

## Time-pressure cues

If you are running 5+ minutes long entering Section 5:

- Compress Section 6 (drop one slide, fold image-integrity content into Section 5)
- Skip the second drill-down on Segmentation if you opened both
- Skip the optional restoration hallucination drill-down (it can also live in Section 6)

If you are running early:

- Open both Segmentation drill-downs
- Open Restoration hallucination drill-down
- Spend more time on Section 4's failure-mode patterns
- Take more questions during Section 5

## Live demo opportunities

The four interactive code cells run in real time:

1. *Setup verification* — runs in 1 second. Use as a "check the room is technically OK" moment.
2. *Train/val/test split visualization* — runs in 2 seconds. Use during Section 3 to make the split concrete.
3. *Cellpose-style result viewer* — runs in 3 seconds. Use during Section 2 Segmentation to make the "good vs bad prediction" concept visible.
4. *Metric vs biology plot* — runs in 2 seconds. Use during Section 5 to make the metrics gap visible *before* attendees see it again in Lab 2.

All four are pure-numpy / pure-matplotlib so they work on any reasonable Python install. No GPU or large packages required.

## Common questions and how to handle them

### "Should I use Cellpose or StarDist?"

Both work. Cellpose-SAM is more general (good zero-shot on diverse cell types); StarDist is stronger on dense, round objects (especially nuclei). Try both on your own data. Lab 1 demonstrates Cellpose-SAM specifically; Lab 2 makes the "compare different models" pattern concrete.

### "How do I know if a model will work on my data?"

You don't, until you validate. Run it on a small held-out set with ground truth. Look at failures qualitatively first (what does it get wrong?), then quantitatively (what's the count error, the IoU, the per-class breakdown?). The morning-lecture validation discipline + Lab 2 are designed to give you the workflow for this.

### "Is foundation-model segmentation always better?"

No. Foundation models are general; task-specific models are tuned. For sample types similar to a pretrained model's training data, the pretrained model usually wins. For novel sample types, foundation models extend reach at the cost of more prompt engineering. Lab 3b makes this trade-off concrete.

### "What about my Zeiss CZI files?"

The pylibCZIrw tutorial in the resources page handles this. Most modern Python tools (Cellpose, ilastik, napari, AICSImageIO, bioio) read CZI directly via Bio-Formats or pylibCZIrw.

### "Do I need a GPU?"

For the workshop today, no. Colab provides a free T4 GPU if you enable it, which speeds Cellpose-SAM and Noise2Void by 5–10×. CPU works for Lab 1 and Lab 2; GPU strongly preferred for Lab 3.

## After the workshop

Remind attendees of three things at the close:

1. The handout URL — bookmark it tonight
2. The image.sc forum — join, search, ask
3. ZeroCostDL4Mic / DL4MicEverywhere — start here for any new method

The first 48 hours after a workshop are when self-directed learning is most likely. A handout that lives at one URL and a community where they can ask questions are worth more than any specific tool we covered.
