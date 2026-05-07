# Prerequisites

The workshop is designed for biologists with technical fluency. The expected entry profile is realistic about the diversity of paths into bioimage analysis: some attendees will arrive from the formally computational route, while others will arrive from biology with computational skills acquired on the job. The prerequisites and self-check below assume the latter as the harder case and ensure both pathways converge on the same starting line for the workshop.

## What attendees should know walking in

**Biology and microscopy.** Active practice, or comparable experience, with light microscopy data. Familiarity with at least one commonly-used modality (widefield, confocal, light-sheet) and the artifacts particular to it. This is the workshop's biological bedrock; we do not teach it.

**Python.** Comfort with the basics: variables, lists and dictionaries, loops, function calls, importing libraries. Comfort opening and running Jupyter / Colab notebooks. Reading existing code and modifying it confidently. Writing code from scratch is *not* required.

**numpy.** Basic array thinking: arrays as multi-dimensional grids of numbers, indexing and slicing, simple arithmetic. We will introduce image-specific numpy patterns during the labs.

**Statistics.** Descriptive statistics (mean, median, standard deviation, distributions) and the idea of comparing distributions. Hypothesis testing and confidence intervals are useful but not required.

**Machine learning.** No prior ML experience required. The morning lecture establishes the conceptual scaffolding from scratch. Attendees who arrive with ML background will see familiar ground in a new context.

## What attendees do *not* need

- ML or deep learning experience
- GPU programming, CUDA, or hardware knowledge
- Linux administration skills beyond basic command line
- Cloud / HPC familiarity
- Prior exposure to Cellpose, Stardist, Noise2Void, or segment-anything

## Accounts and tooling required before arrival

1. **Google account** with access to Google Colab. This is where the labs run. Free-tier Colab is sufficient; a Colab Pro subscription is helpful but not required.
2. **GitHub account** for cloning the workshop repository and saving notebook copies. Free public-tier account is sufficient.
3. **A laptop with a modern browser.** Chrome, Firefox, Safari, or Edge. No local Python install is required for the workshop day.

For attendees who will continue with the materials locally after the workshop, the [environment documentation]({{ site.baseurl }}/env/) walks through setting up a conda environment.

## Self-check notebook

Every attendee runs the [setup self-check notebook]({{ site.baseurl }}/notebooks/00_setup_self_check) before the workshop day. The notebook does three things:

1. Verifies their Colab environment opens, mounts, and runs cells
2. Walks through a short sequence of Python and numpy operations on an image
3. Flags whether they need additional preparation before the workshop

Expected runtime: 20–30 minutes. The notebook self-grades the technical readiness check at the end.

## Optional ramp-up evening (T0 session)

For attendees whose self-check reveals gaps — typically biology-first attendees who have not used Python in a while, or who have not worked with image data programmatically — the [ramp-up evening session]({{ site.baseurl }}/ramp-up/rampup_evening) is a 2.5-hour optional session held the evening before the workshop. It compresses the most relevant content from Tier 0 of the broader curriculum (T0.1 Python, T0.3 ML literacy, T0.4 image data fundamentals) into a single block focused on what the workshop day will require.

Attending the ramp-up does not change anything about the workshop day itself; it is a runway for those who need it. Skipping it is fine for attendees whose self-check shows they are ready.

## Sources of help before and during the workshop

- The workshop instructor team (named on the repository before delivery)
- The [image.sc forum](https://forum.image.sc) for general bioimage analysis questions
- The repository's GitHub issues page for workshop-specific questions
- Lab partners — pair-programming during the workshop is encouraged

## A note on the spectrum of attendees

Attendees come to bioimage AI work from many directions. Some will be PhD biologists whose computational training came from short courses and stack-overflow searches; others will be physics or CS graduates who have only recently encountered live-cell imaging; others will be core-facility staff with broad practical exposure but uneven formal foundations. The workshop is designed to land usefully across this range. The lectures front-load conceptual scaffolding so that no one is left behind on what an AI model actually does, and the labs are structured so that participants progress at their own pace within the time blocks.

If your background is unusual relative to the typical attendee — for example, you have deep ML training but limited microscopy experience — let the instructor team know in advance. We can suggest specific pre-reading or supplementary material.
