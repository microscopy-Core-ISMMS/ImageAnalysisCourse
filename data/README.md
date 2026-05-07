# Data Sources

The workshop is designed so attendees can run every notebook end-to-end **without downloading any large external dataset**. Most labs use one of three sources, each described below.

## What the labs actually use

### scikit-image built-in datasets (Labs 1, 2, 5)

Bundled with the `scikit-image` package. No download required after `pip install scikit-image`.

| Function | What it returns | Used in |
|---|---|---|
| `skimage.data.cell()` | A 2D fluorescence single-cell image | Lab 1 (real-data section) |
| `skimage.data.human_mitosis()` | A 2D fluorescence nuclei image (mitosis stain) | Lab 1 (real-data section), Notebook 05 sample exports |
| `skimage.data.cells3d()` | A 3D confocal stack — `(z=60, c=2, y=256, x=256)` with DAPI + membrane channels | Lab 1 (mid-Z slice for nuclei segmentation), Notebook 05 sample exports |

These are real microscopy images, openly redistributable, and known to be Cellpose-compatible for the easy-case demo.

### Synthetic generators (Labs 1, 2, 3a, 3b, 4)

Each lab that needs a controlled-truth example generates its own synthetic image inline, so the truth count and topology are fully known. No external data, no download.

- **Lab 1** — `make_easy_image()` (12 round well-separated cells) and `make_hard_image()` (18 irregular dense cells, OOD on purpose).
- **Lab 2** — uses Lab 1's saved output or falls back to a synthetic ground-truth set if Lab 1 hasn't been run.
- **Lab 3a** — `make_clean_image()` + `add_noise()` (Poisson + Gaussian) for a low-SNR pair with paired clean reference.
- **Lab 3b** — irregular blob structures designed to fall outside Cellpose's training distribution.
- **Notebook 04** — each of the five mini-workflows (CARE, fnet, pix2pix, Deep-STORM, YOLOv2-style) generates 64–128 paired training samples inline.

### One canonical download with synthetic fallback (Lab 0)

The setup self-check (Notebook 00) tries to download a single small Cellpose example image (`http://www.cellpose.org/static/data/img02.png`). If the network is blocked, the notebook falls back to a synthetic 200×200 image with eight bright blobs so the rest of the self-check still runs.

This is the only network call in any of the workshop notebooks aside from package installs.

### One large model download with simulated fallback (Lab 3b)

Lab 3b downloads the SAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`, ~360 MB) on first run. If the download fails, the notebook drops to `SIMULATE_SAM=True` mode where SAM behavior is mocked with a simple circular mask around each prompt point, so the workflow still demonstrates the prompt-based segmentation pattern.

The `.gitignore` excludes `sam_vit_*.pth` so this checkpoint is never tracked by Git.

### Live registry browse (Notebook 04)

Notebook 04's BioImage Model Zoo section calls `bioimageio.core` to query the BiMZ registry live and download a small pretrained model (a few tens of MB). If the registry is unreachable, the notebook falls back to a curated three-entry list so the cells still demonstrate the API.

## Why no bundled datasets

Two reasons:

1. **Reproducibility.** The notebooks generate or fetch their own data, so the labs survive any change to a dataset URL or hosting service. Earlier drafts of this workshop pointed at BBBC and externally-hosted training images — both are still useful references but neither is required for the labs to run.
2. **Distribution.** Bundling images in the repo would push the size from ~50 MB to several GB. Workshop attendees on free-tier Colab can fetch what they need on demand; nothing has to be pre-staged on a USB drive.

## After the workshop — bringing your own data

The notebooks are written so swapping in your own image typically requires changing **one path** at the top of the relevant notebook. Look for the cells that load `data.cell()` or call `make_easy_image()` and replace with `tifffile.imread("path/to/your_image.tif")`. The downstream cells are mostly format-agnostic (numpy arrays in, numpy arrays out).

For larger projects that need real benchmark datasets:

- **[BBBC — Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/)** — curated benchmark microscopy datasets with documented ground truth. Good for validation work beyond Lab 2.
- **[BioImage Archive (EMBL-EBI)](https://www.ebi.ac.uk/bioimage-archive/)** — public archive for biological image data.
- **[IDR — Image Data Resource](https://idr.openmicroscopy.org)** — public reference imaging datasets.
- **[Cellpose example data](https://www.cellpose.org/static/data/)** — Cellpose-hosted samples used in the original publications.
- **[Allen Cell Image Library](http://www.cellimagelibrary.org)** — curated cell biology images.

These are listed in full on the [Resources page](../resources) with citations.

## Licensing

- **scikit-image bundled images** — distributed under scikit-image's BSD 3-Clause license; reusable for educational and research purposes with attribution.
- **Synthetic generators** — produced inline by code in this repository; covered by this repository's MIT License (for the code) and CC-BY-4.0 (for the documentation around it).
- **Cellpose example image** — courtesy of the Cellpose authors; verify their site for current licensing if you redistribute.
- **SAM checkpoint** — released under the SAM authors' license; see https://github.com/facebookresearch/segment-anything.

Per-notebook citations and attributions are in `acknowledgments.md` at the repo root.
