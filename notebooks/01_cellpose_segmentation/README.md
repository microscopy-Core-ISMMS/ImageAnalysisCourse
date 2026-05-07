# Lab 1 — Pretrained Segmentation with Cellpose-SAM

Working Jupyter notebook for the workshop's first hands-on lab (90 minutes, slot 13:00–14:30).

## Files

- `notebook.ipynb` — the runnable lab notebook (33 cells: 19 markdown, 14 code). **Generated** — do not edit by hand.
- `build_notebook.py` — Python source-of-truth. Edit this and re-run to regenerate the .ipynb.

## Build

```bash
python build_notebook.py
```

## Status

v1.0 (built 2026-05-02). Targets free-tier Google Colab (T4 GPU). Awaiting end-to-end Colab dry run on Day 6 of the pilot build (Thu 2026-05-07).

## Design choices worth knowing

- **Dataset is fully synthetic-from-canonical.** Both the easy and hard cases use `skimage.data.human_mitosis()` — the hard case is the same image with simulated low-SNR noise. No external downloads, no broken URLs, fully deterministic. Trade-off: real data would be more pedagogically rich, but for an internal dry run the reliability win outweighs the realism loss.
- **Cellpose v4 API.** The notebook uses `cellpose.models.CellposeModel` (the v3 `cellpose.models.Cellpose` class was removed). Default pretrained model is `'cpsam'`. The `eval()` method returns 3 values (masks, flows, styles) — the v3 4th return (`diams`) is gone because v4 is robust to diameter.
- **Defensive alternative-model lookup.** The "try a different pretrained model" cell wraps the `'nuclei'` load in try/except and prints `models.MODEL_NAMES` on failure, in case the available pretrained_model strings drift between minor cellpose versions.
- **CPU fallback works but is slow.** Each Cellpose-SAM eval on CPU takes 1–2 minutes vs a few seconds on T4 GPU. The setup cell prints a clear warning if no GPU is detected.

## Outputs the notebook produces

Saves to `./lab1_outputs/` in the notebook's working directory:

- `img_easy.tif`, `masks_easy.tif` — easy-case image and segmentation
- `img_hard.tif`, `masks_hard.tif` — hard-case image and segmentation
- `features_easy.csv`, `features_hard.csv` — per-object feature tables

These are consumed by Lab 2 (validation and quantification).

## Known unknowns to verify in the Day 6 Colab dry run

- Cellpose-SAM model weight download time (~400 MB) on a fresh free-tier Colab runtime. If it consistently exceeds 60s, add a "be patient" hint to the install cell.
- Whether `'nuclei'` is still a valid `pretrained_model` string in the installed Cellpose v4 minor version. If not, swap to whatever `models.MODEL_NAMES` reports.
- The exact `eval()` return tuple length (3 vs 4) on the installed minor version. The build assumes 3 per current docs (4.1.x). The defensive try/except in the alt-model cell will surface a mismatch if it appears.
