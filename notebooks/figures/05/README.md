# notebooks/figures/05/ — screenshots for Notebook 05

This directory holds the 17 GUI screenshots for `notebooks/05_desktop_gui_complements.ipynb` (Trainable Weka, QuPath Pixel Classifier, deepImageJ, StarDist).

**Status (2026-05-21):** 17 placeholder PNGs in this folder (gray boxes labeled "Pending: Screenshot N — …"). They make the live JB render proper image boxes instead of bare `[Screenshot N]` text. Replace them one at a time with real captures — the markdown refs in NB05 already point at the right filenames.

## Naming convention (do not change)

- `screenshot_1.png` through `screenshot_17.png`
- PNG only.
- The notebook markdown uses `![alt text](figures/05/screenshot_N.png)` refs that resolve relative to the notebook file. If you rename the files, the markdown breaks.

## What each screenshot should show

See **`CAPTURE_CHECKLIST.md`** in this directory — full table + capture protocol + suggested order.

## Updating one screenshot

```bash
# Overwrite the placeholder with your real capture, preserving the name
cp ~/Desktop/MyCapture.png notebooks/figures/05/screenshot_3.png
git add notebooks/figures/05/screenshot_3.png
git commit -m "NB05: screenshot 3 — TWS labeled classes"
git push origin 2026-workshop
```

Wait ~90 sec for the GitHub Actions deploy to rebuild + redeploy the JB.

## Tracker reference

- `Claude_Workspace/deliverables.md` — row `JB-CR-04` covers this.
- `Claude_Workspace/review_items.md` — row `#28` (NB05 has 17 unfilled screenshot placeholders) closes when this dir is full of real captures.
