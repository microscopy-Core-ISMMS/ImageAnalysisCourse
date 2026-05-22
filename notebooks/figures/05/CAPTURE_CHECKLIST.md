# NB05 Screenshot Capture Checklist

**Target:** 17 screenshots for `notebooks/05_desktop_gui_complements.ipynb`.
**Save to:** this directory (`notebooks/figures/05/`).
**Filenames:** `screenshot_1.png` through `screenshot_17.png`. The notebook markdown already references these names — drop the files in and the next push triggers a JB rebuild that picks them up. No notebook edits needed.

## Tool versions (use these)

- **Fiji 2.16+** with Trainable Weka Segmentation, deepImageJ (via Updater), StarDist plugin
- **QuPath 0.5+** with built-in Pixel Classifier + StarDist extension

## Sample images for visual continuity

- **Sections A + D (Fiji):** use the same HeLa Cells sample (`File → Open Samples → HeLa Cells`) or one of the TIFFs the appendix generates (`human_mitosis.tif` works well).
- **Sections B + E (QuPath):** the QuPath tutorial H&E slide so anyone replicating can match.

## Capture protocol (5 steps)

1. Stable display 1920×1200+ so windows aren't cropped.
2. PNG only, full-window capture; crop after the fact rather than partial capture.
3. Keep tool panes/dialogs in their default positions — readers will be following along.
4. For overlay screenshots, capture with the overlay ON (don't toggle it off mid-capture).
5. Save with exact name `screenshot_N.png` per the table below.

## The 17 shots

| # | Tool | Section | What to capture |
|---|------|---------|-----------------|
| 1 | Fiji | A.1 | HeLa Cells sample image, single channel selected, full Fiji window. |
| 2 | Fiji TWS | A.2 | TWS main window, two empty default classes, toolbar visible. |
| 3 | Fiji TWS | A.3 | TWS after labeling: brush traces for `cell` (red), `background` (green), `nucleus` (blue). Three-class panel on the right. |
| 4 | Fiji TWS | A.4 | First Train output overlay — deliberately imperfect, with visible boundary errors. |
| 5 | Fiji TWS | A.4 | Probability map output as a stack, one image per class, full intensity range. |
| 6 | Fiji TWS | A.5 | TWS overlay after 3 iterations of refinement — clean class separation. |
| 7 | Fiji TWS | A.6 | "Create result" output: 8-bit indexed image with one value per class. |
| 8 | QuPath | B.1 | Tutorial H&E slide loaded, project pane left, overview thumbnail. |
| 9 | QuPath | B.2 | Slide with `tumor` (red), `stroma` (green), `background` (blue) brush annotations. |
| 10 | QuPath | B.3 | Pixel-classifier dialog open, Live prediction enabled, overlay visible. |
| 11 | QuPath | B.4 | Slide after Create objects — clean polygons. |
| 12 | Fiji updater | C.1 | Updater dialog with deepImageJ update site checked. |
| 13 | Fiji deepImageJ | C.2 | deepImageJ Run dialog with BiMZ models listed, one selected. |
| 14 | Fiji deepImageJ | C.3 | Side-by-side: input image and deepImageJ output, with citation visible. |
| 15 | Fiji StarDist | D.2 | StarDist 2D dialog with pretrained model chosen, segmentation overlay. |
| 16 | QuPath | E.1 | Preferences → Extensions panel showing StarDist installed. |
| 17 | QuPath | E.2 | Script editor with StarDist H&E Groovy script, executed across a slide region, cell objects visible. |

## Suggested capture order

Captured in a single session this is ~45–90 min:
- **Block 1 (Fiji TWS, ~30 min):** 1 → 2 → 3 → 4 → 5 → 6 → 7. Linear workflow; each builds on the previous.
- **Block 2 (QuPath PC, ~20 min):** 8 → 9 → 10 → 11. Same slide throughout.
- **Block 3 (Fiji deepImageJ + StarDist, ~20 min):** 12 → 13 → 14 → 15. Updater dialog first to install both plugins.
- **Block 4 (QuPath StarDist, ~15 min):** 16 → 17. Different QuPath session if you keep the PC and StarDist work separated.

## After capture

```bash
cd notebooks/figures/05/
# verify all 17 are present
ls screenshot_*.png | wc -l   # should print 17
# push
cd ../../..
git add notebooks/figures/05/
git commit -m "NB05: add 17 captured screenshots (JB-CR-04 done)"
git push origin 2026-workshop
```

The deploy workflow picks it up automatically (~90 sec). NB05's rendered page swaps the gray "Pending: ..." placeholders for the real images.

## Replacing one shot at a time

Same flow — drop one PNG with the right name, commit, push. The other 16 placeholders stay.
