# Day 2 — ImageJ / Fiji Hands-On

Student package for **Day 2 / Session 1** of the 2026 MABC Image Analysis Workshop.

- **Date:** Friday, May 8, 2026 — 9:00 AM to 12:30 PM
- **Venue:** Annenberg 5-212
- **Instructor:** Nikos Tzavaras
- **Host:** Microscopy and Advanced Bioimaging Core, Icahn School of Medicine at Mount Sinai

This folder bundles everything attendees need for the morning Fiji session: the guided-exercises handout, all referenced sample images, the macros referenced in the handout, and three quick-reference PDFs.

> **Heads-up:** Day 2 is a Fiji hands-on track and is **not** part of the Jupyter Book that renders the rest of this repo (which covers Day 3 — AI for microscopy image analysis). Browse this folder directly via GitHub's web UI, or `git clone` and open the docx/pdf locally.

## Contents

```
day-2-fiji/
├── MABC_IAWorkshop2026_Day2_Fiji_HandsOn.docx   ← editable handout
├── MABC_IAWorkshop2026_Day2_Fiji_HandsOn.pdf    ← read-only handout
├── README.md                                    ← this file
├── README.txt                                   ← same content, plain-text
├── Exercise Images/   (23 files in 7 topic folders)
│   ├── Background/             flat-field correction (HW 2)
│   ├── BitDepth/               bit-depth exploration (Block 1)
│   ├── Cilia Lengths/          cilia segmentation (HW 4)
│   ├── Counting and Segmentation/   cell counting + Labkit dataset (Blocks 4, 4b, HW 3)
│   ├── Double Stained Cells/   multi-channel co-localization (HW 5)
│   ├── DrosophilaCells/        macros block + HW 8
│   └── Westerns/               band quantification (HW 6)
├── Macros/   (7 files)
│   ├── FiltersDemov001.ijm           Block 3.2
│   ├── FiltersDemov201.ijm           Block 3.2 (anisotropic diffusion)
│   ├── StackMeasureMaxvAvevSum.ijm   Block 5 Z-projection caveat
│   ├── EmptyFolderProcessingMacro.ijm  Block 7 stretch + HW 8
│   ├── Cilia2Dlengths.ijm            HW 4
│   ├── Cilia2Dlengths2.ijm           Block 7 stretch (alt name, same content)
│   └── WesternsforElliot.ijm         HW 6
└── Reference/
    ├── Tools.pdf                              Fiji tool icons + drawing/selection reference (5 pp)
    ├── FIJI - Resources & Documentation.pdf   ImageJ wiki, recommended books, topic guides (3 pp)
    └── Keyboard Shortcuts.pdf                 Fiji keyboard-shortcut cheat sheet (3 pp)
```

## Before the workshop — please install

1. **Fiji.** Download and install from <https://imagej.net/software/fiji/downloads>. Use the latest stable build for your operating system.
2. **Update sites.** Open Fiji, go to `Help → Update → Manage Update Sites`, and tick:
    - **Labkit** (used in Block 4b)
    - **3D ImageJ Suite** (used in Block 6)
    Click *Apply*, then restart Fiji.
3. **Download this folder somewhere with read-write access** (e.g., your Desktop). Working from a read-only location can cause Save / macro-save errors mid-exercise.

## During the workshop

The handout walks through eight blocks of guided exercises (Blocks 1–5, 4b Labkit, 6 — 3D, 7 — Macros). Each block opens with a short Goal box and lists numbered steps with the exact menu paths to click in Fiji. Tip and Key-concept callouts highlight things worth remembering.

File paths in the handout map directly to this folder — for example, `BitDepth/Dapi_WhatBitDepth.tif` lives at `Exercise Images/BitDepth/Dapi_WhatBitDepth.tif`.

## Homework / Extra Practice

The last few pages of the handout contain Extra Practice exercises (HW 1–6) for self-paced practice after the workshop. They use the same images and macros bundled here.

(HW 7 and HW 8 from the original BSR2601 source are now done LIVE in Block 6 and Block 7. They remain in the homework section as reference material with additional detail.)

## Questions

Microscopy and Advanced Bioimaging Core
Icahn School of Medicine at Mount Sinai
**Microscopy.core@mssm.edu**

## License

Content (handout, README, instructions) — Creative Commons Attribution 4.0 International (see top-level `CONTENT_LICENSE.md`).
Macros — see source headers; bundled here for workshop use.
Mount Sinai brand assets — used per institutional brand-usage guidelines, not relicensed.
