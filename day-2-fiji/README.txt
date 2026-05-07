2026 Image Analysis Workshop — Day 2 / Session 1
ImageJ / Fiji Hands-On — Guided Exercises
=================================================

Date:  Friday, May 8, 2026 — 9:00 AM to 12:30 PM
Venue: Annenberg 5-212
Instructor: Nikos Tzavaras
Host: Microscopy and Advanced Bioimaging Core
      Icahn School of Medicine at Mount Sinai

What is in this folder
----------------------

MABC_IAWorkshop2026_Day2_Fiji_HandsOn.docx   The guided-exercises handout (editable Word doc).
MABC_IAWorkshop2026_Day2_Fiji_HandsOn.pdf    Same content, PDF (recommended for reading).

Exercise Images/                   All image files referenced in the handout,
                                   grouped by topic:
   Background/                       Flat-field correction (HW 2)
   BitDepth/                         Bit-depth exploration (Block 1)
   Cilia Lengths/                    Cilia segmentation (HW 4)
   Counting and Segmentation/        Cell counting + Labkit dataset
                                     (Blocks 4, 4b, HW 3)
   Double Stained Cells/             Multi-channel co-localization (HW 5)
   DrosophilaCells/                  Macros block (Block 7) + HW 8
   Westerns/                         Band quantification (HW 6)

Macros/                            ImageJ macros referenced in the handout:
   FiltersDemov001.ijm               Block 3.2 — basic filters demo
   FiltersDemov201.ijm               Block 3.2 — anisotropic diffusion
   StackMeasureMaxvAvevSum.ijm       Block 5.2 — Z-projection caveat
   EmptyFolderProcessingMacro.ijm    Block 7 stretch — batch macro template
   Cilia2Dlengths.ijm                HW 4 — cilia pipeline example
   Cilia2Dlengths2.ijm               Block 7 stretch — same content, alt name
   WesternsforElliot.ijm             HW 6 — band quantification

Reference/                         Quick-reference PDFs:
   Tools.pdf                                       Fiji tool icons + drawing/selection tool reference (5 pages).
   FIJI - Resources & Documentation.pdf            ImageJ/Fiji learning hub: links to the official ImageJ wiki, recommended books, topic-specific guides (segmentation, tracking, deconvolution, etc.), Bio-Formats, scripting, and the image.sc forum (3 pages).
   Keyboard Shortcuts.pdf                          Fiji keyboard-shortcut cheat sheet (3 pages).


Before the workshop — please install
------------------------------------

1. Fiji
   Download and install from https://imagej.net/software/fiji/downloads
   Use the latest stable build for your operating system.

2. Update sites (open Fiji once, then enable):
   Help -> Update -> Manage Update Sites
     - tick "Labkit"             (Block 4b)
     - tick "3D ImageJ Suite"    (Block 6)
   Click Apply, then restart Fiji.

3. Copy this entire folder somewhere with read-write access
   (e.g., your Desktop or a workshop folder). Working from a read-only
   location can cause Save / macro-save errors mid-exercise.


During the workshop
-------------------

The handout walks you through eight blocks of guided exercises. Each block
opens with a short Goal box and lists numbered steps with the exact menu
paths to click in Fiji. Tip and Key-concept callouts highlight things worth
remembering.

Files referenced in the handout live exactly where the handout names them
(e.g., "BitDepth/Dapi_WhatBitDepth.tif" maps to
"Exercise Images/BitDepth/Dapi_WhatBitDepth.tif" inside this folder).


Homework section
----------------

The last few pages of the handout contain Extra Practice exercises (HW 1-6)
designed for self-paced practice after the workshop. They use the same
images and macros bundled here.

(HW 7 and HW 8 from the original BSR2601 set are now done LIVE in Block 6
and Block 7. They remain in the homework section as reference material.)


Questions / contact
-------------------
Microscopy and Advanced Bioimaging Core
Icahn School of Medicine at Mount Sinai
Microscopy.core@mssm.edu
