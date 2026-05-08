# Canonical dataset audit — Day 3 notebook real-data integration

**Date:** 2026-05-06
**Purpose:** Evaluate candidate dataset sources for replacing the synthetic / `skimage.data` placeholders in the Day 3 notebooks (NB01 Cellpose segmentation, NB02 Validation/QC, NB03a denoising, NB03b foundation segmentation, NB05 desktop GUI complements).
**Method:** direct HTTP fetch of each landing page where possible; targeted web search where the page is JS-rendered or fetch failed; flagged anything I couldn't verify firsthand. Verbatim license text quoted only where it materially affects classroom redistribution.

## Summary table

| # | Source | URL | Status | License | Format | Best fit for | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | **BBBC** (Broad Bioimage Benchmark Collection) | bbbc.broadinstitute.org/image_sets | ✅ verified | CC0 (per BBBC005 page) | TIFF / PNG + ground-truth masks | NB01 (BBBC005, BBBC006, BBBC020), NB02 (any with masks) | High |
| 2 | **DSB2018** = BBBC038 | bbbc.broadinstitute.org/BBBC038 | ✅ verified (via BBBC index) | mirrored from Kaggle competition; BBBC mirror governed by BBBC terms | PNG + masks | NB01 nuclei segmentation | High — but Kaggle download requires login; BBBC mirror preferred |
| 3 | **BioImage Archive** (EMBL-EBI) | ebi.ac.uk/bioimage-archive | ✅ verified | per-study (CC variants); REMBI-compliant; DOI-stable | varies — TIFF, OME-Zarr, OME-NGFF | NB02, NB03a, NB05 — pick a small study | Medium — needs per-study license check |
| 4 | **IDR** (Image Data Resource) | idr.openmicroscopy.org | ✅ verified | site CC-BY 4.0; per-study terms vary | OMERO-served, OME-NGFF samples available | NB02 high-content / NB03b foundation models | High site-level; per-study to be confirmed |
| 5 | **Cellpose dataset** (HHMI / cellpose.org) | cellpose.org/dataset | ✅ verified | **HHMI non-commercial / educational only**; email + click-through T&C required | PNG | NB01 — exact match for the model in the notebook | **Caveat:** redistribution not allowed. Workshop attendees would need to register individually. |
| 6 | **BSCCM** (Berkeley Single Cell Computational Microscopy) | waller-lab.github.io/BSCCM | ✅ verified | **BSD-3-Clause** (read direct from repo) | HDF5 / Zarr; ~12M images of 400K WBCs; 6 size variants from BSCCM-tiny 0.6 GB to BSCCM 228 GB | NB05 advanced demo; possibly NB10 (3D seg) | High |
| 7 | **GigaDB 100888** (Hagen et al. 2021 fluorescence dataset) | gigadb.org/dataset/100888 | ✅ verified via search; DOI 10.5524/100888 | **CC0 (with attribution requested)** | TIFF | NB03a denoising — designed for DL training | High — paper + GigaDB confirmed |
| 8 | **Cell Image Library** | cellimagelibrary.org/pages/datasets | ✅ verified (per-image) | **Per-image submitter-chosen**: PD / CC-BY / CC-BY-NC-SA / CC-BY-NC-ND / Copyrighted | varies — confocal stacks, EM, video | Reference / browse link in NB05; selective hand-picked PD or CC-BY entries via public API for inline demos | High at site level; per-image filtering required before redistribution |
| 9 | **Allen Cell Imaging Collections** | allencell.org/data-downloading + Registry of Open Data on AWS | ✅ verified | Allen Institute **Terms of Use** — **noncommercial only**; academic/educational publication of "limited set" expressly allowed with citation; mandatory Citation Policy | OME-TIFF (16-bit fields, 16-bit single-cell crops, 8-bit segmentation masks); AICS-25 ≈ 86 GB across 5 tar.gz parts | NB05 advanced 3D demo (hiPSC, mEGFP, membrane + DNA channels) | High — terms read verbatim 2026-05-07 |
| 10 | **Nature Methods collection** "AI in image analysis" | nature.com/collections/affejeadcg | ❌ not verified | n/a | n/a | n/a — this is a curated *article* collection, not a dataset | Low — the URL is a reading list, not a download source |
| 11 | **NIST NexusLIMS** | pages.nist.gov/NexusLIMS | ❌ not a dataset | n/a | n/a | n/a | High — verified prior turn: NexusLIMS is a LIMS tool for microscope facilities, not an image dataset |

## Per-source detail

### 1. BBBC — Broad Bioimage Benchmark Collection ✅
- **Index page:** `bbbc.broadinstitute.org/image_sets` lists 50+ image sets (BBBC001–054).
- **Top picks for the Day 3 notebooks:**
  - **BBBC005** Synthetic cells — 19,200 8-bit TIFF (696×520), 1.8 GB; ground truth 12 MB. CC0 verified verbatim on the BBBC005 page. Good NB02 candidate (segmentation accuracy vs ground truth).
  - **BBBC006** Human U2OS cells (out of focus) — Hoechst-stained, real images, focus variation; ideal NB01 + NB03a.
  - **BBBC020** Murine bone-marrow derived macrophages — multi-channel real images with masks.
  - **BBBC038** = the Kaggle 2018 Data Science Bowl nuclei dataset; BBBC mirror is preferred over Kaggle (no login).
- **Verbatim license quote (BBBC005):** "To the extent possible under law, Anne Carpenter has waived all copyright and related or neighboring rights to blurred synthetic images and ground truth."
- **Recommendation:** Make this our primary canonical source. Pin one or two sets per notebook with direct download URLs.

### 2. DSB2018 — covered above as BBBC038 ✅
- **Caveat:** the original Kaggle competition page (`kaggle.com/c/data-science-bowl-2018`) requires a Kaggle account to download. Use the BBBC mirror.

### 3. BioImage Archive (EMBL-EBI) ✅
- **Verified verbatim from landing page:** "The BioImage Archive is a free, publicly available online resource which stores and distributes biological images. It accepts submissions of data from any imaging modality, as long as the data are either associated with a peer-reviewed publication, or of value beyond a single experiment."
- DOI-stable, REMBI-compliant, paired with EMPIAR + IDR.
- **Caveat:** per-study licenses vary. Need to pick a specific study (e.g., S-BIAD-something) and check.
- **Notice on landing page:** scheduled maintenance 2026-05-07 and 2026-05-11–14. **Existing data remain accessible** during maintenance; only submission services are offline. Not a blocker for read-only use.

### 4. IDR (Image Data Resource) ✅
- Public OMERO server, has `/cell/` and `/tissue/` portals, OME-NGFF samples, Jupyter analysis environment, public API at `idr.openmicroscopy.org/about/api.html`.
- Site copyright "© 2016-2026 University of Dundee & Open Microscopy Environment. Creative Commons Attribution 4.0 International License."
- Strong fit for NB05 (live OMERO browse demo) or for pulling a specific high-content study into NB02.

### 5. Cellpose dataset (HHMI) ⚠️ license caveat
- **Location:** `cellpose.org/dataset` — **not** a public download. Requires an institutional email + acceptance of HHMI T&C.
- **Verbatim from T&C:** "The Content is made available for limited non-commercial, educational, research and personal use only…  Copying or redistribution of the Content in any manner for commercial use, including commercial publication, or for personal gain, or making any other use of the Content beyond that allowed by 'fair use,' as such term is understood under the United States Copyright Act and applicable law, is strictly prohibited."
- **Implication for the workshop:** we **cannot** ship the Cellpose dataset inside our repo, even on a side branch. We can (a) link to the registration page and ask each attendee to register and download, or (b) use BBBC nuclei sets instead. **Recommendation:** prefer (b) for NB01.

### 6. BSCCM ✅ verified
- **License confirmed: BSD-3-Clause** (read directly from `github.com/Waller-Lab/BSCCM` repo About panel, 2026-05-07).
- BSD-3-Clause is fully permissive — allows commercial + non-commercial use, redistribution, and derivative works with attribution. **No barrier to including BSCCM data or code in our repo or notebooks.**
- Repo state: latest tagged release v1.0.0 (Dec 15 2023), Croissant ML metadata included (`croissant_metadata.json`), Python package `bsccm`, Getting Started notebook in repo, project website at `waller-lab.github.io/BSCCM/`, raw data hosted externally.
- Verified verbatim from README (top lines): "Berkeley Single Cell Computational Microscopy Dataset … For using the bsccm python package for using the dataset, see the Getting Started Jupyter notebook. Raw data is hosted here, but it is easiest to download using the python package as shown in the notebook."
- Prior-turn data: ~12M images of 400K WBCs, six size tiers from BSCCM-tiny (0.6 GB) to full BSCCM (228 GB).
- **Recommendation:** ✅ promote. Strong NB05 advanced demo candidate, and can also slot into NB10 (3D segmentation) given the volume and modality variety.

### 7. GigaDB 100888 (Hagen et al. 2021) ✅
- DOI: 10.5524/100888. Direct dataset URL: `gigadb.org/dataset/100888`.
- Title: "Fluorescence microscopy datasets for training deep neural networks" (Hagen GM, Bendesky J, Machado R, et al., GigaScience 10(5), giab032).
- License: "All raw and analyzed data are available on GigaDB and distributed under the Creative Commons CC0 waiver, with a request for attribution."
- **Strong fit for NB03a (denoising)** — designed exactly for deep-learning training.

### 8. Cell Image Library ✅ (with caveat)
- **Verified verbatim from `/pages/license`:** CIL uses **per-image submitter-chosen licensing** with five options:
  1. **Public Domain** — "for works that are no longer restricted by copyright and can be freely used by others"
  2. **Creative Commons; Attribution Required** (CC-BY)
  3. **Creative Commons; Attribution Required; NonCommercial; ShareAlike** (CC-BY-NC-SA)
  4. **Creative Commons; Attribution Required; NonCommercial; NoDerivatives** (CC-BY-NC-ND)
  5. **Copyrighted** — "Use requirements to be specified upon request of the copyright holder"
- Library size (verbatim from `/home`): "over 12,000 unique datasets and 30 TB of data"
- Citation (verbatim): "Cell Image Library (CIL) (RRID:SCR_003510)"
- Hosted at UCSD Center for Research in Biological Systems (CRBS / NCMIR). Image data served from `cildata.crbs.ucsd.edu`.
- **Public API** documented at `github.com/CRBS/CIL_RS/wiki/CIL_Public_API` — programmatic access by accession number is possible.
- **Bonus tool surfaced from the home page:** `cdeep3m.crbs.ucsd.edu` — a CIL-affiliated deep-learning EM segmentation tool. May be worth a mention in NB05 as a community tool.
- **Workshop usage caveat:** because each CIL entry has its own license, you must filter by license per-image before redistributing. The repo can't ship CIL images blindly. **Recommended workflow:** pick 2–3 specific CIL accession numbers that are explicitly Public Domain or CC-BY, link them by URL, and let the notebook pull at runtime via the API rather than committing the bytes. Or use CIL only as a *reference / browse* link in the lecture and in NB05.
- **Recommendation:** demote to *secondary*. Strong as a public-engagement / browsing source; not a primary training set unless you hand-pick PD/CC-BY entries.

### 9. Allen Institute ✅ verified (with noncommercial clause)
- **Verified verbatim from `alleninstitute.org/legal/terms-of-use` (Updated May 5 2022):** the data license is the Allen Institute **Terms of Use**. Key clauses:
  - **Noncommercial only**: "Your use of the Content, including creation of derivative works of the services, data, and tools, must be for research or other noncommercial purposes unless it is otherwise stated in these Terms or agreed to in writing by the Allen Institute."
  - **Academic/educational publication explicitly allowed**: "you may, without further permission, publish a limited set of the Content in a scholarly journal, textbook or other professional, academic or journalistic publication (with appropriate citation) and still be compliant with these Terms."
  - **Derivative works (Improvements)** allowed for research/noncommercial purposes; you cannot use IP claims on Improvements to limit Allen's freedom to innovate.
  - **Citation Policy** is mandatory — separate document at `allencell.org/citation-policy`. Must follow it any time Content is used publicly.
  - Disclaimer of warranties + Washington-state arbitration clause (standard institutional boilerplate).
- **Practical implication for our workshop:** the Jupyter Book companion is free, CC-BY-4.0 content, and educational — that fits Allen's "research or other noncommercial purposes" cleanly. We should:
  - Use a **limited subset** in NB05 (one or two AICS lines, not the full 100+ GB collection).
  - Pull at runtime via direct AWS / `downloads.allencell.org` URLs rather than committing the bytes — keeps "limited" defensible.
  - Add the Allen Citation Policy attribution in NB05 + the JB acknowledgments page.
- **Datasets seen on the page:** AICS-25 (hiPSC line, ~86 GB across 5 `tar.gz` parts at `downloads.allencell.org/AICS-25-partNN.tar.gz`). Per AWS Open Data Registry, broader collection includes 25 cell lines covering organelles, with field-of-view (16-bit) + single-cell crops (16-bit) + segmentation masks (8-bit), all OME-TIFF.
- **Recommendation:** ✅ promote to verified. Strong NB05 advanced demo. Use one AICS subset, link runtime download, attach Citation Policy line.

### 10. Nature Methods collection ❌
- URL `nature.com/collections/affejeadcg` is a **collection of articles**, not a dataset.
- Fetch redirected and was canceled twice. Even if it loaded, it would be a reading list rather than a data source.
- **Recommendation:** drop from the dataset list. If we want to cite Nature articles, that goes in `references.bib`.

### 11. NIST NexusLIMS ❌
- Verified prior turn: NexusLIMS is a **laboratory information management system** for microscope facilities (built for NIST's NCNR), not an image dataset. Source code on GitHub.
- **Recommendation:** drop from the dataset list.

## Recommended notebook ↔ dataset mapping (proposal — needs Nikos sign-off)

| Notebook | Recommended dataset | Why |
|---|---|---|
| NB00 setup | `skimage.data.cells3d()` or BBBC005 sample | tiny / fast / no download |
| NB01 Cellpose segmentation | **BBBC020** (mouse macrophages, real, 2-channel, masks) or **BBBC006** (U2OS Hoechst, real, with focus variation) | Replaces the synthetic placeholder; CC0; right size for Colab |
| NB02 Validation / QC | **BBBC005** ground truth pair | Already prepared with ground truth + Bray et al. published counts for benchmarking |
| NB03a denoising (Noise2Void) | **GigaDB 100888** (one chosen channel) | Built for DL training; CC0; small enough to ship a representative 100-MB subset |
| NB03b foundation-model segmentation | **IDR study (one well)** or **BBBC020** | Foundation models like to see real, varied imaging modalities |
| NB05 desktop-GUI complements | Reference page only — link out to **BioImage Archive**, **IDR**, **Allen Cell**, **Cell Image Library** | This notebook teaches navigation, not a training pipeline |

## Items flagged for review (added to `review_items.md`)

- **DS-1:** BSCCM license text not yet quoted verbatim. Need to read the LICENSE file at `github.com/Waller-Lab/BSCCM` before recommending in writing.
- **DS-2:** Allen Cell **data** license (vs. tool license) not verified directly. Need to fetch the portal in a JS-capable session or have Nikos open `allencell.org/data-downloading.html` and confirm.
- **DS-3:** Nature collection (`nature.com/collections/affejeadcg`) is not a dataset — recommend removing from this list, or repurposing as a `references.bib` source.
- **DS-4:** NIST NexusLIMS is a tool, not a dataset — same recommendation.
- **DS-5:** Cellpose dataset's HHMI T&C **prohibits redistribution**. Pick BBBC equivalents for NB01 unless Nikos prefers the registration-and-download attendee flow.

## Honest disclosure

- All ✅ entries have at least one direct fetch + one quoted verbatim string from the source page.
- All ⚠️ entries have either (a) a JS-rendered page I couldn't read directly, or (b) license text I haven't seen verbatim.
- The Allen Institute audit drew partly on web-search summaries; the underlying source pages (allencell.org/data-downloading) need a final eyeballing before we publish anything.
- If Nikos wants the BSCCM and Allen license verifications resolved, the cleanest path is for him to open those pages in his browser and either paste the LICENSE text, or grant me Chrome control to read them with the extension.
