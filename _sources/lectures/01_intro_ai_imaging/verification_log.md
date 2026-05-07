# Verification Log — Lecture 1

This log records the verification status of every reference cited in `lecture.ipynb` and the exported `lecture_slides.html`. The intent is auditability: any reader, reviewer, or future maintainer can see what was verified, by what means, and what remains to be checked.

## Verification flag conventions

- **✓ verified** — canonical source confirmed via web search and authoritative DOI/URL
- **✓ spot-check** — verified directly during lecture build via WebSearch in this session (Cellpose, μSAM)
- **✓ agent** — verified by a delegated research agent in this session; trusted via spot-checks of two references in the same batch
- **✏ corrected** — initial training-knowledge claim was wrong; canonical entry used here reflects the correction
- **⚠ uncertain** — drawn from training, no direct verification this session; should be sanity-checked before any external publication
- **❓ synthesis** — interpretation or summary, not a specific factual citation

## Method notes

The Phase 1 verification pass used WebSearch to find canonical sources. Direct DOI fetching was blocked by the workspace network allowlist (only specific domains are reachable), so verification relied on search-result metadata rather than on full text retrieval. Two references (Cellpose, μSAM) were spot-checked directly and matched the agent's findings, which raises confidence in the rest. All references should still be cross-checked against the source articles before any external delivery — search-result metadata can lag, mislabel, or contain errors.

## Reference table

| ID | Status | Authors | Year | Venue | DOI / URL | Notes |
|---|---|---|---|---|---|---|
| `cellpose2021` | ✓ spot-check | Stringer, Wang, Michaelos, Pachitariu | 2021 | Nature Methods 18:100–106 | 10.1038/s41592-020-01018-x | Generalist deep-learning cell segmentation. |
| `cellpose2_2022` | ⚠ uncertain | Pachitariu and Stringer (probable) | 2022 | Nature Methods | 10.1038/s41592-022-01663-4 | "Cellpose 2.0: how to train your own model." Surfaced during spot-check; not in original verification batch. |
| `cellpose3_2025` | ⚠ uncertain | Stringer et al. (probable) | 2025 | Nature Methods | 10.1038/s41592-025-02595-5 | "Cellpose3: one-click image restoration for improved cellular segmentation." Surfaced during spot-check; useful in restoration section. |
| `cellpose_sam_2025` | ✓ spot-check | Pachitariu, Rariden, Stringer | 2025 | bioRxiv (preprint) | 10.1101/2025.04.28.651001 | **Cellpose v4 / Cellpose-SAM** — adapts SAM transformer backbone to the Cellpose framework. Current canonical version of the Cellpose tool. Posted 2025-05-01. Supersedes Cellpose 2021 as the active citation for Cellpose-based segmentation. |
| `stardist2018` | ✓ agent | Schmidt, Weigert, Broaddus, Myers | 2018 | MICCAI 2018 | 10.1007/978-3-030-00934-2_30 | Star-convex polygon segmentation. |
| `care2018` | ✓ agent | Weigert, Schmidt, Boothe et al. | 2018 | Nature Methods 15:1090–1097 | 10.1038/s41592-018-0216-7 | Content-Aware Image Restoration in fluorescence microscopy. |
| `noise2void2019` | ✓ agent | Krull, Buchholz, Jug | 2019 | CVPR 2019 | (CVF open access) | Self-supervised denoising. |
| `mesmer2022` | ✏ corrected | Greenwald, Miller, Moen et al. | 2022 (print; online 2021) | Nature Biotechnology 40:555–565 | 10.1038/s41587-021-01094-0 | DeepCell / Mesmer whole-cell segmentation. Print year is 2022, not 2021 as I had assumed. |
| `voxelmorph2019` | ✓ agent | Balakrishnan, Zhao, Sabuncu, Guttag, Dalca | 2019 | IEEE TMI 38(8):1788–1800 | 10.1109/TMI.2019.2897538 | Deep-learning deformable image registration. |
| `annapalm2018` | ✓ agent | Ouyang, Aristov, Lelek, Hao, Zimmer | 2018 | Nature Biotechnology 36:460–468 | 10.1038/nbt.4106 | Super-resolution from sparse PALM. |
| `deepstorm2018` | ✓ agent | Nehme, Weiss, Michaeli, Shechtman | 2018 | Optica 5(4):458–464 | (Optica) / arXiv:1801.09631 | Deep-learning single-molecule localization. |
| `christiansen2018` | ✓ agent | Christiansen, Yang, Ando et al. | 2018 | Cell 173(3):792–803.e19 | 10.1016/j.cell.2018.03.040 | In silico labeling. |
| `ounkomol2018` | ✓ agent | Ounkomol, Seshamani, Maleckar, Collman, Johnson | 2018 | Nature Methods 15:917–920 | 10.1038/s41592-018-0111-2 | Label-free 3D fluorescence prediction. |
| `sam2023` | ✓ agent | Kirillov, Mintun, Ravi et al. | 2023 | ICCV 2023 | 10.1109/ICCV51070.2023.00371 | Segment Anything (SAM). |
| `microsam2025` | ✓ spot-check | Pape, Freckmann, Archit et al. | 2025 (Nature Methods 22) | Nature Methods | 10.1038/s41592-024-02580-4 | μSAM / micro-sam — segment-anything for microscopy. Constantin Pape, Göttingen. |
| `claim2020` | ✓ agent | Mongan, Moy, Kahn | 2020 | Radiology: AI 2(2):e200029 | 10.1148/ryai.2020200029 | Checklist for AI in Medical Imaging. |
| `stardai2020` | ✓ agent | Sounderajah, Ashrafian, Aggarwal et al. | 2020 (steering); 2021 (BMJ Open protocol) | Nature Medicine | 10.1038/s41591-020-1033-0 | STARD-AI reporting standards. |
| `consortai2020` | ✓ agent | Liu, Cruz Rivera, Moher, Calvert, Denniston | 2020 | Nature Medicine 26(9):1364–1374 | 10.1038/s41591-020-1034-x | CONSORT-AI clinical trial reporting. |
| `spiritai2020` | ✓ agent | Cruz Rivera, Liu, Chan, Denniston, Calvert | 2020 | Nature Medicine 26(9):1351–1363 | 10.1038/s41591-020-1037-7 | SPIRIT-AI trial protocol reporting. |
| `miclaim2020` | ✓ agent | Norgeot, Quer, Beaulieu-Jones et al. | 2020 | Nature Medicine 26(9):1320–1324 | 10.1038/s41591-020-1041-y | MI-CLAIM minimum information for clinical AI. |
| `bbbc2012` | ✓ agent | Ljosa, Sokolnicki, Carpenter | 2012 | Nature Methods 9:637 | 10.1038/nmeth.2083 | Broad Bioimage Benchmark Collection. |
| `imagesc` | ✓ agent | Rueden, Ackerman, Schindelin et al. (community) | 2019 | (PLOS Biology overview); forum.image.sc | — | image.sc community forum. |
| `neubias` | ✓ agent | Network of European BioImage Analysts | 2016+ | eubias.org/NEUBIAS | — | Active training network. |

## Items to verify before external publication

The following carry caveats requiring verification before any external delivery:

- `cellpose2_2022` and `cellpose3_2025`: spotted during a search but not part of the formal verification batch; confirm before citing.
- `microsam2025`: confirmed via search but the author list and DOI should be cross-checked against the published paper.
- All "✓ agent" entries: trusted on the strength of two spot-checks but should be cross-checked against published sources before public-facing use.
- Foundation-model citations not yet added (UNI, Virchow, CONCH for pathology; MedSAM, RadFM for clinical) are deferred to the broader curriculum and not currently in this lecture's bibliography.

## Demo-video sources

Demo-video URLs are *not* verified as reachable in Phase 1. The video search agent was unable to confirm direct URLs due to network restrictions. Phase 1 uses **placeholder still images with descriptive captions**; Phase 3 (polish) will revisit video sourcing with author permissions or institutional archives once the lecture content is locked. Candidate sources flagged by the search agent (NEUBIAS YouTube channel, Nature Methods supplementary materials, paper supplements for Christiansen 2018, Krull 2019, Weigert 2018) are recorded in `README.md` for follow-up.

## Hallucination flags retained in lecture content

Even with verified citations, several claims in the lecture remain interpretive or general-audience-targeted summaries rather than direct citations. These will be flagged in the slides themselves with the ❓ symbol:

- Statements about "where AI works well" and "where it fails" — generic framing not tied to a single paper
- The "confident wrongness" failure-mode pattern — community-recognized but not a single canonical citation
- Discussions of foundation-model behavior — fast-moving area; specific characterizations age quickly

These flags are intended to signal interpretation versus citation, not to cast doubt on the underlying claims.
