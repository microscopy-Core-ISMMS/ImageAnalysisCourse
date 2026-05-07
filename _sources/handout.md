# Workshop Handout

This handout is what you take home from the workshop. Every tool, paper, and platform we mentioned during the day is here, organized so you can follow up on what caught your attention. Bookmark the URL — you'll want to come back to this.

This is a companion to the [Resources page](resources), which is the canonical, more detailed catalog. The handout is the lighter-weight take-home version.

**Print or save the URL of this page now.** Workshop logistics may not be there for you tomorrow morning when you have a question.

## Where to ask questions after today

If you take only one URL away from the workshop:

- **[image.sc forum](https://forum.image.sc)** — community Q&A across all major bioimage tools (Fiji, napari, ilastik, scikit-image, CellProfiler, and many more). Search before posting; if you don't find an answer, ask.

## Where to start exploring next

Pick one based on where you want to go:

- **You want to try methods on Colab without installing anything:** [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) — Colab notebooks for the major DL microscopy methods.
- **You want to run methods locally or on HPC:** [DL4MicEverywhere](https://github.com/HenriquesLab/DL4MicEverywhere) — containerized successor with broader coverage.
- **You want pretrained models you can browse:** [BioImage Model Zoo](https://github.com/bioimage-io/bioimage.io).
- **You want training material organized by topic:** [NEUBIAS training resources](https://github.com/NEUBIAS/training-resources).

## Today's workshop tools

What we used in the labs:

- **[Cellpose / Cellpose-SAM](https://github.com/MouseLand/cellpose)** — current v4 of the generalist cell segmentation tool. Used in Lab 1.
- **[Noise2Void](https://github.com/juglab/n2v)** — self-supervised denoising. Used in Lab 3 option A.
- **[μSAM (micro-sam)](https://github.com/computational-cell-analytics/micro-sam)** — microscopy-tuned Segment Anything. Used in Lab 3 option B.
- **[scikit-image](https://github.com/scikit-image/scikit-image)** — Python library for classical image processing.

## Reading list

Papers in order they were referenced in the lecture. Citation format: first author, year, venue, doi.

### Section 2 — Task taxonomy

- **Cellpose** — Stringer et al., 2021, *Nature Methods* 18:100–106. doi: [10.1038/s41592-020-01018-x](https://doi.org/10.1038/s41592-020-01018-x)
- **Cellpose-SAM** (current v4) — Pachitariu, Rariden, Stringer, 2025, *bioRxiv* preprint. doi: [10.1101/2025.04.28.651001](https://doi.org/10.1101/2025.04.28.651001)
- **StarDist** — Schmidt, Weigert, Broaddus, Myers, 2018, *MICCAI*. doi: [10.1007/978-3-030-00934-2_30](https://doi.org/10.1007/978-3-030-00934-2_30)
- **Mesmer / DeepCell** — Greenwald et al., 2022, *Nature Biotechnology* 40:555–565. doi: [10.1038/s41587-021-01094-0](https://doi.org/10.1038/s41587-021-01094-0)
- **Segment Anything (SAM)** — Kirillov et al., 2023, *ICCV*. doi: [10.1109/ICCV51070.2023.00371](https://doi.org/10.1109/ICCV51070.2023.00371)
- **μSAM (micro-sam)** — Pape et al., 2025, *Nature Methods* 22:579–591. doi: [10.1038/s41592-024-02580-4](https://doi.org/10.1038/s41592-024-02580-4)
- **CARE** — Weigert et al., 2018, *Nature Methods* 15:1090–1097. doi: [10.1038/s41592-018-0216-7](https://doi.org/10.1038/s41592-018-0216-7)
- **Noise2Void** — Krull, Buchholz, Jug, 2019, *CVPR*.
- **ANNA-PALM** — Ouyang et al., 2018, *Nature Biotechnology* 36:460–468. doi: [10.1038/nbt.4106](https://doi.org/10.1038/nbt.4106)
- **Deep-STORM** — Nehme, Weiss, Michaeli, Shechtman, 2018, *Optica* 5:458–464.
- **In silico labeling** — Christiansen et al., 2018, *Cell* 173:792–803. doi: [10.1016/j.cell.2018.03.040](https://doi.org/10.1016/j.cell.2018.03.040)
- **Label-free 3D fluorescence prediction** — Ounkomol et al., 2018, *Nature Methods* 15:917–920. doi: [10.1038/s41592-018-0111-2](https://doi.org/10.1038/s41592-018-0111-2)
- **VoxelMorph** — Balakrishnan, Zhao, Sabuncu, Guttag, Dalca, 2019, *IEEE TMI* 38:1788–1800. doi: [10.1109/TMI.2019.2897538](https://doi.org/10.1109/TMI.2019.2897538)

### Section 5 and 6 — Validation, reproducibility, reporting

- **CLAIM** — Mongan, Moy, Kahn, 2020, *Radiology: AI* 2(2):e200029. doi: [10.1148/ryai.2020200029](https://doi.org/10.1148/ryai.2020200029)
- **CONSORT-AI** — Liu et al., 2020, *Nature Medicine* 26:1364–1374. doi: [10.1038/s41591-020-1034-x](https://doi.org/10.1038/s41591-020-1034-x)
- **SPIRIT-AI** — Cruz Rivera et al., 2020, *Nature Medicine* 26:1351–1363. doi: [10.1038/s41591-020-1037-7](https://doi.org/10.1038/s41591-020-1037-7)
- **STARD-AI** — Sounderajah et al., 2020, *Nature Medicine*. doi: [10.1038/s41591-020-1033-0](https://doi.org/10.1038/s41591-020-1033-0)
- **MI-CLAIM** — Norgeot et al., 2020, *Nature Medicine* 26:1320–1324. doi: [10.1038/s41591-020-1041-y](https://doi.org/10.1038/s41591-020-1041-y)

### Community platforms

- **ZeroCostDL4Mic** — von Chamier et al., 2021, *Nature Communications*. doi: [10.1038/s41467-021-22518-0](https://doi.org/10.1038/s41467-021-22518-0)
- **DL4MicEverywhere** — Hidalgo-Cenalmor et al., 2024, *Nature Methods*. doi: [10.1038/s41592-024-02295-6](https://doi.org/10.1038/s41592-024-02295-6)
- **BBBC** — Ljosa, Sokolnicki, Carpenter, 2012, *Nature Methods* 9:637. doi: [10.1038/nmeth.2083](https://doi.org/10.1038/nmeth.2083)

### Other tools and platforms

- **ilastik** — Berg et al., 2019, *Nature Methods* 16:1226–1232. doi: [10.1038/s41592-019-0582-9](https://doi.org/10.1038/s41592-019-0582-9)
- **deepImageJ** — Gómez-de-Mariscal et al., 2021, *Nature Methods* 18:1192–1195. doi: [10.1038/s41592-021-01262-9](https://doi.org/10.1038/s41592-021-01262-9)
- **BigStitcher** — Hörl et al., 2019, *Nature Methods* 16:870–874. doi: [10.1038/s41592-019-0501-0](https://doi.org/10.1038/s41592-019-0501-0)
- **Mastodon / MaMuT** — Wolff, Tinevez, Pietzsch et al., 2018, *eLife* 7:e34410. doi: [10.7554/eLife.34410](https://doi.org/10.7554/eLife.34410)
- **QuPath** — Bankhead et al., 2017, *Scientific Reports* 7:16878. doi: [10.1038/s41598-017-17204-5](https://doi.org/10.1038/s41598-017-17204-5)
- **nnU-Net** — Isensee et al., 2021, *Nature Methods* 18:203–211. doi: [10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z)
- **TotalSegmentator** — Wasserthal et al., 2023, *Radiology: AI*. doi: [10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024)
- **U-Net** — Ronneberger, Fischer, Brox, 2015, *MICCAI*. arXiv: 1505.04597
- **ClearMap** — Renier, Adams, Kirst et al., 2016, *Cell* 165:1789–1802. doi: [10.1016/j.cell.2016.05.007](https://doi.org/10.1016/j.cell.2016.05.007)

## Tools and platforms — quick reference

Grouped for fast scanning:

### Segmentation
[Cellpose](https://github.com/MouseLand/cellpose) · [StarDist](https://github.com/stardist/stardist) · [Mesmer / DeepCell](https://github.com/vanvalenlab/deepcell-tf) · [μSAM](https://github.com/computational-cell-analytics/micro-sam) · [Segment Anything](https://github.com/facebookresearch/segment-anything) · [AICS Segmentation](https://github.com/AllenCell/aics-segmentation) · [ilastik](https://github.com/ilastik) · [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

### Restoration / denoising
[Noise2Void](https://github.com/juglab/n2v) · [CSBDeep / CARE](https://github.com/CSBDeep/CSBDeep) · [Cellpose 3](https://github.com/MouseLand/cellpose)

### Registration
[VoxelMorph](https://github.com/voxelmorph/voxelmorph) · [BigStitcher](https://github.com/JaneliaSciComp/BigStitcher) · [ClearMap](https://github.com/ChristophKirst/ClearMap)

### Tracking
[Mastodon](https://github.com/mastodon-sc/mastodon) · TrackMate (in Fiji)

### Pathology / clinical
[QuPath](https://github.com/qupath/qupath) · [MONAI](https://github.com/Project-MONAI/MONAI) · [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

### Visualization and platforms
[napari](https://github.com/napari/napari) · [Fiji / ImageJ](https://imagej.net/software/fiji/) · [scikit-image](https://github.com/scikit-image/scikit-image) · [deepImageJ](https://github.com/deepimagej/deepimagej-plugin) · [BioImage Model Zoo](https://github.com/bioimage-io/bioimage.io)

### Vendor format support
[Bio-Formats](https://github.com/ome/bioformats) · [Zeiss pylibCZIrw](https://github.com/ZEISS/pylibczirw) · [AICSImageIO](https://github.com/AllenCellModeling/aicsimageio)

## Datasets and atlases

- **[BBBC — Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/)** — curated benchmark datasets
- **[Allen Brain Atlas](https://portal.brain-map.org)** — reference atlases of mouse and human brain
- **[IDR — Image Data Resource](https://idr.openmicroscopy.org)** — public reference imaging datasets
- **[BioImage Archive (EMBL-EBI)](https://www.ebi.ac.uk/bioimage-archive/)** — public archive for biological image data
- **[MICrONS Explorer](https://www.microns-explorer.org)** — EM connectomics dataset
- **[Allen Cell Image Library](http://www.cellimagelibrary.org)** — curated cell biology images

## Communities to join

- **[image.sc forum](https://forum.image.sc)** — the bioimage analysis community Q&A
- **[NEUBIAS training resources](https://github.com/NEUBIAS/training-resources)** — open educational material; modular 30-min lessons
- **[Henriques Lab](https://henriqueslab.org/)** — ZeroCostDL4Mic, DL4MicEverywhere, and related democratization work
- **[CZ Imaging Institute](https://github.com/czimaginginstitute)** — funding and tooling for the AI-imaging community

## Continuing education

If you want to go deeper on the broader framework this workshop is part of:

- The **[Personas document](https://github.com/your-org/your-repo)** — who this curriculum is designed for and why
- The **[Curriculum framework document](https://github.com/your-org/your-repo)** — full module catalog (60 modules across 5 tiers) with persona pathways

(URLs to be added when the workshop repo is published.)

## A request

If you find errors in this handout — broken links, wrong citations, outdated information — please report them via the workshop's GitHub issues page. The bioimage AI ecosystem moves fast and we'd rather fix things quickly than ship stale material.

## Acknowledgments

This handout pulls together the work of many community members and labs. Specific tool and method credits are in the citations above. The handout itself is licensed CC-BY-4.0 — reuse and adapt freely with attribution.
