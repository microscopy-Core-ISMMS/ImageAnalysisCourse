# Resources

A curated survey of open-source projects, tools, and datasets relevant to AI for scientific image analysis. Organized by institution and topic. Every entry has a GitHub URL where one exists. Verification flags follow the same convention as the lecture's bibliography (✓ verified, ⚠ uncertain, ✏ corrected).

This page is the workshop's distributed-handout resource: attendees receive the URL at workshop close. It is also linked from the lecture's "Resources" appendix.

**Status: Phase 2A draft for stakeholder review.** Citations carry verification flags. Anything tagged ⚠ should be confirmed before external delivery — the field changes fast and metadata can drift.

## Quick legend

- ✓ — verified via web search this session or in Phase 1
- ⚠ — citation or URL surfaced by automated research; should be confirmed before publication
- ✏ — corrected from an initial mistaken claim
- ❓ — interpretive or summary; not a specific citation

---

## Allen Institute

### Allen Cell & Structure Segmenter (AICS)
- **GitHub:** [github.com/AllenCell/aics-segmentation](https://github.com/AllenCell/aics-segmentation) — classical workflow
- **GitHub:** [github.com/AllenCell/aics-ml-segmentation](https://github.com/AllenCell/aics-ml-segmentation) — ML workflow
- **What:** Python toolkit for 3D segmentation of intracellular structures in fluorescence microscopy. Combines classical and deep-learning workflows. ⚠
- **License:** BSD/Apache (verify per repo) ⚠
- **Notes:** Includes a napari plugin; companion lookup table covering ~20 reference structures.

### AICSImageIO / bioio
- **GitHub:** [github.com/AllenCellModeling/aicsimageio](https://github.com/AllenCellModeling/aicsimageio) (now in maintenance mode)
- **Successor:** bioio (per agent finding ⚠)
- **What:** Pure-Python image reading/writing across microscopy formats (OME-TIFF, ND2, DV, CZI, LIF, PNG, GIF). ⚠
- **License:** BSD ⚠
- **Notes:** Foundational data-handling layer for the Allen ecosystem; widely used elsewhere.

### napari-allencell-segmenter
- **GitHub:** [github.com/AllenCell/napari-allencell-segmenter](https://github.com/AllenCell/napari-allencell-segmenter) ⚠
- **What:** napari plugin integrating the AICS Segmenter for interactive 3D work in napari.
- **License:** BSD ⚠

### Allen Neural Dynamics
- **GitHub org:** [github.com/AllenNeuralDynamics](https://github.com/AllenNeuralDynamics) ⚠
- **What:** Behavioral imaging, Neuropixels analysis, and neural-dynamics tools for circuit and whole-brain mapping.
- **Notes:** Includes natural-language brain-data navigation tools; secondary to microscopy but useful for neural-imaging audiences.

### OpenScope databook
- **GitHub:** [github.com/AllenInstitute/openscope_databook](https://github.com/AllenInstitute/openscope_databook) ⚠
- **What:** Two-photon calcium imaging and Neuropixels datasets with reproducible analysis workflows.

### Brain Atlas / AllenSDK
- **Additional reference:** AllenSDK (Allen Institute Software Development Kit) is the canonical Python access layer to the Brain Atlas, Cell Types Database, and other Allen resources. ❓
- **What to investigate before workshop:** which AllenSDK modules are most relevant to microscopy AI versus systems neuroscience.

---

## Chan Zuckerberg Initiative (CZI)

### CZ Imaging Institute (CZII)
- **GitHub org:** [github.com/czimaginginstitute](https://github.com/czimaginginstitute) ⚠
- **What:** Emerging hub for foundational AI models, imaging hardware/software, and training materials. The Imaging Institute is recently founded; the GitHub presence is still expanding.
- **Notes:** Relationship to the broader Chan Zuckerberg Biohub Network worth flagging in presenter notes.

### napari (CZI-supported)
- **GitHub:** [github.com/napari/napari](https://github.com/napari/napari) ✓
- **What:** Multi-dimensional image viewer for Python; the de facto interactive analysis platform in modern bioimage workflows.
- **License:** BSD-3-Clause ✓
- **Notes:** ~100+ plugins; CZI is a primary funder. Plugin engine v2 (npe2) is current.

### CZI funding for community projects
CZI funds many of the projects elsewhere on this page (ZeroCostDL4Mic, DL4MicEverywhere, image.sc, NEUBIAS continuations, BioImage Archive, BioImage Model Zoo). Worth flagging in the lecture as "the funding layer behind the ecosystem." ❓

---

## Janelia Research Campus (HHMI) and affiliated

### Cellpose / Cellpose-SAM (MouseLand — Stringer & Pachitariu)
- **GitHub:** [github.com/MouseLand/cellpose](https://github.com/MouseLand/cellpose) ✓
- **What:** Generalist deep-learning cell segmentation. Current version is **Cellpose-SAM (v4)**, adapting the SAM transformer backbone to the original Cellpose framework.
- **Papers:** Stringer et al., *Nature Methods* 2021 ✓ (original); Pachitariu, Rariden, Stringer, *bioRxiv* 2025 (Cellpose-SAM, preprint) ⚠
- **License:** BSD-3-Clause ✓
- **Notes:** Cellpose has gone through Cellpose, Cellpose 2.0 (2022), Cellpose 3 (2025, image restoration), and Cellpose-SAM (2025). Cite the version you use.

### StarDist (Janelia / MPI-CBG / TU Dresden)
- **GitHub:** [github.com/stardist/stardist](https://github.com/stardist/stardist) ⚠
- **Paper:** Schmidt, Weigert, Broaddus, Myers, *MICCAI* 2018 ✓ (✏ correction: agent draft listed Weigert as first author, the original paper has Schmidt first)
- **What:** Instance segmentation via star-convex polygons; 2D and 3D. Canonical for nuclei detection.
- **License:** BSD-3-Clause ⚠
- **Notes:** napari plugin available; broad adoption.

### FlyEM connectome tools
- **GitHub org:** [github.com/janelia-flyem](https://github.com/janelia-flyem) ⚠
- **What:** EM connectome segmentation, evaluation, and visualization. Includes tooling for the Hemibrain and Male Adult Nerve Cord (MANC) datasets.
- **Notes:** Major connectome projects with public data; relevant for EM-imaging audiences.

### BigStitcher
- **GitHub:** [github.com/JaneliaSciComp/BigStitcher](https://github.com/JaneliaSciComp/BigStitcher) ✓
- **Paper:** Hörl et al., *Nature Methods* 2019 ✓ — doi: 10.1038/s41592-019-0501-0
- **What:** Image stitching and registration for large multi-tile, multi-angle microscopy (lightsheet, confocal). Terabyte-scale.
- **License:** GPL ⚠
- **Notes:** Available via Fiji; BigStitcher-Spark for distributed processing. Relevant for the **registration** task (T1.9).

### BigDataViewer (BDV)
- **GitHub:** integrated within BigStitcher and Fiji repositories; main BDV repo at [github.com/bigdataviewer](https://github.com/bigdataviewer) ⚠
- **What:** Interactive visualization framework for large 3D/4D datasets.
- **Notes:** Foundational dependency of Mastodon, BigStitcher, MoBIE.

### Mastodon
- **GitHub:** [github.com/mastodon-sc/mastodon](https://github.com/mastodon-sc/mastodon) ✓
- **Paper:** Wolff, Tinevez, Pietzsch et al., *eLife* **2018** (✏ corrected: agent draft listed 2022; canonical Mastodon/MaMuT paper is 2018, doi: 10.7554/eLife.34410. The 2022 work is the ELEPHANT deep-learning extension)
- **What:** Large-scale tracking and lineage editing for multi-view developmental and cell-biology imaging. Standalone or via Fiji.
- **License:** GPL ⚠
- **Notes:** Joint development with EMBL groups; ELEPHANT extension for deep-learning-assisted tracking.

---

## EMBL (European Molecular Biology Laboratory)

### ilastik
- **GitHub org:** [github.com/ilastik](https://github.com/ilastik) ✓
- **Paper:** Berg et al., *Nature Methods* 2019 ✓ — doi: 10.1038/s41592-019-0582-9
- **What:** Interactive ML for pixel/object classification, tracking, and semi-automated segmentation. Random-forest backend; no coding required.
- **License:** GPL ⚠
- **Notes:** Anna Kreshuk's lab. ImageJ/Fiji integration; Python API. Long-standing community workhorse.

### deepImageJ
- **GitHub:** [github.com/deepimagej/deepimagej-plugin](https://github.com/deepimagej/deepimagej-plugin) ✓
- **Paper:** Gómez-de-Mariscal et al., *Nature Methods* 2021 ✓ — doi: 10.1038/s41592-021-01262-9
- **What:** ImageJ/Fiji plugin to run deep-learning models without coding (classification, segmentation, denoising, super-resolution). BioImage Model Zoo–compatible.
- **License:** AGPL ⚠

### BioImage Model Zoo (bioimage.io)
- **GitHub:** [github.com/bioimage-io/bioimage.io](https://github.com/bioimage-io/bioimage.io) ✓
- **Paper:** Ouyang et al., 2022, **bioRxiv preprint** (✏ corrected: agent draft listed *Nature Methods*; the canonical reference is a bioRxiv preprint as of verification — confirm publication status before external use)
- **What:** Standardized repository of pre-trained deep-learning models for bioimage analysis. Metadata, versioning, DOI.
- **License:** Various per model (CC-BY, MIT, etc.)
- **Notes:** Hosted on Zenodo; integrates with ilastik, deepImageJ, StarDist, CSBDeep, others.

### MoBIE
- **GitHub:** [github.com/mobie](https://github.com/mobie) ⚠
- **What:** Visualization platform for multi-omics and volumetric data; OME-NGFF/Zarr integration; Fiji plugin.
- **Notes:** COVID-EM and other public projects.

### BioImage Archive (EMBL-EBI)
- **Web:** [www.ebi.ac.uk/bioimage-archive](https://www.ebi.ac.uk/bioimage-archive) ⚠
- **What:** Public archive for biological image data — equivalent to GenBank/PDB for imaging. Submissions assigned accession numbers; growing rapidly.
- **Relevance:** Data-storage and reference-dataset infrastructure.

---

## Blue Brain Project / Open Brain Institute (EPFL)

### NeuroM
- **GitHub:** [github.com/BlueBrain/NeuroM](https://github.com/BlueBrain/NeuroM) ⚠
- **What:** Python toolkit for analysis and processing of neuron morphologies. Feature extraction and validation.
- **License:** LGPLv3 ⚠
- **Notes:** Foundational for computational neuroscience. Migration to OpenBrainInstitute reportedly underway. ⚠

### SONATA format
- **GitHub:** [github.com/BlueBrain/sonata](https://github.com/BlueBrain/sonata) ⚠
- **What:** Scalable open data format for multiscale neuronal network models and simulation output. Co-developed with Allen Institute.

### BluePyOpt
- **GitHub:** [github.com/BlueBrain/BluePyOpt](https://github.com/BlueBrain/BluePyOpt) ✓
- **Paper:** Van Geit et al., *Frontiers in Neuroinformatics* 2016 ✓ — doi: 10.3389/fninf.2016.00017
- **What:** Parameter optimization for computational neuron models.
- **Relevance:** Lower for AI-imaging specifically; included for completeness.

---

## Community platforms and Colab-friendly toolboxes

### ZeroCostDL4Mic
- **GitHub:** [github.com/HenriquesLab/ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) ✓
- **Paper:** von Chamier et al., *Nature Communications* 2021 ✓ — doi: 10.1038/s41467-021-22518-0
- **What:** Google Colab notebooks for deep-learning microscopy methods (U-Net, StarDist, YOLOv2, CARE, Noise2Void, Deep-STORM, label-free prediction / fnet, pix2pix, CycleGAN). No coding required.
- **License:** MIT ⚠
- **Notes:** The community-standard entry point for non-coders.

### DL4MicEverywhere
- **GitHub:** [github.com/HenriquesLab/DL4MicEverywhere](https://github.com/HenriquesLab/DL4MicEverywhere) ✓
- **Paper:** Hidalgo-Cenalmor et al., *Nature Methods* 2024 ✓ — doi: 10.1038/s41592-024-02295-6 (✏ correction: agent draft listed Fuster-Barceló as first author)
- **What:** Containerized successor to ZeroCostDL4Mic. Docker images that run on Colab, local, HPC, or cloud. Adds image registration, 3D SMLM, temporal/spatial upsampling, image generation. ~25 notebooks.
- **License:** CC-BY-4.0 ✓
- **Notes:** AI4Life-affiliated. Probably the right local-runtime path for workshop attendees who want to continue past Colab.

### NEUBIAS training resources
- **GitHub:** [github.com/NEUBIAS/training-resources](https://github.com/NEUBIAS/training-resources) ⚠
- **What:** Modular open educational resource for bioimage analysis. 30-minute lessons across ImageJ, Python, Galaxy, and more.
- **License:** CC-BY ⚠
- **Notes:** Community-driven; NEUBIAS Defragmentation training-school materials available.

### image.sc forum
- **Web:** [forum.image.sc](https://forum.image.sc) ✓
- **What:** Community forum for bioimage analysis; the canonical place to ask questions. Multiple software communities (Fiji, CellProfiler, napari, ilastik, scikit-image, etc.) share the platform.
- **Notes:** The single most useful URL on this page for an attendee continuing self-directed learning.

---

## Foundation models and major DL building blocks

### Segment Anything (SAM)
- **GitHub:** [github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) ⚠
- **Paper:** Kirillov et al., *ICCV* 2023 ✓
- **What:** Vision foundation model for general segmentation. Foundation for downstream microscopy/biomedical adaptations.
- **License:** Apache 2.0 ⚠
- **Notes:** SAM 2 and reportedly SAM 3 released; verify current state before workshop. ⚠

### Segment Anything for Microscopy (μSAM / micro-sam)
- **GitHub:** [github.com/computational-cell-analytics/micro-sam](https://github.com/computational-cell-analytics/micro-sam) ✓
- **Paper:** Pape et al., *Nature Methods* 2025 ✓ (✏ correction: agent draft listed Wolny as first author)
- **What:** SAM fine-tuned on >17,000 microscopy images. Light and electron microscopy; napari plugin.
- **License:** CC-BY-4.0 ⚠
- **Notes:** Constantin Pape, Göttingen. Used in Lab 3 option B.

### U-Net (canonical reference)
- **GitHub example:** [github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) ⚠ (one of many implementations)
- **Paper:** Ronneberger, Fischer, Brox, *MICCAI* 2015 ⚠
- **What:** The seminal encoder-decoder architecture for biomedical image segmentation. Numerous implementations and variants.
- **Notes:** Cite the original paper; there is no canonical reference implementation.

### Cellpose-SAM (current Cellpose)
See [Janelia / MouseLand](#cellpose--cellpose-sam-mouseland--stringer--pachitariu) above.

---

## Restoration and self-supervised denoising

### CARE (Content-Aware Image Restoration)
- **GitHub:** *via CSBDeep / cstring* — verify before workshop ⚠
- **Paper:** Weigert et al., *Nature Methods* 2018 ✓
- **What:** Supervised content-aware restoration for fluorescence microscopy.
- **License:** BSD-3-Clause ⚠

### Noise2Void
- **GitHub:** [github.com/juglab/n2v](https://github.com/juglab/n2v) ⚠
- **Paper:** Krull, Buchholz, Jug, *CVPR* 2019 ✓
- **What:** Self-supervised denoising. Trains directly on noisy data without paired ground truth.
- **License:** BSD-3-Clause ⚠
- **Notes:** Used in Lab 3 option A.

### CSBDeep
- **GitHub:** [github.com/CSBDeep/CSBDeep](https://github.com/CSBDeep/CSBDeep) ⚠
- **What:** Content-aware deep-learning restoration framework underlying CARE and several related methods.

---

## Pathology and digital pathology

### QuPath
- **GitHub:** [github.com/qupath/qupath](https://github.com/qupath/qupath) ✓
- **Paper:** Bankhead et al., *Scientific Reports* 2017 ✓ — doi: 10.1038/s41598-017-17204-5
- **What:** Whole-slide image analysis platform. WSI annotation, segmentation, classification. Deep-learning integrations (StarDist, WSInfer).
- **License:** GPL ⚠
- **Notes:** Wide adoption in digital pathology. WSInfer extension for standardized model deployment.

### WSInfer
- **GitHub:** [github.com/SBU-BMI/wsinfer](https://github.com/SBU-BMI/wsinfer) ⚠
- **What:** Standardized inference of pretrained models on whole-slide images.

### MONAI Pathology
- **GitHub:** [github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI) ⚠
- **What:** PyTorch-based framework for medical AI; pathology and clinical-imaging modules. Core MONAI plus pathology-specific extensions.

---

## Clinical imaging

### MONAI (core)
- **GitHub:** [github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI) ✓
- **Paper:** Cardoso et al., *arXiv* 2022 (arXiv:2211.02701) — ✏ corrected: no canonical Nature Methods paper exists; the framework is documented via the arXiv preprint and project website
- **What:** PyTorch-based framework for clinical AI. DICOM-aware data loaders, transforms, models, and validation utilities.
- **License:** Apache 2.0 ⚠

### nnU-Net
- **GitHub:** [github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) ✓
- **Paper:** Isensee et al., *Nature Methods* 2021 ✓ — doi: 10.1038/s41592-020-01008-z
- **What:** Self-configuring segmentation framework that has dominated medical-imaging segmentation challenges.

### TotalSegmentator
- **GitHub:** [github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator) ✓
- **Paper:** Wasserthal et al., *Radiology: AI* 2023 ✓ — doi: 10.1148/ryai.230024 (✏ corrected: agent draft had year unknown; verified 2023 publication in Radiology: AI)
- **What:** Whole-body CT/MRI multi-organ segmentation tool. Wide clinical adoption.

### VoxelMorph
- **GitHub:** [github.com/voxelmorph/voxelmorph](https://github.com/voxelmorph/voxelmorph) ⚠
- **Paper:** Balakrishnan et al., *IEEE TMI* 2019 ✓
- **What:** Deep-learning deformable image registration. Foundational for clinical-imaging registration.
- **License:** Apache 2.0 ⚠

---

## Visualization and infrastructure

### Fiji / ImageJ
- **GitHub:** [github.com/fiji](https://github.com/fiji) ⚠
- **What:** The dominant community platform for microscopy image analysis. Distribution of ImageJ with curated plugins.
- **License:** GPL ⚠
- **Notes:** Foundational; integrates StarDist, ilastik, BigStitcher, Mastodon, deepImageJ, and many others.

### scikit-image
- **GitHub:** [github.com/scikit-image/scikit-image](https://github.com/scikit-image/scikit-image) ✓
- **What:** Python library for classical image processing. Used heavily in preprocessing pipelines for AI workflows.
- **License:** BSD-3-Clause ✓

### ClearMap
- **GitHub:** [github.com/ChristophKirst/ClearMap](https://github.com/ChristophKirst/ClearMap) ✓
- **Paper:** Renier, Adams, Kirst et al., *Cell* 2016 ✓ — doi: 10.1016/j.cell.2016.05.007 (✏ corrected: agent draft listed Kirst as first author; canonical paper has Renier first; Kirst is contributing author and current GitHub maintainer)
- **What:** Volumetric image analysis, registration, and segmentation for cleared-tissue samples. Allen Brain Atlas integration.

### empanada-napari
- **GitHub:** [github.com/volume-em/empanada-napari](https://github.com/volume-em/empanada-napari) ⚠
- **What:** Panoptic segmentation napari plugin for electron microscopy.

### CellSeg3D
- **GitHub:** [github.com/AdaptiveMotorControlLab/CellSeg3D](https://github.com/AdaptiveMotorControlLab/CellSeg3D) ⚠
- **What:** 3D cell segmentation napari plugin (Mathis Lab).

### Mesmer / DeepCell
- **GitHub:** [github.com/vanvalenlab/deepcell-tf](https://github.com/vanvalenlab/deepcell-tf) ⚠
- **Paper:** Greenwald et al., *Nature Biotechnology* 2022 ✓
- **What:** Whole-cell segmentation across fluorescence and mass spectrometry imaging. Two-channel (nucleus + membrane) input.
- **License:** Apache 2.0 ⚠

---

## Vendor-format support

### Zeiss pylibCZIrw
- **GitHub:** [github.com/ZEISS/pylibczirw](https://github.com/ZEISS/pylibczirw) ⚠
- **Colab tutorial:** [colab.research.google.com/github/zeiss-microscopy/OAD](https://colab.research.google.com/github/zeiss-microscopy/OAD/blob/master/jupyter_notebooks/pylibCZIrw/pylibCZIrw_4_0_0.ipynb) ⚠
- **What:** Read/write CZI files in Python.
- **Notes:** Useful for attendees with Zeiss microscopes.

### Bio-Formats
- **GitHub:** [github.com/ome/bioformats](https://github.com/ome/bioformats) ⚠
- **What:** Java library for reading and writing >150 microscopy and life-science formats. Integrated into Fiji, ImageJ, QuPath, and many Python tools.
- **License:** GPL ⚠

---

## Atlases and big data resources

### Allen Brain Atlas
- **Web:** [portal.brain-map.org](https://portal.brain-map.org) ✓
- **What:** Reference atlases of mouse and human brain. Datasets, ontologies, and APIs (AllenSDK).

### IDR (Image Data Resource)
- **Web:** [idr.openmicroscopy.org](https://idr.openmicroscopy.org) ⚠
- **What:** Public repository of reference imaging datasets.

### MICrONS (Allen + Princeton + Baylor)
- **Web:** [microns-explorer.org](https://www.microns-explorer.org) ⚠
- **What:** EM connectomics dataset with cell-type annotations.

### Cell Image Library
- **Web:** [cellimagelibrary.org](http://www.cellimagelibrary.org) ⚠
- **What:** Curated collection of cell biology images.

---

## Gaps and notes

The agent research pass surfaced 50+ projects. Several areas remain thin:

- **Euro-BioImaging** and **Global BioImaging** are coordination and infrastructure networks rather than tool-publishing organizations. Their value is in linking labs, not GitHub repos. ❓
- **EMBL-EBI imaging tools** are scattered across labs rather than centrally hosted. The BioImage Archive is data-centric; tooling is dispersed.
- **CZ Imaging Institute** is recently founded; the GitHub presence is still expanding. Worth re-checking before any external delivery.
- **Pathology and clinical-imaging foundation models** (UNI, Virchow, CONCH, MedSAM, RadFM, CXR-CLIP, BiomedCLIP) are touched on in the broader curriculum but not in this lecture's primary scope; they are listed in the curriculum module catalog instead.
- **Cryo-EM tooling** (RELION, cryoSPARC, Topaz, ModelAngelo) is deferred per workshop scope.

## What's verified versus flagged

Of the entries above:

- **Fully verified during this build session (✓):** Cellpose, ZeroCostDL4Mic, DL4MicEverywhere, μSAM, napari, scikit-image, image.sc, several papers (Stringer 2021, Schmidt 2018, Weigert 2018, Krull 2019, Kirillov 2023, Pape 2025, Greenwald 2022, von Chamier 2021, Hidalgo-Cenalmor 2024, Balakrishnan 2019), Allen Brain Atlas portal.
- **Flagged ⚠ for verification:** GitHub URLs and license fields for many entries; specific paper citations for ilastik, deepImageJ, BioImage Model Zoo, BigStitcher, Mastodon, Bankhead/QuPath, Berg/ilastik, Hörl/BigStitcher, Wolff/Mastodon, Gómez-de-Mariscal/deepImageJ, Ronneberger U-Net, Kirst/ClearMap, Van Geit/BluePyOpt, Isensee/nnU-Net.

Anything tagged ⚠ should be cross-checked against the source GitHub repository's `README` or the cited paper before public-facing use.

## Citation corrections recorded

Two citation errors in the agent's draft were corrected during compilation:

1. **StarDist** — agent listed Weigert et al. (correct paper has Schmidt et al., 2018, *MICCAI*).
2. **DL4MicEverywhere** — agent listed Fuster-Barceló et al. (correct paper has Hidalgo-Cenalmor et al., 2024, *Nature Methods*).
3. **μSAM** — agent listed Wolny et al. (correct paper has Pape et al., 2025, *Nature Methods*).

These corrections are reflected in the entries above. Other agent-provided citations carry ⚠ pending verification.
