"""Per-notebook MABC sample specs.

Maps each T1 notebook to one of the MABC raw inputs from
`Claude_Workspace/MABC_Raws/` and declares the per-NB transform that turns
that raw into a small, ready-to-ship `data/mabc/<nb_id>.npz`.

`bake_dataset_samples.py` consumes this spec.

License (per `MABC_Raws/Lic.txt`): CC-BY 4.0 for everything in MABC_Raws.
"""

from __future__ import annotations

DEFAULT_HW = (256, 256)
COMMON_LICENSE = "CC-BY 4.0"
COMMON_CITATION = "MABC, Mt Sinai. Workshop sample 2026."
PAM_CITATION = "PAM mice E0771 — MABC, Mt Sinai. Workshop sample 2026."


SAMPLES = {
    # ---- NB01 — Cellpose pretrained segmentation ----
    "01_cellpose_segmentation": {
        "description": "8 fluorescence cell tiles from MABC's Cell counting series.",
        "raw_input": "Exercise Images/Counting and Segmentation/Cell counting 2.tif",
        "n_samples": 8,
        "transform": "tile_crop_to_grayscale",
        "target_hw": DEFAULT_HW,
        "labels": None,
        "license": COMMON_LICENSE,
        "citation": COMMON_CITATION,
    },

    # ---- NB02 — Validation and quantification ----
    "02_validation_quantification": {
        "description": "8 paired tiles from Cell counting 1+2 — same tissue, different views.",
        "raw_input_a": "Exercise Images/Counting and Segmentation/Cell counting 1.tif",
        "raw_input_b": "Exercise Images/Counting and Segmentation/Cell counting 2.tif",
        "n_samples": 8,
        "transform": "tile_crop_pair_to_grayscale",
        "target_hw": DEFAULT_HW,
        "labels": None,
        "license": COMMON_LICENSE,
        "citation": COMMON_CITATION,
    },

    # ---- NB03a — Denoising (Noise2Void) ----
    "03a_denoising_n2v": {
        "description": "16-bit DAPI fluorescence as a real low-noise reference; noisy versions added downstream.",
        "raw_input": "Exercise Images/BitDepth/Dapi_WhatBitDepth.tif",
        "n_samples": 8,
        "transform": "tile_crop_to_grayscale",
        "target_hw": DEFAULT_HW,
        "labels": None,
        "license": COMMON_LICENSE,
        "citation": COMMON_CITATION,
    },

    # ---- NB03b — Foundation-model segmentation (SAM/μSAM) ----
    "03b_foundation_model_segmentation": {
        "description": "4 Drosophila DAPI fluorescence tiles for SAM point/box-prompt demos.",
        "raw_input": "Exercise Images/DrosophilaCells/Dros1Dapi.TIF",
        "n_samples": 4,
        "transform": "tile_crop_to_grayscale",
        "target_hw": DEFAULT_HW,
        "labels": None,
        "license": COMMON_LICENSE,
        "citation": COMMON_CITATION,
    },

    # ---- NB06 — Virtual staining ----
    "06_virtual_staining": {
        "description": "Drosophila cell pairs (3 samples × 4 sub-tiles = 12 paired) — DAPI predicts Tubulin.",
        "raw_input": "Exercise Images/DrosophilaCells/",
        "n_samples": 12,                                         # 3 samples × 4 sub-tiles per sample
        "transform": "drosophila_paired_channels",
        "target_hw": (256, 256),                                 # half native, fits TinyUNet
        "labels": "paired_target_channel",
        "license": COMMON_LICENSE,
        "citation": COMMON_CITATION,
    },

    # ---- NB07 — Widefield super-resolution ----
    "07_widefield_superres": {
        "description": "Real fluorescence as HR; LR synthesized via blur+downsample downstream.",
        "raw_input": "Image_03_ColorDecon.tif",
        "n_samples": 8,
        "transform": "tile_crop_to_grayscale",
        "target_hw": DEFAULT_HW,
        "labels": None,
        "license": COMMON_LICENSE,
        "citation": COMMON_CITATION,
    },

    # ---- NB09 — Cellpose fine-tuning ----
    "09_cellpose_finetune": {
        "description": "Paired fluorescence tiles for human-in-the-loop fine-tuning. Labels generated downstream via baseline.",
        "raw_input_a": "Exercise Images/Counting and Segmentation/Cell counting 1.tif",
        "raw_input_b": "Exercise Images/Counting and Segmentation/Cell counting 2.tif",
        "n_samples": 8,
        "transform": "tile_crop_pair_to_grayscale",
        "target_hw": DEFAULT_HW,
        "labels": None,                                          # init to zeros downstream
        "license": COMMON_LICENSE,
        "citation": COMMON_CITATION,
    },

    # ---- NB10 — 3D segmentation ----
    "10_3d_segmentation": {
        "description": "Real 3D fluorescence: PAM mice E0771 proliferation, 2 channels × 73 Z-slices.",
        "raw_input": "Possible datasets for vendors/PAM mice E0771 prolif ex vivo base good 4_Sent/Tiff/",
        "n_samples": 1,                                          # one full volume
        "transform": "z_stack_to_3d_volume",
        "target_hw": (128, 128),
        "z_subsample": 1,                                        # keep all 73 Z planes
        "labels": None,
        "license": COMMON_LICENSE,
        "citation": PAM_CITATION,
    },

    # ---- NB12 — Deconvolution ----
    "12_deconvolution": {
        "description": "Image_03_ColorDecon — IHC/multistain explicitly named for deconvolution demos.",
        "raw_input": "Image_03_ColorDecon.tif",
        "n_samples": 8,
        "transform": "tile_crop_to_grayscale",
        "target_hw": (64, 64),
        "labels": None,
        "license": COMMON_LICENSE,
        "citation": COMMON_CITATION,
    },

    # ---- NB13 — Validation case study ----
    "13_validation_case_study": {
        "description": "4 diverse fluorescence tiles spanning easy / hard / OOD distributions.",
        "raw_inputs": [
            "Exercise Images/Counting and Segmentation/Cell counting 1.tif",
            "Exercise Images/Counting and Segmentation/Cell counting 2.tif",
            "Exercise Images/DrosophilaCells/Dros1Dapi.TIF",
            "Exercise Images/Background/experimental.tif",
        ],
        "n_samples": 4,
        "transform": "first_tile_per_input_to_grayscale",
        "target_hw": DEFAULT_HW,
        "labels": None,
        "license": COMMON_LICENSE,
        "citation": COMMON_CITATION,
    },

    # ---- NB14 — Spot detection ----
    "14_spot_detection": {
        "description": "Drosophila tubulin tiles — punctate sub-cellular structure for spot detection demos.",
        "raw_input": "Exercise Images/DrosophilaCells/Dros1Tub.TIF",
        "n_samples": 8,
        "transform": "tile_crop_to_grayscale",
        "target_hw": (128, 128),
        "labels": None,                                          # no annotated coords
        "license": COMMON_LICENSE,
        "citation": COMMON_CITATION,
    },

    # ---- NB16 — WSI → transcriptomics ----
    "16_wsi_transcriptomics": {
        "description": "Real H&E WSI tiles from CMU-1 (canonical reference) and LungCancer.",
        "raw_inputs": [
            "Image_04_CMU-1.svs",
            "Image_05_LungCancer.svs",
        ],
        "n_samples": 8,                                          # 4 from each
        "transform": "wsi_tile_extract",
        "target_hw": (224, 224),                                 # ViT input
        "labels": None,
        "license": COMMON_LICENSE,
        "citation": COMMON_CITATION,
    },
}
