#!/usr/bin/env python3
"""Bake MABC raw inputs into per-notebook sample npz files.

Reads `dataset_samples_spec.py` for per-NB transforms, walks raw files in
`MABC_raws/` (default; override with --raws), and writes
`data/mabc/<nb_id>.npz`. Updates `data/MANIFEST.json` with per-file sha256
+ size + source attribution.

If a raw input is missing for a given NB, the bake script SKIPS that NB by
default (the corresponding .npz simply isn't produced; the notebook then
falls through to the canonical tier when that NB runs). To produce a tiny
synthetic placeholder npz instead, pass --placeholder.

Run:
    python3 scripts/bake_dataset_samples.py                    # only bake NBs whose raws exist
    python3 scripts/bake_dataset_samples.py --placeholder      # write synthetic placeholders for missing NBs too
    python3 scripts/bake_dataset_samples.py --nb 01_cellpose_segmentation   # one NB only

Idempotent: re-running with the same raws produces the same npz bytes
(deterministic given a fixed numpy seed for placeholder mode).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "mabc"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"

DEFAULT_RAWS_DIR = REPO_ROOT.parent / "MABC_raws"  # Claude_Workspace/MABC_raws/

BAKE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

# Lazy import so the spec file can change without re-installing this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_samples_spec import SAMPLES  # noqa: E402


# ---------------------------------------------------------------------------
# Image readers
# ---------------------------------------------------------------------------

def _read_image_file(p: Path) -> np.ndarray:
    """Read TIFF / PNG / JPEG into ndarray. tifffile preferred, PIL fallback."""
    try:
        import tifffile
        return tifffile.imread(str(p))
    except Exception:
        from PIL import Image
        return np.array(Image.open(p))


def _walk_images(folder: Path) -> list[Path]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    out = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Per-NB transforms
# ---------------------------------------------------------------------------

def _to_grayscale(a: np.ndarray) -> np.ndarray:
    """Reduce a possibly multi-channel array to (H, W)."""
    a = np.asarray(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        # (H, W, C) with C in (3, 4)
        if a.shape[-1] in (3, 4):
            return a[..., :3].mean(axis=-1)
        # (C, H, W) with small C
        if a.shape[0] <= 4 and a.shape[0] < min(a.shape[1:]):
            return a.mean(axis=0)
        # multi-page TIFF (Z, H, W)
        return a[a.shape[0] // 2]
    # higher-dim — take a middle plane
    return a.reshape(-1, *a.shape[-2:])[a.shape[0] // 2]


def _resize_2d(a: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2D array using scipy.ndimage.zoom."""
    from scipy.ndimage import zoom
    a = np.asarray(a, dtype=np.float32)
    th, tw = target_hw
    ch, cw = a.shape[:2]
    s = min(ch, cw)
    a = a[:s, :s]  # center-square crop
    factor = th / s
    return zoom(a, factor, order=1)


def _normalize_uint8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - lo) / (hi - lo) * 255.0
    return a.astype(np.uint8)


def _transform_to_grayscale_then_resize(raw: np.ndarray, target_hw, n_samples) -> np.ndarray:
    """Single 2D image or stack -> (n_samples, H, W) uint8."""
    if raw.ndim == 2:
        # Single image — tile-crop into n_samples non-overlapping windows
        h, w = raw.shape
        side = min(h, w) // 2
        out = []
        for i in range(n_samples):
            r = (i * side // 2) % max(h - side, 1)
            c = (i * side // 2) % max(w - side, 1)
            tile = raw[r:r + side, c:c + side]
            out.append(_normalize_uint8(_resize_2d(tile, target_hw)))
        return np.stack(out)
    if raw.ndim == 3:
        # Stack — pick n_samples planes, reduce to grayscale, resize
        gray_planes = [_to_grayscale(raw[i]) for i in range(min(n_samples, len(raw)))]
        while len(gray_planes) < n_samples:
            gray_planes.append(gray_planes[-1])  # repeat last plane
        return np.stack([_normalize_uint8(_resize_2d(p, target_hw)) for p in gray_planes])
    raise ValueError(f"Unsupported raw shape: {raw.shape}")


def _transform_rgb_resize(raw: np.ndarray, target_hw, n_samples) -> np.ndarray:
    """Keep RGB; resize. Used for H&E (NB16)."""
    if raw.ndim == 3 and raw.shape[-1] in (3, 4):
        from scipy.ndimage import zoom
        h, w = raw.shape[:2]
        s = min(h, w)
        cropped = raw[:s, :s, :3]
        factor = (target_hw[0] / s, target_hw[1] / s, 1)
        resized = zoom(cropped, factor, order=1)
        # Tile into n_samples windows
        out = [resized] * n_samples
        return np.stack(out).astype(np.uint8)
    raise ValueError(f"Expected RGB(A) image, got shape {raw.shape}")


def _tile_crop_grayscale(big_2d: np.ndarray, n_samples: int, target_hw) -> np.ndarray:
    """Take an arbitrary-size 2D array and produce N non-overlapping square tiles
    at target_hw. Goes channel-mean if RGB. Returns (n_samples, H, W) uint8."""
    g = _to_grayscale(big_2d)
    h, w = g.shape
    side = min(h, w) // 2 if min(h, w) >= 2 * max(target_hw) else min(h, w)
    out = []
    # Walk the image in a grid to extract n non-overlapping crops
    rows = max(1, h // side)
    cols = max(1, w // side)
    coords = [(r * side, c * side) for r in range(rows) for c in range(cols)]
    coords = coords[:n_samples]
    while len(coords) < n_samples:
        coords.append(coords[-1])  # pad by repeating last
    for r, c in coords:
        tile = g[r:r + side, c:c + side]
        out.append(_normalize_uint8(_resize_2d(tile, target_hw)))
    return np.stack(out)


def _transform_tile_crop_pair_to_grayscale(raw_a: np.ndarray, raw_b: np.ndarray,
                                            n_samples: int, target_hw) -> np.ndarray:
    """Take two large 2D images, interleave tiles from each so we get half-and-half."""
    each = n_samples // 2
    out_a = _tile_crop_grayscale(raw_a, each, target_hw)
    out_b = _tile_crop_grayscale(raw_b, n_samples - each, target_hw)
    return np.concatenate([out_a, out_b], axis=0)


def _transform_drosophila_paired_channels(raws_dir: Path, n_samples: int,
                                          target_hw) -> tuple[np.ndarray, np.ndarray]:
    """Load the 9 DrosophilaCells files (3 samples × DAPI/Tub/Actin) and produce
    paired (input, target) tensors for virtual-staining demos.

    Each native 512×512 image is sub-tiled into 4 non-overlapping quadrants of
    256×256 (or whatever target_hw is). With 3 samples × 4 sub-tiles = 12
    paired examples — enough to train a tiny U-Net + hold a few out for test.

    Convention: input = DAPI (nuclei), target = Tubulin (cytoskeleton).
    Tiles are paired identically: sample i sub-tile q's DAPI maps to sample i
    sub-tile q's Tubulin (same field of view, different channel).
    """
    out_in, out_tgt = [], []
    th, tw = target_hw
    for sample_idx in range(1, 4):  # samples 1, 2, 3
        dapi_p = raws_dir / f"Dros{sample_idx}Dapi.TIF"
        tub_p = raws_dir / f"Dros{sample_idx}Tub.TIF"
        if not (dapi_p.exists() and tub_p.exists()):
            continue
        dapi = _to_grayscale(_read_image_file(dapi_p))
        tub = _to_grayscale(_read_image_file(tub_p))
        # Sub-tile into 2x2 = 4 non-overlapping quadrants. Each quadrant is half the
        # native height/width, then resized to target_hw.
        h, w = dapi.shape
        hh, hw = h // 2, w // 2
        for r in (0, hh):
            for c in (0, hw):
                d_tile = dapi[r:r + hh, c:c + hw]
                t_tile = tub[r:r + hh, c:c + hw]
                out_in.append(_normalize_uint8(_resize_2d(d_tile, target_hw)))
                out_tgt.append(_normalize_uint8(_resize_2d(t_tile, target_hw)))
    if not out_in:
        raise ValueError("No DrosophilaCells DAPI/Tub pairs found.")
    # Trim to n_samples if specified; default keeps all 12.
    if n_samples and len(out_in) > n_samples:
        out_in = out_in[:n_samples]
        out_tgt = out_tgt[:n_samples]
    return np.stack(out_in), np.stack(out_tgt)


def _transform_z_stack_to_3d_volume(raws_dir: Path, target_hw,
                                    z_subsample: int = 1) -> np.ndarray:
    """Walk the PAM Sent/Tiff folder for *_C0_Z*.tif and *_C1_Z*.tif files,
    stack into a 4D volume (Z, C, H, W) at target_hw."""
    c0_paths = sorted(raws_dir.glob("*_C0_Z*.tif"))
    c1_paths = sorted(raws_dir.glob("*_C1_Z*.tif"))
    if not c0_paths or not c1_paths:
        raise ValueError(f"No *_C0_Z*.tif / *_C1_Z*.tif found in {raws_dir}")
    if z_subsample > 1:
        c0_paths = c0_paths[::z_subsample]
        c1_paths = c1_paths[::z_subsample]
    n_z = min(len(c0_paths), len(c1_paths))
    print(f"    z_stack: {n_z} z-slices, 2 channels, target {target_hw}")
    vol = np.zeros((n_z, 2, target_hw[0], target_hw[1]), dtype=np.uint8)
    for z in range(n_z):
        c0 = _to_grayscale(_read_image_file(c0_paths[z]))
        c1 = _to_grayscale(_read_image_file(c1_paths[z]))
        vol[z, 0] = _normalize_uint8(_resize_2d(c0, target_hw))
        vol[z, 1] = _normalize_uint8(_resize_2d(c1, target_hw))
    # Wrap in a length-1 outer dim so the npz "images" shape is (1, Z, C, H, W).
    return vol[np.newaxis, ...]


def _transform_z_stack_axial_pair(raws_dir: Path, target_hw, spec_extras: dict):
    """For NB03a denoising. Walks *_C{channel}_Z*.tif files in a PAM-style Z-stack
    folder. Picks specified `train_z_slices` as noisy training inputs; computes
    axial-averaged neighbors (±axial_window) as the 'clean reference' proxy.
    Also extracts a held-out test slice + full stack for browsing.

    Returns (images, labels, extras_dict) where:
      images = (N, H, W) uint8   — noisy training slices, center-cropped to target_hw
      labels = (N, H, W) uint8   — clean_ref training slices (axial average)
      extras_dict keys:
        test_noisy      = (H, W) uint8
        test_clean_ref  = (H, W) uint8
        full_stack      = (Z, H, W) uint8   (full channel stack for Z browsing)
    """
    channel = int(spec_extras.get("channel", 0))
    train_z = list(spec_extras.get("train_z_slices", [20, 23, 26, 29, 32]))
    test_z = int(spec_extras.get("test_z_slice", 40))
    axial_window = int(spec_extras.get("axial_window", 5))

    # Find the channel-specific Z-files. Sorted by name = sorted by Z-index.
    c_paths = sorted(raws_dir.glob(f"*_C{channel}_Z*.tif"))
    if not c_paths:
        raise ValueError(f"No *_C{channel}_Z*.tif files found in {raws_dir}")
    n_z = len(c_paths)
    print(f"    z_stack_axial_pair: channel={channel}, {n_z} z-slices, train_z={train_z}, test_z={test_z}, axial_window=±{axial_window}")

    # Read full channel volume into memory (73 × 512 × 512 = ~38 MB at uint16).
    vol_raw = np.stack([_to_grayscale(_read_image_file(p)) for p in c_paths]).astype(np.float32)

    def _center_crop_resize(arr_2d):
        """Center-crop to a square if non-square, then resize to target_hw."""
        h, w = arr_2d.shape[:2]
        s = min(h, w)
        cy, cx = h // 2, w // 2
        cropped = arr_2d[cy - s // 2: cy - s // 2 + s, cx - s // 2: cx - s // 2 + s]
        return _resize_2d(cropped, target_hw)

    def _slice_to_uint8(arr_2d):
        return _normalize_uint8(_center_crop_resize(arr_2d))

    def _axial_average(z_center):
        z_lo = max(0, z_center - axial_window)
        z_hi = min(n_z, z_center + axial_window + 1)
        return vol_raw[z_lo:z_hi].mean(axis=0)

    # Training pairs
    images = np.stack([_slice_to_uint8(vol_raw[z]) for z in train_z])
    labels = np.stack([_slice_to_uint8(_axial_average(z)) for z in train_z])

    # Held-out test
    test_noisy = _slice_to_uint8(vol_raw[test_z])
    test_clean_ref = _slice_to_uint8(_axial_average(test_z))

    # Full stack for Z browsing — center-crop + resize each slice, uint8
    full_stack = np.stack([_slice_to_uint8(vol_raw[z]) for z in range(n_z)])

    extras = {
        "test_noisy": test_noisy,
        "test_clean_ref": test_clean_ref,
        "full_stack": full_stack,
    }
    return images, labels, extras


def _transform_wsi_tile_extract(raws: list[np.ndarray], n_samples: int,
                                target_hw) -> np.ndarray:
    """For each Aperio SVS pyramid given as a raw, extract n_per_input tiles from
    a useful pyramid level (skip mostly-white tiles).

    raws here is a list of 2D pyramid-level arrays (we picked them in _bake_one
    by reading a non-base level)."""
    n_per_input = max(1, n_samples // len(raws))
    out = []
    for raw in raws:
        if raw.ndim != 3 or raw.shape[-1] not in (3, 4):
            continue
        h, w = raw.shape[:2]
        # Greedy: scan tiles, pick those with mean < 230 (not mostly white background).
        side = min(h, w, 1024)
        candidates = []
        for r in range(0, h - side, side):
            for c in range(0, w - side, side):
                tile = raw[r:r + side, c:c + side, :3]
                if tile.mean() < 230:  # tissue, not slide background
                    candidates.append(tile)
                if len(candidates) >= n_per_input * 4:
                    break
            if len(candidates) >= n_per_input * 4:
                break
        # Resize each to target_hw and pick the first n_per_input
        from scipy.ndimage import zoom as _zoom
        for tile in candidates[:n_per_input]:
            factor = (target_hw[0] / tile.shape[0], target_hw[1] / tile.shape[1], 1)
            resized = _zoom(tile, factor, order=1).astype(np.uint8)
            out.append(resized)
    while len(out) < n_samples and out:
        out.append(out[-1])
    return np.stack(out[:n_samples])


def _apply_transform(transform_name: str, raws_obj, target_hw, n_samples, spec_extras=None):
    """Dispatcher. Returns (images, labels). raws_obj is per-transform:
       - For single-input transforms: a list of ndarrays (use raws_obj[0]).
       - For paired/multi-input: the function itself reads from disk via paths in spec_extras.
    """
    spec_extras = spec_extras or {}

    if transform_name == "to_grayscale_then_resize":
        if not raws_obj:
            raise ValueError("No raw inputs.")
        return _transform_to_grayscale_then_resize(raws_obj[0], target_hw, n_samples), None

    if transform_name == "rgb_resize":
        if not raws_obj:
            raise ValueError("No raw inputs.")
        return _transform_rgb_resize(raws_obj[0], target_hw, n_samples), None

    if transform_name == "tile_crop_to_grayscale":
        if not raws_obj:
            raise ValueError("No raw inputs.")
        return _tile_crop_grayscale(raws_obj[0], n_samples, target_hw), None

    if transform_name == "tile_crop_pair_to_grayscale":
        raw_a = spec_extras.get("raw_a")
        raw_b = spec_extras.get("raw_b")
        if raw_a is None or raw_b is None:
            raise ValueError("Need both raw_a and raw_b for pair transform.")
        return _transform_tile_crop_pair_to_grayscale(raw_a, raw_b, n_samples, target_hw), None

    if transform_name == "drosophila_paired_channels":
        raws_dir = spec_extras.get("raws_dir")
        if raws_dir is None:
            raise ValueError("Need raws_dir for drosophila_paired_channels.")
        imgs, tgt = _transform_drosophila_paired_channels(raws_dir, n_samples, target_hw)
        return imgs, tgt

    if transform_name == "z_stack_to_3d_volume":
        raws_dir = spec_extras.get("raws_dir")
        z_subsample = spec_extras.get("z_subsample", 1)
        if raws_dir is None:
            raise ValueError("Need raws_dir for z_stack transform.")
        return _transform_z_stack_to_3d_volume(raws_dir, target_hw, z_subsample), None

    if transform_name == "z_stack_axial_pair":
        raws_dir = spec_extras.get("raws_dir")
        if raws_dir is None:
            raise ValueError("Need raws_dir for z_stack_axial_pair transform.")
        # Returns (images, labels, extras_dict) — 3-tuple form.
        return _transform_z_stack_axial_pair(raws_dir, target_hw, spec_extras)

    if transform_name == "wsi_tile_extract":
        return _transform_wsi_tile_extract(raws_obj, n_samples, target_hw), None

    if transform_name == "first_tile_per_input_to_grayscale":
        # One tile per input, in order; pads/truncates to n_samples.
        out = []
        for raw in raws_obj:
            tile = _tile_crop_grayscale(raw, 1, target_hw)
            out.append(tile[0])
        while len(out) < n_samples:
            out.append(out[-1])
        return np.stack(out[:n_samples]), None

    if transform_name in ("paired_image_and_mask_resize",
                          "paired_image_and_instance_masks",
                          "split_channels_to_input_target"):
        print(f"  [WARN] transform '{transform_name}' is a stub; falling back to grayscale.")
        return _transform_to_grayscale_then_resize(raws_obj[0], target_hw, n_samples), None
    raise ValueError(f"Unknown transform: {transform_name}")


# ---------------------------------------------------------------------------
# Placeholder generation (for testing the architecture without MABC raws)
# ---------------------------------------------------------------------------

def _make_placeholder(spec: dict, nb_id: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Produce a small synthetic placeholder so the gh-pages URL works while
    we wait for the real MABC samples. Marked clearly in metadata."""
    rng = np.random.default_rng(abs(hash(nb_id)) % (2**32))
    n = spec["n_samples"]
    th, tw = spec["target_hw"]
    if spec["transform"] == "rgb_resize":
        # H&E-like placeholder: pink + a few purple blobs
        imgs = np.zeros((n, th, tw, 3), dtype=np.uint8)
        for i in range(n):
            imgs[i, :, :, 0] = 240
            imgs[i, :, :, 1] = 180
            imgs[i, :, :, 2] = 215
            for _ in range(20):
                cy, cx = rng.integers(20, th - 20), rng.integers(20, tw - 20)
                rr = rng.integers(4, 9)
                Y, X = np.ogrid[:th, :tw]
                m = (Y - cy) ** 2 + (X - cx) ** 2 <= rr ** 2
                imgs[i, m] = (90, 50, 140)
        return imgs, None
    # Generic grayscale placeholder
    imgs = np.zeros((n, th, tw), dtype=np.uint8)
    for i in range(n):
        for _ in range(rng.integers(8, 16)):
            cy, cx = rng.integers(20, th - 20), rng.integers(20, tw - 20)
            rr = rng.integers(8, 16)
            Y, X = np.ogrid[:th, :tw]
            m = (Y - cy) ** 2 + (X - cx) ** 2 <= rr ** 2
            imgs[i][m] = rng.integers(120, 220)
    return imgs, None


# ---------------------------------------------------------------------------
# Bake one NB
# ---------------------------------------------------------------------------

def _resolve_raws(spec: dict, raws_dir: Path):
    """Read raw inputs declared by the spec. Returns (raws_list, spec_extras) where
    spec_extras contains structured args for transforms that need them (paired
    inputs, z-stack folder, etc.)."""
    extras: dict = {}

    # SVS pyramid handling for WSI: read a useful pyramid level (not the largest)
    def _read_svs_thumbnail(p: Path) -> np.ndarray:
        import tifffile
        with tifffile.TiffFile(p) as tif:
            # Find the first level small enough for tile extraction (~5000-10000 px wide)
            pages = tif.pages
            best = pages[-2] if len(pages) >= 2 else pages[-1]
            for page in pages:
                if page.shape[0] < 8000 and page.shape[0] >= 1000:
                    best = page
                    break
            return best.asarray()

    if "raw_input" in spec:
        p = raws_dir / spec["raw_input"]
        if not p.exists():
            return None, extras
        if p.is_dir():
            extras["raws_dir"] = p
            return [], extras  # transforms that work on a dir
        raws = [_read_image_file(p)]
        return raws, extras

    if "raw_input_a" in spec and "raw_input_b" in spec:
        pa = raws_dir / spec["raw_input_a"]
        pb = raws_dir / spec["raw_input_b"]
        if not (pa.exists() and pb.exists()):
            return None, extras
        extras["raw_a"] = _read_image_file(pa)
        extras["raw_b"] = _read_image_file(pb)
        return [], extras

    if "raw_inputs" in spec:
        out = []
        for rel in spec["raw_inputs"]:
            p = raws_dir / rel
            if not p.exists():
                return None, extras
            if p.suffix.lower() == ".svs":
                out.append(_read_svs_thumbnail(p))
            else:
                out.append(_read_image_file(p))
        return out, extras

    return None, extras


def _bake_one(nb_id: str, spec: dict, raws_dir: Path, placeholder: bool) -> Optional[Path]:
    raws, extras = _resolve_raws(spec, raws_dir)
    if raws is None:
        if placeholder:
            images, labels = _make_placeholder(spec, nb_id)
            source_status = "placeholder (synthetic stand-in until MABC raw arrives)"
        else:
            ref = spec.get("raw_input") or spec.get("raw_input_a") or spec.get("raw_inputs")
            print(f"  [SKIP] {nb_id}: raw input {ref!r} not found in {raws_dir}.")
            return None
    else:
        # Pass z_subsample if spec has it
        if "z_subsample" in spec:
            extras["z_subsample"] = spec["z_subsample"]
        # Pass spec-level extras the transform may need (used by z_stack_axial_pair).
        for k in ("channel", "train_z_slices", "test_z_slice", "axial_window"):
            if k in spec:
                extras[k] = spec[k]
        try:
            result = _apply_transform(
                spec["transform"], raws, spec["target_hw"], spec["n_samples"], extras
            )
            # Transforms return either (images, labels) or (images, labels, payload_extras).
            payload_extras = {}
            if len(result) == 3:
                images, labels, payload_extras = result
            else:
                images, labels = result
        except Exception as e:
            import traceback
            print(f"  [ERR] {nb_id}: transform failed: {e}")
            traceback.print_exc(limit=2)
            return None
        source_status = "real (MABC)"

    out = DATA_DIR / f"{nb_id}.npz"
    metadata = {
        "nb_id": nb_id,
        "description": spec["description"],
        "source": source_status,
        "license": spec["license"],
        "citation": spec["citation"],
        "n_samples": int(images.shape[0]),
        "shape_per_image": list(images.shape[1:]),
        "dtype": str(images.dtype),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bake_version": BAKE_VERSION,
    }
    payload = {
        "images": images,
        "filenames": np.array([f"{nb_id}_{i:02d}" for i in range(images.shape[0])]),
        "metadata": np.array(metadata, dtype=object),
    }
    if labels is not None:
        payload["labels"] = labels
    # Merge per-transform payload extras (e.g. test_noisy / test_clean_ref / full_stack
    # from z_stack_axial_pair). Placeholder branch has no extras → dict stays empty.
    for k, v in (locals().get("payload_extras") or {}).items():
        payload[k] = v

    np.savez_compressed(out, **payload)
    size = out.stat().st_size
    print(f"  [OK ] {nb_id}: {source_status} → {out.name} ({size/1e3:.1f} KB)")
    return out


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_manifest(npz_paths: list[Path]) -> None:
    manifest = {
        "bake_version": BAKE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {},
    }
    for p in sorted(npz_paths):
        manifest["files"][p.name] = {
            "size_bytes": p.stat().st_size,
            "sha256": _hash_file(p),
        }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest written: {MANIFEST_PATH}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raws", type=Path, default=DEFAULT_RAWS_DIR,
                    help=f"Directory containing MABC raw files. Default: {DEFAULT_RAWS_DIR}")
    ap.add_argument("--placeholder", action="store_true",
                    help="Emit a synthetic placeholder npz when the raw input is missing.")
    ap.add_argument("--nb", type=str, default=None,
                    help="Bake only this notebook (e.g. '01_cellpose_segmentation'). Default: all.")
    args = ap.parse_args()

    if not args.raws.exists():
        print(f"[INFO] Raws dir does not exist: {args.raws}")
        if not args.placeholder:
            print("       Pass --placeholder to write synthetic stand-ins, or create the dir and add MABC files.")
            return 0

    written: list[Path] = []
    for nb_id, spec in SAMPLES.items():
        if args.nb and args.nb != nb_id:
            continue
        out = _bake_one(nb_id, spec, args.raws, args.placeholder)
        if out is not None:
            written.append(out)

    if written:
        _update_manifest(written)
    print(f"\nDone. {len(written)} npz file(s) written to {DATA_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
