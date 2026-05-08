#!/usr/bin/env python3
"""Patch fixed-range @param sliders that don't auto-scale to real-data image
dimensions.

Bug: Colab `@param` sliders are static (defined at parse time) and can't
reference variables. NBs that hard-coded `min:0, max:255` for pixel
coordinates broke when real-data images were larger than 256 px (e.g.
BBBC020 at ~1388×1040 only got the top-left corner addressed).

Fix: convert pixel-coord sliders to **percent (0-100)** and compute the
actual pixel index from `img.shape` inside the cell. Auto-scales to any
image size. Sentinel-guarded for idempotence.

Run from the repo root:
    python3 scripts/fix_param_sliders.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "notebooks"

SENTINEL = "# SLIDER-RANGE-FIX"


# Per-NB known issues. Each entry replaces a snippet (`old`) with a fixed
# version (`new`). Matching is exact-substring; if `new` already appears in
# the cell, the fix is treated as already applied.
FIXES = {
    "03b_foundation_model_segmentation.ipynb": [
        # --- 1. Slider fix in cell 16 (already applied in v1) ---
        {
            "issue": "prompt_x / prompt_y hardcoded to max=255 — broken on images larger than 256px",
            "old": (
                'prompt_x = 128  # @param {type: "slider", min: 0, max: 255, step: 4}\n'
                'prompt_y = 128  # @param {type: "slider", min: 0, max: 255, step: 4}\n'
                'multimask = True  # @param {type: "boolean"}\n'
                '\n'
                'input_pt = np.array([[prompt_x, prompt_y]])'
            ),
            "new": (
                f'{SENTINEL}\n'
                '# Sliders are in PERCENT of image dimensions so they auto-scale to any image size.\n'
                '# (Colab @param sliders are static and cannot reference variables, so we map\n'
                '# percent → pixel inside the cell.)\n'
                'prompt_x_pct = 50  # @param {type: "slider", min: 0, max: 100, step: 1}\n'
                'prompt_y_pct = 50  # @param {type: "slider", min: 0, max: 100, step: 1}\n'
                'multimask = True  # @param {type: "boolean"}\n'
                '\n'
                'H, W = img.shape[:2]\n'
                'prompt_x = int(prompt_x_pct / 100 * (W - 1))\n'
                'prompt_y = int(prompt_y_pct / 100 * (H - 1))\n'
                'print(f"Image is {W}×{H}. Prompt point: ({prompt_x}, {prompt_y}) px '
                '(= {prompt_x_pct}%, {prompt_y_pct}%).")\n'
                '\n'
                'input_pt = np.array([[prompt_x, prompt_y]])'
            ),
        },
        # --- 2. Cell 14: stale-global `size` in fallback `point` calc ---
        {
            "issue": "cell 14: `point = np.array([size // 2, size // 2])` uses stale synthetic global",
            "old": (
                'else:\n'
                '    point = np.array([size // 2, size // 2])'
            ),
            "new": (
                'else:\n'
                f'    {SENTINEL}: derive from img.shape so this works on any image size\n'
                '    point = np.array([img.shape[0] // 2, img.shape[1] // 2])'
            ),
        },
        # --- 2b. Cell 14 (SIMULATE branch): np.ogrid uses stale `size` ---
        {
            "issue": "cell 14 SIMULATE branch: `np.ogrid[:size, :size]` uses stale synthetic global",
            "old": (
                'else:\n'
                '    # Simulated: threshold around the prompt point\n'
                '    Y, X = np.ogrid[:size, :size]\n'
                '    sam_mask = (Y - point[0])**2 + (X - point[1])**2 <= 25**2'
            ),
            "new": (
                'else:\n'
                '    # Simulated: threshold around the prompt point\n'
                f'    {SENTINEL}: derive bounds from img.shape\n'
                '    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]\n'
                '    sam_mask = (Y - point[0])**2 + (X - point[1])**2 <= 25**2'
            ),
        },
        # --- 3. Cell 16 (SIMULATE branch): np.ogrid uses stale `size` ---
        {
            "issue": "cell 16 SIMULATE branch: `np.ogrid[:size, :size]` uses stale synthetic global",
            "old": (
                'else:\n'
                '    Y, X = np.ogrid[:size, :size]\n'
                '    sim_mask = (Y - prompt_y)**2 + (X - prompt_x)**2 <= 25**2'
            ),
            "new": (
                'else:\n'
                f'    {SENTINEL}: derive bounds from img.shape so simulate works on any image size\n'
                '    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]\n'
                '    sim_mask = (Y - prompt_y)**2 + (X - prompt_x)**2 <= 25**2'
            ),
        },
        # --- 4. Cell 18: bbox y1/x1 clamp uses stale `size` ---
        {
            "issue": "cell 18: y1/x1 bbox clamp uses stale synthetic `size`",
            "old": (
                'y0, x0 = max(point[0] - 30, 0), max(point[1] - 30, 0)\n'
                'y1, x1 = min(point[0] + 30, size), min(point[1] + 30, size)'
            ),
            "new": (
                f'{SENTINEL}: derive bounds from img.shape so the bbox clamps to the real image\n'
                'H_img, W_img = img.shape[:2]\n'
                'y0, x0 = max(point[0] - 30, 0), max(point[1] - 30, 0)\n'
                'y1, x1 = min(point[0] + 30, H_img), min(point[1] + 30, W_img)'
            ),
        },
        # --- 5. Cell 18 (SIMULATE branch): np.ogrid uses stale `size` ---
        {
            "issue": "cell 18 SIMULATE branch: `np.ogrid[:size, :size]` uses stale synthetic global",
            "old": (
                'else:\n'
                '    Y, X = np.ogrid[:size, :size]\n'
                '    sam_box_mask = (Y >= y0) & (Y < y1) & (X >= x0) & (X < x1) & (img > 0.4)'
            ),
            "new": (
                'else:\n'
                f'    {SENTINEL}: derive bounds from img.shape\n'
                '    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]\n'
                '    sam_box_mask = (Y >= y0) & (Y < y1) & (X >= x0) & (X < x1) & (img > 0.4)'
            ),
        },
        # --- 6. Cell 20 (SIMULATE branch, INSIDE for-loop, 8-space indent): np.ogrid uses stale `size` ---
        {
            "issue": "cell 20 SIMULATE branch (inside for-loop): `np.ogrid[:size, :size]` uses stale synthetic global",
            "old": (
                '    else:\n'
                '        Y, X = np.ogrid[:size, :size]\n'
                '        sam_mask_t = (Y - test_point[0])**2 + (X - test_point[1])**2 <= 25**2'
            ),
            "new": (
                '    else:\n'
                f'        {SENTINEL}: derive bounds from img.shape\n'
                '        Y, X = np.ogrid[:img.shape[0], :img.shape[1]]\n'
                '        sam_mask_t = (Y - test_point[0])**2 + (X - test_point[1])**2 <= 25**2'
            ),
        },
    ],
}


def apply_fixes(nb: dict, fn: str, fixes: list, dry_run: bool) -> tuple[int, int]:
    """Returns (applied, skipped)."""
    applied = skipped = 0
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        for fx in fixes:
            if SENTINEL in src and fx["new"] in src:
                # Already fixed.
                skipped += 1
                continue
            if fx["old"] in src:
                if dry_run:
                    print(f"  [DRY ] {fn}: would fix '{fx['issue']}'")
                else:
                    new_src = src.replace(fx["old"], fx["new"], 1)
                    c["source"] = new_src.splitlines(keepends=True)
                    src = new_src
                    print(f"  [FIX ] {fn}: {fx['issue']}")
                applied += 1
    return applied, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total_applied = total_skipped = 0
    for fn, fixes in FIXES.items():
        nb_path = NB_DIR / fn
        if not nb_path.exists():
            print(f"[ERR ] {fn}: not found", file=sys.stderr)
            continue
        with open(nb_path) as f:
            nb = json.load(f)
        applied, skipped = apply_fixes(nb, fn, fixes, args.dry_run)
        total_applied += applied
        total_skipped += skipped
        if applied and not args.dry_run:
            with open(nb_path, "w") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
                f.write("\n")
    print()
    print(f"Done. applied={total_applied}, already-fixed={total_skipped}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
