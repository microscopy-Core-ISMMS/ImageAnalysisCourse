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
GATE_SENTINEL = "# GATE-SYNTHETIC"


# Per-NB known issues. Each entry replaces a snippet (`old`) with a fixed
# version (`new`). Matching is exact-substring; if `new` already appears in
# the cell, the fix is treated as already applied.
FIXES = {
    "01_cellpose_segmentation.ipynb": [
        # Order matters: migrate-from-old-guard first; only if no old guard is present do
        # we apply the fresh fix. Otherwise the fresh fix will substring-match inside the
        # old guard's body and produce a dangling `if`.
        {
            "issue": "cell 14 (migrate): old USE_REAL_FOR_DOWNSTREAM guard -> new real_imgs check",
            "old": (
                f'{SENTINEL}: when real-data swap is on, img_easy is already bound from real_imgs.\n'
                'if not (globals().get("USE_REAL_FOR_DOWNSTREAM") and globals().get("real_imgs")):\n'
                '    img_easy = skio.imread("sample_easy.png")'
            ),
            "new": (
                f'{SENTINEL}: read PNG only if real data is not loaded (real_imgs is None).\n'
                'if globals().get("real_imgs") is None:\n'
                '    img_easy = skio.imread("sample_easy.png")'
            ),
        },
        {
            "issue": "cell 15 (migrate): old USE_REAL_FOR_DOWNSTREAM guard -> new real_imgs check",
            "old": (
                f'{SENTINEL}: when real-data swap is on, img_hard is already bound from real_imgs.\n'
                'if not (globals().get("USE_REAL_FOR_DOWNSTREAM") and globals().get("real_imgs")):\n'
                '    img_hard = skio.imread("sample_hard.png")'
            ),
            "new": (
                f'{SENTINEL}: read PNG only if real data is not loaded (real_imgs is None).\n'
                'if globals().get("real_imgs") is None:\n'
                '    img_hard = skio.imread("sample_hard.png")'
            ),
        },
        {
            "issue": "cell 14 (fresh): skio.imread('sample_easy.png') overwrites real img_easy",
            "old": (
                'img_easy = skio.imread("sample_easy.png")\n'
                '# If the image is RGB, Cellpose expects channels in a specific format'
            ),
            "new": (
                f'{SENTINEL}: read PNG only if real data is not loaded (real_imgs is None).\n'
                'if globals().get("real_imgs") is None:\n'
                '    img_easy = skio.imread("sample_easy.png")\n'
                '# If the image is RGB, Cellpose expects channels in a specific format'
            ),
        },
        {
            "issue": "cell 15 (fresh): skio.imread('sample_hard.png') overwrites real img_hard",
            "old": (
                'img_hard = skio.imread("sample_hard.png")'
            ),
            "new": (
                f'{SENTINEL}: read PNG only if real data is not loaded (real_imgs is None).\n'
                'if globals().get("real_imgs") is None:\n'
                '    img_hard = skio.imread("sample_hard.png")'
            ),
        },
    ],
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
        # --- 7. Cell 18: replace fixed bbox with interactive percent-based sliders ---
        {
            "issue": "cell 18: convert fixed-size bbox to interactive sliders (center + size in % of image)",
            "old": (
                '# Bounding box around (roughly) the same blob\n'
                f'{SENTINEL}: derive bounds from img.shape so the bbox clamps to the real image\n'
                'H_img, W_img = img.shape[:2]\n'
                'y0, x0 = max(point[0] - 30, 0), max(point[1] - 30, 0)\n'
                'y1, x1 = min(point[0] + 30, H_img), min(point[1] + 30, W_img)\n'
                'input_box = np.array([x0, y0, x1, y1])'
            ),
            "new": (
                '# @title Bounding-box prompt — interactive sliders { run: "auto" }\n'
                f'{SENTINEL}: bbox is now controllable by 4 sliders (all in percent of image dims).\n'
                '# Defaults reproduce a small box near the auto-detected `point`.\n'
                'box_center_x_pct = 50  # @param {type: "slider", min: 0, max: 100, step: 1}\n'
                'box_center_y_pct = 50  # @param {type: "slider", min: 0, max: 100, step: 1}\n'
                'box_width_pct = 10  # @param {type: "slider", min: 1, max: 100, step: 1}\n'
                'box_height_pct = 10  # @param {type: "slider", min: 1, max: 100, step: 1}\n'
                '\n'
                'H_img, W_img = img.shape[:2]\n'
                'cx = int(box_center_x_pct / 100 * W_img)\n'
                'cy = int(box_center_y_pct / 100 * H_img)\n'
                'half_w = max(1, int(box_width_pct / 100 * W_img / 2))\n'
                'half_h = max(1, int(box_height_pct / 100 * H_img / 2))\n'
                'x0 = max(cx - half_w, 0)\n'
                'x1 = min(cx + half_w, W_img)\n'
                'y0 = max(cy - half_h, 0)\n'
                'y1 = min(cy + half_h, H_img)\n'
                'input_box = np.array([x0, y0, x1, y1])\n'
                'print(f"Image: {W_img}x{H_img} px. Box: ({x0}, {y0}) -> ({x1}, {y1}), size {x1-x0}x{y1-y0} px.")'
            ),
        },
        # --- 8. Cell 20: replace fixed shifts with slider-driven percent shift ---
        # Two old-string variants because the prior run produced a double-hash version.
        # We try the broken-current-state version first; on a fresh NB the second variant matches.
        {
            "issue": "cell 20: clean up double-hash comment from prior fix run",
            "old": (
                '# @title Move the point around — interactive shift slider { run: "auto" }\n'
                f'# {SENTINEL}: shift amount is now a slider in percent of image width.\n'
            ),
            "new": (
                '# @title Move the point around — interactive shift slider { run: "auto" }\n'
                f'{SENTINEL}: shift amount is now a slider in percent of image width.\n'
            ),
        },
        {
            "issue": "cell 20: convert fixed shift list to interactive percent-based slider",
            "old": (
                '# Move the point around and see how the mask changes\n'
                'shifts = [(-20, 0), (0, 0), (20, 0)]\n'
            ),
            "new": (
                '# @title Move the point around — interactive shift slider { run: "auto" }\n'
                f'{SENTINEL}: shift amount is now a slider in percent of image width.\n'
                '# Three panels show: -shift, 0, +shift along the x-axis from `point`.\n'
                'shift_pct = 5  # @param {type: "slider", min: 0, max: 50, step: 1}\n'
                '\n'
                '_W = img.shape[1]\n'
                'shift_px = int(shift_pct / 100 * _W)\n'
                'shifts = [(0, -shift_px), (0, 0), (0, shift_px)]  # (dy, dx) in pixels\n'
                'print(f"Image width: {_W} px. Shift: {shift_pct}% = {shift_px} px along x-axis.")\n'
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
# Cell-level gating (wraps a synthetic-generation cell so it skips when the
# real-data swap is active). Different shape from FIXES (substring replace) —
# this needs to wrap arbitrary existing content.
# ---------------------------------------------------------------------------

GATE_FIXES = {
    "01_cellpose_segmentation.ipynb": [
        {
            "issue": "cell 10: synthetic image generators (run only as fallback if real data failed)",
            "identifier": "def make_easy_image(seed=0, size=200",
            "display_synth": (
                "    # Display the freshly generated synthetic train pair\n"
                "    import matplotlib.pyplot as _plt\n"
                "    _fig, _axes = _plt.subplots(1, 2, figsize=(10, 5))\n"
                "    _axes[0].imshow(img_easy_synth, cmap='gray'); _axes[0].set_title('img_easy_synth (synthetic)', fontsize=10); _axes[0].axis('off')\n"
                "    _axes[1].imshow(img_hard_synth, cmap='gray'); _axes[1].set_title('img_hard_synth (synthetic)', fontsize=10); _axes[1].axis('off')\n"
                "    _plt.tight_layout(); _plt.show()\n"
            ),
        },
    ],
    "03b_foundation_model_segmentation.ipynb": [
        {
            "issue": "cell 10: synthetic non-canonical image generator (fallback)",
            "identifier": "rng = np.random.default_rng(7)\nsize = 256",
            "display_synth": "    # (display already in original cell — fig/ax/imshow at end of body)\n",
        },
    ],
    "09_cellpose_finetune.ipynb": [
        {
            "issue": "cell 12: synthetic labeled dataset generator (fallback)",
            "identifier": "def make_labeled_image(seed=0, size=200",
            "display_synth": (
                "    # Display the freshly generated synthetic train+test set\n"
                "    import matplotlib.pyplot as _plt\n"
                "    import numpy as _np\n"
                "    _n_show = min(8, len(train_images) + len(test_images))\n"
                "    _imgs = list(train_images[:6]) + list(test_images[:2])\n"
                "    _titles = [f'train {_i}' for _i in range(min(6, len(train_images)))] + [f'test {_i}' for _i in range(min(2, len(test_images)))]\n"
                "    _ncols = 4; _nrows = (_n_show + _ncols - 1) // _ncols\n"
                "    _fig, _axes = _plt.subplots(_nrows, _ncols, figsize=(3*_ncols, 3*_nrows))\n"
                "    for _ax, _im, _t in zip(_axes.flat, _imgs[:_n_show], _titles[:_n_show]):\n"
                "        _ax.imshow(_im, cmap='gray'); _ax.set_title(_t, fontsize=9); _ax.axis('off')\n"
                "    for _ax in _axes.flat[_n_show:]:\n"
                "        _ax.axis('off')\n"
                "    _plt.tight_layout(); _plt.show()\n"
            ),
        },
    ],
    "13_validation_case_study.ipynb": [
        {
            "issue": "cell 12: synthetic TEST_IMAGES registry creation (fallback)",
            "identifier": "def make_synth_easy(seed=0, size=200",
            "display_synth": (
                "    # Display the synthetic TEST_IMAGES entries\n"
                "    import matplotlib.pyplot as _plt\n"
                "    _entries = list(TEST_IMAGES.items())[:8]\n"
                "    _ncols = min(4, len(_entries)); _nrows = (len(_entries) + _ncols - 1) // _ncols\n"
                "    _fig, _axes = _plt.subplots(_nrows, _ncols, figsize=(3*_ncols, 3*_nrows))\n"
                "    _ax_list = _axes.flat if hasattr(_axes, 'flat') else [_axes]\n"
                "    for _ax, (_name, _val) in zip(_ax_list, _entries):\n"
                "        _img = _val[0] if isinstance(_val, tuple) else _val\n"
                "        _ax.imshow(_img, cmap='gray'); _ax.set_title(_name, fontsize=8); _ax.axis('off')\n"
                "    for _ax in list(_ax_list)[len(_entries):]:\n"
                "        _ax.axis('off')\n"
                "    _plt.tight_layout(); _plt.show()\n"
            ),
        },
    ],
}


def _new_gate_wrap(original_src: str, display_synth: str = "") -> str:
    """Build the new-style gate around an existing cell body."""
    indented_body = "\n".join(("    " + line if line else "") for line in original_src.splitlines())
    parts = [
        f"{GATE_SENTINEL}: synthetic generation runs only as a fallback when real data is not loaded.",
        "if globals().get('real_imgs') is not None:",
        "    print('Synthetic generation skipped — real data is loaded into the working variables.')",
        "else:",
        indented_body,
    ]
    if display_synth.strip():
        parts.append(display_synth.rstrip("\n"))
    return "\n".join(parts) + "\n"


def gate_cell(cell: dict, display_synth: str = "") -> bool:
    """Wrap a code cell in a real-vs-synthetic guard. Idempotent: detects either
    the old (USE_REAL_FOR_DOWNSTREAM-based) or new (real_imgs is not None) form
    and rewrites to the new form."""
    src = "".join(cell.get("source", []))
    OLD_GUARD = "if globals().get('USE_REAL_FOR_DOWNSTREAM') and globals().get('real_imgs'):"
    NEW_GUARD = "if globals().get('real_imgs') is not None:"

    if GATE_SENTINEL not in src:
        # Fresh wrap.
        cell["source"] = _new_gate_wrap(src, display_synth).splitlines(keepends=True)
        return True

    # Already wrapped — possibly with the old guard. Migrate if so.
    if OLD_GUARD in src and NEW_GUARD not in src:
        # Strip the old wrapper to recover the original body, then re-wrap with the new guard.
        # Old wrapper shape:
        #   <SENTINEL line>
        #   if globals().get('USE_REAL_FOR_DOWNSTREAM') ...
        #       print(...)
        #   else:
        #       <indented original>
        lines = src.splitlines()
        try:
            else_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "else:")
        except StopIteration:
            return False
        # Lines after `else:` are the indented original body.
        body_indented = lines[else_idx + 1:]
        body_lines = []
        for ln in body_indented:
            if ln.startswith("    "):
                body_lines.append(ln[4:])
            elif ln == "":
                body_lines.append(ln)
            else:
                body_lines.append(ln)
        original_body = "\n".join(body_lines)
        cell["source"] = _new_gate_wrap(original_body, display_synth).splitlines(keepends=True)
        return True

    # Already in new form — check if display_synth is already there.
    if display_synth.strip() and display_synth.strip() not in src:
        # Re-wrap to pick up the display block.
        # Recover the body the same way.
        lines = src.splitlines()
        try:
            else_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "else:")
        except StopIteration:
            return False
        body_indented = lines[else_idx + 1:]
        # Drop trailing display_synth if it was previously applied; otherwise the original body is everything indented.
        body_lines = []
        for ln in body_indented:
            if ln.startswith("    "):
                body_lines.append(ln[4:])
            elif ln == "":
                body_lines.append(ln)
            else:
                # Stop at the first non-indented line — likely the display block we already added previously.
                break
        original_body = "\n".join(body_lines)
        cell["source"] = _new_gate_wrap(original_body, display_synth).splitlines(keepends=True)
        return True

    return False


def apply_gates(nb: dict, fn: str, gates: list, dry_run: bool) -> tuple[int, int]:
    """Returns (applied, skipped). Idempotent — detects already-gated cells via
    indented identifier + GATE_SENTINEL combo. Migrates old-style gates to new."""
    applied = skipped = 0
    OLD_GUARD = "if globals().get('USE_REAL_FOR_DOWNSTREAM') and globals().get('real_imgs'):"
    NEW_GUARD = "if globals().get('real_imgs') is not None:"

    for g in gates:
        target_idx = None
        needs_migration = False
        needs_display = False
        already_done = False
        indented_id = "\n".join("    " + line for line in g["identifier"].splitlines())
        display_synth = g.get("display_synth", "")

        for i, c in enumerate(nb["cells"]):
            if c.get("cell_type") != "code":
                continue
            src = "".join(c.get("source", []))
            # Case A: never gated.
            if g["identifier"] in src and GATE_SENTINEL not in src:
                target_idx = i
                break
            # Case B/C/D: already gated. Identify by indented form + GATE_SENTINEL.
            if GATE_SENTINEL in src and indented_id in src:
                target_idx = i
                if OLD_GUARD in src:
                    needs_migration = True
                elif display_synth.strip() and display_synth.strip() not in src:
                    needs_display = True
                else:
                    already_done = True
                break

        if target_idx is None:
            print(f"  [WARN] {fn}: gate target not found for '{g['issue']}'")
            continue
        if already_done:
            skipped += 1
            continue
        action = "MIGRATE" if needs_migration else ("ADD-DISPLAY" if needs_display else "GATE")
        if dry_run:
            print(f"  [DRY ] {fn}: would {action.lower()} cell {target_idx} — {g['issue']}")
            applied += 1
            continue
        if gate_cell(nb["cells"][target_idx], display_synth):
            print(f"  [{action}] {fn}: cell {target_idx} — {g['issue']}")
            applied += 1
        else:
            skipped += 1
    return applied, skipped


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
    # Combine substring-fix and gate targets so one pass-through writes the file once.
    nbs_to_process = sorted(set(list(FIXES.keys()) + list(GATE_FIXES.keys())))
    for fn in nbs_to_process:
        nb_path = NB_DIR / fn
        if not nb_path.exists():
            print(f"[ERR ] {fn}: not found", file=sys.stderr)
            continue
        with open(nb_path) as f:
            nb = json.load(f)
        applied = skipped = 0
        if fn in FIXES:
            a, s = apply_fixes(nb, fn, FIXES[fn], args.dry_run)
            applied += a
            skipped += s
        if fn in GATE_FIXES:
            a, s = apply_gates(nb, fn, GATE_FIXES[fn], args.dry_run)
            applied += a
            skipped += s
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
