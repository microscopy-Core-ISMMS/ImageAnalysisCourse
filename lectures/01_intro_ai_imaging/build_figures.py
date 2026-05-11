"""
Build all matplotlib figures for the Lecture 1 slides.

Generates synthetic-data PNG figures into figures/ subfolder. The lecture
build script (build_lecture_notebook.py) reads these and embeds them as
base64 data URLs so the exported HTML is fully self-contained.

Run:
    python build_figures.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from scipy.ndimage import gaussian_filter

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Common style — clean for slides
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
})

NAVY = "#1F4E79"
NAVY_LIGHT = "#2E75B6"
ACCENT = "#C44E52"
GOOD = "#55A868"
BAD = "#C44E52"

INSTANCE_CMAP = ListedColormap(["black"] + plt.get_cmap("tab20")(np.linspace(0, 1, 20)).tolist())


def save(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  wrote {path.name} ({path.stat().st_size//1024} KB)")


# --------------------------------------------------------------------------
# Synthetic data helpers
# --------------------------------------------------------------------------
def synth_cells(size=200, n=12, seed=0, irregular=False):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=float)
    masks = np.zeros((size, size), dtype=int)
    centers = rng.uniform(20, size-20, (n, 2))
    radii = rng.uniform(8, 16, n)
    for i, ((cy, cx), r) in enumerate(zip(centers, radii), start=1):
        Y, X = np.ogrid[:size, :size]
        if irregular:
            ry = r * rng.uniform(0.6, 1.4)
            rx = r * rng.uniform(0.6, 1.4)
            mask = ((Y-cy)/ry)**2 + ((X-cx)/rx)**2 <= 1
        else:
            mask = (Y-cy)**2 + (X-cx)**2 <= r**2
        img[mask] = rng.uniform(0.6, 1.0)
        masks[mask] = i
    img = gaussian_filter(img, sigma=0.8)
    img += rng.normal(0, 0.05, img.shape)
    img = np.clip(img, 0, 1)
    return img, masks


# --------------------------------------------------------------------------
# Section 2 — task taxonomy (7 figures)
# --------------------------------------------------------------------------
def fig_classification():
    img, _ = synth_cells(size=180, n=4, seed=2)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.imshow(img, cmap="gray")
    ax.text(0.5, -0.08, "Classification → label per image: phenotype A · mitosis: yes",
            ha="center", va="top", transform=ax.transAxes, color=NAVY, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(NAVY); s.set_linewidth(1.5)
    save(fig, "task_classification")


def fig_detection():
    img, _ = synth_cells(size=200, n=8, seed=3)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.imshow(img, cmap="gray")
    rng = np.random.default_rng(3)
    centers = rng.uniform(30, 170, (5, 2))
    for cy, cx in centers:
        rect = Rectangle((cx-15, cy-15), 30, 30, linewidth=2, edgecolor=ACCENT, facecolor="none")
        ax.add_patch(rect)
    ax.text(0.5, -0.08, "Detection → bounding boxes around objects (e.g., FISH spots)",
            ha="center", va="top", transform=ax.transAxes, color=NAVY, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(NAVY); s.set_linewidth(1.5)
    save(fig, "task_detection")


def fig_segmentation():
    img, masks = synth_cells(size=200, n=12, seed=4)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Image", color=NAVY, fontweight="bold")
    axes[1].imshow(masks, cmap=INSTANCE_CMAP, vmin=0, vmax=20); axes[1].set_title("Instance masks", color=NAVY, fontweight="bold")
    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(np.where(masks > 0, masks, np.nan), cmap=INSTANCE_CMAP, vmin=0, vmax=20, alpha=0.55)
    axes[2].set_title("Overlay", color=NAVY, fontweight="bold")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Segmentation → assign every pixel to an object", color=NAVY, fontsize=13, fontweight="bold", y=0.04)
    save(fig, "task_segmentation")


def fig_restoration():
    """Noisy → restored, 16:9-sized for projector readability. Higher photon-count
    contrast so the difference between panels is dramatic."""
    img, _ = synth_cells(size=240, n=10, seed=5)
    rng = np.random.default_rng(5)
    photons = 8  # lower photon count → more visible noise
    noisy = img * photons
    noisy = rng.poisson(np.clip(noisy, 0, None)).astype(float) / photons
    noisy = noisy + rng.normal(0, 0.15, noisy.shape)
    noisy = np.clip(noisy, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Noisy input (low light)", color=NAVY, fontweight="bold", fontsize=14)
    axes[1].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Restored output (model prediction)", color=NAVY, fontweight="bold", fontsize=14)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Restoration → image-to-image: denoising, deconvolution, super-resolution",
                 color=NAVY, fontsize=15, fontweight="bold", y=1.0)
    save(fig, "task_restoration")


def fig_generation():
    """Brightfield → predicted fluorescence, 16:9-sized with a phase-contrast-style
    transform on the BF panel so cells are visibly dim with bright halos."""
    img, _ = synth_cells(size=240, n=12, seed=6)
    # Phase-contrast-like 'brightfield': dim cells with bright halos
    smooth = gaussian_filter(img, sigma=2.2)
    edge = img - smooth                       # band-pass = halo
    bf = 0.7 - 0.4 * img + 0.6 * edge
    bf = np.clip(gaussian_filter(bf, sigma=0.5), 0, 1)
    # Generated output: predicted fluorescence (use viridis-like colormap)
    gen = img.copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].imshow(bf, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Brightfield input (no labels)", color=NAVY, fontweight="bold", fontsize=14)
    axes[1].imshow(gen, cmap="viridis")
    axes[1].set_title("Predicted fluorescence (generated)", color=NAVY, fontweight="bold", fontsize=14)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Generation → in silico labeling, virtual staining, synthetic data",
                 color=NAVY, fontsize=15, fontweight="bold", y=1.0)
    save(fig, "task_generation")


def fig_registration():
    """4-panel: Image A | Image B (shifted) | overlay before | overlay after alignment.
    16:9-sized so the before-vs-after contrast is visible from the back of the room."""
    img1, _ = synth_cells(size=200, n=8, seed=7)
    shift_y, shift_x = 14, 10
    img2 = np.roll(img1, shift_y, axis=0)
    img2 = np.roll(img2, shift_x, axis=1)
    # Overlay before alignment: A in red channel, B (shifted) in blue channel
    overlay_before = np.zeros((*img1.shape, 3))
    overlay_before[..., 0] = img1
    overlay_before[..., 2] = img2
    # Overlay after alignment: undo the shift on B → both channels co-located → white
    img2_aligned = np.roll(img2, -shift_y, axis=0)
    img2_aligned = np.roll(img2_aligned, -shift_x, axis=1)
    overlay_after = np.zeros((*img1.shape, 3))
    overlay_after[..., 0] = img1
    overlay_after[..., 2] = img2_aligned

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(img1, cmap="Reds_r"); axes[0].set_title("Image A", color=NAVY, fontweight="bold", fontsize=13)
    axes[1].imshow(img2, cmap="Blues_r"); axes[1].set_title("Image B (shifted)", color=NAVY, fontweight="bold", fontsize=13)
    axes[2].imshow(overlay_before); axes[2].set_title("Overlay BEFORE alignment\n(red/blue split visible)", color=BAD, fontweight="bold", fontsize=13)
    axes[3].imshow(overlay_after); axes[3].set_title("Overlay AFTER alignment\n(red/blue overlap → magenta)", color=GOOD, fontweight="bold", fontsize=13)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Registration → align images to a common spatial reference",
                 color=NAVY, fontsize=15, fontweight="bold", y=1.0)
    save(fig, "task_registration")


def fig_tracking():
    rng = np.random.default_rng(8)
    n_tracks = 6
    n_frames = 30
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i in range(n_tracks):
        # Random walk trajectory
        x_start = rng.uniform(0.1, 0.9)
        y_start = rng.uniform(0.1, 0.9)
        xs = [x_start]; ys = [y_start]
        for _ in range(n_frames - 1):
            xs.append(xs[-1] + rng.normal(0, 0.015))
            ys.append(ys[-1] + rng.normal(0, 0.015))
        color = plt.cm.tab10(i)
        ax.plot(xs, ys, "-o", color=color, markersize=3, linewidth=1.5, alpha=0.8)
        ax.plot(xs[0], ys[0], "o", color=color, markersize=8, markeredgecolor="black", markeredgewidth=1)
        ax.plot(xs[-1], ys[-1], "s", color=color, markersize=8, markeredgecolor="black", markeredgewidth=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(NAVY); s.set_linewidth(1.5)
    ax.set_title("Tracking → connect objects across time (cell lineages, particles)",
                 color=NAVY, fontweight="bold", fontsize=12)
    # Legend explaining markers
    ax.plot([], [], "o", color="gray", markersize=8, markeredgecolor="black", label="t = 0 (start)")
    ax.plot([], [], "s", color="gray", markersize=8, markeredgecolor="black", label="t = T (end)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    save(fig, "task_tracking")


# --------------------------------------------------------------------------
# Section 4 — failure mode figures (3)
# --------------------------------------------------------------------------
def fig_failure_domain_shift():
    """Same model on two different sample types. Predictions are SUBSETS of
    each image's true cell labels, so the colored masks always sit on real
    cells — no floating overlays in empty regions."""
    # In-distribution: round, isolated cells. AI gets all of them.
    img_easy, masks_easy = synth_cells(size=160, n=10, seed=11, irregular=False)
    pred_easy = masks_easy
    n_easy = int((np.unique(pred_easy) != 0).sum())

    # Out-of-distribution: irregular, dense. AI misses ~40% — randomly drop labels.
    img_hard, masks_hard = synth_cells(size=160, n=10, seed=12, irregular=True)
    ids_hard = np.array([k for k in np.unique(masks_hard) if k != 0])
    rng = np.random.default_rng(99)
    n_keep = min(6, len(ids_hard))
    keep_hard = set(rng.choice(ids_hard, size=n_keep, replace=False).tolist())
    pred_hard = np.where(np.isin(masks_hard, list(keep_hard)), masks_hard, 0)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes[0, 0].imshow(img_easy, cmap="gray")
    axes[0, 0].set_title("In-distribution image\n(round, isolated cells)",
                         color=NAVY, fontweight="bold", fontsize=11)
    axes[0, 1].imshow(img_easy, cmap="gray")
    axes[0, 1].imshow(np.where(pred_easy > 0, pred_easy, np.nan),
                      cmap=INSTANCE_CMAP, vmin=0, vmax=20, alpha=0.55)
    axes[0, 1].set_title(f"Prediction: {n_easy} objects (correct)",
                         color=GOOD, fontweight="bold", fontsize=11)
    axes[1, 0].imshow(img_hard, cmap="gray")
    axes[1, 0].set_title("Out-of-distribution image\n(irregular, dense)",
                         color=NAVY, fontweight="bold", fontsize=11)
    axes[1, 1].imshow(img_hard, cmap="gray")
    axes[1, 1].imshow(np.where(pred_hard > 0, pred_hard, np.nan),
                      cmap=INSTANCE_CMAP, vmin=0, vmax=20, alpha=0.55)
    axes[1, 1].set_title(f"Prediction: {n_keep} of {len(ids_hard)} objects (under-counts)",
                         color=BAD, fontweight="bold", fontsize=11)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Domain shift: same model, different sample type → silent under-counting",
                 color=NAVY, fontsize=13, fontweight="bold", y=1.0)
    save(fig, "failure_domain_shift")


def fig_failure_hallucination():
    # Synthetic: a noisy image where restoration "invents" features
    rng = np.random.default_rng(14)
    clean = np.zeros((160, 160))
    for cy, cx, r in [(40, 40, 12), (110, 100, 14)]:
        Y, X = np.ogrid[:160, :160]
        clean[(Y-cy)**2 + (X-cx)**2 <= r**2] = 0.9
    clean = gaussian_filter(clean, sigma=1.2)
    noisy = clean + rng.normal(0, 0.25, clean.shape)
    noisy = np.clip(noisy, 0, 1)
    # "Restored" with extra invented blob
    restored = clean.copy()
    Y, X = np.ogrid[:160, :160]
    invented = (Y-80)**2 + (X-130)**2 <= 9**2
    restored[invented] = 0.6  # the hallucinated feature
    restored = gaussian_filter(restored, sigma=0.8)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Noisy input", color=NAVY, fontweight="bold")
    axes[1].imshow(restored, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Restored (looks plausible)", color=NAVY, fontweight="bold")
    axes[2].imshow(clean, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Ground truth (real signal)", color=NAVY, fontweight="bold")
    # Mark the hallucinated region in the restored panel
    axes[1].add_patch(Circle((130, 80), 18, fill=False, edgecolor=BAD, linewidth=2.5))
    axes[1].text(130, 105, "invented", ha="center", color=BAD, fontweight="bold", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Hallucination in restoration: a feature that was never there", color=NAVY, fontsize=13, fontweight="bold", y=0.02)
    save(fig, "failure_hallucination")


def fig_failure_misregistration():
    """16:9-sized, sharper local-distortion artifact so the failure mode is
    obvious at slide scale."""
    img1, _ = synth_cells(size=220, n=8, seed=15)
    Y, X = np.indices(img1.shape, dtype=float)
    cy, cx = img1.shape[0] / 2, img1.shape[1] / 2
    # Stronger local rotation + shift to make the failure more visible
    theta = (np.sqrt((Y-cy)**2 + (X-cx)**2) / (img1.shape[0] / 2)) * 0.25
    Y2 = cy + (Y-cy) * np.cos(theta) - (X-cx) * np.sin(theta) + 12
    X2 = cx + (Y-cy) * np.sin(theta) + (X-cx) * np.cos(theta) + 8
    Y2 = np.clip(Y2, 0, img1.shape[0]-1).astype(int)
    X2 = np.clip(X2, 0, img1.shape[1]-1).astype(int)
    img2_misreg = img1[Y2, X2]

    overlay_good = np.zeros((*img1.shape, 3))
    overlay_good[..., 0] = img1
    overlay_good[..., 2] = img1
    overlay_bad = np.zeros((*img1.shape, 3))
    overlay_bad[..., 0] = img1
    overlay_bad[..., 2] = img2_misreg

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(overlay_good)
    axes[0].set_title("Successful registration\n(structures align → magenta)", color=GOOD, fontweight="bold", fontsize=14)
    axes[1].imshow(overlay_bad)
    axes[1].set_title("Failed registration\n(local distortion → red/blue ghosting)", color=BAD, fontweight="bold", fontsize=14)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Misregistration: alignment looks reasonable globally but distorts locally",
                 color=NAVY, fontsize=15, fontweight="bold", y=1.0)
    save(fig, "failure_misregistration")


# --------------------------------------------------------------------------
# Section 5 — metrics-vs-biology figure
# --------------------------------------------------------------------------
# ---- Shared helpers for the metrics-vs-biology pair ---------------------
METRIC_BLUE = "#4C72B0"   # boundary error
METRIC_RED  = "#C44E52"   # mask merge
METRIC_GRAY = "#888888"   # mixed


def _metrics_four_cell_scene(size=160, seed=200):
    """Stable 4-cell scene used by both metric-error figures.
    Returns (image, ground-truth instance masks).
    Bottom pair sits close enough to motivate the 'merge' error."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size))
    masks = np.zeros((size, size), dtype=int)
    centers = [(45, 45), (45, 115), (115, 60), (115, 105)]
    radii = [16, 16, 18, 18]
    Y, X = np.ogrid[:size, :size]
    for i, ((cy, cx), r) in enumerate(zip(centers, radii), start=1):
        m = (Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2
        img[m] = rng.uniform(0.75, 1.0)
        masks[m] = i
    img = gaussian_filter(img, sigma=0.7)
    img += rng.normal(0, 0.04, img.shape)
    return np.clip(img, 0, 1), masks


def _erode_jagged(masks, iterations=6, keep_rim_prob=0.35, seed=11):
    """Heavily erode each instance and roughen with random rim bumps. Default
    parameters produce visibly smaller, jagged masks (NOT subtle ones)."""
    from scipy.ndimage import binary_erosion, binary_dilation
    rng = np.random.default_rng(seed)
    pred = np.zeros_like(masks)
    for lbl in np.unique(masks):
        if lbl == 0:
            continue
        m = masks == lbl
        eroded = binary_erosion(m, iterations=iterations)
        # Add random outer bumps so the boundary looks ragged (not just shrunken)
        bumpy_band = binary_dilation(eroded, iterations=2) & ~eroded
        keep_bumps = rng.random(bumpy_band.shape) > (1.0 - keep_rim_prob)
        pred[eroded | (bumpy_band & keep_bumps)] = lbl
    return pred


def _bridge_merge(masks, label_a=4, label_b=3, bridge_radius=12):
    """Merge two adjacent instances into a single CONTIGUOUS blob: relabel +
    fill the strip between centroids with the same label. Reads as one big
    mask, not two separate cells sharing a color."""
    out = masks.copy()
    if not ((out == label_a).any() and (out == label_b).any()):
        return out
    coords_a = np.argwhere(masks == label_a)
    coords_b = np.argwhere(masks == label_b)
    cy_a, cx_a = coords_a.mean(axis=0)
    cy_b, cx_b = coords_b.mean(axis=0)
    out[out == label_a] = label_b
    Y, X = np.ogrid[:out.shape[0], :out.shape[1]]
    for t in np.linspace(0, 1, 60):
        cy = cy_a + t * (cy_b - cy_a)
        cx = cx_a + t * (cx_b - cx_a)
        disk = (Y - cy) ** 2 + (X - cx) ** 2 <= bridge_radius ** 2
        out[disk] = label_b
    return out


# Keep old name as alias for any external callers
_merge_pair = _bridge_merge


def _category_cmap(base_color):
    """4-shade palette derived from a single category color. Index 0 = transparent,
    1..4 = darker→lighter variations of the base color so per-instance distinction
    is preserved while the panel reads as one error type."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(base_color)
    # Generate 4 shades by mixing with white at varying ratios
    shades = []
    for t in [0.0, 0.25, 0.5, 0.7]:
        shades.append((r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t))
    return ListedColormap(["black"] + shades)


def _draw_gt_contours(ax, gt_masks, color="white"):
    for lbl in np.unique(gt_masks):
        if lbl == 0:
            continue
        ax.contour(gt_masks == lbl, levels=[0.5], colors=color,
                   linestyles="--", linewidths=1.0)


def fig_metrics_error_examples():
    """Pre-slide figure that defines the visual vocabulary for the scatter.
    Three panels: boundary errors, mask merges, mixed. Each panel shows the
    same 4-cell scene with the AI's prediction overlay + dashed white
    contours marking ground truth."""
    img, gt = _metrics_four_cell_scene()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6))
    fig.suptitle("Three error patterns behind the colored dots   "
                 "(dashed white = ground truth, filled = AI prediction)",
                 color=NAVY, fontsize=13.5, fontweight="bold", y=0.97)

    # DRAMATIC differences so the panels are easy to tell apart at a glance:
    # - boundary: heavy erosion + jagged bumps (predicted masks are visibly
    #   smaller and rougher than the dashed GT contour)
    # - merge: bottom pair becomes ONE contiguous blob via _bridge_merge
    # - mixed: both at once
    pred1 = _erode_jagged(gt, iterations=6, keep_rim_prob=0.35, seed=11)
    pred2 = _bridge_merge(gt, label_a=4, label_b=3, bridge_radius=14)
    pred3 = _bridge_merge(
        _erode_jagged(gt, iterations=5, keep_rim_prob=0.30, seed=22),
        label_a=4, label_b=3, bridge_radius=12,
    )

    panels = [
        (axes[0], pred1, METRIC_BLUE, _category_cmap(METRIC_BLUE),
         "Boundary errors only\nIoU drops  ·  count unchanged (4 cells → 4 masks)"),
        (axes[1], pred2, METRIC_RED, _category_cmap(METRIC_RED),
         "Mask merges only\nIoU still high  ·  count wrong (4 cells → 3 masks)"),
        (axes[2], pred3, METRIC_GRAY, _category_cmap(METRIC_GRAY),
         "Mixed errors\nboth effects active (4 cells → 3 masks)"),
    ]
    for ax, pred, color, cmap, title in panels:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.imshow(np.where(pred > 0, pred, np.nan),
                  cmap=cmap, vmin=0, vmax=4, alpha=0.78)
        _draw_gt_contours(ax, gt)
        ax.set_title(title, color=color, fontweight="bold", fontsize=12, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(color); s.set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    save(fig, "metrics_error_examples")


def fig_metrics_vs_biology():
    """Scatter of IoU vs absolute cell-count error, with three small inset
    thumbnails on the right column showing what each color category actually
    looks like (boundary error / mask merge / mixed)."""
    rng = np.random.default_rng(0)

    # Mode A: boundary errors only — IoU drops, count holds
    n_a = 30
    erosion_a = rng.uniform(0.02, 0.25, n_a)
    ious_a = 1.0 - erosion_a + rng.normal(0, 0.01, n_a)
    count_err_a = rng.poisson(0.3, n_a).astype(float) + rng.normal(0, 0.1, n_a)

    # Mode B: mask merges only — IoU stays near 1, count drops
    n_b = 30
    merges_b = rng.integers(0, 5, n_b)
    ious_b = 0.97 + rng.normal(0, 0.012, n_b)
    count_err_b = merges_b + rng.poisson(0.3, n_b)

    # Mode C: mixed — both effects, scattered
    n_c = 25
    erosion_c = rng.uniform(0, 0.15, n_c)
    merges_c = rng.integers(0, 3, n_c)
    ious_c = 1.0 - erosion_c + rng.normal(0, 0.012, n_c)
    count_err_c = merges_c + rng.poisson(0.3, n_c).astype(float)

    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(3, 5, width_ratios=[2.6, 2.6, 2.6, 0.25, 1.05],
                          hspace=0.45, wspace=0.05,
                          left=0.07, right=0.98, top=0.90, bottom=0.10)
    ax = fig.add_subplot(gs[:, :3])
    ax_b = fig.add_subplot(gs[0, 4])
    ax_m = fig.add_subplot(gs[1, 4])
    ax_x = fig.add_subplot(gs[2, 4])

    ax.scatter(ious_a, count_err_a, c=METRIC_BLUE, s=70, edgecolor="black",
               linewidth=0.4, label="Boundary errors only (hurts IoU)", zorder=3)
    ax.scatter(ious_b, count_err_b, c=METRIC_RED, s=70, edgecolor="black",
               linewidth=0.4, label="Mask merges only (hurts count)", zorder=3)
    ax.scatter(ious_c, count_err_c, c=METRIC_GRAY, s=45, edgecolor="black",
               linewidth=0.4, alpha=0.7, label="Mixed errors", zorder=2)

    ax.set_xlabel("IoU (pixel overlap with ground truth)", fontsize=12)
    ax.set_ylabel("Absolute cell-count error", fontsize=12)
    ax.set_title("High IoU does not guarantee a correct biological count",
                 color=NAVY, fontweight="bold", fontsize=13, pad=10)
    ax.invert_xaxis()
    ax.set_ylim(-0.6, 5.6)
    ax.set_xlim(1.02, 0.73)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9.5)
    ax.annotate("At IoU = 0.97,\ncount error ranges 0–4.\nThe gap is real.",
                xy=(0.965, 3.6), xytext=(0.83, 4.4),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5),
                fontsize=11, color=ACCENT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=ACCENT, lw=1.2))
    for s in ax.spines.values():
        s.set_color(NAVY)

    # ---- Inset thumbnails (same predictions + palettes as the pre-slide) ---
    img_t, gt_t = _metrics_four_cell_scene()
    pred_b = _erode_jagged(gt_t, iterations=6, keep_rim_prob=0.35, seed=11)
    pred_m = _bridge_merge(gt_t, label_a=4, label_b=3, bridge_radius=14)
    pred_x = _bridge_merge(
        _erode_jagged(gt_t, iterations=5, keep_rim_prob=0.30, seed=22),
        label_a=4, label_b=3, bridge_radius=12,
    )

    for ax_th, pred, color, title in [
        (ax_b, pred_b, METRIC_BLUE, "Boundary"),
        (ax_m, pred_m, METRIC_RED,  "Merge"),
        (ax_x, pred_x, METRIC_GRAY, "Mixed"),
    ]:
        ax_th.imshow(img_t, cmap="gray", vmin=0, vmax=1)
        ax_th.imshow(np.where(pred > 0, pred, np.nan),
                     cmap=_category_cmap(color), vmin=0, vmax=4, alpha=0.82)
        _draw_gt_contours(ax_th, gt_t)
        ax_th.set_title(title, color=color, fontsize=10, fontweight="bold", pad=2)
        ax_th.set_xticks([]); ax_th.set_yticks([])
        for s in ax_th.spines.values():
            s.set_color(color); s.set_linewidth(1.8)

    save(fig, "metrics_vs_biology")


# --------------------------------------------------------------------------
# Section 1 — ecosystem visual
# --------------------------------------------------------------------------
def fig_ecosystem():
    # 16:9-friendly canvas; reserve a left gutter for the "stack matures bottom-up"
    # arrow + label so it never collides with the rectangles.
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")

    layers = [
        ("Hardware: GPUs, cloud compute, HPC clusters", 5.0, "#E8E8E8"),
        ("Data: BBBC · IDR · BioImage Archive · Allen Cell Image Library", 3.7, "#D5E8F0"),
        ("Models: pretrained Cellpose-SAM · μSAM · CARE · VoxelMorph · ...", 2.4, "#A6CEE3"),
        ("Platforms: ZeroCostDL4Mic · DL4MicEverywhere · BioImage Model Zoo", 1.1, "#1F78B4"),
    ]
    # Layout: left gutter 0.0–1.6 reserved for arrow + label; rectangles span 1.8–13.5
    for label, y, color in layers:
        rect = Rectangle((1.8, y-0.45), 11.7, 0.9, facecolor=color, edgecolor=NAVY, linewidth=1.2)
        ax.add_patch(rect)
        text_color = "white" if color == "#1F78B4" else NAVY
        ax.text(7.65, y, label, ha="center", va="center", fontsize=12, color=text_color, fontweight="bold")
    # Top title
    ax.text(7.65, 5.85, "Layered ecosystem that made AI-in-microscopy a routine tool",
            ha="center", color=NAVY, fontsize=13.5, fontweight="bold")
    # Up arrow in the left gutter (separated from rectangles)
    ax.annotate("", xy=(1.0, 5.3), xytext=(1.0, 0.7),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=2))
    ax.text(0.5, 3, "stack matures\nbottom-up", rotation=90, ha="center", va="center",
            color=NAVY, fontsize=10, fontweight="bold")
    save(fig, "ecosystem_layers")


# --------------------------------------------------------------------------
# Section 3 — train/val/test diagram
# --------------------------------------------------------------------------
def fig_train_val_test():
    # 16:9-friendly canvas. VAL/TEST captions wrap to two lines so the narrow blocks
    # contain their captions without crossing into each other.
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.5); ax.axis("off")
    # Three blocks: 70 / 15 / 15.
    blocks = [
        ("TRAIN (70%)", 0.5, 9.1, "#4C72B0", "model fits\nto this data"),
        ("VAL (15%)",   9.7, 1.95, "#DD8452", "tune\nhyperparameters"),
        ("TEST (15%)", 11.7, 1.95, "#55A868", "report once,\nat the end"),
    ]
    for label, x, w, color, sub in blocks:
        rect = Rectangle((x, 2.0), w, 1.5, facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, 2.75, label, ha="center", va="center",
                color="white", fontsize=13, fontweight="bold")
        # Caption directly below each block, two lines so VAL/TEST captions fit inside
        # their narrow column without spilling sideways into the neighbor.
        ax.text(x + w/2, 1.75, sub, ha="center", va="top",
                color=NAVY, fontsize=10, style="italic")
    # Top title
    ax.text(7, 4.1, "The three subsets are disjoint. Touch the test set once.",
            ha="center", color=NAVY, fontsize=13, fontweight="bold")
    save(fig, "train_val_test")


# --------------------------------------------------------------------------
# Section 5 — validation gradient
# --------------------------------------------------------------------------
def fig_validation_gradient():
    # 16:9-friendly canvas. Tick labels pushed clearly below the gradient bar
    # with extra y-room so multi-line labels never touch the bar at any width.
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off")
    # Gradient bar spans 1.0 to 13.0
    n_segments = 120
    cmap = plt.cm.RdYlGn
    bar_x0, bar_x1, bar_y, bar_h = 1.0, 13.0, 2.0, 0.55
    bar_w_total = bar_x1 - bar_x0
    for i in range(n_segments):
        rect = Rectangle((bar_x0 + i*bar_w_total/n_segments, bar_y), bar_w_total/n_segments, bar_h,
                         facecolor=cmap(i/n_segments), edgecolor="none")
        ax.add_patch(rect)
    rect = Rectangle((bar_x0, bar_y), bar_w_total, bar_h, fill=False, edgecolor=NAVY, linewidth=1.5)
    ax.add_patch(rect)
    # Tick positions distributed along the bar
    points = [
        (1.0,  "Internal\n(single dataset)"),
        (4.0,  "Multi-condition\nor cross-validation"),
        (7.0,  "External\n(other lab/scanner)"),
        (10.0, "Multi-site\nprospective"),
        (13.0, "Regulatory\nclearance"),
    ]
    for x, label in points:
        ax.plot([x, x], [bar_y - 0.15, bar_y + bar_h + 0.15], color=NAVY, linewidth=1.2)
        # Labels below the bar with generous spacing
        ax.text(x, 1.1, label, ha="center", va="top", fontsize=10, color=NAVY, fontweight="bold")
    # Endpoint labels ABOVE the bar; arrow on a SEPARATE y-row so the two
    # never share a horizontal line (the previous version had the arrow at
    # exactly the label y, producing a visible collision).
    ax.text(1.0, 3.30, "Research-grade", ha="left", color=NAVY,
            fontsize=11.5, fontweight="bold")
    ax.text(13.0, 3.30, "Clinical-grade", ha="right", color=NAVY,
            fontsize=11.5, fontweight="bold")
    # Arrow lives below the labels and above the bar top (bar_y+bar_h = 2.55)
    ax.annotate("", xy=(12.3, 2.85), xytext=(1.7, 2.85),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=2))
    ax.text(7, 3.80, "Validation gradient: the rigor required scales with the stakes",
            ha="center", color=NAVY, fontsize=13, fontweight="bold")
    save(fig, "validation_gradient")


# --------------------------------------------------------------------------
# viz_* figures — added 2026-05-11 (Phase 3 Group A). The original 24 viz_*
# PNGs in figures/figures/ have no preserved source code; these regenerate
# only the figures that need fixes (text overflow, etc.) per the audit.
# --------------------------------------------------------------------------

# Brand palette for the viz_* figures (matches the previous renderings)
VIZ_FOUNDATIONS = "#1FB8E5"   # cyan
VIZ_MECHANICS = "#6E6E6E"     # gray
VIZ_JUDGMENT = "#E03A8C"      # magenta
VIZ_WRAP = "#3A1F5C"          # deep purple


def fig_viz_roadmap():
    """Eight-section lecture roadmap. 8 rounded-rectangle nodes in one row with
    grouping labels beneath. Text must stay INSIDE each node (the original
    rendering had several boxes where 'How they learn' and 'Reproducibility'
    spilled past the box edge)."""
    fig, ax = plt.subplots(figsize=(16, 3.2))
    ax.set_xlim(0, 16); ax.set_ylim(0, 3.2); ax.axis("off")

    nodes = [
        (1,  "Why now?",        VIZ_FOUNDATIONS, "foundations"),
        (2,  "Tasks",            VIZ_FOUNDATIONS, "foundations"),
        (3,  "Models",           VIZ_FOUNDATIONS, "foundations"),
        (4,  "How they\nlearn",  VIZ_MECHANICS,   "mechanics"),
        (5,  "Works/\nfails",    VIZ_JUDGMENT,    "judgment"),
        (6,  "Validation",       VIZ_JUDGMENT,    "judgment"),
        (7,  "Reprod-\nucibility", VIZ_JUDGMENT,  "judgment"),
        (8,  "Closing",          VIZ_WRAP,        "wrap"),
    ]
    # Layout: 8 boxes across 16 units. Each box width = 1.65 units; gap = 0.13 units.
    n = len(nodes)
    box_w = 1.65
    gap = 0.13
    total_w = n * box_w + (n - 1) * gap
    margin = (16 - total_w) / 2
    box_h = 1.6
    box_y = 1.0

    for i, (num, label, color, _grp) in enumerate(nodes):
        x = margin + i * (box_w + gap)
        # Rounded rectangle (fill)
        from matplotlib.patches import FancyBboxPatch
        bbox = FancyBboxPatch((x, box_y), box_w, box_h,
                              boxstyle="round,pad=0.02,rounding_size=0.15",
                              facecolor="#F4FBFF" if color == VIZ_FOUNDATIONS else
                                       "#F2F2F2" if color == VIZ_MECHANICS else
                                       "#FCE8F2" if color == VIZ_JUDGMENT else
                                       "#EBE6F2",
                              edgecolor=color, linewidth=2.5)
        ax.add_patch(bbox)
        # Big number in the top-left of each box
        ax.text(x + box_w*0.22, box_y + box_h*0.72, str(num),
                ha="center", va="center", fontsize=22, fontweight="bold", color=color)
        # Label centered in lower portion; fontsize sized to fit comfortably
        ax.text(x + box_w*0.55, box_y + box_h*0.32, label,
                ha="center", va="center", fontsize=11, color="#222", fontweight="bold",
                linespacing=1.0)
        # Arrow to the next node
        if i < n - 1:
            arrow_x0 = x + box_w + 0.005
            arrow_x1 = x + box_w + gap - 0.005
            ax.annotate("", xy=(arrow_x1, box_y + box_h/2), xytext=(arrow_x0, box_y + box_h/2),
                        arrowprops=dict(arrowstyle="->", color="#888", lw=1.5))

    # Group labels below
    grp_centers = {}
    for i, (_n, _l, color, grp) in enumerate(nodes):
        cx = margin + i * (box_w + gap) + box_w/2
        grp_centers.setdefault(grp, []).append((cx, color))
    for grp, entries in grp_centers.items():
        xs = [e[0] for e in entries]
        color = entries[0][1]
        ax.text(sum(xs)/len(xs), 0.45, grp, ha="center", va="center",
                fontsize=12, style="italic", color=color)

    save(fig, "viz_roadmap")


def fig_viz_s5_patterns():
    """Failure-mode taxonomy. 5 numbered rows with title/description on the left and
    a 'Check: ...' actionable prompt on the right. Original rendering had text
    overflowing the right-column boxes."""
    fig, ax = plt.subplots(figsize=(16, 7.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8); ax.axis("off")

    rows = [
        ("1", "Out-of-distribution",
         "Test images differ from training (modality, sample type, magnification).",
         "Inspect predictions on a\nheld-out OOD strip."),
        ("2", "Rare-category miss",
         "Long-tail classes silently dropped — reviewer never sees them.",
         "Stratified per-class accuracy\n+ confusion matrix."),
        ("3", "Sample-prep drift",
         "Stain, illumination, or fixation shift over time invalidates the model.",
         "Track sample QC metrics\n+ dates."),
        ("4", "Edge effects",
         "Border tiles or border cells lose context → systematic bias.",
         "Compare interior vs\nborder statistics."),
        ("5", "Hallucinated features",
         "Restoration / generative models invent biology that isn't there.",
         "Disclose method in caption;\nreviewer checks raw."),
    ]

    # Title
    ax.text(8, 7.6, "Failure-mode taxonomy — five patterns to recognize",
            ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)

    # Each row
    row_h = 1.25
    row_top = 6.9
    for i, (num, title, body, check) in enumerate(rows):
        y_top = row_top - i * row_h
        y_mid = y_top - row_h/2

        # Number bubble (left)
        bubble = Circle((0.7, y_mid), 0.32, facecolor=ACCENT, edgecolor=ACCENT)
        ax.add_patch(bubble)
        ax.text(0.7, y_mid, num, ha="center", va="center",
                fontsize=16, color="white", fontweight="bold")

        # Title + body on a 7-unit-wide column (1.2 to 8.5)
        ax.text(1.2, y_mid + 0.28, title, ha="left", va="center",
                fontsize=13, color=NAVY, fontweight="bold")
        ax.text(1.2, y_mid - 0.22, body, ha="left", va="center",
                fontsize=10.5, color="#333", wrap=True)

        # 'Check' box on the right (9.0 to 15.5 = 6.5 units wide)
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((9.0, y_mid - 0.45), 6.5, 0.9,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor="#EAF6FF", edgecolor=NAVY_LIGHT, linewidth=1.2)
        ax.add_patch(box)
        ax.text(9.2, y_mid + 0.1, "Check:", ha="left", va="center",
                fontsize=10, color=NAVY_LIGHT, fontweight="bold", style="italic")
        ax.text(9.2, y_mid - 0.18, check, ha="left", va="center",
                fontsize=10, color="#222", linespacing=1.15)

    save(fig, "viz_s5_patterns")


# --------------------------------------------------------------------------
# viz_* figures — Phase 3 Group B regenerations (resize for 16:9 canvas)
# --------------------------------------------------------------------------

def fig_viz_s2_seg_works():
    """Cellpose-SAM success case: clean fluorescence → instance segmentation."""
    img, masks = synth_cells(size=220, n=24, seed=21, irregular=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Synthetic clean fluorescence\n(separated nuclei)",
                      color=NAVY, fontweight="bold", fontsize=14)
    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(np.where(masks > 0, masks, np.nan), cmap=INSTANCE_CMAP,
                   vmin=0, vmax=25, alpha=0.55)
    n_pred = int(masks.max())
    axes[1].set_title(f"Cellpose-SAM-style segmentation\n({n_pred} nuclei recovered, true: {n_pred})",
                      color=GOOD, fontweight="bold", fontsize=14)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Where Cellpose-SAM works: clean fluorescence with separated nuclei",
                 color=NAVY, fontsize=15, fontweight="bold", y=1.0)
    save(fig, "viz_s2_seg_works")


def fig_viz_s2_seg_fails():
    """Cellpose-SAM failure: dense OOD tissue → silent under-counting + merges.
    Detected masks are sampled from the SAME image (subset of true nuclei +
    a few merge pairs), so colored regions coincide with actual cell features."""
    from scipy.ndimage import center_of_mass
    img, masks_true = synth_cells(size=220, n=58, seed=22, irregular=True)

    # synth_cells already assigns each blob a unique integer label, even when
    # blobs overlap. Use those labels directly — DON'T re-do connected-components
    # (which would collapse touching blobs into one component).
    true_ids = np.unique(masks_true)
    true_ids = true_ids[true_ids != 0]
    n_lbl = len(true_ids)
    if n_lbl == 0:
        masks_bad = np.zeros_like(masks_true)
        n_detected = 0
        merge_centroids = []
    else:
        # Centroid per original label ID
        coms = np.array(center_of_mass(masks_true > 0, masks_true, list(true_ids)))
        # For the rest of the code, we treat `lbls` as `masks_true` and use `true_ids`
        # in place of range(1, n_lbl + 1).
        lbls = masks_true

        # Pick ~15 nuclei to be "detected".
        rng = np.random.default_rng(99)
        target_n_detected = min(15, n_lbl)
        kept_ids = rng.choice(true_ids, size=target_n_detected, replace=False)

        # Identify pairs of close kept nuclei (within 30 px) to merge → "merge" callouts.
        # Build an explicit id→centroid lookup so we don't assume ids start at 1.
        id_to_com = {int(tid): coms[i] for i, tid in enumerate(true_ids)}
        kept_coms = np.array([id_to_com[int(k)] for k in kept_ids])
        merge_pairs = []
        used = set()
        for i in range(len(kept_ids)):
            if i in used: continue
            for j in range(i + 1, len(kept_ids)):
                if j in used: continue
                d = np.linalg.norm(kept_coms[i] - kept_coms[j])
                if d < 30:
                    merge_pairs.append((i, j))
                    used.add(i); used.add(j)
                    break
            if len(merge_pairs) >= 2:
                break

        # Assign new contiguous IDs; merge each pair to a single ID
        masks_bad = np.zeros_like(masks_true, dtype=np.int32)
        new_id = 1
        merge_centroids = []
        merged_indices = {idx for pair in merge_pairs for idx in pair}
        # Place merges first
        for (i, j) in merge_pairs:
            masks_bad[lbls == kept_ids[i]] = new_id
            masks_bad[lbls == kept_ids[j]] = new_id
            mid = ((kept_coms[i] + kept_coms[j]) / 2)
            merge_centroids.append((float(mid[0]), float(mid[1])))
            new_id += 1
        # Then non-merged kept nuclei
        for k, oid in enumerate(kept_ids):
            if k in merged_indices: continue
            masks_bad[lbls == oid] = new_id
            new_id += 1
        n_detected = new_id - 1
    n_true = int(n_lbl)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Tissue-like dense overlap\n(synthetic OOD sample)",
                      color=NAVY, fontweight="bold", fontsize=14)
    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(np.where(masks_bad > 0, masks_bad, np.nan), cmap=INSTANCE_CMAP,
                   vmin=0, vmax=20, alpha=0.55)
    axes[1].set_title(f"Pretrained segmenter on OOD: {n_detected} 'cells' detected\n(true = {n_true} — silent under-counting)",
                      color=BAD, fontweight="bold", fontsize=14)
    # Annotate merge regions at the actual centroids of merged pairs
    for cy, cx in merge_centroids:
        axes[1].annotate("merge", xy=(cx, cy), xytext=(cx + 25, cy - 30),
                         fontsize=11, color=BAD, fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color=BAD, lw=1.5))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Where it fails: dense, irregular OOD samples",
                 color=NAVY, fontsize=15, fontweight="bold", y=1.0)
    save(fig, "viz_s2_seg_fails")


def fig_viz_s4_generalization():
    """Underfit / just-right / overfit — 2 rows × 3 columns at 16:9 size."""
    rng = np.random.default_rng(40)
    x_data = np.linspace(0, 2 * np.pi, 30)
    y_true = np.sin(x_data)
    y_data = y_true + rng.normal(0, 0.18, x_data.shape)
    x_fine = np.linspace(0, 2 * np.pi, 200)

    # Three fits: linear (underfit), polynomial degree 4 (just right), degree 18 (overfit)
    y_under = np.poly1d(np.polyfit(x_data, y_data, 1))(x_fine)
    y_right = np.poly1d(np.polyfit(x_data, y_data, 4))(x_fine)
    y_over = np.poly1d(np.polyfit(x_data, y_data, 18))(x_fine)

    # Loss curves: train loss drops in all 3; val loss diverges only in overfit
    epochs = np.arange(1, 51)
    under_train = 0.85 - 0.05 * np.log1p(epochs); under_val = under_train + 0.02
    right_train = 0.55 * np.exp(-epochs / 18) + 0.05; right_val = right_train + 0.03
    over_train = 0.55 * np.exp(-epochs / 8) + 0.02
    over_val = 0.45 * np.exp(-epochs / 12) + 0.05 + 0.01 * np.maximum(0, epochs - 18)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    panels = [
        (axes[0, 0], y_under, "Underfit (too simple)", "High train loss\nHigh val loss", "#222"),
        (axes[0, 1], y_right, "Just right", "Low train loss\nLow val loss", VIZ_FOUNDATIONS),
        (axes[0, 2], y_over, "Overfit (too flexible)", "Low train loss\nHIGH val loss", VIZ_JUDGMENT),
    ]
    for ax, y_pred, title, label, color in panels:
        ax.plot(x_fine, np.sin(x_fine), "--", color="gray", lw=1.2, label="true function")
        ax.plot(x_data, y_data, "o", color="#444", markersize=4, label="train data")
        ax.plot(x_fine, y_pred, "-", color=color, lw=2.2, label="model fit")
        ax.set_title(title, color=color, fontweight="bold", fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylim(-1.8, 1.8)
        # Inline status box
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, linewidth=1.5)
        ax.text(0.04, 0.04, label, transform=ax.transAxes, fontsize=10, color=color,
                fontweight="bold", verticalalignment="bottom", bbox=bbox_props)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        for s in ax.spines.values(): s.set_color(NAVY)

    loss_panels = [
        (axes[1, 0], under_train, under_val, "#222"),
        (axes[1, 1], right_train, right_val, VIZ_FOUNDATIONS),
        (axes[1, 2], over_train, over_val, VIZ_JUDGMENT),
    ]
    for ax, tr, va, color in loss_panels:
        ax.plot(epochs, tr, "-", color=color, lw=2, label="train loss")
        ax.plot(epochs, va, "--", color=color, lw=2, label="val loss")
        ax.fill_between(epochs, tr, va, color=color, alpha=0.12)
        ax.set_xlabel("epoch", fontsize=10); ax.set_ylabel("loss", fontsize=10)
        ax.set_xlim(0, 50); ax.set_ylim(0, 1.0)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Underfit · just-right · overfit — recognize the regime from the loss curves",
                 color=NAVY, fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()
    save(fig, "viz_s4_generalization")


def fig_viz_s4_modes():
    """Three modes of using AI today — pretrained, fine-tune, zero-shot.
    Uses the FULL 16-unit width with generous internal padding so text never
    spills outside its panel."""
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6.5); ax.axis("off")

    # Title
    ax.text(8, 6.2, "Three modes of using AI today",
            ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)

    # Three panels at x=[0.4-5.4], [5.6-10.4], [10.6-15.6]
    from matplotlib.patches import FancyBboxPatch
    panels = [
        (0.4, 5.4, "Pretrained inference", VIZ_FOUNDATIONS, "#F4FBFF",
         ["image", "model (frozen)", "output"],
         "Your data looks like the model's\ntraining data. No gradients,\nno labels — just run.",
         "Lab 1 (Cellpose-SAM out of the box)"),
        (5.6, 10.4, "Fine-tuning", VIZ_WRAP, "#EBE6F2",
         ["pretrained\nbackbone", "+ your\nlabels (~100)", "fine-\ntuned"],
         "Domain-specific data +\nhundreds of labels available.\nCustomize a backbone for your modality.",
         "Notebook 04 (model-zoo recipes)"),
        (10.6, 15.6, "Zero-shot prompting", VIZ_JUDGMENT, "#FCE8F2",
         ["★\nclick", "foundation\nmodel", "mask"],
         "No labels, no training.\nClick / box / text-prompt.\nBest for SAM-style foundation models.",
         "Lab 3b (μSAM)"),
    ]
    for x0, x1, title, color, fill, boxes, descr, lab in panels:
        w = x1 - x0
        # Panel container
        ax.add_patch(FancyBboxPatch((x0, 0.3), w, 5.6,
                                     boxstyle="round,pad=0.05,rounding_size=0.15",
                                     facecolor=fill, edgecolor=color, linewidth=2.5))
        # Panel title
        ax.text(x0 + w/2, 5.2, title, ha="center", va="center",
                fontsize=14, fontweight="bold", color=color)
        # 3 small boxes in a row showing pipeline
        bw = (w - 0.6) / 3 - 0.1
        for i, btxt in enumerate(boxes):
            bx = x0 + 0.3 + i * (bw + 0.15)
            ax.add_patch(FancyBboxPatch((bx, 3.0), bw, 1.2,
                                         boxstyle="round,pad=0.02,rounding_size=0.08",
                                         facecolor="white", edgecolor=color, linewidth=1.4))
            ax.text(bx + bw/2, 3.6, btxt, ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold", linespacing=1.0)
            # Arrow between boxes
            if i < 2:
                ax.annotate("", xy=(bx + bw + 0.13, 3.6), xytext=(bx + bw + 0.02, 3.6),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        # Description text
        ax.text(x0 + w/2, 2.0, descr, ha="center", va="center",
                fontsize=10.5, color="#222", linespacing=1.3)
        # Lab tag at bottom (italic, muted)
        ax.text(x0 + w/2, 0.7, lab, ha="center", va="center",
                fontsize=9.5, color="#666", style="italic")

    save(fig, "viz_s4_modes")


def fig_viz_s5_overview():
    """Works vs fails — 10-panel grid. Each panel has its OWN visual story
    (split panels for restoration, class outlines for classify, click marker
    for μSAM, sparse masks for OOD, dim image for drift, etc.) so the row
    reads as five distinct examples, not five colorings of the same scene."""
    from matplotlib.patches import Rectangle as Rect, Circle as Circ
    from scipy.ndimage import center_of_mass

    fig = plt.figure(figsize=(16, 7.5))
    fig.suptitle("Works vs fails — concrete examples",
                 color=NAVY, fontsize=15, fontweight="bold", y=0.99)
    gs = fig.add_gridspec(2, 6, width_ratios=[0.5, 1, 1, 1, 1, 1],
                          hspace=0.22, wspace=0.10, top=0.92, bottom=0.04,
                          left=0.02, right=0.98)

    # Row labels
    ax_lw = fig.add_subplot(gs[0, 0]); ax_lw.axis("off")
    ax_lw.text(0.5, 0.5, "WORKS", ha="center", va="center", rotation=90,
               fontsize=18, fontweight="bold", color=VIZ_FOUNDATIONS)
    ax_lf = fig.add_subplot(gs[1, 0]); ax_lf.axis("off")
    ax_lf.text(0.5, 0.5, "FAILS", ha="center", va="center", rotation=90,
               fontsize=18, fontweight="bold", color=VIZ_JUDGMENT)

    works_cmap = _category_cmap(VIZ_FOUNDATIONS)
    fails_cmap = _category_cmap(VIZ_JUDGMENT)

    def _frame(ax, color):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(color); s.set_linewidth(1.4)

    # =================== WORKS ROW (cyan family) =========================

    # 1. Segmentation — masks align with cells
    ax = fig.add_subplot(gs[0, 1])
    img, masks = synth_cells(seed=51, n=8)
    ax.imshow(img, cmap="gray")
    ax.imshow(np.where(masks > 0, masks, np.nan),
              cmap=works_cmap, vmin=0, vmax=4, alpha=0.65)
    ax.set_title("Segmentation", fontsize=12, color=VIZ_FOUNDATIONS, fontweight="bold")
    _frame(ax, VIZ_FOUNDATIONS)

    # 2. Restoration — split panel: noisy left | clean right with seam
    ax = fig.add_subplot(gs[0, 2])
    img_r, _ = synth_cells(seed=52, n=6)
    rng_r = np.random.default_rng(52)
    noisy = np.clip(img_r + rng_r.normal(0, 0.30, img_r.shape), 0, 1)
    h, w = img_r.shape
    split = np.hstack([noisy[:, :w // 2], img_r[:, w // 2:]])
    ax.imshow(split, cmap="gray")
    ax.axvline(w // 2 - 0.5, color=VIZ_FOUNDATIONS, lw=2)
    ax.text(w // 4, 6, "noisy", color="white", fontsize=9, fontweight="bold",
            ha="center", va="top")
    ax.text(3 * w // 4, 6, "denoised", color="white", fontsize=9, fontweight="bold",
            ha="center", va="top")
    ax.set_title("Restoration", fontsize=12, color=VIZ_FOUNDATIONS, fontweight="bold")
    _frame(ax, VIZ_FOUNDATIONS)

    # 3. Classification — cells with two-color class outlines
    ax = fig.add_subplot(gs[0, 3])
    img_c, masks_c = synth_cells(seed=53, n=5)
    ax.imshow(img_c, cmap="gray")
    cls_colors = [VIZ_FOUNDATIONS, "#F59E0B"]
    for idx, lbl in enumerate(np.unique(masks_c)):
        if lbl == 0:
            continue
        cy, cx = center_of_mass(masks_c == lbl)
        r = 16
        ax.add_patch(Circ((cx, cy), r + 4, fill=False,
                          edgecolor=cls_colors[idx % 2], lw=2.0))
    ax.text(0.04, 0.06, "A   B", transform=ax.transAxes, color="white",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#222", alpha=0.7,
                      edgecolor="none"))
    ax.set_title("Classify", fontsize=12, color=VIZ_FOUNDATIONS, fontweight="bold")
    _frame(ax, VIZ_FOUNDATIONS)

    # 4. Stitching — two tiles side-by-side with seamless join
    ax = fig.add_subplot(gs[0, 4])
    img_l, _ = synth_cells(size=120, n=3, seed=54)
    img_r2, _ = synth_cells(size=120, n=3, seed=540)
    stitched = np.hstack([img_l, img_r2])
    ax.imshow(stitched, cmap="gray")
    ax.axvline(120 - 0.5, color=VIZ_FOUNDATIONS, lw=1.2, linestyle=":")
    ax.text(60, 8, "tile 1", color="white", fontsize=9, fontweight="bold",
            ha="center", va="top")
    ax.text(180, 8, "tile 2", color="white", fontsize=9, fontweight="bold",
            ha="center", va="top")
    ax.set_title("Stitch", fontsize=12, color=VIZ_FOUNDATIONS, fontweight="bold")
    _frame(ax, VIZ_FOUNDATIONS)

    # 5. μSAM click — single click + mask emanating from it
    ax = fig.add_subplot(gs[0, 5])
    img_s, masks_s = synth_cells(seed=55, n=6)
    ax.imshow(img_s, cmap="gray")
    # Show only ONE mask emanating from a click
    target_lbl = int(np.unique(masks_s)[len(np.unique(masks_s)) // 2])
    one_mask = np.where(masks_s == target_lbl, 1, 0)
    ax.imshow(np.where(one_mask > 0, 1, np.nan), cmap=works_cmap,
              vmin=0, vmax=4, alpha=0.75)
    cy, cx = center_of_mass(masks_s == target_lbl)
    ax.plot(cx, cy, marker="*", markersize=18, color="yellow",
            markeredgecolor="black", markeredgewidth=1.2)
    ax.text(cx + 8, cy, "click", color="yellow", fontsize=9, fontweight="bold",
            va="center")
    ax.set_title("μSAM click", fontsize=12, color=VIZ_FOUNDATIONS, fontweight="bold")
    _frame(ax, VIZ_FOUNDATIONS)

    # =================== FAILS ROW (magenta family) ======================

    rng_keep = np.random.default_rng(101)

    # 1. OOD undercount — 30 cells in image, only ~5 colored
    ax = fig.add_subplot(gs[1, 1])
    img_o, masks_o = synth_cells(seed=61, n=30, irregular=True)
    ids_o = np.array([k for k in np.unique(masks_o) if k != 0])
    keep_o = set(rng_keep.choice(ids_o, size=min(5, len(ids_o)),
                                 replace=False).tolist())
    pred_o = np.where(np.isin(masks_o, list(keep_o)), masks_o, 0)
    ax.imshow(img_o, cmap="gray")
    ax.imshow(np.where(pred_o > 0, pred_o, np.nan), cmap=fails_cmap,
              vmin=0, vmax=4, alpha=0.65)
    ax.set_title("OOD undercount", fontsize=12, color=VIZ_JUDGMENT, fontweight="bold")
    _frame(ax, VIZ_JUDGMENT)

    # 2. Rare miss — round (masked) + elongated (unmasked)
    ax = fig.add_subplot(gs[1, 2])
    img_rm, masks_rm = _scene_rare_category(size=200, n_normal=4, seed=72)
    ax.imshow(img_rm, cmap="gray")
    ax.imshow(np.where(masks_rm > 0, masks_rm, np.nan), cmap=fails_cmap,
              vmin=0, vmax=4, alpha=0.65)
    ax.set_title("Rare miss", fontsize=12, color=VIZ_JUDGMENT, fontweight="bold")
    _frame(ax, VIZ_JUDGMENT)

    # 3. Sample drift — dim image, NO masks (silent fail)
    ax = fig.add_subplot(gs[1, 3])
    img_sd, _ = synth_cells(seed=63, n=4)
    img_sd = img_sd * 0.32 + 0.04  # crush contrast
    ax.imshow(img_sd, cmap="gray", vmin=0, vmax=1)
    ax.text(0.5, 0.5, "(no detections)", transform=ax.transAxes,
            color=VIZ_JUDGMENT, fontsize=10, fontweight="bold",
            ha="center", va="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.7, edgecolor=VIZ_JUDGMENT))
    ax.set_title("Sample drift", fontsize=12, color=VIZ_JUDGMENT, fontweight="bold")
    _frame(ax, VIZ_JUDGMENT)

    # 4. Edge effects — only interior cells masked
    ax = fig.add_subplot(gs[1, 4])
    img_e, masks_e = synth_cells(seed=64, n=12)
    h_e, w_e = img_e.shape
    EDGE = 28
    keep_e = []
    for lbl in np.unique(masks_e):
        if lbl == 0:
            continue
        cy, cx = center_of_mass(masks_e == lbl)
        if EDGE <= cy <= h_e - EDGE and EDGE <= cx <= w_e - EDGE:
            keep_e.append(lbl)
    pred_e = np.where(np.isin(masks_e, keep_e), masks_e, 0)
    ax.imshow(img_e, cmap="gray")
    ax.imshow(np.where(pred_e > 0, pred_e, np.nan), cmap=fails_cmap,
              vmin=0, vmax=4, alpha=0.65)
    ax.set_title("Edge effects", fontsize=12, color=VIZ_JUDGMENT, fontweight="bold")
    _frame(ax, VIZ_JUDGMENT)

    # 5. Hallucinated — magenta blob in empty space
    ax = fig.add_subplot(gs[1, 5])
    img_h, masks_h = synth_cells(seed=65, n=5)
    h_h, w_h = img_h.shape
    H, W = np.ogrid[:h_h, :w_h]
    rng_h = np.random.default_rng(55)
    best_pos, best_val = None, 1.0
    for _ in range(120):
        cy = int(rng_h.integers(30, h_h - 30))
        cx = int(rng_h.integers(30, w_h - 30))
        disk = ((H - cy) ** 2 + (W - cx) ** 2) <= 20 ** 2
        if (masks_h[disk] > 0).any():
            continue
        v = float(img_h[disk].mean())
        if v < best_val:
            best_val, best_pos = v, (cy, cx)
    if best_pos is None:
        best_pos = (h_h // 2, w_h // 2)
    cy_h, cx_h = best_pos
    hallo = ((H - cy_h) ** 2 + (W - cx_h) ** 2) <= 16 ** 2
    ax.imshow(img_h, cmap="gray")
    ax.imshow(np.where(masks_h > 0, masks_h, np.nan), cmap=fails_cmap,
              vmin=0, vmax=4, alpha=0.55)
    ax.imshow(np.where(hallo, 1, np.nan),
              cmap=ListedColormap([VIZ_JUDGMENT]), vmin=0, vmax=1, alpha=0.95)
    ax.text(cx_h, cy_h, "invented", ha="center", va="center", fontsize=8,
            color="white", fontweight="bold", style="italic")
    ax.set_title("Hallucinated", fontsize=12, color=VIZ_JUDGMENT, fontweight="bold")
    _frame(ax, VIZ_JUDGMENT)

    save(fig, "viz_s5_overview")


def _scene_rare_category(size=200, n_normal=4, seed=72):
    """Image with n_normal round cells (labeled) + 2 elongated 'rare-category'
    cells (NOT labeled). Masks contain ONLY the round cells, so any AI overlay
    drawn from `masks` will visibly miss the elongated ones."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=float)
    masks = np.zeros((size, size), dtype=int)
    Y, X = np.ogrid[:size, :size]
    # 4 normal round cells with labels 1..n_normal
    for i in range(1, n_normal + 1):
        cy, cx = rng.uniform(30, size - 30, 2)
        r = rng.uniform(10, 15)
        m = (Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2
        img[m] = rng.uniform(0.75, 1.0)
        masks[m] = i
    # 2 elongated "rare" cells — drawn into the image but NOT into masks
    for _ in range(2):
        cy, cx = rng.uniform(35, size - 35, 2)
        ry, rx = rng.uniform(22, 28), rng.uniform(6, 9)
        m = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1
        img[m] = rng.uniform(0.75, 1.0)
    img = gaussian_filter(img, sigma=0.8)
    img += rng.normal(0, 0.05, img.shape)
    return np.clip(img, 0, 1), masks


def fig_viz_s5_fails():
    """Five failure patterns. Each panel shows the FULL input image in grayscale
    (ground truth) plus the AI's incomplete or wrong prediction overlay. The
    *gap* between what's there and what the model returned is the takeaway."""
    from scipy.ndimage import center_of_mass
    from matplotlib.colors import ListedColormap as _LC

    fig, axes = plt.subplots(1, 5, figsize=(16, 5.0))
    fig.suptitle("Where AI fails — five patterns to recognize",
                 color=NAVY, fontsize=15, fontweight="bold", y=1.0)

    rng = np.random.default_rng(11)

    # ---- Panel 1: OOD undercount — 30 cells in image, only 8 in prediction ----
    img1, masks1 = synth_cells(seed=71, n=30, irregular=True)
    ids1 = np.array([k for k in np.unique(masks1) if k != 0])
    keep1 = set(rng.choice(ids1, size=min(8, len(ids1)), replace=False).tolist())
    pred1 = np.where(np.isin(masks1, list(keep1)), masks1, 0)

    # ---- Panel 2: Rare-category miss — 4 round (predicted) + 2 elongated (missed) ----
    img2, pred2 = _scene_rare_category(seed=72)

    # ---- Panel 3: Sample-prep drift — low-contrast image, prediction is EMPTY ----
    img3, _ = synth_cells(seed=73, n=4, irregular=False)
    img3 = img3 * 0.32 + 0.04  # crush contrast (the "silent fail" regime)
    pred3 = None  # AI returns nothing

    # ---- Panel 4: Edge effects — drop masks whose centroid sits within EDGE_PAD ----
    img4, masks4 = synth_cells(seed=74, n=12, irregular=False)
    h4, w4 = img4.shape
    EDGE_PAD = 28
    keep4 = []
    for lbl in np.unique(masks4):
        if lbl == 0:
            continue
        cy, cx = center_of_mass(masks4 == lbl)
        if EDGE_PAD <= cy <= h4 - EDGE_PAD and EDGE_PAD <= cx <= w4 - EDGE_PAD:
            keep4.append(lbl)
    pred4 = np.where(np.isin(masks4, keep4), masks4, 0)

    # ---- Panel 5: Hallucinated feature — real cells masked correctly + 1 mask in empty space ----
    img5, masks5 = synth_cells(seed=75, n=4, irregular=False)
    h5, w5 = img5.shape
    H5, W5 = np.ogrid[:h5, :w5]
    # Find a dark patch with no real cell content
    rng5 = np.random.default_rng(33)
    best_pos, best_val = None, 1.0
    for _ in range(120):
        cy = int(rng5.integers(40, h5 - 40))
        cx = int(rng5.integers(40, w5 - 40))
        disk = ((H5 - cy) ** 2 + (W5 - cx) ** 2) <= 22 ** 2
        if (masks5[disk] > 0).any():
            continue
        v = float(img5[disk].mean())
        if v < best_val:
            best_val, best_pos = v, (cy, cx)
    if best_pos is None:
        best_pos = (h5 // 2, w5 // 2)
    cy_h, cx_h = best_pos
    hallo_mask = ((H5 - cy_h) ** 2 + (W5 - cx_h) ** 2) <= 18 ** 2
    pred5 = masks5.copy()   # AI also gets the real ones right
    pred5_normal = pred5    # keep separate to render hallucination distinctly
    magenta_cmap = _LC([VIZ_JUDGMENT])

    panels = [
        ("OOD undercount",       img1, pred1, "30 cells → 8 detected"),
        ("Rare-category miss",   img2, pred2, "rare elongated cells → 0 detected"),
        ("Sample-prep drift",    img3, pred3, "low contrast → silent fail (0 detected)"),
        ("Edge effects",         img4, pred4, "border cells dropped"),
        ("Hallucinated feature", img5, pred5_normal, "magenta mask = invented"),
    ]

    for ax, (title, img, pred, caption) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        if pred is not None and pred.max() > 0:
            ax.imshow(np.where(pred > 0, pred, np.nan), cmap=INSTANCE_CMAP,
                      vmin=0, vmax=20, alpha=0.55)
        # Hallucination overlay on the last panel
        if "Halluc" in title:
            ax.imshow(np.where(hallo_mask, 1.0, np.nan), cmap=magenta_cmap,
                      vmin=0, vmax=1, alpha=0.85)
            ax.text(cx_h, cy_h, "invented", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold", style="italic")
        ax.set_title(title, fontsize=12, color=VIZ_JUDGMENT, fontweight="bold")
        ax.text(0.5, -0.05, caption, ha="center", va="top",
                transform=ax.transAxes, fontsize=9, style="italic", color="#444")
        ax.set_xticks([]); ax.set_yticks([])

    # Small legend below the row clarifying what the colored overlay means
    fig.text(0.5, 0.005, "grayscale = input image (ground truth visible)   |   "
             "colored masks = AI prediction",
             ha="center", va="bottom", fontsize=9.5, color="#444", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "viz_s5_fails")


# --------------------------------------------------------------------------
# viz_* figures — Phase 3 Group C regenerations (architecture diagrams + flows)
# --------------------------------------------------------------------------

def fig_viz_s2_overview():
    """Seven task categories. Registration + Tracking panels rebuilt so the
    SAME objects are visibly tracked across frames (was abstract before)."""
    fig = plt.figure(figsize=(16, 8.5))
    fig.suptitle("Seven task categories at a glance",
                 color=NAVY, fontsize=15, fontweight="bold", y=0.99)
    gs = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.18,
                          top=0.92, bottom=0.04, left=0.03, right=0.97)

    # Row 1: Segmentation, Detection, Classification, Restoration
    from scipy.ndimage import label as ndi_label, center_of_mass, find_objects
    from matplotlib.patches import Rectangle as Rect, Circle as Circ
    rng = np.random.default_rng(80)

    def nucleus_centroids_and_radii(img, threshold=0.4):
        """Find each bright nucleus in a synth_cells image. Returns list of (cy, cx, r) tuples."""
        lbls, n_lbl = ndi_label(img > threshold)
        if n_lbl == 0:
            return []
        coms = center_of_mass(img > threshold, lbls, range(1, n_lbl + 1))
        slices = find_objects(lbls)
        out = []
        for (cy, cx), sl in zip(coms, slices):
            # Radius estimate: half the larger of the bounding-box dims
            r = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) / 2
            out.append((cy, cx, r))
        return out

    # Segmentation
    img_s, masks_s = synth_cells(seed=80, n=8)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(img_s, cmap="gray")
    ax.imshow(np.where(masks_s > 0, masks_s, np.nan), cmap=INSTANCE_CMAP, vmin=0, vmax=15, alpha=0.5)
    ax.set_title("Segmentation", color=VIZ_FOUNDATIONS, fontweight="bold", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])

    # Detection — bounding boxes around ACTUAL nuclei
    img_d, _ = synth_cells(seed=81, n=8)
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(img_d, cmap="gray")
    for cy, cx, r in nucleus_centroids_and_radii(img_d):
        pad = r + 4
        ax.add_patch(Rect((cx - pad, cy - pad), 2 * pad, 2 * pad,
                          fill=False, edgecolor=VIZ_FOUNDATIONS, lw=1.8))
    ax.set_title("Detection", color=VIZ_FOUNDATIONS, fontweight="bold", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])

    # Classification — colored circles on ACTUAL nuclei, alternating classes
    img_c, _ = synth_cells(seed=82, n=7)
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(img_c, cmap="gray")
    nuclei_c = nucleus_centroids_and_radii(img_c)
    class_colors = [VIZ_FOUNDATIONS, VIZ_JUDGMENT]
    for i, (cy, cx, r) in enumerate(nuclei_c):
        col = class_colors[i % 2]
        ax.add_patch(Circ((cx, cy), r + 3, fill=False, edgecolor=col, lw=2.2))
    # Add small legend inside the panel
    ax.text(0.03, 0.97, "● class A   ● class B", transform=ax.transAxes,
            va="top", ha="left", fontsize=8, color="#222",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"))
    ax.set_title("Classification", color=VIZ_FOUNDATIONS, fontweight="bold", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    # Restoration
    img_r, _ = synth_cells(seed=83, n=6)
    noisy = np.clip(img_r + rng.normal(0, 0.25, img_r.shape), 0, 1)
    h, w = img_r.shape
    combined = np.hstack([noisy[:, :w//2], img_r[:, w//2:]])
    ax = fig.add_subplot(gs[0, 3]); ax.imshow(combined, cmap="gray")
    ax.axvline(w//2, color=VIZ_JUDGMENT, lw=2)
    ax.set_title("Restoration", color=VIZ_FOUNDATIONS, fontweight="bold", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])

    # Row 2: Registration, Tracking, Generation, (+ many more)
    # Registration: SAME objects in two frames, with a clear correspondence line
    img_reg, _ = synth_cells(seed=84, n=4)
    shift_y, shift_x = 18, 12
    img_reg_shifted = np.roll(img_reg, (shift_y, shift_x), axis=(0, 1))
    h, w = img_reg.shape
    composite = np.zeros((h, w*2 + 8))
    composite[:, :w] = img_reg
    composite[:, w+8:] = img_reg_shifted
    ax = fig.add_subplot(gs[1, 0]); ax.imshow(composite, cmap="gray")
    # Draw 3 correspondence arrows linking the same nucleus across the two panels
    # Pick 3 bright nucleus centroids using simple thresholding
    from scipy.ndimage import label as ndi_label, center_of_mass
    lbls, n_lbl = ndi_label(img_reg > 0.45)
    if n_lbl >= 3:
        centroids_l = center_of_mass(img_reg > 0.45, lbls, range(1, min(n_lbl, 3) + 1))
        for cy, cx in centroids_l:
            ax.annotate("", xy=(cx + w + 8 + shift_x, cy + shift_y),
                        xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->", color=VIZ_JUDGMENT, lw=1.5))
    ax.text(w/2, h+8, "moving", ha="center", va="top", fontsize=10, color=VIZ_JUDGMENT, fontweight="bold")
    ax.text(w + 8 + w/2, h+8, "fixed", ha="center", va="top", fontsize=10, color=VIZ_FOUNDATIONS, fontweight="bold")
    ax.set_title("Registration", color=VIZ_FOUNDATIONS, fontweight="bold", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(h+22, -3)

    # Tracking: same objects across t=0 → t=T with consistent IDs (colors)
    ax = fig.add_subplot(gs[1, 1])
    rng_t = np.random.default_rng(85)
    n_obj = 4
    colors_t = [VIZ_FOUNDATIONS, VIZ_JUDGMENT, VIZ_WRAP, "#E89A00"]
    for i in range(n_obj):
        x_start, y_start = rng_t.uniform(0.15, 0.85, 2)
        xs = [x_start]; ys = [y_start]
        for _ in range(15):
            xs.append(xs[-1] + rng_t.normal(0, 0.025))
            ys.append(ys[-1] + rng_t.normal(0, 0.025))
        # Trajectory line + endpoints labelled with object ID
        ax.plot(xs, ys, "-", color=colors_t[i], lw=1.6, alpha=0.7)
        ax.plot(xs[0], ys[0], "o", color=colors_t[i], ms=12, markeredgecolor="black", mew=1.2)
        ax.plot(xs[-1], ys[-1], "s", color=colors_t[i], ms=12, markeredgecolor="black", mew=1.2)
        ax.text(xs[0], ys[0] + 0.04, f"ID {i+1}", color=colors_t[i], fontsize=8, ha="center", fontweight="bold")
    ax.text(0.05, 0.95, "● t=0  ■ t=T", transform=ax.transAxes, fontsize=9, va="top",
            color=NAVY, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title("Tracking", color=VIZ_FOUNDATIONS, fontweight="bold", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    # Generation
    img_g, _ = synth_cells(seed=86, n=6)
    smooth_g = gaussian_filter(img_g, sigma=2)
    bf_g = np.clip(0.7 - 0.4 * img_g + 0.6 * (img_g - smooth_g), 0, 1)
    combined_g = np.hstack([bf_g[:, :w//2], img_g[:, w//2:]])
    ax = fig.add_subplot(gs[1, 2]); ax.imshow(combined_g, cmap="gray")
    ax.axvline(w//2, color=VIZ_JUDGMENT, lw=2)
    ax.text(w//4, 15, "BF", color="white", fontsize=12, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=NAVY))
    ax.text(3*w//4, 15, "G(z)", color="white", fontsize=12, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=VIZ_JUDGMENT))
    ax.set_title("Generation", color=VIZ_FOUNDATIONS, fontweight="bold", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    # + many more placeholder
    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    ax.text(0.5, 0.5, "+ many\nmore", ha="center", va="center",
            fontsize=18, color="#888", style="italic")

    save(fig, "viz_s2_overview")


def fig_viz_s3_overview():
    """Four model families. Each visual is positioned INSIDE its own quadrant
    via a helper that maps a 0-1 unit-cell coord to absolute coords. The 'Other'
    quadrant is simplified to a single AE bow-tie + small Diffusion-progression
    icon next to it (not two full-size diagrams side by side)."""
    from matplotlib.patches import FancyBboxPatch, Rectangle as Rect, Polygon, Circle as Circ
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.5); ax.axis("off")
    ax.text(8, 8.25, "Common model families behind bioimage AI",
            ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)

    # Quadrant geometry: each quad is at (x0, y0) with width=7.4, height=3.6.
    QUAD_W, QUAD_H = 7.4, 3.6
    quads = [
        ("CNNs",                     0.4, 4.2, VIZ_FOUNDATIONS, "#F4FBFF",
         "Best for: dense prediction (segmentation, restoration)", "Lab 1 backbone"),
        ("Transformers",             8.2, 4.2, VIZ_WRAP,         "#EBE6F2",
         "Best for: long-range context, prompted segmentation", "Lab 3b (SAM)"),
        ("GANs",                     0.4, 0.3, VIZ_JUDGMENT,     "#FCE8F2",
         "Best for: image-to-image translation, stylization", "Notebook 04 demo"),
        ("Other (AE / Diffusion)",   8.2, 0.3, "#333",           "#F2F2F2",
         "Best for: self-supervised denoising, generative priors", "Lab 3a (N2V)"),
    ]
    for title, x0, y0, color, fill, best_for, lab in quads:
        ax.add_patch(FancyBboxPatch((x0, y0), QUAD_W, QUAD_H,
                                     boxstyle="round,pad=0.05,rounding_size=0.15",
                                     facecolor=fill, edgecolor=color, linewidth=2.5))
        # Title at top of quadrant
        ax.text(x0 + 0.3, y0 + QUAD_H - 0.35, title, fontsize=14, fontweight="bold", color=color)
        # "Best for: ..." line at bottom, just above the lab tag
        ax.text(x0 + 0.3, y0 + 0.55, best_for, fontsize=10, color="#222", style="italic")
        # Lab tag at bottom-right
        ax.text(x0 + QUAD_W - 0.2, y0 + 0.2, lab, fontsize=9.5, color="#666", style="italic", ha="right")

    # --- CNN visual (top-left quadrant, y0=4.2) ---
    # 4 pyramidal blocks → arrow, centered in the available drawing area
    cnn_x0, cnn_y0 = 0.4, 4.2  # quadrant origin
    blk_y_center = cnn_y0 + 2.0  # mid-height of quadrant draw area
    blk_x_start = cnn_x0 + 0.9
    for i, (bw, bh) in enumerate([(0.55, 1.5), (0.50, 1.2), (0.45, 0.9), (0.40, 0.6)]):
        bx = blk_x_start + i * 0.65
        by = blk_y_center - bh / 2
        ax.add_patch(Rect((bx, by), bw, bh, facecolor=VIZ_FOUNDATIONS, alpha=0.85))
    # Arrow from last block to right side of quadrant
    arrow_x_start = blk_x_start + 4 * 0.65 + 0.05
    ax.annotate("", xy=(cnn_x0 + QUAD_W - 0.4, blk_y_center),
                xytext=(arrow_x_start, blk_y_center),
                arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=2.5))

    # --- Transformer visual (top-right quadrant, y0=4.2) ---
    tx_q, ty_q = 8.2, 4.2  # quadrant origin
    # 4×4 patch grid centered in upper-middle of the quadrant
    grid_x_start = tx_q + 1.5
    grid_y_start = ty_q + 1.2
    cell_size = 0.34
    for i in range(4):
        for j in range(4):
            color_p = VIZ_WRAP if (i, j) == (2, 1) else "#C9BDD9"
            ax.add_patch(Rect((grid_x_start + j * cell_size, grid_y_start + i * cell_size),
                              cell_size - 0.02, cell_size - 0.02,
                              facecolor=color_p, edgecolor="white", linewidth=1.2))
    # Two attention arrows from the highlighted patch (row 2, col 1) to other patches
    src_x = grid_x_start + 1 * cell_size + cell_size / 2
    src_y = grid_y_start + 2 * cell_size + cell_size / 2
    for dx, dy in [(2 * cell_size, -1 * cell_size), (-1 * cell_size, 1 * cell_size)]:
        ax.annotate("", xy=(src_x + dx, src_y + dy), xytext=(src_x, src_y),
                    arrowprops=dict(arrowstyle="->", color=VIZ_JUDGMENT, lw=1.3))
    ax.text(src_x, src_y - 0.04, "Q", ha="center", va="center", fontsize=10,
            color=VIZ_JUDGMENT, fontweight="bold")

    # --- GAN visual (bottom-left quadrant, y0=0.3) ---
    gx_q, gy_q = 0.4, 0.3
    g_box_y = gy_q + 1.5
    box_w, box_h = 1.3, 0.85
    g_x = gx_q + 1.0
    d_x = gx_q + 4.5
    # Generator G (white fill, magenta outline)
    ax.add_patch(FancyBboxPatch((g_x, g_box_y), box_w, box_h, boxstyle="round,pad=0.05",
                                 facecolor="white", edgecolor=VIZ_JUDGMENT, linewidth=2))
    ax.text(g_x + box_w / 2, g_box_y + box_h / 2, "G",
            ha="center", va="center", fontsize=20, color=VIZ_JUDGMENT, fontweight="bold")
    # Discriminator D (filled magenta)
    ax.add_patch(FancyBboxPatch((d_x, g_box_y), box_w, box_h, boxstyle="round,pad=0.05",
                                 facecolor=VIZ_JUDGMENT, edgecolor=VIZ_JUDGMENT, linewidth=2))
    ax.text(d_x + box_w / 2, g_box_y + box_h / 2, "D",
            ha="center", va="center", fontsize=20, color="white", fontweight="bold")
    # Data flow G → D (solid cyan)
    ax.annotate("", xy=(d_x - 0.02, g_box_y + box_h * 0.65),
                xytext=(g_x + box_w + 0.02, g_box_y + box_h * 0.65),
                arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=2))
    # Gradient flow D → G (dashed magenta)
    ax.annotate("", xy=(g_x + box_w + 0.02, g_box_y + box_h * 0.25),
                xytext=(d_x - 0.02, g_box_y + box_h * 0.25),
                arrowprops=dict(arrowstyle="->", color=VIZ_JUDGMENT, lw=2, linestyle="--"))

    # --- Other visual (bottom-right quadrant, y0=0.3) ---
    # Simplified: AE bow-tie shape on the LEFT half + small 3-stage diffusion on the RIGHT half,
    # both positioned at the same y so the quadrant reads as "two examples of this family".
    ox_q, oy_q = 8.2, 0.3
    # AE bow-tie: two triangles meeting at a small bottleneck dot
    ae_cx = ox_q + 1.4
    ae_cy = oy_q + 1.95
    ae_half_w = 0.85
    ae_half_h = 0.7
    # Encoder triangle (left)
    ax.add_patch(Polygon([[ae_cx - ae_half_w, ae_cy + ae_half_h],
                          [ae_cx, ae_cy],
                          [ae_cx - ae_half_w, ae_cy - ae_half_h]],
                         facecolor=VIZ_FOUNDATIONS, alpha=0.78))
    # Decoder triangle (right)
    ax.add_patch(Polygon([[ae_cx + ae_half_w, ae_cy + ae_half_h],
                          [ae_cx, ae_cy],
                          [ae_cx + ae_half_w, ae_cy - ae_half_h]],
                         facecolor=VIZ_FOUNDATIONS, alpha=0.78))
    # Bottleneck dot
    ax.add_patch(Circ((ae_cx, ae_cy), 0.12, facecolor=VIZ_WRAP, edgecolor="white", linewidth=1.2))
    ax.text(ae_cx, ae_cy - ae_half_h - 0.25, "AE",
            ha="center", fontsize=11, color="#333", fontweight="bold")

    # Diffusion: 3 stages on the right side of the quadrant
    df_x0 = ox_q + 3.6
    df_y = oy_q + 1.55
    df_w, df_h = 0.55, 0.85
    df_stages_noise = [0.9, 0.5, 0.12]
    for i, noise_level in enumerate(df_stages_noise):
        sub_x = df_x0 + i * 0.75
        ax.add_patch(Rect((sub_x, df_y), df_w, df_h, facecolor="#222"))
        n_spk = int(45 * noise_level)
        rng_s = np.random.default_rng(90 + i)
        xs_spk = sub_x + 0.03 + rng_s.uniform(0, df_w - 0.06, n_spk)
        ys_spk = df_y + 0.03 + rng_s.uniform(0, df_h - 0.06, n_spk)
        ax.scatter(xs_spk, ys_spk, s=1.8, c="white")
        if i < len(df_stages_noise) - 1:
            ax.annotate("", xy=(sub_x + df_w + 0.18, df_y + df_h / 2),
                        xytext=(sub_x + df_w + 0.03, df_y + df_h / 2),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.text(df_x0 + (3 * 0.75) / 2, df_y - 0.25, "Diffusion",
            ha="center", fontsize=11, color="#333", fontweight="bold")

    save(fig, "viz_s3_overview")


def fig_viz_s3_cnn():
    """U-Net architecture diagram. Tighter layout so labels stay inside the draw
    area at every column. Includes a Ronneberger 2015 Fig 1 reference inset URL
    in the figure caption (the lecture markdown handles the inset slide)."""
    from matplotlib.patches import Rectangle as Rect
    fig, ax = plt.subplots(figsize=(16, 7))
    # Leave 0.5 unit margin on every side
    ax.set_xlim(-0.5, 16.5); ax.set_ylim(-1, 8); ax.axis("off")
    ax.text(8, 7.7, "U-Net — encoder · bottleneck · decoder · skip connections",
            ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)

    # Encoder blocks (decreasing height, increasing depth/color)
    enc_blocks = [(0.5, 4.5, 1, 64, "572²", "568²"),
                  (1.8, 3.6, 1, 128, "280²", "136²"),
                  (3.1, 2.7, 1, 256, "136²", "64²"),
                  (4.4, 1.8, 1, 512, "64²", "32²")]
    for x, h, w_block, ch, lbl_below, lbl_in in enc_blocks:
        y = 3.0 - h / 2
        ax.add_patch(Rect((x, y), w_block, h, facecolor=VIZ_FOUNDATIONS,
                          edgecolor="white", linewidth=1.5))
        ax.text(x + w_block/2, y + h + 0.15, str(ch), ha="center", va="bottom",
                fontsize=10, color=VIZ_FOUNDATIONS, fontweight="bold")
        ax.text(x + w_block/2, y - 0.2, lbl_below, ha="center", va="top",
                fontsize=8, color="#666", style="italic")

    # Bottleneck
    bn_x = 5.9
    ax.add_patch(Rect((bn_x, 2.4), 0.9, 1.2, facecolor=VIZ_WRAP, edgecolor="white", linewidth=1.5))
    ax.text(bn_x + 0.45, 2.4 + 1.2 + 0.15, "1024", ha="center", va="bottom",
            fontsize=10, color=VIZ_WRAP, fontweight="bold")
    ax.text(bn_x + 0.45, 2.4 - 0.2, "32²", ha="center", va="top",
            fontsize=8, color="#666", style="italic")
    ax.text(bn_x + 0.45, 1.8, "bottleneck", ha="center", va="top",
            fontsize=9, color=VIZ_WRAP, fontweight="bold")

    # Decoder blocks (mirroring encoder)
    dec_blocks = [(7.4, 1.8, 1, 512, "56²"),
                  (8.7, 2.7, 1, 256, "104²"),
                  (10.0, 3.6, 1, 128, "200²"),
                  (11.3, 4.5, 1, 64, "388²")]
    for x, h, w_block, ch, lbl_below in dec_blocks:
        y = 3.0 - h / 2
        ax.add_patch(Rect((x, y), w_block, h, facecolor=VIZ_JUDGMENT,
                          edgecolor="white", linewidth=1.5))
        ax.text(x + w_block/2, y + h + 0.15, str(ch), ha="center", va="bottom",
                fontsize=10, color=VIZ_JUDGMENT, fontweight="bold")
        ax.text(x + w_block/2, y - 0.2, lbl_below, ha="center", va="top",
                fontsize=8, color="#666", style="italic")

    # Output block
    ax.add_patch(Rect((12.7, 2.2), 0.7, 1.6, facecolor="#444",
                      edgecolor="white", linewidth=1.5))
    ax.text(13.05, 2.2 + 1.6 + 0.15, "2", ha="center", va="bottom",
            fontsize=10, color="#444", fontweight="bold")
    ax.text(13.05, 2.2 - 0.2, "388²", ha="center", va="top",
            fontsize=8, color="#666", style="italic")
    ax.text(13.05, 1.6, "mask\n(output)", ha="center", va="top", fontsize=9, color="#444", fontweight="bold")

    # Input
    ax.add_patch(Rect((-0.1, 1.8), 0.4, 2.4, facecolor="#444", edgecolor="white", linewidth=1.5))
    ax.text(0.1, 4.2 + 0.15, "1", ha="center", va="bottom", fontsize=10, color="#444", fontweight="bold")
    ax.text(0.1, 1.6, "image\n(input)", ha="center", va="top", fontsize=9, color="#444", fontweight="bold")
    ax.text(0.1, 1.0, "572²", ha="center", va="top", fontsize=8, color="#666", style="italic")

    # Skip connections (arcs from encoder to decoder)
    skip_pairs = [(enc_blocks[0], dec_blocks[3]),
                  (enc_blocks[1], dec_blocks[2]),
                  (enc_blocks[2], dec_blocks[1]),
                  (enc_blocks[3], dec_blocks[0])]
    for (xe, he, _, _, _, _), (xd, hd, _, _, _) in skip_pairs:
        from matplotlib.patches import FancyArrowPatch
        arc = FancyArrowPatch((xe + 0.5, 3.0 + he/2 + 0.1),
                              (xd + 0.5, 3.0 + hd/2 + 0.1),
                              connectionstyle="arc3,rad=-0.32",
                              arrowstyle="-", color="#666", lw=1.2)
        ax.add_patch(arc)

    # Inter-block arrows (encoder direction)
    for i in range(len(enc_blocks) - 1):
        x0 = enc_blocks[i][0] + 1.0
        x1 = enc_blocks[i+1][0] - 0.02
        ax.annotate("", xy=(x1, 3.0), xytext=(x0, 3.0),
                    arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=1.5))
    # Encoder → bottleneck
    ax.annotate("", xy=(bn_x - 0.02, 3.0), xytext=(enc_blocks[-1][0] + 1.0, 3.0),
                arrowprops=dict(arrowstyle="->", color=VIZ_WRAP, lw=1.7))
    # Bottleneck → decoder
    ax.annotate("", xy=(dec_blocks[0][0] - 0.02, 3.0), xytext=(bn_x + 0.9, 3.0),
                arrowprops=dict(arrowstyle="->", color=VIZ_WRAP, lw=1.7))
    # Inter-decoder arrows
    for i in range(len(dec_blocks) - 1):
        x0 = dec_blocks[i][0] + 1.0
        x1 = dec_blocks[i+1][0] - 0.02
        ax.annotate("", xy=(x1, 3.0), xytext=(x0, 3.0),
                    arrowprops=dict(arrowstyle="->", color=VIZ_JUDGMENT, lw=1.5))
    # Decoder → output
    ax.annotate("", xy=(12.7 - 0.02, 3.0), xytext=(dec_blocks[-1][0] + 1.0, 3.0),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))

    # Axis labels
    ax.text(2.7, 0.2, "encoder (downsample)", ha="center", color=VIZ_FOUNDATIONS, fontsize=11, fontweight="bold")
    ax.text(10, 0.2, "decoder (upsample)", ha="center", color=VIZ_JUDGMENT, fontsize=11, fontweight="bold")

    # Legend at the bottom
    ax.text(8, -0.5, "Channels above each box · spatial size at lower-left",
            ha="center", fontsize=9.5, color="#666", style="italic")
    legend_y = -0.85
    lx = 1.5
    ax.annotate("", xy=(lx + 0.4, legend_y), xytext=(lx, legend_y),
                arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=2))
    ax.text(lx + 0.6, legend_y, "3×3 conv + ReLU (×2), then 2×2 max-pool", va="center", fontsize=9, color="#333")
    lx = 8.5
    ax.annotate("", xy=(lx + 0.4, legend_y), xytext=(lx, legend_y),
                arrowprops=dict(arrowstyle="->", color=VIZ_JUDGMENT, lw=2))
    ax.text(lx + 0.6, legend_y, "2×2 up-conv, then 3×3 conv + ReLU (×2)", va="center", fontsize=9, color="#333")
    legend_y2 = -1.3
    lx = 1.5
    from matplotlib.patches import FancyArrowPatch
    arc_leg = FancyArrowPatch((lx, legend_y2 + 0.05), (lx + 0.4, legend_y2 + 0.05),
                               connectionstyle="arc3,rad=-0.5", arrowstyle="-", color="#666", lw=1.2)
    ax.add_patch(arc_leg)
    ax.text(lx + 0.6, legend_y2, "copy + concat (skip connection)", va="center", fontsize=9, color="#333")
    lx = 8.5
    ax.annotate("", xy=(lx + 0.4, legend_y2), xytext=(lx, legend_y2),
                arrowprops=dict(arrowstyle="->", color="#444", lw=2))
    ax.text(lx + 0.6, legend_y2, "1×1 conv → output classes", va="center", fontsize=9, color="#333")
    ax.set_ylim(-1.5, 8)

    save(fig, "viz_s3_cnn")


def fig_viz_s3_gan():
    """GAN — clean linear pipeline. Two inputs (real, fake-from-noise) converge
    into D, which outputs real/fake. A single training-signal arrow loops back
    with a plain-language caption explaining both steps. No crossing arcs."""
    from matplotlib.patches import FancyBboxPatch, Rectangle as Rect, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6.5); ax.axis("off")
    ax.text(8, 6.2, "GAN — two networks competing",
            ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)

    # ============================================================
    # Layout: two parallel horizontal lanes converging into D.
    # Lane A (top, y≈4.4): real image → D
    # Lane B (bottom, y≈2.4): noise z → G → fake image → D
    # D output on the right. Single training-signal caption below.
    # ============================================================

    lane_a_y = 4.4
    lane_b_y = 2.4
    box_h = 1.1
    half_h = box_h / 2

    # --- Lane A: Real images ---
    real_x, real_w = 0.8, 2.6
    ax.add_patch(FancyBboxPatch((real_x, lane_a_y - half_h), real_w, box_h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="#E0F4FF", edgecolor=VIZ_FOUNDATIONS, linewidth=2))
    ax.text(real_x + real_w/2, lane_a_y + 0.2, "Real images",
            ha="center", fontsize=12, color=VIZ_FOUNDATIONS, fontweight="bold")
    # three small image swatches inside the box
    for i in range(3):
        ax.add_patch(Rect((real_x + 0.25 + i * 0.7, lane_a_y - 0.35), 0.5, 0.4,
                          facecolor="#333"))

    # --- Lane B: noise → Generator → Fake ---
    # Noise z bubble (small, leftmost)
    noise_x, noise_w = 0.4, 1.4
    ax.add_patch(FancyBboxPatch((noise_x, lane_b_y - half_h*0.7), noise_w, box_h*0.7,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="#F5F5F5", edgecolor="#888", linewidth=1.5))
    ax.text(noise_x + noise_w/2, lane_b_y, "noise z",
            ha="center", va="center", fontsize=11, color="#444", style="italic")
    # Generator G
    gen_x, gen_w = 2.6, 2.0
    ax.add_patch(FancyBboxPatch((gen_x, lane_b_y - half_h), gen_w, box_h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="white", edgecolor="#333", linewidth=2))
    ax.text(gen_x + gen_w/2, lane_b_y, "Generator G",
            ha="center", va="center", fontsize=13, color="#333", fontweight="bold")
    # Fake images box
    fake_x, fake_w = 5.4, 2.6
    ax.add_patch(FancyBboxPatch((fake_x, lane_b_y - half_h), fake_w, box_h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="#FCE8F2", edgecolor=VIZ_JUDGMENT, linewidth=2))
    ax.text(fake_x + fake_w/2, lane_b_y + 0.2, "Fake images G(z)",
            ha="center", fontsize=12, color=VIZ_JUDGMENT, fontweight="bold")
    for i in range(3):
        ax.add_patch(Rect((fake_x + 0.25 + i * 0.7, lane_b_y - 0.35), 0.5, 0.4,
                          facecolor="#666"))

    # --- Discriminator D (center-right, spans BOTH lanes vertically) ---
    disc_x, disc_w = 10.0, 2.4
    disc_y_top = lane_a_y + half_h          # aligns with top of Lane A
    disc_y_bot = lane_b_y - half_h          # aligns with bottom of Lane B
    disc_h = disc_y_top - disc_y_bot
    ax.add_patch(FancyBboxPatch((disc_x, disc_y_bot), disc_w, disc_h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="white", edgecolor=VIZ_JUDGMENT, linewidth=2.5))
    ax.text(disc_x + disc_w/2, disc_y_bot + disc_h * 0.72, "Discriminator D",
            ha="center", fontsize=13, color=VIZ_JUDGMENT, fontweight="bold")
    ax.text(disc_x + disc_w/2, disc_y_bot + disc_h * 0.42, "real or fake?",
            ha="center", fontsize=11, color="#333", style="italic")

    # --- D output (rightmost) ---
    out_x, out_y_center = 13.4, (lane_a_y + lane_b_y) / 2
    out_w, out_h = 2.2, 1.2
    ax.add_patch(FancyBboxPatch((out_x, out_y_center - out_h/2), out_w, out_h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="#F8F8F8", edgecolor="#333", linewidth=1.5))
    ax.text(out_x + out_w/2, out_y_center + 0.25, "D(x) ∈ [0, 1]",
            ha="center", fontsize=12, color="#333", fontweight="bold")
    ax.text(out_x + out_w/2, out_y_center - 0.18, "1 = real · 0 = fake",
            ha="center", fontsize=10, color="#666", style="italic")

    # ============================================================
    # DATA-FLOW ARROWS (cyan, solid) — single horizontal direction each
    # ============================================================
    # noise → G
    ax.annotate("", xy=(gen_x - 0.02, lane_b_y),
                xytext=(noise_x + noise_w + 0.02, lane_b_y),
                arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=2))
    # G → Fake
    ax.annotate("", xy=(fake_x - 0.02, lane_b_y),
                xytext=(gen_x + gen_w + 0.02, lane_b_y),
                arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=2))
    # Real → D (gentle slope from Lane A right edge to D's upper-left)
    ax.annotate("", xy=(disc_x - 0.02, disc_y_top - 0.4),
                xytext=(real_x + real_w + 0.02, lane_a_y),
                arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=2.2))
    # Fake → D (gentle slope from Lane B right edge to D's lower-left)
    ax.annotate("", xy=(disc_x - 0.02, disc_y_bot + 0.4),
                xytext=(fake_x + fake_w + 0.02, lane_b_y),
                arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=2.2))
    # D → output
    ax.annotate("", xy=(out_x - 0.02, out_y_center),
                xytext=(disc_x + disc_w + 0.02, out_y_center),
                arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=2.2))

    # ============================================================
    # TRAINING-SIGNAL CAPTION (single, plain-language)
    # No curving arrows — instead, a horizontal magenta band below the diagram
    # with the two-step description spelled out in plain English.
    # ============================================================
    ax.add_patch(FancyBboxPatch((0.4, 0.25), 15.2, 0.95,
                                 boxstyle="round,pad=0.05,rounding_size=0.12",
                                 facecolor="#FCE8F2", edgecolor=VIZ_JUDGMENT, linewidth=1.5))
    ax.text(0.8, 0.95, "Training (alternating steps):",
            fontsize=11, color=VIZ_JUDGMENT, fontweight="bold")
    ax.text(0.8, 0.55,
            "↑  G learns to make D say 'real' for its fakes   ·   "
            "↑  D learns to tell real from fake",
            fontsize=11, color="#333")
    ax.text(15.2, 0.55, "(loss → both)",
            ha="right", fontsize=10, color=VIZ_JUDGMENT, style="italic", fontweight="bold")

    save(fig, "viz_s3_gan")


def fig_viz_s3_other():
    """5 architectures with visibly distinct schematics. Arcs are positioned
    BELOW the encoder/decoder blocks (frowns, not smiles) so they don't collide
    with the panel titles at the top. Adds a Diffusion panel."""
    from matplotlib.patches import Rectangle as Rect, FancyArrowPatch, FancyBboxPatch, Circle as Circ
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.5); ax.axis("off")
    ax.text(8, 8.2, "Recognize the encoder-decoder pattern in any new bioimage paper",
            ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)

    # 2x3 grid layout: panels are 5.0 wide, with 0.3 gap between columns.
    # Top row centered at y=6.2, bottom row centered at y=2.4.
    PANEL_W = 5.0
    PANEL_GAP = 0.3
    panel_xs = [0.4, 5.7, 11.0]   # x_origin per column
    TITLE_Y_TOP = 7.4              # title above top row
    CAPTION_Y_TOP = 4.5            # caption below top row
    TITLE_Y_BOT = 3.6              # title above bottom row
    CAPTION_Y_BOT = 0.7            # caption below bottom row

    def draw_encoder_decoder(x_origin, y_center,
                             encoder_heights=(1.5, 1.2, 0.9, 0.6),
                             decoder_heights=(0.6, 0.9, 1.2, 1.5),
                             block_w=0.38, gap=0.10,
                             skip=True, bottleneck=True,
                             color_enc=VIZ_FOUNDATIONS, color_dec=VIZ_JUDGMENT,
                             arcs_below=True):
        """Draw an encoder-decoder block sequence. When skip=True, arcs are drawn
        BELOW the blocks (rad>0, bowing down) so they never overlap the title."""
        x = x_origin
        enc_xs = []
        for h in encoder_heights:
            y = y_center - h / 2
            ax.add_patch(Rect((x, y), block_w, h, facecolor=color_enc, alpha=0.8,
                              edgecolor="white", lw=1.2))
            enc_xs.append((x, h))
            x += block_w + gap
        if bottleneck:
            ax.add_patch(Rect((x, y_center - 0.22), block_w, 0.44,
                              facecolor=VIZ_WRAP, edgecolor="white", lw=1.2))
            x += block_w + gap
        dec_xs = []
        for h in decoder_heights:
            y = y_center - h / 2
            ax.add_patch(Rect((x, y), block_w, h, facecolor=color_dec, alpha=0.8,
                              edgecolor="white", lw=1.2))
            dec_xs.append((x, h))
            x += block_w + gap
        # Skip connections — BELOW the blocks so they don't collide with titles above
        if skip:
            for (xe, he), (xd, hd) in zip(enc_xs, reversed(dec_xs)):
                if arcs_below:
                    arc = FancyArrowPatch(
                        (xe + block_w/2, y_center - he/2 - 0.08),
                        (xd + block_w/2, y_center - hd/2 - 0.08),
                        connectionstyle="arc3,rad=0.30",
                        arrowstyle="-", color="#666", lw=1.0)
                else:
                    arc = FancyArrowPatch(
                        (xe + block_w/2, y_center + he/2 + 0.05),
                        (xd + block_w/2, y_center + hd/2 + 0.05),
                        connectionstyle="arc3,rad=-0.30",
                        arrowstyle="-", color="#666", lw=1.0)
                ax.add_patch(arc)
        return x_origin, x, dec_xs

    # ===== TOP ROW =====
    # ----- Panel 1: U-Net -----
    p1_x = panel_xs[0]
    p1_center = p1_x + PANEL_W / 2
    ax.text(p1_center, TITLE_Y_TOP, "U-Net",
            fontsize=14, fontweight="bold", color=VIZ_FOUNDATIONS, ha="center")
    draw_encoder_decoder(p1_x + 0.3, 6.2, skip=True, arcs_below=True)
    ax.text(p1_center, CAPTION_Y_TOP, "encoder-decoder + skip connections",
            ha="center", fontsize=10, color="#444", style="italic")

    # ----- Panel 2: Autoencoder -----
    p2_x = panel_xs[1]
    p2_center = p2_x + PANEL_W / 2
    ax.text(p2_center, TITLE_Y_TOP, "Autoencoder",
            fontsize=14, fontweight="bold", color=VIZ_FOUNDATIONS, ha="center")
    _, end_ae, _ = draw_encoder_decoder(p2_x + 0.3, 6.2, skip=False)
    # Latent z indicator at bottleneck
    bn_x_ae = p2_x + 0.3 + 4 * (0.38 + 0.10) + 0.19  # bottleneck center
    ax.add_patch(Circ((bn_x_ae, 6.2), 0.16, facecolor=VIZ_WRAP, edgecolor="white", lw=1.5))
    ax.text(bn_x_ae, 5.75, "z (latent)", ha="center", va="top",
            fontsize=9, color=VIZ_WRAP, fontweight="bold", style="italic")
    ax.text(p2_center, CAPTION_Y_TOP, "no skips — bottleneck forces compression",
            ha="center", fontsize=10, color="#444", style="italic")

    # ----- Panel 3: pix2pix -----
    p3_x = panel_xs[2]
    p3_center = p3_x + PANEL_W / 2
    ax.text(p3_center, TITLE_Y_TOP, "pix2pix",
            fontsize=14, fontweight="bold", color=VIZ_JUDGMENT, ha="center")
    _, end_p2p, dec_xs_p2p = draw_encoder_decoder(p3_x + 0.2, 6.2, skip=True,
                                                   color_dec=VIZ_JUDGMENT, arcs_below=True)
    # Adversarial D box appended on the right
    d_box_x = end_p2p + 0.05
    ax.add_patch(FancyBboxPatch((d_box_x, 5.5), 0.85, 1.4,
                                 boxstyle="round,pad=0.05,rounding_size=0.08",
                                 facecolor="#FCE8F2", edgecolor=VIZ_JUDGMENT, lw=1.8))
    ax.text(d_box_x + 0.42, 6.2, "D\n(adv.)", ha="center", va="center",
            fontsize=10, color=VIZ_JUDGMENT, fontweight="bold")
    # Arrow from decoder output → D
    last_x, _ = dec_xs_p2p[-1]
    ax.annotate("", xy=(d_box_x - 0.02, 6.2), xytext=(last_x + 0.40, 6.2),
                arrowprops=dict(arrowstyle="->", color=VIZ_JUDGMENT, lw=1.4))
    ax.text(p3_center, CAPTION_Y_TOP, "U-Net generator + adversarial discriminator",
            ha="center", fontsize=10, color="#444", style="italic")

    # ===== BOTTOM ROW =====
    # ----- Panel 4: fnet (3D) -----
    p4_x = panel_xs[0]
    p4_center = p4_x + PANEL_W / 2
    ax.text(p4_center, TITLE_Y_BOT, "fnet (3D)",
            fontsize=14, fontweight="bold", color=VIZ_FOUNDATIONS, ha="center")
    # Stack of 3 offset rectangles per block to suggest 3D depth.
    # Tightened to fit within the panel: 9 blocks × ~0.42 ≈ 3.8 units wide.
    bx0 = p4_x + 0.4
    enc_heights = [1.3, 1.05, 0.8, 0.55]
    dec_heights = [0.55, 0.8, 1.05, 1.3]
    block_w = 0.28; gap = 0.08; step = block_w + gap + 0.06  # 0.42 per block
    depth_offset_x = 0.05; depth_offset_y = 0.04
    x = bx0
    for h in enc_heights:
        y = 2.4 - h / 2
        for offset in range(3):
            ax.add_patch(Rect((x + offset*depth_offset_x, y + offset*depth_offset_y),
                              block_w, h,
                              facecolor=VIZ_FOUNDATIONS, alpha=0.45 + offset*0.15,
                              edgecolor="white", lw=0.8))
        x += step
    # Bottleneck stack
    for offset in range(3):
        ax.add_patch(Rect((x + offset*depth_offset_x, 2.4 - 0.20 + offset*depth_offset_y),
                          block_w, 0.40,
                          facecolor=VIZ_WRAP, alpha=0.5 + offset*0.15,
                          edgecolor="white", lw=0.8))
    x += step
    for h in dec_heights:
        y = 2.4 - h / 2
        for offset in range(3):
            ax.add_patch(Rect((x + offset*depth_offset_x, y + offset*depth_offset_y),
                              block_w, h,
                              facecolor=VIZ_JUDGMENT, alpha=0.45 + offset*0.15,
                              edgecolor="white", lw=0.8))
        x += step
    ax.text(p4_center, CAPTION_Y_BOT, "encoder-decoder over volumetric data (Z-stacks → 3D conv)",
            ha="center", fontsize=10, color="#444", style="italic")

    # ----- Panel 5: Diffusion -----
    p5_x = panel_xs[1]
    p5_center = p5_x + PANEL_W / 2
    ax.text(p5_center, TITLE_Y_BOT, "Diffusion",
            fontsize=14, fontweight="bold", color=VIZ_WRAP, ha="center")
    # 4 progressive denoising stages: pure noise → less → less → clean (DAPI-like blobs)
    df_x0 = p5_x + 0.4
    df_y_top = 3.2
    df_w, df_h = 0.85, 1.5
    df_stages_noise = [1.0, 0.7, 0.35, 0.05]  # noise fraction at each stage
    for i, noise_level in enumerate(df_stages_noise):
        sub_x = df_x0 + i * (df_w + 0.15)
        ax.add_patch(Rect((sub_x, df_y_top - df_h), df_w, df_h, facecolor="#222"))
        rng_s = np.random.default_rng(150 + i)
        # White speckles representing noise
        n_spk = int(120 * noise_level)
        xs_spk = sub_x + 0.04 + rng_s.uniform(0, df_w - 0.08, n_spk)
        ys_spk = df_y_top - df_h + 0.04 + rng_s.uniform(0, df_h - 0.08, n_spk)
        ax.scatter(xs_spk, ys_spk, s=2.2, c="white")
        # As noise decreases, add visible "structure" (DAPI-like blobs)
        if noise_level < 0.5:
            structure_alpha = 1.0 - noise_level * 2  # more visible as noise drops
            n_blobs = int(5 * (1 - noise_level))
            for _ in range(n_blobs):
                bx_b = sub_x + 0.15 + rng_s.uniform(0, df_w - 0.30)
                by_b = df_y_top - df_h + 0.15 + rng_s.uniform(0, df_h - 0.30)
                ax.add_patch(Circ((bx_b, by_b), 0.10,
                                   facecolor="white", alpha=structure_alpha, edgecolor="none"))
        if i < len(df_stages_noise) - 1:
            ax.annotate("", xy=(sub_x + df_w + 0.13, df_y_top - df_h/2),
                        xytext=(sub_x + df_w + 0.02, df_y_top - df_h/2),
                        arrowprops=dict(arrowstyle="->", color=VIZ_WRAP, lw=1.4))
    # Stage labels
    ax.text(df_x0 + df_w/2, df_y_top - df_h - 0.18, "T",
            ha="center", fontsize=9, color="#666", style="italic")
    ax.text(df_x0 + 3 * (df_w + 0.15) + df_w/2, df_y_top - df_h - 0.18, "0",
            ha="center", fontsize=9, color="#666", style="italic")
    ax.annotate("", xy=(df_x0 + 3 * (df_w + 0.15) + df_w/2 + 0.25, df_y_top - df_h - 0.18),
                xytext=(df_x0 + df_w/2 - 0.25, df_y_top - df_h - 0.18),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1))
    ax.text(p5_center, CAPTION_Y_BOT, "iteratively denoise pure noise → clean image",
            ha="center", fontsize=10, color="#444", style="italic")

    # ----- Panel 6: takeaway box -----
    p6_x = panel_xs[2]
    p6_center = p6_x + PANEL_W / 2
    ax.text(p6_center, TITLE_Y_BOT, "Common scaffold",
            fontsize=14, fontweight="bold", color=NAVY, ha="center")
    # Takeaway: what to look for in any new paper
    ax.add_patch(FancyBboxPatch((p6_x + 0.2, 1.2), PANEL_W - 0.4, 2.0,
                                 boxstyle="round,pad=0.05,rounding_size=0.12",
                                 facecolor="#F4FBFF", edgecolor=NAVY, linewidth=1.5))
    ax.text(p6_x + 0.4, 2.9, "When you read a new paper, ask:",
            fontsize=10, color=NAVY, fontweight="bold")
    bullets = [
        "• What does it predict?",
        "• What's the loss?",
        "• What data does it need?",
        "• Is there a bottleneck or skips?",
    ]
    for i, b in enumerate(bullets):
        ax.text(p6_x + 0.4, 2.55 - i * 0.32, b, fontsize=10, color="#222")

    save(fig, "viz_s3_other")


def fig_viz_s4_overview():
    """Forward pass + backprop flow. Bigger element sizing; full picture
    visible without crowding."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7); ax.axis("off")
    ax.text(8, 6.7, "How models learn: forward pass + gradient feedback loop",
            ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)

    # "Repeat for thousands of batches" loop indicator
    ax.text(1.2, 5.7, "↻ repeat for thousands of batches",
            ha="left", va="center", fontsize=10.5, color="#666", style="italic")

    # 4 large boxes in a horizontal row at y=3.2..4.8
    boxes = [
        (1.0, "Data",        "labeled examples",       "#E0F4FF", VIZ_FOUNDATIONS),
        (4.5, "Model",       "neural network θ",        "#EBE6F2", VIZ_WRAP),
        (8.0, "Predictions", "ŷ = f(x; θ)",             "#E0F4FF", VIZ_FOUNDATIONS),
        (11.5,"Loss",        "L(ŷ, y)",                 "#FCE8F2", VIZ_JUDGMENT),
    ]
    box_w, box_h = 3.0, 1.8
    box_y = 3.4
    for x, title, sub, fill, color in boxes:
        ax.add_patch(FancyBboxPatch((x, box_y), box_w, box_h,
                                     boxstyle="round,pad=0.05,rounding_size=0.12",
                                     facecolor=fill, edgecolor=color, linewidth=2.5))
        ax.text(x + box_w/2, box_y + box_h - 0.45, title,
                ha="center", fontsize=15, fontweight="bold", color=color)
        ax.text(x + box_w/2, box_y + 0.5, sub,
                ha="center", fontsize=11, color="#333", style="italic")

    # Forward arrows between boxes (cyan, thick)
    forward_color = VIZ_FOUNDATIONS
    for i in range(len(boxes) - 1):
        x_from = boxes[i][0] + box_w + 0.02
        x_to = boxes[i+1][0] - 0.02
        ax.annotate("", xy=(x_to, box_y + box_h/2), xytext=(x_from, box_y + box_h/2),
                    arrowprops=dict(arrowstyle="->", color=forward_color, lw=3))
    # "forward pass →" label above the arrows
    ax.text(8, box_y + box_h + 0.5, "forward pass →",
            ha="center", fontsize=13, color=forward_color, fontweight="bold")

    # Backprop arc: starts from Loss (right), curves DOWN below the boxes,
    # lands on Model (second from left). Arrow points to Model.
    # rad < 0 → arc curves to the right of the from→to direction = below when going right→left.
    backprop = FancyArrowPatch((boxes[3][0] + box_w/2, box_y - 0.05),
                                (boxes[1][0] + box_w/2, box_y - 0.05),
                                connectionstyle="arc3,rad=-0.45",
                                arrowstyle="->", color=VIZ_JUDGMENT, lw=2.8, linestyle="--",
                                mutation_scale=22)
    ax.add_patch(backprop)
    ax.text(8, 1.2, "← backpropagation: ∂L/∂θ updates the weights",
            ha="center", fontsize=12, color=VIZ_JUDGMENT, fontweight="bold", style="italic")
    # Push ymin down so the arc has clearance below the boxes
    ax.set_ylim(0.3, 7)

    save(fig, "viz_s4_overview")


def fig_viz_s4_loss_landscape():
    """Loss landscape with three gradient-descent trajectories: two reach the
    global minimum, one stalls at a local minimum. Two panels: top-down contour
    on the left, 3-D surface on the right. Legend placed BELOW the contour panel
    so it never overlaps any trajectory or annotation."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (projection registration)

    # ---- analytic loss surface: two basins + gentle bowl --------------------
    def loss(xv, yv):
        return (-2.2 * np.exp(-((xv - 1.2) ** 2 + (yv - 0.8) ** 2) / 0.7)
                - 1.0 * np.exp(-((xv + 1.4) ** 2 + (yv + 1.0) ** 2) / 0.5)
                + 0.05 * (xv ** 2 + yv ** 2))

    def grad(pt, h=1e-3):
        fx = (loss(pt[0] + h, pt[1]) - loss(pt[0] - h, pt[1])) / (2 * h)
        fy = (loss(pt[0], pt[1] + h) - loss(pt[0], pt[1] - h)) / (2 * h)
        return np.array([fx, fy])

    def descend(start, lr=0.10, steps=400, tol=1e-3):
        path = [np.array(start, dtype=float)]
        for _ in range(steps):
            g = grad(path[-1])
            path.append(path[-1] - lr * g)
            if np.linalg.norm(g) < tol:
                break
        return np.array(path)

    x = np.linspace(-2.5, 2.5, 220)
    y = np.linspace(-2.5, 2.5, 220)
    X, Y = np.meshgrid(x, y)
    Z = loss(X, Y)

    # init points chosen so trajectories don't overlap each other or the legend
    path_A = descend([1.8, 2.0])    # → global
    path_B = descend([2.2, -1.5])   # → global
    path_C = descend([-2.0, 0.4])   # → local

    A_COLOR = VIZ_FOUNDATIONS   # cyan
    B_COLOR = VIZ_JUDGMENT      # magenta
    C_COLOR = "#222"            # dark — "stuck" trajectory

    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.18)

    # ---- LEFT: top-down contour view ---------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.contourf(X, Y, Z, levels=30, cmap="Purples_r", alpha=0.92)
    ax1.contour(X, Y, Z, levels=10, colors="white", linewidths=0.5, alpha=0.55)

    for path, color, label in [
        (path_A, A_COLOR, "init A → global min"),
        (path_B, B_COLOR, "init B → global min"),
        (path_C, C_COLOR, "init C → local min"),
    ]:
        ax1.plot(path[:, 0], path[:, 1], "-", color=color, lw=2.4, label=label,
                 solid_capstyle="round")
        ax1.plot(path[0, 0], path[0, 1], "o", color=color, markersize=10,
                 markeredgecolor="white", markeredgewidth=1.6, zorder=5)
        ax1.plot(path[-1, 0], path[-1, 1], "*", color=color, markersize=20,
                 markeredgecolor="white", markeredgewidth=1.4, zorder=5)

    # Minima labels — placed OFF to the side of each star so trajectories
    # arriving from above/below never cross the text. Small dark background box
    # keeps them legible against the purple contour fill.
    label_bbox = dict(boxstyle="round,pad=0.18", facecolor="#3A1F5C",
                      edgecolor="white", linewidth=0.8, alpha=0.85)
    ax1.text(0.35, 0.8, "global", ha="right", va="center", fontsize=10.5,
             color="white", fontweight="bold", bbox=label_bbox)
    ax1.text(-0.55, -1.0, "local", ha="left", va="center", fontsize=10.5,
             color="white", fontweight="bold", bbox=label_bbox)

    ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-2.5, 2.5)
    ax1.set_xlabel("weight 1", fontsize=11)
    ax1.set_ylabel("weight 2", fontsize=11)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("Top-down view (lower = better)", fontsize=13, color=NAVY,
                  fontweight="bold", pad=8)
    # Legend BELOW the data area — horizontal, three columns, no frame
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3,
               frameon=False, fontsize=10.5, handlelength=1.6, columnspacing=1.8)

    # ---- RIGHT: 3-D surface view -------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    stride = 4
    ax2.plot_surface(X[::stride, ::stride], Y[::stride, ::stride],
                     Z[::stride, ::stride], cmap="Purples_r",
                     alpha=0.88, linewidth=0, antialiased=True, edgecolor="none")
    z_floor = float(Z.min()) - 0.25
    ax2.contour(X, Y, Z, zdir="z", offset=z_floor, levels=12,
                cmap="Purples_r", alpha=0.55)

    # Project each trajectory onto the surface (slightly above so it reads)
    for path, color in [(path_A, A_COLOR), (path_B, B_COLOR), (path_C, C_COLOR)]:
        zs = np.array([loss(p[0], p[1]) for p in path]) + 0.04
        ax2.plot(path[:, 0], path[:, 1], zs, "-", color=color, lw=2.0,
                 solid_capstyle="round")
        ax2.scatter([path[-1, 0]], [path[-1, 1]], [zs[-1]], color=color,
                    s=90, marker="*", edgecolor="white", linewidth=1.0,
                    depthshade=False)

    ax2.set_xlabel("w₁", fontsize=10, labelpad=-10)
    ax2.set_ylabel("w₂", fontsize=10, labelpad=-10)
    ax2.set_zlabel("loss", fontsize=10, labelpad=-10)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    ax2.set_title("3-D surface view", fontsize=13, color=NAVY,
                  fontweight="bold", pad=8)
    ax2.view_init(elev=32, azim=-58)

    fig.suptitle("Gradient descent walks downhill on the loss surface",
                 color=NAVY, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "viz_s4_loss_landscape")


def fig_viz_s7_habits():
    """Three reproducibility habits, densified with 'what to log' bullets per Nikos's note."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6.5); ax.axis("off")
    ax.text(8, 6.2, "Three reproducibility habits",
            ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)

    habits = [
        (0.4, 5.4, "Version everything", VIZ_FOUNDATIONS, "#F4FBFF",
         "Model · training data · parameters",
         ["• git SHA for code", "• checksum for weights", "• DOI for data", "• random seeds"]),
        (5.6, 10.4, "Document decisions", VIZ_WRAP, "#EBE6F2",
         "Why this model · why this threshold",
         ["• which models tried", "• why this one shipped", "• threshold rationale", "• hyperparameter notes"]),
        (10.6, 15.6, "Reporting standards", VIZ_JUDGMENT, "#FCE8F2",
         "CLAIM · CONSORT-AI · MI-CLAIM",
         ["• CLAIM (medical imaging)", "• CONSORT-AI (trials)", "• STARD-AI (diagnostic)", "• MI-CLAIM (min-info)"]),
    ]
    for x0, x1, title, color, fill, subtitle, bullets in habits:
        w = x1 - x0
        ax.add_patch(FancyBboxPatch((x0, 0.4), w, 5.1,
                                     boxstyle="round,pad=0.05,rounding_size=0.15",
                                     facecolor=fill, edgecolor=color, linewidth=2.5))
        ax.text(x0 + w/2, 4.85, title, ha="center", va="center",
                fontsize=15, fontweight="bold", color=color)
        ax.text(x0 + w/2, 4.25, subtitle, ha="center", va="center",
                fontsize=10.5, color="#555", style="italic")
        # Bullets list
        for i, bullet in enumerate(bullets):
            ax.text(x0 + 0.4, 3.4 - i * 0.55, bullet, ha="left", va="center",
                    fontsize=10.5, color="#222")

    save(fig, "viz_s7_habits")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures...")
    # Section 2 task taxonomy
    fig_classification()
    fig_detection()
    fig_segmentation()
    fig_restoration()
    fig_generation()
    fig_registration()
    fig_tracking()
    # Section 4 failure modes
    fig_failure_domain_shift()
    fig_failure_hallucination()
    fig_failure_misregistration()
    # Section 5 metrics vs biology
    fig_metrics_error_examples()
    fig_metrics_vs_biology()
    # Section 1 ecosystem
    fig_ecosystem()
    # Section 3 train/val/test
    fig_train_val_test()
    # Section 5 validation gradient
    fig_validation_gradient()
    # viz_* figures (Phase 3 Group A regenerations)
    fig_viz_roadmap()
    fig_viz_s5_patterns()
    # viz_* figures (Phase 3 Group B regenerations — resize for 16:9)
    fig_viz_s2_seg_works()
    fig_viz_s2_seg_fails()
    fig_viz_s4_generalization()
    fig_viz_s4_modes()
    fig_viz_s5_overview()
    fig_viz_s5_fails()
    # viz_* figures (Phase 3 Group C regenerations — architecture + flow redraws)
    fig_viz_s2_overview()
    fig_viz_s3_overview()
    fig_viz_s3_cnn()
    fig_viz_s3_gan()
    fig_viz_s3_other()
    fig_viz_s4_overview()
    fig_viz_s4_loss_landscape()
    fig_viz_s7_habits()
    print("Done.")
