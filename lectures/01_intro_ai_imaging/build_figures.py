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
    # Same model on two different sample types
    img_easy, _ = synth_cells(size=160, n=10, seed=11, irregular=False)
    img_hard, _ = synth_cells(size=160, n=10, seed=12, irregular=True)
    # "Predictions": good on easy, missing/merged on hard
    _, pred_easy = synth_cells(size=160, n=10, seed=11, irregular=False)
    _, pred_hard = synth_cells(size=160, n=6, seed=13, irregular=True)  # fewer detections
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes[0,0].imshow(img_easy, cmap="gray"); axes[0,0].set_title("In-distribution image\n(round, isolated cells)", color=NAVY, fontweight="bold", fontsize=11)
    axes[0,1].imshow(img_easy, cmap="gray")
    axes[0,1].imshow(np.where(pred_easy > 0, pred_easy, np.nan), cmap=INSTANCE_CMAP, vmin=0, vmax=20, alpha=0.55)
    axes[0,1].set_title(f"Prediction: {pred_easy.max()} objects (correct)", color=GOOD, fontweight="bold", fontsize=11)
    axes[1,0].imshow(img_hard, cmap="gray"); axes[1,0].set_title("Out-of-distribution image\n(irregular, dense)", color=NAVY, fontweight="bold", fontsize=11)
    axes[1,1].imshow(img_hard, cmap="gray")
    axes[1,1].imshow(np.where(pred_hard > 0, pred_hard, np.nan), cmap=INSTANCE_CMAP, vmin=0, vmax=20, alpha=0.55)
    axes[1,1].set_title(f"Prediction: {pred_hard.max()} objects (under-counts)", color=BAD, fontweight="bold", fontsize=11)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Domain shift: same model, different sample type → silent under-counting", color=NAVY, fontsize=13, fontweight="bold", y=1.0)
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
def fig_metrics_vs_biology():
    """Two-mode construction that actually demonstrates the metric-biology gap.

    Three clouds:
      - Mode A (boundary erosion): drops IoU, leaves count untouched.
        Diagonal cloud at low count error, varying IoU.
      - Mode B (mask merges): leaves IoU near 1.0, drops count.
        Vertical cloud at high IoU, varying count error.
      - Mode C (mixed): both perturbations active. Diffuse cloud in between.

    The gap is the vertical spread of Mode B: at the same high IoU, count
    error covers a wide range. That's the message.
    """
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

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(ious_a, count_err_a, c="#4C72B0", s=70, edgecolor="black", linewidth=0.4,
               label="Boundary errors only (hurts IoU)", zorder=3)
    ax.scatter(ious_b, count_err_b, c="#C44E52", s=70, edgecolor="black", linewidth=0.4,
               label="Mask merges only (hurts count)", zorder=3)
    ax.scatter(ious_c, count_err_c, c="#888888", s=45, edgecolor="black", linewidth=0.4,
               alpha=0.7, label="Mixed errors", zorder=2)

    ax.set_xlabel("IoU (pixel overlap with ground truth)", fontsize=12)
    ax.set_ylabel("Absolute cell-count error", fontsize=12)
    ax.set_title("High IoU does not guarantee a correct biological count",
                 color=NAVY, fontweight="bold", fontsize=13, pad=14)
    ax.invert_xaxis()
    ax.set_ylim(-0.6, 5.6)
    ax.set_xlim(1.02, 0.73)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9.5)

    # Annotate the gap. Position the callout in clear space (lower-right, where IoU is
    # low and few points sit) and point the arrow at the vertical spread of the red
    # cloud at high IoU.
    ax.annotate("At IoU = 0.97,\ncount error ranges 0–4.\nThe gap is real.",
                xy=(0.965, 3.6), xytext=(0.83, 4.4),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5),
                fontsize=11, color=ACCENT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=ACCENT, lw=1.2))
    for s in ax.spines.values(): s.set_color(NAVY)
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
    # Top label + Research → Clinical arrow above the bar
    ax.text(1.0, 3.2, "Research-grade", ha="left", color=NAVY, fontsize=11.5, fontweight="bold")
    ax.text(13.0, 3.2, "Clinical-grade", ha="right", color=NAVY, fontsize=11.5, fontweight="bold")
    ax.annotate("", xy=(13.0, 3.2), xytext=(2.6, 3.2),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=2))
    ax.text(7, 3.7, "Validation gradient: the rigor required scales with the stakes",
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
    """Cellpose-SAM failure: dense OOD tissue → silent under-counting + merges."""
    img, masks_true = synth_cells(size=220, n=58, seed=22, irregular=True)
    # Bad model: only finds ~15 of the 58 nuclei, and several are merged
    _, masks_bad = synth_cells(size=220, n=15, seed=23, irregular=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Tissue-like dense overlap\n(synthetic OOD sample)",
                      color=NAVY, fontweight="bold", fontsize=14)
    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(np.where(masks_bad > 0, masks_bad, np.nan), cmap=INSTANCE_CMAP,
                   vmin=0, vmax=20, alpha=0.55)
    n_detected = int(masks_bad.max())
    n_true = int(masks_true.max())
    axes[1].set_title(f"Pretrained segmenter on OOD: {n_detected} 'cells' detected\n(true = {n_true} — silent under-counting)",
                      color=BAD, fontweight="bold", fontsize=14)
    # Annotate two merge regions
    for cy, cx in [(60, 50), (140, 130)]:
        axes[1].annotate("merge", xy=(cx, cy), xytext=(cx + 25, cy - 25),
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
    """Works vs fails — 10-panel grid (5 works + 5 fails)."""
    fig = plt.figure(figsize=(16, 7.5))
    fig.suptitle("Works vs fails — concrete examples",
                 color=NAVY, fontsize=15, fontweight="bold", y=0.99)

    works = [
        ("Segmentation",  synth_cells(seed=51, n=8)),
        ("Restoration",   synth_cells(seed=52, n=6)),
        ("Classify",      synth_cells(seed=53, n=5)),
        ("Stitch",        synth_cells(seed=54, n=4)),
        ("μSAM click",    synth_cells(seed=55, n=6)),
    ]
    fails = [
        ("OOD undercount", synth_cells(seed=61, n=10, irregular=True)),
        ("Rare miss",      synth_cells(seed=62, n=6, irregular=True)),
        ("Sample drift",   synth_cells(seed=63, n=4)),
        ("Edge effects",   synth_cells(seed=64, n=8)),
        ("Hallucinated",   synth_cells(seed=65, n=5)),
    ]
    gs = fig.add_gridspec(2, 6, width_ratios=[0.5, 1, 1, 1, 1, 1],
                          hspace=0.18, wspace=0.1, top=0.92, bottom=0.04,
                          left=0.02, right=0.98)
    # Left labels
    ax_lw = fig.add_subplot(gs[0, 0]); ax_lw.axis("off")
    ax_lw.text(0.5, 0.5, "WORKS", ha="center", va="center", rotation=90,
               fontsize=18, fontweight="bold", color=VIZ_FOUNDATIONS)
    ax_lf = fig.add_subplot(gs[1, 0]); ax_lf.axis("off")
    ax_lf.text(0.5, 0.5, "FAILS", ha="center", va="center", rotation=90,
               fontsize=18, fontweight="bold", color=VIZ_JUDGMENT)
    # Panels
    for i, (label, (img, masks)) in enumerate(works):
        ax = fig.add_subplot(gs[0, 1 + i])
        ax.imshow(img, cmap="gray")
        ax.imshow(np.where(masks > 0, masks, np.nan), cmap=INSTANCE_CMAP,
                  vmin=0, vmax=20, alpha=0.5)
        ax.set_title(label, fontsize=12, color=VIZ_FOUNDATIONS, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    for i, (label, (img, masks)) in enumerate(fails):
        ax = fig.add_subplot(gs[1, 1 + i])
        ax.imshow(img, cmap="gray")
        # Overlay sparse / wrong markers for failure illustration
        ax.imshow(np.where(masks > 0, masks, np.nan), cmap=INSTANCE_CMAP,
                  vmin=0, vmax=20, alpha=0.4)
        ax.set_title(label, fontsize=12, color=VIZ_JUDGMENT, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    save(fig, "viz_s5_overview")


def fig_viz_s5_fails():
    """Five failure patterns: 5-panel row, paired with viz_s5_works in the deck."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5))
    fig.suptitle("Where AI fails — five patterns to recognize",
                 color=NAVY, fontsize=15, fontweight="bold", y=1.0)

    cases = [
        ("OOD undercount",      synth_cells(seed=71, n=30, irregular=True), "30 cells → 'detected' 8"),
        ("Rare-category miss",  synth_cells(seed=72, n=6, irregular=True),  "(2 cells → 0 detected)"),
        ("Sample-prep drift",   synth_cells(seed=73, n=4, irregular=False), "(low contrast → silent fail)"),
        ("Edge effects",        synth_cells(seed=74, n=8, irregular=False), "(border cells dropped)"),
        ("Hallucinated feature",synth_cells(seed=75, n=4, irregular=False), "(circled spot is invented)"),
    ]
    for i, (title, (img, masks), caption) in enumerate(cases):
        ax = axes[i]
        ax.imshow(img, cmap="gray")
        # For hallucination, draw a red circle on the rightmost panel
        if "Halluc" in title:
            from matplotlib.patches import Circle
            c = Circle((img.shape[1] // 2, img.shape[0] // 2), 22,
                       fill=False, edgecolor=VIZ_JUDGMENT, linewidth=2.5)
            ax.add_patch(c)
            ax.text(img.shape[1] // 2, img.shape[0] // 2, "invented",
                    ha="center", va="center", fontsize=8, color=VIZ_JUDGMENT,
                    fontweight="bold", style="italic")
        else:
            # Overlay sparse correct/missed detections (e.g., for OOD undercount)
            ax.imshow(np.where(masks > 0, masks, np.nan), cmap=INSTANCE_CMAP,
                      vmin=0, vmax=20, alpha=0.45)
        ax.set_title(title, fontsize=12, color=VIZ_JUDGMENT, fontweight="bold")
        ax.text(0.5, -0.05, caption, ha="center", va="top",
                transform=ax.transAxes, fontsize=9, style="italic", color="#444")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
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
    """GAN Generator/Discriminator loop with arrows that clearly land on box edges."""
    from matplotlib.patches import FancyBboxPatch, Rectangle as Rect
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 7); ax.axis("off")
    ax.text(8, 6.7, "GAN — Generator vs Discriminator adversarial loop",
            ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)

    # Boxes (positions chosen so arrows have clean horizontal/vertical paths)
    # Real images box (top-left)
    real_x, real_y, real_w, real_h = 0.5, 4.2, 3.2, 1.6
    ax.add_patch(FancyBboxPatch((real_x, real_y), real_w, real_h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="#E0F4FF", edgecolor=VIZ_FOUNDATIONS, linewidth=2))
    ax.text(real_x + 0.15, real_y + real_h - 0.3, "Real images",
            fontsize=11, color=VIZ_FOUNDATIONS, fontweight="bold")
    # mini imgs
    for i in range(3):
        ax.add_patch(Rect((real_x + 0.3 + i * 0.9, real_y + 0.3), 0.7, 0.7,
                          facecolor="#333"))

    # Discriminator box (top-right)
    disc_x, disc_y, disc_w, disc_h = 9.0, 4.0, 3.2, 1.8
    ax.add_patch(FancyBboxPatch((disc_x, disc_y), disc_w, disc_h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="white", edgecolor=VIZ_JUDGMENT, linewidth=2))
    ax.text(disc_x + disc_w/2, disc_y + disc_h - 0.35, "Discriminator D",
            ha="center", fontsize=12, color=VIZ_JUDGMENT, fontweight="bold")
    ax.text(disc_x + disc_w/2, disc_y + disc_h/2 - 0.05, "real or fake?",
            ha="center", fontsize=10, color="#333", style="italic")

    # Discriminator output bubble (far right)
    out_x = 13.4
    ax.add_patch(FancyBboxPatch((out_x, 4.5), 2.0, 1.0,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="#F8F8F8", edgecolor="#333", linewidth=1.5))
    ax.text(out_x + 1.0, 5.15, "D(x) → [0, 1]", ha="center", fontsize=10,
            color="#333", fontweight="bold")
    ax.text(out_x + 1.0, 4.75, "1 = real\n0 = fake", ha="center", fontsize=8.5,
            color="#666", style="italic", linespacing=1.0)

    # Generator box (bottom-left)
    gen_x, gen_y, gen_w, gen_h = 0.5, 1.0, 3.2, 1.6
    ax.add_patch(FancyBboxPatch((gen_x, gen_y), gen_w, gen_h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="white", edgecolor="#333", linewidth=2))
    ax.text(gen_x + gen_w/2, gen_y + gen_h - 0.3, "Generator G",
            ha="center", fontsize=12, color="#333", fontweight="bold")
    ax.text(gen_x + gen_w/2, gen_y + 0.3, "noise z → image",
            ha="center", fontsize=10, color="#666", style="italic")

    # noise z input (far left)
    ax.text(gen_x - 0.4, gen_y + gen_h/2, "noise z →", ha="right", va="center",
            fontsize=10, color="#666", fontweight="bold")

    # Fake images box (bottom-middle)
    fake_x, fake_y, fake_w, fake_h = 4.6, 1.0, 3.2, 1.6
    ax.add_patch(FancyBboxPatch((fake_x, fake_y), fake_w, fake_h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="#FCE8F2", edgecolor=VIZ_JUDGMENT, linewidth=2))
    ax.text(fake_x + 0.15, fake_y + fake_h - 0.3, "Fake images G(z)",
            fontsize=11, color=VIZ_JUDGMENT, fontweight="bold")
    for i in range(3):
        ax.add_patch(Rect((fake_x + 0.3 + i * 0.9, fake_y + 0.3), 0.7, 0.7,
                          facecolor="#666"))

    # ARROWS — data flow (solid cyan) — landing exactly on box edges
    # Real → D
    ax.annotate("", xy=(disc_x - 0.02, disc_y + disc_h - 0.4),
                xytext=(real_x + real_w + 0.02, real_y + real_h/2),
                arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=2.2))
    # Generator → Fake
    ax.annotate("", xy=(fake_x - 0.02, fake_y + fake_h/2),
                xytext=(gen_x + gen_w + 0.02, gen_y + gen_h/2),
                arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=2.2))
    # Fake → D
    ax.annotate("", xy=(disc_x - 0.02, disc_y + 0.4),
                xytext=(fake_x + fake_w + 0.02, fake_y + fake_h/2),
                arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=2.2))
    # D → output
    ax.annotate("", xy=(out_x - 0.02, out_y_mid := 5.0),
                xytext=(disc_x + disc_w + 0.02, disc_y + disc_h/2),
                arrowprops=dict(arrowstyle="->", color=VIZ_FOUNDATIONS, lw=2.2))

    # GRADIENT FLOW (dashed magenta) — training signal back to G and D
    # G-step: gradient from D back to G (arc curves below the boxes, not through them)
    from matplotlib.patches import FancyArrowPatch
    g_arrow = FancyArrowPatch((disc_x + 0.6, disc_y - 0.02),
                               (gen_x + gen_w - 0.4, gen_y + gen_h + 0.02),
                               connectionstyle="arc3,rad=0.45",
                               arrowstyle="->", color=VIZ_JUDGMENT, lw=1.8, linestyle="--")
    ax.add_patch(g_arrow)
    ax.text(5.5, 3.3, "G-step: improve generator\n(move predictions toward 'real')",
            ha="center", fontsize=8.5, color=VIZ_JUDGMENT, style="italic", fontweight="bold")
    # D-step: gradient from D output back to D itself (small loop on the right side)
    d_arrow = FancyArrowPatch((out_x + 0.2, 4.5),
                               (disc_x + disc_w - 0.3, disc_y + 0.2),
                               connectionstyle="arc3,rad=0.35",
                               arrowstyle="->", color=VIZ_JUDGMENT, lw=1.8, linestyle="--")
    ax.add_patch(d_arrow)
    ax.text(14.3, 3.95, "D-step:\nimprove discriminator", ha="center", va="top",
            fontsize=8.5, color=VIZ_JUDGMENT, style="italic", fontweight="bold")

    # Legend
    ax.plot([13.4, 14.0], [3.5, 3.5], color=VIZ_FOUNDATIONS, lw=2.2)
    ax.text(14.1, 3.5, "data flow", va="center", fontsize=9, color="#333")
    ax.plot([13.4, 14.0], [3.05, 3.05], color=VIZ_JUDGMENT, lw=1.8, linestyle="--")
    ax.text(14.1, 3.05, "gradient flow (training signal)", va="center", fontsize=9, color="#333")

    save(fig, "viz_s3_gan")


def fig_viz_s3_other():
    """Encoder-decoder family — 4 panels with VISIBLY distinct schematics so
    U-Net, Autoencoder, pix2pix, fnet 3D each have an identifying feature.
    Arrows land on block edges, not floating in space."""
    from matplotlib.patches import Rectangle as Rect, FancyArrowPatch, Polygon, FancyBboxPatch, Circle as Circ
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.5); ax.axis("off")
    ax.text(8, 8.2, "Recognize the encoder-decoder pattern in any new bioimage paper",
            ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)

    # Helper: draw an encoder-decoder block sequence
    def draw_encoder_decoder(x_origin, y_origin, encoder_heights=(1.6, 1.3, 1.0, 0.7),
                             decoder_heights=(0.7, 1.0, 1.3, 1.6), block_w=0.4, gap=0.1,
                             skip=True, bottleneck=True, color_enc=VIZ_FOUNDATIONS,
                             color_dec=VIZ_JUDGMENT, label_below=None):
        """Returns the x-range used."""
        x = x_origin
        enc_xs = []
        for h in encoder_heights:
            y = y_origin - h / 2
            ax.add_patch(Rect((x, y), block_w, h, facecolor=color_enc, alpha=0.8,
                              edgecolor="white", lw=1.2))
            enc_xs.append((x, h))
            x += block_w + gap
        if bottleneck:
            ax.add_patch(Rect((x, y_origin - 0.25), 0.4, 0.5, facecolor=VIZ_WRAP,
                              edgecolor="white", lw=1.2))
            x += 0.4 + gap
        dec_xs = []
        for h in decoder_heights:
            y = y_origin - h / 2
            ax.add_patch(Rect((x, y), block_w, h, facecolor=color_dec, alpha=0.8,
                              edgecolor="white", lw=1.2))
            dec_xs.append((x, h))
            x += block_w + gap
        # Skip connections
        if skip:
            for (xe, he), (xd, hd) in zip(enc_xs, reversed(dec_xs)):
                arc = FancyArrowPatch((xe + block_w/2, y_origin + he/2 + 0.05),
                                      (xd + block_w/2, y_origin + hd/2 + 0.05),
                                      connectionstyle="arc3,rad=-0.35",
                                      arrowstyle="-", color="#666", lw=1.0)
                ax.add_patch(arc)
        # Bottleneck label
        if bottleneck:
            bn_idx = len(encoder_heights)
            bn_x = x_origin + bn_idx * (block_w + gap)
            ax.text(bn_x + 0.2, y_origin - 0.5, "z", ha="center", va="top",
                    fontsize=9, style="italic", color=VIZ_WRAP)
        return x_origin, x, dec_xs

    # ----- Panel 1 (top-left): U-Net (encoder + skips + decoder)
    ax.text(2.0, 7.3, "U-Net", fontsize=14, fontweight="bold", color=VIZ_FOUNDATIONS)
    draw_encoder_decoder(0.6, 6.0, skip=True, bottleneck=True, label_below="U-Net")
    ax.text(2.0, 4.7, "encoder-decoder + skip connections",
            ha="center", fontsize=10, color="#444", style="italic")

    # ----- Panel 2 (top-right): Autoencoder (no skips, visible latent dot)
    ax.text(10.0, 7.3, "Autoencoder", fontsize=14, fontweight="bold", color=VIZ_FOUNDATIONS)
    # Draw without skip arcs
    _, end_ae, _ = draw_encoder_decoder(8.6, 6.0, skip=False, bottleneck=True)
    # Bigger latent dot for emphasis
    ax.add_patch(Circ((8.6 + 4 * 0.5 + 0.2, 6.0), 0.18, facecolor=VIZ_WRAP, edgecolor="white", lw=1.5))
    ax.text(8.6 + 4 * 0.5 + 0.2, 5.55, "z (latent)", ha="center", va="top",
            fontsize=9, color=VIZ_WRAP, fontweight="bold", style="italic")
    ax.text(10.0, 4.7, "no skips — bottleneck forces compression",
            ha="center", fontsize=10, color="#444", style="italic")

    # ----- Panel 3 (bottom-left): pix2pix (U-Net generator + adversarial D box)
    ax.text(2.0, 3.5, "pix2pix", fontsize=14, fontweight="bold", color=VIZ_JUDGMENT)
    _, end_p2p, dec_xs_p2p = draw_encoder_decoder(0.6, 2.2, skip=True, bottleneck=True,
                                                   color_dec=VIZ_JUDGMENT)
    # Adversarial D box appended on the right
    d_box_x = end_p2p + 0.1
    ax.add_patch(FancyBboxPatch((d_box_x, 1.5), 0.9, 1.4,
                                 boxstyle="round,pad=0.05,rounding_size=0.08",
                                 facecolor="#FCE8F2", edgecolor=VIZ_JUDGMENT, lw=1.8))
    ax.text(d_box_x + 0.45, 2.2, "D\n(adv.)", ha="center", va="center",
            fontsize=10, color=VIZ_JUDGMENT, fontweight="bold")
    # Arrow from decoder output to D
    last_x, last_h = dec_xs_p2p[-1]
    ax.annotate("", xy=(d_box_x - 0.02, 2.2), xytext=(last_x + 0.42, 2.2),
                arrowprops=dict(arrowstyle="->", color=VIZ_JUDGMENT, lw=1.5))
    ax.text(2.0, 0.9, "U-Net generator + adversarial discriminator",
            ha="center", fontsize=10, color="#444", style="italic")

    # ----- Panel 4 (bottom-right): fnet 3D (stacked planes per block to suggest depth)
    ax.text(10.0, 3.5, "fnet (3D)", fontsize=14, fontweight="bold", color=VIZ_FOUNDATIONS)
    # Draw 4 encoder + bottleneck + 4 decoder, each as a STACK of 3 offset rectangles
    bx0 = 8.0
    enc_heights = [1.5, 1.2, 0.9, 0.6]
    dec_heights = [0.6, 0.9, 1.2, 1.5]
    block_w = 0.4; gap = 0.13
    x = bx0
    enc_xs_fnet = []
    for h in enc_heights:
        y = 2.2 - h / 2
        for offset in range(3):
            ax.add_patch(Rect((x + offset*0.07, y + offset*0.05), block_w, h,
                              facecolor=VIZ_FOUNDATIONS, alpha=0.45 + offset*0.15,
                              edgecolor="white", lw=0.8))
        enc_xs_fnet.append((x, h))
        x += block_w + gap + 0.15  # extra gap for the depth offset
    # Bottleneck stack
    for offset in range(3):
        ax.add_patch(Rect((x + offset*0.07, 2.2 - 0.25 + offset*0.05), 0.4, 0.5,
                          facecolor=VIZ_WRAP, alpha=0.5 + offset*0.15,
                          edgecolor="white", lw=0.8))
    x += 0.4 + gap + 0.15
    dec_xs_fnet = []
    for h in dec_heights:
        y = 2.2 - h / 2
        for offset in range(3):
            ax.add_patch(Rect((x + offset*0.07, y + offset*0.05), block_w, h,
                              facecolor=VIZ_JUDGMENT, alpha=0.45 + offset*0.15,
                              edgecolor="white", lw=0.8))
        dec_xs_fnet.append((x, h))
        x += block_w + gap + 0.15
    ax.text(10.0, 0.9, "encoder-decoder over volumetric data",
            ha="center", fontsize=10, color="#444", style="italic")
    ax.text(11.6, 1.1, "Z-stacks → 3D convolutions",
            ha="center", fontsize=9, color=VIZ_WRAP, style="italic")

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
    fig_viz_s7_habits()
    print("Done.")
