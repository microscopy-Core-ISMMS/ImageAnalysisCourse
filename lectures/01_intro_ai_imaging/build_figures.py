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
    img, _ = synth_cells(size=180, n=8, seed=5)
    rng = np.random.default_rng(5)
    noisy = img * 20  # simulate low photon count
    noisy = rng.poisson(np.clip(noisy, 0, None)).astype(float) / 20
    noisy = noisy + rng.normal(0, 0.08, noisy.shape)
    noisy = np.clip(noisy, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    axes[0].imshow(noisy, cmap="gray"); axes[0].set_title("Noisy input (low light)", color=NAVY, fontweight="bold")
    axes[1].imshow(img, cmap="gray"); axes[1].set_title("Restored output (model prediction)", color=NAVY, fontweight="bold")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Restoration → image-to-image: denoising, deconvolution, super-resolution", color=NAVY, fontsize=13, fontweight="bold", y=0.04)
    save(fig, "task_restoration")


def fig_generation():
    img, _ = synth_cells(size=180, n=10, seed=6)
    # 'Brightfield-like' input: low contrast, no fluorescence pattern
    bf = 0.5 + (img - img.mean()) * 0.3
    bf = np.clip(bf, 0, 1)
    # Generated output: predicted fluorescence
    gen = img.copy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    axes[0].imshow(bf, cmap="gray"); axes[0].set_title("Brightfield input (no labels)", color=NAVY, fontweight="bold")
    axes[1].imshow(gen, cmap="viridis"); axes[1].set_title("Predicted fluorescence (generated)", color=NAVY, fontweight="bold")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Generation → in silico labeling, virtual staining, synthetic data", color=NAVY, fontsize=13, fontweight="bold", y=0.04)
    save(fig, "task_generation")


def fig_registration():
    img1, _ = synth_cells(size=180, n=8, seed=7)
    # Shifted version
    shift = 12
    img2 = np.roll(img1, shift, axis=0)
    img2 = np.roll(img2, shift, axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].imshow(img1, cmap="Reds_r"); axes[0].set_title("Image A", color=NAVY, fontweight="bold")
    axes[1].imshow(img2, cmap="Blues_r"); axes[1].set_title("Image B (shifted)", color=NAVY, fontweight="bold")
    # Composite overlay
    overlay = np.zeros((*img1.shape, 3))
    overlay[..., 0] = img1
    overlay[..., 2] = img2
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay before alignment", color=NAVY, fontweight="bold")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Registration → align images to a common spatial reference", color=NAVY, fontsize=13, fontweight="bold", y=0.04)
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
    img1, _ = synth_cells(size=160, n=8, seed=15)
    # "Bad" registration: alignment is off in non-uniform way
    Y, X = np.indices(img1.shape, dtype=float)
    cy, cx = 80, 80
    # Local rotation + shift
    theta = (np.sqrt((Y-cy)**2 + (X-cx)**2) / 80) * 0.15
    Y2 = cy + (Y-cy) * np.cos(theta) - (X-cx) * np.sin(theta) + 8
    X2 = cx + (Y-cy) * np.sin(theta) + (X-cx) * np.cos(theta) + 4
    Y2 = np.clip(Y2, 0, img1.shape[0]-1).astype(int)
    X2 = np.clip(X2, 0, img1.shape[1]-1).astype(int)
    img2_misreg = img1[Y2, X2]
    # Composite overlay
    overlay = np.zeros((*img1.shape, 3))
    overlay[..., 0] = img1
    overlay[..., 2] = img2_misreg
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    # Good registration: just shifted (still close)
    overlay_good = np.zeros((*img1.shape, 3))
    overlay_good[..., 0] = img1
    overlay_good[..., 2] = img1  # perfect alignment
    axes[0].imshow(overlay_good); axes[0].set_title("Successful registration\n(structures align)", color=GOOD, fontweight="bold")
    axes[1].imshow(overlay); axes[1].set_title("Failed registration\n(local distortion, structures drift)", color=BAD, fontweight="bold")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle("Misregistration: alignment looks reasonable globally but distorts locally", color=NAVY, fontsize=13, fontweight="bold", y=0.02)
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
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    layers = [
        ("Hardware: GPUs, cloud compute, HPC clusters", 5.0, "#E8E8E8"),
        ("Data: BBBC · IDR · BioImage Archive · Allen Cell Image Library", 3.7, "#D5E8F0"),
        ("Models: pretrained Cellpose-SAM · μSAM · CARE · VoxelMorph · ...", 2.4, "#A6CEE3"),
        ("Platforms: ZeroCostDL4Mic · DL4MicEverywhere · BioImage Model Zoo", 1.1, "#1F78B4"),
    ]
    for label, y, color in layers:
        rect = Rectangle((0.5, y-0.45), 9, 0.9, facecolor=color, edgecolor=NAVY, linewidth=1.2)
        ax.add_patch(rect)
        text_color = "white" if color == "#1F78B4" else NAVY
        ax.text(5, y, label, ha="center", va="center", fontsize=11.5, color=text_color, fontweight="bold")
    # Top label
    ax.text(5, 5.85, "Layered ecosystem that made AI-in-microscopy a routine tool",
            ha="center", color=NAVY, fontsize=13, fontweight="bold")
    # Up arrow on the left
    ax.annotate("", xy=(0.2, 5.3), xytext=(0.2, 0.7),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=2))
    ax.text(0.05, 3, "stack matures\nbottom-up", rotation=90, ha="center", va="center",
            color=NAVY, fontsize=10, fontweight="bold")
    save(fig, "ecosystem_layers")


# --------------------------------------------------------------------------
# Section 3 — train/val/test diagram
# --------------------------------------------------------------------------
def fig_train_val_test():
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    # Three blocks
    blocks = [
        ("TRAIN (70%)", 0, 7, "#4C72B0", "model fits to this data"),
        ("VAL (15%)", 7, 1.5, "#DD8452", "tune hyperparameters here"),
        ("TEST (15%)", 8.5, 1.5, "#55A868", "report once, at the end"),
    ]
    for label, x, w, color, sub in blocks:
        rect = Rectangle((x, 1), w, 1.2, facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, 1.6, label, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
        ax.text(x + w/2, 0.6, sub, ha="center", va="center",
                color=NAVY, fontsize=9, style="italic")
    # Arrow showing time/separation
    ax.text(5, 2.6, "The three subsets are disjoint. Touch the test set once.",
            ha="center", color=NAVY, fontsize=11, fontweight="bold")
    save(fig, "train_val_test")


# --------------------------------------------------------------------------
# Section 5 — validation gradient
# --------------------------------------------------------------------------
def fig_validation_gradient():
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    # Gradient bar
    n_segments = 100
    cmap = plt.cm.RdYlGn
    for i in range(n_segments):
        rect = Rectangle((0.5 + i*9/n_segments, 1.2), 9/n_segments, 0.5,
                         facecolor=cmap(i/n_segments), edgecolor="none")
        ax.add_patch(rect)
    rect = Rectangle((0.5, 1.2), 9, 0.5, fill=False, edgecolor=NAVY, linewidth=1.5)
    ax.add_patch(rect)
    # Labels at intervals
    points = [
        (0.5, "Internal\n(single dataset)"),
        (3.0, "Multi-condition\nor cross-validation"),
        (5.5, "External\n(other lab/scanner)"),
        (7.5, "Multi-site\nprospective"),
        (9.5, "Regulatory\nclearance"),
    ]
    for x, label in points:
        ax.plot([x, x], [1.05, 1.85], color=NAVY, linewidth=1.2)
        ax.text(x, 0.85, label, ha="center", va="top", fontsize=9.5, color=NAVY, fontweight="bold")
    # Top label
    ax.text(0.5, 2.3, "Research-grade", ha="left", color=NAVY, fontsize=11, fontweight="bold")
    ax.text(9.5, 2.3, "Clinical-grade", ha="right", color=NAVY, fontsize=11, fontweight="bold")
    ax.annotate("", xy=(9.5, 2.3), xytext=(2.0, 2.3),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=2))
    ax.text(5, 2.7, "Validation gradient: the rigor required scales with the stakes",
            ha="center", color=NAVY, fontsize=12, fontweight="bold")
    save(fig, "validation_gradient")


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
    print("Done.")
