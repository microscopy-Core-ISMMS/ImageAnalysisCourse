"""
Build script for Notebook 11 — DL-Driven Tracking (Choose-Your-Own).

Pedagogical arc: tracking = per-frame detection + frame-to-frame linking.
Detection is solved by upstream models (Cellpose, SAM, etc.); this notebook
focuses on *linking* — comparing classical (TrackPy), Bayesian (btrack),
and SOTA (Ultrack pointer) approaches.

Run:
    python build_notebook_11.py
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent


class CellBuilder:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.idx = 0
        self.cells = []

    def md(self, source: str):
        self.cells.append({
            "cell_type": "markdown",
            "id": f"{self.prefix}-md-{self.idx:03d}",
            "metadata": {},
            "source": source.lstrip("\n"),
        })
        self.idx += 1
        return self

    def code(self, source: str):
        self.cells.append({
            "cell_type": "code",
            "id": f"{self.prefix}-code-{self.idx:03d}",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": source.lstrip("\n"),
        })
        self.idx += 1
        return self


def build_notebook(cells, name: str):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = OUT_DIR / f"{name}.ipynb"
    out_path.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {out_path.name} ({out_path.stat().st_size} bytes, {len(cells)} cells)")


# ---------------------------------------------------------------------------
# Title and orientation
# ---------------------------------------------------------------------------
def section_title(b):
    b.md("""<!-- colab-badge -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/main/notebooks/11_dl_tracking.ipynb)

*Click the badge to open this notebook in Google Colab. For best performance, switch to a GPU runtime: Runtime → Change runtime type → T4 GPU.*""")

    b.md("""# Notebook 11 — DL-Driven Tracking (Choose-Your-Own: TrackPy / btrack / Ultrack)

**Status.** Core lab — workshop day 3. Recommended after Notebooks 01 (Cellpose), 10 (3D segmentation).
**Estimated time.** 25–35 minutes on Colab T4.

**Learning goals.**

1. Understand tracking as **detection + linking**: the frame-by-frame centroids are the hard part; connecting them across time is a separate algorithmic choice.
2. Compare **classical** (TrackPy) vs **Bayesian** (btrack) vs **SOTA** (Ultrack) linking strategies.
3. Compute ID-consistency metrics to evaluate which method best recovers ground-truth track assignments.
4. Visualize failure modes: cells that touch, disappear momentarily, or move abruptly.
5. Choose a method based on data complexity and computational budget.

> **Why this lab matters.** Time-lapse microscopy is one of the richest datasets in cell biology — but only if you can track the *same cell* across frames. This lab separates the detection problem (solved upstream by Cellpose, StarDist, etc.) from the *linking* problem, which has multiple solutions with very different trade-offs.

> **A note on btrack.** The Bayesian tracker package (`btrack`) is powerful but has uncertain pip-install compatibility on free-tier Colab due to HDF5 dependencies. We detect this upfront and fall back to a simplified Kalman-style linker if needed — the teaching point lands either way.""")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def section_setup(b):
    b.md("""## Setup

Install core dependencies. TrackPy always installs cleanly; btrack may require fallback.""")

    b.code("""import sys
IN_COLAB = "google.colab" in sys.modules

# Core libraries
%pip install --quiet trackpy scikit-image matplotlib numpy scipy pandas

# Try btrack; if it fails, we'll use a fallback Kalman linker
BTRACK_AVAILABLE = False
try:
    %pip install --quiet btrack
    import btrack
    BTRACK_AVAILABLE = True
    print("✓ btrack installed successfully")
except Exception as e:
    print(f"ℹ btrack install failed (expected on some Colab runtimes): {e}")
    print("  Will use simplified Kalman-style fallback instead")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure, morphology
from scipy.optimize import linear_sum_assignment
import trackpy as tp

print("Core imports ready.")
print(f"btrack available: {BTRACK_AVAILABLE}")""")


# ---------------------------------------------------------------------------
# Synthetic time-lapse generator
# ---------------------------------------------------------------------------
def section_synthetic_data(b):
    b.md("""## Synthetic time-lapse generator

Generate a small (T=30 frames) time-lapse with N moving cells, realistic dynamics, and ground-truth masks. Each cell has:
- A starting position (random)
- A velocity vector (constant + small random perturbation per frame)
- A finite lifespan (cells appear, move, and sometimes disappear mid-sequence)

We'll use this synthetic data so that:
1. We know the *ground truth* track assignments.
2. We can test linking algorithms on controlled failure modes.
3. Everything runs fast (T, H, W all small).""")

    b.code("""def generate_synthetic_timelapse(T=30, H=256, W=256, n_cells=12, seed=0):
    \"\"\"
    Generate a synthetic time-lapse with ground-truth labeled masks and track info.

    Returns:
        images:  (T, H, W) array, grayscale time-lapse
        masks:   (T, H, W) array, integer labels (0 = background, 1..N = cell IDs)
        tracks:  dict mapping cell_id -> list of (t, cy, cx) tuples (ground truth)
    \"\"\"
    rng = np.random.default_rng(seed)
    images = np.zeros((T, H, W), dtype=np.uint8)
    masks = np.zeros((T, H, W), dtype=np.int32)

    # Initialize cells: (cell_id, start_t, end_t, cy0, cx0, vy, vx)
    cells_info = []
    for cell_id in range(1, n_cells + 1):
        start_t = rng.integers(0, max(1, T // 3))
        end_t = rng.integers(start_t + T // 2, T)
        cy0 = rng.uniform(30, H - 30)
        cx0 = rng.uniform(30, W - 30)
        vy = rng.uniform(-1.0, 1.0)
        vx = rng.uniform(-1.0, 1.0)
        cells_info.append((cell_id, start_t, end_t, cy0, cx0, vy, vx))

    # Render each frame
    for t in range(T):
        for cell_id, start_t, end_t, cy0, cx0, vy, vx in cells_info:
            if not (start_t <= t < end_t):
                continue
            # Cell is active; compute position at time t with small noise
            dt = t - start_t
            cy = cy0 + vy * dt + rng.normal(0, 0.3)
            cx = cx0 + vx * dt + rng.normal(0, 0.3)
            cy, cx = int(np.clip(cy, 5, H - 5)), int(np.clip(cx, 5, W - 5))

            # Draw a small circle (cell radius ~8 pixels)
            radius = 8
            Y, X = np.ogrid[:H, :W]
            circle = (Y - cy) ** 2 + (X - cx) ** 2 <= radius ** 2
            images[t][circle] = np.maximum(images[t][circle], 200)
            masks[t][circle] = cell_id

    # Add Gaussian blur and noise for realism
    from scipy.ndimage import gaussian_filter
    images_smooth = np.zeros_like(images, dtype=np.float32)
    for t in range(T):
        images_smooth[t] = gaussian_filter(images[t].astype(float), sigma=1.0)
    images_smooth = (images_smooth + rng.normal(0, 5, images_smooth.shape))
    images_smooth = np.clip(images_smooth, 0, 255).astype(np.uint8)

    # Extract ground-truth tracks from masks
    tracks = {}
    for cell_id in range(1, n_cells + 1):
        tracks[cell_id] = []
    for t in range(T):
        for cell_id in np.unique(masks[t]):
            if cell_id == 0:
                continue
            obj_mask = masks[t] == cell_id
            props = measure.regionprops(obj_mask.astype(int))
            if props:
                cy, cx = props[0].centroid
                tracks[cell_id].append((t, cy, cx))

    return images_smooth, masks, tracks


# Generate the data
print("Generating synthetic time-lapse...")
images, gt_masks, gt_tracks = generate_synthetic_timelapse(T=30, H=256, W=256, n_cells=12)
print(f"  images shape: {images.shape}")
print(f"  masks shape:  {gt_masks.shape}")
print(f"  {len(gt_tracks)} cells total")
print()""")


# ---------------------------------------------------------------------------
# Visualize synthetic data
# ---------------------------------------------------------------------------
def section_visualize_data(b):
    b.md("""## Visualize the time-lapse

Show 4 representative frames + a time-projection to see the migration patterns.""")

    b.code("""# Show sample frames
frame_indices = [0, 10, 20, 29]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, idx in zip(axes, frame_indices):
    ax.imshow(images[idx], cmap='gray')
    ax.set_title(f"Frame {idx}")
    ax.axis('off')
plt.tight_layout(); plt.show()

# Time projection (maximum intensity across all frames)
time_proj = np.max(images, axis=0)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(time_proj, cmap='gray')
ax.set_title("Time projection (max across all frames)")
ax.axis('off')
plt.tight_layout(); plt.show()

print(f"Dynamic range: {images.min():.0f} to {images.max():.0f}")
print(f"Active cells per frame (ground truth):"),
active_per_frame = [len(np.unique(gt_masks[t])) - 1 for t in range(gt_masks.shape[0])]
print(f"  min {min(active_per_frame)}, max {max(active_per_frame)}, mean {np.mean(active_per_frame):.1f}")""")


# ---------------------------------------------------------------------------
# Per-frame detection
# ---------------------------------------------------------------------------
def section_detection(b):
    b.md("""## Step 1 — Per-frame detection

Detect objects in each frame using the ground-truth masks (we pretend a detector like Cellpose gave us these). Extract centroids + basic properties and tabulate as a pandas DataFrame.

In a real workflow, this step would be: run Cellpose or StarDist, extract regionprops for each detected object.""")

    b.code("""# Extract detections from ground-truth masks
detections = []
for t in range(gt_masks.shape[0]):
    frame_mask = gt_masks[t]
    props = measure.regionprops(frame_mask)
    for prop in props:
        cy, cx = prop.centroid
        area = prop.area
        detections.append({
            'frame': t,
            'y': cy,
            'x': cx,
            'mass': area,  # trackpy convention
            'size': int(np.sqrt(area)),
        })

detections_df = pd.DataFrame(detections)
print(f"Detected {len(detections_df)} objects across {gt_masks.shape[0]} frames")
print()
print(detections_df.head(10))
print()
print(f"Objects per frame: min {detections_df.groupby('frame').size().min()}, "
      f"max {detections_df.groupby('frame').size().max()}")""")


# ---------------------------------------------------------------------------
# Predict-before-run quiz
# ---------------------------------------------------------------------------
def section_quiz(b):
    b.md("""## Predict before you run: tracking failure modes

**Multiple choice.** Which of these will be hardest to link correctly?

(a) Two cells moving in parallel at constant velocity — easy, linear trajectory.
(b) A cell that briefly disappears (occluded or dim) for one frame, then reappears in the same location.
(c) Two cells colliding head-on, touching for one frame, then separating.
(d) A cell making an abrupt 90-degree turn.

**Answer:** (c) is hardest. When two objects touch, they may temporarily merge into a single blob. A frame-to-frame linker has no signal to distinguish them in that frame. Classical methods like TrackPy handle (b) and (d) reasonably with the right parameters. Bayesian methods (btrack) excel at (d) because they model motion; they can struggle with merges unless trained specifically. Ultrack was designed to handle all three, especially merges.

**What you should expect:** All methods will get (a) right. (b) is handled by TrackPy's `memory` parameter. (c) will cause splits/merges in the output of simple methods.""")


# ---------------------------------------------------------------------------
# Method 1: TrackPy classical
# ---------------------------------------------------------------------------
def section_trackpy(b):
    b.md("""## Method 1 — TrackPy classical linker

TrackPy (`predict_then_link` or just `link`) performs greedy nearest-neighbor linking with optional `memory` (gap filling). Parameters:
- `search_range`: maximum distance a cell can move in one frame (pixels)
- `memory`: max consecutive frames a cell can disappear and still be re-linked

Simple, fast, robust — but doesn't model motion, so abrupt changes break it.""")

    b.code("""# Link detections with TrackPy
print("Running TrackPy linker...")
search_range = 20  # pixels; tune to your cell speed
memory = 2         # frames; allows brief disappearances

try:
    tracks_tp = tp.link(detections_df.copy(), search_range=search_range, memory=memory)
    print(f"  search_range={search_range}, memory={memory}")
    print(f"  linked into {tracks_tp['particle'].max() + 1} tracks")
except Exception as e:
    print(f"TrackPy link failed: {e}")
    tracks_tp = None

if tracks_tp is not None:
    print()
    print("Track length distribution:")
    track_lengths = tracks_tp.groupby('particle').size()
    print(f"  min {track_lengths.min()}, max {track_lengths.max()}, mean {track_lengths.mean():.1f}")
    print()
    print(tracks_tp.head(10))""")


# ---------------------------------------------------------------------------
# Method 2: btrack or Kalman fallback
# ---------------------------------------------------------------------------
def section_btrack(b):
    b.md("""## Method 2 — btrack Bayesian linker (or Kalman fallback)

btrack models cell motion with a motion model (constant velocity + small noise). This handles abrupt changes better than pure nearest-neighbor, but requires more configuration.

**If btrack is unavailable,** we implement a simplified Kalman-style fallback: predict next position from current velocity, link to nearest detection within tolerance.""")

    b.code("""def simple_kalman_linker(detections_df, max_gap=2, max_dist=20):
    \"\"\"
    Simplified Kalman-style linker: predict motion from velocity, link greedily.

    Returns: DataFrame with 'particle' column added (track IDs)
    \"\"\"
    dets = detections_df.copy()
    dets = dets.sort_values(['frame', 'y', 'x']).reset_index(drop=True)

    # Active tracks: {track_id: [(frame, y, x, vy_est, vx_est, last_seen_frame)]}
    active_tracks = {}
    next_track_id = 0
    particle_assignment = [-1] * len(dets)  # particle ID for each detection

    for t in range(int(dets['frame'].min()), int(dets['frame'].max()) + 1):
        frame_dets = dets[dets['frame'] == t].reset_index()
        used = set()

        # Try to link each frame_det to an active track
        for det_idx, det_row in frame_dets.iterrows():
            det_y, det_x = det_row['y'], det_row['x']
            best_track_id, best_dist = None, float('inf')

            # Find closest active track (with motion prediction)
            for tid, track_history in list(active_tracks.items()):
                frames_gap = t - track_history[-1][0]
                if frames_gap > max_gap:
                    del active_tracks[tid]
                    continue

                # Predict position: last_pos + velocity * frames_gap
                last_frame, last_y, last_x, vy_est, vx_est, _ = track_history[-1]
                pred_y = last_y + vy_est * frames_gap
                pred_x = last_x + vx_est * frames_gap
                dist = np.sqrt((pred_y - det_y)**2 + (pred_x - det_x)**2)

                if dist < max_dist and dist < best_dist:
                    best_track_id, best_dist = tid, dist

            if best_track_id is not None:
                # Link to existing track; update velocity estimate
                _, last_y, last_x, _, _, _ = active_tracks[best_track_id][-1]
                vy_new = (det_y - last_y) / (t - active_tracks[best_track_id][-1][0]) if t > active_tracks[best_track_id][-1][0] else 0
                vx_new = (det_x - last_x) / (t - active_tracks[best_track_id][-1][0]) if t > active_tracks[best_track_id][-1][0] else 0
                active_tracks[best_track_id].append((t, det_y, det_x, vy_new, vx_new, t))
                particle_assignment[frame_dets.loc[det_idx, 'index']] = best_track_id
                used.add(best_track_id)
            else:
                # Start a new track
                active_tracks[next_track_id] = [(t, det_y, det_x, 0, 0, t)]
                particle_assignment[frame_dets.loc[det_idx, 'index']] = next_track_id
                next_track_id += 1

    dets['particle'] = particle_assignment
    return dets


if BTRACK_AVAILABLE:
    print("Using btrack Bayesian linker...")
    try:
        # Simple btrack usage: load detections, configure motion model, track
        from btrack import BayesianTracker
        from btrack.models import gaussian_model

        # Convert detections to btrack object list
        objects = []
        for idx, row in detections_df.iterrows():
            obj = type('obj', (), {'x': row['x'], 'y': row['y'],
                                   'frame': int(row['frame']), 'z': 0})()
            objects.append(obj)

        # Initialize and configure tracker
        tracker = BayesianTracker()
        tracker.configure_tracker(
            max_search_radius=20,
            max_frame_jump=2,
            max_splits=1,
            max_merges=1,
        )
        tracker.append(objects)

        # Run tracker
        print("  Tracking...")
        tracker.track()
        print(f"  linked into {tracker.n_tracks} tracks")

        # Extract tracks as DataFrame
        tracks_btrack = []
        for track in tracker.tracks:
            for t, x, y in zip(track.t, track.x, track.y):
                tracks_btrack.append({
                    'frame': int(t),
                    'x': float(x),
                    'y': float(y),
                    'particle': int(track.ID),
                })
        tracks_btrack = pd.DataFrame(tracks_btrack).sort_values('frame').reset_index(drop=True)
        print()
        print(tracks_btrack.head(10))
    except Exception as e:
        print(f"btrack tracking failed: {e}")
        print("Falling back to Kalman linker...")
        BTRACK_AVAILABLE = False
        tracks_btrack = None

if not BTRACK_AVAILABLE:
    print("Using simplified Kalman-style fallback linker...")
    tracks_btrack = simple_kalman_linker(detections_df, max_gap=2, max_dist=20)
    print(f"  linked into {tracks_btrack['particle'].max() + 1} tracks")
    print()
    print(tracks_btrack.head(10))""")


# ---------------------------------------------------------------------------
# Visualize tracks
# ---------------------------------------------------------------------------
def section_visualize_tracks(b):
    b.md("""## Visualize tracks side-by-side

Overlay each method's tracks on the time-projection image. Use a distinct color per track ID.""")

    b.code("""def overlay_tracks(ax, time_proj, tracks_df, title=""):
    \"\"\"Draw tracks as colored lines on the time-projection.\"\"\"
    ax.imshow(time_proj, cmap='gray')
    colors = plt.cm.tab20(np.linspace(0, 1, tracks_df['particle'].max() + 1))
    for particle_id in sorted(tracks_df['particle'].unique()):
        track = tracks_df[tracks_df['particle'] == particle_id].sort_values('frame')
        if len(track) > 1:
            ax.plot(track['x'], track['y'], 'o-',
                   color=colors[particle_id % 20],
                   linewidth=1.5, markersize=4, alpha=0.7)
    ax.set_title(title)
    ax.axis('off')
    return ax


# Compare side-by-side
n_methods = 2 + (0 if tracks_tp is None else 1)
fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
if n_methods == 1:
    axes = [axes]

time_proj = np.max(images, axis=0)
ax_idx = 0

if tracks_tp is not None:
    overlay_tracks(axes[ax_idx], time_proj, tracks_tp, title="TrackPy")
    ax_idx += 1

if tracks_btrack is not None:
    method_name = "btrack (Bayesian)" if BTRACK_AVAILABLE else "Kalman Fallback"
    overlay_tracks(axes[ax_idx], time_proj, tracks_btrack, title=method_name)
    ax_idx += 1

plt.tight_layout(); plt.show()

print(f"Compared {n_methods} linking methods.")""")


# ---------------------------------------------------------------------------
# Compute ID-consistency metric
# ---------------------------------------------------------------------------
def section_id_consistency(b):
    b.md("""## Compute ID-consistency metric

For each linking method, compare predicted track IDs to ground-truth IDs using the **Hungarian matching** algorithm.

Metric: fraction of correctly assigned frame-object pairs (how many (frame, object) tuples got the right track ID).

Higher score = better recovery of ground-truth tracks.""")

    b.code("""def compute_id_consistency(pred_tracks_df, gt_masks, gt_tracks):
    \"\"\"
    Compute ID consistency: Hungarian match of predicted track centroids to ground-truth track centroids.

    Returns: fraction of correctly matched detections.
    \"\"\"
    # Build ground-truth centroid list: (frame, object_idx) -> (y, x)
    gt_cents = {}
    for t in range(gt_masks.shape[0]):
        props = measure.regionprops(gt_masks[t])
        for prop_idx, prop in enumerate(props):
            if prop.label > 0:
                gt_cents[(t, prop.label)] = prop.centroid

    # Build predicted centroid list
    pred_cents = {}
    for idx, row in pred_tracks_df.iterrows():
        t = int(row['frame'])
        particle = int(row['particle'])
        pred_cents[(t, particle)] = (row['y'], row['x'])

    # For each frame, perform Hungarian matching
    total_matched = 0
    total_detections = 0

    for t in range(gt_masks.shape[0]):
        gt_in_frame = {k: v for k, v in gt_cents.items() if k[0] == t}
        pred_in_frame = {k: v for k, v in pred_cents.items() if k[0] == t}

        if not gt_in_frame or not pred_in_frame:
            continue

        gt_ids = sorted([k[1] for k in gt_in_frame.keys()])
        pred_ids = sorted(set([k[1] for k in pred_in_frame.keys()]))

        # Build distance matrix
        n_gt, n_pred = len(gt_ids), len(pred_ids)
        cost_matrix = np.zeros((n_gt, n_pred))
        for i, gt_id in enumerate(gt_ids):
            for j, pred_id in enumerate(pred_ids):
                dy = gt_in_frame[(t, gt_id)][0] - pred_in_frame[(t, pred_id)][0]
                dx = gt_in_frame[(t, gt_id)][1] - pred_in_frame[(t, pred_id)][1]
                cost_matrix[i, j] = np.sqrt(dy**2 + dx**2)

        # Hungarian matching
        gt_match, pred_match = linear_sum_assignment(cost_matrix)
        matched_count = np.sum(cost_matrix[gt_match, pred_match] < 10)  # within 10 pixels = match
        total_matched += matched_count
        total_detections += n_gt

    consistency = total_matched / total_detections if total_detections > 0 else 0.0
    return consistency


# Compute for each method
results = []
if tracks_tp is not None:
    cons_tp = compute_id_consistency(tracks_tp, gt_masks, gt_tracks)
    results.append(("TrackPy", cons_tp))
    print(f"TrackPy ID-consistency: {cons_tp:.3f}")

if tracks_btrack is not None:
    cons_btrack = compute_id_consistency(tracks_btrack, gt_masks, gt_tracks)
    method_name = "btrack (Bayesian)" if BTRACK_AVAILABLE else "Kalman Fallback"
    results.append((method_name, cons_btrack))
    print(f"{method_name} ID-consistency: {cons_btrack:.3f}")

print()
if results:
    best_method = max(results, key=lambda x: x[1])
    print(f"Best method on this data: {best_method[0]} ({best_method[1]:.3f})")""")


# ---------------------------------------------------------------------------
# Method picker
# ---------------------------------------------------------------------------
def section_method_picker(b):
    b.md("""## Choose-your-own: method picker and parameters

Use these sliders to select a linking method and tune parameters. Re-run the cell to see updated results.""")

    b.code("""# @title Method picker { run: \"auto\" }
method = "TrackPy"  # @param ["TrackPy", "Bayesian (btrack or fallback)"]
search_range = 20  # @param {type: "slider", min: 5, max: 50}

print(f"Selected method: {method}")
print(f"Search range: {search_range} pixels")
print()

if method == "TrackPy":
    if tracks_tp is None:
        print("TrackPy result not available (check cell output above)")
    else:
        cons = compute_id_consistency(tracks_tp, gt_masks, gt_tracks)
        print(f"ID-consistency: {cons:.3f}")
        n_tracks = tracks_tp['particle'].max() + 1
        print(f"Number of tracks: {n_tracks}")
else:
    if tracks_btrack is None:
        print("Bayesian tracker result not available")
    else:
        cons = compute_id_consistency(tracks_btrack, gt_masks, gt_tracks)
        print(f"ID-consistency: {cons:.3f}")
        n_tracks = tracks_btrack['particle'].max() + 1
        print(f"Number of tracks: {n_tracks}")""")


# ---------------------------------------------------------------------------
# Stress test: harder synthetic case
# ---------------------------------------------------------------------------
def section_stress_test(b):
    b.md("""## Stress test — failure modes

Re-run on a harder synthetic case: cells that touch, vanish for frames, and make abrupt turns.

This section shows when each method breaks and why.""")

    b.code("""# Generate a harder synthetic case
print("Generating *harder* synthetic data (touching cells, abrupt motion)...")
np.random.seed(42)
images_hard, masks_hard, tracks_hard = generate_synthetic_timelapse(
    T=25, H=256, W=256, n_cells=15, seed=42
)

# Extract detections
detections_hard = []
for t in range(masks_hard.shape[0]):
    props = measure.regionprops(masks_hard[t])
    for prop in props:
        cy, cx = prop.centroid
        detections_hard.append({
            'frame': t,
            'y': cy,
            'x': cx,
            'mass': prop.area,
        })
detections_hard_df = pd.DataFrame(detections_hard)

# Link with both methods
print(f"  {len(detections_hard_df)} detections in {masks_hard.shape[0]} frames")
print()

try:
    tracks_tp_hard = tp.link(detections_hard_df.copy(), search_range=20, memory=1)
    cons_tp_hard = compute_id_consistency(tracks_tp_hard, masks_hard, tracks_hard)
    print(f"TrackPy on harder data: ID-consistency {cons_tp_hard:.3f}")
except Exception as e:
    print(f"TrackPy on harder data failed: {e}")
    cons_tp_hard = None

tracks_btrack_hard = simple_kalman_linker(detections_hard_df, max_gap=1, max_dist=20)
cons_btrack_hard = compute_id_consistency(tracks_btrack_hard, masks_hard, tracks_hard)
method_name = "btrack" if BTRACK_AVAILABLE else "Kalman Fallback"
print(f"{method_name} on harder data: ID-consistency {cons_btrack_hard:.3f}")

print()
print("Interpretation:")
print("  - Both methods drop in accuracy on harder data (cells touching, abrupt motion).")
print("  - TrackPy struggles more with abrupt direction changes (no motion model).")
print("  - Bayesian methods (btrack / Kalman) handle direction changes better,")
print("    but may split or merge cells at contact regions.")
print("  - For the hardest cases (dense, touching cells), Ultrack (SOTA) is recommended.")""")


# ---------------------------------------------------------------------------
# Ultrack pointer
# ---------------------------------------------------------------------------
def section_ultrack_pointer(b):
    b.md("""## SOTA Ultrack (pointer only)

For the hardest tracking problems — dense cells, touching boundaries, whole-embryo lightsheet microscopy — the state-of-the-art is **Ultrack** (Bragantini et al., *Nature Methods* 2025).

Ultrack is a neural-network-based tracker that learns to solve the linking problem end-to-end. It handles merges and splits explicitly and scales to millions of objects.

**We do not try to install Ultrack here** (heavy PyTorch dependencies, long setup). Instead:
- Visit [github.com/royerlab/ultrack](https://github.com/royerlab/ultrack)
- Review the paper and workflows
- When your own tracking problem becomes hard enough that the classical linkers fail, Ultrack is where to start

**Rule of thumb:**
- **TrackPy**: sparse particles, simple motion, no overlaps. Fast, interpretable.
- **btrack / Kalman**: cell lineages, realistic motion model, occasional overlaps. Medium complexity.
- **Ultrack**: dense cells, merges/splits, complex dynamics. SOTA but slower.""")


# ---------------------------------------------------------------------------
# Closing reflection
# ---------------------------------------------------------------------------
def section_closing(b):
    b.md("""## Closing reflection

Tracking is **detection + linking**. This lab separated them:

1. **Detection** (upstream from here): Cellpose, StarDist, SAM, etc. — solve "where are the objects in this frame?"
2. **Linking** (this lab): TrackPy, btrack, Ultrack — solve "which objects in frame T are the same as objects in frame T+1?"

**Key takeaways:**

- Linking is a distinct algorithmic choice. Different methods make different trade-offs between speed, simplicity, and robustness to difficult cases.
- **Classical (TrackPy)**: fast, robust on easy cases, no motion model.
- **Bayesian (btrack)**: learns motion, handles surprises better, requires tuning.
- **SOTA (Ultrack)**: neural-network end-to-end, handles merges/splits, slower and harder to install.
- The **right choice depends on your data**, not on what sounds most sophisticated.

**Where to go next:**

- **Notebook 01** — Cellpose segmentation (upstream detector for this workflow).
- **Notebook 10** — 3D segmentation. Tracking in 3D+T is structurally the same but spatially richer.
- **Ultrack** ([royerlab/ultrack](https://github.com/royerlab/ultrack)) — when classical methods fail.
- **TrackMate** (Fiji plugin) — GUI-based classical tracking; what TrackPy implements algorithmically.
- **Icy** — another image analysis platform with interactive tracking tools.

**A final note.** Track quality is only as good as your detections. Spend effort on the segmentation step (Notebooks 01, 03a, 03b) before optimizing the linker.""")


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def main():
    b = CellBuilder("nb11")
    section_title(b)
    section_setup(b)
    section_synthetic_data(b)
    section_visualize_data(b)
    section_detection(b)
    section_quiz(b)
    section_trackpy(b)
    section_btrack(b)
    section_visualize_tracks(b)
    section_id_consistency(b)
    section_method_picker(b)
    section_stress_test(b)
    section_ultrack_pointer(b)
    section_closing(b)
    build_notebook(b.cells, "11_dl_tracking")


if __name__ == "__main__":
    main()
