"""
Build script for the ramp-up evening notebooks.

Generates two Jupyter notebooks for the optional T0 session:
  python_basics.ipynb     — Python and numpy refresher for image work
  image_basics.ipynb      — image-data fundamentals (shape, dtype, formats, display)

Run:
    python build_rampup_notebooks.py
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent


class CellBuilder:
    def __init__(self, prefix):
        self.prefix = prefix
        self.idx = 0
        self.cells = []

    def md(self, source):
        self.cells.append({
            "cell_type": "markdown",
            "id": f"{self.prefix}-md-{self.idx:03d}",
            "metadata": {},
            "source": source.lstrip("\n"),
        })
        self.idx += 1
        return self

    def code(self, source):
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


def _colab_badge(builder, repo_path):
    """Prepend an Open-in-Colab shield cell to a CellBuilder. repo_path is the
    notebook's path relative to the repository root, e.g.
    'ramp-up/notebooks/python_basics.ipynb'."""
    url = (
        "https://colab.research.google.com/github/"
        "microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/"
        f"{repo_path}"
    )
    builder.md(
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})\n\n"
        "*Click the badge to open this notebook in Google Colab and run the cells interactively.*"
    )


def build_notebook(cells, name):
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
# Ramp-up Notebook 1 — Python basics for image work
# ---------------------------------------------------------------------------
def build_python_basics():
    b = CellBuilder("py")
    _colab_badge(b, "ramp-up/notebooks/python_basics.ipynb")
    b.md("""# Ramp-up — Python and numpy basics for image work

**Goal.** Refresh the minimum Python the workshop assumes:
- variables, lists, dictionaries
- numpy arrays as multi-dimensional grids of numbers
- array indexing and slicing
- function definitions and imports
- the "predict before you run" habit

**Time.** ~45 minutes. Hands-on the whole way through.

If most of this feels familiar, you're ready for the workshop. If it feels unfamiliar, you'll be much better off having done this notebook before tomorrow morning.""")

    b.md("## Variables and types")
    b.code("""# Numbers
n_cells = 47
mean_intensity = 0.42
print(type(n_cells), type(mean_intensity))""")

    b.code("""# Strings
sample_name = "C2-WT-rep1"
print(sample_name, type(sample_name))""")

    b.code("""# Booleans
is_processed = True
has_artifact = False
print(is_processed, has_artifact)""")

    b.md("## Lists, indexing, slicing")
    b.code("""# Lists hold ordered collections
intensities = [0.12, 0.45, 0.81, 0.33, 0.67]
print("Length :", len(intensities))
print("First  :", intensities[0])
print("Last   :", intensities[-1])
print("Middle three :", intensities[1:4])""")

    b.md("""**Predict** — what will the next cell print?""")
    b.code("""# Reverse slicing
print(intensities[::-1])
# First three only
print(intensities[:3])
# Skip every other element
print(intensities[::2])""")

    b.md("## Dictionaries")
    b.code("""# Dictionaries hold key-value pairs
sample = {
    "name": "WT-rep1",
    "n_cells": 47,
    "mean_intensity": 0.42,
    "channels": ["DAPI", "GFP", "RFP"],
}
print(sample["name"])
print(sample["channels"])
print("Keys:", list(sample.keys()))""")

    b.md("## numpy: arrays as multi-dimensional grids")
    b.code("""import numpy as np

# A 2D array (think of a grayscale image)
img = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
])
print("Shape :", img.shape)
print("Dtype :", img.dtype)
print("Sum   :", img.sum())
print("Mean  :", img.mean())""")

    b.md("""**Predict** — what shape will the next array have? What will its mean be?""")
    b.code("""# A larger random array (think of a noisy image)
rng = np.random.default_rng(42)
big = rng.integers(0, 256, size=(8, 8))  # 8x8 8-bit-like values
print("Shape :", big.shape)
print("Dtype :", big.dtype)
print("Mean  :", big.mean())
print(big)""")

    b.md("## Array indexing for image work")
    b.code("""# Get a single pixel
print("Pixel at (0,0):", big[0, 0])

# Get a row
print("First row    :", big[0])

# Get a column
print("First column :", big[:, 0])

# Get a subregion (a 3x3 block from the upper-left)
sub = big[:3, :3]
print("Subregion   :")
print(sub)
print("Subregion mean:", sub.mean())""")

    b.md("## Element-wise operations")
    b.code("""# Math on arrays applies element-wise
doubled = big * 2
print("Original mean:", big.mean())
print("Doubled mean :", doubled.mean())

# Boolean indexing — find pixels brighter than the mean
bright = big > big.mean()
print("Brightness mask:")
print(bright.astype(int))
print("Number of bright pixels:", bright.sum())""")

    b.md("## Functions")
    b.code("""def normalize(arr):
    \"\"\"Scale array to [0, 1] using percentile clipping.\"\"\"
    p1, p99 = np.percentile(arr, [1, 99])
    clipped = np.clip(arr, p1, p99)
    return (clipped - p1) / (p99 - p1)

normalized = normalize(big)
print("Normalized min:", normalized.min())
print("Normalized max:", normalized.max())
print("Normalized mean:", normalized.mean().round(3))""")

    b.md("## Imports — how Python finds the tools you'll use")
    b.code("""# The pattern you'll see repeatedly tomorrow
import numpy as np                    # numerical computing
import matplotlib.pyplot as plt        # plotting
from skimage import io as skio         # scientific image I/O

print("numpy version     :", np.__version__)
print("Matplotlib loaded :", plt.__name__)
print("scikit-image loaded:", skio.__name__)""")

    b.md("## Visualizing a small array")
    b.code("""fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(big, cmap='viridis')
ax.set_title("8x8 random 'image'")
plt.colorbar(ax.images[0], ax=ax)
plt.tight_layout(); plt.show()""")

    b.md("""## Closing reflection

You now have the Python and numpy patterns the workshop assumes:

- variables, lists, dictionaries
- numpy arrays, indexing, slicing, element-wise math
- function definitions
- imports

Tomorrow's workshop adds image-specific concepts on top of these. The setup self-check (`notebooks/00_setup_self_check.ipynb`) at the workshop level gives you one more chance to confirm everything works on Colab before the workshop day.""")

    build_notebook(b.cells, "python_basics")


# ---------------------------------------------------------------------------
# Ramp-up Notebook 2 — Image data fundamentals
# ---------------------------------------------------------------------------
def build_image_basics():
    b = CellBuilder("img")
    _colab_badge(b, "ramp-up/notebooks/image_basics.ipynb")
    b.md("""# Ramp-up — Image data fundamentals

**Goal.** Build the minimum image-data fluency the workshop assumes:
- what a digital image is at the data-structure level
- shape, dtype, channels, and dimensions
- common file formats in microscopy
- displaying images with appropriate contrast
- the relationship between pixel coordinates and physical units

**Time.** ~45 minutes. Hands-on the whole way through.""")

    b.md("## A digital image is just an array of numbers")
    b.code("""%pip install --quiet numpy matplotlib scikit-image tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio""")

    b.md("## Make and display a simple image")
    b.code("""# A 2D array — grayscale image
img = np.zeros((100, 100), dtype=np.uint8)
img[20:80, 20:80] = 200   # bright square in the middle
img[40:60, 40:60] = 100   # darker square inside

print("Shape :", img.shape)
print("Dtype :", img.dtype)
print("Min   :", img.min(), "  Max:", img.max())

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(img, cmap='gray', vmin=0, vmax=255)
ax.set_title(f"Grayscale {img.shape}")
plt.tight_layout(); plt.show()""")

    b.md("## Multi-channel images: shape conventions")
    b.code("""# A 3-channel RGB-like image: shape (H, W, C)
rgb = np.zeros((100, 100, 3), dtype=np.uint8)
rgb[..., 0] = 200    # full red
rgb[40:60, 40:60, 1] = 200    # green square in the middle

print("Shape :", rgb.shape)
print("Dtype :", rgb.dtype)
print("Per-channel means:", rgb.mean(axis=(0,1)))

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(rgb)
ax.set_title("RGB image (H, W, 3)")
plt.tight_layout(); plt.show()""")

    b.md("""**Convention warning.** Microscopy data sometimes uses (C, H, W) instead of (H, W, C). Tools differ. Always check the shape and use the cell below to detect channels-first vs channels-last format.""")

    b.code("""def channels_axis(img):
    \"\"\"Heuristic: the smallest axis (size <= 4) is typically the channel axis.\"\"\"
    if img.ndim < 3: return None
    sizes = list(img.shape)
    smallest = min(sizes)
    if smallest <= 4:
        return sizes.index(smallest)
    return None

print("Channels-last RGB (H, W, C):", rgb.shape, "→ channels axis:", channels_axis(rgb))

channels_first = np.transpose(rgb, (2, 0, 1))  # (C, H, W)
print("Channels-first RGB (C, H, W):", channels_first.shape, "→ channels axis:", channels_axis(channels_first))""")

    b.md("## Bit depth")
    b.code("""# 8-bit: values 0-255
img_8bit = np.linspace(0, 255, 256).astype(np.uint8).reshape(16, 16)
print("8-bit  range:", img_8bit.min(), img_8bit.max())

# 16-bit: values 0-65535 (more dynamic range, common in microscopy)
img_16bit = np.linspace(0, 65535, 256).astype(np.uint16).reshape(16, 16)
print("16-bit range:", img_16bit.min(), img_16bit.max())

# Float: values typically 0.0-1.0 (after normalization)
img_float = np.linspace(0, 1, 256).reshape(16, 16)
print("Float range:", img_float.min().round(3), img_float.max().round(3))""")

    b.md("""**Why bit depth matters.** Most microscopes capture 12-bit or 16-bit raw data. Many tools convert to 8-bit for display, which discards 4-8 bits of dynamic range. For *quantitative* analysis you usually want to keep the original bit depth as long as possible.""")

    b.md("## Displaying with appropriate contrast")
    b.code("""# A 16-bit image where most pixels are dark, with a few bright spots
img16 = np.zeros((100, 100), dtype=np.uint16)
img16[40:60, 40:60] = 50000   # bright square
img16 = img16 + np.random.randint(0, 200, img16.shape).astype(np.uint16)

# Default display: contrast is too low because matplotlib auto-scales the bright pixels
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(img16, cmap='gray'); axes[0].set_title("Default (no vmin/vmax)")

# Better: percentile-based contrast
p1, p99 = np.percentile(img16, [1, 99])
axes[1].imshow(img16, cmap='gray', vmin=p1, vmax=p99); axes[1].set_title(f"vmin={int(p1)}, vmax={int(p99)}")

# Best for quantitative: full data range, with a colorbar
axes[2].imshow(img16, cmap='viridis')
plt.colorbar(axes[2].images[0], ax=axes[2])
axes[2].set_title("With colorbar (quantitative)")
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()""")

    b.md("## File formats — TIFF and beyond")
    b.code("""import tifffile

# Save a TIFF and read it back
tifffile.imwrite("test.tif", img)
loaded = tifffile.imread("test.tif")
print("Saved and reloaded TIFF:", loaded.shape, loaded.dtype, "→ matches:", np.array_equal(img, loaded))

# OME-TIFF, NDPI, SVS, CZI, LIF, ND2, etc. are all supported by Bio-Formats
# and tools like AICSImageIO. For the workshop, TIFF and PNG cover most cases.""")

    b.md("## Calibration: pixels and physical units")
    b.code("""# Microscopy images have a pixel size — usually in micrometers per pixel
pixel_size_um = 0.16    # e.g., 0.16 µm/pixel for a 60x objective at typical settings
image_shape = (1024, 1024)

field_of_view_um = (image_shape[0] * pixel_size_um, image_shape[1] * pixel_size_um)
print(f"Image  : {image_shape[0]} × {image_shape[1]} pixels")
print(f"FoV    : {field_of_view_um[0]:.1f} × {field_of_view_um[1]:.1f} µm")
print(f"Area   : {field_of_view_um[0] * field_of_view_um[1]:.1f} µm²")""")

    b.md("""**Key habit.** When you analyze image data quantitatively, the answer almost always needs to come out in physical units, not pixel units. The conversion is straightforward but easy to forget. Document the pixel size with every dataset.""")

    b.md("""## Closing reflection

You now have the image-data patterns the workshop assumes:

- digital images are arrays of numbers — `(H, W)`, `(H, W, C)`, or `(C, H, W)`
- bit depth matters; preserve it for quantitative analysis
- contrast adjustment is for display; original data is what you compute on
- pixel coordinates and physical units are not the same — keep track

Combined with the Python basics notebook, you have what tomorrow's workshop assumes. See you in the morning.""")

    build_notebook(b.cells, "image_basics")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_python_basics()
    build_image_basics()
    print("\nRamp-up notebooks built.")
