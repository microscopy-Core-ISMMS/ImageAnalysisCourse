"""Build script for Notebook T0 — Data Sources Reference.

T0 is the reference notebook attendees consult when they want to use the
workshop's notebooks on data other than the workshop defaults. It contains
copy-paste blocks for: workshop default, local files, Google Drive mount,
arbitrary public URL, HuggingFace Datasets, institutional storage notes,
and a guide to running locally vs Colab.

Run:
    python3 build_notebook_T0.py
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
# Sections
# ---------------------------------------------------------------------------

def section_title(b):
    b.md("""<!-- colab-badge -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/notebooks/00_data_sources.ipynb)

*Click the badge to open this notebook in Google Colab.*""")

    b.md("""# Notebook T0 — Data Sources Reference

**Purpose.** This is a reference notebook, not a tutorial. Use it when you want to run any of the workshop notebooks on **data other than the defaults**. Each section below is a self-contained, copy-paste block for one way to load images.

**How the workshop notebooks consume data.** Every T1 lab notebook (NB01, NB02, NB03a/b, NB06, NB07, NB09, NB12, NB13, NB14, NB16) starts with a "Choose data source" decision block. By default that block fetches from the MABC-hosted samples; if MABC is unavailable, it falls through to a canonical published dataset (BBBC, GigaDB, etc.); if that fails, it falls through to the synthetic generator.

The fourth option in that block is **"My own data → see T0"** — that's what brings you here. The blocks below show you how to load your own data into a Python list called `real_imgs`. Once you've run the appropriate block, **switch back to the lab notebook** and re-run from the cell after the decision block. The lab notebook will pick up `real_imgs` from memory.

> ⚠️  Variable name contract: the lab notebooks expect `real_imgs` to be a list (or array) of NumPy arrays. Each notebook also wants specific working variable names bound from `real_imgs` — e.g. NB01 expects `img_easy`, `img_hard`. See the **Variable name reference** at the end of this notebook for the per-notebook contract.

---""")


def section_overview(b):
    b.md("""## How to use this notebook

1. **Pick the block** that matches where your data lives (local files, Drive, etc.).
2. **Copy the code** into the lab notebook you want to run, or run it here and switch back.
3. **Bind the working variables** the lab notebook expects (see the reference table at the bottom).
4. **Re-run the lab notebook** from the cell *after* its decision block.

The blocks are intentionally minimal — short, well-commented, paste-ready. None of them depend on each other.""")


def section_default(b):
    b.md("""## Block A — Workshop default (MABC samples + canonical fallback)

This is what each lab notebook does by default. Shown here for completeness — you don't normally run this block manually; just leave the lab notebook's decision dropdown set to `"MABC hosted"`.""")

    b.code('''import os, urllib.request, urllib.error, tempfile
import numpy as np

NB_ID = "01_cellpose_segmentation"   # change to your target NB id
MABC_URL = f"https://microscopy-core-ismms.github.io/ImageAnalysisCourse/data/mabc/{NB_ID}.npz"

cache = os.path.join(tempfile.gettempdir(), os.path.basename(MABC_URL))
if not os.path.exists(cache):
    urllib.request.urlretrieve(MABC_URL, cache)

data = np.load(cache, allow_pickle=True)
real_imgs = list(data["images"])
real_filenames = list(data["filenames"]) if "filenames" in data.files else None
metadata = data["metadata"].item() if "metadata" in data.files else {}

print(f"Loaded {len(real_imgs)} images from MABC sample {NB_ID}.")
print(f"  shape per image: {real_imgs[0].shape}, dtype: {real_imgs[0].dtype}")
print(f"  source: {metadata.get('source', '(unknown)')}")
''')


def section_local_files(b):
    b.md("""## Block B — Load your own local files

Use this when your images are on the same machine as the notebook. On Colab, "local" means the runtime VM — not your laptop. To work on your laptop's files in Colab, mount Drive (Block C) or upload them to the runtime first (`from google.colab import files; files.upload()`).""")

    b.code('''from pathlib import Path
import numpy as np

# Edit these paths.
DATA_DIR = Path("/content/my_images")          # folder with your TIFF/PNG files
PATTERN = "*.tif"                                # glob pattern
N_LIMIT = 8                                      # how many to load

# Try tifffile first (handles multi-page, 16-bit), fall back to PIL.
try:
    import tifffile
    _read = lambda p: tifffile.imread(str(p))
except ImportError:
    from PIL import Image
    _read = lambda p: np.array(Image.open(p))

paths = sorted(DATA_DIR.glob(PATTERN))[:N_LIMIT]
real_imgs = [_read(p) for p in paths]
real_filenames = [p.name for p in paths]

print(f"Loaded {len(real_imgs)} files from {DATA_DIR}.")
for fn, im in zip(real_filenames, real_imgs):
    print(f"  {fn}: shape={im.shape}, dtype={im.dtype}")
''')

    b.md("""**TIFF gotchas.** Multi-page TIFF (Z-stacks) returns a 3D array `(Z, H, W)`. Multi-channel TIFF can return `(C, H, W)` or `(H, W, C)` depending on how it was saved. The lab notebooks generally handle both, but if a downstream cell errors on shape, do `print(real_imgs[0].shape)` and slice/transpose to `(H, W)` for grayscale or `(H, W, 3)` for RGB.

**Other formats** — for proprietary microscopy formats (`.lif`, `.nd2`, `.czi`, `.lsm`):

```python
%pip install bioio bioio-lif bioio-nd2  # one of: -lif, -nd2, -czi, -lsm
from bioio import BioImage
img = BioImage(\"/content/my_image.lif\")
real_imgs = [img.get_image_data(\"YX\", T=0, C=0, Z=z) for z in range(img.dims.Z)]
```""")


def section_drive(b):
    b.md("""## Block C — Mount Google Drive (Colab only)

Use this when your images live in your Google Drive. Authentication runs once per Colab runtime; subsequent cells access files at paths like `/content/drive/MyDrive/...`.""")

    b.code('''# Colab-only: mount Google Drive at /content/drive/
from google.colab import drive
drive.mount("/content/drive")

# Now your Drive files are at /content/drive/MyDrive/<your-folder>/...
# Combine with Block B above to load:
from pathlib import Path
DATA_DIR = Path("/content/drive/MyDrive/workshop_images")
PATTERN = "*.tif"
# ... continue with Block B's loading code
''')

    b.md("""**First-time auth.** The cell above pops up a Google sign-in flow. Sign in with the same Google account that owns the Drive folder. Authorization persists for the runtime session (~12 hours).

**Path tip.** If your folder is in a Shared Drive (org account), the path is `/content/drive/Shareddrives/<DriveName>/...` — note the lowercase `d` in `drive` and the capital `Shared`.""")


def section_public_url(b):
    b.md("""## Block D — Pull from any public URL

Use this when your data lives on a public web URL (your lab's website, Zenodo, Figshare, GitHub raw, etc.) — anything reachable via HTTPS without authentication.""")

    b.code('''import os, urllib.request, tempfile, zipfile
from pathlib import Path
import numpy as np

# Edit these.
URL = "https://example.org/path/to/your-data.zip"
N_LIMIT = 8

cache = os.path.join(tempfile.gettempdir(), os.path.basename(URL))
if not os.path.exists(cache):
    print(f"Downloading {URL}...")
    urllib.request.urlretrieve(URL, cache)

# If it's a zip, extract; otherwise just load.
if cache.endswith(".zip"):
    extract_dir = cache + "_extracted"
    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(cache) as zf:
            zf.extractall(extract_dir)
    img_paths = sorted(Path(extract_dir).rglob("*.tif"))[:N_LIMIT]
else:
    img_paths = [Path(cache)]

try:
    import tifffile
    _read = lambda p: tifffile.imread(str(p))
except ImportError:
    from PIL import Image
    _read = lambda p: np.array(Image.open(p))

real_imgs = [_read(p) for p in img_paths]
real_filenames = [p.name for p in img_paths]
print(f"Loaded {len(real_imgs)} images from {URL}.")
''')

    b.md("""**Citation hygiene.** If the URL points at a published dataset (BBBC, GigaDB, BIA, IDR, etc.), capture the dataset name + DOI/URL in your notebook so anyone reading it can trace the data. The workshop's `datasets_audit.md` covers the most common canonical sources.""")


def section_huggingface(b):
    b.md("""## Block E — HuggingFace Datasets

Use this when your dataset is hosted on the [HuggingFace Hub](https://huggingface.co/datasets). Many recent bioimage benchmarks (STimage-1K4M, BIN-1, etc.) live there. HuggingFace caches downloads in `~/.cache/huggingface/` so subsequent runs are fast.""")

    b.code('''%pip install --quiet datasets
from datasets import load_dataset
import numpy as np

# Edit this.
DATASET_ID = "your-org/your-dataset"   # e.g., "polinaeterna/cifar10"
SPLIT = "train"
N_LIMIT = 8

ds = load_dataset(DATASET_ID, split=f"{SPLIT}[:{N_LIMIT}]")
# HF Datasets returns dicts; the image key is usually "image" or "img"
image_key = next((k for k in ds.column_names if k.lower() in ("image", "img", "pixel_values")), None)
if image_key is None:
    raise KeyError(f"No image column found. Available columns: {ds.column_names}")

real_imgs = [np.array(row[image_key]) for row in ds]
real_filenames = [f"{DATASET_ID}#{i}" for i in range(len(real_imgs))]
print(f"Loaded {len(real_imgs)} images from {DATASET_ID}.")
''')

    b.md("""**Gated datasets.** Some HF datasets require accepting a license on the website first, plus a `huggingface-cli login` token. The `load_dataset` call will raise a clear error with a link if so.""")


def section_institutional(b):
    b.md("""## Block F — Mt Sinai Box / OneDrive (institutional storage)

**Short version: you can't fetch directly from Mt Sinai Box or institutional OneDrive in a Colab notebook**, because both require SSO authentication that doesn't work cleanly in a Jupyter cell.

**Workaround.** The cleanest path is to mirror the specific files you need into a personal Google Drive folder, then use Block C to mount that. From the Mt Sinai end:

1. Open the file in Box / OneDrive on your Mac.
2. Right-click → Download (or open in the desktop client and copy out).
3. Drop the file into your `MyDrive/workshop_images/` folder.
4. In Colab, mount Drive (Block C above) and load.

If you need to pull a Mt Sinai Box file at scale (>1 GB or many files), the Box API is the right answer — but configuring Box's OAuth flow inside a workshop slot is friction. **Recommend doing the data prep step on your laptop, not in Colab.**

For purely public Mt Sinai resources (anything published on a Mt Sinai-affiliated GitHub or website), Block D (public URL) works fine.""")


def section_run_locally(b):
    b.md("""## Block G — Run notebooks locally instead of on Colab

Colab is the workshop default — zero-install, free GPU, identical environment for every attendee. But you can also run the lab notebooks on your own machine.

### Why local?
- **Bigger data.** Colab's free tier disconnects after ~12 hours and has limited disk. For multi-GB datasets, local is more practical.
- **Faster iteration.** No re-installing packages every fresh runtime.
- **Privacy.** If your data isn't public, mounting Drive may be inappropriate.

### Setup

```bash
git clone https://github.com/microscopy-Core-ISMMS/ImageAnalysisCourse.git
cd ImageAnalysisCourse

# Conda is recommended:
conda env create -f env/environment.yml
conda activate ai-microscopy-2026

# Or pip (less reliable for some packages — micro-sam, cellpose can be tricky):
pip install -r env/requirements_notebooks.txt

# Launch JupyterLab and open the notebook you want:
jupyter lab notebooks/01_cellpose_segmentation.ipynb
```

### GPU notes
- **CUDA on Linux/Windows.** Most labs benefit from a GPU. Cellpose, SAM, and the larger U-Nets in NB07/12 will run on CPU but slowly (5–30× longer).
- **Apple Silicon Mac.** PyTorch's `mps` backend works for most operations. Set `device = "mps"` and expect ~50–80% of CUDA performance for the small models in this workshop.
- **No GPU.** Everything still runs; expect total runtime around 60–90 min for the labs that train models.

### Where to put data
Put your input images anywhere you like and edit the `DATA_DIR` constant in the lab notebook's loading cell (or in Block B above) to point at it. The lab notebooks don't enforce a specific layout — they just consume `real_imgs` as a list of NumPy arrays.""")


def section_variable_reference(b):
    b.md("""## Variable name reference

After you load your data into `real_imgs` (and optionally `real_filenames`, `real_metadata`), each lab notebook expects specific working variables to be bound. The lab notebook's decision block does this binding automatically when MABC / canonical succeeds. If you load data manually here, you'll need to do the binding yourself.

| NB | Working variables expected | Suggested binding |
|---|---|---|
| **NB01** Cellpose pretrained | `img_easy`, `img_hard`, `img_easy_synth`, `img_hard_synth`, `img` | `img_easy = real_imgs[0]; img_hard = real_imgs[1]; img_easy_synth = real_imgs[0]; img_hard_synth = real_imgs[1]; img = real_imgs[0]` |
| **NB02** Validation/QC | `gt_easy`, `gt_hard` (binary masks) | `gt_easy = (real_imgs[0] > 0).astype(np.uint8); gt_hard = (real_imgs[-1] > 0).astype(np.uint8)` |
| **NB03a** Denoising | `clean`, `noisy` | `clean = real_imgs[0].astype(float)/255; noisy = clean + 0.10*np.random.randn(*clean.shape)` |
| **NB03b** Foundation seg (SAM) | `img` | `img = real_imgs[0].astype(float)/255` |
| **NB06** Virtual staining | `X_train`, `Y_train`, `X_test`, `Y_test` | Split `real_imgs` 6/2 train/test, pair channels appropriately |
| **NB07** Super-resolution | `hr_train`, `hr_test`, `lr_train_small`, `lr_train`, `lr_test_small`, `lr_test`, `n_train`, `n_test` | Split 6/2; LR via Gaussian-blur + 0.5× downsample + bicubic-back-up |
| **NB09** Cellpose finetune | `train_images`, `train_labels`, `test_images`, `test_labels`, `img` | `train_images = real_imgs[:6]; test_images = real_imgs[6:]; train_labels/test_labels = zero masks unless you have GT` |
| **NB12** Deconvolution | `X_train_clean`, `X_train_blurred`, `X_train_blurred_noisy`, `X_test_*`, `n_train`, `n_test` | Same 6/2 split; blur via Gaussian σ≈2, noise via N(0, 0.05) |
| **NB13** Validation case study | `img`, `TEST_IMAGES` | `img = real_imgs[0]; TEST_IMAGES = {f'real_{i}': (real_imgs[i], None) for i in range(min(4, len(real_imgs)))}` |
| **NB14** Spot detection | `train_images`, `train_centers`, `test_images`, `test_centers` | Split 6/2; centers = `[None] * N` if no GT spot coords |
| **NB16** WSI → transcriptomics | `he_tiles` (list of RGB arrays) | `he_tiles = [np.asarray(img) for img in real_imgs]` |

The decision-block code in each lab notebook embeds the right binding logic, so most attendees never touch this table. It's here for the case where you load data manually and need to know what to assign.""")


def section_resources(b):
    b.md("""## Further resources

- **Workshop dataset audit:** [`datasets_audit.md`](https://github.com/microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/datasets_audit.md) — license + format notes for every canonical source the workshop touches (BBBC, BIA, IDR, Allen Cell, BSCCM, GigaDB, CIL, Cellpose, NIST NexusLIMS).
- **MABC manifest:** [`data/MANIFEST.json`](https://github.com/microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/data/MANIFEST.json) — sha256 + size of each `data/mabc/<nb>.npz`.
- **Bake script:** [`scripts/bake_dataset_samples.py`](https://github.com/microscopy-Core-ISMMS/ImageAnalysisCourse/blob/2026-workshop/scripts/bake_dataset_samples.py) — produces the MABC npz files from raw inputs.

---

*This notebook is part of the [AI for Microscopy Image Analysis](https://microscopy-core-ismms.github.io/ImageAnalysisCourse/) workshop.*""")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main():
    b = CellBuilder("nbT0")
    section_title(b)
    section_overview(b)
    section_default(b)
    section_local_files(b)
    section_drive(b)
    section_public_url(b)
    section_huggingface(b)
    section_institutional(b)
    section_run_locally(b)
    section_variable_reference(b)
    section_resources(b)
    build_notebook(b.cells, "00_data_sources")


if __name__ == "__main__":
    main()
