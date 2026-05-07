# scripts/

Helpers for local development and verification of the workshop repo.

| Script | Purpose |
|---|---|
| `dry_run_local.sh` | Build the Jupyter Book site locally and serve at `http://localhost:8000`. Use this to preview what GitHub Pages will look like. Does **not** execute notebook code; renders source. |
| `dry_run_check.py` | Standalone notebook linter. Stdlib-only, runs without any pip-installed deps. Validates JSON, `nbformat==4`, Colab-badge presence, OWNER/REPO placeholder count, stale outputs, nested duplicates. |
| `run_jupyter_lab.sh` | Launch Jupyter Lab pointed at the repo for interactive notebook testing. Default profile is *minimal* (install only `jupyterlab` + a couple support packages); `--prefetch` installs every notebook's runtime dep upfront from `requirements_notebooks.txt`. |
| `test_notebook.sh` | Headlessly execute one notebook via `nbconvert --execute` and report pass/fail. Tight iteration loop without launching Lab. |
| `requirements_notebooks.txt` | Union of all notebook runtime deps. Used by `run_jupyter_lab.sh --prefetch` and as the reference list when a single notebook needs its deps installed manually. |

## Typical workflows

### "I want to preview the Jupyter Book site"
Goal: see what the rendered docs site will look like once published.
```bash
./scripts/dry_run_local.sh
# Then open http://localhost:8000
```
Doesn't execute any notebook code; renders source. Fast (< 1 min).

### "I want to test one notebook quickly"
Goal: confirm a notebook runs end-to-end after I edited it.
```bash
./scripts/run_jupyter_lab.sh --prefetch --no-launch  # one-time, installs all deps
./scripts/test_notebook.sh notebooks/12_deconvolution.ipynb
```
First run takes ~5 GB of installs + a few minutes per notebook; subsequent test runs are just notebook execution time.

### "I want to interactively debug a notebook"
Goal: step through cells, inspect intermediate state.
```bash
./scripts/run_jupyter_lab.sh --prefetch
# Or, to match Colab behavior (each notebook installs its own deps on first run):
./scripts/run_jupyter_lab.sh
```
Lab opens in your browser. Click any notebook in the file tree to start.

### "I'm just sanity-checking the repo state"
Goal: confirm structure, no broken JSON, no stray placeholder, etc.
```bash
./scripts/dry_run_local.sh --check
```
Pre-flight + lint, no install or build. Exits in seconds.

## Convention: verify before installing

All four scripts follow the same principle — **never install anything that's already present**. Each runs a `get_pkg_version()` check via `importlib.metadata.version()` and skips packages that resolve. `--force-reinstall` overrides on every script.

This keeps your environment clean and avoids accidental upgrades. The pinned-and-loose dependency style in `requirements_notebooks.txt` is intentional too — most pins are loose so pip can use what's already in the env, with a few hard pins (e.g., `cellpose<4`, `jupyter-book<2`) where a major-version bump would break the notebooks.
