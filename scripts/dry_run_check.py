#!/usr/bin/env python3
"""
Lint every notebook in the repo before jupyter-book build.

Standalone — uses only the Python standard library so it can run before
you've installed jupyter-book or any other dependency.

Checks per notebook:
  - Valid JSON
  - nbformat == 4
  - At least one cell
  - Colab badge present in cell 0
  - OWNER/REPO placeholder count (across whole notebook)
  - No stale `execution_count` set on code cells (warning, not error)
  - No `outputs` arrays embedded on code cells (warning — bloats repo)

Exit code 0 if no errors; 1 if any errors found.

Usage:
  ./dry_run_check.py              # check repo root (parent of scripts/)
  ./dry_run_check.py /path/to/repo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# ANSI colors. Honored if stdout is a tty.
def _colors_enabled() -> bool:
    return sys.stdout.isatty()


_C = {
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def c(name: str, s: str) -> str:
    return f"{_C[name]}{s}{_C['reset']}" if _colors_enabled() else s


def check_notebook(path: Path) -> tuple[list[tuple[str, str]], dict]:
    """Return (issues, info). issues is a list of (severity, message)."""
    issues: list[tuple[str, str]] = []
    info: dict = {"cells": 0, "size_kb": 0, "has_colab_badge": False,
                  "owner_repo": 0, "stale_outputs": 0, "code_with_outputs": 0}

    raw = path.read_bytes()
    info["size_kb"] = len(raw) // 1024

    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as e:
        issues.append(("ERROR", f"invalid JSON: {e}"))
        return issues, info

    info["nbformat"] = nb.get("nbformat", "?")
    if info["nbformat"] != 4:
        issues.append(("ERROR", f"nbformat {info['nbformat']} (expected 4)"))

    cells = nb.get("cells", [])
    info["cells"] = len(cells)
    if not cells:
        issues.append(("ERROR", "no cells"))
        return issues, info

    # Colab badge in cell 0?
    src0 = cells[0].get("source", "")
    if isinstance(src0, list):
        src0 = "".join(src0)
    info["has_colab_badge"] = "colab.research.google.com" in src0
    if not info["has_colab_badge"]:
        # Many lecture/support notebooks legitimately have no Colab badge
        # (the lecture_prereqs_check, for example). Note as INFO, not WARN.
        # Only warn for files in notebooks/.
        if "notebooks" in path.parts and "ramp-up" not in path.parts:
            issues.append(("WARN", "no Colab badge in cell 0"))

    # OWNER/REPO placeholder count (across whole notebook)
    owner_repo = 0
    code_with_outputs = 0
    stale = 0
    for cell in cells:
        s = cell.get("source", "")
        if isinstance(s, list):
            s = "".join(s)
        owner_repo += s.count("OWNER/REPO")
        if cell.get("cell_type") == "code":
            if cell.get("execution_count") is not None:
                stale += 1
            outs = cell.get("outputs") or []
            if outs:
                code_with_outputs += 1
    info["owner_repo"] = owner_repo
    info["stale_outputs"] = stale
    info["code_with_outputs"] = code_with_outputs

    if stale:
        issues.append(("WARN", f"{stale} code cells have execution_count set"))
    if code_with_outputs:
        issues.append(("WARN", f"{code_with_outputs} code cells have output arrays"))

    return issues, info


def is_nested_duplicate(path: Path, all_paths: set[Path]) -> bool:
    """A notebook at `<dir>/notebook.ipynb` is a nested duplicate if a sibling
    `<dir>.ipynb` exists at the parent level. Original Demo source has one of
    these (legacy build artifact, excluded from the Jupyter Book build via
    `_config.yml`). Skip it to keep the lint output tidy.
    """
    if path.name != "notebook.ipynb":
        return False
    parent = path.parent
    sibling = parent.parent / f"{parent.name}.ipynb"
    return sibling in all_paths


def main() -> int:
    if len(sys.argv) > 1:
        repo_root = Path(sys.argv[1]).resolve()
    else:
        # Default: parent of scripts/
        repo_root = Path(__file__).resolve().parent.parent

    if not (repo_root / "_config.yml").exists():
        print(c("red", f"ERROR: no _config.yml in {repo_root}; not a Jupyter Book repo"),
              file=sys.stderr)
        return 1

    all_paths = sorted(repo_root.glob("**/*.ipynb"))
    all_paths = [p for p in all_paths
                 if "_build" not in p.parts
                 and ".ipynb_checkpoints" not in p.parts
                 and "__pycache__" not in p.parts]
    all_paths_set = set(all_paths)

    skipped: list[tuple[Path, str]] = []
    paths: list[Path] = []
    for p in all_paths:
        if is_nested_duplicate(p, all_paths_set):
            skipped.append((p, "nested duplicate of sibling .ipynb (excluded from JB build)"))
        else:
            paths.append(p)

    print(c("bold", f"Linting {len(paths)} notebooks under {repo_root}"))
    if skipped:
        for sp, reason in skipped:
            rel = sp.relative_to(repo_root)
            print(c("dim", f"  (skipped: {rel} — {reason})"))
    print()

    # Header
    print(f"{'Notebook':<55} {'Cells':>5} {'KB':>5} {'Colab':>5} {'OWNER':>6} {'Issues':>10}")
    print(c("dim", "-" * 95))

    n_errors = 0
    n_warns = 0
    total_cells = 0
    total_owner_repo = 0

    for p in paths:
        rel = p.relative_to(repo_root)
        issues, info = check_notebook(p)
        errors = sum(1 for s, _ in issues if s == "ERROR")
        warns = sum(1 for s, _ in issues if s == "WARN")
        n_errors += errors
        n_warns += warns
        total_cells += info.get("cells", 0)
        total_owner_repo += info.get("owner_repo", 0)

        badge = c("green", "✓") if info.get("has_colab_badge") else c("dim", "—")
        owner_str = str(info.get("owner_repo", 0))
        if info.get("owner_repo", 0) > 0:
            owner_str = c("yellow", owner_str)

        issue_str = ""
        if errors:
            issue_str += c("red", f"{errors}E ")
        if warns:
            issue_str += c("yellow", f"{warns}W")
        if not issue_str:
            issue_str = c("green", "ok")

        print(f"{str(rel):<55} {info.get('cells', '?'):>5} {info.get('size_kb', '?'):>5} "
              f"{badge:>5} {owner_str:>6} {issue_str:>10}")
        for severity, msg in issues:
            color = "red" if severity == "ERROR" else "yellow"
            print(f"      {c(color, '[' + severity + ']')} {msg}")

    print()
    print(c("bold", "Summary:"))
    print(f"  Notebooks    : {len(paths)}")
    print(f"  Total cells  : {total_cells}")
    print(f"  OWNER/REPO   : {total_owner_repo}  (placeholder; replace before publishing)")
    print(f"  Errors       : {c('red' if n_errors else 'green', str(n_errors))}")
    print(f"  Warnings     : {c('yellow' if n_warns else 'green', str(n_warns))}")
    print()

    if n_errors:
        print(c("red", "FAIL"))
        return 1
    if n_warns:
        print(c("yellow", "PASS with warnings"))
        return 0
    print(c("green", "PASS"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
