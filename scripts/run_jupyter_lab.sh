#!/usr/bin/env bash
# Launch Jupyter Lab in the workshop repo for local notebook testing.
#
# Two install profiles:
#   minimal  (default) — install only jupyterlab + ipykernel + ipywidgets.
#                        Each notebook's setup cell still has its own
#                        `%pip install ...` for its specific deps, so the
#                        first run of any notebook installs what it needs.
#                        Matches the Colab workflow exactly. Cheap setup.
#
#   --prefetch          — install everything in requirements_notebooks.txt
#                        upfront (~5 GB). Faster cold start for every
#                        notebook afterward. Useful if you'll iterate on
#                        many notebooks in one sitting.
#
# Common flags:
#   --no-launch         install only; do not start Jupyter Lab
#   --port N            Lab port (default 8888)
#   --force-reinstall   reinstall even if packages are present
#   --help, -h          this message
#
# Examples:
#   ./scripts/run_jupyter_lab.sh                    # quick: minimal install + launch
#   ./scripts/run_jupyter_lab.sh --prefetch         # full upfront install + launch
#   ./scripts/run_jupyter_lab.sh --prefetch --no-launch   # install only, exit
#
# Sister scripts:
#   dry_run_local.sh         — Jupyter Book site preview (different concern)
#   test_notebook.sh PATH    — headless execute one notebook (smoke test)

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
  C_CYAN=$'\033[36m'; C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
else
  C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN=''; C_DIM=''; C_BOLD=''; C_RESET=''
fi

log()  { printf "%s\n" "$*"; }
info() { printf "${C_CYAN}%s${C_RESET}\n" "→ $*"; }
ok()   { printf "${C_GREEN}%s${C_RESET}\n" "✓ $*"; }
warn() { printf "${C_YELLOW}%s${C_RESET}\n" "! $*"; }
err()  { printf "${C_RED}%s${C_RESET}\n" "✗ $*" >&2; }
hdr()  { printf "\n${C_BOLD}%s${C_RESET}\n" "$*"; }

die()  { err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Locate repo
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQS_FILE="$SCRIPT_DIR/requirements_notebooks.txt"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
PROFILE="minimal"
LAUNCH=true
PORT=8888
FORCE_REINSTALL=false

usage() {
  sed -n '2,30p' "$0" | sed 's/^#//' | sed 's/^ //'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefetch)             PROFILE="full"; shift ;;
    --no-launch)            LAUNCH=false; shift ;;
    --port)                 PORT="$2"; shift 2 ;;
    --port=*)               PORT="${1#*=}"; shift ;;
    --force-reinstall)      FORCE_REINSTALL=true; shift ;;
    --help|-h)              usage ;;
    *)                      die "Unknown argument: $1 (try --help)" ;;
  esac
done

hdr "Jupyter Lab — local notebook testing"
log "Repo root: ${REPO_ROOT}"
log "Profile  : ${PROFILE}"
log "Launch   : ${LAUNCH}"
log "Port     : ${PORT}"
echo

# ---------------------------------------------------------------------------
# 1. Repo location
# ---------------------------------------------------------------------------
hdr "1. Repo location"
[[ -f "$REPO_ROOT/_config.yml" ]] || die "Not a Jupyter Book repo (no _config.yml at $REPO_ROOT)"
ok "_config.yml present"
[[ -d "$REPO_ROOT/notebooks"   ]] || die "No notebooks/ directory"
N_NOTEBOOKS="$(find "$REPO_ROOT/notebooks" "$REPO_ROOT/lectures" "$REPO_ROOT/ramp-up" -maxdepth 3 -name '*.ipynb' -not -path '*/\.ipynb_checkpoints/*' 2>/dev/null | wc -l | tr -d ' ')"
ok "Found ${N_NOTEBOOKS} notebooks"

# ---------------------------------------------------------------------------
# 2. Python + venv
# ---------------------------------------------------------------------------
hdr "2. Python"
PY_BIN="$(command -v python3 || true)"
[[ -n "$PY_BIN" ]] || die "python3 not on PATH"
PY_VER="$(python3 --version 2>&1)"
ok "Python: ${PY_BIN} (${PY_VER})"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  ok "In venv: ${VIRTUAL_ENV}"
else
  warn "Not in a virtual environment."
  warn "STRONGLY recommended for notebook testing — installs can be ~5 GB."
  warn "  python3 -m venv ~/.venvs/notebooks-preview"
  warn "  source ~/.venvs/notebooks-preview/bin/activate"
  warn "Continuing anyway. Packages will go into the active Python."
fi

# ---------------------------------------------------------------------------
# 3. Dependency check (verify-existing-installs principle)
# ---------------------------------------------------------------------------
hdr "3. Dependency check"

# Robust version check — works whether or not the module exposes __version__
get_pkg_version() {
  local pkg="$1"
  python3 - <<PY
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        print(version("${pkg}"))
    except PackageNotFoundError:
        print("absent")
except Exception:
    print("unknown")
PY
}

# Always-required for either profile
MINIMAL_PKGS=(jupyterlab ipykernel ipywidgets)

log "Always-required (we will install if missing):"
NEEDS_MINIMAL=false
for pkg in "${MINIMAL_PKGS[@]}"; do
  ver="$(get_pkg_version "$pkg")"
  case "$ver" in
    absent|unknown)
      warn "  ${pkg}  (missing)"
      NEEDS_MINIMAL=true
      ;;
    *)
      ok "  ${pkg}  (already installed: ${ver})"
      ;;
  esac
done

if $FORCE_REINSTALL; then
  warn "--force-reinstall passed; will reinstall regardless"
  NEEDS_MINIMAL=true
fi

if [[ "$PROFILE" == "full" ]]; then
  log
  log "${C_BOLD}Prefetch profile selected.${C_RESET} Will install from:"
  log "  ${REQS_FILE}"
  if [[ ! -f "$REQS_FILE" ]]; then
    die "Requirements file missing: $REQS_FILE"
  fi
  N_PKGS="$(grep -cE '^[a-zA-Z]' "$REQS_FILE" || true)"
  log "  ${N_PKGS} package specifications"
  warn "  Estimated install size: ~5 GB (PyTorch + TensorFlow + cellpose + stardist + diffusers + others)"
  warn "  This is a one-time cost. Subsequent runs reuse the installed packages."
fi

# ---------------------------------------------------------------------------
# 4. Install (only if needed)
# ---------------------------------------------------------------------------
if $NEEDS_MINIMAL; then
  hdr "4a. Install minimal Lab toolchain"
  PIP_CMD=(python3 -m pip install -U)
  $FORCE_REINSTALL && PIP_CMD+=(--force-reinstall)
  PIP_CMD+=("${MINIMAL_PKGS[@]}")
  info "Running: ${PIP_CMD[*]}"
  if ! "${PIP_CMD[@]}"; then
    err "pip install failed"
    exit 3
  fi
  ok "Minimal install complete."
fi

if [[ "$PROFILE" == "full" ]]; then
  hdr "4b. Install full notebook deps (--prefetch)"
  PIP_CMD=(python3 -m pip install -U)
  $FORCE_REINSTALL && PIP_CMD+=(--force-reinstall)
  PIP_CMD+=(-r "$REQS_FILE")
  info "Running: ${PIP_CMD[*]}"
  info "(will be slow first time — go grab coffee)"
  if ! "${PIP_CMD[@]}"; then
    err "Full install failed."
    err "Some optional packages may be incompatible with your Python version."
    err "You can still launch Lab and install the failing pkg manually per-notebook."
    if $LAUNCH; then
      warn "Continuing to Lab launch with whatever installed successfully."
    else
      exit 3
    fi
  else
    ok "Full notebook install complete."
  fi
fi

# ---------------------------------------------------------------------------
# 5. Verify Jupyter Lab is launchable
# ---------------------------------------------------------------------------
hdr "5. Verify Lab"
if ! command -v jupyter >/dev/null 2>&1; then
  err "jupyter command not found after install. Something went wrong."
  err "Try:  python3 -m pip install -U jupyterlab  manually."
  exit 3
fi
JL_VER="$(get_pkg_version jupyterlab)"
ok "jupyterlab: ${JL_VER}"

if ! $LAUNCH; then
  echo
  ok "Install-only mode. To launch Lab manually:"
  log "  cd ${REPO_ROOT}"
  log "  jupyter lab --port=${PORT}"
  exit 0
fi

# ---------------------------------------------------------------------------
# 6. Launch
# ---------------------------------------------------------------------------
# Port collision check
if command -v lsof >/dev/null 2>&1; then
  if lsof -i ":$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    warn "Port $PORT is already in use."
    warn "Re-run with --port=8889 (or any free port), or stop the existing Lab/server."
    exit 0
  fi
fi

hdr "6. Launch Jupyter Lab"
info "Starting Lab in: ${REPO_ROOT}"
info "Will auto-open a browser tab at:  ${C_BOLD}http://localhost:${PORT}${C_RESET}"
info "Press Ctrl-C to stop."
echo
cd "$REPO_ROOT"
exec jupyter lab --port="$PORT"
