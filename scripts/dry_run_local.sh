#!/usr/bin/env bash
# Local dry-run for the Jupyter Book companion.
#
# What this script does:
#   1. Locates the repo root (parent of scripts/)
#   2. Pre-flight checks: directory, Python, venv, dependency state
#   3. Notebook JSON validity check (delegates to dry_run_check.py)
#   4. Installs jupyter-book ONLY if it isn't already importable
#   5. Builds the book (jupyter-book clean . && jupyter-book build .)
#   6. Summarizes warnings from the build log
#   7. Serves _build/html via python3 -m http.server (unless --no-serve)
#
# Modes:
#   (default)              full: pre-flight + check + install-if-needed + build + serve
#   --check, -c            pre-flight + notebook check only; no install/build/serve
#   --no-serve             build only; do not start the local server
#   --port N               serve port (default 8000)
#   --force-reinstall      reinstall jupyter-book even if present
#   --clean                delete _build/ before build
#   --help, -h             usage
#
# Exit codes:
#   0  success
#   1  pre-flight or notebook validity failed
#   2  build failed
#   3  install failed

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors and logging
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
# Locate repo root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODE="full"
PORT=8000
FORCE_REINSTALL=false
CLEAN=false

usage() {
  sed -n '2,21p' "$0" | sed 's/^#//' | sed 's/^ //'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check|-c)             MODE="check"; shift ;;
    --no-serve)             MODE="no-serve"; shift ;;
    --port)                 PORT="$2"; shift 2 ;;
    --port=*)               PORT="${1#*=}"; shift ;;
    --force-reinstall)      FORCE_REINSTALL=true; shift ;;
    --clean)                CLEAN=true; shift ;;
    --help|-h)              usage ;;
    *)                      die "Unknown argument: $1 (try --help)" ;;
  esac
done

hdr "Jupyter Book — local dry-run"
log "Repo root: ${REPO_ROOT}"
log "Mode     : ${MODE}"
log "Port     : ${PORT}"
echo

# ---------------------------------------------------------------------------
# 1. Repo location check
# ---------------------------------------------------------------------------
hdr "1. Repo location"
[[ -f "$REPO_ROOT/_config.yml" ]] || die "Not a Jupyter Book repo (no _config.yml at $REPO_ROOT)"
[[ -f "$REPO_ROOT/_toc.yml"     ]] || die "No _toc.yml at $REPO_ROOT"
ok "_config.yml present"
ok "_toc.yml present"

# ---------------------------------------------------------------------------
# 2. Python check
# ---------------------------------------------------------------------------
hdr "2. Python"
PY_BIN="$(command -v python3 || true)"
[[ -n "$PY_BIN" ]] || die "python3 not on PATH"
PY_VER="$(python3 --version 2>&1)"
ok "Python: ${PY_BIN} (${PY_VER})"

# ---------------------------------------------------------------------------
# 3. Venv check
# ---------------------------------------------------------------------------
hdr "3. Virtual environment"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  ok "In venv: ${VIRTUAL_ENV}"
else
  warn "Not in a virtual environment"
  warn "Recommended (one-time setup):"
  printf "${C_DIM}  python3 -m venv ~/.venvs/jb-preview${C_RESET}\n"
  printf "${C_DIM}  source ~/.venvs/jb-preview/bin/activate${C_RESET}\n"
  warn "  Continuing anyway. Install will go into the active Python."
fi

# ---------------------------------------------------------------------------
# 4. Dependency check (verify-existing-installs principle)
# ---------------------------------------------------------------------------
hdr "4. Dependency check"

# IMPORTANT — we pin to Jupyter Book v1.x (Sphinx-based).
# v2.x is a complete rewrite on MyST-MD + Node.js with a different config
# schema. Our _config.yml and _toc.yml are v1-format, so v2 won't build them.
JB_PIN='jupyter-book>=1.0,<2.0'
REQUIRED_PKGS=(jupyter-book)
INDIRECT_PKGS=(sphinx-book-theme myst-parser myst-nb sphinx-design)

# Robust version check via importlib.metadata (works whether or not the
# module exposes __version__). Returns the installed version, or "absent",
# or "unknown".
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

# Compare a semver-ish string against a major-version threshold.
# Returns 0 (true) if version is >= threshold major, else 1.
version_major_at_least() {
  local ver="$1" min="$2"
  local ver_major="${ver%%.*}"
  [[ "$ver_major" =~ ^[0-9]+$ ]] || return 1
  [[ "$ver_major" -ge "$min" ]]
}

log "Required (we will install if missing):"
NEEDS_INSTALL=false
WRONG_MAJOR=false
for pkg in "${REQUIRED_PKGS[@]}"; do
  ver="$(get_pkg_version "$pkg")"
  case "$ver" in
    absent)
      warn "  ${pkg}  (missing)"
      NEEDS_INSTALL=true
      ;;
    unknown)
      warn "  ${pkg}  (importlib.metadata failed; will reinstall to be safe)"
      NEEDS_INSTALL=true
      ;;
    *)
      if [[ "$pkg" == "jupyter-book" ]]; then
        if version_major_at_least "$ver" 2; then
          err "  ${pkg}  (installed: ${ver} — this is v2, INCOMPATIBLE with our config)"
          err "  Our _config.yml and _toc.yml are v1-format. v2 is MyST-MD/Node based."
          err "  Fix:  pip uninstall -y jupyter-book  (then re-run this script)"
          WRONG_MAJOR=true
        else
          ok "  ${pkg}  (already installed: ${ver}, v1.x — good)"
        fi
      else
        ok "  ${pkg}  (already installed: ${ver})"
      fi
      ;;
  esac
done

if $WRONG_MAJOR; then
  err "Aborting because an incompatible major version of jupyter-book is installed."
  err "Run the uninstall command above, then re-run this script."
  exit 3
fi

log "Indirect dependencies (auto-installed by jupyter-book; we will not install separately):"
for pkg in "${INDIRECT_PKGS[@]}"; do
  ver="$(get_pkg_version "$pkg")"
  case "$ver" in
    absent|unknown)
      log "  ${C_DIM}${pkg}  (will arrive via jupyter-book if installed)${C_RESET}"
      ;;
    *)
      ok "  ${pkg}  (present: ${ver})"
      ;;
  esac
done
echo

if $FORCE_REINSTALL; then
  warn "--force-reinstall passed; will reinstall jupyter-book regardless"
  NEEDS_INSTALL=true
fi

if ! $NEEDS_INSTALL; then
  ok "Decision: nothing to install."
else
  info "Decision: will install/upgrade jupyter-book."
fi

# ---------------------------------------------------------------------------
# 5. Notebook JSON validity
# ---------------------------------------------------------------------------
hdr "5. Notebook validity (no installs needed)"

if ! python3 "$SCRIPT_DIR/dry_run_check.py" "$REPO_ROOT"; then
  err "Notebook validity check failed"
  err "Fix the errors above before continuing"
  exit 1
fi

# ---------------------------------------------------------------------------
# 6. Stop here in --check mode
# ---------------------------------------------------------------------------
if [[ "$MODE" == "check" ]]; then
  echo
  ok "Check-only mode: pre-flight + notebook validity passed. Skipping install/build/serve."
  exit 0
fi

# ---------------------------------------------------------------------------
# 7. Install (only if needed)
# ---------------------------------------------------------------------------
if $NEEDS_INSTALL; then
  hdr "6. Install jupyter-book"
  PIP_CMD=(python3 -m pip install -U)
  if $FORCE_REINSTALL; then
    PIP_CMD+=(--force-reinstall)
  fi
  # Pin to v1.x (Sphinx-based). v2.x is a different product (MyST-MD/Node).
  PIP_CMD+=("$JB_PIN")

  info "Running: ${PIP_CMD[*]}"
  if ! "${PIP_CMD[@]}"; then
    err "pip install failed"
    exit 3
  fi
  ok "Install complete."

  # Re-verify
  JB_VER="$(get_pkg_version jupyter-book)"
  if [[ "$JB_VER" == "absent" || "$JB_VER" == "unknown" ]]; then
    err "jupyter-book still not detectable after install"
    exit 3
  fi
  if version_major_at_least "$JB_VER" 2; then
    err "Install resolved to v${JB_VER} (v2.x). Pin failed somehow."
    err "Try:  pip install -U '$JB_PIN'  manually."
    exit 3
  fi
  ok "jupyter-book: ${JB_VER} (v1.x — correct)"
fi

# ---------------------------------------------------------------------------
# 8. Build
# ---------------------------------------------------------------------------
hdr "7. Build"
cd "$REPO_ROOT"

if $CLEAN || [[ -d "$REPO_ROOT/_build" ]]; then
  info "Cleaning previous build output..."
  jupyter-book clean . >/dev/null
  ok "Cleaned _build/"
fi

BUILD_LOG="$(mktemp -t jb-build.XXXXXX.log)"
info "Building (output captured to ${BUILD_LOG})"

set +e
jupyter-book build . 2>&1 | tee "$BUILD_LOG"
BUILD_RC="${PIPESTATUS[0]}"
set -e

if [[ "$BUILD_RC" -ne 0 ]]; then
  err "jupyter-book build failed (exit ${BUILD_RC})"
  err "Full log: ${BUILD_LOG}"
  exit 2
fi

# Build summary: count warnings/errors in the log
N_WARN="$(grep -cE 'WARNING:'  "$BUILD_LOG" || true)"
N_ERR="$( grep -cE 'ERROR:'    "$BUILD_LOG" || true)"
echo
hdr "Build summary"
log "  Build log : ${BUILD_LOG}"
if [[ "$N_ERR" -gt 0 ]]; then
  warn "  Errors    : ${N_ERR} (review above)"
else
  ok   "  Errors    : 0"
fi
if [[ "$N_WARN" -gt 0 ]]; then
  warn "  Warnings  : ${N_WARN} (most are non-fatal — see notes below)"
else
  ok   "  Warnings  : 0"
fi

# Extract a short list of warning categories
if [[ "$N_WARN" -gt 0 ]]; then
  echo
  log "  Warning categories (first 8):"
  grep -E 'WARNING:' "$BUILD_LOG" | sed 's/.*WARNING: //' | sort -u | head -8 \
    | while IFS= read -r line; do printf "    %s%s%s\n" "$C_DIM" "$line" "$C_RESET"; done
fi

[[ -f "$REPO_ROOT/_build/html/index.html" ]] || die "_build/html/index.html missing — build did not produce HTML"
ok "Built HTML at: $REPO_ROOT/_build/html/"

# ---------------------------------------------------------------------------
# 9. Serve (unless --no-serve)
# ---------------------------------------------------------------------------
if [[ "$MODE" == "no-serve" ]]; then
  echo
  ok "Build-only mode. To preview manually:"
  log "  cd $REPO_ROOT/_build/html && python3 -m http.server $PORT"
  log "  open http://localhost:$PORT"
  exit 0
fi

# Check port availability
if command -v lsof >/dev/null 2>&1; then
  if lsof -i ":$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    warn "Port $PORT is already in use."
    warn "Re-run with --port=8001 (or any free port), or stop the existing server."
    exit 0
  fi
fi

hdr "8. Serve"
SERVE_DIR="$REPO_ROOT/_build/html"
info "Starting HTTP server in ${SERVE_DIR}"
info "Open: ${C_BOLD}http://localhost:${PORT}${C_RESET}"
info "Press Ctrl-C to stop."
echo
cd "$SERVE_DIR"
exec python3 -m http.server "$PORT"
