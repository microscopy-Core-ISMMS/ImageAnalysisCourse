#!/usr/bin/env bash
# Headlessly execute one notebook and report pass/fail. Useful for tight
# iteration loops on a single notebook without launching Jupyter Lab.
#
# Usage:
#   ./scripts/test_notebook.sh notebooks/12_deconvolution.ipynb
#   ./scripts/test_notebook.sh notebooks/12_deconvolution.ipynb --timeout 600
#
# Behavior:
#   - Runs `jupyter nbconvert --to notebook --execute --output /tmp/...`
#   - Captures the rendered notebook to a temp file (does NOT modify the source)
#   - Prints first error if execution fails
#   - Prints success message if it runs clean
#   - Exit 0 on success, 1 on failure
#
# Notes:
#   - Per-cell timeout defaults to 300 s. Override with --timeout.
#   - Notebook deps must already be installed in the active Python.
#     Run ./scripts/run_jupyter_lab.sh --prefetch --no-launch first if needed.

set -euo pipefail

# Colors
if [[ -t 1 ]]; then
  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
  C_CYAN=$'\033[36m'; C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
else
  C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN=''; C_DIM=''; C_BOLD=''; C_RESET=''
fi

ok()   { printf "${C_GREEN}%s${C_RESET}\n" "✓ $*"; }
err()  { printf "${C_RED}%s${C_RESET}\n" "✗ $*" >&2; }
info() { printf "${C_CYAN}%s${C_RESET}\n" "→ $*"; }
warn() { printf "${C_YELLOW}%s${C_RESET}\n" "! $*"; }
die()  { err "$*"; exit 1; }

# Args
NOTEBOOK=""
TIMEOUT=300
KEEP_OUTPUT=false

usage() {
  sed -n '2,21p' "$0" | sed 's/^#//' | sed 's/^ //'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout)         TIMEOUT="$2"; shift 2 ;;
    --timeout=*)       TIMEOUT="${1#*=}"; shift ;;
    --keep)            KEEP_OUTPUT=true; shift ;;
    --help|-h)         usage ;;
    -*)                die "Unknown flag: $1" ;;
    *)                 NOTEBOOK="$1"; shift ;;
  esac
done

[[ -n "$NOTEBOOK" ]] || die "No notebook path given. Try --help."
[[ -f "$NOTEBOOK" ]] || die "File not found: $NOTEBOOK"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve to absolute path
NOTEBOOK="$(cd "$(dirname "$NOTEBOOK")" && pwd)/$(basename "$NOTEBOOK")"
NB_BASENAME="$(basename "$NOTEBOOK" .ipynb)"

OUTPUT_DIR="$(mktemp -d -t jb-test-XXXXXX)"
OUTPUT_NB="$OUTPUT_DIR/${NB_BASENAME}-executed.ipynb"
LOG_FILE="$OUTPUT_DIR/${NB_BASENAME}.log"

info "Notebook: $NOTEBOOK"
info "Timeout : ${TIMEOUT}s per cell"
info "Output  : $OUTPUT_NB"
info "Log     : $LOG_FILE"
echo

# Verify nbconvert is installed
if ! command -v jupyter >/dev/null 2>&1; then
  die "jupyter command not found. Run scripts/run_jupyter_lab.sh first."
fi
if ! python3 -c 'import nbconvert' 2>/dev/null; then
  die "nbconvert not installed. Run scripts/run_jupyter_lab.sh first."
fi

# Run
START_TS="$(date +%s)"
set +e
jupyter nbconvert \
  --to notebook \
  --execute \
  --output "$OUTPUT_NB" \
  --ExecutePreprocessor.timeout="$TIMEOUT" \
  --ExecutePreprocessor.kernel_name=python3 \
  "$NOTEBOOK" \
  > "$LOG_FILE" 2>&1
RC=$?
set -e
END_TS="$(date +%s)"
ELAPSED=$((END_TS - START_TS))

if [[ "$RC" -eq 0 ]]; then
  ok "PASS  (executed in ${ELAPSED}s)"
  ok "Output notebook: $OUTPUT_NB"
  if ! $KEEP_OUTPUT; then
    rm -f "$OUTPUT_NB"
    rm -f "$LOG_FILE"
    rmdir "$OUTPUT_DIR" 2>/dev/null || true
    info "(temp output cleaned; pass --keep to retain)"
  fi
  exit 0
fi

# Failure path — extract the first error from the log
err "FAIL  (exit ${RC}, after ${ELAPSED}s)"
echo
warn "First error in execution log:"
grep -E '^(Error|Cell.*line|.*Error:|.*Exception:|Traceback)' "$LOG_FILE" | head -20 || \
  tail -40 "$LOG_FILE"
echo
warn "Full log: $LOG_FILE"
warn "Partial output notebook: $OUTPUT_NB"
warn "(both kept on failure for inspection)"
exit 1
