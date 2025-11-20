#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Usage:
#   scripts/test_pipelines.sh --pipeline <name|all> \
#       [--organization ORG] [--env ENV] [--token TOKEN] [--report]
#
# Examples:
#   scripts/test_pipelines.sh --pipeline all \
#       --organization test-account --env STAGING --token "$PXL_API_TOKEN"
#
#   scripts/test_pipelines.sh --pipeline yolov8/pre_annotation \
#       --organization test-account --env STAGING --token "$PXL_API_TOKEN" --report
# ---------------------------------------------------------------------------

command -v pxl-pipeline >/dev/null || { echo "❌ 'pxl-pipeline' not found in PATH"; exit 1; }

PIPE="all"                           # all | <top> | <top/sub>
ORGANIZATION="${ORGANIZATION:-test-account}"
ENVIRONMENT="${ENVIRONMENT:-STAGING}"
TOKEN="${PXL_API_TOKEN:-}"           # can also be provided via --token
WRITE_REPORT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline)     PIPE="${2:-all}"; shift 2 ;;
    --organization) ORGANIZATION="${2:?--organization requires a value}"; shift 2 ;;
    --env)          ENVIRONMENT="${2:?--env requires a value}"; shift 2 ;;
    --token)        TOKEN="${2:?--token requires a value}"; shift 2 ;;
    --report)       WRITE_REPORT=true; shift ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TESTS_DIR="$BASE_DIR/tests"
PIPELINES_DIR="$BASE_DIR/pipelines"
REPORT_FILE="$BASE_DIR/test_report.txt"
if [ "$WRITE_REPORT" = true ]; then
  : > "$REPORT_FILE"
fi

RESULTS=()
ANY_FAILURE=false

log_header() {
  echo "🧪 Tests dir:     $TESTS_DIR"
  echo "📦 Pipelines dir: $PIPELINES_DIR"
  echo "🎯 Selection:     ${PIPE}"
  echo "🏢 Organization:  $ORGANIZATION"
  echo "🌍 Environment:   $ENVIRONMENT"
  if [ "$WRITE_REPORT" = true ]; then
    echo "📝 Report file:   $REPORT_FILE"
  fi
}

logr() {
  if [ "$WRITE_REPORT" = true ]; then
    echo -e "$*" | tee -a "$REPORT_FILE"
  else
    echo -e "$*"
  fi
}

# --------------------
# Non-interactive login (optional)
# --------------------
if [[ -n "$TOKEN" ]]; then
  echo "🔐 Performing non-interactive login..."
  set +e
  pxl-pipeline login --organization "$ORGANIZATION" --env "$ENVIRONMENT" --token "$TOKEN"
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    echo "❌ Login failed (organization / env / token invalid?)."
    exit 1
  fi
else
  echo "ℹ️  No token provided (neither --token nor PXL_API_TOKEN)."
  echo "    The script assumes a valid token/context is already configured."
fi

log_header

run_pipeline_test() {
  local PIPELINE_NAME="$1"   # e.g. pre_annotation or training
  local RUN_CONFIG="$2"
  local PIPELINE_DIR="$3"    # e.g. pipelines/yolov8
  local DISPLAY_NAME="$4"    # e.g. yolov8/pre_annotation

  echo
  echo "─────────────────────────────────────────────"
  echo "Pipeline: $DISPLAY_NAME"
  echo "Folder:   $PIPELINE_DIR"
  echo "Config:   $RUN_CONFIG"
  echo "─────────────────────────────────────────────"
  echo

  pushd "$PIPELINE_DIR" >/dev/null || {
    echo "⚠️  Failed to cd into $PIPELINE_DIR, skipping $DISPLAY_NAME"
    RESULTS+=("❌ $DISPLAY_NAME (cd failed)")
    ANY_FAILURE=true
    return
  }

  echo "▶️  pxl-pipeline test $PIPELINE_NAME"
  if pxl-pipeline test "$PIPELINE_NAME" --run-config-file "$RUN_CONFIG"; then
    echo "✅ test passed for $DISPLAY_NAME"
    RESULTS+=("✅ $DISPLAY_NAME")
  else
    echo "❌ test failed for $DISPLAY_NAME"
    RESULTS+=("❌ $DISPLAY_NAME (test)")
    ANY_FAILURE=true
  fi

  popd >/dev/null
}

matched=false

# Discover configs
while IFS= read -r -d '' CONFIG; do
  REL_PATH="${CONFIG#$TESTS_DIR/}"             # e.g. yolov8/pre_annotation/run_config.toml
  REL_NO_SUFFIX="${REL_PATH%/run_config.toml}" # e.g. yolov8/pre_annotation

  # Filtering logic:
  if [[ "$PIPE" != "all" ]]; then
    if [[ "$PIPE" == */* ]]; then
      # --pipeline like "yolov8/pre_annotation": match full relative path
      [[ "$REL_NO_SUFFIX" != "$PIPE" ]] && continue
    else
      # --pipeline like "yolov8": match top-level folder
      TOP="${REL_PATH%%/*}"
      [[ "$TOP" != "$PIPE" ]] && continue
    fi
  fi

  matched=true
  PIPELINE_NAME="$(basename "$(dirname "$CONFIG")")" # e.g. pre_annotation or training
  TOP="${REL_PATH%%/*}"                              # e.g. yolov8
  PIPELINE_DIR="$PIPELINES_DIR/$TOP"                 # e.g. pipelines/yolov8
  DISPLAY_NAME="$REL_NO_SUFFIX"                      # e.g. yolov8/pre_annotation

  if [[ ! -d "$PIPELINES_DIR" ]]; then
    echo "⚠️  Pipelines directory '$PIPELINES_DIR' not found, skipping $DISPLAY_NAME"
    ANY_FAILURE=true
    RESULTS+=("❌ $DISPLAY_NAME (missing pipelines dir)")
    continue
  fi

  if [[ ! -d "$PIPELINE_DIR" ]]; then
    echo "⚠️  $PIPELINE_DIR not found, skipping $DISPLAY_NAME"
    ANY_FAILURE=true
    RESULTS+=("❌ $DISPLAY_NAME (missing pipeline dir)")
    continue
  fi

  run_pipeline_test "$PIPELINE_NAME" "$CONFIG" "$PIPELINE_DIR" "$DISPLAY_NAME"
done < <(find "$TESTS_DIR" -name "run_config.toml" -print0)

if ! $matched; then
  echo "⚠️  No test found for --pipeline '${PIPE}'. Check tests/<pipeline>/.../run_config.toml"
fi

# Summary
{
  echo
  echo "📊 Final summary:"
  if ((${#RESULTS[@]})); then
    for r in "${RESULTS[@]}"; do echo "$r"; done
  else
    echo "— no results (empty set) —"
  fi
} | { [ "$WRITE_REPORT" = true ] && tee -a "$REPORT_FILE" || cat; }

if [ "$WRITE_REPORT" = true ]; then
  echo -e "\n📝 Full report saved in: $REPORT_FILE"
fi

if $ANY_FAILURE; then
  echo
  echo "❌ At least one pipeline test failed."
  exit 1
else
  echo
  echo "✅ All pipeline tests succeeded."
  exit 0
fi
