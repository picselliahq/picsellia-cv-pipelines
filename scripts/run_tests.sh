#!/usr/bin/env bash
set -euo pipefail

# Base directory of the repository (one level above scripts/)
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TESTS_DIR="$BASE_DIR/tests"          # Root folder containing all test configs
PIPELINES_DIR="$BASE_DIR/pipelines"  # Root folder containing all pipelines

REPORT_FILE="$BASE_DIR/test_report.txt"
: > "$REPORT_FILE"   # Clear report file at the start

RESULTS=()  # Collect test results (✅ / ❌)

# Run tests for a single pipeline
# $1 = pipeline name (used in pxl-pipeline command, e.g. "training")
# $2 = run_config file path
# $3 = pipeline directory (where pxl-pipeline must be executed)
# $4 = display name (for logs, e.g. "clip/training")
run_pipeline_test() {
  local PIPELINE_NAME=$1
  local RUN_CONFIG=$2
  local PIPELINE_DIR=$3
  local DISPLAY_NAME=$4

  echo
  echo "─────────────────────────────────────────────"
  echo "Pipeline: $DISPLAY_NAME"
  echo "Folder:   $PIPELINE_DIR"
  echo "Config:   $RUN_CONFIG"
  echo "─────────────────────────────────────────────"
  echo

  cd "$PIPELINE_DIR" || exit 1

  # Run pxl-pipeline test
  echo "▶️  pxl-pipeline test $PIPELINE_NAME"
  if pxl-pipeline test "$PIPELINE_NAME" --run-config-file "$RUN_CONFIG"; then
    echo "✅ test passed for $DISPLAY_NAME"
  else
    echo "❌ test failed for $DISPLAY_NAME"
    RESULTS+=("❌ $DISPLAY_NAME (test)")
    cd "$BASE_DIR"
    return 1
  fi

  # Run pxl-pipeline smoke-test
  echo "▶️  pxl-pipeline smoke-test $PIPELINE_NAME"
  if pxl-pipeline smoke-test "$PIPELINE_NAME" --run-config-file "$RUN_CONFIG"; then
    echo "✅ smoke-test passed for $DISPLAY_NAME"
    RESULTS+=("✅ $DISPLAY_NAME")
  else
    echo "❌ smoke-test failed for $DISPLAY_NAME"
    RESULTS+=("❌ $DISPLAY_NAME (smoke-test)")
  fi

  cd "$BASE_DIR"
}

# Discover all run_config.toml files in tests/ recursively
# For each run_config, compute:
# - PIPELINE_NAME: actual pipeline name (last folder, e.g. "training")
# - DISPLAY_NAME: relative path for logs (e.g. "clip/training")
# - PIPELINE_DIR: directory to cd into (e.g. pipelines/clip)
while IFS= read -r -d '' CONFIG; do
  REL_PATH="${CONFIG#$TESTS_DIR/}"
  PIPELINE_NAME="$(basename "$(dirname "$CONFIG")")"
  PIPELINE_PARENT="$(dirname "$REL_PATH")"
  PIPELINE_DIR="$PIPELINES_DIR/$(dirname "$PIPELINE_PARENT")"

  DISPLAY_NAME="$PIPELINE_PARENT"

  run_pipeline_test "$PIPELINE_NAME" "$CONFIG" "$PIPELINE_DIR" "$DISPLAY_NAME" || true
done < <(find "$TESTS_DIR" -name "run_config.toml" -print0)

# Final summary: last 5 results only
{
  echo
  echo "📊 Final summary:"
  for r in "${RESULTS[@]}"; do
    echo "$r"
  done | tail -n 5
} | tee -a "$REPORT_FILE"

echo
echo "📝 Full report saved in: $REPORT_FILE"
