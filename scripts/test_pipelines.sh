#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/test_pipelines.sh --pipeline <name|all> [--report]
# Ex:
#   scripts/test_pipelines.sh --pipeline bounding_box_cropper
#   scripts/test_pipelines.sh --pipeline all --report

command -v pxl-pipeline >/dev/null || { echo "❌ pxl-pipeline introuvable dans le PATH"; exit 1; }

PIPE="all"
WRITE_REPORT=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline) PIPE="${2:-all}"; shift 2 ;;
    --report)   WRITE_REPORT=true; shift ;;
    *) echo "Arg inconnu: $1"; exit 2 ;;
  esac
done

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TESTS_DIR="$BASE_DIR/tests"
PIPELINES_DIR="$BASE_DIR/pipelines"
REPORT_FILE="$BASE_DIR/test_report.txt"
$WRITE_REPORT && : > "$REPORT_FILE"

RESULTS=()

echo "🧪 Tests dir:     $TESTS_DIR"
echo "📦 Pipelines dir: $PIPELINES_DIR"
echo "🎯 Sélection:     ${PIPE}"

run_pipeline_test() {
  local PIPELINE_NAME="$1"   # e.g. training
  local RUN_CONFIG="$2"
  local PIPELINE_DIR="$3"    # e.g. pipelines/clip
  local DISPLAY_NAME="$4"    # e.g. clip/training

  echo
  echo "─────────────────────────────────────────────"
  echo "Pipeline: $DISPLAY_NAME"
  echo "Folder:   $PIPELINE_DIR"
  echo "Config:   $RUN_CONFIG"
  echo "─────────────────────────────────────────────"
  echo

  pushd "$PIPELINE_DIR" >/dev/null || return 0

  echo "▶️  pxl-pipeline test $PIPELINE_NAME"
  if pxl-pipeline test "$PIPELINE_NAME" --run-config-file "$RUN_CONFIG"; then
    echo "✅ test passed for $DISPLAY_NAME"
    RESULTS+=("✅ $DISPLAY_NAME")
  else
    echo "❌ test failed for $DISPLAY_NAME"
    RESULTS+=("❌ $DISPLAY_NAME (test)")
    # on continue les autres
  fi

  popd >/dev/null
}

matched=false

# Découverte des configs
while IFS= read -r -d '' CONFIG; do
  REL_PATH="${CONFIG#$TESTS_DIR/}"             # ex: bounding_box_cropper/run_config.toml, clip/training/run_config.toml
  TOP="${REL_PATH%%/*}"                        # premier segment
  [[ "$PIPE" != "all" && "$TOP" != "$PIPE" ]] && continue

  matched=true
  PIPELINE_NAME="$(basename "$(dirname "$CONFIG")")" # ex: training (ou le nom si direct)
  PIPELINE_DIR="$PIPELINES_DIR/$TOP"                 # ex: pipelines/bounding_box_cropper
  DISPLAY_NAME="${REL_PATH%/run_config.toml}"

  if [[ ! -d "$PIPELINE_DIR" ]]; then
    echo "⚠️  $PIPELINE_DIR introuvable, skip $DISPLAY_NAME"
    continue
  fi

  run_pipeline_test "$PIPELINE_NAME" "$CONFIG" "$PIPELINE_DIR" "$DISPLAY_NAME"
done < <(find "$TESTS_DIR" -name "run_config.toml" -print0)

if ! $matched; then
  echo "⚠️  Aucun test trouvé pour --pipeline '${PIPE}'. Vérifie tests/<pipeline>/.../run_config.toml"
fi

# Résumé
{
  echo
  echo "📊 Final summary:"
  if ((${#RESULTS[@]})); then
    for r in "${RESULTS[@]}"; do echo "$r"; done
  else
    echo "— aucun résultat (tableau vide) —"
  fi
} | { $WRITE_REPORT && tee -a "$REPORT_FILE" || cat; }

$WRITE_REPORT && echo -e "\n📝 Full report saved in: $REPORT_FILE"
