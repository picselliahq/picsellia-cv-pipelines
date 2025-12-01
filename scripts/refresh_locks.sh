#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/refresh_locks.sh --pipeline <name|all> [--pipeline-dir "<path>"]
# Examples:
#   scripts/refresh_locks.sh --pipeline bounding_box_cropper
#   scripts/refresh_locks.sh --pipeline yolov8/pre_annotation
#   scripts/refresh_locks.sh --pipeline all
#   scripts/refresh_locks.sh --pipeline yolov8 --pipeline-dir "/tmp/my-pipelines"

command -v uv >/dev/null || { echo "❌ 'uv' not found in PATH"; exit 1; }

PIPE="all"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIPELINES_DIR_DEFAULT="$BASE_DIR/pipelines"
PIPELINES_DIR="$PIPELINES_DIR_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline)      PIPE="${2:-all}"; shift 2 ;;
    --pipeline-dir)  PIPELINES_DIR="${2:-$PIPELINES_DIR_DEFAULT}"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done

[[ -d "$PIPELINES_DIR" ]] || { echo "❌ PIPELINES_DIR does not exist: $PIPELINES_DIR"; exit 1; }

echo "📦 Pipelines dir: $PIPELINES_DIR"
echo "🎯 Selection:     $PIPE"
echo

# -------------------------------------------------------------------
# Build the list of actual project directories (those with pyproject.toml)
# -------------------------------------------------------------------
declare -a TARGET_DIRS=()

if [[ "$PIPE" == "all" ]]; then
  # All pipelines: any folder under pipelines/ that contains a pyproject.toml
  while IFS= read -r -d '' pyproj; do
    TARGET_DIRS+=("$(dirname "$pyproj")")
  done < <(find "$PIPELINES_DIR" -mindepth 1 -maxdepth 3 -type f -name "pyproject.toml" -print0)

elif [[ "$PIPE" == */* ]]; then
  # e.g. yolov8/pre_annotation
  dir="$PIPELINES_DIR/$PIPE"
  if [[ -d "$dir" ]]; then
    TARGET_DIRS+=("$dir")
  else
    echo "⚠️  No such pipeline directory: $dir"
  fi

else
  # e.g. PIPE="bounding_box_cropper" or PIPE="yolov8"
  # Case 1: direct project at pipelines/<PIPE>/pyproject.toml
  if [[ -f "$PIPELINES_DIR/$PIPE/pyproject.toml" ]]; then
    TARGET_DIRS+=("$PIPELINES_DIR/$PIPE")
  fi

  # Case 2: nested projects under pipelines/<PIPE>/*/pyproject.toml
  if [[ -d "$PIPELINES_DIR/$PIPE" ]]; then
    while IFS= read -r -d '' pyproj; do
      TARGET_DIRS+=("$(dirname "$pyproj")")
    done < <(find "$PIPELINES_DIR/$PIPE" -mindepth 1 -maxdepth 3 -type f -name "pyproject.toml" -print0)
  fi
fi

# Deduplicate just in case
if ((${#TARGET_DIRS[@]} > 0)); then
  mapfile -t TARGET_DIRS < <(printf '%s\n' "${TARGET_DIRS[@]}" | sort -u)
fi

if ((${#TARGET_DIRS[@]} == 0)); then
  echo "⚠️  No pipeline directories found for selection '$PIPE'."
  exit 0
fi

count=0

for P in "${TARGET_DIRS[@]}"; do
  [[ -d "$P" ]] || continue
  name="$(basename "$P")"

  echo "─────────────────────────────────────────────"
  echo "📁 $name  ($P)"

  if [[ ! -f "$P/pyproject.toml" ]]; then
    echo "⏭️  SKIP: no pyproject.toml at the root."
    continue
  fi

  ((++count))

  if [[ -f "$P/uv.lock" ]]; then
    rm -f "$P/uv.lock"
    echo "🧹 Removed uv.lock"
  else
    echo "ℹ️  No uv.lock found"
  fi

  echo "🔄 Running 'uv lock'"
  # Important: wrap in `if ...; then` so `set -e` doesn't kill the script
  if (cd "$P" && uv lock); then
    echo "✅ Done"
  else
    echo "⚠️  uv lock failed for $name, skipping (exit code $?)"
    # continue with other pipelines instead of exiting the whole script
  fi
done

if [[ $count -eq 0 ]]; then
  echo "⚠️  No pipelines processed (missing pyproject.toml?)."
else
  echo "✨ Completed for $count pipeline(s)."
fi
