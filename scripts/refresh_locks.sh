#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/refresh_locks.sh --pipeline <name|all> [--pipeline-dir "<path>"]
# Examples:
#   scripts/refresh_locks.sh --pipeline bounding_box_cropper
#   scripts/refresh_locks.sh --pipeline all
#   scripts/refresh_locks.sh --pipeline clip --pipeline-dir "/tmp/my-pipelines"

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

filter="*"
[[ "$PIPE" != "all" ]] && filter="$PIPE"

echo "📦 Pipelines dir: $PIPELINES_DIR"
echo "🎯 Selection:     $filter"
echo

shopt -s nullglob
count=0
matched_any=false

for P in "$PIPELINES_DIR"/$filter; do
  [[ -d "$P" ]] || continue
  matched_any=true
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

  echo "🔄 Running 'uv sync'"
  (cd "$P" && uv sync)
  echo "✅ Done"
done

shopt -u nullglob

if ! $matched_any; then
  echo "⚠️  No folder matched '$PIPELINES_DIR/$filter'."
elif [[ $count -eq 0 ]]; then
  echo "⚠️  No pipelines processed (missing pyproject.toml?)."
else
  echo "✨ Completed for $count pipeline(s)."
fi
