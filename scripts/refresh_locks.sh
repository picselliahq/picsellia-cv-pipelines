#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/refresh_locks.sh --pipeline <name|all> [--pipeline-dir "<path>"]
# Ex:
#   scripts/refresh_locks.sh --pipeline bounding_box_cropper
#   scripts/refresh_locks.sh --pipeline all
#   scripts/refresh_locks.sh --pipeline clip --pipeline-dir "/tmp/my-pipelines"

command -v uv >/dev/null || { echo "❌ uv introuvable"; exit 1; }

PIPE="all"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIPELINES_DIR_DEFAULT="$BASE_DIR/pipelines"
PIPELINES_DIR="$PIPELINES_DIR_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline) PIPE="${2:-all}"; shift 2 ;;
    --pipeline-dir) PIPELINES_DIR="${2:-$PIPELINES_DIR_DEFAULT}"; shift 2 ;;
    *) echo "Arg inconnu: $1"; exit 2 ;;
  esac
done

[[ -d "$PIPELINES_DIR" ]] || { echo "❌ PIPELINES_DIR inexistant: $PIPELINES_DIR"; exit 1; }

filter="*"; [[ "$PIPE" != "all" ]] && filter="$PIPE"
echo "📦 Pipelines dir: $PIPELINES_DIR"
echo "🎯 Sélection:     $filter"
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
    echo "⏭️  SKIP: pas de pyproject.toml à la racine."
    continue
  fi

  ((++count))
  if [[ -f "$P/uv.lock" ]]; then
    rm -f "$P/uv.lock"
    echo "🧹 uv.lock supprimé"
  else
    echo "ℹ️  pas de uv.lock"
  fi

  echo "🔄 uv sync"
  (cd "$P" && uv sync)
  echo "✅ done"
done
shopt -u nullglob

if ! $matched_any; then
  echo "⚠️  Aucun dossier ne matche '$PIPELINES_DIR/$filter'."
elif [[ $count -eq 0 ]]; then
  echo "⚠️  Aucune pipeline traitée (pyproject.toml manquant ?)."
else
  echo "✨ Terminé pour $count pipeline(s)."
fi
