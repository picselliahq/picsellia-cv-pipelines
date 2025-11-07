#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/deploy_pipelines.sh --pipeline <name|all> [--pipeline-dir "<path>"] [--yes] [--organization "<org>"] [--env "<ENV>"]
# Examples:
#   scripts/deploy_pipelines.sh --pipeline bounding_box_cropper --yes --organization test-account --env STAGING
#   scripts/deploy_pipelines.sh --pipeline all --yes --organization my-org --env PROD

command -v pxl-pipeline >/dev/null || { echo "❌ pxl-pipeline introuvable dans le PATH"; exit 1; }

PIPE="all"
ASSUME_YES=false
ORG=""
ENV_NAME=""

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIPELINES_DIR_DEFAULT="$BASE_DIR/pipelines"
PIPELINES_DIR="$PIPELINES_DIR_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline)      PIPE="${2:-all}"; shift 2 ;;
    --pipeline-dir)  PIPELINES_DIR="${2:-$PIPELINES_DIR_DEFAULT}"; shift 2 ;;
    --yes)           ASSUME_YES=true; shift ;;
    --organization)  ORG="${2:-}"; shift 2 ;;
    --env)           ENV_NAME="${2:-}"; shift 2 ;;
    *) echo "Arg inconnu: $1"; exit 2 ;;
  esac
done

[[ -d "$PIPELINES_DIR" ]] || { echo "❌ PIPELINES_DIR inexistant: $PIPELINES_DIR"; exit 1; }

filter="*"; [[ "$PIPE" != "all" ]] && filter="$PIPE"

confirm() {
  $ASSUME_YES && return 0
  read -rp "⚠️  Confirmer le déploiement ? (y/N) " ans
  [[ "${ans,,}" == "y" || "${ans,,}" == "yes" ]]
}

deploy_one() {
  local PIPELINE_DIR="$1"
  local PIPELINE_NAME; PIPELINE_NAME="$(basename "$PIPELINE_DIR")"

  echo
  echo "🚀 Déploiement: $PIPELINE_NAME"
  echo "   Dossier:    $PIPELINE_DIR"
  [[ -n "$ORG"      ]] && echo "   Organization: $ORG"
  [[ -n "$ENV_NAME" ]] && echo "   Env:          $ENV_NAME"
  echo

  if confirm; then
    pushd "$PIPELINE_DIR" >/dev/null

    # Construire la commande avec options facultatives
    CMD=(pxl-pipeline deploy "$PIPELINE_NAME")
    [[ -n "$ORG"      ]] && CMD+=(--organization "$ORG")
    [[ -n "$ENV_NAME" ]] && CMD+=(--env "$ENV_NAME")   # si votre CLI utilise --environment, remplacez ici

    echo "▶️  ${CMD[*]}"
    "${CMD[@]}"

    popd >/dev/null
    echo "✅ Déployé: $PIPELINE_NAME"
  else
    echo "⏩ Skip (non confirmé): $PIPELINE_NAME"
  fi
}

echo "📦 Pipelines dir: $PIPELINES_DIR"
echo "🎯 Sélection:     $filter"

shopt -s nullglob
matched=false
for P in "$PIPELINES_DIR"/$filter; do
  [[ -d "$P" && -f "$P/pyproject.toml" ]] || continue
  matched=true
  deploy_one "$P"
done
shopt -u nullglob

if ! $matched; then
  echo "⚠️  Aucune pipeline ne matche '$PIPELINES_DIR/$filter' (ou pas de pyproject.toml)."
fi
