#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Usage:
#   scripts/deploy_pipelines.sh --pipeline <name|all> \
#       [--pipeline-dir "<path>"] \
#       [--organization "<org>"] [--env "<ENV>"] \
#       [--token TOKEN] \
#       [--bump <patch|minor|major|rc|final>]
#
# Examples:
#   scripts/deploy_pipelines.sh --pipeline bounding_box_cropper \
#       --organization test-account --env STAGING --token "$PXL_API_TOKEN" --bump final
#
#   scripts/deploy_pipelines.sh --pipeline all \
#       --organization my-org --env PROD --token "$PXL_API_TOKEN" --bump patch
# ---------------------------------------------------------------------------

command -v pxl-pipeline >/dev/null || { echo "❌ 'pxl-pipeline' not found in PATH"; exit 1; }

PIPE="all"
# Same defaults as your test script for convenience
ORG="${ORGANIZATION:-test-account}"
ENV_NAME="${ENVIRONMENT:-STAGING}"
BUMP="${PIPELINE_BUMP:-final}"     # patch | minor | major | rc | final
TOKEN="${PXL_API_TOKEN:-}"         # can also be provided via --token
HF_TOKEN="${HF_TOKEN:-}"

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIPELINES_DIR_DEFAULT="$BASE_DIR/pipelines"
PIPELINES_DIR="$PIPELINES_DIR_DEFAULT"

# ----------------------
# Parse arguments
# ----------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline)      PIPE="${2:-all}"; shift 2 ;;
    --pipeline-dir)  PIPELINES_DIR="${2:-$PIPELINES_DIR_DEFAULT}"; shift 2 ;;
    --yes)           shift ;;  # kept for compatibility, no-op
    --organization)  ORG="${2:-}"; shift 2 ;;
    --env)           ENV_NAME="${2:-}"; shift 2 ;;
    --bump)          BUMP="${2:-}"; shift 2 ;;
    --token)         TOKEN="${2:-}"; shift 2 ;;
    --hf-token)      HF_TOKEN="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done

# ----------------------
# Validate bump value
# ----------------------
case "$BUMP" in
  patch|minor|major|rc|final) ;;
  *) echo "❌ Invalid bump value: '$BUMP' (must be patch|minor|major|rc|final)"; exit 2 ;;
esac

if [[ ! -d "$PIPELINES_DIR" ]]; then
  echo "❌ PIPELINES_DIR does not exist: $PIPELINES_DIR"
  exit 1
fi

filter="*"
if [[ "$PIPE" != "all" ]]; then
  filter="$PIPE"
fi

# ----------------------
# Optional non-interactive login
# ----------------------
if [[ -n "$TOKEN" ]]; then
  echo "🔐 Performing non-interactive login for deploy..."
  set +e
  if [[ -n "$ORG" && -n "$ENV_NAME" ]]; then
    pxl-pipeline login --organization "$ORG" --env "$ENV_NAME" --token "$TOKEN"
  else
    echo "⚠️  Token provided but organization or env missing; skipping login."
  fi
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    echo "❌ Login failed (organization / env / token invalid?)."
    exit 1
  fi
else
  echo "ℹ️  No token provided (neither --token nor PXL_API_TOKEN)."
  echo "    Deploy will use the existing CLI context (pxl-pipeline login)."
fi

echo "📦 Pipelines dir: $PIPELINES_DIR"
echo "🎯 Selection:     $filter"
echo "🏢 Organization:  ${ORG:-<current context>}"
echo "🌍 Environment:   ${ENV_NAME:-<current context>}"
echo "📌 Version bump:  $BUMP"

# ----------------------
# Deploy helper
# ----------------------
declare -a RESULTS
ANY_FAILURE=false
shopt -s nullglob
matched=false

deploy_one() {
  local PIPELINE_DIR="$1"
  local PIPELINE_NAME; PIPELINE_NAME="$(basename "$PIPELINE_DIR")"

  echo
  echo "🚀 Deploying: $PIPELINE_NAME"
  echo "   Folder:        $PIPELINE_DIR"
  [[ -n "$ORG"      ]] && echo "   Organization: $ORG"
  [[ -n "$ENV_NAME" ]] && echo "   Env:          $ENV_NAME"
  echo "   Version bump:  $BUMP"
  echo

  pushd "$PIPELINE_DIR" >/dev/null || {
    echo "⚠️  Failed to cd into $PIPELINE_DIR, skipping $PIPELINE_NAME"
    RESULTS+=("❌ $PIPELINE_NAME (cd failed)")
    ANY_FAILURE=true
    return
  }

  # If HF_TOKEN is provided, create a .env in the pipeline folder
  # so it gets included in the docker build context and baked into the final image.
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "🔐 Writing .env with HF token for $PIPELINE_NAME (will be baked into image)"
    {
      echo "HF_TOKEN=${HF_TOKEN}"
      echo "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"
    } > ".env"
  fi


  # Build command with optional flags
  CMD=(pxl-pipeline deploy "$PIPELINE_NAME" --bump "$BUMP")
  [[ -n "$ORG"      ]] && CMD+=(--organization "$ORG")
  [[ -n "$ENV_NAME" ]] && CMD+=(--env "$ENV_NAME")

  echo "▶️  ${CMD[*]}"
  if "${CMD[@]}"; then
    echo "✅ Deployed: $PIPELINE_NAME"
    RESULTS+=("✅ $PIPELINE_NAME")
  else
    echo "❌ Deployment failed: $PIPELINE_NAME"
    RESULTS+=("❌ $PIPELINE_NAME (deploy)")
    ANY_FAILURE=true
  fi

  popd >/dev/null

  if [[ -n "${HF_TOKEN:-}" ]]; then
    rm -f ".env"
  fi

}

# ----------------------
# Selection + deploy
# ----------------------
for P in "$PIPELINES_DIR"/$filter; do
  [[ -d "$P" && -f "$P/pyproject.toml" ]] || continue
  matched=true
  deploy_one "$P"
done
shopt -u nullglob

if ! $matched; then
  echo "⚠️  No pipeline matched '$PIPELINES_DIR/$filter' (or no pyproject.toml found)."
fi

# ----------------------
# Final summary + exit code
# ----------------------
echo
echo "📊 Deploy summary:"
if ((${#RESULTS[@]})); then
  for r in "${RESULTS[@]}"; do
    echo "$r"
  done
else
  echo "— no deployments performed —"
fi

if $ANY_FAILURE; then
  echo
  echo "❌ At least one deployment failed."
  exit 1
else
  echo
  echo "✅ All deployments succeeded."
  exit 0
fi
