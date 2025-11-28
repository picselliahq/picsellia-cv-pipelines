#!/usr/bin/env bash
set -euo pipefail

# Reads changed files from STDIN and uses CHANGED_FLAG env
# to decide whether to build a JSON matrix of pipelines.
#
# Expects:
#   - env CHANGED_FLAG = "true" | "false"
#   - stdin = newline-separated list of files (from dorny pipelines_files)
#
# Writes:
#   - pipelines=<json_array> into $GITHUB_OUTPUT

CHANGED_FLAG="${CHANGED_FLAG:-false}"

echo "CHANGED_FLAG=$CHANGED_FLAG"
echo "Reading changed files from stdin..."

if [[ "$CHANGED_FLAG" != "true" ]]; then
  echo "No changes under pipelines/ or tests/ (flag is not 'true')."
  echo 'pipelines=[]' >> "$GITHUB_OUTPUT"
  exit 0
fi

declare -A PIPELINE_SET=()

while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  echo "  file: $file"

  # pipelines/<top>/...
  if [[ "$file" == pipelines/* ]]; then
    rest="${file#pipelines/}"
    top="${rest%%/*}"
    [[ -n "$top" ]] && PIPELINE_SET["$top"]=1
  fi

  # tests/<top>/...
  if [[ "$file" == tests/* ]]; then
    rest="${file#tests/}"
    top="${rest%%/*}"
    [[ -n "$top" ]] && PIPELINE_SET["$top"]=1
  fi
done

entries=()
for name in "${!PIPELINE_SET[@]}"; do
  entries+=( "\"$name\"" )
done

if [ ${#entries[@]} -eq 0 ]; then
  echo "No pipelines deduced from changed files."
  echo 'pipelines=[]' >> "$GITHUB_OUTPUT"
  exit 0
fi

json="[$(IFS=,; echo "${entries[*]}")]"
echo "Computed pipelines JSON: $json"
echo "pipelines=$json" >> "$GITHUB_OUTPUT"
