#!/usr/bin/env bash
set -euo pipefail

echo "Reading changed files from stdin..."

declare -A PIPELINE_SET=()

# We may get:
# - one path per line
# - OR one line with multiple paths separated by spaces
# So: read line by line, then split each line into tokens.
while IFS= read -r line; do
  [[ -z "$line" ]] && continue

  for file in $line; do
    [[ -z "$file" ]] && continue

    # Case 1: pipelines/<top>/...
    if [[ "$file" == pipelines/* ]]; then
      rest="${file#pipelines/}"
      top="${rest%%/*}"
      [[ -n "$top" ]] && PIPELINE_SET["$top"]=1
    fi

    # Case 2: tests/<top>/...
    if [[ "$file" == tests/* ]]; then
      rest="${file#tests/}"
      top="${rest%%/*}"
      [[ -n "$top" ]] && PIPELINE_SET["$top"]=1
    fi
  done
done

if [ ${#PIPELINE_SET[@]} -eq 0 ]; then
  echo "No pipelines deduced from changed files."
  echo 'pipelines=[]' >> "$GITHUB_OUTPUT"
  exit 0
fi

entries=()
for name in "${!PIPELINE_SET[@]}"; do
  entries+=( "\"$name\"" )
done

json="[$(IFS=,; echo "${entries[*]}")]"
echo "Computed pipelines JSON: $json"
echo "pipelines=$json" >> "$GITHUB_OUTPUT"
