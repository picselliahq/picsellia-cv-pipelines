#!/usr/bin/env bash
set -euo pipefail

ALL_JSON="${ALL_JSON:-}"

if [[ -z "$ALL_JSON" || "$ALL_JSON" == "[]" ]]; then
  echo "processing_pipelines=[]" >> "$GITHUB_OUTPUT"
  echo "training_pipelines=[]" >> "$GITHUB_OUTPUT"
  exit 0
fi

python << 'PY' >> "$GITHUB_OUTPUT"
import json
import os
import pathlib

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

raw = os.environ.get("ALL_JSON", "")
if not raw:
    processing = []
    training = []
else:
    pipelines = json.loads(raw)
    processing = []
    training = []

    for p in pipelines:
        if "/" in p:
            top, sub = p.split("/", 1)
            cfg_path = pathlib.Path("pipelines") / top / sub / "config.toml"
        else:
            cfg_path = pathlib.Path("pipelines") / p / "config.toml"

        ptype = "UNKNOWN"
        if cfg_path.is_file():
            try:
                with cfg_path.open("rb") as f:
                    data = tomllib.load(f)
                meta = data.get("metadata") or {}
                ptype = meta.get("type", "UNKNOWN")
            except Exception:
                ptype = "UNKNOWN"

        if ptype == "TRAINING":
            training.append(p)
        else:
            processing.append(p)

print(f"processing_pipelines={json.dumps(processing)}")
print(f"training_pipelines={json.dumps(training)}")
PY
