#!/usr/bin/env bash
set -euo pipefail

echo "All changed pipelines (JSON): ${ALL_JSON:-}"

python - << 'PY' >> "$GITHUB_OUTPUT"
import json
import os
import pathlib

# tomllib for Python 3.11+, tomli fallback otherwise
try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

raw = os.environ.get("ALL_JSON", "")
if not raw:
    all_pipelines = []
else:
    all_pipelines = json.loads(raw)

# GPU-only processings (by name, e.g. "grounding_dino", "foo/bar" possible)
gpu_only_raw = os.environ.get("GPU_ONLY", "")
gpu_only = {name.strip() for name in gpu_only_raw.split(",") if name.strip()}

processing_cpu = []
processing_gpu = []
training = []

for p in all_pipelines:
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
        if p in gpu_only:
            processing_gpu.append(p)
        else:
            processing_cpu.append(p)

print(f"processing_pipelines={json.dumps(processing_cpu)}")
print(f"training_pipelines={json.dumps(training)}")
print(f"gpu_processing_pipelines={json.dumps(processing_gpu)}")
PY
