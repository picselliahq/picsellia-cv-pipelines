#!/usr/bin/env bash
set -euo pipefail

echo "All changed pipelines (JSON): ${ALL_JSON:-}"

python - << 'PY' >> "$GITHUB_OUTPUT"
import json
import os
import pathlib

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

raw = os.environ.get("ALL_JSON", "")
if not raw:
    all_pipelines = []
else:
    all_pipelines = json.loads(raw)

cpu_processing = []
gpu_processing = []
training = []

for p in all_pipelines:
    if "/" in p:
        top, sub = p.split("/", 1)
        cfg_path = pathlib.Path("pipelines") / top / sub / "config.toml"
    else:
        cfg_path = pathlib.Path("pipelines") / p / "config.toml"

    ptype = "UNKNOWN"
    gpu_count = 0

    if cfg_path.is_file():
        try:
            with cfg_path.open("rb") as f:
                data = tomllib.load(f)
            meta = data.get("metadata") or {}
            ptype = meta.get("type", "UNKNOWN")

            docker_cfg = data.get("docker") or {}
            gpu_raw = docker_cfg.get("gpu", 0)

            if isinstance(gpu_raw, str):
                try:
                    gpu_count = int(gpu_raw)
                except ValueError:
                    gpu_count = 0
            elif isinstance(gpu_raw, (int, float)):
                gpu_count = int(gpu_raw)
            else:
                gpu_count = 0
        except Exception:
            pass

    if ptype == "TRAINING":
        training.append(p)
    else:
        if gpu_count > 0:
            gpu_processing.append(p)
        else:
            cpu_processing.append(p)

def emit(name, value):
    print(f"{name}={json.dumps(value)}")

emit("processing_pipelines", cpu_processing)
emit("training_pipelines", training)
emit("gpu_processing_pipelines", gpu_processing)
PY
