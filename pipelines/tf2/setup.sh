#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_MODELS_DIR="${ROOT_DIR}/models"

# Prefer the project's uv venv python if it exists
PYTHON="python"
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
fi

ensure_pip() {
  if "$PYTHON" -c "import pip" >/dev/null 2>&1; then
    return 0
  fi

  echo "⚙️  pip not found for $PYTHON — bootstrapping with ensurepip..."
  if "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
    :
  else
    echo "⚠️  ensurepip unavailable; trying to proceed anyway."
  fi

  if "$PYTHON" -c "import pip" >/dev/null 2>&1; then
    "$PYTHON" -m pip install --upgrade pip >/dev/null 2>&1 || true
    return 0
  fi

  echo "❌ Could not install pip for $PYTHON"
  exit 1
}

patch_tf_slim() {
  echo "🩹 Patching tf_slim to be compatible with TF >= 2.14 (control_flow_ops.case/cond)..."

  TFSLIM_FILE="$("$PYTHON" - <<'PY'
import os
import tf_slim
base = os.path.dirname(tf_slim.__file__)
print(os.path.join(base, "data", "tfexample_decoder.py"))
PY
)"

  if [[ ! -f "$TFSLIM_FILE" ]]; then
    echo "⚠️  tf_slim file not found at: $TFSLIM_FILE"
    echo "    Skipping patch."
    return 0
  fi

  if grep -q "control_flow_ops_cond" "$TFSLIM_FILE" && grep -q "control_flow_case" "$TFSLIM_FILE"; then
    echo "✅ tf_slim already patched: $TFSLIM_FILE"
    return 0
  fi

  # Pass path via env var to avoid heredoc interpolation issues
  TFSLIM_FILE="$TFSLIM_FILE" "$PYTHON" - <<'PY'
from __future__ import annotations
import os
from pathlib import Path

path_str = os.environ["TFSLIM_FILE"]
path = Path(path_str)

txt = path.read_text(encoding="utf-8")

# Replace imports
if "from tensorflow.python.ops import control_flow_ops" in txt:
    txt = txt.replace(
        "from tensorflow.python.ops import control_flow_ops",
        "from tensorflow.python.ops import cond as control_flow_ops_cond\n"
        "from tensorflow.python.ops import control_flow_case",
    )

# Replace call sites
txt = txt.replace("control_flow_ops.cond(", "control_flow_ops_cond.cond(")
txt = txt.replace("control_flow_ops.case(", "control_flow_case.case(")

path.write_text(txt, encoding="utf-8")
print(f"patched: {path}")
PY

  echo "✅ Patched tf_slim file: $TFSLIM_FILE"
}


if [ ! -d "${TF_MODELS_DIR}/research/object_detection" ]; then
  rm -rf "${TF_MODELS_DIR}"
  git clone --depth 1 https://github.com/tensorflow/models.git "${TF_MODELS_DIR}"
fi

cd "${TF_MODELS_DIR}/research"

ensure_pip

# (optional) Install whatever you need in the env
"$PYTHON" -m pip install -q "grpcio-tools==1.59.3"

# ✅ Patch tf_slim *after* deps are installed in the venv
patch_tf_slim

rm -f object_detection/protos/*_pb2.py object_detection/protos/*_pb2_grpc.py

"$PYTHON" -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  object_detection/protos/*.proto

echo "---- sanity check (should NOT mention runtime_version / Protobuf 6) ----"
head -n 15 object_detection/protos/anchor_generator_pb2.py || true
echo "-----------------------------------------------------------------------"

cd "${ROOT_DIR}"
rm -rf "${ROOT_DIR}/object_detection" "${ROOT_DIR}/slim" "${ROOT_DIR}/official"
cp -R "${TF_MODELS_DIR}/research/object_detection" "${ROOT_DIR}/object_detection"
cp -R "${TF_MODELS_DIR}/research/slim" "${ROOT_DIR}/slim"
cp -R "${TF_MODELS_DIR}/official" "${ROOT_DIR}/official"

echo "✅ setup.sh OK (grpcio-tools 1.59.3 + tf_slim patch)"
