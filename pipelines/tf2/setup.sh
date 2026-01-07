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
  # pip import check
  if "$PYTHON" -c "import pip" >/dev/null 2>&1; then
    return 0
  fi

  echo "⚙️  pip not found for $PYTHON — bootstrapping with ensurepip..."
  # ensurepip may not exist in some minimal builds, so fallback safely
  if "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
    :
  else
    echo "⚠️  ensurepip unavailable; trying to proceed anyway."
  fi

  # Try upgrading pip if it now exists
  if "$PYTHON" -c "import pip" >/dev/null 2>&1; then
    "$PYTHON" -m pip install --upgrade pip >/dev/null 2>&1 || true
    return 0
  fi

  echo "❌ Could not install pip for $PYTHON"
  exit 1
}

if [ ! -d "${TF_MODELS_DIR}/research/object_detection" ]; then
  rm -rf "${TF_MODELS_DIR}"
  git clone --depth 1 https://github.com/tensorflow/models.git "${TF_MODELS_DIR}"
fi

cd "${TF_MODELS_DIR}/research"

ensure_pip

"$PYTHON" -m pip install -q "grpcio-tools==1.59.3"

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

echo "✅ setup.sh OK (grpcio-tools 1.59.3)"
