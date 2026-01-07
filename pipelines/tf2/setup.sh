#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_MODELS_DIR="${ROOT_DIR}/models"

if [ ! -d "${TF_MODELS_DIR}/research/object_detection" ]; then
  rm -rf "${TF_MODELS_DIR}"
  git clone --depth 1 https://github.com/tensorflow/models.git "${TF_MODELS_DIR}"
fi

cd "${TF_MODELS_DIR}/research"

python -m pip install -q "grpcio-tools==1.59.3"

rm -f object_detection/protos/*_pb2.py object_detection/protos/*_pb2_grpc.py

python -m grpc_tools.protoc \
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
