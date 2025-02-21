#!/bin/bash

export host=https://staging.picsellia.com
export organization_id=0189266d-64f5-79b4-9b80-6fcf8a3b498b
pip install picsellia --upgrade

if [ $# -eq 0 ]; then
  echo "specify one model to test or all"
  exit 1
fi

if [ "$1" = "yolov8" ]
then
  echo "Running tests for YOLOV8 Object Detecton.."
  pip install -r "yolov8-detection/requirements.txt"
  python -m unittest -v "yolov8-detection/tests.py"

  echo "Running tests for YOLOV8 Classification.."
  pip install -r "yolov8-classification/requirements.txt"
  python -m unittest -v "yolov8-classification/tests.py"

  echo "Running tests for YOLOV8 Segmentation.."
  pip install -r "yolov8-segmentation/requirements.txt"
  python -m unittest -v "yolov8-segmentation/tests.py"

elif [ "$1" = "yolov8-detection" ]
then
  echo "Running tests for YOLOV8 Object Detecton.."
  pip install -r "yolov8-detection/requirements.txt"
  python -m unittest -v "yolov8-detection/tests.py"

elif [ "$1" = "yolov8-segmentation" ]
then
  echo "Running tests for YOLOV8 Segmentation.."
  pip install -r "yolov8-segmentation/requirements.txt"
  python -m unittest -v "yolov8-segmentation/tests.py"

elif [ "$1" = "yolov8-classification" ]
then
  echo "Running tests for YOLOV8 classification.."
  pip install -r "yolov8-classification/requirements.txt"
  python -m unittest -v "yolov8-classification/tests.py"

elif [ "$1" = "unet" ]
then
  echo "Running tests for UNET.."
  export SM_FRAMEWORK="tf.keras"
  pip install -r "unet-instance-segmentation/requirements.txt"
  python -m unittest -v "unet-instance-segmentation/tests.py"

elif [ "$1" = "utils" ]
then
  echo "Running tests for core utils.."
  pip install -r "core_utils/requirements.txt"
  python -m unittest -v "core_utils/tests.py"

elif [ "$1" = "all" ]
then
  echo "Running all unit tests.."
  requirements=("core_utils/requirements.txt" "yolov8-classification/requirements.txt" "yolov8-detection/requirements.txt" "yolov8-segmentation/requirements.txt" "unet-instance-segmentation/requirements.txt")
  for requirement_file in "${requirements[@]}"; do
    if [ -f "$requirement_file" ]; then
      echo "Installing packages from $requirement_file"
      pip install -r "$requirement_file"
    else
      echo "File not found: $requirement_file"
    fi
  done
  python -m unittest -v "core_utils/tests.py"
  python -m unittest -v "yolov8-classification/tests.py"
  python -m unittest -v "yolov8-detection/tests.py"
  python -m unittest -v "yolov8-segmentation/tests.py"
  python -m unittest -v "unet-instance-segmentation/tests.py"

fi
