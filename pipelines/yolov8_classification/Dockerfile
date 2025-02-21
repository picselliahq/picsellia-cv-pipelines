FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

ARG REBUILD_ALL
COPY ./src/pipelines/yolov8_classification/requirements.txt ./src/pipelines/yolov8_classification/requirements.txt
ARG REBUILD_PICSELLIA

RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.10) --no-cache -r ./src/pipelines/yolov8_classification/requirements.txt
RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.10) --no-cache torch==2.2.1+cu118 torchaudio==2.2.1+cu118 torchvision==0.17.1+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html

WORKDIR /experiment

COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src
COPY ./src/pipelines/yolov8_classification ./src/pipelines/yolov8_classification

ENV PYTHONPATH=":/experiment/src"

ENTRYPOINT ["run", "python3.10", "src/pipelines/yolov8_classification/training_pipeline.py"]
