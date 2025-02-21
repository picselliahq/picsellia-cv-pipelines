FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

ARG REBUILD_ALL
COPY ./src/pipelines/yolov7_segmentation/requirements.txt ./src/pipelines/yolov7_segmentation/requirements.txt
ARG REBUILD_PICSELLIA

RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.10) --no-cache -r ./src/pipelines/yolov7_segmentation/requirements.txt

WORKDIR /experiment

COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src
COPY ./src/pipelines/yolov7_segmentation ./src/pipelines/yolov7_segmentation

RUN cd /experiment/src/pipelines/yolov7_segmentation && git clone https://github.com/WongKinYiu/yolov7 --branch u7

COPY ./src/pipelines/yolov7_segmentation/yolov7_changes/ /experiment/src/pipelines/yolov7_segmentation/yolov7/seg/

ENV PYTHONPATH=".:/experiment/src/pipelines/yolov7_segmentation/yolov7/seg:/experiment/src"

ENTRYPOINT ["run", "python3.10", "src/pipelines/yolov7_segmentation/training_pipeline.py"]
