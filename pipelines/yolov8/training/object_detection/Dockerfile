FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone -b feat/add-yolo-dataset-loader https://github.com/picselliahq/picsellia-cv-engine.git
RUN git clone https://github.com/picselliahq/picsellia-pipelines-cli.git
RUN uv pip install --python=$(which python3.10) -e /picsellia-cv-engine

WORKDIR /experiment

ARG REBUILD_ALL
COPY ./pipelines/yolov8/training/pipeline_utils ./pipelines/yolov8/training/pipeline_utils
COPY ./pipelines/yolov8/training/object_detection ./pipelines/yolov8/training/object_detection
ARG REBUILD_PICSELLIA

RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.10) --no-cache -r ./pipelines/yolov8/training/object_detection/requirements.txt
RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.10) --no-cache torch==2.2.1+cu118 torchaudio==2.2.1+cu118 torchvision==0.17.1+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html


ENV PYTHONPATH=":/experiment"

ENTRYPOINT ["run", "python3.10", "pipelines/yolov8/training/object_detection/training_pipeline.py"]
