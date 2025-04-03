FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone -b feat/add-utils-for-training-template https://github.com/picselliahq/picsellia-cv-engine.git /picsellia-cv-engine
RUN git clone https://github.com/picselliahq/picsellia-pipelines-cli.git
RUN uv pip install --python=$(which python3.10) -e /picsellia-cv-engine

WORKDIR /experiment

ARG REBUILD_ALL
COPY ./pipelines/yolov8 ./pipelines/yolov8
ARG REBUILD_PICSELLIA

RUN uv pip install --python=$(which python3.10) --no-cache -r ./pipelines/yolov8/pre_annotation/requirements.txt

ENV PYTHONPATH=":/experiment"

ENTRYPOINT ["run", "python3.10", "pipelines/yolov8/pre_annotation/processing_pipeline.py"]
