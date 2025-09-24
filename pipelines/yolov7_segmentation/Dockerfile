FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/./yolov7_segmentation/.venv/bin/activate' /experiment/run.sh

ARG REBUILD_ALL
COPY ./ ./yolov7_segmentation
ARG REBUILD_PICSELLIA

RUN uv sync --python=$(which python3.10) --project yolov7_segmentation

RUN bash yolov7_segmentation/setup.sh

ENV PYTHONPATH=".:/experiment/yolov7_segmentation/yolov7/seg:/experiment"

ENTRYPOINT ["run", "python3.10", "yolov7_segmentation/pipeline.py"]
