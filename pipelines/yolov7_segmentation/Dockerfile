FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --python=$(which python3.10) git+https://github.com/picselliahq/picsellia-cv-engine.git@main

ARG REBUILD_ALL
COPY ./ ./yolov7_segmentation
ARG REBUILD_PICSELLIA

RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.10) --no-cache -r ./yolov7_segmentation/requirements.txt

WORKDIR /experiment

RUN cd /experiment/yolov7_segmentation && git clone https://github.com/WongKinYiu/yolov7 --branch u7

COPY ./yolov7_segmentation/yolov7_changes/ /experiment/yolov7_segmentation/yolov7/seg/

ENV PYTHONPATH=".:/experiment/yolov7_segmentation/yolov7/seg:/experiment"

ENTRYPOINT ["run", "python3.10", "yolov7_segmentation/picsellia_pipeline.py"]
