FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./yolov5-detection/requirements.txt .

ARG REBUILD_ALL
ARG REBUILD_PICSELLIA

RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.10) -r ./requirements.txt --no-cache-dir

WORKDIR /experiment

RUN git clone https://github.com/ultralytics/yolov5

COPY ./yolov5-detection/experiment/yolov5_utils/train.py /experiment/yolov5/train.py

COPY ./yolov5-detection/experiment ./
COPY ./core_utils ./core_utils

ENV PYTHONPATH="/experiment/yolov5/:/experiment/yolov5/utils:${PYTHONPATH}"

RUN chown -R 42420:42420 /experiment

ENTRYPOINT ["run", "python3.10", "main.py"]
