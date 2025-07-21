FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --python=$(which python3.10) git+https://github.com/picselliahq/picsellia-cv-engine.git@main

WORKDIR /experiment

ARG REBUILD_ALL
COPY ./ ./yolov8
ARG REBUILD_PICSELLIA

RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.10) --no-cache -r ./yolov8/training/requirements.txt
RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.10) --no-cache torch==2.2.1+cu118 torchaudio==2.2.1+cu118 torchvision==0.17.1+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html


ENV PYTHONPATH=":/experiment"

ENTRYPOINT ["run", "python3.10", "./yolov8/training/picsellia_pipeline.py"]
