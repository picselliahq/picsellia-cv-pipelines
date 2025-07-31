FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment
RUN sed -i '1 a source /experiment/fine_tuning/.venv/bin/activate' /experiment/run.sh

ARG REBUILD_ALL
COPY ./ fine_tuning
ARG REBUILD_PICSELLIA

RUN git clone https://github.com/facebookresearch/sam2.git /experiment/fine_tuning/sam2 \
    && chmod -R a+w /experiment/fine_tuning/sam2

COPY train.yaml /experiment/fine_tuning/sam2/sam2/configs/train.yaml
COPY train.py /experiment/fine_tuning/sam2/training/train.py

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.10) --project fine_tuning

ENV PYTHONPATH=":/experiment"

ENTRYPOINT ["run", "python3.10", "fine_tuning/picsellia_pipeline.py"]
