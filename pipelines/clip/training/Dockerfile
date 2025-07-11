FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

ENV HF_HOME=/experiment/.cache/huggingface
RUN mkdir -p $HF_HOME

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment
RUN sed -i '1 a source /experiment/training/.venv/bin/activate' /experiment/run.sh

ARG REBUILD_ALL
COPY ./ training
ARG REBUILD_PICSELLIA

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.10) --project training

ENV PYTHONPATH=":/experiment"

ENTRYPOINT ["run", "python3.10", "training/picsellia_pipeline.py"]
