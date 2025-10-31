FROM picsellia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04-python3.11

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

RUN bash fine_tuning/setup.sh

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.11) --project fine_tuning

ENV PYTHONPATH="/experiment/fine_tuning/sam2:/experiment"

ENTRYPOINT ["run", "python3.11", "fine_tuning/pipeline.py"]
