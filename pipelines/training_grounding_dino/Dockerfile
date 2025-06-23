FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools uv

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment
RUN sed -i '1 a source /experiment/./training_grounding_dino/.venv/bin/activate' /experiment/run.sh
RUN ln -s /experiment/run.sh /usr/bin/run && \
    chmod +x /experiment/run.sh && \
    chown -R 42420:42420 /experiment

ARG REBUILD_ALL
COPY ./ ./training_grounding_dino
ARG REBUILD_PICSELLIA

RUN git clone https://github.com/longzw1997/Open-GroundingDino.git ./training_grounding_dino/Open-GroundingDino

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.10) --project ./training_grounding_dino

ENV PYTHONPATH=":/experiment"

ENTRYPOINT ["run", "python3.10", "./training_grounding_dino/picsellia_pipeline.py"]
