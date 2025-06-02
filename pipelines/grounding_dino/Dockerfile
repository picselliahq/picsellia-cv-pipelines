FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx git wget curl unzip libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* \

RUN pip install -U pip setuptools uv

WORKDIR /experiment

ADD https://raw.githubusercontent.com/picselliahq/picsellia-cv-base-docker/main/base/run.sh /experiment/run.sh
RUN ln -s /experiment/run.sh /usr/bin/run && \
    chmod +x /experiment/run.sh && \
    chown -R 42420:42420 /experiment

ARG REBUILD_ALL
COPY ./ ./pipelines/grounding_dino
ARG REBUILD_PICSELLIA

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3) --project pipelines/grounding_dino

ENV PYTHONPATH="/experiment"
ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

ENTRYPOINT ["run", "python3", "pipelines/grounding_dino/picsellia_pipeline.py"]
