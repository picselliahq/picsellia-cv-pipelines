FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools uv poetry

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker && \
    cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/grounding_dino/.venv/bin/activate' /experiment/run.sh

RUN ln -s /experiment/run.sh /usr/bin/run && \
    chmod +x /experiment/run.sh && \
    chown -R 42420:42420 /experiment

ARG REBUILD_ALL
COPY ./ ./grounding_dino
ARG REBUILD_PICSELLIA

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.10) --project grounding_dino
RUN uv add --python=$(which python3.10) git+https://github.com/IDEA-Research/GroundingDINO.git --project grounding_dino --no-build-isolation

ENV PYTHONPATH="/experiment"
ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

ENTRYPOINT ["run", "python3.10", "grounding_dino/picsellia_pipeline.py"]
