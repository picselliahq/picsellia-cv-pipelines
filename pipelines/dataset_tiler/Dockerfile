FROM picsellia/cpu:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/./dataset_tiler/.venv/bin/activate' /experiment/run.sh

ARG REBUILD_ALL
COPY ./ ./dataset_tiler
ARG REBUILD_PICSELLIA

RUN uv sync --python=$(which python3.10) --project dataset_tiler

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.10", "dataset_tiler/pipeline.py"]
