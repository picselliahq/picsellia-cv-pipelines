FROM picsellia/cpu:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --python=$(which python3.10) git+https://github.com/picselliahq/picsellia-cv-engine.git@main

ARG REBUILD_ALL
COPY ./pipelines/dataset_tiler/requirements.txt ./pipelines/dataset_tiler/requirements.txt
ARG REBUILD_PICSELLIA

RUN uv pip install --python=$(which python3.10) --no-cache -r ./pipelines/dataset_tiler/requirements.txt

WORKDIR /experiment

COPY ./pipelines/dataset_tiler ./pipelines/dataset_tiler

ENV PYTHONPATH=":/experiment"

ENTRYPOINT ["run", "python3.10", "pipelines/dataset_tiler/processing_pipeline.py"]
