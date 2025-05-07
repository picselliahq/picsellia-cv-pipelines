FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.11

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --python=$(which python3.11) git+https://github.com/picselliahq/picsellia-cv-engine.git@main

ARG REBUILD_ALL
COPY ./pipelines/diversified_dataset_extractor/requirements.txt ./pipelines/diversified_dataset_extractor/requirements.txt
ARG REBUILD_PICSELLIA

RUN uv pip install --python=$(which python3.11) --no-cache -r ./pipelines/diversified_dataset_extractor/requirements.txt

WORKDIR /experiment

RUN mkdir -p /workspace
RUN chmod 777 /workspace

ENV PYTHONPATH=":/experiment"

ENTRYPOINT ["run", "python3.11", "pipelines/diversified_dataset_extractor/processing_pipeline.py"]
