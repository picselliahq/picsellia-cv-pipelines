FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.11

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

ARG REBUILD_ALL
COPY ./pipelines/datalake_autotagging/requirements.txt ./pipelines/datalake_autotagging/requirements.txt
ARG REBUILD_PICSELLIA

RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.11) --no-cache -r ./pipelines/datalake_autotagging/requirements.txt

WORKDIR /experiment

COPY ./pipelines/datalake_autotagging ./pipelines/datalake_autotagging

RUN mkdir -p /workspace
RUN chmod 777 /workspace

ENV PYTHONPATH=":/experiment"

ENTRYPOINT ["run", "python3.11", "pipelines/datalake_autotagging/processing_pipeline.py"]
