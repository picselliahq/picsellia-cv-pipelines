FROM picsellia/cpu:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

ARG REBUILD_ALL
COPY ./src/pipelines/bounding_box_cropper/requirements.txt ./src/pipelines/bounding_box_cropper/requirements.txt
ARG REBUILD_PICSELLIA

RUN uv pip install --python=$(which python3.10) --no-cache -r ./src/pipelines/bounding_box_cropper/requirements.txt

WORKDIR /experiment

COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src
COPY ./src/pipelines/bounding_box_cropper ./src/pipelines/bounding_box_cropper

ENV PYTHONPATH=":/experiment/src"

ENTRYPOINT ["run", "python3.10", "src/pipelines/bounding_box_cropper/processing_pipeline.py"]
