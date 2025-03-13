FROM picsellia/cpu:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY ./src/pipelines/augmentations_pipeline/requirements.txt ./src/pipelines/augmentations_pipeline/requirements.txt

RUN  uv pip install --python=$(which python3.10) --no-cache -r ./src/pipelines/augmentations_pipeline/requirements.txt

WORKDIR /experiment

COPY ./examples ./examples
COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src
COPY ./src/pipelines/augmentations_pipeline ./src/pipelines/augmentations_pipeline

ENV PYTHONPATH=":/experiment/src"

ENTRYPOINT ["run", "python3.10", "src/pipelines/augmentations_pipeline/processing_pipeline.py"]
