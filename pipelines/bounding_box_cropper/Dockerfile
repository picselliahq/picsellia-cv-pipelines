FROM picsellia/cpu:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN git clone -b feat/add-base-scripts https://github.com/picselliahq/picsellia-cv-engine.git /picsellia-cv-engine
RUN pip install -e /picsellia-cv-engine

WORKDIR /experiment

ARG REBUILD_ALL
COPY ./pipelines/bounding_box_cropper ./pipelines/bounding_box_cropper
ARG REBUILD_PICSELLIA

RUN uv pip install --python=$(which python3.10) --no-cache -r ./pipelines/bounding_box_cropper/requirements.txt

ENV PYTHONPATH=":/experiment"

ENTRYPOINT ["run", "python3.10", "pipelines/bounding_box_cropper/processing_pipeline.py"]
