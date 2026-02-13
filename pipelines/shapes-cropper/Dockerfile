FROM picsellia/cpu:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/./bounding_box_cropper/.venv/bin/activate' /experiment/run.sh


ARG REBUILD_ALL
COPY ./ ./bbox-or-polygon-cropper
ARG REBUILD_PICSELLIA

RUN uv sync --python=$(which python3.10) --project bbox-or-polygon-cropper

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.10", "bbox-or-polygon-cropper/pipeline.py"]
