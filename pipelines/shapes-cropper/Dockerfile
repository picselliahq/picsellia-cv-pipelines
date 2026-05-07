FROM picsellia/cpu:python3.14

RUN apt-get update && apt-get install -y \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/./shapes-cropper/.venv/bin/activate' /experiment/run.sh


ARG REBUILD_ALL
COPY ./ ./shapes-cropper
ARG REBUILD_PICSELLIA

RUN uv sync --python=$(which python3.14) --project shapes-cropper

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.14", "shapes-cropper/pipeline.py"]
