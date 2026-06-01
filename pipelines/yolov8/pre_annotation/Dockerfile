FROM picsellia/cuda:12.6.3-cudnn-ubuntu24.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/pre_annotation/.venv/bin/activate' /experiment/run.sh

COPY ./uv.lock pre_annotation/uv.lock
COPY ./pyproject.toml pre_annotation/pyproject.toml

RUN uv sync --python=$(which python3.10) --project pre_annotation

COPY ./ pre_annotation

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.10", "pre_annotation/pipeline.py"]
