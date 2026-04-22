FROM picsellia/cpu:python3.14

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/albumentations_processing/.venv/bin/activate' /experiment/run.sh

COPY ./uv.lock albumentations_processing/uv.lock
COPY ./pyproject.toml albumentations_processing/pyproject.toml

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.10) --project albumentations_processing

COPY ./ albumentations_processing

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.10", "albumentations_processing/pipeline.py"]
