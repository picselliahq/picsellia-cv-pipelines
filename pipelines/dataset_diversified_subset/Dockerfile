FROM picsellia/cpu:python3.14

RUN apt-get update && apt-get install -y \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/dataset_diversified_subset/.venv/bin/activate' /experiment/run.sh

COPY ./uv.lock dataset_diversified_subset/uv.lock
COPY ./pyproject.toml dataset_diversified_subset/pyproject.toml

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.14) --project dataset_diversified_subset

COPY ./ dataset_diversified_subset

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.14", "dataset_diversified_subset/pipeline.py"]
