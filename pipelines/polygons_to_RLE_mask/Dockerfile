FROM picsellia/cpu:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/polygons_to_RLE_mask/.venv/bin/activate' /experiment/run.sh

COPY ./uv.lock polygons_to_RLE_mask/uv.lock
COPY ./pyproject.toml polygons_to_RLE_mask/pyproject.toml

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.10) --project polygons_to_RLE_mask

COPY ./ polygons_to_RLE_mask

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.10", "polygons_to_RLE_mask/pipeline.py"]
