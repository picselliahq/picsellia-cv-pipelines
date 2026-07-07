FROM picsellia/cpu:python3.14

RUN apt-get update && apt-get install -y \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/split_video/.venv/bin/activate' /experiment/run.sh

COPY ./uv.lock split_video/uv.lock
COPY ./pyproject.toml split_video/pyproject.toml

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.14) --project split_video

COPY ./ split_video

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.14", "split_video/pipeline.py"]
