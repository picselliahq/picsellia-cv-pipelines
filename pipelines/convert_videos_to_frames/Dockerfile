FROM picsellia/cpu:python3.14

RUN apt-get update && apt-get install -y \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/convert_videos_to_frames/.venv/bin/activate' /experiment/run.sh

COPY ./uv.lock convert_videos_to_frames/uv.lock
COPY ./pyproject.toml convert_videos_to_frames/pyproject.toml

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.14) --project convert_videos_to_frames

COPY ./ convert_videos_to_frames

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.14", "convert_videos_to_frames/pipeline.py"]
