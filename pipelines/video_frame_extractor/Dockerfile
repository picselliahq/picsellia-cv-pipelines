FROM picsellia/cpu:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/video_frame_extractor/.venv/bin/activate' /experiment/run.sh

COPY ./uv.lock video_frame_extractor/uv.lock
COPY ./pyproject.toml video_frame_extractor/pyproject.toml

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.10) --project video_frame_extractor

COPY ./ video_frame_extractor

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.10", "video_frame_extractor/pipeline.py"]
