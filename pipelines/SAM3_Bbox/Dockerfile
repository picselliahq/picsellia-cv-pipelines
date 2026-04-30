FROM picsellia/cuda:12.6.3-cudnn-ubuntu24.04-python3.14

RUN apt-get update && apt-get install -y \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment
RUN sed -i '1 a source /experiment/SAM3_Bbox/.venv/bin/activate' /experiment/run.sh

COPY ./uv.lock SAM3_Bbox/uv.lock
COPY ./pyproject.toml SAM3_Bbox/pyproject.toml

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.14) --project SAM3_Bbox  --prerelease allow

COPY ./ SAM3_Bbox

ENV HF_HOME="/experiment/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/experiment/.cache/huggingface/transformers"
ENV HF_DATASETS_CACHE="/experiment/.cache/huggingface/datasets"
ENV HUGGINGFACE_HUB_CACHE="/experiment/.cache/huggingface/hub"

# Create cache directories with proper permissions
RUN mkdir -p /experiment/.cache/huggingface/transformers \
    && mkdir -p /experiment/.cache/huggingface/datasets \
    && mkdir -p /experiment/.cache/huggingface/hub \
    && chmod -R 777 /experiment/.cache

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.14", "SAM3_Bbox/pipeline.py"]
