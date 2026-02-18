FROM picsellia/cpu:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker &&     cp -r /tmp/base-docker/base/. /experiment
RUN sed -i '1 a source /experiment/split_train_test_val/.venv/bin/activate' /experiment/run.sh

ARG REBUILD_ALL
COPY ./ split_train_test_val
ARG REBUILD_PICSELLIA

# Sync from uv.lock (assumes uv lock has already been created)
RUN uv sync --python=$(which python3.10) --project split_train_test_val

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.10", "split_train_test_val/pipeline.py"]
