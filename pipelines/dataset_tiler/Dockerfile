FROM picsellia/cpu:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --python=$(which python3.10) git+https://github.com/picselliahq/picsellia-cv-engine.git@main

ARG REBUILD_ALL
COPY ./ ./dataset_tiler
ARG REBUILD_PICSELLIA

RUN uv pip install --python=$(which python3.10) --no-cache -r ./dataset_tiler/requirements.txt

WORKDIR /experiment


ENV PYTHONPATH=":/experiment"

ENTRYPOINT ["run", "python3.10", "./dataset_tiler/picsellia_pipeline.py"]
