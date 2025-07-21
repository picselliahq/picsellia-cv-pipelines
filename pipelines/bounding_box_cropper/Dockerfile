FROM picsellia/cpu:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --python=$(which python3.10) git+https://github.com/picselliahq/picsellia-cv-engine.git@main

WORKDIR /experiment

ARG REBUILD_ALL
COPY ./ ./bounding_box_cropper
ARG REBUILD_PICSELLIA

RUN uv pip install --python=$(which python3.10) --no-cache -r ./bounding_box_cropper/requirements.txt

ENV PYTHONPATH=":/experiment"

ENTRYPOINT ["run", "python3.10", "./bounding_box_cropper/picsellia_pipeline.py"]
