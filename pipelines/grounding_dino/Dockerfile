FROM picsellia/cpu:python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --python=$(which python3.10) git+https://github.com/picselliahq/picsellia-cv-engine.git@main
RUN uv pip install --python=$(which python3.10) torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /experiment

ARG REBUILD_ALL
COPY ./ ./pipelines/grounding_dino
ARG REBUILD_PICSELLIA

RUN uv pip install --python=$(which python3.10) --no-cache -r ./pipelines/grounding_dino/requirements.txt

RUN git clone https://github.com/IDEA-Research/GroundingDINO.git && \
    cd GroundingDINO && uv pip install --python=$(which python3.10) -e . --no-build-isolation

ENV PYTHONPATH="/experiment"

ENTRYPOINT ["run", "python3.10", "pipelines/grounding_dino/picsellia_pipeline.py"]
