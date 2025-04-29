FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04-python3.10

COPY ./ViT_classification/requirements.txt .

ARG REBUILD_ALL
RUN python3.10 -m install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN python3.10 -m pip install --no-cache-dir picsellia
WORKDIR /experiment

COPY ./ViT_classification/experiment ./
COPY ./core_utils ./core_utils

ENTRYPOINT ["run", "python3.10", "main.py"]
