FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04-python3.10

COPY ./ViT-classification/requirements.txt .

ARG REBUILD_ALL
RUN python3.10 -m install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN python3.10 -m pip install --no-cache-dir picsellia
RUN python3.10 -m pip install -U accelerate
RUN python3.10 -m pip install -U transformers

WORKDIR /experiment

COPY ./ViT-classification/experiment ./

ENTRYPOINT ["run", "python3.10", "main.py"]
