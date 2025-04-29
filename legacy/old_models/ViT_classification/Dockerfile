FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04-python3.10

COPY ./ViT_classification/requirements.txt .

ARG REBUILD_ALL
RUN python3.10 -m pip install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA


RUN python3.10 -m pip install --no-cache-dir picsellia
RUN python3.10 -m pip install picsellia --upgrade

WORKDIR /experiment
COPY ./ViT_classification/experiment ./
RUN mkdir -p /workspace
COPY ./abstract_trainer ./abstract_trainer
COPY ./evaluator ./evaluator
COPY ./core_utils ./core_utils
RUN chown -R 42420:42420 /workspace

ENTRYPOINT ["run", "python3.10", "main.py"]
