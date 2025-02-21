FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04-python3.10

COPY ./ViT-classification/requirements.txt .

ARG REBUILD_ALL
RUN python3.10 -m pip install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA


RUN python3.10 -m pip install --no-cache-dir picsellia
RUN python3.10 -m install picsellia --upgrade

RUN python3.10 -m pip install -U accelerate
RUN python3.10 -m pip install -U transformers
RUN python3.10 -m pip install optimum[onnxruntime]

WORKDIR /experiment
COPY ./ViT-classification/experiment ./
RUN mkdir -p /workspace
COPY ./abstract_trainer ./abstract_trainer
COPY ./evaluator ./evaluator
RUN chown -R 42420:42420 /workspace

ENTRYPOINT ["run", "python3.10", "main.py"]
