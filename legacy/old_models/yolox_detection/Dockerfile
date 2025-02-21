FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04-python3.10

RUN apt-get -y update
RUN apt-get -y install git

COPY ./yolox-detection/requirements.txt .

ARG REBUILD_ALL
RUN python3.10 -m pip install -r requirements.txt --no-cache-dir

WORKDIR /experiment

COPY ./yolox-detection/experiment ./
COPY ./evaluator ./evaluator

RUN mkdir -p /workspace
RUN chmod 777 /workspace

ENTRYPOINT ["run", "python3.10", "main.py"]
