FROM registry.baidubce.com/paddlepaddle/paddle:2.6.1-gpu-cuda11.7-cudnn8.4-trt8.4

RUN apt-get update && apt-get upgrade -y && \
    apt-get install --no-install-recommends -y build-essential software-properties-common wget && \
    wget -q -O - https://packages.sury.org/php/apt.gpg | apt-key add - && \
    echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu focal main" > /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-focal.list && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        python3-pip libpq-dev libgtk2.0-dev libgl1-mesa-dev gcc \
        python3.10 python3.10-dev python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip && python3.10 -m pip install -U setuptools

ENV PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

COPY ./base /experiment

RUN ln -s /experiment/run.sh /usr/bin/run && \
    chmod +x /experiment/run.sh && \
    chown -R 42420:42420 /experiment

ARG REBUILD_ALL
COPY ./src/pipelines/paddle_ocr/requirements.txt ./src/pipelines/paddle_ocr/requirements.txt
ARG REBUILD_PICSELLIA

RUN $(which python3.10) -m pip install --no-cache -r ./src/pipelines/paddle_ocr/requirements.txt

RUN mkdir -p /workspace && chmod -R 777 /workspace
RUN mkdir -p /root && chmod -R 777 /root

WORKDIR /experiment

COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src
COPY ./src/pipelines/paddle_ocr ./src/pipelines/paddle_ocr

RUN cd /experiment/src/pipelines/paddle_ocr && git clone https://github.com/PaddlePaddle/PaddleOCR.git --branch release/2.8

ENV PYTHONPATH=":/experiment/src"

ENTRYPOINT ["run", "python3.10", "src/pipelines/paddle_ocr/training_pipeline.py"]
