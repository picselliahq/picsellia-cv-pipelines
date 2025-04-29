FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    unzip \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment
RUN git clone https://github.com/tensorflow/models.git

COPY ./tf2/experiment ./
COPY ./tf2/requirements.txt ./
COPY ./evaluator ./evaluator
COPY ./core_utils ./core_utils

COPY ./tf2/experiment/tf2_utils/model_lib_v2.py /experiment/models/research/object_detection/model_lib_v2.py

RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v23.4/protoc-23.4-linux-x86_64.zip \
    && unzip protoc-23.4-linux-x86_64.zip -d /usr/local/protoc \
    && rm protoc-23.4-linux-x86_64.zip

ENV PATH="/usr/local/protoc/bin:${PATH}"

RUN cd /experiment/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

ARG REBUILD_ALL
ARG REBUILD_PICSELLIA

RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.10) --no-cache -r requirements.txt

ENV PYTHONPATH="/experiment/models/research"

ENTRYPOINT ["run", "python3.10", "main.py"]
