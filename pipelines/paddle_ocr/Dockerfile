FROM registry.baidubce.com/paddlepaddle/paddle:2.6.1-gpu-cuda11.7-cudnn8.4-trt8.4

WORKDIR /experiment

# --- Base system & Python setup ---
RUN apt-get update && apt-get upgrade -y && \
    apt-get install --no-install-recommends -y \
        build-essential software-properties-common wget \
        python3-pip libpq-dev libgtk2.0-dev libgl1-mesa-dev gcc \
        python3.10 python3.10-dev python3.10-distutils && \
    wget -q -O - https://packages.sury.org/php/apt.gpg | apt-key add - && \
    echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu focal main" > /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-focal.list && \
    apt-get update && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install -U setuptools && \
    python3.10 -m pip install uv

RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker && \
    cp -r /tmp/base-docker/base/. /experiment

RUN sed -i '1 a source /experiment/paddle_ocr/.venv/bin/activate' /experiment/run.sh

RUN ln -s /experiment/run.sh /usr/bin/run && \
    chmod +x /experiment/run.sh && \
    chown -R 42420:42420 /experiment

# --- Requirements & source ---
ARG REBUILD_ALL
COPY ./ ./paddle_ocr
ARG REBUILD_PICSELLIA

# --- External clone ---
RUN git clone https://github.com/PaddlePaddle/PaddleOCR.git --branch release/2.8 ./paddle_ocr/PaddleOCR

RUN uv sync --python=$(which python3.10) --project paddle_ocr

# --- Workspace and env ---
RUN mkdir -p /workspace /root && chmod -R 777 /workspace /root

ENV PYTHONPATH="/experiment" \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

ENTRYPOINT ["run", "python3.10", "paddle_ocr/pipeline.py"]
