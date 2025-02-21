FROM picsellia/cuda:11.7.1-cudnn8-ubuntu20.04-python3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

COPY ./keras-classification/experiment ./
COPY ./keras-classification/requirements.txt ./

ARG REBUILD_ALL
ARG REBUILD_PICSELLIA

RUN export UV_HTTP_TIMEOUT=900 && uv pip install --python=$(which python3.10) --no-cache -r requirements.txt

ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

ENTRYPOINT ["run", "python3.10", "main.py"]
