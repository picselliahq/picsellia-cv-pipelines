FROM picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10

COPY ./unet_instance_segmentation/requirements.txt .

ARG REBUILD_ALL
RUN python3.10 -m pip install -r ./requirements.txt --no-cache-dir
RUN python3.10 -m pip install picsellia --upgrade

WORKDIR /experiment
ENV SM_FRAMEWORK="tf.keras"

COPY ./unet_instance_segmentation/experiment ./
COPY ./abstract_trainer ./abstract_trainer
COPY ./core_utils ./core_utils

ENTRYPOINT ["run", "python3.10", "main.py"]
