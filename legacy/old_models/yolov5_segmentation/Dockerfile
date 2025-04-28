FROM picsellia/cuda:11.4.3-cudnn8-ubuntu20.04-python3.8

COPY ./yolov5-segmentation/requirements.txt .

ARG REBUILD_ALL
RUN python3.8 -m pip install -r ./requirements.txt --no-cache-dir
ARG REBUILD_PICSELLIA

RUN python3.8 -m pip install picsellia-yolov5==0.2.8
RUN python3.8 -m pip install picsellia --upgrade

WORKDIR /experiment

COPY ./yolov5-segmentation/experiment ./
COPY ./core_utils ./core_utils

ENTRYPOINT ["run", "python3.8", "main.py"]
