import logging
import os

from trainer import Yolov8SegmentationTrainer

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
logging.getLogger("picsellia").setLevel(logging.INFO)


training_pipeline = Yolov8SegmentationTrainer()
training_pipeline.prepare_data_for_training()
training_pipeline.train()
training_pipeline.eval()
