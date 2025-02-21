import logging
import os

import keras
from picsellia.types.enums import LogType
from trainer import UnetSegmentationTrainer

os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
logging.getLogger("picsellia").setLevel(logging.INFO)

training_pipeline = UnetSegmentationTrainer()
training_pipeline.prepare_data_for_training()
experiment = training_pipeline.experiment


class LogTrainingMetrics(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("logs on_epoch_end: ", logs)
        for metric_name in logs.keys():
            experiment.log(
                name=metric_name, type=LogType.LINE, data=float(logs[metric_name])
            )


training_pipeline.callbacks.append(LogTrainingMetrics())
training_pipeline.train()
training_pipeline.eval()
