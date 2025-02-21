import logging
import os

from helpers import TrainingPipeline
from transformers import TrainerCallback

os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
logging.getLogger("picsellia").setLevel(logging.INFO)


class LogObjectDetectionMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            for metric_name, value in logs.items():
                if value < 1000:
                    if metric_name in [
                        "train_loss",
                        "total_flos",
                        "train_steps_per_second",
                        "train_samples_per_second",
                        "train_runtime",
                    ]:
                        experiment.log(str(metric_name), float(value), "value")
                    else:
                        experiment.log(str(metric_name), float(value), "line")


training_pipeline = TrainingPipeline()
experiment = training_pipeline.get_experiment()
train_test_valid_dataset, dataset = training_pipeline.prepare_data_for_training()
trainer = training_pipeline.train(
    train_test_valid_dataset=train_test_valid_dataset,
    callback_list=[LogObjectDetectionMetricsCallback],
)
image_processor, model = training_pipeline.test(
    train_test_valid_dataset=train_test_valid_dataset
)
training_pipeline.evaluate(
    dataset=dataset, train_test_valid_dataset=train_test_valid_dataset, model=model
)
