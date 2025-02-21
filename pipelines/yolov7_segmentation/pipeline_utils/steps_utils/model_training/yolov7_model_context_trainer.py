import os
import subprocess

from picsellia import Experiment

from pipelines.yolov7_segmentation.pipeline_utils.dataset.yolov7_dataset_collection import (
    Yolov7DatasetCollection,
)
from pipelines.yolov7_segmentation.pipeline_utils.model.yolov7_model_context import (
    Yolov7ModelContext,
)
from pipelines.yolov7_segmentation.pipeline_utils.parameters.yolov7_hyper_parameters import (
    Yolov7HyperParameters,
)


def handle_training_failure(process: subprocess.Popen):
    """
    Handles training failure by printing error messages from the training process.

    Args:
        process (subprocess.Popen): The process object running the training command.
    """
    print("Training failed with errors")
    if process.stderr:
        errors = process.stderr.read()
        print(errors)


class Yolov7ModelContextTrainer:
    def __init__(self, model_context: Yolov7ModelContext, experiment: Experiment):
        """
        Initializes the trainer with a model context and experiment.

        Args:
            model_context (ModelContext): The context for the PaddleOCR model being trained.
            experiment (Experiment): The Picsellia experiment to log training metrics.
        """
        self.model_context = model_context
        self.experiment = experiment

    def train_model_context(
        self,
        dataset_collection: Yolov7DatasetCollection,
        hyperparameters: Yolov7HyperParameters,
        api_token: str,
        organization_id: str,
        host: str,
        experiment_id: str,
    ):
        if not self.model_context.pretrained_weights_path or not os.path.exists(
            self.model_context.pretrained_weights_path
        ):
            raise ValueError("Pretrained weights file not found.")

        if not self.model_context.config_path or not os.path.exists(
            self.model_context.config_path
        ):
            raise ValueError("Configuration file not found.")

        if not self.model_context.hyperparameters_path or not os.path.exists(
            self.model_context.hyperparameters_path
        ):
            raise ValueError("Hyperparameters file not found.")

        if not dataset_collection.config_path or not os.path.exists(
            dataset_collection.config_path
        ):
            raise ValueError("Dataset configuration file not found.")

        if not self.model_context.results_dir or not os.path.exists(
            self.model_context.results_dir
        ):
            raise ValueError("Results directory not found.")

        project_dir = os.path.join(self.model_context.results_dir, "training")
        os.makedirs(project_dir, exist_ok=True)

        train_file_path = os.path.abspath(
            "src/pipelines/yolov7_segmentation/yolov7/seg/segment/train.py"
        )

        command = [
            "python3.10",
            train_file_path,
            "--weights",
            self.model_context.pretrained_weights_path,
            "--cfg",
            self.model_context.config_path,
            "--data",
            dataset_collection.config_path,
            "--hyp",
            self.model_context.hyperparameters_path,
            "--epochs",
            str(hyperparameters.epochs),
            "--batch-size",
            str(hyperparameters.batch_size),
            "--img-size",
            str(hyperparameters.image_size),
            "--device",
            str(hyperparameters.device),
            "--project",
            project_dir,
            "--name",
            self.model_context.model_name,
            "--api_token",
            api_token,
            "--organization_id",
            organization_id,
            "--host",
            host,
            "--experiment_id",
            experiment_id,
        ]

        process = subprocess.Popen(command, stdout=None, stderr=None, text=True)

        return_code = process.wait()
        if return_code != 0:
            print("Training failed with errors.")
        else:
            print("Training completed successfully.")
