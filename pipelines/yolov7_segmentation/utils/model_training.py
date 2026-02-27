import os
import subprocess

from picsellia import Experiment

from .dataset import (
    Yolov7DatasetCollection,
)
from .model import (
    Yolov7Model,
)
from .parameters import (
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


class Yolov7ModelTrainer:
    def __init__(self, model: Yolov7Model, experiment: Experiment):
        """
        Initializes the trainer with a model and experiment.

        Args:
            model (Model): The context for the PaddleOCR model being trained.
            experiment (Experiment): The Picsellia experiment to log training metrics.
        """
        self.model = model
        self.experiment = experiment

    def train_model(
        self,
        dataset_collection: Yolov7DatasetCollection,
        hyperparameters: Yolov7HyperParameters,
        api_token: str,
        organization_id: str,
        host: str,
        experiment_id: str,
    ):
        if not self.model.pretrained_weights_path or not os.path.exists(
            self.model.pretrained_weights_path
        ):
            raise ValueError("Pretrained weights file not found.")

        if not self.model.config_path or not os.path.exists(self.model.config_path):
            raise ValueError("Configuration file not found.")

        if not self.model.hyperparameters_path or not os.path.exists(
            self.model.hyperparameters_path
        ):
            raise ValueError("Hyperparameters file not found.")

        if not dataset_collection.config_path or not os.path.exists(
            dataset_collection.config_path
        ):
            raise ValueError("Dataset configuration file not found.")

        if not self.model.results_dir or not os.path.exists(self.model.results_dir):
            raise ValueError("Results directory not found.")

        project_dir = os.path.join(self.model.results_dir, "training")
        os.makedirs(project_dir, exist_ok=True)
        _pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # .../yolov7_segmentation/utils/model_training.py → .../yolov7_segmentation/

        train_file_path = os.path.join(_pipeline_dir, "yolov7/seg/segment/train.py")
        venv_python = os.path.join(_pipeline_dir, ".venv/bin/python")
        # train_file_path = os.path.abspath(
        #     "yolov7_segmentation/yolov7/seg/segment/train.py"
        # )
        #
        # venv_python = os.path.abspath("yolov7_segmentation/.venv/bin/python")
        print(f"cwd: {os.getcwd()}")
        print(f"venv_python: {venv_python}")
        print(f"train_file_path: {train_file_path}")
        print(f"venv exists: {os.path.exists(venv_python)}")
        print(f"train exists: {os.path.exists(train_file_path)}")
        yolov7_seg_dir = os.path.join(_pipeline_dir, "yolov7/seg")
        env = os.environ.copy()
        # Set PYTHONPATH to only yolov7/seg — do NOT inherit existing PYTHONPATH.
        # The pipeline's utils/ package (yolov7_segmentation/utils/) is a regular
        # package that would shadow yolov7's utils/ namespace package if included,
        # causing "No module named 'utils.dataloaders'".
        env["PYTHONPATH"] = yolov7_seg_dir
        print(f"PYTHONPATH: {env.get('PYTHONPATH', 'NOT SET')}")
        utils_dir = os.path.join(_pipeline_dir, "yolov7/seg/utils")
        if os.path.exists(utils_dir):
            print(f"yolov7/seg/utils/ contents: {os.listdir(utils_dir)}")
        else:
            print("yolov7/seg/utils/ does NOT exist")
        command = [
            venv_python,
            train_file_path,
            "--weights",
            self.model.pretrained_weights_path,
            "--cfg",
            self.model.config_path,
            "--data",
            dataset_collection.config_path,
            "--hyp",
            self.model.hyperparameters_path,
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
            self.model.name,
            "--api_token",
            str(api_token),
            "--organization_id",
            str(organization_id),
            "--host",
            host,
            "--experiment_id",
            str(experiment_id),
        ]

        process = subprocess.Popen(command, stdout=None, stderr=None, text=True, env=env)

        return_code = process.wait()
        if return_code != 0:
            print("Training failed with errors.")
        else:
            print("Training completed successfully.")
