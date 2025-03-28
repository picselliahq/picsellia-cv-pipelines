import os
import subprocess

from picsellia import Experiment
from picsellia.sdk.log import LogType
from picsellia_cv_engine.models.model.model import Model


def extract_and_log_metrics(log_line: str) -> dict[str, str | int | float]:
    """
    Extracts metrics from a log line by parsing key-value pairs.

    The function is designed to process log lines output from the training process, where metrics
    are logged in a specific format. It handles different types of values such as integers, floats,
    and strings.

    Args:
        log_line (str): A single line of log output from the training process.

    Returns:
        Dict[str, Union[str, int, float]]: Extracted metrics as a dictionary, where the keys are
        metric names and the values are either integers, floats, or strings depending on the type of the metric.
    """
    log_line = log_line.split("ppocr INFO:")[-1].strip()
    metrics: dict[str, str | int | float] = {}
    key_value_pairs = log_line.split(",")

    for pair in key_value_pairs:
        if ":" in pair:
            key, value = pair.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                if key == "epoch":
                    metrics[key] = int(value.replace("[", "").split("/")[0])
                elif "." in value:
                    metrics[key] = float(value)
                else:
                    metrics[key] = int(value)
            except ValueError:
                metrics[key] = value

    return metrics


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


class PaddleOCRModelTrainer:
    """
    Trainer for PaddleOCR models. This class manages the model training process and captures
    metrics during training, logging them to the Picsellia experiment.

    Attributes:
        model (Model): The context containing the model configuration and paths.
        experiment (Experiment): The Picsellia experiment where logs will be recorded.
        last_logged_epoch (Union[int, None]): The last epoch for which metrics were logged.
    """

    def __init__(self, model: Model, experiment: Experiment):
        """
        Initializes the trainer with a model and experiment.

        Args:
            model (Model): The context for the PaddleOCR model being trained.
            experiment (Experiment): The Picsellia experiment to log training metrics.
        """
        self.model = model
        self.experiment = experiment
        self.last_logged_epoch: int | None = None  # Last epoch that was logged

    def train_model(self):
        """
        Trains the PaddleOCR model using the configuration provided in the model.

        This method constructs a command to execute the training script, processes the output logs,
        and logs relevant metrics to the experiment. If the configuration file path is missing,
        it raises a ValueError.

        Raises:
            ValueError: If no configuration file path is found in the model.
        """
        config_path = self.model.config_path
        if not config_path:
            raise ValueError(
                f"No configuration file path found in {self.model.name} model"
            )

        command = [
            "python3.10",
            "src/pipelines/paddle_ocr/PaddleOCR/tools/train.py",
            "-c",
            config_path,
        ]

        current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f".:{current_pythonpath}"
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        try:
            self._process_training_output(process, self.model)
        except Exception as e:
            print("Error during model training:", e)
        finally:
            process.wait()
            if process.returncode != 0:
                handle_training_failure(process)

            os.environ["PYTHONPATH"] = current_pythonpath

    def _process_training_output(self, process: subprocess.Popen, model: Model):
        """
        Processes the output from the training subprocess and extracts metrics.

        This method reads the training logs line by line, extracts relevant metrics, and logs
        them to the Picsellia experiment. Only new metrics from previously unlogged epochs are recorded.

        Args:
            process (subprocess.Popen): The subprocess object running the training command.
            model (Model): The model containing information about the model being trained.
        """
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                print(line.strip())
                if "epoch:" in line:
                    metrics = extract_and_log_metrics(line)
                    current_epoch = metrics.get("epoch")
                    if (
                        current_epoch is not None
                        and isinstance(current_epoch, int)
                        and current_epoch != self.last_logged_epoch
                    ):
                        self.last_logged_epoch = current_epoch
                        metrics = {
                            k: v
                            for k, v in metrics.items()
                            if k not in ["epoch", "global_step"]
                        }
                        for key, value in metrics.items():
                            if isinstance(value, int | float):
                                self.experiment.log(
                                    name=f"{model.name}/{key}",
                                    data=value,
                                    type=LogType.LINE,
                                )
