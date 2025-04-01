import os

from picsellia import Label, ModelVersion
from picsellia_cv_engine.core.models import Model


def find_latest_run_dir(dir: str, model_name: str):
    """
    Finds the latest run directory in the given directory.
    """
    run_dirs = os.listdir(dir)
    run_dirs = [f for f in run_dirs if f.startswith(model_name)]
    if not run_dirs:
        raise ValueError("No results folder found")
    elif len(run_dirs) == 1:
        return os.path.join(dir, run_dirs[0])

    return os.path.join(
        dir,
        sorted([f for f in run_dirs if f[-1].isdigit()])[-1],
    )


class UltralyticsModel(Model):
    def __init__(
        self,
        name: str,
        model_version: ModelVersion,
        pretrained_weights_name: str | None = None,
        trained_weights_name: str | None = None,
        config_name: str | None = None,
        exported_weights_name: str | None = None,
        labelmap: dict[str, Label] | None = None,
    ):
        super().__init__(
            name=name,
            model_version=model_version,
            pretrained_weights_name=pretrained_weights_name,
            trained_weights_name=trained_weights_name,
            config_name=config_name,
            exported_weights_name=exported_weights_name,
            labelmap=labelmap,
        )
        self.latest_run_dir: str | None = None

    def set_latest_run_dir(self):
        """
        Sets the latest run directory in the given results directory.
        """
        if not self.results_dir or not os.path.exists(self.results_dir):
            raise ValueError("The results directory is not set.")
        self.latest_run_dir = find_latest_run_dir(self.results_dir, self.name)

    def set_trained_weights_path(self):
        """
        Sets the path to the trained weights file using the latest run directory.
        """
        if not self.results_dir or not os.path.exists(self.results_dir):
            raise ValueError("The results directory is not set.")
        if not self.latest_run_dir or not os.path.exists(self.latest_run_dir):
            raise ValueError("The latest run directory is not set.")
        trained_weights_dir = os.path.join(self.latest_run_dir, "weights")
        self.trained_weights_path = os.path.join(trained_weights_dir, "best.pt")
