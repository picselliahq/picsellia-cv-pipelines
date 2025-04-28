from abc import ABC, abstractmethod

from core_utils.picsellia_utils import get_experiment


class AbstractTrainer(ABC):
    def __init__(self):
        self.experiment = get_experiment()
        self.dataset_list = self.experiment.list_attached_dataset_versions()
        self.parameters = self.experiment.get_log("parameters").data
        self.labelmap = {}

    @abstractmethod
    def prepare_data_for_training(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass
