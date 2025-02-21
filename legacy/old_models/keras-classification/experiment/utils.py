import os
import shutil
from typing import List, Tuple, Optional

import numpy as np
import picsellia
import tensorflow as tf
from keras import backend as K
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia.sdk.experiment import Experiment
from picsellia.types.enums import LogType
from PIL import Image
from pycocotools.coco import COCO
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.utils import to_categorical


def dataset(experiment, assets, count, split_type, new_size, n_classes):
    X = []
    for i in range(len(assets)):
        path = os.path.join(experiment.png_dir, split_type, assets[i].filename)
        x = Image.open(path).convert("RGB")
        x = x.resize(new_size)
        x = np.array(x)
        X.append(x)

    y = np.zeros(len(assets))
    indices = np.cumsum(count["y"])
    for i in range(len(indices) - 1):
        y[indices[i] : indices[i + 1]] = np.ones(
            len(y[indices[i] : indices[i + 1]])
        ) * (i)
    y = to_categorical(y, n_classes)

    return np.asarray(X), y


def get_experiment() -> Experiment:
    if "api_token" not in os.environ:
        raise Exception("You must set an api_token to run this image")
    api_token = os.environ["api_token"]

    if "host" not in os.environ:
        host = "https://app.picsellia.com"
    else:
        host = os.environ["host"]

    if "organization_id" not in os.environ:
        organization_id = None
    else:
        organization_id = os.environ["organization_id"]

    client = picsellia.Client(
        api_token=api_token, host=host, organization_id=organization_id
    )

    if "experiment_name" in os.environ:
        experiment_name = os.environ["experiment_name"]
        if "project_token" in os.environ:
            project_token = os.environ["project_token"]
            project = client.get_project_by_id(project_token)
        elif "project_name" in os.environ:
            project_name = os.environ["project_name"]
            project = client.get_project(project_name)
        experiment = project.get_experiment(experiment_name)
    else:
        Exception("You must set the project_token or project_name and experiment_name")
    return experiment


def get_train_test_eval_datasets_from_experiment(
    experiment: Experiment,
) -> Tuple[bool, DatasetVersion, DatasetVersion, DatasetVersion]:
    is_split = _is_train_test_eval_dataset(experiment)
    if is_split:
        print("We found 3 datasets:")
        train = experiment.get_dataset("train")
        print(f"{train.name}/{train.version} for training")
        test = experiment.get_dataset("test")
        print(f"{test.name}/{test.version} for testing")
        eval_dataset = experiment.get_dataset("eval")
        print(f"{eval_dataset.name}/{eval_dataset.version} for evaluation")
    else:
        print(
            "We only found one dataset inside your experiment, the train/test/split will be performed automatically."
        )
        train = experiment.list_attached_dataset_versions()[0]
        test = None
        eval_dataset = None
    return is_split, train, test, eval_dataset


def _is_train_test_eval_dataset(experiment: Experiment) -> bool:
    datasets: List[DatasetVersion] = experiment.list_attached_dataset_versions()
    if len(datasets) < 3:
        return False
    template = ["train", "test", "eval"]
    ok_counter = 0
    for alias in template:
        try:
            experiment.get_dataset(alias)
            ok_counter += 1
        except Exception:
            pass
    return ok_counter == 3


def _move_files_in_class_directories(
    coco: COCO, base_imdir: Optional[str] = None
) -> str:
    if base_imdir is None:
        raise ValueError("base_imdir must be provided")
    fnames = os.listdir(base_imdir)
    for i in coco.cats:
        cat = coco.cats[i]
        class_folder = os.path.join(base_imdir, cat["name"])
        if not os.path.isdir(class_folder):
            os.mkdir(class_folder)
    print(f"Formatting {base_imdir} ..")
    for i in coco.imgs:
        im = coco.imgs[i]
        if im["file_name"] not in fnames:
            continue
        ann = coco.loadAnns(coco.getAnnIds(im["id"]))
        if len(ann) > 1:
            print(f"{im['file_name']} has more than one class. Skipping")
        ann = ann[0]
        cat = coco.loadCats(ann["category_id"])[0]
        fpath = os.path.join(base_imdir, im["file_name"])
        new_fpath = os.path.join(base_imdir, cat["name"], im["file_name"])
        try:
            shutil.move(fpath, new_fpath)
        except Exception:
            print(f"{im['file_name']} skipped.")
    print(f"Formatting {base_imdir} .. OK")
    return base_imdir


def order_repartition_according_labelmap(labelmap, repartition):
    ordered_rep = {"x": list(labelmap.values()), "y": []}
    for name in ordered_rep["x"]:
        ordered_rep["y"].append(repartition["y"][repartition["x"].index(name)])
    return ordered_rep


def recall_m(y_true, y_pred):
    """
    This function returns recall_score between y_true and y_pred
    This function is ported as a metric to the Neural Network Models
    Keras backend is used to take care of batch type training, the metric takes in a batch of y_pred and corresponding y_pred
    as input and returns recall score of the batch
    """
    true_positives = K.sum(
        K.round(K.clip(y_true * y_pred, 0, 1))
    )  # calculates number of true positives
    # calculates number of actual positives
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # K.epsilon takes care of non-zero divisions
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """
    This function returns precison_score between y_true and y_pred
    This function is ported as a metric to the Neural Network Models
    Keras backend is used to take care of batch type training, the metric takes in a batch of y_pred and corresponding y_pred
    as input and returns prediction score of the batch
    """
    true_positives = K.sum(
        K.round(K.clip(y_true * y_pred, 0, 1))
    )  # calculates number of true positives
    # calculates number of predicted positives
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # K.epsilon takes care of non-zero divisions
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_micro(y_true, y_pred):
    """
    This function returns f1_score between y_true and y_pred
    This
    This function is ported as a metric to the Neural Network Models
    Keras backend is used to take care of batch type training, the metric takes in a batch of y_pred and corresponding y_pred
    as input and returns f1 score of the batch
    """
    precision = precision_m(
        y_true, y_pred
    )  # calls precision metric and takes the score of precision of the batch
    # calls recall metric and takes the score of precision of the batch
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


dependencies = {"recall_m": recall_m, "precision_m": precision_m, "f1_micro": f1_micro}


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, val_data, batch_size=None, experiment: Experiment = None):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        self.experiment = experiment

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        batches = len(self.validation_data)
        # total = batches * self.batch_size

        val_pred = []
        val_true = []
        for batch in range(batches):
            xVal, yVal = next(self.validation_data)

            # val_pred_batch = np.zeros((len(xVal)))
            # val_true_batch = np.zeros((len(xVal)))

            val_pred_batch = np.argmax(np.asarray(self.model.predict(xVal)), axis=1)
            val_true_batch = np.argmax(yVal, axis=1)

            val_pred.append(val_pred_batch)
            val_true.append(val_true_batch)

        import itertools

        val_pred = np.asarray(list(itertools.chain.from_iterable(val_pred)))
        val_true = np.asarray(list(itertools.chain.from_iterable(val_true)))

        _val_f1 = f1_score(val_true, val_pred, average="macro")
        _val_precision = precision_score(val_true, val_pred, average="macro")
        _val_recall = recall_score(val_true, val_pred, average="macro")

        logs["val_f1"] = _val_f1
        logs["val_recall"] = _val_recall
        logs["val_precision"] = _val_precision
        print(
            " - val_f1_macro: %f - val_precision_macro: %f - val_recall_macro: %f"
            % (_val_f1, _val_precision, _val_recall)
        )

        if self.experiment is not None:
            self.experiment.log(name="val_f1", data=[float(_val_f1)], type=LogType.LINE)
            self.experiment.log(
                name="val_recall", data=[float(_val_recall)], type=LogType.LINE
            )
            self.experiment.log(
                name="val_precision", data=[float(_val_precision)], type=LogType.LINE
            )

        # predIdxs = self.model.predict(self.validation_data)
        # predIdxs = np.argmax(predIdxs, axis=1)
        # cr = classification_report(self.validation_data.classes, predIdxs, target_names=np.unique(self.validation_data.classes))

        # original_stdout = sys.stdout  # Save a reference to the original standard output
        # with open('cm_by_epoch.txt', 'a') as f:
        #     sys.stdout = f # Change the standard output to the file we created.
        #     print("Confusion matrix on validation data on epoch "+str(epoch+1)+"\n"+"--------------------------------------------------------"+"\n")
        #     print(cr)
        #     print("\n")
        #     sys.stdout = original_stdout # Reset the standard output to its original value​
        return
