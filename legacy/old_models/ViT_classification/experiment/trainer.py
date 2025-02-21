import os

from datasets import load_dataset
from optimum.onnxruntime import ORTModelForImageClassification
from picsellia.exceptions import ResourceNotFoundError
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia.types.enums import InferenceType
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ToTensor
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    pipeline,
)
from utils import (
    _move_all_files_in_class_directories,
    compute_metrics,
    find_asset_by_filename,
    get_asset_filename_from_path,
    get_predicted_label_confidence,
    get_train_test_eval_datasets_from_experiment,
    log_labelmap,
    prepare_datasets_with_annotation,
    split_single_dataset,
)

from abstract_trainer.trainer import AbstractTrainer


class VitClassificationTrainer(AbstractTrainer):
    checkpoint = "google/vit-base-patch16-224-in21k"
    model_artifact_name = "pytorch_model"

    def __init__(self):
        super().__init__()
        self.train_path = os.path.join(self.experiment.base_dir, "data/train")
        self.test_path = os.path.join(self.experiment.base_dir, "data/test")
        self.eval_path = os.path.join(self.experiment.base_dir, "data/val")
        self.train_test_eval_path = {
            "train_path": self.train_path,
            "test_path": self.test_path,
            "eval_path": self.eval_path,
        }
        self.output_model_dir = os.path.join(self.experiment.checkpoint_dir)
        self.data_dir = os.path.join(self.experiment.base_dir, "data")
        self.evaluation_assets = None
        self.evaluation_ds = None
        self.loaded_dataset = None
        self.image_processor = None
        self.dataset_labels = None
        self.model = None
        self.trainer = None
        self.nbr_epochs = int(self.parameters.get("epochs", 5))
        self.batch_size = int(self.parameters.get("batch_size", 16))
        self.learning_rate = self.parameters.get("learning_rate", 5e-5)

    def prepare_data_for_training(self):
        (
            has_one_dataset,
            has_three_datasets,
            train_set,
            test_set,
            eval_set,
        ) = get_train_test_eval_datasets_from_experiment(experiment=self.experiment)

        if has_three_datasets:
            self.download_triple_dataset(train_set, test_set, eval_set)
            (
                self.evaluation_ds,
                self.evaluation_assets,
            ) = prepare_datasets_with_annotation(
                train_set=train_set,
                test_set=test_set,
                val_set=eval_set,
                train_test_eval_path_dict=self.train_test_eval_path,
            )

        elif has_one_dataset and not has_three_datasets:
            train_set.download("images")
            (
                train_assets,
                test_assets,
                eval_assets,
                train_rep,
                test_rep,
                val_rep,
                labels,
            ) = split_single_dataset(
                parameters=self.parameters,
                train_set=train_set,
                train_test_eval_path_dict=self.train_test_eval_path,
            )
            _move_all_files_in_class_directories(
                train_set=train_set, train_test_eval_path_dict=self.train_test_eval_path
            )
            self.evaluation_ds = train_set
            self.evaluation_assets = eval_assets
        else:
            raise Exception(
                "You must either have only one Dataset, 2 (train, test) or 3 datasets (train, test, eval)"
            )
        self.dataset_labels = {
            label.name: label for label in self.evaluation_ds.list_labels()
        }

        self.loaded_dataset = load_dataset(
            "imagefolder",
            data_dir=self.data_dir,
            cache_dir=self.experiment.base_dir,
        )
        loaded_checkpoint_folder_path = self._download_model_artifacts_if_available()
        if loaded_checkpoint_folder_path:
            self.checkpoint = loaded_checkpoint_folder_path
        self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self.loaded_dataset = self.loaded_dataset.with_transform(self.transforms)

    def download_triple_dataset(
        self,
        train_set: DatasetVersion,
        test_set: DatasetVersion,
        eval_set: DatasetVersion,
    ) -> None:
        for data_path, dataset in {
            self.train_path: train_set,
            self.test_path: test_set,
            self.eval_path: eval_set,
        }.items():
            dataset.download(target_path=data_path, max_workers=8)

    def transforms(self, examples):
        normalize = Normalize(
            mean=self.image_processor.image_mean, std=self.image_processor.image_std
        )
        size = (
            self.image_processor.size["shortest_edge"]
            if "shortest_edge" in self.image_processor.size
            else (
                self.image_processor.size["height"],
                self.image_processor.size["width"],
            )
        )
        _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
        examples["pixel_values"] = [
            _transforms(img.convert("RGB")) for img in examples["image"]
        ]
        del examples["image"]
        return examples

    def _download_model_artifacts_if_available(self) -> str:
        try:
            _ = self.experiment.get_artifact(self.model_artifact_name)
            loaded_checkpoint_folder_path = self.experiment.base_dir
            self.experiment.download_artifacts(with_tree=False)
        except ResourceNotFoundError as e:
            print(e)
            loaded_checkpoint_folder_path = None
        return loaded_checkpoint_folder_path

    def init_train(self):
        data_collator = DefaultDataCollator()
        label2id, id2label = self._get_label2id_id2label()
        log_labelmap(id2label=id2label, experiment=self.experiment)

        self.model = AutoModelForImageClassification.from_pretrained(
            self.checkpoint,
            num_labels=len(self.labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        training_args = TrainingArguments(
            output_dir=self.output_model_dir,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=16,
            num_train_epochs=self.nbr_epochs,
            warmup_ratio=0.1,
            logging_steps=5,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            save_total_limit=2,
            save_strategy="no",
            load_best_model_at_end=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.loaded_dataset["train"],
            eval_dataset=self.loaded_dataset["test"],
            tokenizer=self.image_processor,
            compute_metrics=compute_metrics,
        )

    def train(self):
        self.trainer.train()
        self.trainer.save_model(output_dir=self.output_model_dir)

        for artifact in os.listdir(path=self.output_model_dir):
            self.experiment.store(
                name=artifact.split(".")[0],
                path=os.path.join(self.output_model_dir, artifact),
            )

    def _get_label2id_id2label(self) -> tuple[dict, dict]:
        self.labels = self.loaded_dataset["train"].features["label"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(self.labels):
            label2id[label] = str(i)
            id2label[str(i)] = label
        return label2id, id2label

    def eval(self):
        classifier = pipeline("image-classification", model=self.output_model_dir)

        self.create_and_store_onnx_model(classifier)

        for path, subdirs, file_list in os.walk(self.eval_path):
            for file in file_list:
                self._run_one_asset_evaluation(
                    path=path, file=file, classifier=classifier
                )

        self.experiment.compute_evaluations_metrics(InferenceType.CLASSIFICATION)

    def create_and_store_onnx_model(self, classifier):
        bin_weights_directory_path = os.path.join(
            self.experiment.base_dir, "bin_weights"
        )
        classifier.save_pretrained(bin_weights_directory_path)
        onnx_weights_directory_path = os.path.join(
            self.experiment.base_dir, "onnx_weights"
        )
        ort_model = ORTModelForImageClassification.from_pretrained(
            bin_weights_directory_path, export=True
        )
        ort_model.save_pretrained(onnx_weights_directory_path)

        self.experiment.store(
            name="model-latest",
            path=os.path.join(onnx_weights_directory_path, "model.onnx"),
        )

    def _run_one_asset_evaluation(self, path: str, file: str, classifier):
        file_path = os.path.join(path, file)
        current_prediction = classifier(str(file_path))

        pred_label, pred_conf = get_predicted_label_confidence(current_prediction)
        asset_filename = get_asset_filename_from_path(file_path=file_path)
        self._find_asset_send_evaluation_by_filename(
            asset_filename=asset_filename,
            pred_label=pred_label,
            pred_conf=pred_conf,
        )

    def _find_asset_send_evaluation_by_filename(
        self, asset_filename: str, pred_label: str, pred_conf: float
    ):
        asset = find_asset_by_filename(
            filename=asset_filename, dataset=self.evaluation_ds
        )
        classification_data = (self.dataset_labels[pred_label], float(pred_conf))
        if asset is not None:
            self.experiment.add_evaluation(
                asset=asset,
                classifications=[classification_data],
            )
        print(f"Asset: {asset_filename} evaluated.")
