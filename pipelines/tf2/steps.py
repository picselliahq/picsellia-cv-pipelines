import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pxl_tf
import pxl_utils
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core.data.dataset.coco_dataset import CocoDataset
from picsellia_cv_engine.core.data.dataset.dataset_collection import DatasetCollection
from picsellia_cv_engine.core.models import Model

# -------------------------
# PYTHONPATH TF OD API
# -------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))  # dossier tf2/
RESEARCH = os.path.join(ROOT, "models", "research")
SLIM = os.path.join(RESEARCH, "slim")
sys.path.insert(0, RESEARCH)
sys.path.insert(0, SLIM)


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _maybe_unpack_to_dir(artifact_path: str, dst_dir: str) -> str:
    if not artifact_path:
        raise ValueError("artifact_path is empty")

    p = Path(artifact_path)
    _safe_mkdir(dst_dir)

    if not p.exists():
        if p.parent.exists():
            p = p.parent
        else:
            raise FileNotFoundError(f"artifact_path not found: {artifact_path}")

    if p.is_dir():
        return str(p)

    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(dst_dir)
        return dst_dir

    dst = Path(dst_dir) / p.name

    # ✅ Avoid copying onto itself
    try:
        if p.resolve() == dst.resolve():
            return dst_dir
    except FileNotFoundError:
        if str(p) == str(dst):
            return dst_dir

    shutil.copy(str(p), str(dst))
    return dst_dir


def _tar_dir(src_dir: str, out_path: str) -> str:
    src_dir = str(Path(src_dir))
    out_path = str(Path(out_path))
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(src_dir, arcname=Path(src_dir).name)
    return out_path


def _get_split(ds_col: DatasetCollection[CocoDataset], name: str) -> CocoDataset:
    if name in ds_col.datasets:
        return ds_col.datasets[name]
    if name == "val" and "eval" in ds_col.datasets:
        return ds_col.datasets["eval"]
    raise KeyError(
        f"Dataset split '{name}' not found. Available: {list(ds_col.datasets.keys())}"
    )


def _build_labelmap_from_label_names(label_names: list[str]) -> dict[str, str]:
    return {str(i + 1): name for i, name in enumerate(label_names)}


def _count_annotations_per_label(
    coco_data: dict, label_names: list[str]
) -> dict[str, int]:
    if not coco_data:
        return dict.fromkeys(label_names, 0)

    cat_id_to_name: dict[int, str] = {}
    for c in coco_data.get("categories", []):
        if "id" in c and "name" in c:
            cat_id_to_name[int(c["id"])] = c["name"]

    counts = dict.fromkeys(label_names, 0)
    for ann in coco_data.get("annotations", []):
        cid = ann.get("category_id")
        if cid is None:
            continue
        name = cat_id_to_name.get(int(cid))
        if name is None:
            continue
        if name not in counts:
            counts[name] = 0
        counts[name] += 1

    for n in label_names:
        counts.setdefault(n, 0)

    return counts


def _split_bar_payload(coco_data: dict, label_names: list[str]) -> dict:
    counts = _count_annotations_per_label(coco_data, label_names)
    return {"x": label_names, "y": [counts.get(n, 0) for n in label_names]}


def _experiment_log(
    experiment: Any, name: str, data: Any, chart: str, replace: bool = True
) -> None:
    if experiment is None:
        return
    if hasattr(experiment, "log"):
        experiment.log(name, data, chart, replace=replace)


def _experiment_chapter(experiment: Any, title: str) -> None:
    if experiment is None:
        return
    if hasattr(experiment, "start_logging_chapter"):
        experiment.start_logging_chapter(title)


def _experiment_buffer_start(experiment: Any, size: int = 9) -> None:
    if experiment is None:
        return
    if hasattr(experiment, "start_logging_buffer"):
        experiment.start_logging_buffer(size)


def _experiment_buffer_end(experiment: Any) -> None:
    if experiment is None:
        return
    if hasattr(experiment, "end_logging_buffer"):
        experiment.end_logging_buffer()


def _experiment_store_like_before(
    experiment: Any,
    picsellia_model: Model,
    training_config_dir: str,
    results_dir: str,
    exported_model_dir: str,
) -> None:
    if experiment is not None and hasattr(experiment, "store"):
        experiment.store("model-latest")
        experiment.store("config")
        experiment.store("checkpoint-data-latest")
        experiment.store("checkpoint-index-latest")
        return

    print("⚠️ context.experiment.store() non disponible → fallback upload d'artifacts.")
    try:
        saved_model_path = os.path.join(exported_model_dir, "saved_model")
        if os.path.isdir(saved_model_path):
            tar_path = _tar_dir(
                saved_model_path,
                os.path.join(picsellia_model.results_dir, "model-latest.tar.gz"),
            )
            picsellia_model.save_artifact_to_experiment(
                experiment=experiment,
                artifact_name="model-latest",
                artifact_path=tar_path,
            )

        tar_cfg = _tar_dir(
            training_config_dir,
            os.path.join(picsellia_model.results_dir, "config.tar.gz"),
        )
        picsellia_model.save_artifact_to_experiment(
            experiment=experiment,
            artifact_name="config",
            artifact_path=tar_cfg,
        )

        tar_ckpt = _tar_dir(
            results_dir, os.path.join(picsellia_model.results_dir, "checkpoints.tar.gz")
        )
        picsellia_model.save_artifact_to_experiment(
            experiment=experiment,
            artifact_name="checkpoints",
            artifact_path=tar_ckpt,
        )
    except Exception as e:
        print("⚠️ fallback artifact upload failed:", repr(e))


# -------------------------
# Train helpers (refacto for C901)
# -------------------------
def _load_splits(
    picsellia_datasets: DatasetCollection[CocoDataset],
) -> tuple[CocoDataset, CocoDataset, CocoDataset]:
    train_ds = _get_split(picsellia_datasets, "train")
    val_ds = _get_split(picsellia_datasets, "val")
    test_ds = _get_split(picsellia_datasets, "test")

    train_ds.load_coco_file_data()
    val_ds.load_coco_file_data()
    test_ds.load_coco_file_data()

    if not train_ds.images_dir or not val_ds.images_dir or not test_ds.images_dir:
        raise RuntimeError(
            "images_dir is missing on one of the datasets (train/val/test)."
        )

    return train_ds, val_ds, test_ds


def _prepare_labelmap_and_logs(
    experiment: Any,
    picsellia_model: Model,
    train_ds: CocoDataset,
    val_ds: CocoDataset,
    test_ds: CocoDataset,
) -> tuple[list[str], dict[str, str], str]:
    label_names = list(train_ds.labelmap.keys()) if train_ds.labelmap else []
    if not label_names:
        cats = (train_ds.coco_data or {}).get("categories", [])
        label_names = [c["name"] for c in cats]

    labelmap = _build_labelmap_from_label_names(label_names)
    label_path = pxl_utils.generate_label_map(
        classes=label_names,
        output_path=picsellia_model.results_dir,
    )

    _experiment_log(experiment, "labelmap", labelmap, "labelmap", replace=True)
    _experiment_log(
        experiment,
        "train-split",
        pxl_utils.sort_split(
            _split_bar_payload(train_ds.coco_data, label_names), label_names
        ),
        "bar",
        replace=True,
    )
    _experiment_log(
        experiment,
        "eval-split",
        pxl_utils.sort_split(
            _split_bar_payload(val_ds.coco_data, label_names), label_names
        ),
        "bar",
        replace=True,
    )
    _experiment_log(
        experiment,
        "test-split",
        pxl_utils.sort_split(
            _split_bar_payload(test_ds.coco_data, label_names), label_names
        ),
        "bar",
        replace=True,
    )

    return label_names, labelmap, label_path


def _create_records(
    context: Any,
    picsellia_model: Model,
    train_ds: CocoDataset,
    val_ds: CocoDataset,
    test_ds: CocoDataset,
    label_path: str,
) -> str:
    _experiment_chapter(getattr(context, "experiment", None), "Create records")

    record_dir = os.path.join(picsellia_model.results_dir, "records")
    _safe_mkdir(record_dir)

    pxl_utils.create_record_files(
        train_annotations=train_ds.coco_data,
        eval_annotations=val_ds.coco_data,  # "eval" = val split
        test_annotations=test_ds.coco_data,  # "test" = test split
        label_path=label_path,
        record_dir=record_dir,
        tfExample_generator=pxl_tf.tf_vars_generator,
        annotation_type=context.hyperparameters.annotation_type,
        images_dir_map={
            "train": train_ds.images_dir,
            "eval": val_ds.images_dir,
            "test": test_ds.images_dir,
        },
    )

    return record_dir


def _prepare_configs(
    picsellia_model: Model,
) -> tuple[str, str, str]:
    if not picsellia_model.config_path:
        raise RuntimeError("No config file found (pipeline.config or zip).")

    training_config_dir = picsellia_model.config_dir
    _safe_mkdir(training_config_dir)
    _maybe_unpack_to_dir(picsellia_model.config_path, training_config_dir)

    pipeline_config_path = None
    for root, _, files in os.walk(training_config_dir):
        if "pipeline.config" in files:
            pipeline_config_path = os.path.join(root, "pipeline.config")
            break
    if not pipeline_config_path:
        raise RuntimeError("pipeline.config not found after unpacking config artifact.")

    eval_config_dir = os.path.join(picsellia_model.results_dir, "eval_config")
    _safe_mkdir(eval_config_dir)
    shutil.copy(pipeline_config_path, os.path.join(eval_config_dir, "pipeline.config"))

    return training_config_dir, eval_config_dir, pipeline_config_path


def _edit_configs(
    context: Any,
    record_dir: str,
    checkpoint_dir: str,
    training_config_dir: str,
    eval_config_dir: str,
    label_path: str,
) -> None:
    params = (
        context.hyperparameters.model_dump()
        if hasattr(context.hyperparameters, "model_dump")
        else vars(context.hyperparameters)
    )

    pxl_utils.edit_config(
        model_selected=checkpoint_dir,
        input_config_dir=training_config_dir,
        output_config_dir=training_config_dir,
        train_record_path=os.path.join(record_dir, "train.record"),
        eval_record_path=os.path.join(record_dir, "test.record"),
        label_map_path=label_path,
        num_steps=context.hyperparameters.steps,
        batch_size=context.hyperparameters.batch_size,
        learning_rate=context.hyperparameters.learning_rate,
        annotation_type=context.hyperparameters.annotation_type,
        parameters=params,
    )

    pxl_utils.edit_config(
        model_selected=checkpoint_dir,
        input_config_dir=eval_config_dir,
        output_config_dir=eval_config_dir,
        train_record_path=os.path.join(record_dir, "train.record"),
        eval_record_path=os.path.join(record_dir, "eval.record"),
        label_map_path=label_path,
        num_steps=context.hyperparameters.steps,
        batch_size=context.hyperparameters.batch_size,
        learning_rate=context.hyperparameters.learning_rate,
        annotation_type=context.hyperparameters.annotation_type,
        parameters=params,
    )


def _run_train_and_export(
    context: Any,
    picsellia_model: Model,
    experiment: Any,
    training_config_dir: str,
) -> tuple[str, str]:
    _experiment_chapter(experiment, "Start training")

    results_dir = os.path.join(picsellia_model.results_dir, "tf2_results")
    _safe_mkdir(results_dir)

    pxl_utils.train(
        model_dir=results_dir,
        config_dir=training_config_dir,
        log_real_time=experiment,
        evaluate_fn=pxl_utils.evaluate,
        log_metrics=pxl_utils.log_metrics,
        checkpoint_every_n=context.hyperparameters.checkpoint_every_n,
    )

    _experiment_chapter(experiment, "Store artifacts")

    exported_model_dir = os.path.join(picsellia_model.results_dir, "exported_model")
    _safe_mkdir(exported_model_dir)

    pxl_utils.export_graph(
        ckpt_dir=results_dir,
        exported_model_dir=exported_model_dir,
        config_dir=training_config_dir,
    )

    return results_dir, exported_model_dir


def _compute_metrics_and_confusion(
    experiment: Any,
    picsellia_model: Model,
    record_dir: str,
    eval_config_dir: str,
    results_dir: str,
    exported_model_dir: str,
    labelmap: dict[str, str],
) -> None:
    _experiment_chapter(experiment, "Computing metrics on test dataset")
    _experiment_buffer_start(experiment, 9)

    eval_metrics_dir = os.path.join(picsellia_model.results_dir, "eval_metrics")
    _safe_mkdir(eval_metrics_dir)

    pxl_utils.evaluate(
        metrics_dir=eval_metrics_dir,
        config=eval_config_dir,
        ckpt_dir=results_dir,
    )

    metrics = pxl_utils.tf_events_to_dict(
        os.path.join(eval_metrics_dir, "eval"), "eval"
    )
    _experiment_log(experiment, "Evaluation/Metrics", metrics, "table", replace=True)

    conf, _ = pxl_utils.get_confusion_matrix(
        input_tfrecord_path=os.path.join(record_dir, "eval.record"),
        model=os.path.join(exported_model_dir, "saved_model"),
        labelmap=labelmap,
    )
    confusion = {"categories": list(labelmap.values()), "values": conf.tolist()}
    _experiment_log(
        experiment, "Evaluation/confusion-matrix", confusion, "heatmap", replace=True
    )

    _experiment_buffer_end(experiment)


def _run_optional_evaluators(experiment: Any, val_ds: CocoDataset) -> None:
    _experiment_chapter(experiment, "Starting Evaluation")

    try:
        from evaluator.tf_evaluator import (
            DetectionTensorflowEvaluator,
            SegmentationTensorflowEvaluator,
        )
        from picsellia.types.enums import InferenceType
    except Exception as e:
        print("⚠️ evaluators not available in this environment:", repr(e))
        return

    inference_type = None
    try:
        if experiment is not None and hasattr(experiment, "get_base_model_version"):
            inference_type = experiment.get_base_model_version().type
    except Exception:
        inference_type = None

    eval_assets = None
    try:
        if hasattr(val_ds, "list_assets"):
            eval_assets = val_ds.list_assets()
    except Exception:
        eval_assets = None

    if inference_type == InferenceType.OBJECT_DETECTION:
        DetectionTensorflowEvaluator(
            experiment=experiment,
            dataset=val_ds,
            asset_list=eval_assets,
            confidence_threshold=0.1,
        ).evaluate()
        return

    if inference_type == InferenceType.SEGMENTATION:
        SegmentationTensorflowEvaluator(
            experiment=experiment,
            dataset=val_ds,
            asset_list=eval_assets,
            confidence_threshold=0.1,
        ).evaluate()
        return

    print(
        "The only supported inference types for evaluation are object detection and segmentation. "
        "Please add inference type to model if you haven't already."
    )


@step()
def train(picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]):
    context = Pipeline.get_active_context()
    experiment = getattr(context, "experiment", None)

    train_ds, val_ds, test_ds = _load_splits(picsellia_datasets)
    _, labelmap, label_path = _prepare_labelmap_and_logs(
        experiment, picsellia_model, train_ds, val_ds, test_ds
    )

    record_dir = _create_records(
        context, picsellia_model, train_ds, val_ds, test_ds, label_path
    )

    training_config_dir, eval_config_dir, _ = _prepare_configs(picsellia_model)

    if not picsellia_model.pretrained_weights_path:
        raise RuntimeError("No pretrained weights found (checkpoint zip or dir).")

    checkpoint_dir = picsellia_model.weights_dir
    print("✅ checkpoint_dir:", checkpoint_dir)

    _edit_configs(
        context=context,
        record_dir=record_dir,
        checkpoint_dir=checkpoint_dir,
        training_config_dir=training_config_dir,
        eval_config_dir=eval_config_dir,
        label_path=label_path,
    )

    results_dir, exported_model_dir = _run_train_and_export(
        context=context,
        picsellia_model=picsellia_model,
        experiment=experiment,
        training_config_dir=training_config_dir,
    )

    _experiment_store_like_before(
        experiment=experiment,
        picsellia_model=picsellia_model,
        training_config_dir=training_config_dir,
        results_dir=results_dir,
        exported_model_dir=exported_model_dir,
    )

    _compute_metrics_and_confusion(
        experiment=experiment,
        picsellia_model=picsellia_model,
        record_dir=record_dir,
        eval_config_dir=eval_config_dir,
        results_dir=results_dir,
        exported_model_dir=exported_model_dir,
        labelmap=labelmap,
    )

    _run_optional_evaluators(experiment=experiment, val_ds=val_ds)
