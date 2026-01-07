import logging
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    # si le path n'existe pas, on remonte au parent (cas: .../pretrained_weights/saved_model)
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

    shutil.copy(str(p), os.path.join(dst_dir, p.name))
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


def _build_labelmap_from_label_names(label_names: List[str]) -> Dict[str, str]:
    # même convention que l'ancien script (id à partir de 1, clés en str)
    return {str(i + 1): name for i, name in enumerate(label_names)}


def _count_annotations_per_label(coco_data: dict, label_names: List[str]) -> Dict[str, int]:
    """
    Retourne un dict {label_name: count} sur base des annotations COCO.
    On s'appuie sur categories + annotations.category_id.
    """
    if not coco_data:
        return {n: 0 for n in label_names}

    cat_id_to_name = {}
    for c in coco_data.get("categories", []):
        if "id" in c and "name" in c:
            cat_id_to_name[int(c["id"])] = c["name"]

    counts = {n: 0 for n in label_names}
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

    # garantit l'ordre/présence
    for n in label_names:
        counts.setdefault(n, 0)

    return counts


def _split_bar_payload(coco_data: dict, label_names: List[str]) -> dict:
    counts = _count_annotations_per_label(coco_data, label_names)
    return {"x": label_names, "y": [counts.get(n, 0) for n in label_names]}


def _experiment_log(experiment, name: str, data, chart: str, replace: bool = True) -> None:
    if experiment is None:
        return
    if hasattr(experiment, "log"):
        experiment.log(name, data, chart, replace=replace)


def _experiment_chapter(experiment, title: str) -> None:
    if experiment is None:
        return
    if hasattr(experiment, "start_logging_chapter"):
        experiment.start_logging_chapter(title)


def _experiment_buffer_start(experiment, size: int = 9) -> None:
    if experiment is None:
        return
    if hasattr(experiment, "start_logging_buffer"):
        experiment.start_logging_buffer(size)


def _experiment_buffer_end(experiment) -> None:
    if experiment is None:
        return
    if hasattr(experiment, "end_logging_buffer"):
        experiment.end_logging_buffer()


def _experiment_store_like_before(
    experiment,
    picsellia_model: Model,
    training_config_dir: str,
    results_dir: str,
    exported_model_dir: str,
) -> None:
    if experiment is not None and hasattr(experiment, "store"):
        # Même appels que l'ancien script
        experiment.store("model-latest")
        experiment.store("config")
        experiment.store("checkpoint-data-latest")
        experiment.store("checkpoint-index-latest")
        return

    # Fallback propre si pas de store dispo
    # (tu voulais “comme avant”, mais au moins ça ne bloque pas)
    print("⚠️ context.experiment.store() non disponible → fallback upload d'artifacts.")
    try:
        # saved_model
        saved_model_path = os.path.join(exported_model_dir, "saved_model")
        if os.path.isdir(saved_model_path):
            tar_path = _tar_dir(saved_model_path, os.path.join(picsellia_model.results_dir, "model-latest.tar.gz"))
            picsellia_model.save_artifact_to_experiment(
                experiment=experiment,
                artifact_name="model-latest",
                artifact_path=tar_path,
            )

        # config
        tar_cfg = _tar_dir(training_config_dir, os.path.join(picsellia_model.results_dir, "config.tar.gz"))
        picsellia_model.save_artifact_to_experiment(
            experiment=experiment,
            artifact_name="config",
            artifact_path=tar_cfg,
        )

        # checkpoints
        tar_ckpt = _tar_dir(results_dir, os.path.join(picsellia_model.results_dir, "checkpoints.tar.gz"))
        picsellia_model.save_artifact_to_experiment(
            experiment=experiment,
            artifact_name="checkpoints",
            artifact_path=tar_ckpt,
        )
    except Exception as e:
        print("⚠️ fallback artifact upload failed:", repr(e))


@step()
def train(picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]):
    context = Pipeline.get_active_context()
    experiment = getattr(context, "experiment", None)

    # ---- 1) COCO splits (train/test/val|eval)
    train_ds = _get_split(picsellia_datasets, "train")
    val_ds = _get_split(picsellia_datasets, "val")
    test_ds = _get_split(picsellia_datasets, "test")

    train_ds.load_coco_file_data()
    val_ds.load_coco_file_data()
    test_ds.load_coco_file_data()

    if not train_ds.images_dir or not val_ds.images_dir or not test_ds.images_dir:
        raise RuntimeError("images_dir is missing on one of the datasets (train/val/test).")

    # ---- 2) label map pbtxt + logs (comme avant)
    label_names = list(train_ds.labelmap.keys()) if train_ds.labelmap else []
    if not label_names:
        cats = (train_ds.coco_data or {}).get("categories", [])
        label_names = [c["name"] for c in cats]

    labelmap = _build_labelmap_from_label_names(label_names)
    label_path = pxl_utils.generate_label_map(
        classes=label_names,
        output_path=picsellia_model.results_dir,
    )

    # logs identiques
    _experiment_log(experiment, "labelmap", labelmap, "labelmap", replace=True)
    _experiment_log(experiment, "train-split", pxl_utils.sort_split(_split_bar_payload(train_ds.coco_data, label_names), label_names), "bar", replace=True)
    _experiment_log(experiment, "eval-split",  pxl_utils.sort_split(_split_bar_payload(val_ds.coco_data, label_names), label_names), "bar", replace=True)
    _experiment_log(experiment, "test-split",  pxl_utils.sort_split(_split_bar_payload(test_ds.coco_data, label_names), label_names), "bar", replace=True)

    print("\n")
    _experiment_chapter(experiment, "Create records")

    # ---- 3) TFRecords (train/test/eval) (comme avant)
    record_dir = os.path.join(picsellia_model.results_dir, "records")
    _safe_mkdir(record_dir)

    pxl_utils.create_record_files(
        train_annotations=train_ds.coco_data,
        eval_annotations=val_ds.coco_data,   # "eval" = val split
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

    # ---- 4) pipeline.config (training_config_dir + eval_config) (comme avant)
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

    # eval_config dir séparé + copie pipeline.config (exactement comme avant)
    eval_config_dir = os.path.join(picsellia_model.results_dir, "eval_config")
    _safe_mkdir(eval_config_dir)
    shutil.copy(pipeline_config_path, os.path.join(eval_config_dir, "pipeline.config"))

    # ---- 5) pretrained weights
    if not picsellia_model.pretrained_weights_path:
        raise RuntimeError("No pretrained weights found (checkpoint zip or dir).")

    # IMPORTANT: ton build_model te met déjà les poids dans ces dossiers.
    checkpoint_dir = picsellia_model.weights_dir  # contient ckpt-0.index + ckpt-0.data...
    print("✅ checkpoint_dir:", checkpoint_dir)

    # ---- 6) edit training config (train_record + test_record) (comme avant)
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
        parameters=context.hyperparameters.model_dump()
        if hasattr(context.hyperparameters, "model_dump")
        else vars(context.hyperparameters),
    )

    # ---- 6bis) edit final test config (train_record + eval_record) (comme avant)
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
        parameters=context.hyperparameters.model_dump()
        if hasattr(context.hyperparameters, "model_dump")
        else vars(context.hyperparameters),
    )

    print("\n")
    _experiment_chapter(experiment, "Start training")

    # ---- 7) train (log_real_time=experiment) (comme avant)
    results_dir = os.path.join(picsellia_model.results_dir, "tf2_results")
    _safe_mkdir(results_dir)

    pxl_utils.train(
        model_dir=results_dir,
        config_dir=training_config_dir,
        log_real_time=experiment,  # <= important: identique à l'ancien
        evaluate_fn=pxl_utils.evaluate,
        log_metrics=pxl_utils.log_metrics,
        checkpoint_every_n=context.hyperparameters.checkpoint_every_n,
    )

    print("\n")
    _experiment_chapter(experiment, "Store artifacts")

    # ---- 8) export SavedModel (comme avant)
    exported_model_dir = os.path.join(picsellia_model.results_dir, "exported_model")
    _safe_mkdir(exported_model_dir)

    pxl_utils.export_graph(
        ckpt_dir=results_dir,
        exported_model_dir=exported_model_dir,
        config_dir=training_config_dir,
    )

    # ---- 9) store artifacts EXACTEMENT comme avant si possible
    _experiment_store_like_before(
        experiment=experiment,
        picsellia_model=picsellia_model,
        training_config_dir=training_config_dir,
        results_dir=results_dir,
        exported_model_dir=exported_model_dir,
    )

    print("\n")
    _experiment_chapter(experiment, "Computing metrics on test dataset")

    _experiment_buffer_start(experiment, 9)

    # ---- 10) eval TF2 (comme avant)
    eval_metrics_dir = os.path.join(picsellia_model.results_dir, "eval_metrics")
    _safe_mkdir(eval_metrics_dir)

    pxl_utils.evaluate(
        metrics_dir=eval_metrics_dir,
        config=eval_config_dir,     # <= le dossier eval_config
        ckpt_dir=results_dir,
    )

    metrics = pxl_utils.tf_events_to_dict(os.path.join(eval_metrics_dir, "eval"), "eval")
    _experiment_log(experiment, "Evaluation/Metrics", metrics, "table", replace=True)

    # confusion matrix (comme avant)
    conf, _ = pxl_utils.get_confusion_matrix(
        input_tfrecord_path=os.path.join(record_dir, "eval.record"),
        model=os.path.join(exported_model_dir, "saved_model"),
        labelmap=labelmap,
    )
    confusion = {"categories": list(labelmap.values()), "values": conf.tolist()}
    _experiment_log(experiment, "Evaluation/confusion-matrix", confusion, "heatmap", replace=True)

    _experiment_buffer_end(experiment)

    print("\n")
    _experiment_chapter(experiment, "Starting Evaluation")

    # ---- 11) Evaluator (comme avant) – best effort (imports optionnels)
    try:
        from picsellia.types.enums import InferenceType
        from evaluator.tf_evaluator import DetectionTensorflowEvaluator, SegmentationTensorflowEvaluator
    except Exception as e:
        print("⚠️ evaluators not available in this environment:", repr(e))
        return

    # on essaye de récupérer le type (comme avant)
    inference_type = None
    try:
        if experiment is not None and hasattr(experiment, "get_base_model_version"):
            inference_type = experiment.get_base_model_version().type
    except Exception:
        inference_type = None

    # best effort: assets list (selon l'implémentation CocoDataset)
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

    elif inference_type == InferenceType.SEGMENTATION:
        SegmentationTensorflowEvaluator(
            experiment=experiment,
            dataset=val_ds,
            asset_list=eval_assets,
            confidence_threshold=0.1,
        ).evaluate()

    else:
        print(
            "The only supported inference types for evaluation are object detection and segmentation. "
            "Please add inference type to model if you haven't already."
        )
