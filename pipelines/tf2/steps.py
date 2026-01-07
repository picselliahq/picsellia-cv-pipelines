import os
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

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
    """
    Si artifact_path est un zip => extract dans dst_dir, renvoie dst_dir.
    Si c'est un dossier => renvoie ce dossier.
    Si c'est un fichier => le copie dans dst_dir, renvoie dst_dir.
    """
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
    raise KeyError(f"Dataset split '{name}' not found. Available: {list(ds_col.datasets.keys())}")


@step()
def train(picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]):
    context = Pipeline.get_active_context()

    # ---- 1) COCO splits
    train_ds = _get_split(picsellia_datasets, "train")
    val_ds = _get_split(picsellia_datasets, "val")
    test_ds = _get_split(picsellia_datasets, "test")

    train_ds.load_coco_file_data()
    val_ds.load_coco_file_data()
    test_ds.load_coco_file_data()

    if not train_ds.images_dir or not val_ds.images_dir or not test_ds.images_dir:
        raise RuntimeError("images_dir is missing on one of the datasets (train/val/test).")

    # ---- 2) label map pbtxt
    label_names = list(train_ds.labelmap.keys()) if train_ds.labelmap else []
    if not label_names:
        cats = (train_ds.coco_data or {}).get("categories", [])
        label_names = [c["name"] for c in cats]

    label_path = pxl_utils.generate_label_map(classes=label_names, output_path=picsellia_model.results_dir)

    # ---- 3) TFRecords
    record_dir = os.path.join(picsellia_model.results_dir, "records")
    _safe_mkdir(record_dir)

    pxl_utils.create_record_files(
        train_annotations=train_ds.coco_data,
        eval_annotations=val_ds.coco_data,
        test_annotations=test_ds.coco_data,
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

    # ---- 4) pipeline.config
    if not picsellia_model.config_path:
        raise RuntimeError("No config file found (pipeline.config or zip).")

    config_workdir = os.path.join(picsellia_model.results_dir, "config")
    _safe_mkdir(config_workdir)
    _maybe_unpack_to_dir(picsellia_model.config_path, config_workdir)

    pipeline_config_path = None
    for root, _, files in os.walk(config_workdir):
        if "pipeline.config" in files:
            pipeline_config_path = os.path.join(root, "pipeline.config")
            break
    if not pipeline_config_path:
        raise RuntimeError("pipeline.config not found after unpacking config artifact.")

    # ---- 5) pretrained weights (checkpoint + saved_model)
    if not picsellia_model.pretrained_weights_path:
        raise RuntimeError("No pretrained weights found (checkpoint zip or dir).")

    weights_root_dir = picsellia_model.weights_dir
    ckpt_prefix = os.path.join(picsellia_model.weights_dir, "ckpt-0")
    saved_model_dir = picsellia_model.pretrained_weights_dir

    print("✅ weights_root_dir:", weights_root_dir)
    print("✅ ckpt_prefix:", ckpt_prefix)
    if saved_model_dir:
        print("✅ saved_model_dir:", saved_model_dir)

    # ---- 6) edit config (IMPORTANT: model_selected = dossier qui contient ckpt-0.index + ckpt-0.data...)
    # pxl_utils.edit_config va choisir le dernier ckpt dans model_selected
    pxl_utils.edit_config(
        model_selected=weights_root_dir,
        input_config_dir=os.path.dirname(pipeline_config_path),
        output_config_dir=os.path.dirname(pipeline_config_path),
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

    # ---- 7) train
    results_dir = os.path.join(picsellia_model.results_dir, "tf2_results")
    _safe_mkdir(results_dir)

    pxl_utils.train(
        model_dir=results_dir,
        config_dir=os.path.dirname(pipeline_config_path),
        log_real_time=getattr(context, "experiment", None),
        evaluate_fn=pxl_utils.evaluate,
        log_metrics=pxl_utils.log_metrics,
        checkpoint_every_n=context.hyperparameters.checkpoint_every_n,
    )

    # ---- 8) export SavedModel
    exported_model_dir = os.path.join(picsellia_model.results_dir, "exported_model")
    _safe_mkdir(exported_model_dir)

    pxl_utils.export_graph(
        ckpt_dir=results_dir,
        exported_model_dir=exported_model_dir,
        config_dir=os.path.dirname(pipeline_config_path),
    )

    # ---- 9) upload artifacts
    saved_model_path = os.path.join(exported_model_dir, "saved_model")
    if os.path.isdir(saved_model_path):
        tar_path = _tar_dir(saved_model_path, os.path.join(picsellia_model.results_dir, "saved_model.tar.gz"))
        picsellia_model.save_artifact_to_experiment(
            experiment=context.experiment,
            artifact_name="saved-model",
            artifact_path=tar_path,
        )

    tar_ckpt = _tar_dir(results_dir, os.path.join(picsellia_model.results_dir, "checkpoints.tar.gz"))
    picsellia_model.save_artifact_to_experiment(
        experiment=context.experiment,
        artifact_name="tf2-checkpoints",
        artifact_path=tar_ckpt,
    )

    tar_cfg = _tar_dir(os.path.dirname(pipeline_config_path), os.path.join(picsellia_model.results_dir, "config_used.tar.gz"))
    picsellia_model.save_artifact_to_experiment(
        experiment=context.experiment,
        artifact_name="tf2-config",
        artifact_path=tar_cfg,
    )
