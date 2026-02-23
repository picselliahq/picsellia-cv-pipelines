from __future__ import annotations

import os
import sys
from argparse import Namespace

# Ensure the YOLOX vendored package is importable
_pipeline_dir = os.path.dirname(os.path.abspath(__file__))
if _pipeline_dir not in sys.path:
    sys.path.insert(0, _pipeline_dir)

import torch
from picsellia.types.enums import InferenceType
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from picsellia_cv_engine.core.models import PicselliaRectanglePrediction
from picsellia_cv_engine.core.services.model.utils import evaluate_model_impl

from utils.data import get_annotation_paths, prepare_coco_directories
from utils.steps_utils import (
    build_label_maps,
    run_inference_on_asset,
    save_and_upload_artifacts,
)

VALID_ARCHITECTURES = (
    "yolox-s",
    "yolox-m",
    "yolox-l",
    "yolox-x",
    "yolox-tiny",
    "yolox-nano",
)


def _build_yolox_args(
    hp,
    export_params,
    data_dir: str,
    annotation_paths: dict[str, str],
    num_classes: int,
    experiment,
    pretrained_weights_path: str | None,
) -> Namespace:
    """Build the args namespace expected by YOLOX's experiment config and trainer."""
    architecture = hp.architecture
    if architecture not in VALID_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose from: {', '.join(VALID_ARCHITECTURES)}"
        )

    args = Namespace(
        name=architecture,
        experiment_name=architecture.replace("-", "_"),
        # Data paths
        data_dir=data_dir,
        train_ann=annotation_paths["train"],
        val_ann=annotation_paths["val"],
        test_ann=annotation_paths.get("test", annotation_paths["val"]),
        # Model config
        num_classes=num_classes,
        ckpt=pretrained_weights_path,
        # Training hyperparameters
        learning_rate=hp.learning_rate,
        batch_size=hp.batch_size,
        epochs=hp.epochs,
        image_size=(hp.image_size, hp.image_size),
        eval_interval=hp.eval_interval,
        enable_weather_transform=hp.enable_weather_transform,
        # Picsellia integration
        picsellia_experiment=experiment,
        # Distributed training defaults
        dist_backend="nccl",
        dist_url=None,
        devices=None,
        num_machines=1,
        machine_rank=0,
        # Training options
        exp_file=None,
        resume=False,
        start_epoch=None,
        fp16=torch.cuda.is_available(),
        cache=None,
        occupy=False,
        logger="tensorboard",
        opts=[],
    )
    return args


@step()
def train(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
) -> None:
    """Train YOLOX on COCO datasets and upload checkpoint + ONNX model."""
    ctx = Pipeline.get_active_context()
    hp = ctx.hyperparameters
    export_params = ctx.export_parameters

    print("=" * 60)
    print("YOLOX TRAINING PIPELINE")
    print("=" * 60)

    # 1. Build label maps
    print("\n[1/7] Building label maps...")
    id2label, label2id = build_label_maps(ds=picsellia_datasets["train"])
    num_classes = len(id2label)
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Labels: {list(id2label.values())}")

    logged_labelmap = {str(i): name for i, name in id2label.items()}
    ctx.experiment.log("labelmap", logged_labelmap, "labelmap", replace=True)

    # 2. Prepare COCO directory structure
    print("\n[2/7] Preparing COCO directory structure...")
    data_dir = prepare_coco_directories(picsellia_datasets)
    annotation_paths = get_annotation_paths(picsellia_datasets)
    print(f"  - Data directory: {data_dir}")
    for split, path in annotation_paths.items():
        print(f"  - {split} annotations: {path}")

    # 3. Resolve pretrained weights
    print("\n[3/7] Resolving pretrained weights...")
    pretrained_weights_path = picsellia_model.pretrained_weights_path
    if pretrained_weights_path and os.path.isfile(pretrained_weights_path):
        print(f"  - Using pretrained weights: {pretrained_weights_path}")
    else:
        pretrained_weights_path = None
        print("  - No pretrained weights found, training from scratch")

    # 4. Configure YOLOX experiment
    print(f"\n[4/7] Configuring YOLOX experiment ({hp.architecture})...")
    args = _build_yolox_args(
        hp=hp,
        export_params=export_params,
        data_dir=data_dir,
        annotation_paths=annotation_paths,
        num_classes=num_classes,
        experiment=ctx.experiment,
        pretrained_weights_path=pretrained_weights_path,
    )

    from YOLOX.yolox.exp.build import get_exp_by_name
    from YOLOX.yolox.exp import check_exp_value
    from YOLOX.yolox.utils import configure_module, get_num_devices

    configure_module()
    exp = get_exp_by_name(args)
    exp.merge(args.opts)
    check_exp_value(exp)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # Override output dir to use picsellia_model results dir
    exp.output_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name)
    os.makedirs(exp.output_dir, exist_ok=True)

    print(f"  - Architecture: {hp.architecture}")
    print(f"  - Epochs: {hp.epochs}")
    print(f"  - Batch size: {hp.batch_size}")
    print(f"  - Image size: {hp.image_size}")
    print(f"  - Learning rate: {hp.learning_rate}")
    print(f"  - Eval interval: {hp.eval_interval}")
    print(f"  - Weather augmentations: {hp.enable_weather_transform}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device: {device}")
    if device.type == "cuda":
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")

    # 5. Launch YOLOX training
    print("\n[5/7] Starting training...")
    print("-" * 60)

    from YOLOX.tools.train import main as yolox_train
    from YOLOX.yolox.core import launch

    num_gpu = get_num_devices() if args.devices is None else args.devices

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        yolox_train,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )

    print("\n" + "=" * 60)

    # 6. Save and upload artifacts (checkpoint + ONNX)
    print("[6/7] Saving and uploading model artifacts...")

    enable_dynamic_axis = getattr(export_params, "enable_dynamic_axis", False)

    # Find the trainer's best_epoch by inspecting the checkpoint
    class _TrainerProxy:
        best_epoch = "best"

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    best_ckpt_path = os.path.join(file_name, "best_ckpt.pth")
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        _TrainerProxy.best_epoch = ckpt.get("start_epoch", "best")

    save_and_upload_artifacts(
        picsellia_model=picsellia_model,
        experiment=ctx.experiment,
        exp=exp,
        args=args,
        trainer=_TrainerProxy,
        image_size=hp.image_size,
        enable_dynamic_axis=enable_dynamic_axis,
    )
    print("  - Artifacts uploaded successfully")

    # 7. Log training parameters
    print("\n[7/7] Logging training parameters...")
    ctx.experiment.log_parameters(hp.to_dict())
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)


@step()
def evaluate(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
) -> None:
    """Run inference on test split and evaluate in Picsellia."""
    ctx = Pipeline.get_active_context()
    hp = ctx.hyperparameters

    print("\n" + "=" * 60)
    print("YOLOX EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load trained model
    print("\n[1/4] Loading trained model...")

    from YOLOX.yolox.exp.build import get_exp_by_name
    from YOLOX.yolox.utils import configure_module

    configure_module()

    # Rebuild the args for getting the experiment config
    annotation_paths = get_annotation_paths(picsellia_datasets)
    data_dir = picsellia_datasets.dataset_path

    args = _build_yolox_args(
        hp=hp,
        export_params=ctx.export_parameters,
        data_dir=data_dir,
        annotation_paths=annotation_paths,
        num_classes=len(picsellia_datasets["train"].labelmap),
        experiment=ctx.experiment,
        pretrained_weights_path=None,
    )

    exp = get_exp_by_name(args)

    # Find best checkpoint
    file_name = os.path.join(
        picsellia_model.results_dir,
        picsellia_model.name,
        args.experiment_name,
    )
    ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    if not os.path.isfile(ckpt_file):
        ckpt_file = os.path.join(file_name, "last_epoch_ckpt.pth")

    print(f"  - Checkpoint: {ckpt_file}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_file, map_location=device)

    model = exp.get_model()
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"  - Model loaded on device: {device}")

    # 2. Build predictor
    print("\n[2/4] Building predictor...")
    id2label, _ = build_label_maps(picsellia_datasets["test"])
    label_names = [name for name in id2label.values()]
    prediction_device = "gpu" if device.type == "cuda" else "cpu"

    from YOLOX.tools.demo import Predictor

    yolox_predictor = Predictor(
        model=model,
        exp=exp,
        cls_names=label_names,
        device=prediction_device,
    )
    print(f"  - Predictor ready with {len(label_names)} classes")

    # 3. Run inference on test set
    print("\n[3/4] Running inference on test set...")
    ds = picsellia_datasets["test"]
    total_assets = len(ds.assets)
    print(f"  - Test set size: {total_assets} images")
    print(f"  - Confidence threshold: 0.1")

    import time

    predictions: list[PicselliaRectanglePrediction] = []
    inference_times: list[float] = []

    for i, asset in enumerate(ds.assets):
        if (i + 1) % 10 == 0 or (i + 1) == total_assets:
            print(f"  - Processing: {i + 1}/{total_assets} images...")

        start_time = time.perf_counter()
        pred = run_inference_on_asset(
            ds=ds,
            asset=asset,
            predictor=yolox_predictor,
            id2label=id2label,
            conf_thresh=0.1,
        )
        inference_times.append(time.perf_counter() - start_time)

        if pred:
            predictions.append(pred)

    mean_inference_time_ms = (
        (sum(inference_times) / len(inference_times)) * 1000
        if inference_times
        else 0.0
    )

    print(f"\n  Inference complete:")
    print(f"  - Images with predictions: {len(predictions)}/{total_assets}")
    print(f"  - Mean inference time: {mean_inference_time_ms:.2f} ms")

    from picsellia.types.enums import LogType

    ctx.experiment.log(
        name="mean_inference_time_ms", data=mean_inference_time_ms, type=LogType.VALUE
    )

    # 4. Evaluate predictions
    print("\n[4/4] Evaluating predictions...")
    if predictions:
        training_labelmap = ctx.experiment.get_log("labelmap").data
        print(
            f"  - Sending {len(predictions)} predictions to Picsellia for evaluation..."
        )
        evaluate_model_impl(
            context=ctx,
            picsellia_predictions=predictions,
            inference_type=InferenceType.OBJECT_DETECTION,
            assets=ds.assets,
            output_dir=os.path.join(ctx.working_dir, "evaluation"),
            training_labelmap=training_labelmap,
        )
        print("  - Evaluation metrics computed and logged")
    else:
        print("  - WARNING: No predictions to evaluate!")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)
