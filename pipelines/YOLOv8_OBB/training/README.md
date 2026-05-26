# YOLOv8 OBB — Training Pipeline

Trains a YOLOv8 Oriented Bounding Box detector on a Picsellia dataset whose
ground truth is stored as **polygons** (e.g. the output of the
`mask_to_obb` processing pipeline). Polygons are converted to YOLO-OBB
labels on the fly, the model is trained with full metric/plot logging to
the experiment, the best checkpoint is uploaded as `best-model`, and a
final test-set evaluation is pushed to the Picsellia "Evaluations" tab
as 4-point polygon predictions.

---

## What the pipeline does

```
load_coco_datasets ──► prepare_obb_dataset ──► YOLO(...).train ──► evaluate test split
        │                       │                     │                     │
        │                       │                     │                     │
  COCO json per split   labels/<split>/*.txt   per-epoch metrics    polygon predictions
  (polygons)            (YOLO OBB format)      + val plots          + COCO metrics
                                               in experiment        in Evaluations tab
```

Pipeline file: `pipeline.py`

1. **`load_coco_datasets()`** — downloads each split (`train`, `val`,
   `test`) as a `CocoDataset` with polygon ground truth.
2. **`prepare_obb_dataset()`** (`utils/data.py`) — for each annotation,
   computes the minimum-area rotated rectangle with
   `cv2.minAreaRect` + `cv2.boxPoints`, normalizes the 4 corners to
   `[0, 1]`, and writes `dataset/labels/<split>/<image_stem>.txt` in
   YOLO OBB format (`class x1 y1 x2 y2 x3 y3 x4 y4`). Also writes
   `dataset/data.yaml`.
3. **`load_ultralytics_model()`** — fetches the `pretrained-weights`
   file attached to the experiment's base model version and loads it
   into a YOLO instance.
4. **`UltralyticsObbCallbacks`** (`utils/callbacks.py`) — registered on
   the YOLO instance before training. Logs per-epoch losses, LRs,
   `mAP50(B)`, `mAP50-95(B)`, precision/recall, training/val plot
   images, and a per-class metrics table to the experiment.
5. **`YOLO(...).train(...)`** — trains the OBB model with the
   user-configured hyperparameters and augmentations.
6. **Best checkpoint upload** — `best.pt` from `runs/.../weights/` is
   stored on the experiment as the `best-model` artifact.
7. **`_evaluate_on_test_split()`** (`steps.py`) — runs inference on
   the test split with `UltralyticsObbModelPredictor`
   (`utils/predictor.py`), converts each OBB
   (`result.obb.xyxyxyxy`) into a 4-point
   `PicselliaPolygonPrediction`, and calls `evaluate_model_impl` so the
   predictions land in the Picsellia "Evaluations" tab and COCO
   metrics (mAP50, mAP50-95, mAR50-95) are logged under
   `phase="test"`.

---

## Input requirements

- **Dataset format**: Picsellia dataset version(s) with **polygon**
  annotations.
  - Attach 1 dataset → split automatically into train/val/test using
    `train_set_split_ratio`.
  - Attach 2 datasets (aliases `train`, `test`) → first is split into
    train/val.
  - Attach 3 datasets (aliases `train`, `val`, `test`) → used as-is.
- **Pretrained weights**: a YOLOv8 **OBB** checkpoint
  (`yolov8{n,s,m,l,x}-obb.pt`) attached to the experiment's base model
  version as the file named `pretrained-weights`. The five variants
  are vendored at the root of this folder for convenience.
- **Model version `inference_type`**: `SEGMENTATION` (polygons are the
  supported Picsellia type for OBB-as-polygon evaluations).

---

## Parameters

All parameters are exposed via `utils/parameters.py:TrainingHyperParameters`
(inherits the full `UltralyticsHyperParameters` surface) and
`UltralyticsAugmentationParameters`. Below is the subset most worth
tuning for OBB.

### Core training

| Param | Default | Notes |
|---|---|---|
| `epochs` | 10 | Total epochs. |
| `batch_size` | 8 | `-1` lets Ultralytics auto-batch on GPU. |
| `image_size` | 640 | Square training resolution. |
| `patience` | 100 | Early stop patience (epochs without improvement). |
| `time` | `None` | Wall-clock cap in hours (overrides `epochs` if set). |
| `device` | `"cuda:0"` | Set to `"cpu"` or `"mps"` for local runs. |
| `workers` | 8 | DataLoader workers. |
| `cache` | `false` | Cache images in RAM for speed. |
| `save_period` | 100 | Upload `best.pt` to the experiment every N epochs. The final upload always happens. |
| `validate` | `false` | Run per-epoch validation (the val callbacks rely on this). |
| `plots` | `true` | Generate training/val plots (logged to experiment). |
| `seed` | 0 | RNG seed. |
| `deterministic` | `true` | Force deterministic ops. |
| `train_set_split_ratio` | 0.8 | Only used when a single dataset is attached. |

### Optimizer & schedule

| Param | Default | Notes |
|---|---|---|
| `optimizer` | `"auto"` | `auto`, `SGD`, `Adam`, `AdamW`, `RMSProp`. |
| `lr0` | 0.01 | Initial learning rate. |
| `lrf` | 0.1 | Final LR fraction (`final = lr0 * lrf`). |
| `cos_lr` | `false` | Cosine LR schedule. |
| `momentum` | 0.937 | SGD momentum / Adam β1. |
| `weight_decay` | 0.0005 | L2 regularization. |
| `warmup_epochs` | 3.0 | Linear warmup duration. |
| `warmup_momentum` | 0.8 | Initial warmup momentum. |
| `warmup_bias_lr` | 0.1 | Initial warmup bias LR. |
| `amp` | `true` | Mixed-precision training. |
| `freeze` | `None` | Freeze first N backbone layers. |
| `rect` | `false` | Rectangular training (less padding, faster). |
| `single_cls` | `false` | Treat all classes as one. |
| `dropout` | 0.0 | Regularization dropout. |
| `fraction` | 1.0 | Use only a fraction of the dataset. |
| `nbs` | 64 | Nominal batch size for loss scaling. |
| `label_smoothing` | 0.0 | Label smoothing factor. |

### Loss gains (OBB-relevant)

| Param | Default | Notes |
|---|---|---|
| `box` | 7.5 | Oriented-box regression loss weight. |
| `cls` | 0.5 | Classification loss weight. |
| `dfl` | 1.5 | Distribution Focal Loss weight. |

### Augmentations

For OBB, **`degrees`** is the headline augmentation — set it >0 so the
model learns rotated layouts your data exhibits.

| Param | Default | Range |
|---|---|---|
| `hsv_h` / `hsv_s` / `hsv_v` | 0.015 / 0.7 / 0.4 | HSV jitter. |
| `degrees` | 0.0 | Rotation degrees (±). **Raise for OBB.** |
| `translate` | 0.1 | Fraction of image. |
| `scale` | 0.5 | Scale jitter. |
| `shear` | 0.0 | Shear degrees. |
| `perspective` | 0.0 | Perspective warp (0–0.001). |
| `flipud` / `fliplr` | 0.0 / 0.5 | Vertical / horizontal flip probability. |
| `mosaic` | 1.0 | Mosaic augmentation probability. |
| `close_mosaic` | 10 | Disable mosaic for the last N epochs. |
| `mixup` | 0.0 | MixUp probability. |
| `copy_paste` | 0.0 | Copy-Paste probability. |
| `auto_augment` | `"randaugment"` | `randaugment`, `autoaugment`, or `augmix`. |
| `erasing` | 0.4 | Random erasing probability. |

---

## Running locally

`runs/run_config.toml` controls a local invocation. Minimum required
fields:

```toml
[input.train_dataset_version]
id = "<dataset_version_uuid>"

[input.model_version]
id = "<model_version_uuid_holding_pretrained-weights>"

[output.experiment]
id = "<experiment_uuid>"

[hyperparameters]
epochs = 30
batch_size = 8
image_size = 640
device = "mps"   # or "cpu" / "cuda:0"
```

Then:

```bash
cd pipelines/YOLOv8_OBB
uv sync           # only the first time (or after pyproject changes)
python pipeline.py --mode local --config-file runs/run_config.toml
```

---

## Running on Picsellia

1. Upload one of the vendored weight files
   (`yolov8{n,s,m,l,x}-obb.pt`) as the `pretrained-weights` file of a
   model version. Set the model version `inference_type` to
   `SEGMENTATION`.
2. Build & push the Docker image declared in `config.toml`:
   ```bash
   uv lock      # if dependencies changed
   pxl-pipeline build .   # or your usual build command
   ```
3. Launch a training job from the Picsellia UI: pick the experiment,
   attach the dataset(s) with the right split aliases, and set
   hyperparameters in the form (any key from the tables above is
   accepted — defaults apply otherwise).

---

## What gets logged to the experiment

- **Per-epoch (train)**: `box_loss`, `cls_loss`, `dfl_loss`, learning
  rates, epoch time, `train_batch*` / `labels*` images.
- **Per-epoch (val)**: `val/box_loss`, `val/cls_loss`, `val/dfl_loss`,
  `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`,
  `metrics/mAP50-95(B)`, val plots (`PR_curve`, `F1_curve`,
  `val_batch*`).
- **End of training**: a per-class metrics table (`P`, `R`, `mAP50`,
  `mAP50-95`) under `metrics`.
- **End of pipeline (`phase="test"`)**:
  - The `best-model` artifact (best.pt).
  - Polygon predictions on every test asset, visible in the
    Evaluations tab.
  - COCO metrics table and scalar `mAP50(B)` / `mAP50-95(B)` /
    `mAR50-95(B)` summaries.

---

## File map

```
pipelines/YOLOv8_OBB/
├── pipeline.py              # Pipeline definition (context + 3 steps).
├── steps.py                 # `train` step + final test-set evaluation.
├── utils/
│   ├── data.py              # COCO polygons → YOLO-OBB labels + data.yaml.
│   ├── callbacks.py         # OBB-safe Picsellia training callbacks.
│   ├── predictor.py         # OBB → 4-point polygon predictions.
│   └── parameters.py        # TrainingHyperParameters (full Ultralytics surface).
├── config.toml              # Pipeline metadata, parameters class, model version.
├── pyproject.toml           # Python deps (ultralytics, opencv, numpy, pyyaml).
├── Dockerfile               # picsellia/cuda:11.8 base + uv sync.
├── runs/run_config.toml     # Local-run config (IDs + hyperparameter overrides).
└── yolov8{n,s,m,l,x}-obb.pt # Pretrained weights for each model size.
```

---

## Notes & caveats

- The engine's `UltralyticsModelTrainer` and `evaluate_ultralytics_model`
  do **not** support the `"obb"` task — they explicitly only accept
  `classify` / `detect` / `segment`. That is why this pipeline wires
  callbacks onto the YOLO instance directly (`utils/callbacks.py`) and
  ships its own OBB-aware predictor (`utils/predictor.py`).
- The `OBBValidator` does not expose `nt_per_image` / `nt_per_class`,
  so the per-class metrics table omits the count columns and only
  reports `P`, `R`, `mAP50`, `mAP50-95` from
  `validator.metrics.class_result(i)`.
- OBB predictions are stored as Picsellia **polygons** (4 points). The
  matching `InferenceType` is `SEGMENTATION`; there is no dedicated
  OBB type on the platform.
