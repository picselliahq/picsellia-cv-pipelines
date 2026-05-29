# SAM 3 Fine-tuning

Fine-tunes Meta's [SAM 3](https://huggingface.co/facebook/sam3) promptable
concept-segmentation model on a custom Picsellia segmentation dataset.

SAM 3 is text-prompted: a single forward pass segments every instance of one
concept described by a text prompt. This pipeline uses each dataset **label name**
as a prompt and fine-tunes the model so its zero-shot concepts align with your
own classes.

## How it works

`facebook/sam3` (via 🤗 `transformers`) is an inference-only DETR-style detector:
its `forward` returns `pred_logits`, `pred_boxes`, `pred_masks` and
`presence_logits` but computes **no training loss**. This pipeline therefore adds
a DETR-style set-prediction criterion (`utils/criterion.py`):

1. **Hungarian matching** between the model's queries and the ground-truth
   instances of the prompted concept (classification + L1 + GIoU cost).
2. **Losses** on the matched pairs: focal classification on the per-query concept
   score, a presence BCE, box L1 + GIoU, and mask focal + dice.

Boxes and masks are built in the **original image frame** because SAM 3's outputs
are normalized to the original size (boxes scaled by original width/height, masks
resized directly to the original size).

By default the ViT vision encoder and CLIP text encoder are **frozen**, and only
the DETR encoder/decoder, mask decoder and scoring heads are trained. This keeps
the pretrained zero-shot features and avoids backpropagating through the encoders
at 1008×1008 resolution. Set `freeze_vision_encoder` / `freeze_text_encoder` to
`false` for full fine-tuning (much higher GPU memory).

## Pipeline steps

1. `load_coco_datasets` — download the attached dataset versions (`train` / `test`).
2. `load_sam3_model` — load `facebook/sam3` weights and processor from Hugging Face.
3. `train_sam3_model` — fine-tune and store the model archive (`model-latest`) on
   the experiment.
4. `evaluate_sam3_model` — run the fine-tuned model on the test split and submit
   polygon predictions for evaluation in Picsellia.

## Hugging Face access

`facebook/sam3` is a gated model. Provide a token with access via the
`HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) environment variable, or a `.env` file in
this directory containing `HF_TOKEN=...`.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | `10` | Number of training epochs. |
| `batch_size` | `2` | Images per batch (1008×1008 inputs are memory-heavy). |
| `learning_rate` | `1e-4` | AdamW learning rate. |
| `weight_decay` | `1e-4` | AdamW weight decay. |
| `grad_accumulation_steps` | `1` | Optimizer-step accumulation. |
| `max_grad_norm` | `1.0` | Gradient clipping norm. |
| `warmup_ratio` | `0.0` | Fraction of steps for linear LR warmup (then cosine decay). |
| `freeze_vision_encoder` | `true` | Freeze the ViT vision encoder. |
| `freeze_text_encoder` | `true` | Freeze the CLIP text encoder. |
| `class_loss_weight` | `2.0` | Weight of the focal classification loss. |
| `presence_loss_weight` | `1.0` | Weight of the presence BCE loss. |
| `bbox_loss_weight` | `5.0` | Weight of the box L1 loss. |
| `giou_loss_weight` | `2.0` | Weight of the box GIoU loss. |
| `mask_loss_weight` | `5.0` | Weight of the mask focal loss. |
| `dice_loss_weight` | `5.0` | Weight of the mask dice loss. |
| `focal_alpha` | `0.25` | Focal-loss alpha. |
| `focal_gamma` | `2.0` | Focal-loss gamma. |
| `mask_resolution` | `288` | Resolution of rasterized ground-truth masks. |
| `include_negative_samples` | `false` | Add (image, concept) pairs for absent concepts. |
| `eval_score_threshold` | `0.3` | Score threshold during evaluation. |
| `eval_mask_threshold` | `0.5` | Mask binarization threshold during evaluation. |
| `eval_min_polygon_area` | `20.0` | Minimum polygon area (px) kept during evaluation. |

## Local testing

```bash
pxl-pipeline test SAM_3 --run-config-file tests/SAM_3/run_config.toml
```
