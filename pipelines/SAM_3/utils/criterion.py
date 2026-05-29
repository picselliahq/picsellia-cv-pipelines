from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


def box_xyxy_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU between two sets of xyxy boxes. Returns a [N, M] matrix."""
    area1 = box_xyxy_area(boxes1)
    area2 = box_xyxy_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-7)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]

    return iou - (area_c - union) / area_c.clamp(min=1e-7)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Dice loss over flattened masks. Returns the summed loss across masks."""
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).sum()


class HungarianMatcher(nn.Module):
    """Bipartite matching between SAM-3 queries and ground-truth instances."""

    def __init__(
        self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def __call__(
        self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor, tgt_boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match a single image's predictions.

        Args:
            pred_logits: (num_queries,) foreground logits for the queried concept.
            pred_boxes: (num_queries, 4) xyxy normalized boxes.
            tgt_boxes: (num_targets, 4) xyxy normalized ground-truth boxes.

        Returns:
            (query_indices, target_indices) as long tensors.
        """
        num_targets = tgt_boxes.shape[0]
        device = pred_boxes.device
        if num_targets == 0:
            empty = torch.as_tensor([], dtype=torch.long, device=device)
            return empty, empty

        prob = pred_logits.sigmoid()  # (Q,)
        cost_class = -prob[:, None].expand(-1, num_targets)
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)
        cost_giou = -generalized_box_iou(pred_boxes, tgt_boxes)

        cost = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        cost = torch.nan_to_num(cost, nan=0.0, posinf=1e4, neginf=-1e4)

        query_idx, target_idx = linear_sum_assignment(cost.cpu().numpy())
        return (
            torch.as_tensor(query_idx, dtype=torch.long, device=device),
            torch.as_tensor(target_idx, dtype=torch.long, device=device),
        )


class Sam3SetCriterion(nn.Module):
    """DETR-style set-prediction loss for fine-tuning SAM-3.

    SAM-3's ``forward`` is inference-only (no built-in loss), so we reconstruct a
    Hungarian-matched criterion over its DETR outputs: a focal classification loss
    on the per-query concept score, a presence BCE, box L1 + GIoU losses, and
    mask focal + dice losses on matched queries.
    """

    def __init__(self, hp) -> None:
        super().__init__()
        self.matcher = HungarianMatcher(
            cost_class=hp.class_loss_weight,
            cost_bbox=hp.bbox_loss_weight,
            cost_giou=hp.giou_loss_weight,
        )
        self.w_class = hp.class_loss_weight
        self.w_presence = hp.presence_loss_weight
        self.w_bbox = hp.bbox_loss_weight
        self.w_giou = hp.giou_loss_weight
        self.w_mask = hp.mask_loss_weight
        self.w_dice = hp.dice_loss_weight
        self.focal_alpha = hp.focal_alpha
        self.focal_gamma = hp.focal_gamma

    def forward(self, outputs, targets: list[dict]) -> dict[str, torch.Tensor]:
        pred_logits = outputs.pred_logits  # (B, Q)
        pred_boxes = outputs.pred_boxes  # (B, Q, 4)
        pred_masks = outputs.pred_masks  # (B, Q, h, w)
        presence_logits = outputs.presence_logits  # (B, 1)
        device = pred_logits.device
        batch_size, num_queries = pred_logits.shape

        num_boxes = sum(t["boxes"].shape[0] for t in targets)
        num_boxes = max(num_boxes, 1)

        loss_class = torch.zeros((), device=device)
        loss_presence = torch.zeros((), device=device)
        loss_bbox = torch.zeros((), device=device)
        loss_giou = torch.zeros((), device=device)
        loss_mask = torch.zeros((), device=device)
        loss_dice = torch.zeros((), device=device)

        for b in range(batch_size):
            tgt_boxes = targets[b]["boxes"].to(device)
            tgt_masks = targets[b]["masks"].to(device)
            n_tgt = tgt_boxes.shape[0]

            # Presence: does the concept appear in this image at all?
            presence_target = torch.ones((1,), device=device) if n_tgt > 0 else torch.zeros((1,), device=device)
            loss_presence = loss_presence + F.binary_cross_entropy_with_logits(
                presence_logits[b].reshape(1), presence_target
            )

            class_target = torch.zeros(num_queries, device=device)
            if n_tgt == 0:
                loss_class = loss_class + sigmoid_focal_loss(
                    pred_logits[b], class_target, self.focal_alpha, self.focal_gamma
                )
                continue

            q_idx, t_idx = self.matcher(pred_logits[b], pred_boxes[b], tgt_boxes)
            class_target[q_idx] = 1.0
            loss_class = loss_class + sigmoid_focal_loss(
                pred_logits[b], class_target, self.focal_alpha, self.focal_gamma
            )

            matched_pred_boxes = pred_boxes[b][q_idx]
            matched_tgt_boxes = tgt_boxes[t_idx]
            loss_bbox = loss_bbox + F.l1_loss(
                matched_pred_boxes, matched_tgt_boxes, reduction="sum"
            )
            giou = torch.diag(
                generalized_box_iou(matched_pred_boxes, matched_tgt_boxes)
            )
            loss_giou = loss_giou + (1 - giou).sum()

            matched_pred_masks = pred_masks[b][q_idx]  # (M, h, w)
            tgt_resized = F.interpolate(
                tgt_masks[t_idx].unsqueeze(1),
                size=matched_pred_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            loss_mask = loss_mask + sigmoid_focal_loss(
                matched_pred_masks.flatten(1),
                tgt_resized.flatten(1),
                self.focal_alpha,
                self.focal_gamma,
            ) / (matched_pred_masks.shape[-1] * matched_pred_masks.shape[-2])
            loss_dice = loss_dice + dice_loss(matched_pred_masks, tgt_resized)

        losses = {
            "loss_class": self.w_class * loss_class / num_boxes,
            "loss_presence": self.w_presence * loss_presence / batch_size,
            "loss_bbox": self.w_bbox * loss_bbox / num_boxes,
            "loss_giou": self.w_giou * loss_giou / num_boxes,
            "loss_mask": self.w_mask * loss_mask / num_boxes,
            "loss_dice": self.w_dice * loss_dice / num_boxes,
        }
        losses["loss"] = sum(losses.values())
        return losses
