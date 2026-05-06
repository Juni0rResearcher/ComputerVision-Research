from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .constants import (
    BDD100K_COLOR_TO_TRAINID,
    BDD100K_TRAINID_TO_CLASS,
    CLASS_SPECIFIC_MIN_AREA,
    TARGET_CLASSES,
)


def convert_color_mask_to_trainid(color_mask: np.ndarray) -> np.ndarray:
    h, w = color_mask.shape[:2]
    trainid_mask = np.zeros((h, w), dtype=np.uint8)
    for rgb, train_id in BDD100K_COLOR_TO_TRAINID.items():
        trainid_mask[np.all(color_mask == rgb, axis=-1)] = train_id
    return trainid_mask


def load_gt_mask(mask_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    gt_mask = Image.open(mask_path).resize(target_size, Image.NEAREST)
    gt_array = np.array(gt_mask)
    if gt_array.ndim == 3 and gt_array.shape[2] >= 3:
        gt_array = convert_color_mask_to_trainid(gt_array[:, :, :3])
    elif gt_array.ndim == 3 and gt_array.shape[2] == 1:
        gt_array = gt_array[:, :, 0]
    return gt_array.astype(np.uint8)


def compute_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def get_instance_masks_from_gt(gt_mask: np.ndarray, target_classes: List[str]) -> Dict[str, List[np.ndarray]]:
    instance_masks: Dict[str, List[np.ndarray]] = {}
    for cls in target_classes:
        train_ids = [tid for tid, name in BDD100K_TRAINID_TO_CLASS.items() if name == cls]
        all_masks: List[np.ndarray] = []
        for train_id in train_ids:
            class_mask = gt_mask == train_id
            if not class_mask.any():
                continue
            if cls in ["person", "bicycle", "motorcycle"]:
                class_mask = ndimage.binary_closing(class_mask, structure=np.ones((3, 3), dtype=bool))
            labeled, num_components = ndimage.label(class_mask)
            min_area = CLASS_SPECIFIC_MIN_AREA.get(cls, 100)
            for i in range(1, num_components + 1):
                instance_mask = labeled == i
                if instance_mask.sum() >= min_area:
                    all_masks.append(instance_mask.astype(np.uint8))
        instance_masks[cls] = all_masks
    return instance_masks


def match_predictions_to_gt(
    pred_masks: List[np.ndarray],
    pred_classes: List[str],
    gt_masks_per_class: Dict[str, List[np.ndarray]],
) -> Tuple[List[float], List[Optional[int]], List[Optional[str]]]:
    if not pred_masks:
        return [], [], []

    n_pred = len(pred_masks)
    ious = np.zeros(n_pred)
    matched_indices: List[Optional[int]] = [None] * n_pred
    matched_classes: List[Optional[str]] = [None] * n_pred

    pred_by_class: Dict[str, List[int]] = {}
    for i, (_, cls) in enumerate(zip(pred_masks, pred_classes)):
        pred_by_class.setdefault(cls, []).append(i)

    for cls, pred_indices in pred_by_class.items():
        gt_masks = gt_masks_per_class.get(cls, [])
        if not gt_masks:
            continue

        n_pred_cls, n_gt_cls = len(pred_indices), len(gt_masks)
        cost_matrix = np.zeros((n_pred_cls, n_gt_cls))

        for i, pred_idx in enumerate(pred_indices):
            for j, gt_mask in enumerate(gt_masks):
                cost_matrix[i, j] = 1 - compute_iou(pred_masks[pred_idx], gt_mask)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            if r < n_pred_cls and c < n_gt_cls:
                pred_idx = pred_indices[r]
                ious[pred_idx] = 1 - cost_matrix[r, c]
                matched_indices[pred_idx] = c
                matched_classes[pred_idx] = cls

    return ious.tolist(), matched_indices, matched_classes


def compute_class_level_iou(
    pred_masks: List[np.ndarray],
    pred_classes: List[str],
    gt_mask: np.ndarray,
    target_classes: List[str],
) -> Tuple[Dict[str, float], float]:
    class_ious: Dict[str, float] = {}
    for cls in target_classes:
        pred_union = np.zeros(gt_mask.shape, dtype=bool)
        for mask, pc in zip(pred_masks, pred_classes):
            if pc == cls and mask.shape == gt_mask.shape:
                pred_union = np.logical_or(pred_union, mask)

        train_ids = [tid for tid, name in BDD100K_TRAINID_TO_CLASS.items() if name == cls]
        gt_class = np.zeros(gt_mask.shape, dtype=bool)
        for tid in train_ids:
            gt_class = np.logical_or(gt_class, gt_mask == tid)

        if pred_union.any() or gt_class.any():
            class_ious[cls] = compute_iou(pred_union.astype(np.uint8), gt_class.astype(np.uint8))
        else:
            class_ious[cls] = 1.0

    return class_ious, float(np.mean(list(class_ious.values())))


def compute_iou_for_pipeline(pred_masks: List[np.ndarray], pred_classes: List[str], gt_mask: np.ndarray) -> Dict:
    gt_instances = get_instance_masks_from_gt(gt_mask, TARGET_CLASSES)
    ious, matched_indices, matched_classes = match_predictions_to_gt(pred_masks, pred_classes, gt_instances)

    per_class_iou: Dict[str, float] = {}
    per_class_counts: Dict[str, int] = {}
    for cls in TARGET_CLASSES:
        cls_ious = [iou for iou, pc in zip(ious, pred_classes) if pc == cls]
        per_class_iou[cls] = float(np.mean(cls_ious)) if cls_ious else 0.0
        per_class_counts[cls] = len(cls_ious)

    valid_ious = [iou for iou, pc in zip(ious, pred_classes) if pc in TARGET_CLASSES]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0

    return {
        "per_class_iou": per_class_iou,
        "per_class_counts": per_class_counts,
        "miou": miou,
        "ious": ious,
        "matched_indices": matched_indices,
        "matched_classes": matched_classes,
        "total_predictions": len(ious),
        "matched_predictions": sum(1 for idx in matched_indices if idx is not None),
    }
