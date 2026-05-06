from __future__ import annotations

import gc
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from .constants import TARGET_CLASSES, YOLO_CLASS_NAMES
from .metrics import compute_class_level_iou, compute_iou_for_pipeline, load_gt_mask
from .visibility import VisibilityCondition, detect_visibility_condition, preprocess_for_visibility


class YoloEfficientSAM3Pipeline:
    """Research pipeline refactored from Yolo_SAM3_improve-3.ipynb."""

    def __init__(
        self,
        sam_builder,
        sam_processor_cls,
        sam_checkpoint: str,
        yolo_model: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        backbone_type: str = "efficientvit",
        model_name: str = "b0",
        cache_dir: str = "./cache",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.yolo = YOLO(yolo_model).to(self.device)
        if torch.cuda.is_available() and hasattr(self.yolo, "model"):
            self.yolo.model.half()

        self.sam = sam_builder(
            checkpoint_path=sam_checkpoint,
            backbone_type=backbone_type,
            model_name=model_name,
            enable_inst_interactivity=True,
        ).to(self.device).eval()

        self.processor = sam_processor_cls(self.sam)
        self.confidence_threshold = confidence_threshold

    def _get_cached_path(self, source_path: str) -> str:
        local_path = os.path.join(self.cache_dir, Path(source_path).name)
        if not os.path.exists(local_path):
            shutil.copy2(source_path, local_path)
        return local_path

    def preload_images(self, image_paths: List[str]) -> None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(self._get_cached_path, image_paths))

    @staticmethod
    def _process_mask(mask: np.ndarray) -> np.ndarray:
        mask = np.squeeze(mask)
        if mask.ndim == 3:
            mask = mask[0] if mask.shape[0] == 3 else mask[:, :, 0]
        if mask.dtype == bool:
            return mask.astype(np.uint8)
        if mask.dtype in [np.float32, np.float64, float]:
            return (mask > 0.5).astype(np.uint8)
        return (mask > 127).astype(np.uint8)

    @staticmethod
    def _resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_resized = mask_pil.resize(target_size, Image.NEAREST)
        return (np.array(mask_resized) > 127).astype(np.uint8)

    def _process_sam_result(self, result, num_objects: int, original_size: Tuple[int, int], yolo_confs: List[float]):
        masks_list: List[np.ndarray] = []
        scores_list: List[float] = []

        if isinstance(result, (tuple, list)) and len(result) >= 2:
            all_masks, all_scores = result[0], result[1]
        else:
            all_masks, all_scores = result, None

        if all_masks is None:
            return [], []

        if isinstance(all_masks, torch.Tensor):
            all_masks = all_masks.cpu().numpy()
        if isinstance(all_scores, torch.Tensor):
            all_scores = all_scores.cpu().numpy()

        num_masks_per_obj = all_masks.shape[1] if isinstance(all_masks, np.ndarray) and all_masks.ndim == 4 else 1

        for i in range(num_objects):
            best_mask, best_score = None, 0.0
            for j in range(num_masks_per_obj):
                mask = all_masks[i, j] if (isinstance(all_masks, np.ndarray) and all_masks.ndim == 4) else all_masks
                score = yolo_confs[i]
                if all_scores is not None and isinstance(all_scores, np.ndarray) and all_scores.ndim == 2:
                    if i < all_scores.shape[0] and j < all_scores.shape[1]:
                        score = float(all_scores[i, j])

                mask_processed = self._process_mask(mask)
                if score > best_score:
                    best_score = score
                    best_mask = mask_processed

            if best_mask is not None:
                masks_list.append(self._resize_mask(best_mask, original_size))
                scores_list.append(best_score)
            else:
                masks_list.append(np.zeros((original_size[1], original_size[0]), dtype=np.uint8))
                scores_list.append(yolo_confs[i])

        return masks_list, scores_list

    def process_single(self, image_path: str, preprocess: bool = False):
        local_path = self._get_cached_path(image_path)
        metrics = {"yolo_time": 0.0, "sam_time": 0.0, "num_objects": 0}

        image_np = np.array(Image.open(local_path).convert("RGB"))
        if preprocess:
            condition = detect_visibility_condition(image_np)
            image_np = preprocess_for_visibility(image_np, condition)
            metrics["visibility"] = condition.value

        image = Image.fromarray(image_np)
        original_size = image.size

        t0 = time.perf_counter()
        results = self.yolo(image_np, conf=self.confidence_threshold, verbose=False)
        boxes = results[0].boxes
        metrics["yolo_time"] = time.perf_counter() - t0

        if boxes is None or len(boxes) == 0:
            return [], [], [], metrics, []

        all_bboxes: List[List[float]] = []
        classes_list: List[str] = []
        yolo_confs: List[float] = []

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = YOLO_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()

            all_bboxes.append(coords)
            classes_list.append(cls_name)
            yolo_confs.append(conf)

        metrics["num_objects"] = len(all_bboxes)

        t0 = time.perf_counter()
        inference_state = self.processor.set_image(image)
        with torch.inference_mode():
            result = self.sam.predict_inst(inference_state, box=all_bboxes)
        masks_list, scores_list = self._process_sam_result(result, len(all_bboxes), original_size, yolo_confs)
        metrics["sam_time"] = time.perf_counter() - t0

        return masks_list, classes_list, scores_list, metrics, all_bboxes

    def process_batch_with_iou(self, image_paths: List[str], gt_mask_paths: List[Optional[str]], preprocess: bool = False) -> Dict:
        self.preload_images(image_paths)

        stats = {
            "total_images": len(image_paths),
            "successful": 0,
            "failed": 0,
            "total_objects": 0,
            "total_time": 0.0,
            "yolo_time": 0.0,
            "sam_time": 0.0,
            "all_ious": [],
            "miou_values": [],
            "class_miou_values": [],
            "class_iou_per_class": {cls: [] for cls in TARGET_CLASSES},
            "images_with_iou": 0,
            "visibility_stats": {c.value: {"count": 0, "miou": []} for c in VisibilityCondition},
        }

        if image_paths:
            _ = self.process_single(image_paths[0], preprocess=preprocess)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        start = time.perf_counter()
        for img_path, gt_path in zip(image_paths, gt_mask_paths):
            try:
                masks, classes, _scores, m, _ = self.process_single(img_path, preprocess=preprocess)
                stats["successful"] += 1
                stats["total_objects"] += m["num_objects"]
                stats["yolo_time"] += m["yolo_time"]
                stats["sam_time"] += m["sam_time"]

                if gt_path and os.path.exists(gt_path) and masks:
                    gt_mask = load_gt_mask(gt_path, Image.open(self._get_cached_path(img_path)).size)
                    iou_metrics = compute_iou_for_pipeline(masks, classes, gt_mask)
                    stats["all_ious"].extend(iou_metrics["ious"])
                    stats["miou_values"].append(iou_metrics["miou"])

                    class_ious, class_miou = compute_class_level_iou(masks, classes, gt_mask, TARGET_CLASSES)
                    stats["class_miou_values"].append(class_miou)
                    stats["images_with_iou"] += 1
                    for cls in TARGET_CLASSES:
                        if cls in class_ious and any(c == cls for c in classes):
                            stats["class_iou_per_class"][cls].append(class_ious[cls])

                    if m.get("visibility"):
                        vis = m["visibility"]
                        stats["visibility_stats"][vis]["count"] += 1
                        stats["visibility_stats"][vis]["miou"].append(class_miou)
            except Exception:
                stats["failed"] += 1
            gc.collect()

        stats["total_time"] = time.perf_counter() - start

        if stats["successful"] > 0:
            stats["fps"] = stats["successful"] / stats["total_time"]
            stats["avg_yolo_ms"] = stats["yolo_time"] / stats["successful"] * 1000
            stats["avg_sam_ms"] = stats["sam_time"] / stats["successful"] * 1000
            stats["avg_objects"] = stats["total_objects"] / stats["successful"]
            stats["class_miou_avg"] = float(np.mean(stats["class_miou_values"])) if stats["class_miou_values"] else 0.0
            stats["overall_miou"] = float(np.mean(stats["miou_values"])) if stats["miou_values"] else 0.0
            stats["overall_mean_iou_per_prediction"] = float(np.mean(stats["all_ious"])) if stats["all_ious"] else 0.0
            stats["mean_class_iou"] = {
                cls: (float(np.mean(vals)) if vals else 0.0)
                for cls, vals in stats["class_iou_per_class"].items()
            }

        return stats

    def cleanup(self) -> None:
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
