import argparse
import glob
import json
import os
from pathlib import Path

from .pipeline import YoloEfficientSAM3Pipeline


def resolve_gt_masks(image_paths, mask_dir: str):
    gt_mask_paths = []
    for img_path in image_paths:
        img_name = Path(img_path).stem
        found = None
        for suffix in ["_train_color.png", "_train_id.png", ".png"]:
            p = os.path.join(mask_dir, f"{img_name}{suffix}")
            if os.path.exists(p):
                found = p
                break
        if found is None:
            matches = glob.glob(os.path.join(mask_dir, f"{img_name}*.png"))
            found = matches[0] if matches else None
        gt_mask_paths.append(found)
    return gt_mask_paths


def main():
    parser = argparse.ArgumentParser(description="YOLO + EfficientSAM3 evaluation")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--masks", required=True, help="Path to BDD100K masks directory")
    parser.add_argument("--sam-checkpoint", required=True, help="Path to EfficientSAM3 checkpoint")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--output", default="metrics.json")
    args = parser.parse_args()

    from sam3.model_builder import build_efficientsam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    image_paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))[: args.limit]
    gt_mask_paths = resolve_gt_masks(image_paths, args.masks)

    pipeline = YoloEfficientSAM3Pipeline(
        sam_builder=build_efficientsam3_image_model,
        sam_processor_cls=Sam3Processor,
        sam_checkpoint=args.sam_checkpoint,
        yolo_model=args.yolo_model,
        confidence_threshold=args.conf,
        cache_dir=args.cache_dir,
    )

    metrics = pipeline.process_batch_with_iou(image_paths, gt_mask_paths, preprocess=args.preprocess)
    pipeline.cleanup()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "images": len(image_paths),
        "fps": metrics.get("fps", 0),
        "class_miou": metrics.get("class_miou_avg", 0),
        "yolo_ms": metrics.get("avg_yolo_ms", 0),
        "sam_ms": metrics.get("avg_sam_ms", 0),
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
