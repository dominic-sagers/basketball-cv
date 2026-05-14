"""
train.py — fine-tune a detection model from config.yaml.

Supports two backends via config.yaml training.backend:
  - "yolo"   : Ultralytics YOLOv11 fine-tuning (YOLO-format dataset)
  - "rfdetr" : RF-DETR-L fine-tuning (COCO-format dataset)

All training parameters are read from config.yaml — nothing is hardcoded here.
To switch datasets or hyperparameters, edit config.yaml and re-run.

Usage:
    python src/train.py                      # uses config.yaml in project root
    python src/train.py --config config.yaml # explicit config path
    python src/train.py --resume             # resume from last checkpoint (YOLO only)

Dataset layout:
    store/dataset/
        basketball-srfkd/          <- YOLO format  (for backend: yolo)
            data.yaml
            train/images/
            valid/images/
        basketball-srfkd-coco/     <- COCO format  (for backend: rfdetr)
            train/
                images/
                annotations.json
            val/
                images/
                annotations.json

To export the Roboflow dataset in COCO format for RF-DETR:
    1. Go to your Roboflow project → Versions → Export
    2. Select "COCO JSON" format
    3. Place under store/dataset/basketball-srfkd-coco/
    4. Set training.dataset in config.yaml to that path
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a detection model from config.yaml")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (YOLO only — RF-DETR resumes via checkpoint path)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config.yaml not found at %s — run from project root", config_path.resolve())
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    backend = train_cfg.get("backend", "yolo").lower()

    if backend == "yolo":
        _train_yolo(train_cfg, resume=args.resume)
    elif backend == "rfdetr":
        if args.resume:
            logger.warning(
                "--resume is not supported for RF-DETR via this flag. "
                "Set training.weights in config.yaml to your checkpoint path instead."
            )
        _train_rfdetr(train_cfg)
    else:
        logger.error("Unknown training.backend '%s'. Use 'yolo' or 'rfdetr'.", backend)
        sys.exit(1)


def _train_yolo(train_cfg: dict, resume: bool = False) -> None:
    """Fine-tune YOLOv11 from config.yaml training section."""
    from ultralytics import YOLO

    dataset     = train_cfg.get("dataset")
    base_model  = train_cfg.get("base_model", "yolo11m.pt")
    epochs      = train_cfg.get("epochs", 50)
    imgsz       = train_cfg.get("imgsz", 960)
    batch       = train_cfg.get("batch", 8)
    device      = train_cfg.get("device", 0)
    output_name = train_cfg.get("output_name", "basketball-ft")

    if not dataset:
        logger.error("training.dataset is not set in config.yaml")
        sys.exit(1)

    dataset_path = Path(dataset)
    if not dataset_path.exists():
        logger.error(
            "Dataset not found: %s\n"
            "  Run: dvc pull   to restore store/ from the remote\n"
            "  Or download the dataset and place it under store/dataset/<name>/",
            dataset_path.resolve(),
        )
        sys.exit(1)

    output_dir = Path("store/weights")
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        last_ckpt = output_dir / output_name / "weights" / "last.pt"
        if not last_ckpt.exists():
            logger.error("No checkpoint to resume from: %s", last_ckpt.resolve())
            sys.exit(1)
        logger.info("Resuming YOLO training from %s", last_ckpt)
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
        return

    logger.info("=== YOLOv11 training ===")
    logger.info("  Dataset    : %s", dataset)
    logger.info("  Base model : %s", base_model)
    logger.info("  Epochs     : %d", epochs)
    logger.info("  imgsz      : %d", imgsz)
    logger.info("  Batch      : %d", batch)
    logger.info("  Device     : %s", device)
    logger.info("  Output     : store/weights/%s/", output_name)

    model = YOLO(base_model)
    model.train(
        data=str(dataset_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(output_dir),
        name=output_name,
        exist_ok=True,
    )

    best = output_dir / output_name / "weights" / "best.pt"
    logger.info("Training complete. Best weights: %s", best)
    logger.info(
        "To use: set config.yaml → model.backend: yolo  and  model.weights: \"%s\"", best
    )


def _train_rfdetr(train_cfg: dict) -> None:
    """Fine-tune RF-DETR-L from config.yaml training section."""
    try:
        from rfdetr import RFDETRLarge
    except ImportError as exc:
        raise ImportError(
            "rfdetr not installed — run: pip install rfdetr"
        ) from exc

    dataset     = train_cfg.get("dataset")
    epochs      = train_cfg.get("epochs", 50)
    batch       = train_cfg.get("batch", 4)
    grad_accum  = train_cfg.get("grad_accum", 4)
    lr          = train_cfg.get("lr", 1e-4)
    device      = train_cfg.get("device", 0)
    output_name = train_cfg.get("output_name", "basketball-rfdetr-l")

    if not dataset:
        logger.error("training.dataset is not set in config.yaml")
        sys.exit(1)

    dataset_path = Path(dataset)
    if not dataset_path.exists():
        logger.error(
            "Dataset not found: %s\n"
            "  RF-DETR requires COCO JSON format, not YOLO format.\n"
            "  Export from Roboflow as 'COCO JSON' and place under store/dataset/<name>/\n"
            "  Expected layout:\n"
            "    %s/train/images/\n"
            "    %s/train/annotations.json\n"
            "    %s/val/images/\n"
            "    %s/val/annotations.json",
            dataset_path.resolve(), dataset, dataset, dataset, dataset,
        )
        sys.exit(1)

    output_dir = Path("store/weights") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== RF-DETR-L training ===")
    logger.info("  Dataset      : %s (COCO format)", dataset)
    logger.info("  Epochs       : %d", epochs)
    logger.info("  Batch        : %d  (effective: %d with grad_accum=%d)", batch, batch * grad_accum, grad_accum)
    logger.info("  LR           : %g", lr)
    logger.info("  Device       : %s", device)
    logger.info("  Output       : %s/", output_dir)

    import os
    device_str = f"cuda:{device}" if isinstance(device, int) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device) if isinstance(device, int) else "0"

    model = RFDETRLarge(pretrained=True)
    model.train(
        dataset_dir=str(dataset_path),
        epochs=epochs,
        batch_size=batch,
        grad_accum_steps=grad_accum,
        lr=lr,
        output_dir=str(output_dir),
    )

    best = output_dir / "best.pth"
    logger.info("Training complete. Best weights: %s", best)
    logger.info(
        "To use: set config.yaml → model.backend: rfdetr  and  model.weights: \"%s\"", best
    )
    logger.info(
        "Also update model.input_size to 560 and model.class_map to match your training classes."
    )


if __name__ == "__main__":
    main()
