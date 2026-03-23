"""
train.py — launch YOLOv11 fine-tuning from config.yaml.

Reads all training parameters from the `training:` section of config.yaml so
that switching datasets, base models, or hyperparameters never requires touching
this file. The only thing that changes between training runs is config.yaml.

Usage:
    python src/train.py                          # uses config.yaml in project root
    python src/train.py --config config.yaml     # explicit config path
    python src/train.py --resume                 # resume from last checkpoint

To switch dataset:
    Edit config.yaml → training.dataset to point at a different data.yaml,
    e.g. store/dataset/my-gym-footage/data.yaml
    Then re-run python src/train.py

Dataset layout (each dataset in its own subdirectory):
    store/dataset/
        basketball-srfkd/      ← Roboflow basketball-detection-srfkd
            data.yaml
            train/images/
            valid/images/
            test/images/
        my-gym-footage/        ← custom-labelled gym footage (future)
            data.yaml
            train/images/
            ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv11 from config.yaml")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (store/weights/<output_name>/weights/last.pt)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config.yaml not found at %s — run from project root", config_path.resolve())
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})

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

    if args.resume:
        last_ckpt = output_dir / output_name / "weights" / "last.pt"
        if not last_ckpt.exists():
            logger.error("No checkpoint to resume from: %s", last_ckpt.resolve())
            sys.exit(1)
        logger.info("Resuming from %s", last_ckpt)
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
        return

    logger.info("=== Training run ===")
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
        "To use: update config.yaml → model.weights: \"%s\"",
        str(output_dir / output_name / "weights" / "best.pt"),
    )


if __name__ == "__main__":
    main()
