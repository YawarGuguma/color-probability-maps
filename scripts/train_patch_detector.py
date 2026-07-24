#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_sod.pipeline import verify_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--project", default=ROOT / "runs")
    parser.add_argument("--name", default="patch_detector")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    verify_model(model)
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.learning_rate,
        patience=args.patience,
        amp=True,
        seed=args.seed,
        deterministic=True,
        close_mosaic=5,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        save_period=-1,
        plots=True,
    )
    verify_model(YOLO(Path(result.save_dir) / "weights/best.pt"))


if __name__ == "__main__":
    main()
