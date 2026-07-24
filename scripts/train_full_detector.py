#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_sod.pipeline import verify_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default=ROOT / "configs/yolo_sod_p1p4.yaml")
    parser.add_argument("--weights")
    parser.add_argument("--standard-weights", default="yolov5nu.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--project", default=ROOT / "runs")
    parser.add_argument("--name", default="full_detector")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def transfer_module(source_model, target_model, source_index, target_index):
    source = source_model.model.model[source_index]
    target = target_model.model.model[target_index]
    target.load_state_dict(source.state_dict(), strict=True)


def initialize_model(model_yaml, standard_weights):
    from ultralytics import YOLO

    target = YOLO(model_yaml)
    target.load(standard_weights)
    source = YOLO(standard_weights)
    layer_map = [(10, 11), (13, 14), (14, 15), (17, 18), (18, 33), (20, 35)]
    for source_index, target_index in layer_map:
        transfer_module(source, target, source_index, target_index)
    return target


def main():
    args = parse_args()
    from ultralytics import YOLO

    if args.weights:
        model = YOLO(args.weights)
    else:
        model = initialize_model(args.model, args.standard_weights)
    verify_model(model)
    options = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "patience": args.patience,
        "amp": True,
        "cache": False,
        "seed": args.seed,
        "deterministic": True,
        "close_mosaic": 10,
        "max_det": 1000,
        "val": True,
        "plots": True,
        "save": True,
        "save_period": -1,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
    }
    if args.learning_rate is not None:
        options["lr0"] = args.learning_rate
    result = model.train(
        **options,
    )
    verify_model(YOLO(Path(result.save_dir) / "weights/best.pt"))


if __name__ == "__main__":
    main()
