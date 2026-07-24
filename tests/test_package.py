#!/usr/bin/env python3

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_sod.fuzzy_rules import fuzzy_output
from yolo_sod.pipeline import load_method, verify_model


def check_files():
    paths = [
        ROOT / "configs/yolo_sod_p1p4.yaml",
        ROOT / "weights/wider/full_detector.pt",
        ROOT / "weights/wider/patch_detector.pt",
        ROOT / "weights/local/full_detector.pt",
        ROOT / "weights/local/patch_detector.pt",
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)


def check_configs():
    for profile in ("wider", "local"):
        method = load_method(ROOT / f"configs/{profile}")
        likelihood = method["distribution"]["likelihood"]
        if len(likelihood) != 256:
            raise RuntimeError(f"{profile} Cr likelihood must contain 256 entries")
        values = []
        fuzzy = method["fuzzy"]
        limits = fuzzy["normalization"]
        for count in limits["cluster_count"]:
            for density in limits["cluster_density"]:
                values.append(fuzzy_output(count, density, fuzzy)[0])
        if not set(values).issubset({4, 6, 8}):
            raise RuntimeError(f"{profile} fuzzy output is invalid")


def check_models():
    from ultralytics import YOLO

    verify_model(YOLO(ROOT / "configs/yolo_sod_p1p4.yaml"))
    for profile in ("wider", "local"):
        for name in ("full_detector.pt", "patch_detector.pt"):
            path = ROOT / f"weights/{profile}/{name}"
            model = YOLO(path)
            verify_model(model)


def check_summaries():
    summary = json.loads(
        (ROOT / "datasets/local_face/split_summary.json").read_text()
    )
    rows = summary["statistics"].values()
    if sum(row["images"] for row in rows) != 436:
        raise RuntimeError("Local dataset image count does not match the release split")
    if sum(row["faces"] for row in rows) != 350:
        raise RuntimeError("Local dataset face count does not match the release split")


def main():
    check_files()
    check_configs()
    check_models()
    check_summaries()
    print("Package checks passed")


if __name__ == "__main__":
    main()
