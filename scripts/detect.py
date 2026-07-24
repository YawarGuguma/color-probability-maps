#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_sod.boxes import IMAGE_SUFFIXES, image_files
from yolo_sod.pipeline import load_method, run_image, verify_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--profile", choices=("wider", "local"), default="wider")
    parser.add_argument("--output", default=ROOT / "results/detections", type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def source_images(source):
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return image_files(source, recursive=True)
    raise FileNotFoundError(source)


def draw_result(image, result, confidence):
    output = image.copy()
    for x1, y1, x2, y2 in result["regions"]:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 255), 2)
    for x1, y1, x2, y2, score in result["combined"]:
        if score < confidence:
            continue
        cv2.rectangle(
            output,
            (round(x1), round(y1)),
            (round(x2), round(y2)),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            output,
            f"{score:.2f}",
            (round(x1), max(round(y1) - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    decision = result["decision"]
    text = f"patches={decision['patch_count']} rule={decision['rule_name']}"
    cv2.putText(
        output,
        text,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main():
    args = parse_args()
    from ultralytics import YOLO

    full_weights = ROOT / f"weights/{args.profile}/full_detector.pt"
    patch_weights = ROOT / f"weights/{args.profile}/patch_detector.pt"
    full_model = YOLO(full_weights)
    patch_model = YOLO(patch_weights)
    verify_model(full_model)
    verify_model(patch_model)
    method = load_method(ROOT / f"configs/{args.profile}")
    args.output.mkdir(parents=True, exist_ok=True)

    paths = source_images(args.source)
    for index, path in enumerate(paths, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        result = run_image(
            full_model,
            patch_model,
            image,
            method,
            args.device,
            args.half,
        )
        rendered = draw_result(image, result, args.confidence)
        output_path = args.output / path.name
        cv2.imwrite(str(output_path), rendered)
        print(f"{index}/{len(paths)} {path.name}: {len(result['combined'])} raw detections")


if __name__ == "__main__":
    main()
