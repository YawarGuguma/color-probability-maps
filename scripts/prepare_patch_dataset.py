#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_sod.boxes import image_files, load_yolo_labels
from yolo_sod.patch_generation import proposal_regions
from yolo_sod.pipeline import load_method, predict_full, verify_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--full-weights", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--maximum-regions", type=int, default=8)
    parser.add_argument("--negative-regions", type=int, default=2)
    parser.add_argument("--minimum-coverage", type=float, default=0.80)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def label_path(images, labels, image_path):
    return labels / image_path.relative_to(images).with_suffix(".txt")


def face_coverage(face, region):
    left = max(face[0], region[0])
    top = max(face[1], region[1])
    right = min(face[2], region[2])
    bottom = min(face[3], region[3])
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    area = max((face[2] - face[0]) * (face[3] - face[1]), 1e-12)
    return intersection / area


def crop_labels(boxes, region, minimum_coverage):
    x1, y1, x2, y2 = region
    width = x2 - x1
    height = y2 - y1
    labels = []
    for face in boxes:
        if face_coverage(face, region) < minimum_coverage:
            continue
        left = np.clip(face[0], x1, x2) - x1
        top = np.clip(face[1], y1, y2) - y1
        right = np.clip(face[2], x1, x2) - x1
        bottom = np.clip(face[3], y1, y2) - y1
        center_x = (left + right) / (2.0 * width)
        center_y = (top + bottom) / (2.0 * height)
        box_width = (right - left) / width
        box_height = (bottom - top) / height
        labels.append((0, center_x, center_y, box_width, box_height))
    return labels


def write_labels(path, labels):
    lines = []
    for class_id, center_x, center_y, width, height in labels:
        lines.append(
            f"{class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_data_yaml(output):
    text = (
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: face\n"
    )
    (output / "patch_dataset.yaml").write_text(text)


def main():
    args = parse_args()
    from ultralytics import YOLO

    full_model = YOLO(args.full_weights)
    verify_model(full_model)
    method = load_method(args.config)
    image_output = args.output / f"images/{args.split}"
    label_output = args.output / f"labels/{args.split}"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    paths = image_files(args.images, recursive=True)
    saved_positive = 0
    saved_negative = 0

    for image_index, image_path in enumerate(paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        boxes, _ = load_yolo_labels(
            label_path(args.images, args.labels, image_path),
            image.shape,
        )
        full = predict_full(
            full_model,
            image,
            method["settings"],
            args.device,
            args.half,
        )
        regions, _, _, _ = proposal_regions(
            image,
            full,
            method["distribution"],
            method["ranker"],
        )
        negative_count = 0
        for rank, region in enumerate(regions[: args.maximum_regions], 1):
            labels = crop_labels(boxes, region, args.minimum_coverage)
            if not labels:
                if negative_count >= args.negative_regions:
                    continue
                negative_count += 1
                saved_negative += 1
            else:
                saved_positive += 1
            x1, y1, x2, y2 = region
            crop = image[y1:y2, x1:x2]
            name = f"{image_path.stem}_r{rank:02d}"
            cv2.imwrite(str(image_output / f"{name}.jpg"), crop)
            write_labels(label_output / f"{name}.txt", labels)
        if image_index % 25 == 0 or image_index == len(paths):
            print(f"Patch preparation {image_index}/{len(paths)}", flush=True)

    write_data_yaml(args.output)
    print(f"Positive patches: {saved_positive}")
    print(f"Hard-negative patches: {saved_negative}")
    print(f"Saved patch dataset to {args.output.resolve()}")


if __name__ == "__main__":
    main()
