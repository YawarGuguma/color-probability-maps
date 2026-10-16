#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_sod.boxes import image_files, load_yolo_labels
from yolo_sod.cr_probability import (
    build_distribution,
    face_background_masks,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=61)
    return parser.parse_args()


def label_path(images, labels, image_path):
    return labels / image_path.relative_to(images).with_suffix(".txt")


def main():
    args = parse_args()
    face_histogram = np.ones(256, dtype=np.float64)
    background_histogram = np.ones(256, dtype=np.float64)
    face_pixels = 0
    background_pixels = 0
    generator = np.random.default_rng(args.seed)
    paths = image_files(args.images, recursive=True)

    for index, image_path in enumerate(paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        boxes, _ = load_yolo_labels(
            label_path(args.images, args.labels, image_path),
            image.shape,
        )
        cr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 1]
        face_mask, background_mask = face_background_masks(image.shape, boxes)
        face_values = cr[face_mask]
        background_values = cr[background_mask]
        if len(background_values) > 10000:
            background_values = generator.choice(background_values, 10000, replace=False)
        face_histogram += np.bincount(face_values, minlength=256)
        background_histogram += np.bincount(background_values, minlength=256)
        face_pixels += len(face_values)
        background_pixels += len(background_values)
        if index % 25 == 0 or index == len(paths):
            print(f"Cr distribution {index}/{len(paths)}", flush=True)

    face_pdf, background_pdf, likelihood, threshold = build_distribution(
        face_histogram,
        background_histogram,
    )
    output = {
        "color_space": "OpenCV YCrCb",
        "channel": "Cr",
        "face_pdf": face_pdf.tolist(),
        "background_pdf": background_pdf.tolist(),
        "likelihood": likelihood.tolist(),
        "mask_threshold": threshold,
        "training_images": len(paths),
        "sampled_face_pixels": face_pixels,
        "sampled_background_pixels": background_pixels,
        "source_split": "train only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Learned Cr threshold: {threshold:.3f}")
    print(f"Saved distribution to {args.output.resolve()}")


if __name__ == "__main__":
    main()
