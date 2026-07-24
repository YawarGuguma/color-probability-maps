#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_sod.boxes import image_files, load_yolo_labels, save_yolo_predictions
from yolo_sod.metrics import compact, evaluate_local
from yolo_sod.pipeline import load_method, run_image, verify_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=ROOT / "datasets/local_face/local_face_dataset",
        type=Path,
    )
    parser.add_argument(
        "--full-weights",
        default=ROOT / "weights/local/full_detector.pt",
        type=Path,
    )
    parser.add_argument(
        "--patch-weights",
        default=ROOT / "weights/local/patch_detector.pt",
        type=Path,
    )
    parser.add_argument("--config", default=ROOT / "configs/local", type=Path)
    parser.add_argument("--output", default=ROOT / "results/local", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label_path(dataset, image_path):
    relative = image_path.relative_to(dataset / "images/test")
    return dataset / "labels/test" / relative.with_suffix(".txt")


def save_graphs(output, full_metrics, combined_metrics):
    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    axis.plot(
        full_metrics["recall"],
        full_metrics["precision"],
        linewidth=2.5,
        label=f"Modified P1-P4 ({full_metrics['ap50']:.4f})",
    )
    axis.plot(
        combined_metrics["recall"],
        combined_metrics["precision"],
        linewidth=2.5,
        label=f"YOLO-SOD ({combined_metrics['ap50']:.4f})",
    )
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output / "precision_recall.pdf")
    figure.savefig(output / "precision_recall.png", dpi=300)
    plt.close(figure)

    groups = ["all", "small", "medium", "large"]
    x = np.arange(len(groups))
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    axis.bar(
        x - 0.18,
        [full_metrics["by_size"][group]["ap50"] for group in groups],
        0.36,
        label="Modified P1-P4",
    )
    axis.bar(
        x + 0.18,
        [combined_metrics["by_size"][group]["ap50"] for group in groups],
        0.36,
        label="YOLO-SOD",
    )
    axis.set(
        xticks=x,
        xticklabels=[name.title() for name in groups],
        ylabel="AP@0.50",
        ylim=(0, 1),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "face_size_ap50.pdf")
    figure.savefig(output / "face_size_ap50.png", dpi=300)
    plt.close(figure)


def main():
    args = parse_args()
    from ultralytics import YOLO

    image_root = args.dataset / "images/test"
    label_root = args.dataset / "labels/test"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            "Extract the controlled local dataset before running evaluation"
        )

    full_model = YOLO(args.full_weights)
    patch_model = YOLO(args.patch_weights)
    verify_model(full_model)
    verify_model(patch_model)
    method = load_method(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    samples = []
    full_predictions = []
    combined_predictions = []
    decisions = []
    paths = image_files(image_root, recursive=True)
    for index, image_path in enumerate(paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        ground_truth, ratios = load_yolo_labels(
            label_path(args.dataset, image_path),
            image.shape,
        )
        result = run_image(
            full_model,
            patch_model,
            image,
            method,
            args.device,
            args.half,
        )
        samples.append({"ground_truth": ground_truth, "height_ratios": ratios})
        full_predictions.append(result["full"])
        combined_predictions.append(result["combined"])
        decision = result["decision"]
        decisions.append(
            {
                "image": str(image_path.relative_to(image_root)),
                "faces": len(ground_truth),
                "patches": decision["patch_count"],
                "applied_patches": len(result["regions"]),
                "reliable_regions": result["reliable_regions"],
                "cluster_count": decision["cluster_count"],
                "cluster_density": decision["cluster_density"],
                "dominant_rule": decision["dominant_rule"],
                "rule_name": decision["rule_name"],
                "accepted_novel": result["statistics"]["accepted_novel"],
                "accepted_refinement": result["statistics"]["accepted_refinement"],
                "rejected": result["statistics"]["rejected"],
            }
        )
        relative = image_path.relative_to(image_root).with_suffix(".txt")
        save_yolo_predictions(
            args.output / "predictions_full" / relative,
            result["full"],
            image.shape,
        )
        save_yolo_predictions(
            args.output / "predictions_yolo_sod" / relative,
            result["combined"],
            image.shape,
        )
        if index % 25 == 0 or index == len(paths):
            print(f"Local evaluation {index}/{len(paths)}", flush=True)

    metrics = {}
    table = []
    for name, predictions in (
        ("modified_p1p4", full_predictions),
        ("yolo_sod", combined_predictions),
    ):
        by_size = {}
        for group in ("all", "small", "medium", "large"):
            value = evaluate_local(samples, predictions, group)
            by_size[group] = compact(value)
            table.append({"method": name, "group": group, **compact(value)})
        all_values = evaluate_local(samples, predictions, "all")
        metrics[name] = {
            **compact(all_values),
            "precision": all_values["precision"],
            "recall": all_values["recall"],
            "scores": all_values["scores"],
            "by_size": by_size,
        }

    summary = {
        "dataset": "controlled local test set",
        "images": len(paths),
        "faces": sum(len(sample["ground_truth"]) for sample in samples),
        "negative_images": sum(len(sample["ground_truth"]) == 0 for sample in samples),
        "evaluation_iou": 0.50,
        "sahi": False,
        "metrics": {
            name: {
                key: value
                for key, value in row.items()
                if key not in ("precision", "recall", "scores")
            }
            for name, row in metrics.items()
        },
        "patch_selection": {
            str(count): sum(row["patches"] == count for row in decisions)
            for count in (0, 4, 6, 8)
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_csv(args.output / "metrics.csv", table)
    write_csv(args.output / "patch_decisions.csv", decisions)
    save_graphs(args.output, metrics["modified_p1p4"], metrics["yolo_sod"])

    result = metrics["yolo_sod"]
    print(f"YOLO-SOD AP@0.50: {result['ap50']:.6f}")
    print(f"Precision: {result['best_precision']:.6f}")
    print(f"Recall: {result['best_recall']:.6f}")
    print(f"F1: {result['best_f1']:.6f}")
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
