#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_sod.metrics import voc_ap
from yolo_sod.pipeline import load_method, run_image, verify_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--full-weights",
        default=ROOT / "weights/wider/full_detector.pt",
        type=Path,
    )
    parser.add_argument(
        "--patch-weights",
        default=ROOT / "weights/wider/patch_detector.pt",
        type=Path,
    )
    parser.add_argument("--config", default=ROOT / "configs/wider", type=Path)
    parser.add_argument("--output", default=ROOT / "results/wider", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def text_value(item):
    return str(item[0][0])


def load_ground_truth(directory):
    required = [
        "wider_face_val.mat",
        "wider_easy_val.mat",
        "wider_medium_val.mat",
        "wider_hard_val.mat",
    ]
    for name in required:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    validation = loadmat(directory / "wider_face_val.mat")
    return (
        validation["face_bbx_list"],
        validation["event_list"],
        validation["file_list"],
        {
            "Easy": loadmat(directory / "wider_easy_val.mat")["gt_list"],
            "Medium": loadmat(directory / "wider_medium_val.mat")["gt_list"],
            "Hard": loadmat(directory / "wider_hard_val.mat")["gt_list"],
        },
    )


def image_records(event_list, file_list, image_root):
    records = []
    for event_index in range(len(event_list)):
        event = text_value(event_list[event_index])
        for item in file_list[event_index][0]:
            image_name = text_value(item)
            image_path = image_root / event / f"{image_name}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            records.append((event, image_name, image_path))
    return records


def to_xywh(boxes):
    if len(boxes) == 0:
        return np.empty((0, 5), dtype=np.float64)
    return np.column_stack(
        (
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
            boxes[:, 4],
        )
    )


def save_wider_predictions(directory, event, image_name, boxes):
    path = directory / event / f"{image_name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{event}/{image_name}.jpg", str(len(boxes))]
    for x, y, width, height, score in boxes:
        lines.append(f"{x:.3f} {y:.3f} {width:.3f} {height:.3f} {score:.8f}")
    path.write_text("\n".join(lines) + "\n")


def normalize_scores(predictions):
    values = [boxes[:, 4] for boxes in predictions.values() if len(boxes)]
    if not values:
        return
    minimum = min(scores.min() for scores in values)
    maximum = max(scores.max() for scores in values)
    difference = maximum - minimum
    for boxes in predictions.values():
        if len(boxes):
            boxes[:, 4] = (boxes[:, 4] - minimum) / max(difference, 1e-12)


def box_overlaps(boxes, ground_truth):
    if len(boxes) == 0 or len(ground_truth) == 0:
        return np.zeros((len(boxes), len(ground_truth)), dtype=np.float64)
    left = np.maximum(boxes[:, None, 0], ground_truth[None, :, 0])
    top = np.maximum(boxes[:, None, 1], ground_truth[None, :, 1])
    right = np.minimum(boxes[:, None, 2], ground_truth[None, :, 2])
    bottom = np.minimum(boxes[:, None, 3], ground_truth[None, :, 3])
    width = np.maximum(right - left + 1.0, 0.0)
    height = np.maximum(bottom - top + 1.0, 0.0)
    intersection = width * height
    area = (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
    other = (ground_truth[:, 2] - ground_truth[:, 0] + 1.0) * (
        ground_truth[:, 3] - ground_truth[:, 1] + 1.0
    )
    return intersection / np.maximum(area[:, None] + other[None, :] - intersection, 1e-12)


def evaluate_image(predictions, ground_truth, valid_ground_truth):
    predictions = predictions.copy()
    ground_truth = ground_truth.copy()
    predictions[:, 2] += predictions[:, 0]
    predictions[:, 3] += predictions[:, 1]
    ground_truth[:, 2] += ground_truth[:, 0]
    ground_truth[:, 3] += ground_truth[:, 1]
    overlap = box_overlaps(predictions[:, :4], ground_truth)
    detected = np.zeros(len(ground_truth), dtype=np.int8)
    proposal = np.ones(len(predictions), dtype=np.int8)
    cumulative_recall = np.zeros(len(predictions), dtype=np.int32)
    for index in range(len(predictions)):
        best = int(np.argmax(overlap[index]))
        if overlap[index, best] >= 0.50:
            if not valid_ground_truth[best]:
                detected[best] = -1
                proposal[index] = -1
            elif detected[best] == 0:
                detected[best] = 1
        cumulative_recall[index] = np.count_nonzero(detected == 1)
    return cumulative_recall, proposal


def image_curve(predictions, proposal, cumulative_recall, thresholds):
    output = np.zeros((len(thresholds), 2), dtype=np.float64)
    for index, threshold in enumerate(thresholds):
        selected = np.flatnonzero(predictions[:, 4] >= threshold)
        if len(selected):
            last = selected[-1]
            output[index, 0] = np.count_nonzero(proposal[: last + 1] == 1)
            output[index, 1] = cumulative_recall[last]
    return output


def evaluate_subset(predictions, face_boxes, event_list, file_list, subset):
    thresholds = 1.0 - (np.arange(1000, dtype=np.float64) + 1.0) / 1000.0
    total_curve = np.zeros((len(thresholds), 2), dtype=np.float64)
    face_count = 0
    for event_index in range(len(event_list)):
        event = text_value(event_list[event_index])
        names = file_list[event_index][0]
        boxes = face_boxes[event_index][0]
        valid = subset[event_index][0]
        for image_index in range(len(names)):
            image_name = text_value(names[image_index])
            prediction = predictions[(event, image_name)]
            ground_truth = boxes[image_index][0].astype(np.float64)
            keep = valid[image_index][0].astype(np.int64).reshape(-1) - 1
            face_count += len(keep)
            if len(prediction) == 0 or len(ground_truth) == 0:
                continue
            valid_ground_truth = np.zeros(len(ground_truth), dtype=bool)
            valid_ground_truth[keep] = True
            cumulative_recall, proposal = evaluate_image(
                prediction,
                ground_truth,
                valid_ground_truth,
            )
            total_curve += image_curve(
                prediction,
                proposal,
                cumulative_recall,
                thresholds,
            )
    true_positive = total_curve[:, 1]
    precision = np.divide(
        true_positive,
        total_curve[:, 0],
        out=np.zeros_like(true_positive),
        where=total_curve[:, 0] > 0,
    )
    recall = true_positive / max(face_count, 1)
    return {
        "faces": face_count,
        "ap50": voc_ap(recall, precision),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def evaluate_all(predictions, face_boxes, event_list, file_list, subsets):
    values = {key: boxes.copy() for key, boxes in predictions.items()}
    normalize_scores(values)
    output = {}
    for name in ("Easy", "Medium", "Hard"):
        output[name] = evaluate_subset(
            values,
            face_boxes,
            event_list,
            file_list,
            subsets[name],
        )
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_graphs(output, full_metrics, combined_metrics):
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for axis, difficulty in zip(axes, ("Easy", "Medium", "Hard")):
        full = full_metrics[difficulty]
        combined = combined_metrics[difficulty]
        axis.plot(
            full["recall"],
            full["precision"],
            linewidth=2.5,
            label=f"Modified P1-P4 ({full['ap50']:.4f})",
        )
        axis.plot(
            combined["recall"],
            combined["precision"],
            linewidth=2.5,
            label=f"YOLO-SOD ({combined['ap50']:.4f})",
        )
        axis.set(
            title=difficulty,
            xlabel="Recall",
            ylabel="Precision",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output / "precision_recall.pdf")
    figure.savefig(output / "precision_recall.png", dpi=300)
    plt.close(figure)

    labels = ["Easy", "Medium", "Hard"]
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    axis.bar(
        x - 0.18,
        [full_metrics[name]["ap50"] for name in labels],
        0.36,
        label="Modified P1-P4",
    )
    axis.bar(
        x + 0.18,
        [combined_metrics[name]["ap50"] for name in labels],
        0.36,
        label="YOLO-SOD",
    )
    axis.set(xticks=x, xticklabels=labels, ylabel="AP@0.50", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "ap50_comparison.pdf")
    figure.savefig(output / "ap50_comparison.png", dpi=300)
    plt.close(figure)


def main():
    args = parse_args()
    from ultralytics import YOLO

    ground_truth_root = args.dataset / "eval_tools/ground_truth"
    image_root = args.dataset / "WIDER_val/images"
    face_boxes, event_list, file_list, subsets = load_ground_truth(ground_truth_root)
    records = image_records(event_list, file_list, image_root)
    full_model = YOLO(args.full_weights)
    patch_model = YOLO(args.patch_weights)
    verify_model(full_model)
    verify_model(patch_model)
    method = load_method(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    full_predictions = {}
    combined_predictions = {}
    decisions = []
    for index, (event, image_name, image_path) in enumerate(records, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        result = run_image(
            full_model,
            patch_model,
            image,
            method,
            args.device,
            args.half,
        )
        full = to_xywh(result["full"])
        combined = to_xywh(result["combined"])
        full_predictions[(event, image_name)] = full
        combined_predictions[(event, image_name)] = combined
        save_wider_predictions(
            args.output / "predictions_full",
            event,
            image_name,
            full,
        )
        save_wider_predictions(
            args.output / "predictions_yolo_sod",
            event,
            image_name,
            combined,
        )
        decision = result["decision"]
        decisions.append(
            {
                "event": event,
                "image": image_name,
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
        if index % 100 == 0 or index == len(records):
            print(f"WIDER evaluation {index}/{len(records)}", flush=True)

    full_metrics = evaluate_all(
        full_predictions,
        face_boxes,
        event_list,
        file_list,
        subsets,
    )
    combined_metrics = evaluate_all(
        combined_predictions,
        face_boxes,
        event_list,
        file_list,
        subsets,
    )
    rows = []
    for name in ("Easy", "Medium", "Hard"):
        rows.append(
            {
                "difficulty": name,
                "faces": combined_metrics[name]["faces"],
                "modified_p1p4_ap50": full_metrics[name]["ap50"],
                "yolo_sod_ap50": combined_metrics[name]["ap50"],
                "difference": combined_metrics[name]["ap50"]
                - full_metrics[name]["ap50"],
            }
        )
        with (args.output / f"pr_{name.lower()}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "threshold",
                    "full_precision",
                    "full_recall",
                    "yolo_sod_precision",
                    "yolo_sod_recall",
                ]
            )
            writer.writerows(
                zip(
                    combined_metrics[name]["thresholds"],
                    full_metrics[name]["precision"],
                    full_metrics[name]["recall"],
                    combined_metrics[name]["precision"],
                    combined_metrics[name]["recall"],
                )
            )

    summary = {
        "dataset": "WIDER FACE validation",
        "images": len(records),
        "evaluation_iou": 0.50,
        "thresholds": 1000,
        "sahi": False,
        "modified_p1p4": {
            name: {
                "faces": full_metrics[name]["faces"],
                "ap50": full_metrics[name]["ap50"],
            }
            for name in ("Easy", "Medium", "Hard")
        },
        "yolo_sod": {
            name: {
                "faces": combined_metrics[name]["faces"],
                "ap50": combined_metrics[name]["ap50"],
            }
            for name in ("Easy", "Medium", "Hard")
        },
        "patch_selection": {
            str(count): sum(row["patches"] == count for row in decisions)
            for count in (0, 4, 6, 8)
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_csv(args.output / "metrics.csv", rows)
    write_csv(args.output / "patch_decisions.csv", decisions)
    save_graphs(args.output, full_metrics, combined_metrics)
    for row in rows:
        print(f"{row['difficulty']:6s} AP@0.50: {row['yolo_sod_ap50']:.6f}")
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
