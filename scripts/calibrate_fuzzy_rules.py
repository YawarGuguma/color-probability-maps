#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_sod.boxes import image_files
from yolo_sod.cr_probability import load_distribution
from yolo_sod.fuzzy_rules import (
    cluster_measurements,
    fuzzy_output,
    nearest_patch_count,
)
from yolo_sod.patch_generation import proposal_regions
from yolo_sod.pipeline import load_json, predict_full, verify_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-images", required=True, type=Path)
    parser.add_argument("--tune-images", required=True, type=Path)
    parser.add_argument("--full-weights", required=True, type=Path)
    parser.add_argument("--distribution", required=True, type=Path)
    parser.add_argument("--ranker", required=True, type=Path)
    parser.add_argument("--merge-rules", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-name", default="custom")
    parser.add_argument("--trials", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def candidate_thresholds(distribution):
    values = [
        distribution["mask_threshold"],
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]
    return sorted({round(float(value), 4) for value in values if 0.0 < value < 1.0})


def extract_records(
    images,
    model,
    distribution,
    ranker,
    rules,
    settings,
    device,
    half,
):
    thresholds = candidate_thresholds(distribution)
    records = []
    paths = image_files(images, recursive=True)
    for index, path in enumerate(paths, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        full = predict_full(model, image, settings, device, half)
        _, scores, _, cr_map = proposal_regions(image, full, distribution, ranker)
        reliable = 0
        for score in scores:
            if score >= rules["region_threshold"]:
                reliable += 1
        features = {}
        for threshold in thresholds:
            features[str(threshold)] = cluster_measurements(cr_map, threshold)
        records.append(
            {
                "image": str(path),
                "reliable_regions": reliable,
                "target": nearest_patch_count(reliable),
                "features": features,
            }
        )
        if index % 25 == 0 or index == len(paths):
            print(f"Fuzzy calibration {index}/{len(paths)}", flush=True)
    return records, thresholds


def value_limits(values):
    low, high = np.quantile(values, (0.05, 0.95))
    if high <= low:
        high = low + 1.0
    return [float(low), float(high)]


def random_group(generator):
    centers = sorted(
        [
            generator.uniform(0.00, 0.35),
            generator.uniform(0.25, 0.75),
            generator.uniform(0.65, 1.00),
        ]
    )
    widths = generator.uniform(0.08, 0.55, 3)
    return {
        "centers": dict(zip(("low", "medium", "high"), map(float, centers))),
        "widths": dict(zip(("low", "medium", "high"), map(float, widths))),
    }


def predict_records(records, threshold, model):
    output = []
    for record in records:
        if record["reliable_regions"] < 4:
            output.append(0)
            continue
        count, density, _ = record["features"][str(threshold)]
        output.append(fuzzy_output(count, density, model)[0])
    return np.asarray(output)


def prediction_score(records, predictions):
    targets = np.asarray([record["target"] for record in records])
    accuracy = float(np.mean(targets == predictions))
    recalls = []
    for value in (4, 6, 8):
        selected = targets == value
        if np.any(selected):
            recalls.append(float(np.mean(predictions[selected] == value)))
    macro_recall = float(np.mean(recalls)) if recalls else accuracy
    score = 0.7 * accuracy + 0.3 * macro_recall
    return score, accuracy, macro_recall


def fit_model(fit_records, tune_records, thresholds, trials, seed):
    generator = np.random.default_rng(seed)
    best = None
    nonzero = [record for record in fit_records if record["reliable_regions"] >= 4]
    if not nonzero:
        raise RuntimeError("The calibration set produced no reliable patch regions")

    for threshold in thresholds:
        counts = np.asarray([record["features"][str(threshold)][0] for record in nonzero])
        densities = np.asarray(
            [record["features"][str(threshold)][1] for record in nonzero]
        )
        normalization = {
            "cluster_count": value_limits(counts),
            "cluster_density": value_limits(densities),
        }
        iterations = max(trials // len(thresholds), 1)
        for _ in range(iterations):
            model = {
                "normalization": normalization,
                "membership": {
                    "cluster_count": random_group(generator),
                    "cluster_density": random_group(generator),
                },
            }
            fit_prediction = predict_records(fit_records, threshold, model)
            tune_prediction = predict_records(tune_records, threshold, model)
            fit_score, fit_accuracy, fit_macro = prediction_score(
                fit_records,
                fit_prediction,
            )
            tune_score, tune_accuracy, tune_macro = prediction_score(
                tune_records,
                tune_prediction,
            )
            objective = 0.4 * fit_score + 0.6 * tune_score
            row = {
                "objective": objective,
                "component_threshold": threshold,
                "normalization": normalization,
                "membership": model["membership"],
                "fit_accuracy": fit_accuracy,
                "fit_macro_recall": fit_macro,
                "tune_accuracy": tune_accuracy,
                "tune_macro_recall": tune_macro,
            }
            if best is None or objective > best["objective"]:
                best = row

    best.update(
        {
            "min_area_ratio": 1e-5,
            "inference": "zero_order_sugeno",
            "activation_minimum_regions": 4,
            "patch_scale": {"4": 0.25, "6": 0.20, "8": 0.15},
            "rule_outputs": [8, 4, 6, 6, 8],
            "seed": seed,
            "trials": trials,
        }
    )
    return best


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.full_weights)
    verify_model(model)
    distribution = load_distribution(args.distribution)
    ranker = load_json(args.ranker)
    rules = load_json(args.merge_rules)
    settings = load_json(args.settings)
    fit_records, thresholds = extract_records(
        args.fit_images,
        model,
        distribution,
        ranker,
        rules,
        settings,
        args.device,
        args.half,
    )
    tune_records, _ = extract_records(
        args.tune_images,
        model,
        distribution,
        ranker,
        rules,
        settings,
        args.device,
        args.half,
    )
    fuzzy = fit_model(
        fit_records,
        tune_records,
        thresholds,
        args.trials,
        args.seed,
    )
    fuzzy["dataset"] = args.dataset_name
    fuzzy["baseline_region_threshold"] = rules["region_threshold"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fuzzy, indent=2))
    print(json.dumps(fuzzy, indent=2))
    print(f"Saved fuzzy calibration to {args.output.resolve()}")


if __name__ == "__main__":
    main()
