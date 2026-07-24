import json
from pathlib import Path

import numpy as np

from .boxes import map_patch_boxes, model_boxes
from .cr_probability import load_distribution
from .detection_merge import detection_features, merge_detections
from .fuzzy_rules import nearest_patch_count, patch_decision, resize_regions
from .patch_generation import proposal_regions


def load_json(path):
    return json.loads(Path(path).read_text())


def load_method(config_directory):
    config_directory = Path(config_directory)
    return {
        "distribution": load_distribution(config_directory / "cr_distribution.json"),
        "ranker": load_json(config_directory / "region_ranker.json"),
        "rules": load_json(config_directory / "merge_rules.json"),
        "fuzzy": load_json(config_directory / "fuzzy5.json"),
        "settings": load_json(config_directory / "settings.json"),
    }


def verify_model(model):
    strides = [float(value) for value in model.model.stride.tolist()]
    if strides != [2.0, 4.0, 8.0, 16.0]:
        raise RuntimeError(f"Expected YOLO-SOD P1-P4 strides [2, 4, 8, 16], found {strides}")
    return strides


def predict_full(model, image, settings, device, half):
    result = model.predict(
        source=[image],
        imgsz=settings["full_image_size"],
        batch=1,
        conf=settings["raw_confidence"],
        iou=settings["nms_iou"],
        max_det=settings["full_max_detections"],
        device=device,
        half=half,
        verbose=False,
    )[0]
    return model_boxes(result)


def predict_patches(model, image, regions, scores, settings, device, half):
    if not regions:
        return []
    crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in regions]
    results = model.predict(
        source=crops,
        imgsz=settings["patch_image_size"],
        batch=len(crops),
        conf=settings["raw_confidence"],
        iou=settings["nms_iou"],
        max_det=settings["patch_max_detections"],
        device=device,
        half=half,
        verbose=False,
    )

    groups = []
    short_side = min(image.shape[:2])
    for rank, (region, score, result) in enumerate(zip(regions, scores, results), 1):
        boxes = map_patch_boxes(
            model_boxes(result),
            region,
            image.shape,
            settings["patch_edge_margin"],
            settings["maximum_patch_face_ratio"],
        )
        if len(boxes):
            scale = max(region[2] - region[0], region[3] - region[1]) / short_side
            groups.append(
                {
                    "boxes": boxes,
                    "rank": rank,
                    "scale": scale,
                    "proposal_score": float(score),
                }
            )
    return groups


def run_image(full_model, patch_model, image, method, device="0", half=False):
    full = predict_full(full_model, image, method["settings"], device, half)
    regions, scores, _, cr_map = proposal_regions(
        image,
        full,
        method["distribution"],
        method["ranker"],
    )
    reliable = 0
    for score in scores:
        if score >= method["rules"]["region_threshold"]:
            reliable += 1

    decision = patch_decision(cr_map, reliable, method["fuzzy"])
    selected_regions = resize_regions(regions, decision["patch_count"], image.shape)
    applied = len(selected_regions)
    groups = predict_patches(
        patch_model,
        image,
        selected_regions,
        scores[:applied],
        method["settings"],
        device,
        half,
    )
    boxes, novel_features, refinement_features = detection_features(full, groups, cr_map)
    combined, statistics = merge_detections(
        full,
        boxes,
        novel_features,
        refinement_features,
        applied,
        method["rules"],
    )
    return {
        "full": full,
        "combined": combined,
        "cr_map": cr_map,
        "regions": selected_regions,
        "region_scores": scores[:applied],
        "reliable_regions": reliable,
        "baseline_patch_count": nearest_patch_count(reliable),
        "decision": decision,
        "statistics": statistics,
    }
