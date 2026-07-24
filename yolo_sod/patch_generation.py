import cv2
import numpy as np

from .boxes import box_iou
from .cr_probability import cr_likelihood_map, normalize_map, region_contrast, region_mean
from .fuzzy_rules import square_region


REGION_FEATURES = [
    "cr_peak",
    "cr_mean",
    "cr_contrast",
    "weak_mean",
    "weak_peak",
    "combined_peak",
    "crop_scale",
]


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def predict_probability(features, classifier):
    if len(features) == 0:
        return np.empty(0, dtype=np.float64)
    mean = np.asarray(classifier["mean"])
    scale = np.asarray(classifier["scale"])
    coefficients = np.asarray(classifier["coefficients"])
    values = (features - mean) / scale
    return sigmoid(values @ coefficients + float(classifier["bias"]))


def weak_map(full_boxes, image_shape, grid_size=160, confidence=0.20, size_ratio=0.10):
    height, width = image_shape[:2]
    output = np.zeros((grid_size, grid_size), dtype=np.float32)
    centers = []
    for box in full_boxes:
        relative_width = (box[2] - box[0]) / width
        relative_height = (box[3] - box[1]) / height
        if box[4] >= confidence or max(relative_width, relative_height) > size_ratio:
            continue
        center_x = (box[0] + box[2]) * 0.5 / width * grid_size
        center_y = (box[1] + box[3]) * 0.5 / height * grid_size
        sigma = max(relative_height * grid_size, 1.5)
        radius = max(round(3.0 * sigma), 3)
        x1 = max(round(center_x) - radius, 0)
        y1 = max(round(center_y) - radius, 0)
        x2 = min(round(center_x) + radius + 1, grid_size)
        y2 = min(round(center_y) + radius + 1, grid_size)
        yy, xx = np.mgrid[y1:y2, x1:x2]
        strength = float(np.sqrt(max(box[4], 0.001) / confidence))
        output[y1:y2, x1:x2] += strength * np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2)
        )
        centers.append((center_x, center_y, relative_height, strength))
    return normalize_map(output), centers


def raw_regions(image, full_boxes, distribution, scales=(0.15, 0.20, 0.25), maximum_peaks=18):
    height, width = image.shape[:2]
    short_side = min(height, width)
    grid_size = 160
    cr_full = cr_likelihood_map(image, distribution)
    cr_grid = cv2.resize(cr_full, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    cr_grid = normalize_map(cv2.GaussianBlur(cr_grid, (0, 0), 2.0))
    weak, weak_centers = weak_map(full_boxes, image.shape, grid_size)
    combined = normalize_map(0.65 * cr_grid + 0.35 * weak)
    maxima = combined >= cv2.dilate(combined, np.ones((9, 9), dtype=np.uint8))

    peaks = []
    for y, x in np.argwhere(maxima & (combined > 0.03)):
        peaks.append((float(combined[y, x]), float(x), float(y), None))
    for center_x, center_y, relative_height, strength in weak_centers:
        x = int(np.clip(round(center_x), 0, grid_size - 1))
        y = int(np.clip(round(center_y), 0, grid_size - 1))
        peaks.append((float(combined[y, x]) + strength, center_x, center_y, relative_height))
    peaks.sort(key=lambda item: item[0], reverse=True)

    regions = []
    features = []
    for _, grid_x, grid_y, weak_height in peaks[:maximum_peaks]:
        center_x = grid_x / grid_size * width
        center_y = grid_y / grid_size * height
        candidate_scales = scales
        if weak_height is not None:
            required = weak_height * 640.0 / 48.0
            candidate_scales = [min(scales, key=lambda value: abs(value - required))]
        for scale in candidate_scales:
            region = square_region(center_x, center_y, scale * short_side, width, height)
            map_box = (
                region[0] / width * grid_size,
                region[1] / height * grid_size,
                region[2] / width * grid_size,
                region[3] / height * grid_size,
            )
            x = int(np.clip(round(grid_x), 0, grid_size - 1))
            y = int(np.clip(round(grid_y), 0, grid_size - 1))
            features.append(
                [
                    float(cr_grid[y, x]),
                    region_mean(cr_grid, map_box),
                    region_contrast(cr_grid, map_box),
                    region_mean(weak, map_box),
                    float(weak[y, x]),
                    float(combined[y, x]),
                    float(scale),
                ]
            )
            regions.append(region)
    return regions, np.asarray(features, dtype=np.float64), cr_full


def rank_regions(regions, features, ranker, maximum=8, nms_iou=0.60):
    if not regions:
        return [], [], np.empty((0, len(REGION_FEATURES)))
    probability = predict_probability(features, ranker)
    order = np.argsort(-probability)
    selected = []
    selected_scores = []
    selected_features = []
    for index in order:
        region = regions[index]
        if selected:
            overlap = box_iou(np.asarray([region]), np.asarray(selected))[0]
            if np.max(overlap) >= nms_iou:
                continue
        selected.append(region)
        selected_scores.append(float(probability[index]))
        selected_features.append(features[index])
        if len(selected) >= maximum:
            break
    return selected, selected_scores, np.asarray(selected_features)


def proposal_regions(image, full_boxes, distribution, ranker):
    regions, features, cr_map = raw_regions(image, full_boxes, distribution)
    selected, scores, selected_features = rank_regions(regions, features, ranker, 8)
    return selected, scores, selected_features, cr_map
