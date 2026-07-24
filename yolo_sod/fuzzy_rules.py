import cv2
import numpy as np


RULE_NAMES = [
    "high_count_low_density_small",
    "low_count_high_density_large",
    "medium_count_medium_density_medium",
    "high_count_high_density_medium",
    "low_count_low_density_small",
]
RULE_OUTPUTS = np.asarray([8.0, 4.0, 6.0, 6.0, 8.0], dtype=np.float64)
PATCH_SCALE = {4: 0.25, 6: 0.20, 8: 0.15}


def gaussian(value, center, width):
    width = max(float(width), 1e-6)
    return np.exp(-0.5 * ((value - center) / width) ** 2)


def normalize_value(value, limits):
    low, high = limits
    return float(np.clip((value - low) / max(high - low, 1e-9), 0.0, 1.0))


def memberships(value, parameters):
    output = []
    for name in ("low", "medium", "high"):
        output.append(gaussian(value, parameters["centers"][name], parameters["widths"][name]))
    return np.asarray(output)


def fuzzy_output(cluster_count, cluster_density, model):
    count = normalize_value(cluster_count, model["normalization"]["cluster_count"])
    density = normalize_value(cluster_density, model["normalization"]["cluster_density"])
    low_count, medium_count, high_count = memberships(count, model["membership"]["cluster_count"])
    low_density, medium_density, high_density = memberships(
        density, model["membership"]["cluster_density"]
    )
    firing = np.asarray(
        [
            high_count * low_density,
            low_count * high_density,
            medium_count * medium_density,
            high_count * high_density,
            low_count * low_density,
        ]
    )
    value = float(np.sum(firing * RULE_OUTPUTS) / max(np.sum(firing), 1e-12))
    patch_count = min((4, 6, 8), key=lambda item: abs(item - value))
    dominant = int(np.argmax(firing))
    return patch_count, value, dominant, firing


def cluster_measurements(cr_map, threshold, minimum_area_ratio=1e-5):
    height, width = cr_map.shape
    image_area = height * width
    mask = (cr_map >= threshold).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    number, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, 8)
    minimum_area = max(4, round(image_area * minimum_area_ratio))
    kept = []
    for index in range(1, number):
        if statistics[index, cv2.CC_STAT_AREA] >= minimum_area:
            kept.append(index)
    if not kept:
        return 0.0, 0.0, 0

    total_mass = 0.0
    for index in kept:
        total_mass += float(cr_map[labels == index].sum())
    megapixels = image_area / 1e6
    count = len(kept) / max(megapixels, 1e-9)
    density = total_mass / len(kept) / max(megapixels, 1e-9)
    return count, density, len(kept)


def patch_decision(cr_map, reliable_regions, model):
    count, density, raw_count = cluster_measurements(
        cr_map,
        model["component_threshold"],
        model["min_area_ratio"],
    )
    if reliable_regions < 4:
        return {
            "patch_count": 0,
            "sugeno_value": 0.0,
            "dominant_rule": -1,
            "rule_name": "activation_gate",
            "cluster_count": count,
            "cluster_density": density,
            "raw_cluster_count": raw_count,
            "firing": [0.0] * 5,
        }

    patch_count, value, dominant, firing = fuzzy_output(count, density, model)
    return {
        "patch_count": patch_count,
        "sugeno_value": value,
        "dominant_rule": dominant,
        "rule_name": RULE_NAMES[dominant],
        "cluster_count": count,
        "cluster_density": density,
        "raw_cluster_count": raw_count,
        "firing": firing.tolist(),
    }


def square_region(center_x, center_y, side, image_width, image_height):
    side = min(float(side), float(image_width), float(image_height))
    left = np.clip(center_x - side / 2.0, 0, image_width - side)
    top = np.clip(center_y - side / 2.0, 0, image_height - side)
    return tuple(map(int, (round(left), round(top), round(left + side), round(top + side))))


def resize_regions(regions, patch_count, image_shape):
    if patch_count == 0:
        return []
    height, width = image_shape[:2]
    side = PATCH_SCALE[patch_count] * min(height, width)
    output = []
    for x1, y1, x2, y2 in regions[:patch_count]:
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        output.append(square_region(center_x, center_y, side, width, height))
    return output


def nearest_patch_count(reliable_regions):
    if reliable_regions >= 8:
        return 8
    if reliable_regions >= 6:
        return 6
    if reliable_regions >= 4:
        return 4
    return 0
"""Five-rule fuzzy functions. SPDX-License-Identifier: MIT."""
