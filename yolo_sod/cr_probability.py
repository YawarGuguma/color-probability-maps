import json
from pathlib import Path

import cv2
import numpy as np


def load_distribution(path):
    return json.loads(Path(path).read_text())


def normalize_map(values):
    maximum = float(values.max()) if values.size else 0.0
    if maximum <= 0.0:
        return values
    return values / maximum


def cr_likelihood_map(image, distribution):
    cr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 1]
    likelihood = np.asarray(distribution["likelihood"], dtype=np.float32)
    return likelihood[cr]


def region_mean(values, box):
    height, width = values.shape
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(np.ceil(x2), x1 + 1, width))
    y2 = int(np.clip(np.ceil(y2), y1 + 1, height))
    return float(values[y1:y2, x1:x2].mean())


def region_contrast(values, box):
    height, width = values.shape
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * 0.25
    pad_y = (y2 - y1) * 0.25
    outer = (
        max(x1 - pad_x, 0),
        max(y1 - pad_y, 0),
        min(x2 + pad_x, width),
        min(y2 + pad_y, height),
    )
    return region_mean(values, box) - region_mean(values, outer)


def face_background_masks(image_shape, boxes):
    height, width = image_shape[:2]
    face = np.zeros((height, width), dtype=np.uint8)
    excluded = np.zeros_like(face)
    for x1, y1, x2, y2 in boxes:
        center = (round((x1 + x2) / 2.0), round((y1 + y2) / 2.0))
        axes = (max(round((x2 - x1) * 0.34), 1), max(round((y2 - y1) * 0.38), 1))
        cv2.ellipse(face, center, axes, 0, 0, 360, 1, -1)
        pad_x = (x2 - x1) * 0.4
        pad_y = (y2 - y1) * 0.4
        left = max(round(x1 - pad_x), 0)
        top = max(round(y1 - pad_y), 0)
        right = min(round(x2 + pad_x), width)
        bottom = min(round(y2 + pad_y), height)
        excluded[top:bottom, left:right] = 1
    return face.astype(bool), ~excluded.astype(bool)


def smooth_histogram(values):
    kernel = np.asarray([1, 2, 3, 4, 3, 2, 1], dtype=np.float64)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def build_distribution(face_histogram, background_histogram):
    face_pdf = smooth_histogram(face_histogram)
    background_pdf = smooth_histogram(background_histogram)
    face_pdf /= face_pdf.sum()
    background_pdf /= background_pdf.sum()
    log_ratio = np.log(face_pdf / np.maximum(background_pdf, 1e-12))
    center = float(np.median(log_ratio))
    scale = max(float(np.std(log_ratio)), 1e-6)
    likelihood = 1.0 / (1.0 + np.exp(-np.clip((log_ratio - center) / scale, -20, 20)))

    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        selected = likelihood >= threshold
        score = float(face_pdf[selected].sum() - background_pdf[selected].sum())
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score
    return face_pdf, background_pdf, likelihood, best_threshold
"""Cr probability functions. SPDX-License-Identifier: MIT."""
