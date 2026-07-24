from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def image_files(directory, recursive=False):
    directory = Path(directory)
    paths = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def model_boxes(result):
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 5), dtype=np.float64)
    coordinates = result.boxes.xyxy.cpu().numpy().astype(np.float64)
    confidence = result.boxes.conf.cpu().numpy().astype(np.float64)
    return np.column_stack((coordinates, confidence))


def box_iou(boxes, reference):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0 or len(reference) == 0:
        return np.zeros((len(boxes), len(reference)), dtype=np.float64)

    left = np.maximum(boxes[:, None, 0], reference[None, :, 0])
    top = np.maximum(boxes[:, None, 1], reference[None, :, 1])
    right = np.minimum(boxes[:, None, 2], reference[None, :, 2])
    bottom = np.minimum(boxes[:, None, 3], reference[None, :, 3])
    intersection = np.maximum(right - left, 0.0) * np.maximum(bottom - top, 0.0)
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    other = np.maximum(reference[:, 2] - reference[:, 0], 0.0) * np.maximum(
        reference[:, 3] - reference[:, 1], 0.0
    )
    return intersection / np.maximum(area[:, None] + other[None, :] - intersection, 1e-12)


def candidate_nms(boxes, scores, threshold):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 5)
    order = np.argsort(-np.asarray(scores))
    keep = []
    while len(order):
        selected = int(order[0])
        keep.append(selected)
        if len(order) == 1:
            break
        overlap = box_iou(boxes[order[1:], :4], boxes[selected : selected + 1, :4])[:, 0]
        order = order[1:][overlap < threshold]
    return np.asarray(keep, dtype=np.int64)


def map_patch_boxes(local_boxes, region, image_shape, edge_margin, maximum_face_ratio):
    if len(local_boxes) == 0:
        return np.empty((0, 5), dtype=np.float64)

    x1, y1, x2, y2 = region
    patch_width = x2 - x1
    patch_height = y2 - y1
    image_height, image_width = image_shape[:2]
    keep = np.ones(len(local_boxes), dtype=bool)

    if x1 > 0:
        keep &= local_boxes[:, 0] > edge_margin
    if y1 > 0:
        keep &= local_boxes[:, 1] > edge_margin
    if x2 < image_width:
        keep &= local_boxes[:, 2] < patch_width - edge_margin
    if y2 < image_height:
        keep &= local_boxes[:, 3] < patch_height - edge_margin

    boxes = local_boxes[keep].copy()
    boxes[:, [0, 2]] += x1
    boxes[:, [1, 3]] += y1
    relative_width = (boxes[:, 2] - boxes[:, 0]) / max(image_width, 1)
    relative_height = (boxes[:, 3] - boxes[:, 1]) / max(image_height, 1)
    return boxes[np.maximum(relative_width, relative_height) <= maximum_face_ratio]


def load_yolo_labels(path, image_shape):
    path = Path(path)
    if not path.is_file() or not path.read_text().strip():
        return np.empty((0, 4), dtype=np.float64), np.empty(0, dtype=np.float64)

    labels = np.loadtxt(path, dtype=np.float64, ndmin=2).reshape(-1, 5)
    height, width = image_shape[:2]
    boxes = np.column_stack(
        (
            (labels[:, 1] - labels[:, 3] / 2.0) * width,
            (labels[:, 2] - labels[:, 4] / 2.0) * height,
            (labels[:, 1] + labels[:, 3] / 2.0) * width,
            (labels[:, 2] + labels[:, 4] / 2.0) * height,
        )
    )
    return boxes, labels[:, 4]


def save_yolo_predictions(path, boxes, image_shape):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image_shape[:2]
    lines = []
    for x1, y1, x2, y2, score in boxes:
        center_x = (x1 + x2) / (2.0 * width)
        center_y = (y1 + y2) / (2.0 * height)
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f} {score:.8f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
