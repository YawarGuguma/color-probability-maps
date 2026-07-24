import numpy as np

from .boxes import box_iou


def voc_ap(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(
        np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1])
    )


def size_mask(height_ratios, group):
    if group == "all":
        return np.ones(len(height_ratios), dtype=bool)
    if group == "small":
        return height_ratios <= 0.03
    if group == "medium":
        return (height_ratios > 0.03) & (height_ratios <= 0.06)
    return height_ratios > 0.06


def evaluate_local(samples, predictions, group="all", iou_threshold=0.50):
    masks = [size_mask(sample["height_ratios"], group) for sample in samples]
    total_faces = int(sum(np.count_nonzero(mask) for mask in masks))
    detections = []
    for image_index, boxes in enumerate(predictions):
        for box in boxes:
            detections.append((float(box[4]), image_index, box[:4]))
    detections.sort(key=lambda item: item[0], reverse=True)

    matched = [np.zeros(len(sample["ground_truth"]), dtype=bool) for sample in samples]
    true_positive = []
    false_positive = []
    scores = []
    for score, image_index, box in detections:
        ground_truth = samples[image_index]["ground_truth"]
        if len(ground_truth) == 0:
            true_positive.append(0.0)
            false_positive.append(1.0)
            scores.append(score)
            continue

        overlap = box_iou(box[None, :], ground_truth)[0]
        best = int(np.argmax(overlap))
        if overlap[best] >= iou_threshold and not masks[image_index][best]:
            continue
        correct = overlap[best] >= iou_threshold and not matched[image_index][best]
        if correct:
            matched[image_index][best] = True
        true_positive.append(float(correct))
        false_positive.append(float(not correct))
        scores.append(score)

    if not scores:
        return {
            "faces": total_faces,
            "ap50": 0.0,
            "max_recall": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
            "best_f1": 0.0,
            "best_confidence": 1.0,
            "precision": [],
            "recall": [],
            "scores": [],
        }

    true_positive = np.cumsum(true_positive)
    false_positive = np.cumsum(false_positive)
    recall = true_positive / max(total_faces, 1)
    precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return {
        "faces": total_faces,
        "ap50": voc_ap(recall, precision),
        "max_recall": float(recall[-1]),
        "best_precision": float(precision[best]),
        "best_recall": float(recall[best]),
        "best_f1": float(f1[best]),
        "best_confidence": float(scores[best]),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "scores": scores,
    }


def compact(metric):
    return {
        key: value
        for key, value in metric.items()
        if key not in ("precision", "recall", "scores")
    }
