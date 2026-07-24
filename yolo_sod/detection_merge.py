import numpy as np

from .boxes import box_iou, candidate_nms
from .cr_probability import region_contrast, region_mean
from .patch_generation import predict_probability


NOVEL_FEATURES = [
    "logit_confidence",
    "crop_rank",
    "crop_scale",
    "cr_mean",
    "cr_contrast",
    "cross_crop_agreement",
    "proposal_score",
]
REFINEMENT_FEATURES = NOVEL_FEATURES + ["full_iou", "full_confidence"]


def detection_features(full_boxes, groups, cr_map):
    if not groups:
        return (
            np.empty((0, 5)),
            np.empty((0, len(NOVEL_FEATURES))),
            np.empty((0, len(REFINEMENT_FEATURES))),
        )

    boxes = np.concatenate([group["boxes"] for group in groups])
    ranks = np.concatenate([np.full(len(group["boxes"]), group["rank"]) for group in groups])
    scales = np.concatenate([np.full(len(group["boxes"]), group["scale"]) for group in groups])
    scores = np.concatenate(
        [np.full(len(group["boxes"]), group["proposal_score"]) for group in groups]
    )
    overlap = box_iou(boxes[:, :4], boxes[:, :4])
    agreement = np.zeros(len(boxes))
    for index in range(len(boxes)):
        other = ranks != ranks[index]
        if np.any(other):
            agreement[index] = float(np.max(overlap[index, other]))

    means = np.asarray([region_mean(cr_map, box[:4]) for box in boxes])
    contrasts = np.asarray([region_contrast(cr_map, box[:4]) for box in boxes])
    raw = np.clip(boxes[:, 4], 1e-6, 1.0 - 1e-6)
    novel = np.column_stack(
        (
            np.log(raw / (1.0 - raw)),
            ranks / 8.0,
            scales,
            means,
            contrasts,
            agreement,
            scores,
        )
    )

    full_overlap = box_iou(boxes[:, :4], full_boxes[:, :4])
    if len(full_boxes):
        indices = np.argmax(full_overlap, axis=1)
        best_iou = full_overlap[np.arange(len(boxes)), indices]
        full_scores = full_boxes[indices, 4]
    else:
        best_iou = np.zeros(len(boxes))
        full_scores = np.zeros(len(boxes))
    refinement = np.column_stack((novel, best_iou, full_scores))
    keep = candidate_nms(boxes, boxes[:, 4], 0.50)
    return boxes[keep], novel[keep], refinement[keep]


def merge_detections(full_boxes, patch_boxes, novel_features, refinement_features, limit, rules):
    output = full_boxes.copy()
    statistics = {"accepted_novel": 0, "accepted_refinement": 0, "rejected": 0}
    if limit == 0 or len(patch_boxes) == 0:
        return output, statistics

    within_limit = novel_features[:, 1] <= limit / 8.0 + 1e-9
    patch_boxes = patch_boxes[within_limit].copy()
    novel_features = novel_features[within_limit]
    refinement_features = refinement_features[within_limit]
    novel_probability = predict_probability(novel_features, rules["novel_classifier"])
    refinement_probability = predict_probability(
        refinement_features, rules["refinement_classifier"]
    )
    ranking = np.maximum(novel_probability, refinement_probability)
    keep = candidate_nms(patch_boxes, ranking, 0.50)

    for box, novel_score, refinement_score in zip(
        patch_boxes[keep],
        novel_probability[keep],
        refinement_probability[keep],
    ):
        overlap = box_iou(box[None, :4], output[:, :4])[0] if len(output) else np.empty(0)
        best = int(np.argmax(overlap)) if len(overlap) else -1
        best_iou = float(overlap[best]) if len(overlap) else 0.0

        if best_iou >= rules["fusion_iou"]:
            score = refinement_score * rules["refinement_score_scale"]
            accept = (
                refinement_score >= rules["refinement_threshold"]
                and output[best, 4] < rules["reliable_full_confidence"]
                and score > output[best, 4]
            )
            if accept:
                output[best, :4] = box[:4]
                output[best, 4] = score
                statistics["accepted_refinement"] += 1
            else:
                statistics["rejected"] += 1
            continue

        score = novel_score * rules["novel_score_scale"]
        if novel_score >= rules["novel_threshold"]:
            candidate = box.copy()
            candidate[4] = score
            output = np.vstack((output, candidate))
            statistics["accepted_novel"] += 1
        else:
            statistics["rejected"] += 1

    if len(output):
        output = output[np.argsort(-output[:, 4])][:1500]
    else:
        output = np.empty((0, 5))
    return output, statistics
