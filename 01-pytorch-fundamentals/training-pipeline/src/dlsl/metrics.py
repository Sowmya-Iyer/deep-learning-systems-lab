"""Detection evaluation metrics."""

from collections import defaultdict


def _iou_xywh(first, second):
    """Calculate IoU between two COCO-format xywh boxes."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second

    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    intersection_width = max(
        0.0,
        min(ax + aw, bx + bw) - max(ax, bx),
    )
    intersection_height = max(
        0.0,
        min(ay + ah, by + bh) - max(ay, by),
    )

    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection

    return intersection / union

def false_positives_per_image(
    gt,
    predictions,
    score_threshold=0.5,
    iou_threshold=0.5,
):
    """Compute a simple greedy FP/image metric for non-crowd annotations."""
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1")

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")

    image_ids = {image["id"] for image in gt["images"]}
    categories = {
        category["id"]: category["name"]
        for category in gt["categories"]
    }

    if not image_ids:
        raise ValueError("Cannot evaluate a dataset with no images")

    ground_truth = defaultdict(list)
    grouped_predictions = defaultdict(list)

    for annotation in gt["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        if image_id not in image_ids:
            raise ValueError(f"Unknown ground-truth image ID: {image_id}")

        if category_id not in categories:
            raise ValueError(
                f"Unknown ground-truth category ID: {category_id}"
            )

        # Crowd/ignore regions need special matching rules.
        # Fail explicitly rather than silently counting them incorrectly.
        if annotation.get("iscrowd", 0) or annotation.get("ignore", 0):
            raise ValueError(
                "This FP/image implementation supports only "
                "non-crowd, non-ignored ground truth"
            )

        ground_truth[(image_id, category_id)].append(annotation["bbox"])

    for prediction in predictions:
        image_id = prediction["image_id"]
        category_id = prediction["category_id"]

        if image_id not in image_ids:
            raise ValueError(f"Unknown prediction image ID: {image_id}")

        if category_id not in categories:
            raise ValueError(
                f"Unknown prediction category ID: {category_id}"
            )

        if prediction["score"] >= score_threshold:
            grouped_predictions[(image_id, category_id)].append(
                prediction
            )

    false_positives = {
        category_id: 0
        for category_id in categories
    }

    for key, detections in grouped_predictions.items():
        _, category_id = key
        boxes = ground_truth[key]
        matched = set()

        detections = sorted(
            detections,
            key=lambda prediction: prediction["score"],
            reverse=True,
        )

        for detection in detections:
            best_index = None
            best_iou = -1.0

            for index, box in enumerate(boxes):
                if index in matched:
                    continue

                overlap = _iou_xywh(detection["bbox"], box)

                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index

            if best_index is not None and best_iou >= iou_threshold:
                matched.add(best_index)
            else:
                false_positives[category_id] += 1

    image_count = len(image_ids)
    total_fp = sum(false_positives.values())

    return {
        "false_positives_per_image": total_fp / image_count,
        "false_positive_count": total_fp,
        "per_class": {
            categories[category_id]: count / image_count
            for category_id, count in false_positives.items()
        },
        "image_count": image_count,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
    }