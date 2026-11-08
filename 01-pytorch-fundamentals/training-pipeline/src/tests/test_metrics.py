import pytest

from dlsl.metrics import false_positives_per_image


@pytest.fixture
def ground_truth():
    return {
        "images": [
            {"id": 1},
            {"id": 2},
        ],
        "categories": [
            {"id": 1, "name": "pallet"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
    }


def prediction(box, score=0.9, image_id=1):
    return {
        "image_id": image_id,
        "category_id": 1,
        "bbox": box,
        "score": score,
    }


def test_perfect_prediction_has_no_false_positives(ground_truth):
    result = false_positives_per_image(
        ground_truth,
        [prediction([0, 0, 10, 10])],
    )

    assert result["false_positive_count"] == 0
    assert result["false_positives_per_image"] == 0.0


def test_duplicate_prediction_is_false_positive(ground_truth):
    result = false_positives_per_image(
        ground_truth,
        [
            prediction([0, 0, 10, 10], score=0.9),
            prediction([0, 0, 10, 10], score=0.8),
        ],
    )

    assert result["false_positive_count"] == 1

    # One false positive across two evaluated images.
    assert result["false_positives_per_image"] == pytest.approx(0.5)
    assert result["per_class"]["pallet"] == pytest.approx(0.5)


def test_insufficient_overlap_is_false_positive(ground_truth):
    # Intersection=60, union=140, IoU≈0.429, below 0.5.
    result = false_positives_per_image(
        ground_truth,
        [prediction([4, 0, 10, 10])],
    )

    assert result["false_positive_count"] == 1


def test_prediction_in_empty_image_is_false_positive(ground_truth):
    result = false_positives_per_image(
        ground_truth,
        [prediction([0, 0, 10, 10], image_id=2)],
    )

    assert result["false_positive_count"] == 1


def test_low_confidence_prediction_is_excluded(ground_truth):
    result = false_positives_per_image(
        ground_truth,
        [prediction([50, 50, 10, 10], score=0.2)],
        score_threshold=0.5,
    )

    assert result["false_positive_count"] == 0


def test_empty_predictions(ground_truth):
    result = false_positives_per_image(ground_truth, [])

    assert result["false_positive_count"] == 0
    assert result["image_count"] == 2