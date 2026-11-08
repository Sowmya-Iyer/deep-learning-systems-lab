import torch

from models.detectors import build_fasterrcnn


def test_fasterrcnn_training_and_evaluation():
    # No pretrained downloads. Small images keep this smoke test lighter.
    # With no pretrained weights, use all five trainable backbone stages.
    model = build_fasterrcnn(
        num_classes=6,
        weights=None,
        min_size=64,
        max_size=64,
        trainable_backbone_layers=5,
    )

    images = [
        torch.rand(3, 64, 64),
        torch.rand(3, 64, 64),
    ]

    targets = [
        {
            "boxes": torch.tensor(
                [[10.0, 10.0, 40.0, 45.0]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor(
                [[15.0, 12.0, 50.0, 55.0]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([3], dtype=torch.int64),
        },
    ]

    assert model.roi_heads.box_predictor.cls_score.out_features == 6

    model.train()

    # This smoke test checks loss outputs, without storing a backward graph.
    with torch.no_grad():
        losses = model(images, targets)

    assert isinstance(losses, dict)
    assert losses

    for name, loss in losses.items():
        assert loss.ndim == 0, name
        assert torch.isfinite(loss).item(), name

    model.eval()

    with torch.no_grad():
        predictions = model(images)

    assert len(predictions) == len(images)

    for prediction in predictions:
        assert {"boxes", "labels", "scores"} <= prediction.keys()

        boxes = prediction["boxes"]
        labels = prediction["labels"]
        scores = prediction["scores"]

        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        assert labels.shape == scores.shape == (len(boxes),)

        assert torch.isfinite(boxes).all().item()
        assert ((scores >= 0) & (scores <= 1)).all().item()
        assert ((labels >= 1) & (labels <= 5)).all().item()