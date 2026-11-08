"""Factories for object-detection models."""

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn(
    num_classes=6,
    weights="COCO_V1",
    min_size=800,
    max_size=1333,
    trainable_backbone_layers=3,
):
    """Build Faster R-CNN with a replacement classification/box head."""
    if num_classes < 2:
        raise ValueError(
            "num_classes must include background and at least one object class"
        )

    if weights is None:
        pretrained_weights = None
    elif weights == "DEFAULT":
        pretrained_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    else:
        pretrained_weights = FasterRCNN_ResNet50_FPN_V2_Weights[weights]

    model = fasterrcnn_resnet50_fpn_v2(
        weights=pretrained_weights,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # Read the feature width expected by the existing predictor.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the COCO predictor with our task-specific predictor.
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    return model