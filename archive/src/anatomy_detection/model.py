from typing import Optional

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

try:
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
except Exception:  # pragma: no cover - for older torchvision releases
    MaskRCNN_ResNet50_FPN_Weights = None


def build_maskrcnn(
    num_classes: int,
    pretrained: bool = True,
    min_size: int = 640,
    max_size: int = 1024,
    trainable_backbone_layers: Optional[int] = 3,
):
    """Build Mask R-CNN with COCO heads replaced for anatomy classes."""
    weights = None
    if pretrained and MaskRCNN_ResNet50_FPN_Weights is not None:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        min_size=int(min_size),
        max_size=int(max_size),
        trainable_backbone_layers=trainable_backbone_layers if weights is not None else None,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(num_classes))

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, int(num_classes))
    return model
