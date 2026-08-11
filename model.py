import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from config import PRETRAINED


def build_model(num_classes):
    """
    Creates a MobileNetV3 Small model for fine-tuning.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if PRETRAINED else None
    model = mobilenet_v3_small(weights=weights)

    # Unfreeze the final feature extraction blocks (11, 12, 13) and the classifier
    for name, param in model.named_parameters():
        if any(b in name for b in ["features.11", "features.12", "features.13", "classifier"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Get input features of final layer
    in_features = model.classifier[3].in_features

    # Replace classifier with Dropout + Linear layer to reduce overfitting
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=in_features, out_features=num_classes)
    )

    return model