# model.py
import torch
import torch.nn as nn

def create_model(num_classes, dropout=0.3, use_timm=False):
    """
    Returns model. Tries torchvision EfficientNet-B0, falls back to timm.
    """
    try:
        from torchvision import models
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    except Exception:
        # fallback to timm
        import timm
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    return model
