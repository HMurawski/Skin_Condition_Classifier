
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_model(num_classes: int):
    """
    Build a ResNet18 model with ImageNet weights and replace the final layer.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
