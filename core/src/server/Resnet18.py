import torch
import torch.nn as nn
from torchvision import models

class ResNet18Classifier(nn.Module):
    def __init__(self,
                 num_classes:int = 12,
                 pre_trained: bool = True,
                 dropout: float = 0.2):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pre_trained else None
        self.backbone = models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        if dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor):
        return self.backbone(x)
    
    
