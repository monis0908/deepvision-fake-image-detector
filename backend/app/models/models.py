import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FakeImageDetector(nn.Module):

    def __init__(self):

        super().__init__()

        # Load pretrained ResNet18
        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Number of input features of the last layer
        in_features = self.model.fc.in_features

        # Replace the last layer for binary classification
        self.model.fc = nn.Linear(
            in_features,
            2
        )

    def forward(self, x):

        return self.model(x)