import torch
import torch.nn as nn

class SelmanMLP(nn.Module):
    """
    3-Stage Multilayer Perceptron with Gaussian Error Linear Units (GELU)
    Dimensions: 784 -> 320 -> 80 -> 10
    """
    def __init__(self, in_features=784, h1=320, h2=80, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, h1),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
