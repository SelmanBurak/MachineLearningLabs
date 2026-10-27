import torch
import torch.nn as nn

class SelmanCNN(nn.Module):
    """
    Batch-Normalized Convolutional Vision Model
    Block 1: Conv2d(1, 32, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)
    Block 2: Conv2d(32, 64, k=3, p=1) -> BatchNorm2d -> ReLU -> AdaptiveAvgPool2d((3, 3))
    Head: Flatten -> Linear(64*3*3, 128) -> ReLU -> Linear(128, 10)
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3))
        )
        self.dense_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.dense_head(self.conv_stack(x))
