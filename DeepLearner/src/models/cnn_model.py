import torch
import torch.nn as nn

from monai.networks.blocks import Convolution


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=2, dropout=0.05, n_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = Convolution(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=16,
            adn_ordering="ADN",
            dropout=dropout
        )
        self.conv2 = Convolution(
            spatial_dims=2,
            in_channels=16,
            out_channels=16,
            adn_ordering="ADN",
            dropout=dropout
        )
        self.conv3 = Convolution(
            spatial_dims=2,
            in_channels=16,
            out_channels=16,
            adn_ordering="ADN",
            dropout=dropout
        )
        self.mxpool = nn.MaxPool2d(4)
        self.out_head = nn.Sequential(
            nn.Linear(1024, 64),
            nn.PReLU(),
            nn.Linear(64, 64),
            nn.PReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        out = self.mxpool(self.conv1(x))
        out = self.mxpool(self.conv2(out))
        out = self.mxpool(self.conv3(out))
        out = nn.Flatten(start_dim=1)(out)
        out = self.out_head(out)
        return out
