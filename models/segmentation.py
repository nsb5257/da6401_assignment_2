"""Segmentation model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11UNet(nn.Module):
    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5, use_bn: bool = True):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batch_norm=use_bn)

        self.up1_trans = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up1_conv = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.up2_trans = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2_conv = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.up3_trans = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3_conv = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.up4_trans = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4_conv = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.up5_trans = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up5_conv = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _pad_and_cat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return torch.cat([skip, x], dim=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        bottleneck, features = self.encoder(x, return_features=True)
        d = self.up1_trans(bottleneck)
        d = self._pad_and_cat(d, features['block5'])
        d = self.up1_conv(d)
        d = self.up2_trans(d)
        d = self._pad_and_cat(d, features['block4'])
        d = self.up2_conv(d)
        d = self.up3_trans(d)
        d = self._pad_and_cat(d, features['block3'])
        d = self.up3_conv(d)
        d = self.up4_trans(d)
        d = self._pad_and_cat(d, features['block2'])
        d = self.up4_conv(d)
        d = self.up5_trans(d)
        d = self._pad_and_cat(d, features['block1'])
        d = self.up5_conv(d)
        return self.final_conv(d)