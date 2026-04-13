"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3, use_batch_norm: bool = True):
        """Initialize the VGG11Encoder model."""
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        # VGG-11 Convolutional Blocks (as per Simonyan & Zisserman, 2014)
        self.block1 = self._make_block(in_channels, 64, num_convs=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block2 = self._make_block(64, 128, num_convs=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block3 = self._make_block(128, 256, num_convs=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block4 = self._make_block(256, 512, num_convs=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block5 = self._make_block(512, 512, num_convs=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def _make_block(self, in_c: int, out_c: int, num_convs: int) -> nn.Sequential:
        """Helper to construct sequential conv blocks with optional batch norm."""
        layers = []
        for i in range(num_convs):
            current_in = in_c if i == 0 else out_c
            layers.append(nn.Conv2d(current_in, out_c, kernel_size=3, padding=1))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # TODO: Implement forward pass.
        features = {}
        
        b1 = self.block1(x)
        features['block1'] = b1
        p1 = self.pool1(b1)
        
        b2 = self.block2(p1)
        features['block2'] = b2
        p2 = self.pool2(b2)
        
        b3 = self.block3(p2)
        features['block3'] = b3
        p3 = self.pool3(b3)
        
        b4 = self.block4(p3)
        features['block4'] = b4
        p4 = self.pool4(b4)
        
        b5 = self.block5(p4)
        features['block5'] = b5
        bottleneck = self.pool5(b5)
        
        if return_features:
            return bottleneck, features
            
        return bottleneck

# Alias to ensure autograder compatibility as per README
VGG11 = VGG11Encoder