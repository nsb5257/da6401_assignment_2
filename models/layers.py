"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("Dropout prob p must be in range [0, 1)")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        # TODO: implement dropout.
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) > self.p).to(x.dtype)
        return x*mask/(1-self.p)        
