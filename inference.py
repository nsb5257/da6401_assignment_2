"""Inference and evaluation
"""

import torch
import numpy as np
from sklearn.metrics import f1_score

class MetricsCalculator:
    """Utility to compute automated grading metrics."""

    @staticmethod
    def calculate_macro_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Satisfies Task 4 Classification requirement: Macro F1-Score.
        Utilizes scikit-learn as specified in requirements.txt.
        
        Args:
            preds: Logits [B, 37]
            targets: Ground truth class indices [B]
        """
        # Convert logits to predicted class indices
        pred_classes = torch.argmax(preds, dim=1).cpu().numpy()
        target_classes = targets.cpu().numpy()
        
        # Calculate macro F1 using scikit-learn
        # zero_division=0 prevents warnings early in training when a class isn't predicted
        return f1_score(target_classes, pred_classes, average='macro', zero_division=0)

    @staticmethod
    def calculate_dice_score(pred_masks: torch.Tensor, target_masks: torch.Tensor, num_classes: int = 3, eps: float = 1e-6) -> float:
        """
        Satisfies Task 4 Segmentation requirement: Dice Similarity Coefficient.
        Calculates per-class Dice and averages them (macro average).
        
        Args:
            pred_masks: Segmentation logits [B, C, H, W]
            target_masks: Ground truth trimap indices [B, H, W]
        """
        preds = torch.argmax(pred_masks, dim=1) # Shape: [B, H, W]
        dice_total = 0.0
        
        for c in range(num_classes):
            # Create binary masks for the current class
            p = (preds == c).float()
            t = (target_masks == c).float()
            
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            
            # Dice formula: 2 * |A intersect B| / (|A| + |B|)
            dice_score = (2.0 * intersection + eps) / (union + eps)
            dice_total += dice_score
            
        return (dice_total / num_classes).item()

    @staticmethod
    def pixel_accuracy(pred_masks: torch.Tensor, target_masks: torch.Tensor) -> float:
        """
        Helper metric required for W&B Report Q2.6 (Dice vs. Pixel Accuracy).
        """
        preds = torch.argmax(pred_masks, dim=1)
        correct = (preds == target_masks).float().sum()
        total = torch.numel(target_masks)
        return (correct / total).item()