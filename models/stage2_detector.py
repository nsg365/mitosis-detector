"""
Stage 2 Object Detector - Mitosis Localization

This module provides the object detector for the second stage of the 
mitosis detection pipeline. It localizes and classifies mitotic figures
within patches identified by Stage 1.

Architecture: Faster R-CNN with ResNet50-FPN backbone
Input: 512×512 RGB patch
Output: Bounding boxes + confidence scores
Anchors: Custom sizes [8, 16, 32, 64, 128]px for small objects
Loss: Multi-task (classification + regression)
Optimization: AdamW + Cosine Annealing
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator


class Stage2Detector(nn.Module):
    """
    Faster R-CNN based object detector for mitosis localization.
    
    Args:
        num_classes (int): Number of classes (default: 2, background + mitosis)
        pretrained (bool): Load COCO pretrained weights
        pretrained_backbone (bool): Load ImageNet pretrained backbone
    """
    
    def __init__(self, num_classes=2, pretrained=False, pretrained_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        
        # Custom anchors for small objects (mitosis ~20-30 pixels)
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        
        # Load Faster R-CNN with ResNet50-FPN
        self.model = fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator
        )
    
    def forward(self, images, targets=None):
        """
        Forward pass.
        
        Args:
            images: List of tensors of shape [3, H, W]
            targets: List of dicts with 'boxes' and 'labels' (required during training)
        
        Returns:
            During training: dict with 'loss_classifier', 'loss_box_reg', etc.
            During inference: list of dicts with 'boxes', 'labels', 'scores'
        """
        return self.model(images, targets)
    
    def freeze_backbone(self):
        """Freeze all but the FPN and detection heads."""
        for name, param in self.model.backbone.named_parameters():
            if 'layer4' not in name:
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
