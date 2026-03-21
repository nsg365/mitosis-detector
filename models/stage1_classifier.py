"""
Stage 1 Binary Classifier - Mitosis Screening

This module provides the binary classifier for the first stage of the 
mitosis detection pipeline. It filters patches to identify those likely
to contain mitotic figures.

Architecture: ResNet50 or EfficientNet-B3 with 2-class output
Input: 64×64 RGB patch
Output: Binary probability (contains mitosis vs. background)
Loss: Focal Loss for class imbalance
Optimization: AdamW + Cosine Annealing
"""

import torch
import torch.nn as nn
import torchvision.models as models


class Stage1Classifier(nn.Module):
    """
    Binary classifier for mitosis screening.
    
    Args:
        backbone (str): 'resnet50' or 'efficientnet_b3'
        pretrained (bool): Load ImageNet weights
        num_classes (int): Output classes (default: 2)
    """
    
    def __init__(self, backbone='efficientnet_b3', pretrained=True, num_classes=2):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        if backbone == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            self.model = model
            
        elif backbone == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=pretrained)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            self.model = model
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def forward(self, x):
        return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all but the final layer."""
        for name, param in self.model.named_parameters():
            if not name.startswith('fc') and not name.startswith('classifier'):
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
