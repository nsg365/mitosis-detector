#Configuration settings for Stage 1 and Stage 2 training.
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Stage1Config:
    #Configuration for Stage 1 Binary Classifier.
    
    # Data
    patch_size: int = 64
    num_classes: int = 2
    
    # Model
    backbone: str = "efficientnet_b3"  # or "resnet50"
    pretrained: bool = True
    
    # Training
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Optimization
    warmup_epochs: int = 3
    
    # Augmentation
    random_crop: bool = True
    random_rotation: float = 15.0
    random_affine: bool = True
    color_jitter: Tuple = (0.2, 0.2, 0.2, 0.1)
    
    # Checkpoint
    save_interval: int = 1
    checkpoint_dir: str = "checkpoints"


@dataclass
class Stage2Config:
    #Configuration for Stage 2 Object Detector.
    
    # Data
    patch_size: int = 512
    num_classes: int = 2  # background + mitosis
    
    # Model
    backbone: str = "resnet50_fpn"
    pretrained: bool = False
    pretrained_backbone: bool = True
    
    # Anchors
    anchor_sizes: Tuple = (8, 16, 32, 64, 128)
    aspect_ratios: Tuple = (0.5, 1.0, 2.0)
    
    # Training
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    
    # Optimization
    warmup_epochs: int = 5
    
    # Augmentation
    random_flip: bool = True
    random_rotation: float = 10.0
    color_jitter: Tuple = (0.2, 0.2, 0.2, 0.1)
    
    # NMS
    nms_thresh: float = 0.4
    score_thresh: float = 0.4
    
    # Checkpoint
    save_interval: int = 1
    checkpoint_dir: str = "checkpoints"


@dataclass
class PipelineConfig:
    #Configuration for end-to-end pipeline.
    
    # Stage 1 threshold
    stage1_threshold: float = 0.5087
    
    # Stage 2 score threshold
    stage2_score_threshold: float = 0.4
    
    # IoU threshold for evaluation
    iou_threshold: float = 0.5
    
    # Use both stages or just Stage 2
    use_stage1_filtering: bool = False
    
    # Number of workers for data loading
    num_workers: int = 4
