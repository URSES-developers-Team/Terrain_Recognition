"""
Utility functions for FasterRCNN models
Provides reusable components and helper functions for model construction and enhancement
"""

import torch
import torch.nn as nn
from typing import Dict, Any


def replace_relu_with_elu(module: nn.Module, alpha: float = 1.0, inplace: bool = True) -> None:
    """
    Recursively replace all ReLU activations with ELU for better gradient flow
    
    Args:
        module: PyTorch module to modify
        alpha: ELU alpha parameter
        inplace: Whether to use inplace operations
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ELU(alpha=alpha, inplace=inplace))
        else:
            replace_relu_with_elu(child, alpha=alpha, inplace=inplace)


def initialize_predictor_weights(predictor: nn.Module) -> None:
    """
    Initialize predictor weights for training stability
    
    Args:
        predictor: FastRCNNPredictor module to initialize
    """
    nn.init.normal_(predictor.cls_score.weight, std=0.01)
    nn.init.normal_(predictor.bbox_pred.weight, std=0.001)
    nn.init.constant_(predictor.cls_score.bias, 0)
    nn.init.constant_(predictor.bbox_pred.bias, 0)


def optimize_model_for_satellite_imagery(model: nn.Module) -> None:
    """
    Apply satellite imagery specific optimizations to a FasterRCNN model
    
    Args:
        model: FasterRCNN model to optimize
    """
    # RPN optimization for dense satellite scenes
    model.rpn.pre_nms_top_n_train = 6000
    model.rpn.post_nms_top_n_train = 4000
    model.rpn.pre_nms_top_n_test = 3000
    model.rpn.post_nms_top_n_test = 2000
    
    # Detection parameters optimized for small objects
    model.roi_heads.nms_thresh = 0.3
    model.roi_heads.detections_per_img = 300
    model.roi_heads.score_thresh = 0.01
    
    # Batch sampling optimization
    if hasattr(model.roi_heads, 'fg_bg_sampler'):
        model.roi_heads.fg_bg_sampler.batch_size_per_image = 512
        model.roi_heads.fg_bg_sampler.positive_fraction = 0.25


def compute_class_counts_from_dataframe(df, class_column: str = 'class_id') -> Dict[int, int]:
    """
    Compute class counts from annotation dataframe
    
    Args:
        df: DataFrame with annotations
        class_column: Column name containing class IDs
    
    Returns:
        dict: {class_id: count} mapping
    """
    class_counts = df[class_column].value_counts().to_dict()
    
    # Ensure background class (0) is included
    if 0 not in class_counts:
        class_counts[0] = 1
    
    print(f"📊 Class distribution computed:")
    print(f"   Total classes: {len(class_counts)}")
    print(f"   Most common: {max(class_counts.values())} samples")
    print(f"   Least common: {min(class_counts.values())} samples")
    print(f"   Imbalance ratio: {max(class_counts.values())/min(class_counts.values()):.1f}:1")
    
    return class_counts


def get_training_recommendations() -> Dict[str, Any]:
    """
    Get recommended training hyperparameters for satellite imagery
    
    Returns:
        dict: Training configuration parameters
    """
    return {
        'learning_rate': 0.0005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'batch_size': 2,
        'gradient_clip_norm': 1.0,
        'warmup_epochs': 3,
        'scheduler': 'StepLR',
        'step_size': 7,
        'gamma': 0.1,
        'early_stopping_patience': 5,
        'amp_enabled': True,  # Mixed precision training
    }


def handle_image_input_formats(images, device: str = 'cpu'):
    """
    Handle different input formats for images and convert to proper format
    
    Args:
        images: List of images, single image, or ImageList
        device: Target device
        
    Returns:
        Processed images list
    """
    if isinstance(images, (list, tuple)):
        # Ensure all images are tensors and on the correct device
        processed_images = []
        for img in images:
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)
            img = img.to(device)
            processed_images.append(img)
        return processed_images
    elif isinstance(images, torch.Tensor):
        # Single tensor - convert to list
        return [images.to(device)]
    else:
        raise ValueError(f"Unsupported image format: {type(images)}")


def move_targets_to_device(targets, device: str):
    """
    Move target dictionaries to the specified device
    
    Args:
        targets: List of target dictionaries
        device: Target device
        
    Returns:
        Targets moved to device
    """
    if targets is not None:
        return [{k: v.to(device) for k, v in t.items()} for t in targets]
    return None


def filter_predictions_by_confidence(predictions, confidence_threshold: float = 0.5):
    """
    Filter predictions by confidence threshold
    
    Args:
        predictions: List of prediction dictionaries
        confidence_threshold: Minimum confidence score
        
    Returns:
        Filtered predictions
    """
    filtered_predictions = []
    for pred in predictions:
        mask = pred['scores'] >= confidence_threshold
        filtered_pred = {
            'boxes': pred['boxes'][mask],
            'labels': pred['labels'][mask],
            'scores': pred['scores'][mask]
        }
        filtered_predictions.append(filtered_pred)
    return filtered_predictions
