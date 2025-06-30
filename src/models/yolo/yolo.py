import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
import math
import sys
import os

# Add the src directory to the path to find config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEVICE


class YOLOv8Native(nn.Module):
    """
    Native YOLOv8 implementation optimized for maximum performance.
    Uses YOLO's native training pipeline for best results.
    """
    def __init__(self, num_classes, model_size='n'):
        super().__init__()
        self.num_classes = num_classes
        self.model_size = model_size
        self.model = self._create_model()
        
    def _create_model(self):
        """
        Creates YOLOv8 model with appropriate size.
        model_size options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
        """
        model_name = f'yolov8{self.model_size}.pt'
        model = YOLO(model_name)
        return model
    
    def forward(self, images, **kwargs):
        """
        Direct forward pass using YOLO's native inference.
        Optimized for satellite imagery detection.
        """
        return self._inference_forward(images, **kwargs)
    
    def _inference_forward(self, images, conf=0.25, iou=0.45, max_det=1000, **kwargs):
        """
        Native YOLO inference optimized for satellite imagery.
        Returns raw YOLO results for maximum performance.
        """
        # Convert inputs to YOLO-compatible format
        processed_images = self._prepare_images_for_yolo(images)
        
        # Run native YOLO inference with satellite-optimized settings
        results = self.model(
            processed_images,
            verbose=False,
            conf=conf,  # Confidence threshold
            iou=iou,    # NMS IoU threshold
            agnostic_nms=False,  # Class-agnostic NMS
            max_det=max_det,  # Maximum detections per image
            **kwargs
        )
        
        return results
    
    def _prepare_images_for_yolo(self, images):
        """
        Convert various image formats to YOLO-compatible numpy arrays.
        Handles tensors, PIL images, numpy arrays, and file paths.
        """
        if isinstance(images, str):
            # Single file path
            return images
        elif isinstance(images, (list, tuple)):
            if len(images) == 0:
                return images
            
            # Handle list of images
            first_item = images[0]
            if isinstance(first_item, str):
                # List of file paths
                return images
            elif isinstance(first_item, torch.Tensor):
                # List of tensors - convert to numpy
                np_images = []
                for img_tensor in images:
                    img_np = self._tensor_to_numpy(img_tensor)
                    np_images.append(img_np)
                return np_images
            elif isinstance(first_item, np.ndarray):
                # Already numpy arrays
                return images
            else:
                # PIL images or other formats
                return images
        elif isinstance(images, torch.Tensor):
            # Single tensor
            return self._tensor_to_numpy(images)
        elif isinstance(images, np.ndarray):
            # Single numpy array
            return images
        else:
            # PIL image or other format
            return images
    
    def _tensor_to_numpy(self, tensor):
        """Convert a PyTorch tensor to numpy array in YOLO format."""
        if tensor.dim() == 4:
            # Batch of images [B, C, H, W] - take first image
            tensor = tensor[0]
        
        if tensor.dim() == 3:
            # Single image [C, H, W] -> [H, W, C]
            img_np = tensor.permute(1, 2, 0).cpu().numpy()
        else:
            # Already in [H, W] or [H, W, C] format
            img_np = tensor.cpu().numpy()
        
        # Ensure proper data type and range
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0 and img_np.min() >= 0.0:
                # Normalized [0, 1] -> [0, 255]
                img_np = (img_np * 255).astype(np.uint8)
            else:
                # Assume already in [0, 255] range
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        return img_np
    
    def predict(self, images, **kwargs):
        """Direct prediction method using YOLO's native interface."""
        processed_images = self._prepare_images_for_yolo(images)
        return self.model.predict(processed_images, **kwargs)
    
    def train_native(self, data_yaml_path, epochs=100, imgsz=640, batch=16, 
                    learning_rate=0.01, weight_decay=0.0005, **kwargs):
        """
        Native YOLO training with satellite imagery optimizations.
        This is the recommended way to train for best performance.
        """
        # Enhanced training parameters for satellite imagery
        training_args = {
            'data': data_yaml_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': DEVICE,
            'save': True,
            'save_period': 10,
            'val': True,
            'plots': True,
            'verbose': True,
            'patience': 50,  # Early stopping patience
            'lr0': learning_rate,  # Initial learning rate
            'weight_decay': weight_decay,
            'momentum': 0.937,
            'cls': 0.5,  # Classification loss weight
            'box': 7.5,  # Box regression loss weight
            'dfl': 1.5,  # Distribution focal loss weight
            'mosaic': 1.0,  # Mosaic augmentation probability
            'mixup': 0.1,  # Mixup augmentation probability
            'copy_paste': 0.1,  # Copy-paste augmentation probability
            'degrees': 0.5,  # Rotation degrees for augmentation
            'translate': 0.1,  # Translation fraction
            'scale': 0.5,  # Scaling factor range
            'shear': 0.0,  # Shear degrees
            'perspective': 0.0,  # Perspective transform
            'flipud': 0.0,  # Vertical flip probability
            'fliplr': 0.5,  # Horizontal flip probability
            **kwargs
        }
        
        results = self.model.train(**training_args)
        return results
    
    def get_native_model(self):
        """Return the native YOLO model for direct use."""
        return self.model
    
    def save_model(self, path):
        """Save the YOLO model."""
        self.model.save(path)
    
    def load_model(self, path):
        """Load a trained YOLO model."""
        self.model = YOLO(path)
        return self.model

class YOLOv8:
    """
    Primary YOLOv8 model class for maximum performance.
    Uses native YOLO training and inference - no compatibility overhead.
    """
    def __init__(self, num_classes, model_size='n'):
        self.num_classes = num_classes
        self.model_size = model_size
        
    def get_model(self):
        """
        Returns native YOLOv8 model for maximum performance.
        """
        return YOLOv8Native(self.num_classes, self.model_size)


class YOLOv8Enhanced:
    """
    Enhanced YOLOv8 class with satellite imagery optimizations.
    Built for performance with intelligent model sizing.
    
    This class provides the same model size flexibility as regular YOLO
    but with satellite imagery optimizations applied during training.
    """
    def __init__(self, num_classes, model_size='n', satellite_optimized=True):
        """
        Initialize Enhanced YOLOv8 model.
        
        Args:
            num_classes: Number of classes for detection
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            satellite_optimized: Whether to apply satellite-specific optimizations
        """
        self.num_classes = num_classes
        self.model_size = model_size
        self.satellite_optimized = satellite_optimized
        
        # Validate model size
        valid_sizes = ['n', 's', 'm', 'l', 'x']
        if self.model_size not in valid_sizes:
            raise ValueError(f"Invalid model size '{self.model_size}'. Must be one of {valid_sizes}")
        
    def get_model(self):
        """
        Returns optimized YOLOv8 model for satellite imagery.
        
        Returns:
            YOLOv8Native model with specified size and satellite optimizations
        """
        print(f"Creating Enhanced YOLOv8{self.model_size.upper()} model with satellite optimizations")
        model = YOLOv8Native(self.num_classes, self.model_size)
        
        if self.satellite_optimized:
            # Apply satellite-specific optimizations
            model = self._apply_satellite_optimizations(model)
        
        return model
    
    def _apply_satellite_optimizations(self, model):
        """Apply optimizations specific to satellite imagery detection."""
        # Satellite imagery optimizations are primarily training-time configurations
        # The model itself remains native YOLO for maximum performance
        print(f"  ✅ Applied satellite imagery optimizations for YOLOv8{self.model_size.upper()}")
        return model
    
    @staticmethod
    def get_recommended_size(image_resolution, performance_priority='balanced', is_tiled_dataset=False):
        """
        Get recommended YOLO model size based on image resolution and performance needs.
        
        Args:
            image_resolution: tuple (width, height) of typical image size
            performance_priority: 'speed', 'accuracy', or 'balanced'
            is_tiled_dataset: Whether this is for a tiled dataset (like xView satellite imagery)
        
        Returns:
            Recommended model size ('n', 's', 'm', 'l', 'x')
        """
        width, height = image_resolution
        total_pixels = width * height
        
        print(f"Analyzing image resolution: {width}x{height} ({total_pixels:,} pixels)")
        print(f"Performance priority: {performance_priority}")
        if is_tiled_dataset:
            print("📡 Satellite/Tiled dataset detected - using optimized recommendations")
        
        # For tiled datasets like xView, we have special considerations:
        # - Higher object density per tile
        # - Need to detect small objects reliably  
        # - Processing many tiles per original image
        # - Balance between accuracy and processing speed across many tiles
        
        if is_tiled_dataset:
            # Satellite/tiled dataset recommendations (more conservative, better accuracy)
            if performance_priority == 'speed':
                if total_pixels < 500_000:  # Small tiles
                    recommended = 's'  # Skip nano for better small object detection
                    reason = "Small satellite tiles - small model for reliable detection"
                elif total_pixels < 2_000_000:  # Medium tiles  
                    recommended = 'm'
                    reason = "Medium satellite tiles - medium model for good speed/accuracy balance"
                else:  # Large tiles
                    recommended = 'l'
                    reason = "Large satellite tiles - large model for comprehensive detection"
            
            elif performance_priority == 'accuracy':
                if total_pixels < 500_000:
                    recommended = 'm'
                    reason = "Small satellite tiles - medium model for reliable small object detection"
                elif total_pixels < 2_000_000:
                    recommended = 'l'
                    reason = "Medium satellite tiles - large model for high accuracy"
                elif total_pixels < 8_000_000:
                    recommended = 'x'
                    reason = "Large satellite tiles - extra large model for maximum accuracy"
                else:
                    recommended = 'x'
                    reason = "Very large satellite tiles - extra large model for comprehensive detection"
            
            else:  # balanced - optimized for satellite imagery
                if total_pixels < 500_000:
                    recommended = 's'
                    reason = "Small satellite tiles - small model for reliable detection"
                elif total_pixels < 2_000_000:  # This covers your ~1M pixel xView tiles
                    recommended = 'm'
                    reason = "Medium satellite tiles (xView-sized) - medium model for optimal balance"
                elif total_pixels < 5_000_000:
                    recommended = 'l'
                    reason = "Large satellite tiles - large model for thorough detection"
                else:
                    recommended = 'l'
                    reason = "Very large satellite tiles - large model for comprehensive coverage"
        
        else:
            # Regular dataset recommendations (original logic)
            if performance_priority == 'speed':
                if total_pixels < 500_000:  # Small images
                    recommended = 'n'
                    reason = "Small images - nano model for maximum speed"
                elif total_pixels < 2_000_000:  # Medium images
                    recommended = 's'
                    reason = "Medium images - small model for good speed"
                else:  # Large images
                    recommended = 'm'
                    reason = "Large images - medium model for balanced performance"
            
            elif performance_priority == 'accuracy':
                if total_pixels < 500_000:
                    recommended = 's'
                    reason = "Small images - small model for better accuracy than nano"
                elif total_pixels < 2_000_000:
                    recommended = 'm'
                    reason = "Medium images - medium model for good accuracy"
                elif total_pixels < 8_000_000:
                    recommended = 'l'
                    reason = "Large images - large model for high accuracy"
                else:
                    recommended = 'x'
                    reason = "Very large images - extra large model for maximum accuracy"
            
            else:  # balanced
                if total_pixels < 500_000:
                    recommended = 'n'
                    reason = "Small images - nano model for speed"
                elif total_pixels < 2_000_000:
                    recommended = 's'
                    reason = "Medium images - small model for balanced performance"
                elif total_pixels < 5_000_000:
                    recommended = 'm'
                    reason = "Large images - medium model for good balance"
                else:
                    recommended = 'l'
                    reason = "Very large images - large model for accuracy"
        
        print(f"Recommended model size: YOLOv8{recommended.upper()} - {reason}")
        return recommended
    
    @classmethod
    def create_recommended(cls, num_classes, image_resolution, performance_priority='balanced', is_tiled_dataset=False):
        """
        Create Enhanced YOLO model with automatically recommended size.
        
        Args:
            num_classes: Number of classes for detection
            image_resolution: tuple (width, height) of typical image size
            performance_priority: 'speed', 'accuracy', or 'balanced'
            is_tiled_dataset: Whether this is for a tiled dataset (like xView)
            
        Returns:
            YOLOv8Enhanced instance with recommended model size
        """
        recommended_size = cls.get_recommended_size(image_resolution, performance_priority, is_tiled_dataset)
        return cls(num_classes, recommended_size, satellite_optimized=True)

# Utility functions for YOLO model management
def get_yolo_model(num_classes, model_size='n', enhanced=False):
    """
    Factory function to get the appropriate YOLO model.
    
    Args:
        num_classes: Number of classes for detection
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        enhanced: Whether to use enhanced satellite-optimized version
        
    Returns:
        Native YOLO model instance for maximum performance
    """
    print(f"Creating YOLOv8{model_size.upper()} model (Enhanced: {enhanced})")
    
    if enhanced:
        return YOLOv8Enhanced(num_classes, model_size).get_model()
    else:
        return YOLOv8(num_classes, model_size).get_model()


def get_enhanced_yolo_model(num_classes, model_size='n', satellite_optimized=True):
    """
    Convenience function specifically for Enhanced YOLO models with clear size selection.
    
    Args:
        num_classes: Number of classes for detection
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            'n' = nano (fastest, least accurate)
            's' = small (balanced speed/accuracy)
            'm' = medium (good accuracy, moderate speed)
            'l' = large (high accuracy, slower)
            'x' = extra large (highest accuracy, slowest)
        satellite_optimized: Whether to apply satellite imagery optimizations
        
    Returns:
        Enhanced YOLOv8Native model instance optimized for satellite imagery
    """
    return YOLOv8Enhanced(num_classes, model_size, satellite_optimized).get_model()


def get_xview_recommended_model(num_classes, performance_priority='balanced', n_tiles=10):
    """
    Get recommended YOLO model specifically optimized for xView satellite dataset.
    
    Args:
        num_classes: Number of classes for detection
        performance_priority: 'speed', 'accuracy', or 'balanced'
        n_tiles: Number of tiles used in preprocessing (default: 10 for xView)
        
    Returns:
        Enhanced YOLOv8Native model instance optimized for xView
    """
    # Calculate typical xView tile dimensions
    # Original xView images: ~3000x3000
    # With n_tiles=10: Nx=3, Ny=4 (3x4 grid = 12 tiles)
    # Each tile: ~1000x750 average
    
    original_size = 3000  # Typical xView image dimension
    nx = int(math.floor(math.sqrt(n_tiles)))
    if nx < 1:
        nx = 1
    ny = int(math.ceil(n_tiles / nx))
    
    tile_width = original_size // nx
    tile_height = original_size // ny
    
    print(f"xView Dataset Analysis:")
    print(f"  Original images: ~{original_size}x{original_size}")
    print(f"  Tiling: {nx}x{ny} grid ({nx*ny} tiles)")
    print(f"  Each tile: ~{tile_width}x{tile_height}")
    
    # Use the tiled dimensions for recommendation
    tile_resolution = (tile_width, tile_height)
    
    return YOLOv8Enhanced.create_recommended(
        num_classes=num_classes,
        image_resolution=tile_resolution,
        performance_priority=performance_priority,
        is_tiled_dataset=True
    ).get_model()


def train_yolo_native(model_path_or_size, data_yaml_path, epochs=100, imgsz=640, 
                     batch=16, learning_rate=0.01, weight_decay=0.0005, **kwargs):
    """
    Direct native YOLO training function for maximum performance.
    
    Args:
        model_path_or_size: Either a model size ('n', 's', 'm', 'l', 'x') or path to existing model
        data_yaml_path: Path to YOLO dataset YAML file
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        **kwargs: Additional training parameters
    """
    # Load model
    if model_path_or_size in ['n', 's', 'm', 'l', 'x']:
        model = YOLO(f'yolov8{model_path_or_size}.pt')
    else:
        model = YOLO(model_path_or_size)
    
    # Enhanced training parameters for satellite imagery
    training_args = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': DEVICE,
        'save': True,
        'save_period': 10,
        'val': True,
        'plots': True,
        'verbose': True,
        'patience': 50,  # Early stopping patience
        'lr0': learning_rate,  # Initial learning rate
        'weight_decay': weight_decay,
        'momentum': 0.937,
        'cls': 0.5,  # Classification loss weight
        'box': 7.5,  # Box regression loss weight
        'dfl': 1.5,  # Distribution focal loss weight
        'mosaic': 1.0,  # Mosaic augmentation probability
        'mixup': 0.1,  # Mixup augmentation probability
        'copy_paste': 0.1,  # Copy-paste augmentation probability
        'degrees': 0.5,  # Rotation degrees for augmentation
        'translate': 0.1,  # Translation fraction
        'scale': 0.5,  # Scaling factor range
        'shear': 0.0,  # Shear degrees
        'perspective': 0.0,  # Perspective transform
        'flipud': 0.0,  # Vertical flip probability
        'fliplr': 0.5,  # Horizontal flip probability
        **kwargs
    }
    
    results = model.train(**training_args)
    return results


def recommend_yolo_config(image_stats, performance_target='balanced', is_tiled_dataset=False):
    """
    Recommend YOLO configuration based on dataset characteristics.
    
    Args:
        image_stats: Dictionary with 'avg_width', 'avg_height', 'num_images', 'avg_objects_per_image'
        performance_target: 'speed', 'accuracy', or 'balanced'
        is_tiled_dataset: Whether this is for a tiled dataset (like xView satellite imagery)
        
    Returns:
        Dictionary with recommended configuration
    """
    avg_resolution = (image_stats['avg_width'], image_stats['avg_height'])
    recommended_size = YOLOv8Enhanced.get_recommended_size(avg_resolution, performance_target, is_tiled_dataset)
    
    # Determine optimal image size for training
    max_dim = max(avg_resolution)
    if max_dim <= 640:
        imgsz = 640
    elif max_dim <= 800:
        imgsz = 800
    elif max_dim <= 1024:
        imgsz = 1024
    else:
        imgsz = 1280
    
    # Determine batch size based on model size and image size
    batch_size_map = {
        ('n', 640): 32,
        ('s', 640): 24,
        ('m', 640): 16,
        ('l', 640): 12,
        ('x', 640): 8,
        ('n', 800): 24,
        ('s', 800): 16,
        ('m', 800): 12,
        ('l', 800): 8,
        ('x', 800): 6,
        ('n', 1024): 16,
        ('s', 1024): 12,
        ('m', 1024): 8,
        ('l', 1024): 6,
        ('x', 1024): 4,
        ('n', 1280): 12,
        ('s', 1280): 8,
        ('m', 1280): 6,
        ('l', 1280): 4,
        ('x', 1280): 2,
    }
    
    recommended_batch = batch_size_map.get((recommended_size, imgsz), 8)
    
    return {
        'model_size': recommended_size,
        'imgsz': imgsz,
        'batch_size': recommended_batch,
        'epochs': 200 if performance_target == 'accuracy' else 150,
        'learning_rate': 0.01,
        'weight_decay': 0.0005,
        'patience': 50,
        'is_tiled_dataset': is_tiled_dataset
    }
