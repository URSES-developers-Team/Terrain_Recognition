import sys
import os
from data import preprocessing
from . import yolo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from .faster_rcnn import base, enhanced, ultimate

def get_model(model_name, num_classes):
    """
    Model factory for programmatic model creation.
    
    Supported model names:
    
    FasterRCNN:
    - "base": Standard FasterRCNN
    - "enhanced": FasterRCNN with satellite optimizations
    
    YOLO:
    - "yolo", "yolov8": YOLOv8 nano
    - "yolo_enhanced", "yolov8_enhanced": Enhanced YOLOv8 nano  
    - "yolo_<size>": YOLOv8 with size (n/s/m/l/x)
    - "yolo_enhanced_<size>": Enhanced YOLOv8 with size
    """
    # FasterRCNN models
    if model_name == "ultimate":
        # Ultimate model should be created directly in train.py to avoid preprocessing duplication
        raise ValueError(
            "Ultimate model requires class_counts parameter. "
            "For training, use: python src/train.py --model ultimate. "
            "For inference, import UltimateFasterRCNN directly and provide class_counts."
        )
    elif model_name == "apex":
        # Apex model should be created directly in train.py to avoid preprocessing duplication
        raise ValueError(
            "Apex model requires class_counts parameter. "
            "For training, use: python src/train.py --model apex. "
            "For inference, import ApexFasterRCNN directly and provide class_counts."
        )
    
    if model_name == "enhanced":
        return enhanced.FasterRCNN_Enhanced(num_classes).get_model(
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            elu_alpha=ELU_ALPHA
        )
    elif model_name == "base":
        return base.FasterRCNN(num_classes).get_model()
    
    # YOLO models
    elif model_name == "yolo" or model_name == "yolov8":
        return yolo.YOLOv8(num_classes).get_model()
    elif model_name == "yolo_enhanced" or model_name == "yolov8_enhanced":
        return yolo.YOLOv8Enhanced(num_classes, model_size='n', satellite_optimized=True).get_model()
    elif model_name.startswith("yolo_") or model_name.startswith("yolov8_"):
        parts = model_name.split("_")
        
        if "enhanced" in parts:
            # Enhanced YOLO with size: yolo_enhanced_s, yolov8_enhanced_m, etc.
            size_candidates = [p for p in parts if p in ['n', 's', 'm', 'l', 'x']]
            size = size_candidates[0] if size_candidates else 'n'
            return yolo.YOLOv8Enhanced(num_classes, model_size=size, satellite_optimized=True).get_model()
        else:
            # Regular YOLO with size: yolo_s, yolov8_m, etc.
            size = parts[-1]
            if size in ['n', 's', 'm', 'l', 'x']:
                return yolo.YOLOv8(num_classes, model_size=size).get_model()
            else:
                return yolo.YOLOv8(num_classes).get_model()
    
    # Default fallback
    return base.FasterRCNN(num_classes).get_model()
