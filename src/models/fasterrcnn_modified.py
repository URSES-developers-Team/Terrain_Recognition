from torchvision import models
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config import *


class FasterRCNN_Advanced:
    """
    Advanced Faster R-CNN with COCO pretraining + satellite imagery optimizations:
    - COCO pretraining for feature extraction
    - Custom anchor scales for small objects
    - Lower NMS threshold for dense objects
    - Optimized for satellite imagery characteristics
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def get_model(self):
        # Start with COCO pretrained weights - this is crucial for performance
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        
        # Replace classifier head for our number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Satellite imagery optimizations
        self._optimize_for_satellite_imagery(model)
        
        return model
    
    def _optimize_for_satellite_imagery(self, model):
        """
        Optimize model parameters for satellite imagery characteristics:
        - Small objects
        - Dense object distribution
        - High resolution imagery
        """
        # Lower NMS threshold for dense satellite imagery
        model.roi_heads.nms_thresh = 0.3  # Default: 0.5
        
        # Increase detections per image for dense satellite scenes
        model.roi_heads.detections_per_img = 300  # Default: 100
        
        # Lower score threshold for small objects
        model.roi_heads.score_thresh = 0.01  # Default: 0.05
        
        # Optimize anchor generator for small objects
        if hasattr(model.rpn, 'anchor_generator'):
            # Use smaller anchor scales for satellite imagery small objects
            model.rpn.anchor_generator.sizes = ((16, 32, 64, 128, 256),) * 5
            model.rpn.anchor_generator.aspect_ratios = ((0.5, 1.0, 2.0),) * 5
        
        # More proposals for dense imagery
        model.rpn.pre_nms_top_n_train = 4000  # Default: 2000
        model.rpn.post_nms_top_n_train = 4000  # Default: 2000
        model.rpn.pre_nms_top_n_test = 2000   # Default: 1000
        model.rpn.post_nms_top_n_test = 2000  # Default: 1000


class FasterRCNN_MultiScale:
    """
    Multi-scale Faster R-CNN optimized for satellite imagery with different resolutions
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def get_model(self):
        # Always start with COCO pretraining - proven to be beneficial
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = models.detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            # Custom transform for satellite imagery
            min_size=512,      # Minimum image size
            max_size=1024,     # Maximum image size  
            image_mean=[0.485, 0.456, 0.406],  # ImageNet means work well
            image_std=[0.229, 0.224, 0.225]    # ImageNet stds work well
        )
        
        # Replace classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        return model
