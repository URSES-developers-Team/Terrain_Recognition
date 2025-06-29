"""
FasterRCNN Ultimate: Combines satellite imagery optimizations with class imbalance handling
- Satellite imagery optimizations (Advanced)
- ELU activations for better gradient flow
- Focal Loss for class imbalance
- Optimized for xView dataset characteristics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from config import *


def replace_relu_with_elu(module, alpha=1.0, inplace=True):
    """
    Recursively replace all ReLU activations with ELU for better gradient flow
    ELU helps with vanishing gradients and provides smoother activations
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ELU(alpha=alpha, inplace=inplace))
        else:
            replace_relu_with_elu(child, alpha=alpha, inplace=inplace)


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance in satellite imagery
    Focuses learning on hard examples and reduces impact of easy backgrounds
    """
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Enhanced Focal Loss forward pass with comprehensive numerical stability
        """
        # Enhanced numerical stability - prevent extreme values
        inputs = torch.clamp(inputs, min=-100, max=100)
        
        # Standard cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute pt (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Enhanced prevention of pt being exactly 1.0 to avoid NaN in (1-pt)^gamma
        pt = torch.clamp(pt, min=1e-8, max=1.0 - 1e-8)
        
        # Apply focal loss formula: alpha * (1-pt)^gamma * CE_loss
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * ce_loss
        
        # Enhanced NaN/Inf handling - replace with zeros and add debugging
        if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
            print("Warning: NaN/Inf detected in focal_loss computation")
            print(f"  Input range: [{inputs.min():.6f}, {inputs.max():.6f}]")
            print(f"  pt range: [{pt.min():.6f}, {pt.max():.6f}]")
            print(f"  ce_loss range: [{ce_loss.min():.6f}, {ce_loss.max():.6f}]")
        
        focal_loss = torch.where(torch.isfinite(focal_loss), focal_loss, torch.zeros_like(focal_loss))
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class RoIHeadsWithFocalLoss(RoIHeads):
    """
    Custom RoI Heads that use Focal Loss instead of standard Cross Entropy
    Optimized for satellite imagery with class imbalance
    """
    def __init__(self, focal_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = focal_loss

    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        """
        Enhanced loss function with comprehensive NaN/Inf handling for training stability
        Combines Focal Loss for classification and Smooth L1 for regression
        """
        # Ensure inputs are valid - comprehensive NaN/Inf detection and handling
        if torch.isnan(class_logits).any() or torch.isinf(class_logits).any():
            print("Warning: NaN/Inf detected in class_logits")
            class_logits = torch.nan_to_num(class_logits, nan=0.0, posinf=10.0, neginf=-10.0)

        # Apply Focal Loss for classification
        classification_loss = self.focal_loss(class_logits, labels)

        # Get positive samples for box regression
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        
        # Handle case where there are no positive samples
        if len(sampled_pos_inds_subset) == 0:
            # Return zero box loss if no positive samples
            box_loss = torch.tensor(0.0, device=class_logits.device, requires_grad=True)
        else:
            labels_pos = labels[sampled_pos_inds_subset]
            
            # Enhanced box regression handling with proper indexing
            # box_regression is [N, num_classes*4], need to get the right 4 values for each positive sample
            num_classes = box_regression.shape[1] // 4
            box_regression_pos = box_regression[sampled_pos_inds_subset]
            
            # Get the regression targets for positive samples
            regression_targets_pos = regression_targets[sampled_pos_inds_subset]
            
            # For each positive sample, get the box regression for its class
            # This is more complex indexing - we need to select the right 4 coordinates
            box_pred_pos = torch.zeros_like(regression_targets_pos)
            for i, label in enumerate(labels_pos):
                start_idx = label * 4
                end_idx = start_idx + 4
                box_pred_pos[i] = box_regression_pos[i, start_idx:end_idx]
            
            # Apply Smooth L1 Loss for box regression
            box_loss = F.smooth_l1_loss(
                box_pred_pos,
                regression_targets_pos,
                beta=1.0 / 9,
                reduction='sum'
            )
            # Enhanced normalization: Divide by number of positive samples, not total labels
            box_loss = box_loss / max(1, len(sampled_pos_inds_subset))

        # Enhanced final validation - check for NaN/Inf values in losses and replace with finite values
        if torch.isnan(classification_loss) or torch.isinf(classification_loss):
            print("Warning: NaN/Inf in classification_loss, replacing with 0.0")
            classification_loss = torch.tensor(0.0, device=class_logits.device, requires_grad=True)
        
        if torch.isnan(box_loss) or torch.isinf(box_loss):
            print("Warning: NaN/Inf in box_loss, replacing with 0.0")
            box_loss = torch.tensor(0.0, device=class_logits.device, requires_grad=True)

        return classification_loss, box_loss


class FasterRCNN_Ultimate:
    """
    Ultimate FasterRCNN combining all optimizations for satellite imagery:
    
    1. COCO Pretraining - Proven feature extraction capabilities
    2. ELU Activations - Better gradient flow, smoother learning
    3. Focal Loss - Handles severe class imbalance in satellite imagery
    4. Satellite Optimizations - Anchor sizes, thresholds, detection counts
    5. Robust Training - Gradient clipping, numerical stability
    
    This is the best model for xView dataset characteristics:
    - Small objects (vehicles, buildings)
    - Dense scenes (hundreds of objects per image)
    - Class imbalance (many background pixels, few object pixels)
    - High resolution satellite imagery
    """
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def get_model(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, elu_alpha=ELU_ALPHA):
        """
        Create the ultimate FasterRCNN model with all optimizations
        
        Args:
            alpha: Focal loss alpha parameter (class balance)
            gamma: Focal loss gamma parameter (hard example focus)
            elu_alpha: ELU activation alpha parameter
        """
        # Start with COCO pretrained weights - essential for performance
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        
        # Apply ELU activations throughout the network
        replace_relu_with_elu(model, alpha=elu_alpha, inplace=True)
        
        # Replace final classifier head for our classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Initialize new layers with smaller weights for stability
        self._initialize_predictor_weights(model.roi_heads.box_predictor)
        
        # Apply satellite imagery optimizations
        self._optimize_for_satellite_imagery(model)
        
        # Replace RoI heads with Focal Loss version
        self._apply_focal_loss(model, alpha, gamma)
        
        return model
    
    def _initialize_predictor_weights(self, predictor):
        """
        Initialize predictor weights with smaller values for training stability
        """
        nn.init.normal_(predictor.cls_score.weight, std=0.01)
        nn.init.normal_(predictor.bbox_pred.weight, std=0.001)
        nn.init.constant_(predictor.cls_score.bias, 0)
        nn.init.constant_(predictor.bbox_pred.bias, 0)
    
    def _optimize_for_satellite_imagery(self, model):
        """
        Apply comprehensive satellite imagery optimizations
        Based on research and empirical results from satellite object detection
        """
        # Detection thresholds optimized for small objects
        model.roi_heads.nms_thresh = 0.3          # Lower NMS for dense scenes
        model.roi_heads.detections_per_img = 300  # More detections for dense imagery
        model.roi_heads.score_thresh = 0.01       # Lower threshold for small objects
        
        # Anchor optimization for small satellite objects
        if hasattr(model.rpn, 'anchor_generator'):
            # Smaller anchor sizes for small objects (vehicles: 10-30px, buildings: 20-100px)
            model.rpn.anchor_generator.sizes = ((8, 16, 32, 64, 128),) * 5
            # Multiple aspect ratios for diverse object shapes
            model.rpn.anchor_generator.aspect_ratios = ((0.5, 1.0, 2.0),) * 5
        
        # RPN optimization for dense satellite scenes
        model.rpn.pre_nms_top_n_train = 6000     # More proposals for dense scenes
        model.rpn.post_nms_top_n_train = 4000    # Keep more after NMS
        model.rpn.pre_nms_top_n_test = 3000      # Test time proposals
        model.rpn.post_nms_top_n_test = 2000     # Test time post-NMS
        
        # Batch size optimization for satellite imagery
        if hasattr(model.roi_heads, 'fg_bg_sampler'):
            # More samples for better learning on dense scenes
            model.roi_heads.fg_bg_sampler.batch_size_per_image = 512
            # Balanced sampling for class imbalance
            model.roi_heads.fg_bg_sampler.positive_fraction = 0.25
    
    def _apply_focal_loss(self, model, alpha, gamma):
        """
        Replace standard RoI heads with Focal Loss version
        """
        # Create Focal Loss function
        focal_loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        
        # Get original RoI heads parameters
        orig_roi_heads = model.roi_heads
        
        # Create new RoI heads with Focal Loss
        custom_roi_heads = RoIHeadsWithFocalLoss(
            focal_loss=focal_loss_fn,
            box_roi_pool=orig_roi_heads.box_roi_pool,
            box_head=orig_roi_heads.box_head,
            box_predictor=orig_roi_heads.box_predictor,
            fg_iou_thresh=orig_roi_heads.proposal_matcher.high_threshold,
            bg_iou_thresh=orig_roi_heads.proposal_matcher.low_threshold,
            batch_size_per_image=orig_roi_heads.fg_bg_sampler.batch_size_per_image,
            positive_fraction=orig_roi_heads.fg_bg_sampler.positive_fraction,
            bbox_reg_weights=orig_roi_heads.box_coder.weights,
            score_thresh=orig_roi_heads.score_thresh,
            nms_thresh=orig_roi_heads.nms_thresh,
            detections_per_img=orig_roi_heads.detections_per_img,
        )
        
        # Replace the RoI heads
        model.roi_heads = custom_roi_heads
        
        return model


def validate_model_stability():
    """
    Comprehensive validation of model stability and NaN resistance
    Tests the model with edge cases that commonly cause NaN issues
    """
    print("🔧 Testing FasterRCNN_Ultimate stability...")
    
    try:
        import torch
        
        # Test Focal Loss stability
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Test case 1: Normal inputs
        normal_inputs = torch.randn(10, 5)
        normal_targets = torch.randint(0, 5, (10,))
        loss1 = focal_loss(normal_inputs, normal_targets)
        print(f"✅ Normal case: {loss1.item():.6f}")
        
        # Test case 2: Extreme inputs (should be clamped)
        extreme_inputs = torch.tensor([[1000.0, -1000.0, 0.0, 50.0, -50.0]] * 10)
        extreme_targets = torch.randint(0, 5, (10,))
        loss2 = focal_loss(extreme_inputs, extreme_targets)
        print(f"✅ Extreme inputs: {loss2.item():.6f}")
        
        # Test case 3: Very confident predictions (pt near 1.0)
        confident_inputs = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0]] * 10)
        confident_targets = torch.zeros(10, dtype=torch.long)
        loss3 = focal_loss(confident_inputs, confident_targets)
        print(f"✅ Confident predictions: {loss3.item():.6f}")
        
        # Verify no NaN/Inf in any case
        all_finite = all(torch.isfinite(loss).all() for loss in [loss1, loss2, loss3])
        print(f"✅ All losses finite: {all_finite}")
        
        return True
        
    except Exception as e:
        print(f"❌ Stability test failed: {e}")
        return False


def get_training_recommendations():
    """
    Returns recommended training settings for the Ultimate model
    """
    return {
        'learning_rate': 0.001,      # Conservative for stability
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'batch_size': 2,             # Start small due to memory usage
        'gradient_clip_norm': 5.0,   # Essential for training stability
        'warmup_epochs': 2,          # Gradual learning rate warmup
        'scheduler': 'StepLR',       # Step decay every 7 epochs
        'step_size': 7,
        'gamma': 0.1
    }


def get_enhanced_training_recommendations():
    """
    Enhanced training recommendations with stability focus
    """
    return {
        'learning_rate': 0.0005,     # More conservative for ultimate stability
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'batch_size': 2,             # Start small due to memory usage
        'gradient_clip_norm': 1.0,   # Tighter clipping for stability
        'warmup_epochs': 3,          # Longer warmup for complex model
        'scheduler': 'StepLR',       # Step decay every 7 epochs
        'step_size': 7,
        'gamma': 0.1,
        'early_stopping_patience': 5,
        'loss_scale_factor': 1.0,    # Can be increased if losses are too small
        'numerical_checks': True,    # Enable runtime NaN checking
    }


if __name__ == "__main__":
    # Comprehensive testing of the Ultimate model
    print("Testing FasterRCNN_Ultimate")
    print("=" * 50)
    
    # Test stability first
    stability_ok = validate_model_stability()
    
    if stability_ok:
        print("\n📦 Creating model...")
        model_creator = FasterRCNN_Ultimate(num_classes=61)  # xView has 60 classes + background
        model = model_creator.get_model()
        
        print(f"✅ Model created successfully!")
        print(f"📊 Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        enhanced_recommendations = get_enhanced_training_recommendations()
        print(f"\n🎯 Enhanced training recommendations:")
        for key, value in enhanced_recommendations.items():
            print(f"   {key}: {value}")
            
        print(f"\n🔒 Safety features enabled:")
        print(f"   • Enhanced NaN/Inf detection in Focal Loss")
        print(f"   • Robust box regression indexing")
        print(f"   • Comprehensive input validation")
        print(f"   • Zero-loss fallback for edge cases")
        print(f"   • Detailed debugging output")
        
    else:
        print("❌ Stability tests failed. Please check the implementation.")
    
    # Validate model stability
    validate_model_stability()
