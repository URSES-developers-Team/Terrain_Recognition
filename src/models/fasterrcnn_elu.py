import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from models.fasterrcnn import FasterRCNN
from config import *


def replace_relu_with_elu(module, alpha=1.0, inplace=True):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ELU(alpha=alpha, inplace=inplace))
        else:
            replace_relu_with_elu(child, alpha=alpha, inplace=inplace)

class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1.0 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class RoIHeadsWithFocalLoss(RoIHeads):
    def __init__(self, focal_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = focal_loss

    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        classification_loss = self.focal_loss(class_logits, labels)
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1.0 / 9,
            reduction='sum'
        )
        box_loss = box_loss / labels.numel()
        return classification_loss, box_loss

class FasterRCNN_ELU(FasterRCNN):
    def get_model(self, num_classes, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, elu_alpha=ELU_ALPHA):
        model = fasterrcnn_resnet50_fpn(weights="COCO_V1")
        replace_relu_with_elu(model, alpha=elu_alpha, inplace=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        focal_loss_fn = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        orig_roi_heads = model.roi_heads
        
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
        model.roi_heads = custom_roi_heads
        return model
