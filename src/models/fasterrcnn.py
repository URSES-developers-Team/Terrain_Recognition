from torchvision import models

class FasterRCNN:
    """
    Standard Faster R-CNN model (ResNet50 backbone, FPN, COCO-80 pre-trained weights).
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self._create_model(num_classes)

    def _create_model(self, num_classes):
        weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        return model

    def get_model(self):
        return self.model