from config import *
from . import fasterrcnn, fasterrcnn_elu

def get_model(model_name, num_classes):
    if model_name == "fasterrcnn_elu":
        return fasterrcnn_elu.FasterRCNN_ELU(num_classes).get_model(
            num_classes=num_classes,
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            elu_alpha=ELU_ALPHA
        )
    return fasterrcnn.FasterRCNN(num_classes).get_model()