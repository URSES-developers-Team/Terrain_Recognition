import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pickle
import os
from config import MODEL_SAVE_DIR

def plot_metrics(train_losses, val_maps, save_path=None):
    """
    Plots training loss and validation mAP curves.
    """
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_maps, label="Val mAP")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss and Validation mAP")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_checkpoint(model, optimizer, epoch, path):
    """
    Saves model and optimizer state.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def load_checkpoint(model, optimizer, path):
    """
    Loads model and optimizer state using DEVICE from config.
    """
    from config import DEVICE
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    return model, optimizer, epoch

def set_seed(seed):
    """
    Sets random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log(msg):
    """
    Simple logger.
    """
    print(f"[LOG] {msg}")

def get_map_metric():
    return MeanAveragePrecision()

def save_metrics(metrics, model_name):
    from config import MODEL_SAVE_DIR
    model_dir = os.path.join(MODEL_SAVE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    metrics_path = os.path.join(model_dir, "model_metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

def load_metrics(model_name):
    from config import MODEL_SAVE_DIR
    metrics_path = os.path.join(MODEL_SAVE_DIR, model_name, "model_metrics.pkl")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found for model: {model_name}")
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)
    return metrics