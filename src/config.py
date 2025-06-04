import os
from dotenv import load_dotenv
import torch

load_dotenv(dotenv_path="/Volumes/T7 Shield/rps_github/terrarian_recognition/.env")

# Data paths
DATASET_DIR = os.path.join("data", "dataset") 
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "train_images")
TILED_IMAGES_DIR = os.path.join(DATASET_DIR, "tiled_images")
ANNOTATIONS_PATH = os.path.join("data", "xView_train.geojson")
MODEL_SAVE_DIR = os.path.join("data", "models")

# Model selection
MODEL_NAME = os.getenv("MODEL_NAME", "fasterrcnn")  
MODEL_DIR = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "model.pth")
MODEL_METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.pkl")

# Inference output directory
default_output_dir = os.path.join("data", "output")
INFERENCE_OUTPUT_DIR = os.getenv("INFERENCE_OUTPUT_DIR", default_output_dir)

# Hyperparameters
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 15))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.01))
MOMENTUM = float(os.getenv("MOMENTUM", 0.9))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.0005))
N_TILES = int(os.getenv("N_TILES", 10))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", 0.2))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# Focal/ELU params
FOCAL_ALPHA = float(os.getenv("FOCAL_ALPHA", 1.0))
FOCAL_GAMMA = float(os.getenv("FOCAL_GAMMA", 2.0))
ELU_ALPHA = float(os.getenv("ELU_ALPHA", 1.0))

NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))

DEVICE = torch.device("cuda" if torch.cuda.is_available() 
                    #   else "mps" if torch.backends.mps.is_available()
                      else "cpu")