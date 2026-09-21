from pathlib import Path
import torch

#Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#Data
DATA_DIR = PROJECT_ROOT/"data"
RAW_DATA_DIR = DATA_DIR/"raw"
PROCESSED_DATA_DIR = DATA_DIR/"processed"

#Models
MODEL_DIR = PROJECT_ROOT/"models"
CHECKPOINT_DIR = MODEL_DIR/"checkpoints"

#Outputs
OUTPUT_DIR = PROJECT_ROOT/"outputs"

#Training
BATCH_SIZE = 64
NUM_WORKERS = 2
IMAGE_SIZE = 32

#Training Hyperparameters
EPOCHS = 20
LEARNING_RATE = 0.001

#Random Seed
SEED = 42

#Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#Create directories if they don't exist
for directory in [

    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)