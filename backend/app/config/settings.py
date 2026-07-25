from pathlib import Path
import torch


# ----------------------------
# Project Paths
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = PROJECT_ROOT / "dataset"

CHECKPOINTS_PATH = PROJECT_ROOT / "backend" / "checkpoints"

CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Training Hyperparameters
# ----------------------------

BATCH_SIZE = 8

LEARNING_RATE = 0.001

NUM_EPOCHS = 10


# ----------------------------
# Model
# ----------------------------

NUM_CLASSES = 2


# ----------------------------
# Device
# ----------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ----------------------------
# Checkpoint File
# ----------------------------

BEST_MODEL_PATH = CHECKPOINTS_PATH / "best_model.pth"