from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config.settings import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHECKPOINTS_PATH,
    DATASET_PATH,
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
)


def main() -> None:
    """Display the configured project settings."""
    print("Dataset Path :", DATASET_PATH)
    print("Checkpoint Path :", CHECKPOINTS_PATH)
    print("Best Model :", BEST_MODEL_PATH)
    print("Batch Size :", BATCH_SIZE)
    print("Learning Rate :", LEARNING_RATE)
    print("Epochs :", NUM_EPOCHS)
    print("Device :", DEVICE)


if __name__ == "__main__":
    main()
