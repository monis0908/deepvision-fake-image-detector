from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.trainer import Trainer


def main() -> None:
    """Instantiate the trainer and display its configured components."""
    trainer = Trainer()

    print("Model Loaded Successfully!")
    print("Device:", next(trainer.model.parameters()).device)
    print("Loss Function:", trainer.criterion)
    print("Optimizer:", trainer.optimizer)


if __name__ == "__main__":
    main()
