from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.dataloader.loader import get_dataloaders
from app.config.settings import NUM_EPOCHS, BEST_MODEL_PATH
from training.trainer import Trainer


def main():

    # Load DataLoaders
    train_loader, validation_loader, _ = get_dataloaders()

    # Initialize Trainer
    trainer = Trainer()

    best_validation_accuracy = 0.0

    print("=" * 60)
    print("Starting Training...")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} starting...")
        train_loss, train_accuracy = trainer.train_one_epoch(train_loader)

        validation_loss, validation_accuracy = trainer.validate(validation_loader)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            trainer.save_model(BEST_MODEL_PATH)
            print("✅ Best model saved!")

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Train Accuracy  : {train_accuracy:.2f}%")
        print(f"Validation Loss : {validation_loss:.4f}")
        print(f"Validation Acc. : {validation_accuracy:.2f}%")
        print(f"Best Validation : {best_validation_accuracy:.2f}%")

    print("\nTraining Completed!")


if __name__ == "__main__":
    main()
