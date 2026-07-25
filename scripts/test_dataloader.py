from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))

from app.dataloader.loader import get_dataloaders

train_loader, validation_loader, test_loader = get_dataloaders()

print("Train Batches:", len(train_loader))
print("Validation Batches:", len(validation_loader))
print("Test Batches:", len(test_loader))

images, labels = next(iter(train_loader))

print("Batch Shape:", images.shape)
print("Labels Shape:", labels.shape)

print(labels[:10])