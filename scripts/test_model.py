from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))

from app.models.models import FakeImageDetector
from app.dataloader.loader import get_dataloaders

train_loader, _, _ = get_dataloaders()

images, labels = next(iter(train_loader))

model = FakeImageDetector()

outputs = model(images)

print("Input Shape :", images.shape)
print("Output Shape:", outputs.shape)
