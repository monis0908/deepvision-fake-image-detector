from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))

from app.dataset.dataset import FakeImageDataset
from app.preprocessing.transforms import train_transform

dataset = FakeImageDataset(
    dataset_path=PROJECT_ROOT / "dataset" / "train",
    transform=train_transform
)

print("Dataset Size:", len(dataset))

image, label = dataset[0]

print("Image Shape:", image.shape)
print("Label:", label)