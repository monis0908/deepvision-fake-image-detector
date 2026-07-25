from pathlib import Path

from torch.utils.data import DataLoader

from app.config.settings import BATCH_SIZE
from app.dataset.dataset import FakeImageDataset
from app.preprocessing.transforms import (
    train_transform,
    test_transform
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "dataset"


def get_dataloaders(batch_size=BATCH_SIZE):

    train_dataset = FakeImageDataset(
        DATASET_PATH / "train",
        transform=train_transform
    )

    validation_dataset = FakeImageDataset(
        DATASET_PATH / "validation",
        transform=test_transform
    )

    test_dataset = FakeImageDataset(
        DATASET_PATH / "test",
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        validation_loader,
        test_loader
    )
