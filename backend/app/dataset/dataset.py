from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset


class FakeImageDataset(Dataset):

    def __init__(self, dataset_path, transform=None):
        """
        dataset_path:
            dataset/train
            dataset/validation
            dataset/test

        transform:
            train_transform
            test_transform
        """

        self.dataset_path = Path(dataset_path)
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self.class_to_label = {
            "REAL": 0,
            "FAKE": 1
        }

        self.load_dataset()

    def load_dataset(self):

        for class_name in self.class_to_label:

            class_folder = self.dataset_path / class_name

            for image_path in class_folder.iterdir():

                if image_path.is_file():

                    self.image_paths.append(image_path)

                    self.labels.append(
                        self.class_to_label[class_name]
                    )

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label