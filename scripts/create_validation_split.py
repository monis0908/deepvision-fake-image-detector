from pathlib import Path
import random
import shutil

# -----------------------------
# Configuration
# -----------------------------
VALIDATION_SPLIT = 0.15

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"

TRAIN_PATH = DATASET_PATH / "train"
VALIDATION_PATH = DATASET_PATH / "validation"

CLASSES = ["REAL", "FAKE"]

random.seed(42)  # Makes the split reproducible


def create_validation_split():
    print("=" * 50)
    print("Creating Validation Dataset")
    print("=" * 50)

    for class_name in CLASSES:

        train_folder = TRAIN_PATH / class_name
        validation_folder = VALIDATION_PATH / class_name

        # Get all image files
        image_files = [
            file for file in train_folder.iterdir()
            if file.is_file()
        ]

        total_images = len(image_files)

        validation_count = int(total_images * VALIDATION_SPLIT)

        # Shuffle images randomly
        random.shuffle(image_files)

        # Select images for validation
        validation_images = image_files[:validation_count]

        # Move selected images
        for image in validation_images:
            shutil.move(
                str(image),
                str(validation_folder / image.name)
            )

        print(f"{class_name}")
        print(f"Original Images : {total_images}")
        print(f"Moved           : {validation_count}")
        print(f"Remaining Train : {total_images - validation_count}")
        print()

    print("Validation dataset created successfully!")


if __name__ == "__main__":
    create_validation_split()