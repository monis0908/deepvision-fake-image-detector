from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_folder(folder_path):
    valid_images = 0
    corrupted_images = []

    for image_path in folder_path.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                with Image.open(image_path) as img:
                    img.verify()

                valid_images += 1

            except Exception:
                corrupted_images.append(image_path)

    return valid_images, corrupted_images


for split in ["train", "validation", "test"]:
    print(f"\n========== {split.upper()} ==========")

    for label in ["REAL", "FAKE"]:
        folder = DATASET_PATH / split / label

        valid, corrupted = validate_folder(folder)

        print(f"{label}")
        print(f"Valid Images     : {valid}")
        print(f"Corrupted Images : {len(corrupted)}")

        if corrupted:
            print("Corrupted Files:")
            for file in corrupted:
                print(file)