from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder_path):
    count = 0

    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1

    return count


def print_split(split_name):
    real_folder = DATASET_PATH / split_name / "REAL"
    fake_folder = DATASET_PATH / split_name / "FAKE"

    real_count = count_images(real_folder)
    fake_count = count_images(fake_folder)

    print("=" * 40)
    print(f"{split_name.upper()} DATASET")
    print("=" * 40)
    print(f"REAL Images : {real_count}")
    print(f"FAKE Images : {fake_count}")
    print(f"TOTAL Images : {real_count + fake_count}")
    print()


print_split("train")
print_split("validation")
print_split("test")