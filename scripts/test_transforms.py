from pathlib import Path
from PIL import Image

import sys

# Add backend to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))

from app.preprocessing.transforms import train_transform

# Get one sample image
sample_image = next((PROJECT_ROOT / "dataset" / "train" / "REAL").iterdir())

print(f"Image: {sample_image.name}")

# Open image
image = Image.open(sample_image)

print("Original Size:", image.size)

# Apply preprocessing
tensor = train_transform(image)

print("Tensor Shape:", tensor.shape)
print("Tensor Data Type:", tensor.dtype)