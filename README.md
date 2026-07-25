# DeepVision - Fake Image Detector

DeepVision is a machine learning project for detecting whether an image is real or AI-generated. The repository includes a PyTorch-based training pipeline, custom dataset and dataloader modules, a ResNet18 classifier, and supporting training utilities.

## Features

- Binary image classification with PyTorch and ResNet18
- Custom dataset and dataloader implementation
- Training and validation pipeline
- Checkpoint saving for best model weights
- Simple frontend and backend structure for experimentation and deployment

## Project Architecture

- backend/app/config: project settings and configuration
- backend/app/dataloader: data loading utilities
- backend/app/dataset: dataset class
- backend/app/models: model definition
- backend/app/preprocessing: image transforms
- backend/training: training and validation logic
- scripts: dataset and training verification scripts
- frontend: web interface assets

## Folder Structure

```text
backend/
  app/
    config/
    dataloader/
    dataset/
    models/
    preprocessing/
  training/
  requirements.txt
scripts/
frontend/
dataset/
README.md
LICENSE
```

## Installation

Clone the repository and navigate into it:

```bash
git clone <repository-url>
cd DeepVision-Fake-Image-Detector
```

## Virtual Environment Setup

Create and activate a virtual environment:

```bash
python -m venv backend/venv
backend/venv/Scripts/activate
```

## Dependency Installation

Install the required Python packages:

```bash
pip install -r backend/requirements.txt
```

## Dataset Instructions

Place the training, validation, and test image folders under the dataset directory with the following structure:

```text
dataset/
  train/
    REAL/
    FAKE/
  validation/
    REAL/
    FAKE/
  test/
    REAL/
    FAKE/
```

## How to Run

Train the model with:

```bash
python backend/training/train.py
```

## Training Instructions

The training script loads the dataset, initializes the ResNet18-based model, runs training and validation, and saves the best checkpoint to backend/checkpoints/best_model.pth.

## Tech Stack

- Python
- PyTorch
- TorchVision
- FastAPI
- React
- Pillow

## Future Improvements

- Add experiment tracking and logging
- Support distributed training
- Improve inference API and deployment workflow
- Expand evaluation metrics and visualization

## License

This project is licensed under the MIT License. See the LICENSE file for details.
