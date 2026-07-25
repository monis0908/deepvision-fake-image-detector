import torch
import torch.nn as nn
import torch.optim as optim

from app.models.models import FakeImageDetector
from app.config.settings import (
    DEVICE,
    LEARNING_RATE
)


class Trainer:

    def __init__(self):

        # Initialize Model
        self.model = FakeImageDetector().to(DEVICE)

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

    def train_one_epoch(self, train_loader):

        # Put model in training mode
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        batch_idx = 0

        for images, labels in train_loader:

            batch_idx += 1

            if batch_idx == 1:
                print("First batch started...")

            # Move data to CPU/GPU
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Clear previous gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update model weights
            self.optimizer.step()

            if batch_idx == 1:
                print("First batch completed.")

            # Store loss
            running_loss += loss.item()

            # Predicted class
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        return epoch_loss, epoch_accuracy

    def validate(self, validation_loader):

        # Put model in evaluation mode
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        # Disable gradient calculation
        with torch.no_grad():

            for images, labels in validation_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        validation_loss = running_loss / len(validation_loader)
        validation_accuracy = 100 * correct / total

        return validation_loss, validation_accuracy

    def save_model(self, path):

        torch.save(
            self.model.state_dict(),
            path
        )