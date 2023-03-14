from __future__ import annotations

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from nonodepth.utils.image import SSIM, gradient_loss
from nonodepth.utils.model import trainable_parameters


def fit(device: torch.device,
        model: nn.Module,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        learning_rate: float) -> None:

    # Print parameter information.
    print(f'Parameters for training')
    print(f'- Device={device}')
    print(f'- Training batches={len(training_loader)}')
    print(f'- Validation batches={len(validation_loader)}')
    print(f'- Epochs={epochs}')
    print(f'- Learning rate={learning_rate}')
    print(f'- Trainable parameters={trainable_parameters(model)}')

    # Setup optimizer.
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)

    # Setup loss function.
    ssim = SSIM()

    def loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ssim_loss_weight = 0.85
        grad_loss_weight = 0.15

        ssim_loss = ssim.loss(predictions, targets)

        grad_loss, _, _ = gradient_loss(
            predictions, targets)
        # print(f' ssim_loss={ssim_loss.mean()} grad_loss={grad_loss.mean()}',
        #      end=' ', flush=True)

        return ssim_loss * ssim_loss_weight + grad_loss * grad_loss_weight
        # return ssim_loss

    # Run training/validation loop.
    for epoch in range(epochs):
        print(f'Epoch={epoch + 1}/{epochs}', end=': ', flush=True)

        training_batches = 0
        training_loss = 0.
        for images, targets, _masks in training_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_batches += 1

        print(f'loss={training_loss / training_batches}', end=' ', flush=True)

        with torch.no_grad():
            validation_batches = 0
            validation_loss = 0
            for images, targets, _masks in validation_loader:
                images, targets = images.to(device), targets.to(device)

                predictions = model(images)
                loss = loss_fn(predictions, targets)

                validation_loss += loss.item()
                validation_batches += 1

        print(f'val_loss={validation_loss / validation_batches}')
