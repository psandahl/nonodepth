from __future__ import annotations

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from nonodepth.utils.image import SSIM, gradient_loss
from nonodepth.utils.model import trainable_parameters


def fit(model: nn.Module,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        learning_rate: float) -> None:

    # Print parameter information.
    print(f'Parameters for training')
    print(f'- Training batches={len(training_loader)}')
    print(f'- Validation batches={len(validation_loader)}')
    print(f'- Epochs={epochs}')
    print(f'- Learning rate={learning_rate}')
    print(f'- Trainable parameters={trainable_parameters(model)}')

    # Setup optimizer.
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           amsgrad=False)

    # Setup loss function.
    ssim = SSIM()

    def loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ssim_loss_weight = 1.0
        grad_loss_weight = 0.5

        ssim_loss = ssim.loss(predictions, targets)
        # print(ssim_loss)

        grad_loss, _, _ = gradient_loss(
            predictions, targets)
        # print(grad_loss)

        return ssim_loss * ssim_loss_weight + grad_loss * grad_loss_weight

    # Run training loop.
    for epoch in range(epochs):
        print(f'Epoch={epoch + 1} of {epochs}', end=':', flush=True)
        batches = 0
        running_loss = 0.
        for batch, (images, targets, _masks) in enumerate(training_loader):

            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        print(f' training loss={running_loss / batches}')
