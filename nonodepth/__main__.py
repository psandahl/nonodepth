import argparse

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
import sys
import torch
from torchvision.transforms import Resize, InterpolationMode
from torch.utils.data import DataLoader

from nonodepth.datasets.diode import Diode
from nonodepth.networks.basicunet import BasicUNet
import nonodepth.trainer as trainer
from nonodepth.utils.image import gradients, gradient_loss, tensor_to_np_image, SSIM
from nonodepth.utils.model import total_parameters, trainable_parameters, print_trainable_parameters


def view_triplet(sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    rgb, depth, mask = sample

    plt.figure(figsize=(9, 9))

    plt.subplot(1, 3, 1)
    plt.imshow(tensor_to_np_image(rgb))
    plt.title('RGB Image')

    plt.subplot(1, 3, 2)
    plt.imshow(tensor_to_np_image(depth), cmap='gray',
               vmin=torch.min(depth), vmax=torch.max(depth))
    plt.title('Adapted Depth Map')

    plt.subplot(1, 3, 3)
    plt.imshow(tensor_to_np_image(mask), cmap='gray')
    plt.title('Depth Mask')


def view_ssim(x: torch.Tensor, y: torch.Tensor) -> None:
    plt.figure(figsize=(9, 9))

    plt.subplot(1, 3, 1)
    plt.imshow(tensor_to_np_image(x), cmap='gray',
               vmin=torch.min(x), vmax=torch.max(x))
    plt.title('Image X')

    plt.subplot(1, 3, 2)
    plt.imshow(tensor_to_np_image(y), cmap='gray',
               vmin=torch.min(y), vmax=torch.max(y))
    plt.title('Image Y')

    ssim = SSIM()
    ssim_img = ssim(x, y)

    plt.subplot(1, 3, 3)
    plt.imshow(tensor_to_np_image(ssim_img), cmap='gray',
               vmin=torch.min(ssim_img), vmax=torch.max(ssim_img))
    plt.title('SSIM')

    ssim_loss = ssim.loss(x, y)
    plt.suptitle(f'SSIM loss={ssim_loss}')


def view_gradients(x: torch.Tensor) -> None:
    plt.figure(figsize=(9, 9))

    xx = x.expand((1,) + x.shape)
    dx, dy = gradients(xx)

    dx = tensor_to_np_image(dx[0])
    dy = tensor_to_np_image(dy[0])

    plt.subplot(1, 3, 1)
    plt.imshow(tensor_to_np_image(x), cmap='gray',
               vmin=torch.min(x), vmax=torch.max(x))
    plt.title('depth')

    plt.subplot(1, 3, 2)
    plt.imshow(dx, vmin=np.min(dx), vmax=np.max(dx), cmap='gray')
    plt.title('dx')

    plt.subplot(1, 3, 3)
    plt.imshow(dy, vmin=np.min(dy), vmax=np.max(dy), cmap='gray')
    plt.title('dy')


def view_gradient_loss(x: torch.Tensor) -> None:
    plt.figure(figsize=(9, 9))

    xx = x.expand((1,) + x.shape)
    yy = xx.clone()
    yy[:, :, 150, :] = torch.max(xx)

    plt.subplot(1, 4, 1)
    plt.imshow(tensor_to_np_image(x), cmap='gray',
               vmin=torch.min(x), vmax=torch.max(x))
    plt.title('target')

    plt.subplot(1, 4, 2)
    plt.imshow(tensor_to_np_image(yy[0]), cmap='gray',
               vmin=torch.min(yy[0]), vmax=torch.max(yy[0]))
    plt.title('pred')

    loss, diff_x, diff_y = gradient_loss(yy, xx)

    plt.suptitle(f'Gradient loss={loss.item()}')

    plt.subplot(1, 4, 3)
    plt.imshow(tensor_to_np_image(diff_x[0]), cmap='gray',
               vmin=torch.min(diff_x[0]), vmax=torch.max(diff_x[0]))
    plt.title('diff x')

    plt.subplot(1, 4, 4)
    plt.imshow(tensor_to_np_image(diff_y[0]), cmap='gray',
               vmin=torch.min(diff_y[0]), vmax=torch.max(diff_y[0]))
    plt.title('diff y')


def basic_visualization(path: pathlib.Path) -> None:
    transform = Resize(
        (384, 384), interpolation=InterpolationMode.BILINEAR, antialias=True)
    target_transform = Resize(
        (384, 384), interpolation=InterpolationMode.BILINEAR, antialias=True)
    diode = Diode(root=path, shuffle=True,
                  transform=transform,
                  target_transform=target_transform)

    print(f'Available samples={len(diode)}')

    triplet = diode[44]
    view_triplet(triplet)
    view_ssim(triplet[1], triplet[1])
    view_ssim(triplet[1], triplet[2])
    view_gradients(triplet[1])
    view_gradient_loss(triplet[1])

    plt.show()


def train_diode(dataset_path: pathlib.Path,
                model_path: pathlib.Path,
                image_size: tuple[int, int],
                training_batch: int,
                validation_batch: int,
                epochs: int,
                learning_rate: float,
                no_cuda: bool
                ) -> bool:
    # Configure device.
    if no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    print(type(device))

    # Setup datasets and dataloaders.
    transform = Resize(
        image_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
    target_transform = Resize(
        image_size, interpolation=InterpolationMode.BILINEAR, antialias=True)

    random.seed(1598)
    training_dataset = Diode(root=dataset_path,
                             split='train',
                             transform=transform,
                             target_transform=target_transform)
    random.seed(1598)  # To get same initial shuffle.
    validation_dataset = Diode(root=dataset_path,
                               split='test',
                               transform=transform,
                               target_transform=target_transform)

    training_loader = DataLoader(
        training_dataset, batch_size=training_batch, shuffle=True, num_workers=3)
    validation_loader = DataLoader(
        validation_dataset, batch_size=validation_batch, shuffle=False, num_workers=3)

    # Create model.
    model = BasicUNet(in_channels=3)
    model.to(device)

    # Run training.
    trainer.fit(device=device,
                model=model,
                training_loader=training_loader,
                validation_loader=validation_loader,
                epochs=epochs,
                learning_rate=learning_rate)

    return True


def main() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--action', choices=['train-diode'], required=True, help='Available actions')
    parser.add_argument('--dataset', type=pathlib.Path,
                        help='Dataset path')
    parser.add_argument('--training-batch', type=int,
                        default=16, help='Batch size for the training data')
    parser.add_argument('--validation-batch', type=int,
                        default=8, help='Batch size for the validation data')
    parser.add_argument('--image-width', type=int, default=384,
                        help='Image width as input to network')
    parser.add_argument('--image-height', type=int, default=384,
                        help='Image height as input to network')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float,
                        default=1e-03, help='Learning rate')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    opts = parser.parse_args()

    if opts.action == 'train-diode':
        dataset = opts.dataset
        if dataset is None:
            parser.print_usage()
            print('Error: Dataset option is missing')
            return False

        if dataset.exists() and dataset.is_dir():
            image_size = opts.image_width, opts.image_height
            return train_diode(dataset_path=dataset,
                               model_path=pathlib.Path(),
                               image_size=image_size,
                               training_batch=opts.training_batch,
                               validation_batch=opts.validation_batch,
                               epochs=opts.epochs,
                               learning_rate=opts.learning_rate,
                               no_cuda=opts.no_cuda)
        else:
            print('Error: The DIODE dataset must be an existing directory')
            return False

    return True


if __name__ == '__main__':
    if main() == True:
        sys.exit(0)
    else:
        sys.exit(1)
