import numpy as np
import pathlib
import torch
from torchvision.transforms import Resize, InterpolationMode
import matplotlib.pyplot as plt

from nonodepth.datasets.diode import Diode
from nonodepth.utils.image import tensor_to_np_image, SSIM


def view_triplet(sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    rgb, depth, mask = sample

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

    plt.show()


def view_ssim(x: torch.Tensor, y: torch.Tensor) -> None:

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

    plt.show()

    ssim_loss = ssim.loss(x, y)

    print(f'ssim_loss={ssim_loss.item()}')


if __name__ == '__main__':
    path = pathlib.Path('C:\\Users\\patri\\datasets\\val')

    transform = Resize(
        (384, 384), interpolation=InterpolationMode.BILINEAR, antialias=True)
    target_transform = Resize(
        (384, 384), interpolation=InterpolationMode.BILINEAR, antialias=True)
    diode = Diode(root=path, transform=transform,
                  target_transform=target_transform)

    print(f'Available samples={len(diode)}')

    triplet = diode[44]
    view_triplet(triplet)
    view_ssim(triplet[1], triplet[1])
