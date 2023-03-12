import numpy as np
import pathlib
import torch
from torchvision.transforms import Resize, InterpolationMode
import matplotlib.pyplot as plt

from nonodepth.datasets.diode import Diode
from nonodepth.networks.basicunet import BasicUNet
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


def model_eval() -> None:
    model = BasicUNet(in_channels=3)

    print(f'Total number of parameters={total_parameters(model)}')
    print(f'Trainable number of parameters={trainable_parameters(model)}')
    print_trainable_parameters(model)

    t = torch.rand((3, 3, 384, 384), dtype=torch.float32)
    tt = model(t)

    print(tt.shape)


if __name__ == '__main__':
    # path = pathlib.Path('C:\\Users\\patri\\datasets\\val')
    # basic_visualization(path)

    model_eval()
