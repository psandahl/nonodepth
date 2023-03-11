from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def tensor_to_np_image(img: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor, representing an image, to a numpy image.

    Parameters:
        img: Tensor image.

    Returns:
        Numpy image.
    """
    return np.transpose(img.numpy(), (1, 2, 0))


def np_to_tensor_image(img: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy image to a tensor representing an image.

    Parameters:
        img: Numpy image.

    Returns:
        Tensor image.
    """
    if img.ndim == 3:
        return torch.from_numpy(np.transpose(img, (2, 0, 1)))
    elif img.ndim == 2:
        # Special case if no explicit channel.
        img = np.expand_dims(img, 0)
        return torch.from_numpy(img)
    else:
        return None


def gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradients in x and y directions for the given tensor.

    Parameters:
        x: Input tensor. Must have dimensions (n c h w)

    Returns:
        dx, dy
    """
    right = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    bottom = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]

    dx, dy = right - x, bottom - x

    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the gradient loss, i.e. the L1 loss for the gradient images.
    """
    dx_pred, dy_pred = gradients(pred)
    dx_target, dy_target = gradients(target)

    diff_x = torch.abs(dx_target - dx_pred)
    diff_y = torch.abs(dy_target - dy_pred)

    score = torch.mean(diff_x + diff_y)

    return score, diff_x, diff_y


class SSIM(nn.Module):
    """
    Layer to compute the SSIM image from a pair of images. Inspired from: Monodepth2.
    """

    def __init__(self: SSIM) -> None:
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self: SSIM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        # return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return SSIM_n / SSIM_d

    def loss(self: SSIM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute an inverse SSIM score, where zero is perfect match and one
        is negative match.
        """
        ssim_img = self.forward(x, y)
        return torch.clamp((1. - ssim_img.mean()) / 2., 0., 1.0)
