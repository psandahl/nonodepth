import numpy as np
import torch


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
