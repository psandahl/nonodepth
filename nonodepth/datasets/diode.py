from __future__ import annotations

import numpy as np
import pathlib
import random
import torch
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from typing import Callable, Optional

from nonodepth.utils.image import np_to_tensor_image


class Diode(VisionDataset):
    """
    Dataset class to load Diode data: https://diode-dataset.org/
    """

    # TODO:
    # Shuffle after each epoch?
    # Augment images?
    # Logarithmic depth?

    def __init__(self: Diode,
                 root: pathlib.Path,
                 split: str = 'train',
                 shuffle: bool = True,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        """
        Initialize a Diode dataset from an unpacked diode directory. The
        class will recursively search for png files, and expect that
        for each png file there's a _depth.npy and _depth_mask.npy.

        Parameters:
            root: The path to the root for the Diode dataset. Must exist and be a directory.
            split: Split the dataset into 'train' (90%), 'test' (10%), or not split at all (100%).
            shuffle: Shuffle the files randomly.
            transforms: Functions to transform both image and target. Must be None.
            transform: Transforms performed on the loaded image Tensor.
            target_transform: Transforms performed on the loaded depth and mask images.
        """
        assert root.is_dir() and root.exists()
        assert transforms is None

        super().__init__(root, transforms, transform, target_transform)

        # Build list of png file names.
        self.files = list(self.root.glob('**/*.png'))
        if shuffle:
            random.shuffle(self.files)

        self.num_files = len(self.files)

        if split == 'train':
            # Take the first 90 percent.
            self.num_files = int(round(self.num_files * 0.9))
            self.files = self.files[:self.num_files]
        elif split == 'test':
            # Take the last 10 percent.
            start = int(round(self.num_files * 0.9))
            self.files = self.files[start:]
            self.num_files = len(self.files)

    def __len__(self: Diode) -> int:
        return self.num_files

    def __getitem__(self: Diode, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        png_path = self.root / self.files[index]

        # Always convert to float and normalize.
        rgb_img = read_image(str(png_path)).to(torch.float32) / 255.
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        base = png_path.parent
        stem = png_path.stem

        depth_path = base / pathlib.Path(stem + '_depth.npy')
        mask_path = base / pathlib.Path(stem + '_depth_mask.npy')

        depth_img = np_to_tensor_image(np.load(depth_path))
        mask_img = np_to_tensor_image(np.load(mask_path))

        # Use the mask image to set zeros where mask is zero.
        depth_img = depth_img * mask_img

        if self.target_transform is not None:
            depth_img = self.target_transform(depth_img)
            mask_img = self.target_transform(mask_img)

        return rgb_img, depth_img, mask_img
