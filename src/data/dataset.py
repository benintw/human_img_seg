"""Dataset and data loading utilities for human image segmentation.

This module provides the core data handling components for the human image segmentation project:
    - SegmentationDataset: Custom PyTorch Dataset for loading image-mask pairs
    - Data transformation pipelines using Albumentations
    - DataLoader creation with configurable parameters

The data pipeline expects:
    - Images in RGB format
    - Binary segmentation masks
    - A CSV file containing paths to image-mask pairs

Typical usage:
    ```python
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_loader, val_loader = get_dataloaders(config)
    ```
"""

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from icecream import ic
from typing import Any

# to be removed later
from typing import Final
import matplotlib.pyplot as plt
import yaml


class SegmentationDataset(Dataset):
    """Custom Dataset class for image segmentation tasks.

    This dataset loads image and mask pairs from a DataFrame containing their file paths.
    Images are loaded in RGB format and masks are normalized to [0, 1] range.

    Args:
        df (pd.DataFrame): DataFrame containing 'images' and 'masks' columns with file paths
        transform (A.Compose | None): Albumentations transformations to apply to images and masks
    """

    def __init__(self, df: pd.DataFrame, transform: A.Compose | None = None) -> None:
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Loads and preprocesses an image-mask pair.

        Args:
            idx (int): Index of the sample to load

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Preprocessed image and mask tensors
                - image: Shape (C, H, W), normalized to [0, 1]
                - mask: Shape (1, H, W), normalized to [0, 1]
        """
        row = self.df.iloc[idx]
        img_path = row["images"]
        mask_path = row["masks"]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # (h,w,c)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # (h,w)
        mask = np.expand_dims(mask, axis=-1)  # (h,w,c=1) # mask = mask[..., np.newaxis]

        if self.transform is not None:
            data = self.transform(image=img, mask=mask)
            img = data["image"]
            mask = data["mask"]

        # we want to return (channel, h, w)
        img = torch.from_numpy(img).permute(2, 0, 1).type(torch.float32) / 255.0
        mask = torch.from_numpy(mask).permute(2, 0, 1).type(torch.float32) / 255.0

        return img, mask


def get_transforms(config: dict[str, Any]) -> tuple[A.Compose, A.Compose]:

    resize_dim = config["data"]["augmentation"]["train"].get("resize_dim", 320)

    train_transform: A.Compose = A.Compose(
        [
            A.Resize(height=resize_dim, width=resize_dim),
            A.HorizontalFlip(
                p=config["data"]["augmentation"]["train"].get("horizontal_flip", 0.0)
            ),
            A.VerticalFlip(
                p=config["data"]["augmentation"]["train"].get("vertical_flip", 0.0)
            ),
        ],
        is_check_shapes=False,
    )

    val_transform: A.Compose = A.Compose(
        [A.Resize(height=resize_dim, width=resize_dim)], is_check_shapes=False
    )

    return train_transform, val_transform


def get_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    df = pd.read_csv(config["data"]["csv_file"])
    
    train_df, val_df = train_test_split(
        df,
        test_size=config["data"]["train_val_split"],
        random_state=config["random_seed"],
    )

    train_transform, val_transform = get_transforms(config)

    train_dataset = SegmentationDataset(train_df, train_transform)
    val_dataset = SegmentationDataset(val_df, val_transform)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    return train_dataloader, val_dataloader


def main() -> None:

    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_loader, val_loader = get_dataloaders(config)

    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    # df = pd.read_csv(config["data"]["csv_file"])
    # ic(df.shape)
    # ic(len(df))

    # train_df, val_df = train_test_split(df, test_size=config["data"]["train_val_split"])
    # ic(train_df.shape)
    # ic(val_df.shape)

    # train_ds = SegmentationDataset(train_df)
    # val_ds = SegmentationDataset(val_df)

    # ic(len(train_ds))
    # ic(len(val_ds))

    # # get a sample from train_ds
    # img, mask = train_ds[0]
    # ic(img.shape)
    # ic(mask.shape)

    # idx = 0
    # row = df.iloc[idx]
    # ic(row)
    # img_path = row["images"]
    # mask_path = row["masks"]

    # img = cv2.imread(img_path)
    # ic(type(img))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # ic(img.shape)

    # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) / 255.0
    # ic(mask.shape)

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # ax1.set_title("Image")
    # ax1.imshow(img)

    # ax2.set_title("Ground Truth")
    # ax2.imshow(mask)

    # plt.show()
    ...


if __name__ == "__main__":
    main()
