import pytest
import pandas as pd
import torch
import numpy as np
import os
from pathlib import Path
import cv2
import albumentations as A

from torch.utils.data import DataLoader
from src.data.dataset import SegmentationDataset
from src.data.dataset import get_dataloaders
from icecream import ic


@pytest.fixture
def mock_config():
    return {
        "data": {"csv_file": "./data/train.csv", "train_val_split": 0.2},
        "training": {"batch_size": 4, "num_workers": 0, "random_seed": 42},
        "augmentation": {
            "train": {"resize_dim": 320, "horizontal_flip": 0.5, "vertical_flip": 0.5}
        },
    }


@pytest.fixture
def sample_df():
    # create a sample dataframe
    current_dir = Path(__file__).parent
    data = {
        "images": [str(current_dir) + "/test_training.jpg"],
        "masks": [str(current_dir) + "/test_gt.png"],
    }
    df = pd.DataFrame(data)
    ic(df)
    ic(df["images"])
    ic(df["masks"])
    return df


@pytest.fixture
def sample_dataset(sample_df):
    return SegmentationDataset(sample_df)


@pytest.fixture
def sample_augmented_dataset(sample_df):
    transform = A.Compose(
        [
            A.Resize(height=320, width=320),
        ]
    )
    return SegmentationDataset(sample_df, transform=transform)


def test_dataset_initialization(sample_dataset):
    assert isinstance(sample_dataset, SegmentationDataset)
    assert len(sample_dataset) == 1


def test_dataset_length(sample_df):
    dataset = SegmentationDataset(sample_df)
    assert len(dataset) == len(sample_df)


def test_dataset_getitem(sample_dataset):
    img, mask = sample_dataset[0]
    ic(img.shape)

    # check return types
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    # check shapes
    assert img.dim() == 3
    assert mask.dim() == 3
    assert img.shape[0] == 3  # RGB channels
    assert mask.shape[0] == 1  # grayscale single channel

    # check value ranges
    assert torch.all(img >= 0.0) and torch.all(img <= 1.0)
    assert torch.all(mask >= 0.0) and torch.all(mask <= 1.0)


def test_dataset_with_transform(sample_augmented_dataset):

    img, mask = sample_augmented_dataset[0]

    # Check if the transforms were applied (size should be 256x256)
    assert img.shape[1] == 320  # Height
    assert img.shape[2] == 320  # Width
    assert mask.shape[1] == 320  # Height
    assert mask.shape[2] == 320  # Width


def test_get_dataloaders_return_correct_types(mock_config):
    train_dataloader, val_dataloader = get_dataloaders(mock_config)

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(val_dataloader, DataLoader)


def test_get_dataloaders_correct_split(mock_config):
    train_dataloader, val_dataloader = get_dataloaders(mock_config)

    assert len(train_dataloader.dataset) >= 0
    assert len(val_dataloader.dataset) < len(train_dataloader.dataset)


def test_get_dataloaders_batch_size(mock_config):
    train_dataloader, val_dataloader = get_dataloaders(mock_config)

    assert train_dataloader.batch_size == mock_config["training"]["batch_size"]
    assert val_dataloader.batch_size == mock_config["training"]["batch_size"]


def test_dataloader_output_shapes(mock_config): ...
