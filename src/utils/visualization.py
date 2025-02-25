import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as A
from pathlib import Path
import numpy as np


def show_images(
    image: torch.Tensor, mask: torch.Tensor, pred_image: torch.Tensor | None = None
) -> None:
    """Display image segmentation results side by side.

    Args:
        image: Input image tensor of shape (C, H, W)
        mask: Ground truth mask tensor of shape (C, H, W)
        pred_image: Optional predicted mask tensor of shape (C, H, W)
    """
    # Validate inputs
    if not all(
        x.dim() == 3
        for x in [image, mask] + ([pred_image] if pred_image is not None else [])
    ):
        raise ValueError("All input tensors must have 3 dimensions (C, H, W)")

    ncols = 3 if pred_image is not None else 2
    f, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(5 * ncols, 5))

    # plot input image and ground truth
    axes[0].set_title("Image")
    axes[0].imshow(image.permute(1, 2, 0).squeeze(0), cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].imshow(mask.permute(1, 2, 0).squeeze(0), cmap="gray")

    # plot prediction if provided
    if pred_image is not None:
        axes[2].set_title("Prediction")
        axes[2].imshow(pred_image.permute(1, 2, 0).squeeze(0), cmap="gray")

    plt.show()


def show_original_and_mask(df: pd.DataFrame, idx: int = 0) -> None:

    row = df.iloc[idx]

    image_path = row[
        "images"
    ]  # 'Human-Segmentation-Dataset-master/Training_Images/1.jpg'
    mask_path = row["masks"]  # 'Human-Segmentation-Dataset-master/Ground_Truth/1.png'

    image = cv2.imread(image_path)  # <class 'numpy.ndarray'> (183, 276, 3)
    print(
        f"Before COLOR_BGR2RGB\nImage type: {type(image)} | Image shape: {image.shape}"
    )
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # <class 'numpy.ndarray'> (183, 276, 3)
    print(
        f"After COLOR_BGR2RGB\nImage type: {type(image)} | Image shape: {image.shape}\n"
    )

    mask = cv2.imread(mask_path)  # <class 'numpy.ndarray'> (183, 276, 3)
    print(
        f"Before IMREAD_UNCHANGED\nMask type: {type(mask)} | Mask shape: {mask.shape}"
    )
    mask = (
        cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) / 255.0
    )  # <class 'numpy.ndarray'> (183, 276)
    print(f"After IMREAD_UNCHANGED\nMask type: {type(mask)} | Mask shape: {mask.shape}")

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax1.set_title("Image")
    ax1.imshow(image)
    ax2.set_title("Ground Truth")
    ax2.imshow(mask)

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_original_and_augmented(
    idx: int, df: pd.DataFrame, augmentation: A.Compose
) -> None:

    row = df.iloc[idx]

    image_path = row["images"]
    mask_path = row["masks"]

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) / 255.0

    transformed = augmentation(image=image, mask=mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]

    imgs_to_plot = [image, mask, transformed_image, transformed_mask]
    img_titles = ["image", "mask", "transformed_image", "transformed_mask"]

    for i, img in enumerate(imgs_to_plot):
        print(f"{img_titles[i]} shape: {img.shape}")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs_to_plot[i])
        ax.set_title(f"{(img_titles[i])}")
        ax.label_outer()

    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict, save_path: Path | str | None = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Validation")
    ax1.set_title("Loss History")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Validation")
    ax2.set_title("Accuracy History")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_validation_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    save_path: Path,
    num_samples: int = 5,
) -> None:
    """Plot original images, ground truth masks, and predicted masks in a grid.

    Args:
        images: Tensor of shape (B, C, H, W) containing the input images
        masks: Tensor of shape (B, 1, H, W) containing the ground truth masks
        predictions: Tensor of shape (B, 1, H, W) containing the predicted masks
        save_path: Path where to save the visualization
        num_samples: Number of samples to plot (default: 4)
    """
    # move tensors to cpu and convert to numpy
    images_np = images.cpu().detach().numpy()
    masks_np = masks.cpu().detach().numpy()
    predictions_np = predictions.cpu().detach().numpy()

    # takae onky the specified number of samples
    images_np = images_np[:num_samples]
    masks_np = masks_np[:num_samples]
    predictions_np = predictions_np[:num_samples]

    fig, axes = plt.subplots(nrows=num_samples, ncols=3, figsize=(15, 4 * num_samples))

    for idx in range(num_samples):
        img = images_np[idx].transpose(1, 2, 0)  # (CHW) to (HWC)
        if img.max() > 1:
            img = img / 255.0
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title("Original Image")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(masks_np[idx, 0], cmap="gray")  # take the first channel
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(predictions_np[idx, 0], cmap="gray")
        axes[idx, 2].set_title("Prediction")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_predictions(
    original_image: np.ndarray, pred_mask: np.ndarray, save_path: Path
) -> None:
    # TODO: finish this function
    """Plot and save the original image alongside its predicted segmentation mask.


    Args:
        original_image: Input image as a numpy array (H, W, C) dim() = 3
        pred_mask: Predicted segmentation mask as numpy array (H, W) dim() = 2
        save_path: Path where the visualization will be saved
    """

    pred_mask = np.expand_dims(pred_mask, axis=2)  # (H,W,1)

    # Create a figure with two subplots side by side
    plt.figure(figsize=(12, 6))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # Plot predicted mask
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    # Ensure the output directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
