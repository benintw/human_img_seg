"""Validation script for the human image segmentation model.

This module implements the validation pipeline for a deep learning-based image segmentation
model designed to segment human figures from images. It includes a Validator class that
handles model evaluation, metric computation, and result logging.

Example:
    To validate the model using default configuration:
        $ python validate.py --config configs/validate_config.yaml

    To specify a different device:
        $ python validate.py --config configs/validate_config.yaml --device cuda
"""

import argparse
import yaml
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.data.dataset import get_dataloaders
from src.models.model import SegmentationModel
from src.utils.device import get_device
from src.utils.visualization import plot_validation_predictions


class Validator:
    """Handles the validation process for the segmentation model.

    This class implements the validation pipeline including model loading,
    metric computation, and result visualization.

    Args:
        config: Dictionary containing all configuration parameters for validation
        device_name: Optional string specifying the device to use (cpu, cuda, or mps)
    """

    def __init__(self, config, device_name: str | None = None) -> None:
        self.config = config
        self.device = get_device(device_name)
        print(f"Using device: {self.device}")
        self.setup_seeds()

        self.model = SegmentationModel(config).to(self.device)
        self.train_dataloader, self.val_dataloader = get_dataloaders(config)
        self.metrics = {"val_loss": []}

    def setup_seeds(self) -> None:
        torch.manual_seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config["random_seed"])
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.config["random_seed"])

    def load_checkpoint(self) -> None:
        checkpoint_path = (
            Path(self.config["training"]["checkpoint"]["save_dir"])
            / self.config["training"]["checkpoint"]["save_name"]
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.metrics["val_loss"].append(checkpoint["val_loss"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def validate(self) -> dict:
        self.model.eval()
        total_loss: float = 0.0

        num_batches = len(self.val_dataloader)

        with torch.inference_mode():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                batch_images, batch_masks = batch
                batch_images = batch_images.to(self.device)
                batch_masks = batch_masks.to(self.device)

                logits, loss = self.model(batch_images, batch_masks)
                total_loss += loss.item()

            self.save_predictions(batch_images, batch_masks, logits)

        return {"val_loss": total_loss / num_batches}

    def save_predictions(
        self, images: torch.Tensor, masks: torch.Tensor, logits: torch.Tensor
    ):
        predictions = torch.sigmoid(logits) > 0.5
        save_path = Path(self.config["logging"]["log_dir"]) / "validation_samples.png"
        plot_validation_predictions(images, masks, predictions, save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a segmentation model")
    parser.add_argument("--config", type=str, default="configs/val_config.yaml")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Device to use (cpu, cuda, mps)",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    validator = Validator(config, device_name=args.device)
    validator.load_checkpoint()
    results = validator.validate()

    print("\nValidation Results:")
    print(f"Loss: {results['val_loss']:.4f}")


if __name__ == "__main__":
    main()
