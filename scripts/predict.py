"""Prediction script for the human image segmentation model.

This module implements the prediction pipeline for a deep learning-based image segmentation
model designed to segment human figures from images.

Example:
    To run prediction on a single image:
        $ python predict.py --config configs/predict_config.yaml --input path/to/image.jpg

    To specify a different device:
        $ python predict.py --config configs/predict_config.yaml --input path/to/image.jpg --device cuda
"""

import argparse
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path

from src.models.model import SegmentationModel
from src.utils.device import get_device
from src.utils.visualization import plot_predictions

from icecream import ic


class Predictor:
    """Handles the prediction process for the segmentation model.

    This class implements the prediction pipeline including model loading
    and result visualization for single images.

    Args:
        config: Dictionary containing all configuration parameters for prediction
        device_name: Optional string specifying the device to use (cpu, cuda, or mps)
    """

    def __init__(self, config, device_name: str | None = None) -> None:
        self.config = config
        self.device = get_device(device_name)
        print(f"Using device: {self.device}")

        self.model = SegmentationModel(config).to(self.device)

    def load_checkpoint(self) -> None:
        checkpoint_path = (
            Path(self.config["training"]["checkpoint"]["save_dir"])
            / self.config["training"]["checkpoint"]["save_name"]
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def _preprocess_image(self, original_img_np: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input.

        Args:
            original_img_np (np.ndarray): Input image as numpy array (H, W, C)

        Returns:
            torch.Tensor: Preprocessed image tensor (1, C, H, W)

        Raises:
            ValueError: If input image is None or empty
        """
        if original_img_np is None or original_img_np.size == 0:
            raise ValueError("Input image is empty or None")
        # Ensure image is RGB
        if len(original_img_np.shape) != 3:
            raise ValueError("Input image must be a 3-channel RGB image")

        image_np = cv2.resize(
            original_img_np,
            (self.config["data"]["image_size"], self.config["data"]["image_size"]),
            interpolation=cv2.INTER_NEAREST,
        )
        image_tensor: torch.Tensor = (torch.from_numpy(image_np).permute(2, 0, 1)).type(
            torch.float32
        )
        image_tensor = image_tensor / 255.0
        return image_tensor.unsqueeze(dim=0)

    def predict(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Make prediction on a single image.

        Args:
            image_path: Path to the input image file

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing (original_image, prediction_mask)

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self.model.eval()
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        input_tensor = self._preprocess_image(original_image)

        with torch.inference_mode():
            logits = self.model(input_tensor.to(self.device))
        pred_mask = torch.sigmoid(logits) > 0.5
        pred_mask = (pred_mask > 0.5).float()

        # Convert to numpy and resize back to original dimensions
        pred_mask_np = pred_mask.squeeze(dim=0).cpu().detach().numpy()
        pred_mask_np = (pred_mask_np * 255).astype(np.uint8)  # (1, 320, 320)
        pred_mask_np = pred_mask_np.transpose(1, 2, 0)  # (320, 320, 1)
        pred_mask_np = cv2.resize(
            pred_mask_np,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        return original_image, pred_mask_np

    def save_predictions(
        self, original_image: np.ndarray, pred_mask: np.ndarray
    ) -> None:
        """Save prediction visualization to file.

        Args:
            original_image: Input image as numpy array
            pred_mask: Prediction mask as numpy array
        """
        save_dir = Path(self.config["prediction"]["output_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "predictions.png"
        plot_predictions(original_image, pred_mask, save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("--config", type=str, default="configs/test_config.yaml")
    parser.add_argument("--input", type=str, default="data/images/image.jpg")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for prediction",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    predictor = Predictor(config, device_name=args.device)
    predictor.load_checkpoint()
    original_image, pred_mask = predictor.predict(args.input)
    predictor.save_predictions(original_image, pred_mask)


if __name__ == "__main__":
    main()
