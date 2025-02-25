"""Training script for the human image segmentation model.

This module implements the training pipeline for a deep learning-based image segmentation
model designed to segment human figures from images. It includes a Trainer class that
handles the training loop, validation, checkpointing, and logging functionality.

Example:
    To train the model using default configuration:
        $ python train.py --config configs/train_config.yaml

    To specify a different device:
        $ python train.py --config configs/train_config.yaml --device cuda
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
from src.utils.visualization import plot_training_history


class Trainer:
    """Handles the training process for the segmentation model.

    This class implements the complete training pipeline including model initialization,
    training loop, validation, learning rate scheduling, early stopping, and checkpointing.

    Args:
        config: Dictionary containing all configuration parameters for training
        device_name: Optional string specifying the device to use (cpu, cuda, or mps)

    Attributes:
        model: The segmentation model instance
        optimizer: The optimizer for training
        lr_scheduler: Learning rate scheduler
        history: Dictionary tracking training and validation metrics
        device: The device being used for training
    """

    def __init__(self, config, device_name: str | None = None) -> None:
        self.config = config
        self.device = get_device(device_name)
        print(f"Using device: {self.device}")
        self.setup_seeds()

        self.model = SegmentationModel(config).to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        self.train_dataloader, self.val_dataloader = get_dataloaders(config)

        self.history: dict[str, list[float | np.ndarray]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=self.config["training"].get("lr_patience", 5),
        )

        self.grad_clip = self.config["training"].get("grad_clip", 1.0)

    def setup_seeds(self) -> None:
        torch.manual_seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config["random_seed"])
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.config["random_seed"])

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss: float = 0.0

        for batch in tqdm(self.train_dataloader, desc="Training"):
            batch_images, batch_masks = batch
            batch_images = batch_images.to(self.device)
            batch_masks = batch_masks.to(self.device)

            self.optimizer.zero_grad()
            logits, loss = self.model(batch_images, batch_masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_dataloader)

    def validate(self) -> float:
        self.model.eval()
        total_loss: float = 0.0

        with torch.inference_mode():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                batch_images, batch_masks = batch
                batch_images = batch_images.to(self.device)
                batch_masks = batch_masks.to(self.device)

                logits, loss = self.model(batch_images, batch_masks)
                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def train(self) -> None:

        best_val_loss = float("inf")
        patience_counter = 0
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Add progress bar for epochs
        epochs_pbar = tqdm(
            range(self.config["training"]["epochs"]), desc="Training Progress"
        )

        for epoch in epochs_pbar:
            print(f"\nEpoch {epoch + 1:03d}/{self.config['training']['epochs']:03d}")

            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            # Update learning rate scheduler and check for changes
            old_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]["lr"]

            if new_lr != old_lr:
                print(f"\nLearning rate decreased from {old_lr:.6f} to {new_lr:.6f}")

            # update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1

            # early stopping
            if patience_counter >= self.config["training"]["early_stopping_patience"]:
                print(f"Early stopping triggered at epoch {epoch + 1} epochs")
                break

            # Update progress bar description
            epochs_pbar.set_description(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
            )

        plot_training_history(
            self.history,
            save_path=Path(self.config["logging"]["log_dir"]) / "training_history.png",
        )

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }

        save_path = (
            Path(self.config["training"]["checkpoint"]["save_dir"])
            / self.config["training"]["checkpoint"]["save_name"]
        )

        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
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

    trainer = Trainer(config, device_name=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
