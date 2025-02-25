import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

import yaml
from icecream import ic


class SegmentationModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super(SegmentationModel, self).__init__()

        # Validate config
        required_keys = [
            "encoder",
            "weights",
            "activation",
            "in_channels",
            "dice_mode",
            "dice_weight",
            "bce_weight",
            "dice_mode",
        ]
        if not all(key in config["model"] for key in required_keys):
            raise ValueError(f"Config must contain all required keys: {required_keys}")

        self.loss_weights = {
            "dice": config["model"].get("dice_weight", 1.0),
            "bce": config["model"].get("bce_weight", 1.0),
        }

        self.architecture = smp.Unet(
            encoder_name=config["model"]["encoder"],
            encoder_weights=config["model"]["weights"],
            in_channels=config["model"]["in_channels"],
            classes=1,
            activation=config["model"]["activation"],  # outputs raw logits
        )

        self.dice_loss = DiceLoss(mode=config["model"]["dice_mode"])
        self.bce_loss = nn.BCEWithLogitsLoss()

    def compute_loss(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # Compute weighted combination of losses.
        dice_loss = self.dice_loss(logits, masks) * self.loss_weights["dice"]
        bce_loss = self.bce_loss(logits, masks) * self.loss_weights["bce"]

        return dice_loss + bce_loss

    def forward(
        self, images: torch.Tensor, masks: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        logits = self.architecture(images)

        if masks is not None:
            combined_loss = self.compute_loss(logits, masks)
            return logits, combined_loss

        return logits


def main() -> None:

    with open("./configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = SegmentationModel(config)

    # test forward pass
    images = torch.randn(1, 3, 224, 224)
    masks = torch.randn(1, 1, 224, 224)
    logits, loss = model(images, masks)
    ic(logits.shape, loss)


if __name__ == "__main__":
    main()
