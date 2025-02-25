#!/bin/bash

# default values
CONFIG_PATH="configs/train_config.yaml"
DEVICE="mps"


# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config) CONFIG_PATH="$2"; shift ;;
        -d|--device) DEVICE="$2"; shift ;;

        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done


# Create necessary directories
mkdir -p checkpoints
mkdir -p logs

# Set up timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"

# Print training configuration
echo "Starting training with:"
echo "  Config: $CONFIG_PATH"
echo "  Device: $DEVICE"
echo

# Run the training script
python scripts/train.py \
    --config "$CONFIG_PATH" \
    --device "$DEVICE" \

echo "Training completed!"
echo "Check the logs and results in the logs directory."