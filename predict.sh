#!/bin/bash

# Default values
CONFIG="configs/test_config.yaml"
DEVICE="mps"
INPUT_IMAGE="predict_tony.jpg"
OUTPUT_DIR="predictions"

mkdir -p "$OUTPUT_DIR"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --input)
            INPUT_IMAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Check if input image is provided
if [ -z "$INPUT_IMAGE" ]; then
    echo "Error: Input image path is required"
    echo "Usage: ./predict.sh --input path/to/image.jpg [--config path/to/config.yaml] [--device cpu|cuda|mps]"
    exit 1
fi

# Run prediction
python scripts/predict.py \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --input "$INPUT_IMAGE"

echo "Prediction complete. Results saved to $OUTPUT_DIR"