# Dataset for Image Segmentation

This repo contains csv file, Training Images, Ground Truth Images (Mask) and helper.py (for visualization)

Dataset Credit : https://github.com/VikramShenoy97/Human-Segmentation-Dataset

# Human Image Segmentation

A web application for segmenting humans from images using deep learning.

## Features

- FastAPI web interface
- PyTorch-based segmentation model
- Docker deployment support

## Pull Docker Image

docker pull benchen666/human-img-seg:latest

## Run Docker Container

docker run -d -p 8000:8000 benchen666/human-img-seg:latest

## Stop Docker Container

docker stop human-img-seg

## Model Weights

The model weights are not included in this repository due to size constraints. You can:

use Docker: `docker pull benchen666/human-img-seg:latest`
