from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from pathlib import Path
import yaml
import torch
import uuid
import os

from src.models.model import SegmentationModel
from src.utils.device import get_device
from src.utils.visualization import plot_predictions

app = FastAPI(title="Human Image Segmentation")

# mount statis files
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# Initialize model
with open("configs/test_config.yaml") as f:
    config = yaml.safe_load(f)

device = get_device("cpu")
model = SegmentationModel(config).to(device)

checkpoint_path = (
    Path(config["training"]["checkpoint"]["save_dir"])
    / config["training"]["checkpoint"]["save_name"]
)

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# create output directory if it doesnt exist
UPLOAD_DIR = Path("web/static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict(file: UploadFile = File(...)) -> dict[str, str]:
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"input_{file_id}.jpg"
    output_path = UPLOAD_DIR / f"output_{file_id}.png"

    # save uploaded file
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    # read and process image
    original_image = cv2.imread(str(input_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # pre-process and predict
    input_tensor = preprocess_image(original_image)

    with torch.inference_mode():
        logits = model(input_tensor.to(device))
        pred_mask = (torch.sigmoid(logits) > 0.5).float()

    # convert prediction to numpy
    pred_mask_np = (pred_mask.squeeze(dim=0).cpu().numpy() * 255).astype(np.uint8)
    pred_mask_np = pred_mask_np.transpose(1, 2, 0)
    pred_mask_np = cv2.resize(
        pred_mask_np,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # save result
    plot_predictions(original_image, pred_mask_np, output_path)

    # return results
    return {
        "input_image": f"/static/uploads/input_{file_id}.jpg",
        "output_image": f"/static/uploads/output_{file_id}.png",
    }


def preprocess_image(original_image_np: np.ndarray) -> torch.Tensor:
    image_np = cv2.resize(
        original_image_np,
        (config["data"]["image_size"], config["data"]["image_size"]),
        interpolation=cv2.INTER_NEAREST,
    )
    image_tensor: torch.Tensor = (torch.from_numpy(image_np).permute(2, 0, 1)).type(
        torch.float32
    )
    image_tensor = image_tensor / 255.0
    return image_tensor.unsqueeze(dim=0)
