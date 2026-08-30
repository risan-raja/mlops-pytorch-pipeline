import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image
from torchvision import transforms

from model import get_model

MODEL_PATH_ENV = os.environ.get("MODEL_PATH", "/app/checkpoints/classifier_v1.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ]
)

model: torch.nn.Module | None = None


def load_model_from_checkpoint(path_str: str) -> torch.nn.Module:
    path = Path(path_str)
    if not path.exists():
        # Fallback local paths check
        local_fallback = Path("checkpoints/classifier_v1.pt")
        if local_fallback.exists():
            path = local_fallback
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {path_str}")

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    architecture = checkpoint.get("architecture", "resnet18")
    num_classes = checkpoint.get("num_classes", 10)

    loaded_model = get_model(architecture=architecture, num_classes=num_classes)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(DEVICE)
    loaded_model.eval()
    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager handling application startup and shutdown."""
    global model
    try:
        model = load_model_from_checkpoint(MODEL_PATH_ENV)
        print(f"Successfully loaded model from {MODEL_PATH_ENV}")
    except Exception as e:  # noqa: BLE001
        print(
            f"Warning: Model not loaded on startup ({e}). Ready to load dynamically if available."
        )
    yield
    model = None


app = FastAPI(
    title="MLOps CIFAR-10 Model Serving API",
    description="Inference API exposing prediction and health check endpoints.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health() -> dict[str, str]:
    global model
    if model is None:
        try:
            model = load_model_from_checkpoint(MODEL_PATH_ENV)
        except Exception:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded or checkpoint not available",
            ) from None
    return {"status": "healthy", "model_loaded": "true"}


@app.post("/predict")
async def predict(image: Annotated[UploadFile, File(...)]) -> dict:
    global model
    if model is None:
        try:
            model = load_model_from_checkpoint(MODEL_PATH_ENV)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not loaded: {e!s}",
            ) from e

    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {e!s}",
        ) from e

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0).tolist()

    class_probs = {
        CIFAR10_CLASSES[i] if i < len(CIFAR10_CLASSES) else f"class_{i}": round(prob, 4)
        for i, prob in enumerate(probabilities)
    }
    predicted_idx = int(torch.argmax(outputs, dim=1).item())
    predicted_class = (
        CIFAR10_CLASSES[predicted_idx]
        if predicted_idx < len(CIFAR10_CLASSES)
        else f"class_{predicted_idx}"
    )

    return {
        "prediction": predicted_class,
        "class_id": predicted_idx,
        "probabilities": class_probs,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
