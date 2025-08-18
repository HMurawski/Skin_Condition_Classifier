from functools import lru_cache

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from .config import BEST_WEIGHTS, CLASS_INDEX_PATH, DEVICE, IMG_SIZE, PRED_THRESHOLD
from .model import build_model

# ---- I/O helpers ----------


def load_classes():
    """Read class names (one per line) from artifacts/classes.txt."""
    if not CLASS_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing {CLASS_INDEX_PATH}. Train first.")
    with open(CLASS_INDEX_PATH, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# Cache the eval transform (building it every call is wasteful)
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_tf():
    """Return the cached eval transform."""
    return _TRANSFORM


# ---- Model loading (cached) -------------------------------------------------


@lru_cache(maxsize=1)
def _load_model_cached():
    """Load model once per process; cache across calls."""
    classes = load_classes()
    model = build_model(len(classes))
    device = torch.device(DEVICE if (DEVICE.lower() == "cuda" and torch.cuda.is_available()) else "cpu")
    state = torch.load(BEST_WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    # Return classes as tuple for lru_cache friendliness
    return model, tuple(classes), device


def load_model():
    """Public wrapper that returns the cached (model, classes, device)."""
    return _load_model_cached()


# ---- Prediction API ---------------------------------------------------------


def predict_image(path: str, threshold: float = None):
    """Predict from a file path."""
    img = Image.open(path).convert("RGB")
    return predict_pil(img, threshold)


def predict_pil(img: Image.Image, threshold: float = None):
    """
    Predict directly from a PIL Image without touching disk.
    Returns (final_label, final_conf, probs_dict_sorted, extra_info).
    """
    thr = PRED_THRESHOLD if threshold is None else threshold
    model, classes, device = load_model()
    tf = get_tf()

    # Honor EXIF orientation (common for phone photos)
    img = ImageOps.exif_transpose(img)

    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().squeeze().numpy()

    # Sort probabilities (desc)
    order = probs.argsort()[::-1]
    top1_idx, top2_idx = order[0], order[1]
    top1_p, top2_p = float(probs[top1_idx]), float(probs[top2_idx])
    top1, top2 = classes[top1_idx], classes[top2_idx]

    uncertain = top1_p < thr
    borderline = (top1_p - top2_p) < 0.05

    final_label = "uncertain/healthy" if uncertain else top1
    final_conf = top1_p

    # Build a dict sorted by prob
    probs_dict = {classes[i]: float(probs[i]) for i in order}

    extra = {
        "top1": (top1, top1_p),
        "top2": (top2, top2_p),
        "borderline": borderline,
        "threshold": thr,
    }
    return final_label, final_conf, probs_dict, extra
