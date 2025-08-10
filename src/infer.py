# src/infer.py
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from .config import BEST_WEIGHTS, CLASS_INDEX_PATH, IMG_SIZE, DEVICE, PRED_THRESHOLD
from .model import build_model

def load_classes():
    with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines()]

def get_tf():
    # Keep eval transform consistent with training eval pipeline
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def load_model():
    classes = load_classes()
    model = build_model(len(classes))
    device = torch.device(DEVICE if (DEVICE.lower()=="cuda" and torch.cuda.is_available()) else "cpu")
    model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=device))
    model.eval().to(device)
    return model, classes, device

def predict_image(path: str, threshold: float = None):
    """Return (final_label, confidence, probs_dict, extra_info)."""
    thr = PRED_THRESHOLD if threshold is None else threshold
    model, classes, device = load_model()
    tf = get_tf()

    # Respect EXIF orientation (common on phone photos)
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)

    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().squeeze().numpy()

    # Sort classes by probability
    order = probs.argsort()[::-1]
    top1_idx, top2_idx = order[0], order[1]
    top1_p, top2_p = float(probs[top1_idx]), float(probs[top2_idx])
    top1, top2 = classes[top1_idx], classes[top2_idx]

    # Uncertainty rule: below threshold -> "uncertain/healthy"
    uncertain = top1_p < thr
    # Tie-break rule: if very close, flag as borderline (useful for UI messaging)
    borderline = (top1_p - top2_p) < 0.05

    final_label = "uncertain/healthy" if uncertain else top1
    final_conf  = top1_p if not uncertain else top1_p  # we still show top1 prob

    probs_dict = {classes[i]: float(probs[i]) for i in order}  # sorted by prob (desc)
    extra = {"top1": (top1, top1_p), "top2": (top2, top2_p), "borderline": borderline, "threshold": thr}
    return final_label, final_conf, probs_dict, extra
