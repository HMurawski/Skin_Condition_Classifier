import torch
from src.model import build_model
from src.config import IMG_SIZE

def test_forward_shape():
    m = build_model(num_classes=3).eval()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    y = m(x)
    assert y.shape == (2, 3)