import pytest
import numpy as np
from PIL import Image
from src.infer import predict_pil
from src.config import BEST_WEIGHTS, CLASS_INDEX_PATH

@pytest.mark.skipif(
    not BEST_WEIGHTS.exists() or not CLASS_INDEX_PATH.exists(),
    reason="requires trained artifacts (best_resnet18.pt, classes.txt)"
)
def test_uncertain_logic_with_artifacts():
    # Black image -> high threshold should force 'uncertain'
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    label, conf, probs, extra = predict_pil(img, threshold=0.99)
    assert label == "uncertain/healthy"

