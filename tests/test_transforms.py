import numpy as np
from PIL import Image
from src.data import get_transforms
from src.config import IMG_SIZE

def test_eval_transform_shape_and_scale():
    _, eval_tf = get_transforms()
    img = Image.fromarray((np.random.rand(100, 120, 3) * 255).astype("uint8"))
    x = eval_tf(img)
    assert x.shape == (3, IMG_SIZE, IMG_SIZE)
    # after normalization the values should be around [-3, 3]
    assert float(x.min()) > -5 and float(x.max()) < 5
