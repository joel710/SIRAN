import torch
from PIL import Image

from core.transforms import get_train_transforms, get_eval_transforms


def test_train_transform_output():
    img = Image.new("RGB", (640, 480), color=(128, 64, 32))
    transform = get_train_transforms()
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_eval_transform_output():
    img = Image.new("RGB", (320, 240), color=(200, 100, 50))
    transform = get_eval_transforms()
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
