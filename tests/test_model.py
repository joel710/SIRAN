import torch
from core.model import build_siran_model


def test_model_output_shape():
    model = build_siran_model(pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1)


def test_model_output_range():
    model = build_siran_model(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    prob = torch.sigmoid(out)
    assert 0.0 <= prob.item() <= 1.0
