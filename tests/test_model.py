import numpy as np
import pytest

from kneemil.constants import N_SLICES
from kneemil.volume import window_centers

torch = pytest.importorskip("torch")
pytest.importorskip("timm")


def test_model_forward_masking_and_attention():
    from kneemil.model import KneeMILModel, gpu_augment, predict_volume, to_input

    model = KneeMILModel(pretrained=False)
    x = torch.randint(0, 255, (2, 6, 3, 64, 64), dtype=torch.uint8)
    xi = gpu_augment(to_input(x, torch.device("cpu")))
    pos = torch.randint(1, 63, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 4:] = False
    logits, att = model(xi, pos, mask, return_attention=True)
    assert logits.shape == (2, 12) and att.shape == (2, 6, 12)
    assert torch.allclose(att.sum(1), torch.ones(2, 12), atol=1e-5)  # softmax over windows
    assert att[1, 4:].abs().sum() == 0  # masked windows get no attention
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.rand(2, 12)).backward()

    model.eval()
    vol = np.random.randint(0, 255, (N_SLICES, 64, 64), dtype=np.uint8)
    m = np.ones(N_SLICES, bool)
    p = predict_volume(model, vol, m, torch.device("cpu"), window_centers)
    assert p.shape == (12,) and np.all((p > 0) & (p < 1))
    assert predict_volume(model, vol, np.zeros(N_SLICES, bool), torch.device("cpu"), window_centers) is None
