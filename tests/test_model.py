import importlib.util
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

def test_model_shape_and_mlp_contract():
    assert (ROOT / 'mlp.py').exists(), 'mlp.py dosyası bulunamadı.'
    from mlp import MLP
    model = MLP([64, 32])
    assert model(torch.zeros(3, 1, 28, 28)).shape == (3, 10)
    assert sum(isinstance(m, torch.nn.Linear) for m in model.modules()) == 3
    assert sum(isinstance(m, torch.nn.ReLU) for m in model.modules()) == 2
    assert not any(isinstance(m, torch.nn.modules.conv._ConvNd) for m in model.modules())
    for sizes in ([], [4], [4, 0], [-1, 4]):
        with pytest.raises(ValueError):
            MLP(sizes)
