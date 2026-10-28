from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from mlp import MLP

ROOT = Path(__file__).resolve().parents[1]

def test_training_primitives(tmp_path):
    assert (ROOT / 'core.py').exists(), 'core.py dosyası bulunamadı.'
    from core import make_split, run_epoch, save_checkpoint, load_checkpoint, validate_data_dir
    train, val = make_split(100, 20, 7, 30, 10)
    assert (train, val) == make_split(100, 20, 7, 30, 10)
    assert not set(train) & set(val)
    assert len(train) == 30 and len(val) == 10
    for args in [(10, 0, 7, None, None), (10, 10, 7, None, None), (10, 2, 7, 0, None), (10, 2, 7, 9, None), (10, 2, 7, None, 3)]:
        with pytest.raises(ValueError):
            make_split(*args)
    # Son minibatch tek örnek: batch ortalamasının ortalaması yanlış olur.
    logits = torch.tensor([[4., 0.], [4., 0.], [4., 0.]])
    labels = torch.tensor([0, 0, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    metrics = run_epoch(torch.nn.Identity(), loader)
    assert metrics['loss'] == pytest.approx(torch.nn.functional.cross_entropy(logits, labels).item())
    assert metrics['accuracy'] == pytest.approx(2 / 3)
    assert metrics['samples'] == 3
    with pytest.raises(ValueError):
        run_epoch(torch.nn.Identity(), [])
    model = MLP([8, 4])
    before = model(torch.ones(2, 1, 28, 28)).detach().clone()
    destination = tmp_path / 'best.pt'
    save_checkpoint(destination, model, {'hidden_sizes': [8, 4], 'seed': 7}, train, val, 1, 2.0)
    with torch.no_grad():
        next(model.parameters()).add_(100)
    restored, checkpoint = load_checkpoint(destination)
    assert torch.equal(restored(torch.ones(2, 1, 28, 28)), before)
    assert checkpoint['train_indices'] == train
    assert checkpoint['val_indices'] == val
    assert checkpoint['versions']['python']
    with pytest.raises(ValueError):
        validate_data_dir(ROOT / 'data')
    assert validate_data_dir(Path.home() / 'datasets/mnist').is_absolute()
