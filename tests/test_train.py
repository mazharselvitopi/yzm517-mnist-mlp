"""Küçük sentetik fixture; MNIST başarı ölçümü DEĞİLDİR."""
from pathlib import Path
import json
import importlib
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_training_cli_keeps_test_closed_and_writes_artifacts(tmp_path, monkeypatch):
    assert (ROOT / 'train.py').exists(), 'Eğitim CLI henüz yok'
    train = importlib.import_module('train')
    calls = []
    class SmallDataset:
        def __init__(self, root, train, download, transform):
            calls.append(train)
            assert train is True, 'Eğitim sırasında test kümesine erişildi!'
        def __len__(self):
            return 30
        def __getitem__(self, index):
            return torch.full((1, 28, 28), index / 30), index % 10
    monkeypatch.setattr(train.datasets, 'MNIST', SmallDataset)
    output = tmp_path / 'run'
    args = ['--data-dir', str(Path.home() / 'datasets/mnist'), '--output', str(output),
            '--epochs', '2', '--val-size', '6', '--limit-train', '12', '--limit-val', '4',
            '--batch-size', '5', '--hidden-sizes', '8', '4']
    train.main(args)
    assert calls == [True]
    assert (output / 'best.pt').exists()
    history = json.loads((output / 'history.json').read_text())
    assert len(history) == 2
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['best_val_loss'] == min(row['val_loss'] for row in history)
    assert summary['test_accessed'] is False
    from core import load_checkpoint
    restored, checkpoint = load_checkpoint(output / 'best.pt')
    assert checkpoint['val_loss'] == summary['best_val_loss']
    assert checkpoint['epoch'] == summary['best_epoch']
    assert next(restored.parameters()).device.type == 'cpu'
    split = json.loads((output / 'split.json').read_text())
    assert not set(split['train_indices']) & set(split['val_indices'])
    for file in ('loss.png', 'accuracy.png'):
        with Image.open(output / file) as image:
            assert image.format == 'PNG' and image.width > 100
    assert (output / 'history.csv').read_text().count('\n') == 3
    with pytest.raises(FileExistsError):
        train.main(args)
    with pytest.raises(SystemExit):
        train.main(['--epochs', '0', '--data-dir', str(Path.home() / 'datasets/mnist'), '--output', str(tmp_path / 'invalid')])
