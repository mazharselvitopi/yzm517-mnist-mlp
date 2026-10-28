from pathlib import Path
import importlib
import json
import pytest
import torch
from core import save_checkpoint
from mlp import MLP

ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_requires_final_selection_and_keeps_confusion_counts(tmp_path, monkeypatch):
    assert (ROOT / 'evaluate.py').exists(), 'Ayrı final değerlendirme CLI henüz yok'
    evaluate = importlib.import_module('evaluate')
    calls = []
    class TestDataset:
        def __init__(self, root, train, download, transform):
            calls.append(train)
            assert train is False
        def __len__(self):
            return 3
        def __getitem__(self, index):
            return torch.zeros(1, 28, 28), index
    monkeypatch.setattr(evaluate.datasets, 'MNIST', TestDataset)
    checkpoint = tmp_path / 'best.pt'
    model = MLP([4, 2])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    save_checkpoint(checkpoint, model, {'hidden_sizes': [4, 2], 'seed': 1}, [0], [1], 1, 1.0)
    output = tmp_path / 'eval'
    args = ['--checkpoint', str(checkpoint), '--data-dir', str(Path.home() / 'datasets/mnist'), '--output', str(output)]
    with pytest.raises(SystemExit):
        evaluate.main(args)
    assert calls == []
    evaluate.main(args + ['--final-model-selected'])
    assert calls == [False]
    result = json.loads((output / 'test_metrics.json').read_text())
    assert result['accuracy'] == pytest.approx(1 / 3)
    assert result['samples'] == 3
    assert len(result['checkpoint_sha256']) == 64
    matrix = json.loads((output / 'confusion_matrix.json').read_text())
    assert len(matrix) == 10 and all(len(row) == 10 for row in matrix)
    assert sum(map(sum, matrix)) == 3
    assert matrix[0][0] == matrix[1][0] == matrix[2][0] == 1
    assert (output / 'confusion_matrix.png').exists()
    with pytest.raises(FileExistsError):
        evaluate.main(args + ['--final-model-selected'])
