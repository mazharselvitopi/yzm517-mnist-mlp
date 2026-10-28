from pathlib import Path
import importlib
import json
import pytest
import torch
from PIL import Image, ImageDraw, ImageOps
from core import MEAN, STD, save_checkpoint
from mlp import MLP

ROOT = Path(__file__).resolve().parents[1]


def test_single_digit_preprocessing_and_prediction(tmp_path):
    assert (ROOT / 'predict.py').exists(), 'predict.py dosyası bulunamadı.'
    predict = importlib.import_module('predict')
    # Sentetik çubuk yalnız piksel sözleşmesini sınar; gerçek el yazısı örneği değildir.
    dark = Image.new('L', (60, 90), 255)
    ImageDraw.Draw(dark).rectangle((9, 20, 16, 74), fill=0)
    dark.save(tmp_path / 'dark.png')
    ImageOps.invert(dark).save(tmp_path / 'light.png')
    a, prepared = predict.preprocess_image(tmp_path / 'dark.png')
    b, inverse_prepared = predict.preprocess_image(tmp_path / 'light.png')
    assert a.shape == (1, 1, 28, 28)
    assert torch.equal(a, b)
    assert prepared.size == (28, 28) and prepared.mode == 'L'
    assert float(a.min()) == pytest.approx((0 - MEAN) / STD)
    assert float(a.max()) <= (1 - MEAN) / STD + 1e-5
    raw = torch.tensor(list(prepared.get_flattened_data()), dtype=torch.float32).reshape(28, 28)
    center_y = (raw.sum(1) * torch.arange(28)).sum() / raw.sum()
    center_x = (raw.sum(0) * torch.arange(28)).sum() / raw.sum()
    assert abs(center_y.item() - 13.5) <= 0.6
    assert abs(center_x.item() - 13.5) <= 0.6
    for color in (0, 255):
        path = tmp_path / f'blank{color}.png'
        Image.new('L', (32, 32), color).save(path)
        with pytest.raises(ValueError, match='[Bb]oş|[Kk]ontrast'):
            predict.preprocess_image(path)
    transparent = tmp_path / 'transparent.png'
    Image.new('RGBA', (32, 32), (0, 0, 0, 0)).save(transparent)
    with pytest.raises(ValueError):
        predict.preprocess_image(transparent)
    invalid = tmp_path / 'input.txt'
    invalid.write_text('not an image')
    with pytest.raises(ValueError):
        predict.preprocess_image(invalid)
    corrupt = tmp_path / 'corrupt.png'
    corrupt.write_bytes(b'not a png')
    with pytest.raises(ValueError):
        predict.preprocess_image(corrupt)
    checkpoint = tmp_path / 'best.pt'
    save_checkpoint(checkpoint, MLP([4, 2]), {'hidden_sizes': [4, 2], 'seed': 1}, [0], [1], 1, 1.0)
    output = tmp_path / 'prediction'
    args = ['--checkpoint', str(checkpoint), '--image', str(tmp_path / 'dark.png'), '--output', str(output)]
    predict.main(args)
    result = json.loads((output / 'predictions.json').read_text())
    assert len(result) == 1 and 0 <= result[0]['prediction'] <= 9
    assert sum(result[0]['probabilities']) == pytest.approx(1)
    assert (output / '000_prepared.png').exists()
    with pytest.raises(FileExistsError):
        predict.main(args)
