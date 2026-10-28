"""Ortak CPU eğitim, bölme ve güvenli checkpoint yardımcıları."""
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from mlp import MLP

MEAN, STD = 0.1307, 0.3081


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def versions():
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'torchvision': str(torchvision.__version__), 'numpy': str(np.__version__),
            'platform': platform.platform(), 'machine': platform.machine(),
            'device': 'cpu', 'threads': torch.get_num_threads()}


def validate_data_dir(path):
    path = Path(path).expanduser().resolve()
    # Verinin bu deponun veya bir üst Git deposunun içine yazılmasını önle.
    for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (parent / '.git').exists() and path.is_relative_to(parent):
            raise ValueError('--data-dir Git deposunun dışında olmalı (örn. ~/datasets/mnist).')
    if path.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError('--data-dir proje dışında olmalı.')
    return path


def make_split(total, val_size, seed, limit_train=None, limit_val=None):
    sizes = [total, val_size, *(n for n in (limit_train, limit_val) if n is not None)]
    if any(type(n) is not int for n in sizes):
        raise ValueError('Veri ve altküme boyutları tamsayı olmalı; bool/ondalık kabul edilmez.')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed 0..2**32-1 arasında tamsayı olmalı.')
    if not 0 < val_size < total:
        raise ValueError('0 < val_size < veri sayısı gerekir.')
    if limit_train is not None and not 0 < limit_train <= total - val_size:
        raise ValueError('limit_train eğitim bölümü boyutunu aşmayan pozitif sayı olmalı.')
    if limit_val is not None and not 0 < limit_val <= val_size:
        raise ValueError('limit_val doğrulama boyutunu aşmayan pozitif sayı olmalı.')
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    val, train = order[:val_size], order[val_size:]
    return train[:limit_train], val[:limit_val]


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to('cpu'), labels.to('cpu')
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            n = labels.size(0)
            total_loss += loss.item() * n
            correct += (logits.argmax(1) == labels).sum().item()
            count += n
    if not count:
        raise ValueError('Boş veri yükleyicisi ölçülemez.')
    return {'loss': total_loss / count, 'accuracy': correct / count, 'samples': count}


def save_checkpoint(path, model, config, train_indices, val_indices, epoch, val_loss):
    # detach+clone: sonraki optimizer adımları en iyi ağırlıkları değiştirmesin.
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save({'state_dict': state, 'config': dict(config), 'seed': config['seed'],
                'train_indices': list(train_indices), 'val_indices': list(val_indices),
                'epoch': epoch, 'val_loss': val_loss, 'versions': versions(),
                'normalization': {'mean': MEAN, 'std': STD}}, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['normalization'] != {'mean': MEAN, 'std': STD}:
        raise ValueError('Checkpoint normalizasyonu bu uygulamayla uyuşmuyor.')
    model = MLP(checkpoint['config']['hidden_sizes']).to('cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
