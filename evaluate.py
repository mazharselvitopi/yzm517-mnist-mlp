"""Validation ile final model seçildikten sonra tek seferlik MNIST test ölçümü."""
import argparse
import csv
import hashlib
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from core import MEAN, STD, load_checkpoint, seed_everything, validate_data_dir, versions, write_json


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--checkpoint', type=Path, required=True, help='Yalnız kendi ürettiğiniz güvenilir best.pt')
    cli.add_argument('--data-dir', type=Path, required=True, help='Git dışındaki MNIST önbelleği')
    cli.add_argument('--output', type=Path, required=True, help='Var olmayan final değerlendirme klasörü')
    cli.add_argument('--batch-size', type=int, default=256)
    cli.add_argument('--final-model-selected', action='store_true', help='Modeli validation ile seçtim; test ile ayar yapmayacağım')
    args = cli.parse_args(argv)
    if not args.final_model_selected:
        cli.error('Önce validation karşılaştırmasını tamamlayın; --final-model-selected sonra eklenir.')
    if args.batch_size <= 0:
        cli.error('batch-size pozitif olmalı.')
    try:
        data_dir = validate_data_dir(args.data_dir)
    except ValueError as exc:
        cli.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    model, checkpoint = load_checkpoint(args.checkpoint)
    seed_everything(checkpoint['seed'])
    dataset = datasets.MNIST(root=str(data_dir), train=False, download=True,
                             transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    loss_sum, correct, count = 0.0, 0, 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    started = time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to('cpu'), labels.to('cpu')
            logits = model(images)
            predictions = logits.argmax(1)
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
            correct += (predictions == labels).sum().item()
            count += len(labels)
            confusion += torch.bincount(10 * labels + predictions, minlength=100).reshape(10, 10)
    if not count:
        raise ValueError('Boş test kümesi.')
    metrics = {'loss': loss_sum / count, 'accuracy': correct / count, 'samples': count,
               'evaluation_seconds': time.perf_counter() - started,
               'checkpoint': str(args.checkpoint.resolve()),
               'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               'selected_epoch': checkpoint['epoch'], 'selection_metric': 'val_loss',
               'final_model_selected': True, 'versions': versions()}
    write_json(args.output / 'test_metrics.json', metrics)
    write_json(args.output / 'confusion_matrix.json', confusion.tolist())
    with (args.output / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['actual / predicted', *range(10)])
        for actual, row in enumerate(confusion.tolist()):
            writer.writerow([actual, *row])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(confusion.numpy(), cmap='Blues')
    ax.set(xlabel='Tahmin', ylabel='Gerçek sınıf', xticks=range(10), yticks=range(10), title='MNIST test confusion matrix')
    for actual in range(10):
        for predicted in range(10):
            ax.text(predicted, actual, str(confusion[actual, predicted].item()), ha='center', va='center', fontsize=7)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(args.output / 'confusion_matrix.png', dpi=120)
    plt.close(fig)
    print(f"test_loss={metrics['loss']:.6f} test_accuracy={metrics['accuracy']:.4f} samples={count}")
    return metrics


if __name__ == '__main__':
    main()
