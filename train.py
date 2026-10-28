"""Yalnız MNIST eğitim havuzundan train/validation; test burada açılmaz."""
import argparse
import csv
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from core import (MEAN, STD, make_split, run_epoch, save_checkpoint,
                  seed_everything, validate_data_dir, versions, write_json)
from mlp import MLP


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data-dir', type=Path, required=True, help='Git dışında MNIST önbelleği')
    result.add_argument('--output', type=Path, required=True, help='Henüz var olmayan deney klasörü')
    result.add_argument('--epochs', type=int, default=10)
    result.add_argument('--batch-size', type=int, default=64)
    result.add_argument('--hidden-sizes', nargs='+', type=int, default=[128, 64])
    result.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam')
    result.add_argument('--lr', type=float, default=0.001)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--val-size', type=int, default=10000, help='60000 eğitim örneğinden validation sayısı')
    result.add_argument('--limit-train', type=int, help='Yalnız smoke: train altkümesini küçült')
    result.add_argument('--limit-val', type=int, help='Yalnız smoke: validation altkümesini küçült')
    return result


def plot_history(history, output):
    for metric in ('loss', 'accuracy'):
        fig, ax = plt.subplots(figsize=(7, 4))
        for subset in ('train', 'val'):
            ax.plot([row['epoch'] for row in history], [row[f'{subset}_{metric}'] for row in history],
                    marker='o', label=subset)
        ax.set(xlabel='Epoch', ylabel=metric, title=f'MNIST MLP {metric}')
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f'{metric}.png', dpi=120)
        plt.close(fig)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if (args.epochs <= 0 or args.batch_size <= 0 or not 0 < args.lr < float('inf')
            or not 0 <= args.seed < 2**32):
        cli.error('epochs/batch-size/lr pozitif; seed 0..2**32-1 olmalı.')
    if args.val_size <= 0 or any(n is not None and n <= 0 for n in (args.limit_train, args.limit_val)):
        cli.error('Bölüm ve limit boyutları pozitif olmalı.')
    try:
        data_dir = validate_data_dir(args.data_dir)
        model = MLP(args.hidden_sizes).to('cpu')
    except ValueError as exc:
        cli.error(str(exc))
    # Var olan run hiçbir koşulda ezilmez. Başarısız koşu klasörü de korunur.
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    model = MLP(args.hidden_sizes).to('cpu')
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config['data_dir'] = str(data_dir)
    config['device'] = 'cpu'
    write_json(args.output / 'config.json', config)
    write_json(args.output / 'environment.json', versions())
    # Test kümesi yalnız evaluate.py içinde, final seçimden sonra yüklenir.
    dataset = datasets.MNIST(root=str(data_dir), train=True, download=True,
                             transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]))
    train_indices, val_indices = make_split(len(dataset), args.val_size, args.seed, args.limit_train, args.limit_val)
    write_json(args.output / 'split.json', {'seed': args.seed, 'pool': 'MNIST/train',
                                           'train_indices': train_indices, 'val_indices': val_indices})
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer_cls = torch.optim.Adam if args.optimizer == 'adam' else torch.optim.SGD
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    history, best_loss, best_epoch = [], float('inf'), 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        val_metrics = run_epoch(model, val_loader)
        row = {'epoch': epoch, 'train_loss': train_metrics['loss'], 'val_loss': val_metrics['loss'],
               'train_accuracy': train_metrics['accuracy'], 'val_accuracy': val_metrics['accuracy'],
               'elapsed_seconds': time.perf_counter() - started}
        history.append(row)
        if val_metrics['loss'] < best_loss:
            best_loss, best_epoch = val_metrics['loss'], epoch
            save_checkpoint(args.output / 'best.pt', model, config, train_indices, val_indices, epoch, best_loss)
        write_json(args.output / 'history.json', history)
        with (args.output / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"epoch={epoch} train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f} "
              f"train_accuracy={row['train_accuracy']:.4f} val_accuracy={row['val_accuracy']:.4f}")
    seconds = time.perf_counter() - started
    plot_history(history, args.output)
    summary = {'best_epoch': best_epoch, 'best_val_loss': best_loss, 'training_seconds': seconds,
               'training_seconds_scope': 'epoch döngüsü ve checkpoint/log I/O; veri indirme ve grafik hariç',
               'train_samples': len(train_indices), 'val_samples': len(val_indices), 'test_accessed': False,
               'run_kind': 'smoke' if args.limit_train or args.limit_val else 'experiment'}
    write_json(args.output / 'summary.json', summary)
    print(f'best_epoch={best_epoch} training_seconds={seconds:.3f} output={args.output}')
    return summary


if __name__ == '__main__':
    main()
