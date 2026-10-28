"""Bir dosya = bir rakam. Tam öğrenci numarası segmentasyonu yapılmaz."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from core import MEAN, STD, load_checkpoint, seed_everything, write_json


EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def preprocess_image(path, polarity='auto'):
    path = Path(path)
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError('Desteklenmeyen uzantı. PNG/JPEG/BMP/TIFF kullanın.')
    if polarity not in ('auto', 'dark', 'light'):
        raise ValueError('polarity auto/dark/light olmalı.')
    try:
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source)
            rgba = source.convert('RGBA')
            alpha = np.asarray(rgba.getchannel('A'))
            if alpha.max() == 0:
                raise ValueError('Boş/şeffaf görüntüde rakam yok.')
            background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            gray = Image.alpha_composite(background, rgba).convert('L')
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f'Görüntü okunamadı: {path}') from exc
    pixels = np.asarray(gray, dtype=np.float32)
    if pixels.max() - pixels.min() < 25:
        raise ValueError('Boş veya düşük kontrastlı görüntü.')
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    invert = polarity == 'dark' or (polarity == 'auto' and np.median(border) > 127)
    if invert:
        pixels = 255 - pixels
    # Beyaz rakam/siyah arka plan; hafif tarayıcı arka planını temizle.
    pixels = pixels - np.median(np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]]))
    pixels = np.clip(pixels, 0, 255)
    if pixels.max() < 25:
        raise ValueError('Boş görüntü: arka plandan ayrılan rakam yok.')
    pixels = pixels * (255.0 / pixels.max())
    mask = pixels > 25
    ys, xs = np.where(mask)
    if len(xs) < 3:
        raise ValueError('Boş veya çok küçük rakam alanı.')
    crop = Image.fromarray(pixels.astype(np.uint8)).crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    scale = 20 / max(crop.size)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    prepared = Image.new('L', (28, 28), 0)
    prepared.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))
    mass = np.asarray(prepared, dtype=np.float32)
    y_center = float((mass.sum(axis=1) * np.arange(28)).sum() / mass.sum())
    x_center = float((mass.sum(axis=0) * np.arange(28)).sum() / mass.sum())
    centered = Image.new('L', (28, 28), 0)
    centered.paste(prepared, (round(13.5 - x_center), round(13.5 - y_center)))
    tensor = torch.from_numpy(np.asarray(centered, dtype=np.float32).copy()).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - MEAN) / STD
    return tensor, centered


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--checkpoint', type=Path, required=True, help='Yalnız kendi güvenilir best.pt dosyanız')
    cli.add_argument('--image', nargs='+', type=Path, required=True, help='Her dosyada yalnız tek el yazısı rakamı')
    cli.add_argument('--output', type=Path, required=True, help='Var olmayan, kişisel sonuçlar için yerel klasör')
    cli.add_argument('--polarity', choices=['auto', 'dark', 'light'], default='auto', help='Rakamın girişteki rengi (auto kenardan tahmin eder)')
    args = cli.parse_args(argv)
    # Önce tüm girdileri doğrula; bozuk dosya varsa kısmi tahmin üretme.
    prepared_inputs = [preprocess_image(path, args.polarity) for path in args.image]
    args.output.mkdir(parents=True, exist_ok=False)
    model, checkpoint = load_checkpoint(args.checkpoint)
    seed_everything(checkpoint['seed'])
    results = []
    with torch.no_grad():
        for index, (path, (tensor, image)) in enumerate(zip(args.image, prepared_inputs)):
            probabilities = model(tensor.to('cpu')).softmax(dim=1)[0]
            prediction = probabilities.argmax().item()
            image.save(args.output / f'{index:03d}_prepared.png')
            results.append({'input': str(path.resolve()), 'prediction': prediction,
                            'probabilities': probabilities.tolist(),
                            'maximum_probability': probabilities.max().item(), 'polarity': args.polarity})
            print(f'{path.name}: prediction={prediction} (softmax olasılığı kalibre güven garantisi değildir)')
    write_json(args.output / 'predictions.json', results)
    return results


if __name__ == '__main__':
    main()
