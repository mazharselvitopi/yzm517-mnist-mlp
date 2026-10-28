# Kaynaklar ve kapsam

## Asıl ödev yönergesi

`ODEV.md`, Classroom Ödev 1 yönergesinin yerel kopyasıdır. Son teslim tarihi ve içerik **28 Ekim 2026 tarihinde canlı Classroom API üzerinden** yeniden kontrol edilmiştir: 28 Ekim 2026, 23.59 Europe/Istanbul. Kendi el yazısı, çalıştırılabilir kod, MLP ve teslim şartlarında öğretmen yönergesi esastır.

## Teknik kaynaklar

Aşağıdaki kaynaklar ilk atölye hazırlığında 22 Ekim 2026 tarihinde ilgili bölümleriyle incelenmiştir. Bu repo için tüm belgelerin yeniden okunduğu veya tüm sitenin tarandığı iddia edilmez.

### [S01] PyTorch — Build the Neural Network

https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

İncelenen kapsam: `nn.Module`, `forward`, `Flatten`, `Linear`, `ReLU`, `Sequential`, logits ve softmax. Buradaki örnek veri FashionMNIST’tir; bu proje **MNIST** kullanır. Resmî örneğin sonuçları bu projenin sonucu değildir.

### [S02] PyTorch — CrossEntropyLoss

https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/loss.py

Sürüme sabitlenmiş resmî kaynak kodundaki `CrossEntropyLoss` docstring kapsamı: normalize edilmemiş logits, sınıf-indeksli target, log-softmax + NLLLoss eşdeğerliği ve reduction. Bu nedenle model eğitimde softmax değil logits döndürür.

### [S03] Torchvision — MNIST

https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html

İncelenen kapsam: `root`, `train`, `transform`, `download` ve görüntü/etiket dönüşü. Train/validation ayrımı ve deney sonuçları bu API belgesinden değil projenin gerçek kodundan/çalıştırmasından gelir.

## Deney kanıtının sınırı

- `results/` altındaki dosyalar varsa gerçek deneylerin kopyasıdır; küçük fixture testlerinin sonucu değildir.
- `evidence/lesson-probes.txt` ilk hazırlık sırasında gerçekten çalıştırılan tensor/CE/gradient problarını içerir.
- `tests/` veri indirmeyen küçük testlerle kod sözleşmelerini denetler; testlerin geçmesi MNIST doğruluğunu veya öğrencinin konuyu öğrendiğini kanıtlamaz.
- `ANLATIM.md` açıklama ve alıştırma notudur. Nihai ölçümler için `RAPOR.md` ve ilgili JSON/CSV dosyaları esas alınır.
