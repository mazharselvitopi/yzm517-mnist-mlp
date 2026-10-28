# Kaynaklar

## Teknik kaynaklar

Aşağıdaki kaynakları, yapay zekâ desteğiyle hazırlanan çalışmanın ilgili kod bölümleri için referans olarak kullandım. Kaynakların kullanım kapsamını her başlığın altında belirttim.

### [S01] PyTorch: Build the Neural Network

https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

İlgili konular: `nn.Module`, `forward`, `Flatten`, `Linear`, `ReLU`, `Sequential`, logits ve softmax.

Bu kaynak, `mlp.py` içindeki model yapısı ve [ANLATIM.md](ANLATIM.md) dosyasındaki tensor boyutları açıklamasıyla ilişkilidir. Kaynaktaki örnek FashionMNIST kullanır; bu projede kullanılan veri seti MNIST'tir. Kaynaktaki örnek sonuçları kendi deney sonucum olarak kullanmadım.

### [S02] PyTorch: CrossEntropyLoss

https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/loss.py

İlgili konular: `CrossEntropyLoss` sınıfının açıklaması, normalize edilmemiş logits beklentisi, sınıf indeksli etiketler ve reduction işlemi. Kullanılan kaynak PyTorch v2.14.1 sürümüne sabitlenmiştir.

Eğitimde logits'e ayrıca softmax eklenmemesi ve kaybın örnek sayısıyla ağırlıklandırılması bu açıklamalarla ilişkilidir.

### [S03] Torchvision: MNIST

https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html

İlgili konular: `root`, `train`, `transform`, `download` parametreleri ve görüntü/etiket dönüşü.

`train.py` eğitim havuzunu, `evaluate.py` ise ayrı test havuzunu yükler. Train/validation bölmesi bu projenin kodunda yapılır; kaynak dokümanın verdiği hazır bir bölüm değildir.

## Projeye ait kayıtlar

- `results/`: Gerçek eğitim ve test sonuçları, hiperparametreler, veri bölmesi, grafikler ve checkpoint'ler.
- `handwriting/`: Kendi el yazımın orijinal fotoğrafı, hane kırpmaları ve tahmin kayıtları.
- `evidence/lesson-probes.txt`: Tensor boyutu, cross-entropy ve gradient için önceki küçük çalıştırmaların çıktıları.
- `tests/`: Küçük ve sentetik girdilerle yapılan kod kontrolleri. Bunların sonuçları, modelin MNIST veya el yazısı doğruluğu değildir.

Kaynakların ilgili bölümleri ilk hazırlıkta 22 Ekim 2026 tarihinde incelenmiştir. Bu liste, belgelerin tamamının veya bütün PyTorch sitesinin okunduğu anlamına gelmez. Rapordaki deney sonuçları bu kaynaklardan değil, proje kodunun gerçek çalıştırmalarından alınmıştır.
