# MNIST ile MLP Tabanlı El Yazısı Rakam Tanıma

**Ders:** YZM517 — Yapay Sinir Ağları · **Ödev 1**

Bu proje, MNIST rakamlarını yalnız tam bağlı katmanlardan oluşan bir Multi-Layer Perceptron (MLP) ile sınıflandırır. Modeller sıfırdan eğitilir; CNN veya hazır eğitilmiş ağırlık kullanılmaz. İki gizli katman genişliği, aynı veri bölümü ve aynı eğitim bütçesi altında karşılaştırılır.

> **Sonuç:** İki tam eğitim ve final test tamamlandı. Seçilen model MNIST testinde **%97.80**, öğrencinin gerçek el yazısında **8/9 hane (%88.89)** doğruluk verdi. Deneyler ve yorumlar [RAPOR.md](RAPOR.md) içinde; ham ölçümler `results/`, gerçek el yazısı `handwriting/` altında. Classroom teslimi henüz yapılmadı; öğrencinin konuyu anlatabilmesi ayrıca çalışılacaktır.

## 1. Deney düzeni

| Özellik | Değer |
|---|---|
| Girdi | 28 × 28 gri görüntü, düzleştirilmiş 784 özellik |
| Deney A | 784 → 128 → 64 → 10 |
| Deney B | 784 → 256 → 128 → 10 |
| Gizli aktivasyon | ReLU |
| Çıkış | 10 logits; eğitimden önce softmax eklenmez |
| Loss | CrossEntropyLoss |
| Optimizer | Adam, öğrenme oranı 0.001 |
| Eğitim bütçesi | Her model için 12 epoch, batch size 128 |
| Tohum | 42 |
| Train / validation | MNIST eğitim havuzundan ayrık 50.000 / 10.000 örnek |
| Test | Resmî ayrı 10.000 örnek; yalnız model seçiminden sonra |
| Normalizasyon | `(pixel / 255 − 0.1307) / 0.3081` |
| Checkpoint seçimi | Her koşuda minimum validation loss; sonra A/B arasında minimum validation loss |
| Donanım | CPU, iki PyTorch thread |

Tek değişken gizli katman genişliğidir. Tüm modeller aynı seed ile oluşturulmuş aynı indeksleri kullanır. Test doğruluğu model/epoch/hiperparametre seçmek için kullanılmaz. Tek tohumla karşılaştırma, istatistiksel üstünlük kanıtı değildir.

## 2. Kurulum ve test

Python sanal ortamı kullanın. Bu sunucuda doğrulanan sürümler `requirements.txt`, tam ortam dökümü `environment-lock.txt` içindedir. Diğer işletim sistemlerinde aynı sürümlerin kurulabilirliği ayrıca doğrulanmamıştır. GPU gerekmez; kod CPU kullanır.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Bu çalışmanın sunucudaki Python yolu:

```bash
PY=/home/ubuntu/.local/share/education-venv/bin/python
"$PY" -m pytest -q
```

## 3. Eğitimi yeniden üretme

Komutları repo kökünde çalıştırın. MNIST dizini Git deposunun dışında olmalıdır. İlk indirme internet bağlantısı gerektirir. Çıktı klasörü zaten varsa üzerine yazılmaz; yeni bir ad seçin.

```bash
python train.py --data-dir "$HOME/datasets/mnist" \
  --epochs 12 --batch-size 128 --hidden-sizes 128 64 \
  --optimizer adam --lr 0.001 --seed 42 --output runs/deney-a

python train.py --data-dir "$HOME/datasets/mnist" \
  --epochs 12 --batch-size 128 --hidden-sizes 256 128 \
  --optimizer adam --lr 0.001 --seed 42 --output runs/deney-b
```

Her koşu `config.json`, `environment.json`, `split.json`, `history.csv`, `history.json`, `summary.json`, `loss.png`, `accuracy.png` ve `best.pt` üretir. MNIST indirmenin test arşivini de önbelleğe alması test örneklerinin eğitimde kullanıldığı anlamına gelmez; eğitim yordamı yalnız `train=True` havuzunu yükler.

**Ölçüm ayrıntısı:** Training loss/accuracy, epoch içindeki minibatch güncellemeleri sırasında hesaplanır; tek bir sabit final modelin train ölçümü değildir. Validation ölçümü her epoch sonunda sabit modelle yapılır. Loss/accuracy, örnek sayılarıyla ağırlıklandırılır. Eğitim süresi epoch döngüsü ile checkpoint/log yazımını kapsar; veri indirme ve grafik üretimi hariçtir.

## 4. Model seçimi ve final test

Önce her koşunun `summary.json` dosyasındaki `best_val_loss` değerlerini karşılaştırın. Küçük olanı seçin, kararınızı kaydedin; ancak bundan sonra seçilen checkpoint üzerinde test çalıştırın:

```bash
# A seçilmişse; B seçilirse checkpoint yolunu değiştirin.
python evaluate.py --checkpoint runs/deney-a/best.pt \
  --data-dir "$HOME/datasets/mnist" \
  --output runs/final-test --final-model-selected
```

Bu bayrak yöntem disiplininin yerini tutmaz. Test sonucundan sonra ayar değiştirilirse aynı test artık tarafsız final test olarak sunulamaz. Çıktı, test loss/accuracy, ölçüm süresi, checkpoint SHA-256 ve confusion matrix dosyalarını içerir. Confusion matrix satırları gerçek, sütunları tahmin edilen sınıftır.

## 5. Kendi el yazısı deneyi

Bu bölümde yalnız öğrencinin gerçekten kendi yazdığı görüntüler kullanılmalıdır. MNIST örnekleri, bilgisayar fontları veya yapay üretilmiş rakamlar kişisel el yazısı yerine sunulmaz.

1. Öğrenci numarasını sade arka plana koyu kalemle, rakamlar birbirine değmeden yazın.
2. Net fotoğrafı `kisisel/` altında saklayın; her rakamı ayrı görüntü olarak kırpın.
3. Seçilen checkpoint ile çalıştırın:

```bash
python predict.py --checkpoint runs/deney-a/best.pt \
  --image kisisel/hane-01.png kisisel/hane-02.png \
  --output kisisel/sonuclar
```

Ön işleme: EXIF yönünü düzeltme → gri ton → arka plan/rakam rengini uyarlama → rakam alanını kırpma → uzun kenarı 20 piksele ölçekleme → 28 × 28 siyah tuvale yerleştirme → ağırlık merkezine göre ortalama → eğitimle aynı normalizasyon. Her dosya tek rakam içermelidir; tam numara segmentasyonu bu CLI tarafından yapılmaz.

`*_prepared.png` görüntülerini gözle kontrol edin. Gerçek etiketleri tahminlerle karşılaştırın; yalnız softmax maksimumunu doğruluk saymayın. Softmax olasılığı kalibre bir güven garantisi değildir. Yanlış tahminleri gizlemeyin: fotoğraf ışığı, kalem kalınlığı, eğim, kırpma ve MNIST ile dağılım farkı nedeniyle hata oluşabilir.

**Gizlilik ve teslim:** Öğrenci, numara görsellerinin de **public GitHub deposunda yayımlanmasını açıkça onaylamıştır**. Bu nedenle teslim için gerekli orijinal fotoğraf, hane kırpmaları, hazırlanmış görüntüler ve sonuç karşılaştırması `handwriting/` altında paylaşılır. Yerel çalışma kopyaları `kisisel/` altında Git dışında kalır. Depo herkese açıktır; öğretmenin görüntülemek için davete ihtiyacı yoktur.

### Hazır modelle kişisel tahmini yeniden çalıştırma

Bu komut yeni eğitim yapmadan repodaki seçilmiş checkpoint ile aynı dokuz kırpmayı değerlendirir. `--output` dizini önceden var olmamalıdır.

```bash
python predict.py --checkpoint results/deney-b/best.pt \
  --image handwriting/hane-01.png handwriting/hane-02.png \
          handwriting/hane-03.png handwriting/hane-04.png \
          handwriting/hane-05.png handwriting/hane-06.png \
          handwriting/hane-07.png handwriting/hane-08.png handwriting/hane-09.png \
  --output kisisel/yeniden-tahmin
```

`handwriting/metrics.json` gerçek etiketlerle karşılaştırmayı, `handwriting/tahminler/predictions.json` softmax dağılımlarını içerir. Modelin 6. hanedeki 7’yi 4 olarak tanıması raporda korunmuştur.

## 6. Kod haritası

| Dosya | Sorumluluk |
|---|---|
| `mlp.py` | En az iki ReLU gizli katmanlı MLP |
| `core.py` | Seed, ayrık veri bölümü, eğitim ölçümü, checkpoint |
| `train.py` | Yalnız train/validation eğitimi, grafikler |
| `evaluate.py` | Final seçimin ardından ayrı test kümesi |
| `predict.py` | Gerçek tek-rakam görselini hazırlama ve tahmin |
| `tests/` | Mimari, veri ayrımı, CLI, ölçüm ve görüntü testleri |
| `ANLATIM.md` | Kavramları tartışmak için Türkçe çalışma notu |
| `KAYNAKLAR.md` | Kaynaklar |

## 7. Yapay zekâ desteği ve öğrenme

Kod, deney yürütme ve rapor hazırlığında yapay zekâ desteği kullanılmıştır. Sayısal sonuçlar gerçek çalıştırma dosyalarından alınır; kişisel el yazısı öğrenciye aittir. Öğrencinin mimariyi, train/validation/test ayrımını, loss/accuracy eğrilerini ve kendi el yazısındaki hataları açıklaması ayrıca çalışılacak bir öğrenme adımıdır; kodun hazır olması bu yetkinliği otomatik kanıtlamaz.
