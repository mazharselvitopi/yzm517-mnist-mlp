# Deney Raporu — MNIST ile MLP Tabanlı El Yazısı Rakam Tanıma

## 1. Amaç ve yöntem

Amaç, MNIST veri setindeki 0–9 el yazısı rakamlarını CNN kullanmadan sınıflandırmak, iki MLP yapılandırmasını karşılaştırmak ve seçilen modeli öğrencinin gerçek el yazısına uygulamaktır. Kod, deney ve rapor hazırlığında yapay zekâ desteği kullanılmıştır. Tablolardaki sayılar gerçek çalıştırmaların JSON/CSV kayıtlarından alınmıştır; kişisel görüntü öğrenci tarafından sağlanmıştır. Bu metin, öğrencinin konuyu öğrendiğine ilişkin bir değerlendirme değildir.

## 2. Veri ve deney kontrolü

MNIST’in 60.000 örnekli eğitim havuzu, seed 42 ile tek permütasyon üzerinden **50.000 train ve 10.000 validation** olarak bölündü. İndeksler `results/deney-{a,b}/split.json` içinde kayıtlıdır. İki deneyin train/validation indekslerinin birebir aynı olduğu ve bölümlerin kesişmediği programatik olarak doğrulandı. Resmî ayrı **10.000 test örneği**, model seçiminden sonra yalnız seçilen model için ölçüldü.

Her görüntü gri tonlamalı 28 × 28 pikseldir. Yoğunluklar önce [0,1] aralığına dönüştürülüp `(x−0.1307)/0.3081` ile normalize edildi. Bu sabitler deneyler veya test sonucu kullanılarak yeniden ayarlanmadı. Bölme rastgeledir, stratified değildir.

## 3. Modeller ve hiperparametreler

| Özellik | Deney A | Deney B |
|---|---:|---:|
| Mimari | 784 → 128 → 64 → 10 | 784 → 256 → 128 → 10 |
| Öğrenilebilir parametre | 109.386 | 235.146 |
| Gizli katman sayısı | 2 | 2 |
| Aktivasyon | ReLU | ReLU |
| Optimizer | Adam | Adam |
| Öğrenme oranı | 0.001 | 0.001 |
| Batch size | 128 | 128 |
| Epoch | 12 | 12 |
| Seed | 42 | 42 |
| Kayıp | CrossEntropyLoss | CrossEntropyLoss |

Tek karşılaştırma değişkeni gizli katman genişliğidir. Ağırlıklar sıfırdan başlatıldı. Çıkışta 10 logits üretilir; `CrossEntropyLoss` öncesinde softmax uygulanmaz. Eğitim CPU üzerinde iki PyTorch thread ile yapıldı. Sürüm ve platform kayıtları her deneyin `environment.json` dosyası ile `environment-lock.txt` içindedir.

## 4. Eğitim ve validation sonuçları

Her koşuda en düşük validation loss elde edilen checkpoint saklandı. Aşağıdaki train değerleri, seçilen epoch’un minibatch güncellemeleri sırasında toplanan ölçümlerdir; sabit checkpoint’in train kümesi üzerinde ayrıca ölçülmesi değildir. Validation değerleri epoch sonunda sabit modelle hesaplandı.

| Ölçüt | Deney A | Deney B |
|---|---:|---:|
| En iyi epoch (minimum validation loss) | 9 | 7 |
| Seçilen epoch training loss | 0.026690 | 0.027351 |
| Seçilen epoch validation loss | 0.083207 | **0.075181** |
| Seçilen epoch training accuracy | %99.150 | %99.122 |
| Seçilen epoch validation accuracy | %97.570 | **%97.800** |
| Son epoch training loss | 0.019649 | 0.015053 |
| Son epoch validation loss | 0.104741 | 0.093236 |
| Son epoch training accuracy | %99.306 | %99.492 |
| Son epoch validation accuracy | %97.370 | %97.900 |
| 12 epoch eğitim süresi | 147.998 saniye | 151.576 saniye |

Eğitim süresi epoch döngüsü ile checkpoint/log yazımını kapsar; indirme ve grafik üretimi hariçtir. Bu süreler aynı sunucudaki tek koşulardır; genel bir donanım benchmark’ı veya istatistiksel hız üstünlüğü olarak yorumlanmamalıdır.

### Eğitim grafikleri

**Deney A — loss ve accuracy:**

![A loss](results/deney-a/loss.png)

![A accuracy](results/deney-a/accuracy.png)

**Deney B — loss ve accuracy:**

![B loss](results/deney-b/loss.png)

![B accuracy](results/deney-b/accuracy.png)

### Model seçimi ve yorum

**Deney B’nin 7. epoch checkpoint’i seçildi**, çünkü A’nın en iyi validation loss’u 0.083207 iken B’ninki 0.075181 idi. Seçim kaydı `results/selection.json` dosyasına test çalıştırılmadan önce yazıldı. Test accuracy bu kararı vermek için kullanılmadı.

Her iki modelde train loss azalmaya devam ederken en iyi epoch’tan sonraki validation loss daha yüksek ve oynak kaldı. Bu örüntü aşırı öğrenmeyle uyumludur; tek bir koşu tek başına nedenin kesin kanıtı değildir. Daha büyük B modelinin validation loss’u daha düşüktür, fakat buradan tüm büyük modellerin daha iyi olduğu sonucu çıkmaz.

B’nin 12. epoch validation accuracy’si %97.9 ile seçilen 7. epoch’un %97.8 değerinden yüksektir; buna rağmen validation loss daha kötüdür. Bu çelişki değildir: accuracy yalnız doğru sınıfı seçip seçmediğine bakar; cross-entropy ise tahmin olasılıklarının büyüklüğüne de duyarlıdır. Seçim kriteri önceden **validation loss** olarak sabitlendiği için son epoch’a geçilmedi.

İlk B çalıştırması 8. epoch sonrasında dış kesintiyle durdu. Yarım koşu karşılaştırmaya alınmadı; aynı ayarlarla sıfırdan başlatılan ve 12 epoch’u tamamlayan ikinci çalıştırma kullanıldı. Bu, checkpoint’ten devam ettirilmiş bir koşu değildir.

## 5. Ayrı test kümesi sonucu

Seçilen B/epoch 7 modeli, 10.000 test görüntüsü üzerinde ölçüldü:

| Ölçüt | Sonuç |
|---|---:|
| Test loss | 0.077727 |
| Test accuracy | **%97.80** |
| Doğru / toplam | **9.780 / 10.000** |
| Yanlış | 220 |

Ham ölçüm: `results/final-test/test_metrics.json`. Checkpoint SHA-256 bu kayıtta yer alır. Confusion matrix toplamının 10.000 ve köşegen toplamının 9.780 olduğu ayrıca doğrulandı.

![Test confusion matrix](results/final-test/confusion_matrix.png)

Satırlar gerçek sınıf, sütunlar tahmin edilen sınıftır. En sık gözlenen yanlış eşleştirme, gerçek 9’un 4 tahmin edilmesidir (24 örnek). Diğer sık çiftler 3→9 (9), 2→7 (9), 7→9 (8), 7→1 (7) oldu. Bunlar görüntüleri tek tek incelemeden hataların fiziksel nedenini kanıtlamaz.

Test sonucundan sonra mimari, epoch veya hiperparametre değiştirilmedi.

## 6. Öğrencinin kendi el yazısı

Öğrenci kendi yazdığı dokuz haneli numaranın fotoğrafını sağladı. Orijinal görüntü `handwriting/orijinal.png` içinde korunur. İlk iki rakamın çizgileri birbirine temas ettiği için her hane **insan tarafından belirlenen dikdörtgenlerle** kırpıldı. Rakam çizilmedi, yeniden üretilmedi veya model sonucuna göre düzeltilmedi. Kırpma koordinatları ve insanın okuduğu gerçek etiketler `handwriting/kirpma-kaydi.json` içindedir; etiketler model tahmininden türetilmedi.

Ön işleme sırası: EXIF yönü → gri ton → renk yönünü uyarlama → rakam alanını kırpma → uzun kenarı 20 piksele ölçekleme → 28 × 28 tuval → ağırlık merkeziyle ortalama → eğitimle aynı normalizasyon. Hazırlanan dokuz görüntü görsel olarak incelendi; rakamlar okunabilir durumda ve arka plan belirgin biçimde ayrılmıştı. Bu gözlem ilk iki hanenin kırpılmasındaki manuel kararı kusursuz bir otomatik segmentasyon yöntemi yapmaz.

![Gerçek ve tahmin edilen kişisel rakamlar](handwriting/karsilastirma.png)

| Hane | Gerçek | Tahmin | Sonuç |
|---|---:|---:|---|
| 1 | 2 | 2 | Doğru |
| 2 | 6 | 6 | Doğru |
| 3 | 9 | 9 | Doğru |
| 4 | 7 | 7 | Doğru |
| 5 | 9 | 9 | Doğru |
| 6 | 7 | **4** | **Yanlış** |
| 7 | 0 | 0 | Doğru |
| 8 | 1 | 1 | Doğru |
| 9 | 3 | 3 | Doğru |

**Hane bazlı doğruluk: 8/9 = %88.89.** Tüm numaranın eksiksiz tanınması açısından sonuç başarılı değildir; bir hane yanlıştır. Dokuz haneli tek bir örnekten gerçek dünyadaki genel doğruluk için güvenilir bir oran çıkarılamaz.

Yanlış tahmin edilen altıncı hanede modelin 4 sınıfı için maksimum softmax çıktısı yaklaşık **0.9171** oldu. Yüksek softmax çıktısı doğru tahmin veya kalibre güven anlamına gelmez. El yazısındaki orta çizgi, kamera/kâğıt koşulları ve MNIST’ten dağılım farkı olası açıklamalardır; bu deney bunlardan hangisinin nedensel olduğunu ayırmaz. Özellikle aynı fotoğraftaki diğer 7 doğru tanındığından, yalnız rakam sınıfına bakarak kesin neden söylemek mümkün değildir. Sonucu iyileştirmek için test sonrası yeniden eğitim veya seçici fotoğraf değiştirme yapılmadı.

## 7. Sınırlılıklar ve sonuç

- İki mimari, tek seed ve tek veri bölümü karşılaştırıldı; çoklu seed belirsizliği ölçülmedi.
- Validation ile model ve epoch seçildiği için validation sonucu tarafsız final test sonucu değildir.
- Rastgele bölme stratified değildir; sınıf dengesi zorunlu olarak sabitlenmez.
- Kişisel örnek sayısı yalnız dokuzdur ve bazı rakamlar tekrarlıdır; tüm rakam sınıflarını kapsamaz.
- Kırpma manueldir; uçtan uca çok-haneli OCR sistemi yapılmadı.
- MLP’nin yüksek MNIST doğruluğu, kamera fotoğraflarında aynı başarıyı garanti etmedi.

Sonuç olarak iki gizli katmanlı, sıfırdan eğitilmiş MLP ile MNIST test kümesinde **%97.80**, öğrencinin sağladığı gerçek el yazısında **8/9 hane doğruluğu** elde edildi. Modelin başarısı ile öğrencinin bu süreci açıklayabilmesi ayrı değerlendirilmelidir.
