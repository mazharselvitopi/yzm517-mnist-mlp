# MLP'nin çalışma mantığına ilişkin notlarım

Bu dosyada projedeki veri akışını ve model seçimi için kullandığım kavramları özetledim. Deneylerin sayısal sonuçları [RAPOR.md](RAPOR.md), kaynaklar ise [KAYNAKLAR.md](KAYNAKLAR.md) içinde bulunuyor. Komutlar, README'deki kurulum tamamlandıktan sonra repo kökünde ve sanal ortam etkinken çalıştırılır.

## 1. Görüntüden sınıf skoruna

Modelin girdisi, gri tonlamalı bir rakam görüntüsüdür. Çıktısı ise 0–9 sınıfları için üretilen on skordur. Eğitim sırasında model, etiketli görüntülerden bu skorları belirleyen ağırlık ve bias değerlerini öğrenir.

MNIST'te yüksek başarı elde edilmesi, fotoğrafla alınan her el yazısının doğru tanınacağı anlamına gelmez. Kendi el yazımdaki altıncı hanenin yanlış tahmin edilmesi bu çalışmadaki örneğidir.

## 2. Tensor boyutları ve parametreler

A modeli için veri akışı şu şekildedir:

```text
Görüntü batch'i: (B, 1, 28, 28)
Flatten:         (B, 784)
Linear + ReLU:   (B, 128)
Linear + ReLU:   (B, 64)
Çıkış:           (B, 10)
```

Burada `B` batch boyutu, `1` gri ton kanal sayısıdır. Flatten işlemi batch eksenini korur; görüntünün diğer eksenlerini tek özellik ekseninde birleştirir. [S01]

Bu akışı eğitim veya veri indirme yapmadan kontrol etmek için:

```bash
python -c 'import torch; from mlp import MLP; x=torch.zeros(2,1,28,28); m=MLP([128,64]); print(tuple(x.shape),tuple(x.flatten(1).shape),tuple(m(x).shape)); print(sum(p.numel() for p in m.parameters()))'
```

Kayıtlı çıktı:

```text
(2, 1, 28, 28) (2, 784) (2, 10)
109386
```

Bir `Linear` katmanında parametre sayısı `girdi × çıktı + çıktı` şeklindedir. Son terim bias sayısıdır. A modelinde ağırlık matrisleri `(128,784)`, `(64,128)` ve `(10,64)` boyutlarındadır. B modelinde gizli katmanlar 256 ve 128 nöron olduğu için bu boyutlar değişir; girdi ve sınıf sayısı aynı kalır.

İlk katmanın hesabı:

```text
Z1 = X W1ᵀ + b1
H1 = ReLU(Z1)
```

Her nöron giriş piksellerinin ağırlıklı toplamını alır. Bir nöronun belirli bir çizgiyi veya şekli tanıdığını yalnızca mimarisine bakarak söylemek mümkün değildir; bunun ayrıca incelenmesi gerekir.

## 3. ReLU'nun görevi

ReLU, `max(0,z)` işlemini uygular. Gizli katmanlar arasına doğrusal olmayan dönüşüm ekler. Yalnız affine katmanları art arda kullanmak, bunların tek bir affine dönüşüm olarak birleştirilebilmesine yol açar. Bu nedenle sadece katman sayısını artırmak yeterli değildir. [S01]

İki modelde de her gizli katmandan sonra ReLU bulunur. Çıkış katmanından sonra ReLU yoktur. Negatif bölgede ReLU türevi sıfır olduğundan sürekli bu bölgede kalan bir nöronun öğrenmeye katkısı sınırlanabilir.

## 4. Logits, softmax ve cross-entropy

Çıkıştaki on değer logits'tir. Negatif olabilirler ve toplamlarının bir olması gerekmez. En yüksek logit'in indeksi sınıf tahminini verir. Softmax, bu skorları sınıflar üzerinde toplamı bir olan değerlere dönüştürür; projede kişisel tahminleri incelerken kullanılır.

`CrossEntropyLoss`, sınıf indeksleriyle kullanıldığında logits bekler. Bu yüzden eğitimde softmax eklemedim. Etiketlerin boyutu `(B,)`, logits'in boyutu `(B,10)` şeklindedir. [S02]

```bash
python -c 'import torch; z=torch.tensor([[2.,0.,-1.]]); y=torch.tensor([0]); f=torch.nn.functional.cross_entropy; print(round(f(z,y).item(),6),round(f(z.softmax(1),y).item(),6))'
```

Kayıtlı çıktı:

```text
0.169846 0.657846
```

İkinci hesap Python hatası üretmiyor; ancak softmax çıktısını yeniden logits olarak ele aldığı için farklı bir kayıp hesaplıyor.

Accuracy, en yüksek skora sahip sınıfın doğru olup olmadığına bakar. Cross-entropy ise doğru sınıfa verilen olasılığa da duyarlıdır. Bu yüzden accuracy değişmeden loss değişebilir. Benim deneyimde de B modelinin son epoch'ta daha yüksek validation accuracy vermesine rağmen daha yüksek validation loss üretmesi, iki ölçütün aynı bilgiyi vermediğini gösteriyor.

## 5. Geri yayılım ve ağırlık güncellemesi

Eğitim döngüsündeki temel adımlar:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

- `zero_grad`, önceki batch'ten kalan gradient'leri temizler.
- Forward işlemi, mevcut ağırlıklarla logits üretir.
- Loss, bu çıktılarla gerçek etiketler arasındaki kaybı hesaplar.
- `backward`, zincir kuralıyla parametrelerin gradient'lerini hesaplar.
- `step`, optimizer üzerinden parametreleri günceller.

Gradient, rakamın sınıf tahmini değildir. Kayıp fonksiyonunun bir parametreye göre yerel değişimini ifade eder. Basit bir örnek:

```bash
python -c 'import torch; w=torch.tensor([2.],requires_grad=True); loss=(w-5).pow(2).sum(); loss.backward(); print(w.grad.item())'
```

Kayıtlı çıktı `-6.0` değeridir. Basit SGD güncellemesinde `w_yeni = w − lr × gradient` kullanılır. Bu örnekte negatif gradient, ağırlığın artması yönünde güncelleme üretir. Adam ise gradient geçmişinden de yararlanır; farklı optimizer'larda aynı öğrenme oranının aynı davranışı vermesi beklenmez.

Validation sırasında ağırlık güncellemesi yapılmaz. Kodda değerlendirme modu ve kapalı gradient hesabı birlikte kullanılır. `eval()` çağrısının tek başına gradient hesabını kapatmadığına dikkat etmek gerekir.

## 6. Train, validation ve test ayrımı

Çalışmadaki üç bölümün kullanım amacı farklıdır:

| Bölüm | Kullanım |
|---|---|
| Train | Ağırlıkların güncellenmesi |
| Validation | Mimari ve checkpoint seçimi |
| Test | Seçim tamamlandıktan sonra final değerlendirme |

Torchvision'da `train=True`, MNIST'in eğitim havuzunu; `train=False`, ayrı test havuzunu yükler. Eğitim havuzunu 50.000 train ve 10.000 validation olarak böldüm. Test kümesi 10.000 örnek olarak ayrı kaldı. [S03]

Seed; başlangıç ağırlıkları, veri sırası ve bölme işlemi açısından önemlidir. Projede Python, NumPy ve PyTorch seed'leri ayarlanır, CPU ve iki thread kullanılır, bölme indeksleri kaydedilir. Farklı sürüm veya donanımda bit düzeyinde aynı sonucu garanti etmek yine de mümkün değildir.

A/B seçimini test accuracy değerlerine göre yapmak, test kümesini fiilen validation amacıyla kullanmak olurdu. Bu nedenle `train.py` test örneklerini yüklemez; final test ayrı komutla çalıştırılır. `--final-model-selected` bayrağı bu ayrımı kullanım düzeyinde belirtir; tek başına yöntem disiplinini garanti etmez.

## 7. Loss hesabı ve checkpoint seçimi

Batch boyutları eşit olmayabileceği için epoch loss'u, batch loss değerlerinin doğrudan ortalaması değildir. Kodda kullanılan hesap:

```text
epoch_loss = Σ(batch_loss × batch_örnek_sayısı) / toplam_örnek
accuracy   = toplam_doğru / toplam_örnek
```

`tests/test_core.py`, son batch'in daha küçük olduğu bir örnekle bu hesabı kontrol eder. Bu ifade, projede kullanılan eşit örnek ağırlıklı cross-entropy hesabı içindir.

Her epoch sonunda validation loss ölçülür ve iyileşme varsa checkpoint kaydedilir. Böylece son epoch yerine en düşük validation loss değerine sahip ağırlıklar korunur. A modelinde bu epoch 9, B modelinde 7 oldu.

Checkpoint kaydedilirken ağırlıklar klonlanır. Amaç, sonraki optimizer adımlarının kaydedilmiş en iyi ağırlıkları değiştirmemesidir. İlgili test, modelin ağırlıkları sonradan değiştirildiğinde checkpoint'ten eski çıktının geri geldiğini kontrol eder.

## 8. Kendi el yazımın hazırlanması

Fotoğrafın MNIST ile aynı boyutta olması tek başına yeterli değildir. Arka plan, ışık, kalem kalınlığı, eğim ve rakamın konumu da girdiyi değiştirir. Bu nedenle her hane için gri ton, renk yönü, kırpma, ölçekleme ve merkezleme işlemleri uygulanır.

`predict.py` tek rakam içeren dosyalarla çalışır. Fotoğraftaki haneleri otomatik bölmez. Hazırlanan görüntüler 28 × 28 boyutundadır ve eğitimle aynı normalizasyondan geçirilir. Boş, tamamen şeffaf veya düşük kontrastlı görüntüler hata olarak reddedilir.

Kendi el yazımdaki 7'nin 4 olarak tahmin edilmesi, ön işleme ve dağılım farkı açısından değerlendirilmesi gereken bir sonuçtur. Bu hatayı tek başına optimizer'a veya model kapasitesine bağlamak mümkün değildir. Yanlış tahminde maksimum softmax çıktısının yaklaşık 0.9171 olması da yüksek çıktının doğruluk garantisi olmadığını gösterir.

## Terimler

| Terim | Kısa açıklama |
|---|---|
| Batch / minibatch | Bir ileri ve geri adımda işlenen örnek grubu |
| Epoch | Train bölümünden bir tam geçiş |
| Hidden layer | Giriş ile çıktı arasındaki öğrenilebilir katman |
| Weight / bias | Dönüşüm katsayısı ve sabit terim |
| Logit | Normalize edilmemiş sınıf skoru |
| Gradient | Kaybın parametreye göre yerel değişimi |
| Learning rate | Optimizer güncellemesinin adım ölçeği |
| Overfitting | Eğitim verisine uyumun yeni örneklere aynı ölçüde taşınmaması |
| Distribution shift | Geliştirme ve kullanım verilerinin dağılımlarının farklılaşması |
| Checkpoint | Belirli bir aşamadaki model ağırlıkları ve yeniden kurma bilgileri |

Küçük hesapların önceki gerçek çalıştırma kaydı `evidence/lesson-probes.txt` dosyasında bulunuyor. Bunlar modelin MNIST başarımının yerine geçmez; eğitim ve test sonuçları raporda ayrı olarak veriliyor.
