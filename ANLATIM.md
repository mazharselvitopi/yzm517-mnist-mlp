# Pikselden karara: MLP’yi çalıştırmak ile açıklayabilmek farklıdır

## 1. Mazhar, önce çözmeye çalıştığımız sorun

Elinde açık zemin üzerine yazılmış bir rakam var. Programın bunu `7` diye adlandırmasını istiyorsun. İlk akla gelen yaklaşım, bir örnek `7` görüntüsünü saklayıp yeni resmin pikselleriyle farkını karşılaştırmak olabilir. Ama rakam birkaç piksel sağa kayınca aynı insan yazısı çok farklı piksel vektörüne dönüşür. Her yazı stiline bir `if` eklemek sürdürülebilir değildir.

Burada öğrendiğimiz şey bir resim arşivi değil, **etiketli örneklerden karar üreten parametrelerdir**. Modelin görevi her girdiye on sınıf için skor üretmek; eğitim yordamının görevi yanlış kararların kaybını azaltacak ağırlık güncellemeleri yapmaktır. MNIST üzerinde başarı, kendi kalemin/kameran üzerinde başarıyı garanti etmez. Bu ödevin kişisel görüntü bölümü tam bu ayrımı görünür kılar.

> **AI ile çalışma başlangıcı:** “MNIST MLP taslağındaki `MLP.forward` ve `run_epoch` kodunu açıklamaya çalışacağım. Önce benden tensor boyutlarını tahmin etmemi iste. Cevabımı bekle; yanlışım varsa tek bir karşı örnek ver. Ben söylemeden test başarısını öğrenme kanıtı sayma.”

## 2. Deneyelim: model görüntüyü hangi boyutlarda görüyor?

README’deki `PY` değişkeni bu shell’de tanımlı olmalı. Bu komut veri indirmez, eğitim yapmaz:

```bash
"$PY" -c 'import torch; from mlp import MLP; x=torch.zeros(2,1,28,28); m=MLP([128,64]); print(tuple(x.shape),tuple(x.flatten(1).shape),tuple(m(x).shape)); print(sum(p.numel() for p in m.parameters()))'
```

Bu ortamda gerçek ölçüm: `(2, 1, 28, 28)` → `(2, 784)` → `(2, 10)` ve **109386 parametre** (`evidence/lesson-probes.txt`). Burada `2` batch boyutudur, sınıf sayısı değil. Kanal `1` gri görüntüyü ifade eder. Flatten batch eksenini korur, görüntünün geri kalan eksenlerini tek özellik ekseninde birleştirir. [S01]

Model girişine matematikte `X ∈ R^(B×784)` diyebiliriz. İlk katmanın her nöronu 784 pikselin ağırlıklı toplamını alır:

`Z1 = X W1ᵀ + b1`, `H1 = ReLU(Z1)`.

Bu projede `W1` shape’i `(128, 784)`, `b1` shape’i `(128,)`; dolayısıyla `Z1` shape’i `(B, 128)` olur. Sonra `W2` `(64,128)`, `W3` `(10,64)` şekillidir. Bias her batch örneğine aynı vektör olarak eklenir. W’nin bir satırı, tek nöronun tüm girişlere verdiği katsayılardır. Bunları “şu nöron kesin üst çizgiyi tanır” diye peşinen adlandırma; bu yorum ancak ölçüm/incelemeyle savunulabilir.

**Kendin dene:** `--hidden-sizes 32 16` olsaydı üç ağırlık matrisinin boyutlarını ve toplam parametre sayısı hesabını kağıtta yaz. Sonra komutta `[128,64]` yerine `[32,16]` koyarak kontrol et. Bir `Linear` katmanında neden `girdi×çıktı + çıktı` parametre var?

## 3. Neden araya ReLU koyuyoruz?

Sadece doğrusal dönüşümleri art arda koymak, bias’larla birlikte yine tek affine dönüşüm olarak birleştirilebilir. Daha çok katman yazmış olmak tek başına daha esnek karar sınırı vermez. `ReLU(z)=max(0,z)` bu birleştirilebilirliği kıran doğrusal olmayan adımdır. Bu taslakta iki hidden layer’ın ikisinden sonra ReLU vardır; çıktı katmanından sonra ReLU yoktur. [S01]

Bir nöronun negatif değerini sıfırlamak görüntünün kendisini silmek değildir; ara özellik skoruna uygulanır. Pozitif bölgede küçük değişimler ileri taşınır; negatif bölgede türev sıfırdır. Öğrenme hızı veya başlangıç nedeniyle sürekli negatif kalan nöronların katkısı sınırlanabilir. Bu durumdan “ReLU her zaman kötü” sonucu çıkmaz; belirtiyi ve deney koşulunu incelemek gerekir.

**Ara aktif hatırlama — cevaplarını komuttan önce yaz:**

1. Flatten sıralamayı korurken MLP neden CNN’deki yerel bağlantı/paylaşılan filtre varsayımını kullanmıyor?
2. İlk hidden boyutunu büyütmek neden giriş görüntüsünün çözünürlüğünü artırmıyor?
3. ReLU’ları çıkarsan üç affine katmanı tek affine dönüşüme hangi cebirsel nedenle indirebilirsin?

Bu ödev CNN’yi yasaklıyor. “Başarımı artırayım” diye Conv2d veya pretrained sınıflandırıcı eklemek hedefi kaçırır; `tests/test_model.py` bu mimari sapmayı yakalar.

## 4. Logits, softmax ve CrossEntropyLoss birbirinin yerine geçmez

Çıkıştaki on değer **logits**’tir: negatif olabilirler, toplamları bir olmak zorunda değildir. En yüksek logit’in indeksi sınıf tahminidir. İnsan için sınıflar arası dağılım göstermek istediğimizde softmax kullanırız. `CrossEntropyLoss` sınıf-indeksli etiketlerle logits bekler; log-softmax ve NLLLoss birleşimine eşdeğer hesap yapar. Etiketler bu projede `long` tamsayılardır; shape `(B,)`, logits shape `(B,10)` olur. [S02]

```bash
"$PY" -c 'import torch; z=torch.tensor([[2.,0.,-1.]]); y=torch.tensor([0]); f=torch.nn.functional.cross_entropy; print(round(f(z,y).item(),6),round(f(z.softmax(1),y).item(),6))'
```

Gerçek küçük probun çıktısı: `0.169846 0.657846`. İkinci hesap çalışır, ancak softmax dağılımını yeniden logits sanarak farklı bir amaç fonksiyonu hesaplar. Hata her zaman Python exception’ı değildir. Tek örnekte CE doğru sınıfa atanan olasılığın negatif logaritmasıdır; yanlış ve çok emin tahminlerin cezası büyüyebilir. Accuracy yalnız argmax’in doğru olup olmadığına bakar; olasılıkların ne kadar değiştiğini göstermez. Bu nedenle accuracy aynı kalırken loss düşebilir.

**Görev:** Bir örneğin doğru sınıfının skoru artıyor ama argmax zaten doğruysa accuracy ne olur? Loss neden değişebilir? Cevabını “çünkü daha iyi” demeden, hesaplanan büyüklükler üzerinden yaz.

## 5. Backprop: model kendi kendine düşünmüyor

`run_epoch` içindeki sıra önemlidir:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

Forward, mevcut parametrelerle skorları hesaplar. Loss, etiketlerle karşılaştırılan skaler amaçtır. `backward()` zincir kuralıyla her öğrenilebilir parametrenin bu amaca yerel etkisini, yani gradient’ini hesaplar. `step()` gradient bilgisini kullanarak parametreyi değiştirir. Gradient bir rakam tahmini değildir; bir parametreyi küçük miktar değiştirmenin kaybı hangi yönde etkilediğinin bilgisidir. Parametre güncellenmezse yalnız gradient hesaplamış olursun.

```bash
"$PY" -c 'import torch; w=torch.tensor([2.],requires_grad=True); loss=(w-5).pow(2).sum(); loss.backward(); print(w.grad.item())'
```

Bu koşuda `-6.0` çıktı. SGD’nin basit fikri `w_yeni = w - lr×gradient` olduğundan negatif gradient bu örnekte w’yi artırır. Gerçek ağda bu hesap çok sayıda matrise dağılır. Adam geçmiş gradient bilgilerini de kullanır; her optimizer için aynı learning rate’in aynı davranışı vereceğini varsayma.

Gradient’ler varsayılan olarak birikebildiğinden `zero_grad` sıradan temizlik değildir: önceki batch’in katkısını istemiyorsak sıfırlamamız gerekir. Validation’da `optimizer` verilmez; `model.train(False)` ve kapalı gradient modu ile ölçüm yapılır. `eval()` tek başına gradient hesabını kapatmaz; bu projede iki davranış da açık seçilir.

**Kısa uygulama:** Yukarıdaki probu iki kez `backward()` çağıran iki ayrı forward ile değiştir; gradient’i sıfırlamazsan ne beklediğini önceden yaz. Bu mini deney checkpoint’e veya gerçek deney verisine dokunmasın.

## 6. Üç veri bölümünün üç farklı sorusu

- **Train:** ağırlıkları hangi örneklerle güncelledim?
- **Validation:** hangi mimari/epoch’u seçtim?
- **Test:** artık seçilmiş modelin daha önce karar vermek için kullanmadığım bölümdeki sonucu nedir?

Torchvision `train=True` ile MNIST eğitim havuzunu, `train=False` ile resmi test havuzunu açar. Biz eğitim havuzunu seed’li permütasyonla 50000 train + 10000 validation olarak böleriz. Smoke limitleri bölme oluşturulduktan sonra her bölümden bir önek seçer; ikisi yine kesişmez. [S03] Bu bir stratified bölme değildir; sınıf dağılımları ayrıca incelenebilir.

Seed’i yalnız rapora yazmak yetmez. Model başlangıcı, veri sırası ve split üretimi kontrol edilmelidir. Kod Python/NumPy/Torch seed’lerini ayarlar, CPU kullanır, thread sayısını iki yapar, indeksleri dosyaya kaydeder. Buna rağmen farklı sürüm/donanımda bit bit aynı sonucu garanti etmeyiz. Yeniden üretilebilirlik “aynı koşulları kaydet ve karşılaştır” disiplinidir; tek sihirli sayı değildir.

Testi her modelden sonra ölçüp en yüksek test accuracy’yi seçersen test, adı değişmese de fiilen validation haline gelir. Bu yüzden `train.py` test dataset’ini hiç yüklemez; ayrı CLI final model seçimi onayı ister. Onay bayrağı bilimsel dürüstlüğü yazılım yoluyla bütünüyle zorlayamaz; kullanım sorumluluğu sende kalır.

## 7. Ölçümde küçük bir batch hatası büyük bir rapor hatasıdır

Bir epoch’un örnek başına kaybı `Σ(batch_loss × batch_örnek_sayısı) / toplam_örnek` olarak hesaplanır. Son batch küçük olabileceğinden batch ortalamalarının düz ortalaması yanlıştır. Accuracy ise `toplam_doğru / toplam_örnek`tir. `tests/test_core.py` iki örnekli batch + tek örnekli son batch ile bu riski sınar. Bunlar CE’nin class weight/ignore_index kullanılmayan, örnek ağırlıkları eşit olan bu taslağı içindir.

Grafikte train loss düşerken validation loss yükselmesi genellemeye ilişkin bir uyarı olabilir; otomatik “kesin overfit” etiketi koymadan eğrinin süresini, oynaklığı ve örnek sayısını incele. Her iki kayıp da yüksekse az eğitim, uygunsuz lr veya kapasite sınırı olası açıklamalardır. Tek grafikten neden kanıtlanmaz.

Kod her iyileşmede en düşük validation loss checkpoint’ini kaydeder. Son epoch en iyi epoch olmak zorunda değildir. `state_dict` tensörleri clone edilir; aynı model güncellenirken en iyi anın ağırlıklarını referans üzerinden yanlışlıkla değiştirmemek hedeflenir. Checkpoint’i tekrar yükleyen test, sonradan ağırlığı değiştirip eski çıktının geri geldiğini sınar.

**Deney sonrası yaz:** En iyi epoch ile son epoch farklı mı? Train accuracy’nin minibatch güncellemeleri sırasında, validation accuracy’nin sabit epoch-sonu ağırlıklarıyla ölçülmesi karşılaştırmanı nasıl etkiler?

## 8. Kendi yazın neden farklı bir problemdir?

Kamerada gölge, kâğıtta çizgi, ters renk, farklı kalınlık ve yanlış crop olabilir. MNIST üzerindeki modelin giriş biçimine yalnız şekil olarak uymak yeterli değildir. 28×28 boyutlu ama siyah rakam/beyaz fonlu bir görüntü, eğitimdeki yoğunluk düzeninin tersidir. `preprocess_image` kenarlardan arka planı tahmin eder, gerekirse ters çevirir, boş alanı kırpar, merkezler ve aynı normalize formülünü kullanır.

Bu adımlar kusursuz bir segmentasyon/algılama sistemi değildir. Fotoğrafta iki rakam varsa kod bunu güvenilir biçimde ayıramaz. **Bir dosya bir rakam** şartını kullanıcı sağlamalıdır. Tam numara görselinden hane kesme otomasyonu yapılmadı. Tamamen şeffaf veya boş görüntünün geçerli tahmin gibi raporlanmaması için hata verilir. Görüntüyü tekrar tekrar değiştirerek kişisel sonucu güzelleştiriyorsan bunu raporda açıkça belirt; ilk ve düzeltilmiş hazırlığı ayır.

En büyük softmax olasılığı yüksek olsa bile dış dağılımdan görüntüde yanlış sınıf seçilebilir. “Model %99 emin” cümlesi yerine “maksimum softmax çıktısı …; gerçek etiket …; sonuç doğru/yanlış” diye yaz. Kendi el yazısı yoksa kişisel bölüm eksiktir; sentetik test çubuğu onun yerine geçmez.

## 9. Testten öğrenci öğrenmesine hangi kanıt eksik?

Testler shape, kurallar ve hesap sözleşmesini denetler. İyi accuracy model davranışı hakkında veri verir. Senin öğrenmene ilişkin kanıt ise kendi açıklaman, değişiklik gerekçen ve sonucu yorumlamandır. Hazır kodun çalışması bu üçüncü kanıtı vermez. Rapor taslağını okuyup aynen aktarmak yerine kendi koşunu ve gerekçeni ekle.

**Kümülatif alıştırma:** A/B deneylerini çalıştır, yalnız validation ile birini seç, seçimi yaz, final testi bir kez ölç, kendi gerçek el yazını her hane için etiketle ve prepared görselleri denetle. Kabul: komut/config/split/seed/sürüm kayıtları; loss/accuracy grafiklerine atıf; testin seçimden sonra geldiğine ilişkin not; en az bir hata veya sınırlılık açıklaması. Bir yanlış tahmini neden kesinlikle optimizer’a bağlayamayacağını tartış.

## 10. Son aktif hatırlama — cevabı henüz yazılmadı

1. `X` shape’i `(64,784)` iken `W1` `(128,784)` ise neden transpoz gerekir?
2. Logits’e softmax ekleyip CE’ye verirsen neden hesaplanan hedef değişir?
3. `backward()` ve `step()` arasındaki görev farkı nedir?
4. Eğitimde test dataset’ini açmamak hangi yanlış kararı önler?
5. Bir epoch’un son batch’i yarım boyuttaysa loss’u nasıl birleştirirsin?
6. En iyi checkpoint’i val accuracy yerine val loss ile seçmek hangi tercihi sabitler?
7. Kendi yazın yanlışsa preprocessing hatası ile distribution shift’i ayırmak için hangi gözlemleri toplarsın?

Cevaplarını kendin yaz; sonra kod üzerinden denetle. Öğrenildi işaretlemesi bu dosyada yapılmaz.

### Küçük terim sözlüğü

**Batch/minibatch:** tek ileri/geri adımda işlenen örnek grubu. **Epoch:** seçilen train bölümünden bir geçiş. **Hidden layer:** giriş ve son sınıf skorları arasındaki öğrenilebilir katman. **Weight/bias:** dönüşüm katsayısı/sabit kaydırma. **Logit:** normalize edilmemiş sınıf skoru. **Gradient:** amacın parametreye göre yerel değişimi. **Learning rate:** optimizer adım ölçeği. **Overfitting:** eğitim örüntülerine uyumun yeni örneklere taşınmaması. **Distribution shift:** geliştirme ve kullanım girdilerinin dağılımlarının farklılaşması. **Checkpoint:** belirli andaki model ve yeniden kurma bilgisi.

Kaynak sınırı: [S01–S03] için `KAYNAKLAR.md`; açıklamaların diğer kısmı bu taslağın kod akışı ve küçük probların yorumudur. Sınırlı scope’ta okunan kaynak, tüm kitap/site okundu anlamına gelmez. İlgili belgeler: `README.md`, `RAPOR.md`, `ODEV.md`.
