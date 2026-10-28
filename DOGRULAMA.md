# Test ve doğrulama kayıtları

Bu dosyada çalışmanın test sonuçlarını ve kayıtlı deney dosyaları için yapılan tutarlılık kontrollerini özetledim. Kod ve deneylerde olduğu gibi bu kontrollerde de yapay zekâ desteği kullandım.

## Kod testleri

Repo kökünde ve sanal ortam etkinken:

```bash
python -m pytest -q
```

Test paketindeki **14 test geçti**. Testler; MLP mimarisini, ayrık veri bölmesini, örnek sayısıyla ağırlıklandırılmış metrikleri, checkpoint kaydını, komut satırı arayüzlerini ve görüntü ön işlemeyi kontrol ediyor. Küçük sentetik test girdileri, gerçek el yazısı veya MNIST başarımı olarak sunulmuyor.

## Eğitim kayıtlarının tutarlılığı

- İki deney de 12 epoch, 50.000 train ve 10.000 validation örneğiyle tamamlandı.
- İki deneyin `split.json` dosyalarındaki indeksler birebir aynı; train ve validation kümeleri kesişmiyor.
- Eğitim kayıtlarında `run_kind=experiment` ve `test_accessed=false` değerleri bulunuyor.
- Her iki `best.pt` dosyasındaki epoch ve validation loss, ilgili eğitim geçmişindeki minimum validation loss ile eşleşiyor.
- A modelinde en iyi epoch 9, B modelinde 7; her ikisinde de son epoch 12. Dolayısıyla kaydedilen modelin yalnızca son epoch olmadığı da doğrulanabiliyor.
- Testten önce yazılan [seçim kaydı](results/selection.json), B modelinin minimum validation loss ölçütüyle seçildiğini gösteriyor.

## Test kümesi sonuçları

| Kontrol | Sonuç |
|---|---:|
| Test örneği | 10.000 |
| Test loss | 0.077727 |
| Test accuracy | %97.80 |
| Confusion matrix boyutu | 10 × 10 |
| Matrisin toplamı | 10.000 |
| Köşegen toplamı | 9.780 |

Köşegen toplamı, kaydedilen accuracy ile tutarlı. Ölçüm dosyasındaki SHA-256 değeri de kullanılan `results/deney-b/best.pt` dosyasıyla eşleşiyor.

## El yazısı sonuçları

Dokuz hane için dokuz gerçek etiket ve dokuz tahmin kaydı bulunuyor. Bunların sekizi doğru. Altıncı hanedeki 7→4 hatası hem sonuç dosyasında hem raporda aynı şekilde yer alıyor.

Repo içindeki `handwriting/` görüntüleri ile seçilen checkpoint kullanılarak tahmin yeniden çalıştırıldığında, ilk çalıştırmadaki dokuz tahminin tamamı ve tüm softmax değerleri birebir eşleşti. Hazırlanmış PNG dosyalarının baytları da aynı çıktı.

Orijinal fotoğraf ve hazırlanmış haneler görsel olarak kontrol edildi. Orijinal görüntü korunuyor; rakamlar modelin tahminine göre yeniden çizilmedi veya değiştirilmedi.

## Belgeler ve ortam

- `requirements.txt` içindeki altı doğrudan bağımlılığın sürümü, deney ortamındaki paketlerle eşleşiyor.
- Belgelerdeki göreli bağlantılar ve görseller dosya sistemi üzerinde kontrol edildi.
- Model ağırlıkları ve ham sonuçlar belge düzenlemeleri sırasında korunuyor.
- MNIST veri arşivleri Git deposunun dışında tutuluyor.
- Kod inceleme kayıtları `evidence/independent-review.json` ve `evidence/delivery-audit.json` içinde bulunuyor. Bunlar yapay zekâ destekli teknik inceleme kayıtlarıdır; ders notu veya deney ölçümü değildir.

Doğrulanan ortam Linux ARM64 ve CPU'dur. Mac, Windows, Colab kurulumu ve GitHub Actions çalıştırması ayrıca doğrulanmamıştır. Testlerin geçmesi, tüm olası hataların veya güvenlik sorunlarının bulunmadığı anlamına gelmez.
