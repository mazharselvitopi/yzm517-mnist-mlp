# Doğrulama kaydı — 28 Ekim 2026

## Gerçek çalıştırmalar

- Ayrı teslim reposunda: `/home/ubuntu/.local/share/education-venv/bin/python -m pytest -q` → **14 passed in 13.74s**.
- İki eğitim: 12’şer epoch, 50.000 train + 10.000 validation; `run_kind=experiment`, eğitimde `test_accessed=false`.
- İki koşunun split indeksleri birebir aynı; train/validation kesişimi yok.
- Her koşunun `best.pt` checkpoint’inin epoch ve val_loss’u, history içindeki minimum validation loss ile eşleşti. A’da epoch 9, B’de epoch 7; son epoch 12 olduğu için gerçek koşularda minimumun son epoch dışında saklandığı da doğrulandı.
- Seçim testten önce `results/selection.json` içine kaydedildi: B, minimum validation loss.
- Final test: **test_loss=0.077727; test_accuracy=0.9780; samples=10000**.
- Confusion matrix 10×10; toplam 10.000, köşegen toplamı 9.780; test metriğiyle eşleşti.
- Kişisel deney: 9 tahmin / 9 insan etiketi; 8 doğru, bir yanlış. Hiçbir rakam yapay üretilmedi.
- Repo içindeki `handwriting/` kırpmaları ve `results/deney-b/best.pt` ile tahmin yeniden çalıştırıldı: 9 tahminin tamamı, tüm softmax dağılımları ve hazırlanmış PNG baytları ilk kişisel koşuyla birebir eşleşti.
- `requirements.txt` içindeki altı paket sürümü gerçek eğitim ortamındaki sürümlerle karşılaştırıldı; tamamı eşleşti.

## İnceleme ve sınırlar

- Beş Python modülü ile testler ayrı bir ajan tarafından salt-okunur incelendi; güvenlik veya sonuç geçerliliğini bozan mantık hatası bildirilmedi (`evidence/independent-review.json`).
- Özel ham çalışma dizinleri hariç 69 teslim adayı dosyada GitHub token / private key / shell injection örüntü taraması yapıldı; eşleşme yok. Bu sınırlı tarama tüm olası güvenlik sorunlarının yokluğu garantisi değildir.
- Markdown bağlantıları ve grafik hedefleri dosya sistemi üzerinde kontrol edildi; kırık hedef bulunmadı. En uzun göreli yol 40, en uzun dosya adı 24 karakter; repository yol limitleri sağlandı.
- Orijinal el yazısı ve hazırlanan dokuz rakam montajı görsel olarak incelendi. Altıncı hanedeki yanlış 7→4 tahmini raporda aynen korundu.
- Bu sunucudaki test/CPU çalıştırması doğrulandı. Mac/Windows/Colab kurulumu ve GitHub Actions CI ayrıca çalıştırılmış değildir.
- MNIST verisi repo dışındadır; veri arşivleri ve yarım kalmış B koşusu teslim reposuna eklenmedi.
- Öğrenme yeterliği veya Classroom teslimi bu teknik doğrulamayla kanıtlanmaz.
