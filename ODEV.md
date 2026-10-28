# YZM517 — Ödev 1: asıl yönerge metni

Kaynak: Classroom, 22 Ekim 2026 anlık görüntüsü. Son tarih: 28 Ekim 2026 23.59 Europe/Istanbul.

ÖDEV: MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma

Bu ödevin amacı, MNIST el yazısı rakam veri seti kullanılarak bir Multi-Layer Perceptron (MLP) modeli geliştirmek, modeli eğitmek ve model yapılandırmalarının performansını karşılaştırmaktır.

Öğrenciler ayrıca kendi el yazılarıyla Öğrenci okul numaralarını yazıp modele vererek, eğitilen modelin gerçek hayattaki el yazıları üzerindeki başarısını değerlendirecektir.

Ödev kapsamında PyTorch, TensorFlow/Keras veya benzeri derin öğrenme kütüphaneleri kullanılabilir.


2. Kullanılacak Veri Seti

Deneylerde MNIST veri seti kullanılacaktır.


0–9 arasındaki el yazısı rakamlardan oluşur.
Görüntüler 28 × 28 piksel boyutundadır.
Her görüntü gri seviyelidir.
Toplam 10 sınıf vardır: `0, 1, ..., 9`.

Not: CNN kullanılmayacaktır. Bu ödevde temel model olarak MLP kullanılmalıdır.

Deney 1 – Temel MLP Modeli

İlk deneyde temel bir MLP modeli oluşturulacaktır.

En az 2 hidden layer kullanılmalıdır.
Aktivasyon fonksiyonu olarak ReLU kullanılmalıdır.
Loss fonksiyonu olarak uygun bir sınıflandırma kayıp fonksiyonu kullanılmalıdır.
Optimizer olarak SGD veya Adam kullanılabilir.
Train, validation ve test veri kümeleri oluşturulmalıdır.

Öğrenciler aşağıdaki sonuçları raporlamalıdır:

Training loss
Validation loss
Training accuracy
Validation accuracy
Test accuracy
Eğitim süresi

Ayrıca eğitim sürecindeki loss ve accuracy grafiklerini göstermelidir.



Kendi El Yazısı Görüntülerinin Hazırlanması

Kendi el yazınızla oluşturduğunuz görüntüler MNIST görüntülerine mümkün olduğunca benzer şekilde hazırlanmalıdır.

Görüntüler:

Gri tonlamalı olmalıdır.
28 × 28 piksele dönüştürülmelidir.
Arka plan ve rakam renkleri MNIST formatıyla uyumlu hale getirilmelidir.
Normalizasyon uygulanmalıdır.


Teslim Edilecekler

Ödev teslimi için GitHub veya benzeri platformlarda yer alan proje deposunun (repository) bağlantısını paylaşmanız yeterlidir.
Repo içerisinde proje kodlarının yanı sıra, kendi el yazınızla yazdığınızı öğrenci numaranızın görselleri ve deney sonucu hakkında genel değerlendirmenizin bulunması gerekmektedir. 


Önemli Kurallar

1. CNN kullanılmayacaktır. Ödevin temel amacı MLP mimarisini anlamaktır.
2. Hazır eğitilmiş bir model kullanılmayacaktır.
3. Model sıfırdan eğitilmelidir.
4. Train, validation ve test ayrımı açıkça gösterilmelidir.
5. Test verisi model geliştirme sürecinde kullanılmamalıdır.
6. Kendi el yazısı görüntüleri öğrenci tarafından oluşturulmalıdır.
7. Kullanılan hiperparametreler raporda belirtilmelidir.
8. Kodun çalıştırılabilir olması gerekmektedir.
9. Öğrenci yaptığı deneyleri ve elde ettiği sonuçları açıklayabilmelidir.

Beklenen Öğrenme Çıktıları

Ödev sonunda öğrencinin:

MNIST veri setini kullanabilmesi,
MLP mimarisi oluşturabilmesi,
Train/validation/test ayrımını anlayabilmesi,
Loss ve accuracy kavramlarını yorumlayabilmesi,
Hyperparameter değişikliklerinin etkisini analiz edebilmesi,
Model performansını değerlendirebilmesi,
Kendi verisini modele uygulayabilmesi,
Training data ile gerçek dünya verisi arasındaki farkı (distribution shift/generalization) gözlemleyebilmesi


beklenmektedir.
