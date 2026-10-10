# AIMV — AI Media Verifier

نظام لتدريب نماذج تكشف هل الصورة أو المستند **حقيقي** أم **مولّد أو معدّل بالذكاء الاصطناعي**
(Gemini، ChatGPT، Grok، Midjourney، FLUX، ...)، مع تركيز على **البيانات الليبية**:
الإيصالات المصرفية، البطاقات الشخصية، الجوازات، الشهادات، كتيب العائلة، والصور المتداولة على مواقع التواصل.

> ⚠️ لا يوجد كاشف دقيق 100%. النتيجة احتمالية، وتصدر مع درجة ثقة وخريطة للمناطق المشبوهة.
> في القرارات المهمة (رفض إيصال، اتهام شخص) يجب أن يراجعها إنسان.

## كيف يعمل

| المكوّن | الوظيفة |
|---|---|
| **نموذج الآثار** (`photos_artifact`, `documents`) | شبكة ConvNeXt/EfficientNetV2 على مقاطع بالدقة الأصلية، مع قنوات بقايا الضوضاء SRM، فتكشف بصمات المولّدات والتعديل الموضعي (مثل تغيير رقم في إيصال) |
| **النموذج الدلالي** (`photos_clip`) | خصائص CLIP ViT-L/14، يُدرَّب منها آخر 4 طبقات. هو الأفضل في كشف أدوات لم يرها النموذج أثناء التدريب |
| **التقييم بالمربعات** | الصور الكبيرة لا تُصغَّر، بل تُقسَّم إلى مربعات بدقتها الأصلية، ويُخرَج احتمال كلي وخريطة حرارية للمناطق المشبوهة |
| **الدمج (Ensemble)** | يجمع عدة نماذج بعد معايرة كل نموذج (temperature scaling) |
| **فحص البيانات الوصفية** | EXIF، ووسوم IPTC `trainedAlgorithmicMedia`، وC2PA، وإعدادات ComfyUI/A1111. هذه قرائن فقط، لأن واتساب يحذفها |

### ما الذي يجعله "احترافياً"
- **محاكاة الواقع**: ضغط JPEG متكرر بأسلوب واتساب وفيسبوك، وتغيير الحجم، وتصوير الشاشة، والتشويش، ويُطبَّق كل ذلك على **الصنفين معاً**.
- **منع الطرق المختصرة**: كل صورة يُعاد ضغطها بـ JPEG (حتى لا يتعلّم النموذج أن PNG يعني مزيّف)، وتُلصق علامات مائية شبيهة بنجمة Gemini على الصنفين.
- **لا تسرّب بين المجموعات**: الأصل ونسخه المعدّلة ونسخه المتشابهة (perceptual hash) تبقى كلها في نفس المجموعة (train/val/test).
- **اختبار التعميم**: يمكن حجب أداة توليد كاملة للاختبار فقط (`--holdout-generators`).
- **نقطة تشغيل مضبوطة**: يُختار الحد بحيث لا تتجاوز نسبة الصور الحقيقية المتهمة خطأً 1% (و0.5% للمستندات).
- **تقنيات التدريب**: AMP، EMA، cosine LR مع warmup، معدل تعلّم أقل للعمود الفقري، label smoothing، إيقاف مبكر، وعيّنات متوازنة لكل (فئة × صنف).
- **تقرير تقييم مفصّل**: AUC وAP وEER، لكل فئة ولكل أداة توليد، وتحت ضغط واتساب وفيسبوك.

## التثبيت

```bash
pip install -r requirements.txt
```

التدريب الحقيقي يحتاج GPU (مثل RTX 3090/4090 أو A100، أو Colab وKaggle المجانيين).

## خطوات التدريب

**1. اجمع البيانات** حسب [`docs/LIBYAN_DATA_GUIDE.md`](docs/LIBYAN_DATA_GUIDE.md):

```
data/raw/real/receipt/moamalat/IMG_0042.jpg
data/raw/fake/receipt/gemini/IMG_0042__amount.jpg     # تعديل للأصل (نفس الاسم + __)
data/raw/real/photo/whatsapp/...
data/raw/fake/photo/midjourney/...
```

**2. (اختياري) أنشئ تعديلات تلقائية على المستندات:**

```bash
python scripts/make_tamper_fakes.py --real data/raw/real/receipt --out data/raw/fake/receipt --per-image 2
```

**3. ابنِ ملف البيانات (manifest)** مع التقسيم:

```bash
python scripts/build_manifest.py --root data/raw --out data/manifest.csv --val 0.1 --test 0.1 \
    --holdout-generators grok
```

**4. درّب النماذج:**

```bash
python -m aimv.train --config configs/documents.yaml        # المستندات الليبية
python -m aimv.train --config configs/photos_artifact.yaml  # الصور: الآثار
python -m aimv.train --config configs/photos_clip.yaml      # الصور: الدلالي (CLIP)
# لتغيير أي إعداد من سطر الأوامر:
python -m aimv.train --config configs/documents.yaml --set train.epochs=40 train.batch_size=16
```

**5. قيّم النماذج:**

```bash
python -m aimv.evaluate --ckpt runs/documents/best.pt --split test --out runs/documents/eval.json
```

**6. افحص صوراً جديدة:**

```bash
python -m aimv.predict --ckpt runs/photos_clip/best.pt runs/photos_artifact/best.pt \
    --input suspicious/ --out report.json --heatmaps heatmaps/
python -m aimv.predict --ckpt runs/documents/best.pt --input receipt.jpg --heatmaps heatmaps/
```

مثال للمخرجات:

```
0.973  AI-generated / AI-edited     receipt.jpg   [no useful metadata (normal for images sent via WhatsApp/Facebook)]
0.041  likely real                  exam.jpg      [camera EXIF present (samsung SM-A546E) — weak evidence of a real capture]
```

## الاختبارات

```bash
pytest -q        # اختبار شامل على بيانات اصطناعية: بناء البيانات ← التدريب ← التقييم ← الفحص، على CPU
```

## هيكل المشروع

```
aimv/
  config.py      إعدادات YAML وتعديلها من سطر الأوامر
  augment.py     محاكاة واتساب وفيسبوك وتصوير الشاشة، ومنع الطرق المختصرة
  data.py        manifest، والبيانات، والعيّنات المتوازنة
  models.py      Detector: عمود فقري من timm مع قنوات SRM، وتجميد جزئي لـ CLIP وDINOv2
  metrics.py     AUC/AP/EER، والمعايرة، والحد عند نسبة إيجابيات كاذبة محددة
  train.py       حلقة التدريب
  evaluate.py    تقرير مفصّل ومقاومة الضغط
  inference.py   التقييم بالمربعات، والخريطة الحرارية، والدمج
  forensics.py   فحص البيانات الوصفية والعلامات
  predict.py     واجهة الفحص
scripts/
  build_manifest.py     بناء البيانات، وإزالة التكرار، والتقسيم دون تسرّب
  make_tamper_fakes.py  تعديلات موضعية تلقائية على المستندات الحقيقية
configs/                documents / photos_artifact / photos_clip / debug_cpu
docs/LIBYAN_DATA_GUIDE.md
```
