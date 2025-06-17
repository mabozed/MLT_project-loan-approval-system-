# نظام التنبؤ بالموافقة على القروض
## Loan Approval Prediction System

### 🌟 نظرة عامة | Overview
نظام ذكي يستخدم تقنيات التعلم الآلي للتنبؤ بالموافقة على القروض. يعتمد النظام على تحليل بيانات المتقدمين للقروض وتاريخهم الائتماني لاتخاذ قرارات دقيقة ومتوازنة.

### 📊 المميزات | Features
- معالجة متقدمة للبيانات
- تحليل شامل للمتغيرات
- مقارنة بين عدة نماذج للتعلم الآلي
- تقييم دقيق لأداء النماذج
- واجهة تقارير احترافية

### 🛠️ التقنيات المستخدمة | Technologies
- Python 3.8+
- scikit-learn
- XGBoost
- pandas
- numpy
- matplotlib
- seaborn
- python-docx

### 📋 متطلبات النظام | Requirements
```bash
pip install -r requirements.txt
```

### 🚀 كيفية التشغيل | How to Run
1. تحضير البيانات:
```bash
python data_preprocessing.py
```

2. تدريب النماذج:
```bash
python model_training.py
```

3. إنشاء التقرير:
```bash
python create_report.py
```

### 📁 هيكل المشروع | Project Structure
```
├── data_preprocessing.py     # معالجة البيانات
├── model_training.py        # تدريب النماذج
├── create_report.py         # إنشاء التقرير
├── requirements.txt         # المكتبات المطلوبة
├── loan_prediction.csv      # البيانات الأصلية
├── processed_loan_data.csv  # البيانات المعالجة
└── best_loan_model.joblib   # النموذج المدرب
```

### 📈 النتائج | Results
- دقة النموذج: 87%
- أفضل نموذج: Random Forest
- معدل F1: 86%

### 🔍 المميزات المستخدمة | Features Used
- معلومات شخصية (الجنس، الحالة الاجتماعية، عدد المعالين)
- معلومات مالية (الدخل، دخل الشريك، مبلغ القرض)
- معلومات عن الممتلكات
- التاريخ الائتماني

### 📊 النماذج المستخدمة | Models Used
1. Random Forest
2. Gradient Boosting
3. XGBoost
4. Logistic Regression

### 🎯 الاستخدام | Usage
```python
# مثال على استخدام النموذج للتنبؤ
import joblib

# تحميل النموذج
model = joblib.load('best_loan_model.joblib')

# التنبؤ
prediction = model.predict(data)
```

