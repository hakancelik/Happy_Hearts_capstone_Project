import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükleme (örnek olarak CSV dosyası kullanılıyor)
df = pd.read_csv("PROJE/data/heart.csv")

# Veri seti bilgileri
print("="*40)
print(f"Heart Disease Data :\n\t\t{df.shape[0]} Number of Rows \n\t\t&\n\t\t{df.shape[1]} Number of Columns")
print("="*40)

# Target Görselleştirme
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

df["target"].replace({0: "No Heart Disease", 1: "Heart Disease"}).value_counts().plot(
    kind="pie", colors=["salmon", "lightblue"], ax=axes[0], explode=[0, 0.1], autopct='%1.1f%%', shadow=True
)
axes[0].set_ylabel('')
df["target"].replace({0: "No Heart Disease", 1: "Heart Disease"}).value_counts().plot(
    kind="bar", ax=axes[1], color=["salmon", "lightblue"]
)
axes[1].set_ylabel('')
axes[1].set_xlabel('')
plt.show()

# Cinsiyet ve target arasındaki ilişki
plt.figure(figsize=(15, 8))
pd.crosstab(df.sex, df.target).plot(kind="bar", figsize=(15, 8), color=["salmon", "lightblue"])
plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Disease, 1= Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0)
plt.show()

# Özellik mühendisliği
df['age_max_heart_rate_ratio'] = df['age'] / df['thalach']
df['age_range'] = pd.cut(df['age'], bins=[29, 39, 49, 59, 69, 79], labels=['30-39', '40-49', '50-59', '60-69', '70-79'])
df['cholesterol_hdl_ratio'] = df['chol'] / df['thalach']
df['heart_rate_reserve'] = df['thalach'] - df['trestbps']

# EDA (Veri Keşfi)
sns.scatterplot(x='age_max_heart_rate_ratio', y='target', data=df)
plt.show()

# Yeni oluşturulan özellikleri inceleme
plt.figure(figsize=(15, 8))
sns.boxplot(x='age_range', y='thalach', hue='target', data=df)
plt.show()


# Özellik mühendisliği: Yeni özellikler oluşturma
df['age_max_heart_rate_ratio'] = df['age'] / df['thalach']
df['age_range'] = pd.cut(df['age'], bins=[29, 39, 49, 59, 69, 79], labels=[1, 2, 3, 4, 5])  # Kategorik veriyi nümerik hale getirme
df['cholesterol_hdl_ratio'] = df['chol'] / df['thalach']
df['heart_rate_reserve'] = df['thalach'] - df['trestbps']

# Özellikler ve hedef değişkeni tanımlama
X = df.drop(['target', 'age_range'], axis=1)  # 'target' sütununu ve 'age_range' sütununu özelliklerden çıkar
y = df['target']  # 'target' sütunu hedef değişken

# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestClassifier kullanarak modeli eğitme
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Modeli test seti üzerinde değerlendirme
y_pred = model.predict(X_test)

# Model performansını değerlendirme
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


from sklearn.model_selection import GridSearchCV

# Hiperparametre aralıkları
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# En iyi hiperparametreleri görelim
print("Best Hyperparameters:", grid_search.best_params_)

# En iyi modeli seçelim
best_model = grid_search.best_estimator_

# En iyi modeli test seti üzerinde değerlendirelim
y_pred_best = best_model.predict(X_test)

# Model performansını değerlendirme
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

feature_importances = best_model.feature_importances_
feature_names = X_train.columns

# Özellik önem sıralamasını görselleştirme
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=feature_names, orient='h', palette='viridis')
plt.title("Feature Importances")
plt.show()



errors = X_test[y_test != y_pred_best]

# Hatalı sınıflandırılan örneklerin özelliklerini inceleme
print(errors)


###ROC EĞRİSİ##
from sklearn.metrics import roc_curve, roc_auc_score

# Olasılık skorlarını alalım
y_probs = best_model.predict_proba(X_test)[:, 1]

# ROC eğrisini çizme
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random', color='red')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# AUC skoru
auc_score = roc_auc_score(y_test, y_probs)
print("AUC Score:", auc_score)

import joblib

# Modeli kaydetme
joblib.dump(model, 'heart_disease_model.joblib')


import joblib
import pandas as pd

# Kaydedilmiş modeli yükleme
loaded_model = joblib.load('heart_disease_model.joblib')

# Kullanıcının giriş verilerini alın
age = int(input("Age: "))
sex_Male = int(input("Sex (1 for Male, 0 for Female): "))
cp = int(input("Chest Pain Type (0-3): "))
trestbps = int(input("Resting Blood Pressure (mm Hg): "))
chol = int(input("Serum Cholesterol (mg/dl): "))
fbs = int(input("Fasting Blood Sugar (> 120 mg/dl, 1=True, 0=False): "))
restecg = int(input("Resting Electrocardiographic Results (0-2): "))
thalach = int(input("Maximum Heart Rate Achieved: "))
exang = int(input("Exercise Induced Angina (1=Yes, 0=No): "))
oldpeak = float(input("ST Depression Induced by Exercise Relative to Rest: "))
slope = int(input("Slope of the Peak Exercise ST Segment (0-2): "))
ca = int(input("Number of Major Vessels Colored by Fluoroscopy (0-3): "))
thal = int(input("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect): "))

# Kullanıcının girişini modele uygun formata getirme
input_data = pd.DataFrame({
    'age': [age],
    'sex_Male': [sex_Male],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Modelin beklediği özellik adlarını kontrol et
expected_features = set(input_data.columns)
if set(expected_features) != set(loaded_model.feature_names_in_):
    raise ValueError("The feature names do not match those that were passed during fit.")

# Tahmin yapma
prediction = loaded_model.predict(input_data)

# Sonucu gösterme
if prediction[0] == 0:
    print("Prediction: No Heart Disease")
else:
    print("Prediction: Heart Disease")