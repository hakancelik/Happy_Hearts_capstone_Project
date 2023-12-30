import streamlit as st
import pandas as pd
import numpy as np
from copy import deepcopy
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("PROJE/heartfeature.csv")

# Model dosya yollarını sabit olarak tanımla
MODEL_PATHS = {
    "Logistic Regression": "PROJE/logistic_regression_best_model.joblib",
    "K-Nearest Neighbors": "PROJE/k-nearest_neighbors_best_model.joblib",
    "Decision Tree": "PROJE/decision_tree_best_model.joblib",
    "Random Forest": "PROJE/random_forest_best_model.joblib",
    "Gradient Boosting": "PROJE/gradient_boosting_best_model.joblib",
    "XGBoost": "PROJE/xgboost_best_model.joblib",
}

# Modelleri yükleme fonksiyonu
import builtins

@st.cache(hash_funcs={builtins.dict: lambda _: None})
def load_models():
    return {name: joblib.load(model_path) for name, model_path in MODEL_PATHS.items()}

# Tahmin fonksiyonu
def predict(model, input_data):
    loaded_model = model
    prediction = loaded_model.predict(input_data)
    return prediction

# Streamlit arayüzü
st.set_page_config(
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modelleri yükle
models = load_models()

# Ana uygulama fonksiyonu
def main():
    gif_path = "kalp.gif"

    # Sidebar'a GIF'i ekleyin
    st.sidebar.image(gif_path, use_column_width=True, caption="")

    # Sol tarafta menü oluşturma
    menu = ["Ana Sayfa", "Görselleştirme Sayfası", "Sunum Sayfası", "Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Ana Sayfa":
        home()

    elif choice == "Görselleştirme Sayfası":
        visualization()

    elif choice == "Sunum Sayfası":
        presentation()

    elif choice == "Predict":
        # Kullanıcıdan giriş verilerini al
        input_data, sex = get_user_input()

        # Model seçme kutusu
        selected_model = st.sidebar.selectbox("Select a model", list(MODEL_PATHS.keys()), key="model_selectbox")

        # İlk tahminden önce orijinal modelin kopyasını al
        initial_model = deepcopy(models[selected_model])

        # Tahmin sonucunu sıfırlamak için değişken
        prediction_result = None

        if st.button("Predict", key="predict_button"):
            # Model tahmini yapma
            prediction = predict(models[selected_model], input_data)

            # Tahmin sonucunu sakla
            prediction_result = "KALP RAHATSIZLIĞI YOK 💖" if prediction[0] == 0 else "KALP RAHATISZLIĞI VAR 💔"

            # Tahmin sonucunu ekrana yazdırma
            if prediction_result == "KALP RAHATISZLIĞI VAR 💔":
                st.error(f"Kullanılan Model {selected_model}: {prediction_result}")
                # Cinsiyete göre resim göster
                if sex == "Kadın":
                    st.image("kadin.jpg", caption="", use_column_width=False)
                elif sex == "Erkek":
                    st.image("erkek.jpg", caption="", use_column_width=False)
            else:
                st.success(f"Kullanılan Model {selected_model}: {prediction_result}")

                # No Heart Disease durumunda kalp simgesi gösterme ve balonları ekleme
                if prediction_result == "KALP RAHATSIZLIĞI YOK 💖":
                    heart_image_path = "health.jpg"  # Kalp simgesinin gerçek yolunu belirtin
                    if os.path.exists(heart_image_path):
                        st.image(heart_image_path, caption="", use_column_width=False)
                        st.balloons()
                    else:
                        st.warning("Warning: Heart image not found at the specified path.")

            # Modeli ve tahmin sonucunu sıfırla
            models[selected_model] = initial_model
            prediction_result = None

# ... Diğer kodlar ...

def get_user_input():
    age = st.slider("Yaş:", min_value=29, max_value=79, value=40)
    sex = st.radio("Cinsiyet:", options=["Kadın", "Erkek"])
    sex_Male = 1 if sex == "Erkek" else 0
    cp = st.slider("Göğüs Ağrısı Tipleri (0 : Kan akışının azalmasından kaynaklı - 1 : Kalbe bağlı olmayan ağrı - 2 : Yemek borusu spazmından kaynaklı  3 : Hastalık belirtisi göstermeyen ağrı):", min_value=0, max_value=3, value=1)
    trestbps = st.slider("İstirahat Kan Basıncı (mm Hg):", min_value=90, max_value=200, value=120)
    chol = st.slider("Kolesterol (mg/dl):", min_value=50, max_value=600, value=200)
    fbs = st.radio("Açlık Kan Şekeri (> 120 mg/dl):", options=["Hayır", "Evet"])
    fbs = 1 if fbs == "Evet" else 0
    restecg = st.slider("İstirahat EKG Sonuçlar (0 : Normal - 1 : Orta - 2 : Yüksek):", min_value=0, max_value=2, value=1)
    thalach = st.slider("Maksimum Kalp Atış Hızı:", min_value=70, max_value=220, value=150)
    exang = st.radio("Egzersize Bağlı Göğüs Ağrısı (0 : Göğüs ağrısı yok - 1 : Göğüs ağrısı var):", options=["Hayır", "Evet"])
    exang = 1 if exang == "Evet" else 0
    oldpeak = st.slider("Egzersizle ST Depresyonu (İskemi; kan akışının zayıflaması):", min_value=0.0, max_value=6.2, value=0.0)
    slope = st.slider("Egzersiz EKG esnasında ST Segmentinin Eğimi (0 : Eğim Düz - 1 : Eğim yavaş artan - 2 : Eğim hızlı artan):", min_value=0, max_value=2, value=1)
    ca = st.slider(" Damarlardaki Kalsiyum Birikimi (0 : Yok - 1 : Çok az - 2 : Orta - 3 : yüksek ):", min_value=0, max_value=3, value=0)
    thal = st.slider("Thalassemi, kandaki hemoglobin proteini etkilenme (1 : Az - 2 : Orta - 3 : Yüksek)):", min_value=1, max_value=3, value=2)

    # Kullanıcının girişini modele uygun formata getir
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_Male],
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
        'thal': [thal],
        'age_max_heart_rate_ratio': [age / thalach],
        'cholesterol_hdl_ratio': [chol / thalach],
        'heart_rate_reserve': [thalach - trestbps]
    })
    return input_data, sex

# Uygulamayı başlat
if __name__ == '__main__':
    main()
