
import streamlit as st
import pandas as pd
import joblib
from recommender import recommend_caption
from preprocessing import clean_caption

st.set_page_config(page_title="Rekomendasi Caption Mobil", layout="wide")
st.title("🚗 Rekomendasi Caption Digital untuk Brand Mobil")

# Load vectorizer dan data caption top
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
top_df = pd.read_csv("output/top_caption.csv")
top_df['clean_caption'] = top_df['clean_caption'].astype(str)

# Load model prediksi performa
vectorizer_pred = joblib.load("models/vectorizer_model.joblib")
model_pred = joblib.load("models/performance_model.joblib")

# Dropdown brand
brand_options = ["Semua"] + sorted(top_df['brand'].dropna().unique())
selected_brand = st.selectbox("Filter berdasarkan brand:", brand_options)

if selected_brand != "Semua":
    filtered_df = top_df[top_df['brand'] == selected_brand]
    X_top = vectorizer.transform(filtered_df['clean_caption'])
else:
    filtered_df = top_df
    X_top = vectorizer.transform(top_df['clean_caption'])

# Input caption
caption_input = st.text_area("Masukkan caption baru:")

if st.button("Rekomendasikan"):
    if caption_input.strip() != "":
        # Prediksi performa
        clean_input = clean_caption(caption_input)
        input_vec = vectorizer_pred.transform([clean_input])
        pred = model_pred.predict_proba(input_vec)[0][1]
        st.markdown(f"### 🔮 Prediksi Performa: `{pred:.2%}` kemungkinan akan perform tinggi")

        # Rekomendasi caption
        results = recommend_caption(caption_input, filtered_df, vectorizer, X_top)
        st.write("### 💡 Caption Mirip Berperforma Tinggi")
        st.dataframe(results)
    else:
        st.warning("Caption tidak boleh kosong.")
