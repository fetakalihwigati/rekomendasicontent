
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from preprocessing import clean_caption

df = pd.read_excel("data/data_content.xlsx")
df = df[['Caption', 'brand', 'Reactions, Comments & Shares']].dropna(subset=['Caption'])
df['clean_caption'] = df['Caption'].apply(clean_caption)
df = df.drop_duplicates()

threshold = np.percentile(df['Reactions, Comments & Shares'], 90)
top_df = df[df['Reactions, Comments & Shares'] >= threshold]

top_df.to_csv("output/top_caption.csv", index=False)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
vectorizer.fit(top_df['clean_caption'])
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
