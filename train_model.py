
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from preprocessing import clean_caption

df = pd.read_excel("data/data_content.xlsx")
df = df[['Caption', 'Reactions, Comments & Shares']].dropna(subset=['Caption'])
df['clean_caption'] = df['Caption'].apply(clean_caption)
df = df.drop_duplicates()

threshold = df['Reactions, Comments & Shares'].median()
df['label'] = (df['Reactions, Comments & Shares'] >= threshold).astype(int)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df['clean_caption'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

joblib.dump(vectorizer, "models/vectorizer_model.joblib")
joblib.dump(clf, "models/performance_model.joblib")
