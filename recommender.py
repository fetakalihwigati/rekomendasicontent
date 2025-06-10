
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_caption

def recommend_caption(input_text, top_df, vectorizer, X_top, top_n=5):
    input_clean = clean_caption(input_text)
    input_vec = vectorizer.transform([input_clean])
    sims = cosine_similarity(input_vec, X_top).flatten()
    top_indices = sims.argsort()[-top_n:][::-1]
    return top_df.iloc[top_indices][['Caption', 'brand', 'Reactions, Comments & Shares']]
