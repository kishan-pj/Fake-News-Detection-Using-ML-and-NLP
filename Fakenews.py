import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------------------
# Load TF-IDF vectorizer and trained model
# ----------------------------
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    with open('best_fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"❌ Error loading model or vectorizer: {e}")
    st.stop()

# ----------------------------
# NLTK setup (no downloads here)
# ----------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ----------------------------
# Text preprocessing function (no punkt)
# ----------------------------
def transform_text(text):
    text = str(text).lower()
    tokens = re.findall(r'\b\w+\b', text)  # regex-based tokenization
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    tokens = [ps.stem(word) for word in tokens]  # stemming
    return " ".join(tokens)

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection")

input_content = st.text_area("Enter news article/content:")

if st.button("Predict"):
    if input_content.strip() == "":
        st.warning("⚠️ Please enter a valid news article or content.")
    else:
        transformed_content = transform_text(input_content)
        vector_input = tfidf.transform([transformed_content])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.success("✅ This news is TRUE.")
        else:
            st.error("⚠️ This news is FAKE.")
