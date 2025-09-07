import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------------------
# Load TF-IDF vectorizer and trained model
# ----------------------------
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('best_fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ----------------------------
# NLTK setup
# ----------------------------
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ----------------------------
# Text preprocessing function
# ----------------------------
def transform_text(text):
    text = str(text).lower()                 # Lowercase
    tokens = nltk.word_tokenize(text)        # Tokenize
    tokens = [word for word in tokens if word.isalnum()]     # Remove non-alphanumeric
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [ps.stem(word) for word in tokens]              # Stemming
    return " ".join(tokens)

# ----------------------------
# Streamlit App UI
# ----------------------------
st.title("📰 Fake News Detection")

input_content = st.text_area("Enter news article/content:")

if st.button("Predict"):
    if input_content.strip() == "":
        st.warning("⚠️ Please enter a valid news article or content.")
    else:
        # Preprocess the input
        transformed_content = transform_text(input_content)
        vector_input = tfidf.transform([transformed_content])
        
        # Predict
        result = model.predict(vector_input)[0]

        # Show prediction
        if result == 1:
            st.success("✅ This news is TRUE.")
        else:
            st.error("⚠️ This news is FAKE.")

