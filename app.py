import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

st.set_page_config(page_title="Fake News Detection System")

st.title("Fake News Detection System")
st.write("Paste a news article or headline below to check if it is real or fake.")

news_input = st.text_area("News Text", height=200, placeholder="Paste news article here...")

if st.button("Analyze", type="primary"):
    if news_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)[0]

        decision = model.decision_function(vectorized)[0]
        confidence = min(abs(float(decision)) * 20, 100)

        st.divider()
        if result == 1:
            st.error("FAKE NEWS")
        else:
            st.success("REAL NEWS")

        st.metric("Confidence Score", f"{confidence:.1f}%")
        st.progress(int(confidence))