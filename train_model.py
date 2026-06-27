import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
import joblib
import re
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Load data
df = pd.read_csv('dataset/WELFake_Dataset.csv')
df = df.dropna()

# 2. Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

df['clean'] = df['text'].apply(clean_text)

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42
)

# 4. Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# 6. Accuracy
preds = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")

# 7. Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
print("Model saved!")