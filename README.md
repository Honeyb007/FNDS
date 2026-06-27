# Fake News Detection System

A machine learning web app that detects whether a news article is real or fake with 94.74% accuracy.

## What it does

- Paste any news article or headline and get an instant real/fake prediction
- Shows confidence score for each prediction
- Trained on the WELFake dataset — over 70,000 news articles
- Clean, simple Streamlit interface

## How it works

Text is vectorized using TF-IDF then classified by a Passive Aggressive Classifier — a fast, online learning algorithm well-suited for text classification tasks.

## Built With

- **Frontend/UI** — Streamlit
- **ML Model** — scikit-learn (Passive Aggressive Classifier)
- **Vectorizer** — TF-IDF
- **Dataset** — WELFake (70,000+ articles)
- **Language** — Python 3.11

## Getting Started

1. Clone the repo
```bash
   git clone https://github.com/Honeyb007/fake-news-detection.git
   cd fake-news-detection
```

2. Install dependencies
```bash
   pip install -r requirements.txt
```

3. Download the WELFake dataset and place it in the project root as `WELFake_Dataset.csv`

4. Run the app
```bash
   streamlit run app.py
```

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 94.74% |
| Dataset | WELFake |
| Algorithm | Passive Aggressive Classifier |

## Author

[Kudirat Ovayami](https://github.com/Honeyb007)
