# Fake News Detection System

A full-stack machine learning web application that classifies news articles as **Real** or **Fake** using TF-IDF vectorization and Logistic Regression.

## Live App

Try the deployed app: [Fake News Detector](https://fake-news-detector-5apv.onrender.com/)

## Project Structure

```
fake-news-detector/
├── data/              # Datasets
│   ├── Fake.csv
│   └── True.csv
├── model/             # Saved ML artifacts (generated after training)
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
├── backend/
│   ├── train_model.py # Data processing + model training pipeline
│   └── app.py         # Flask REST API
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
└── README.md
```

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
cd backend
python train_model.py
```

This will:

- Load and preprocess both datasets
- Train Logistic Regression and PassiveAggressiveClassifier
- Print accuracy, confusion matrix, and classification report
- Save the best model and TF-IDF vectorizer to `model/`

### 3. Start the app

```bash
cd backend
python app.py
```

Open `http://localhost:5000` in your browser. Flask serves the frontend and API from the same origin.

### 5. Use the app

1. Paste a news article into the text box.
2. Click **Check News**.
3. See the prediction (Real / Fake) and confidence score.

## API Reference

### GET `/health`

Returns the backend and loaded model status.

### POST `/predict`

**Request:**

```json
{ "text": "Your news article text here..." }
```

**Response:**

```json
{ "prediction": "Real", "confidence": 0.9734 }
```

## Tech Stack

| Layer    | Technology                 |
| -------- | -------------------------- |
| ML       | scikit-learn, NLTK, pandas |
| Backend  | Flask, flask-cors          |
| Frontend | HTML, CSS, JavaScript      |
