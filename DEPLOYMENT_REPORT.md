# Deployment Inspection Report

Date: 2026-11-02

## Application

- Backend: Flask app in `backend/app.py` loads `model/fake_news_model.pkl` and `model/vectorizer.pkl` at startup.
- Frontend: Flask serves `frontend/index.html` and its static assets from the same origin as the API.
- Browser prediction URL: `/predict` (relative to the page origin).
- Health endpoint: `GET /health`.
- Prediction endpoint: `POST /predict`, JSON body `{ "text": "..." }`.
- Start locally from the repository root with `cd backend && python app.py`; open `http://localhost:5000`.

## Model Training

`backend/train_model.py` reads `data/Fake.csv` and `data/True.csv`, preprocesses and vectorizes the articles, trains Logistic Regression, Passive Aggressive, and Linear SVC classifiers, then saves the best model and TF-IDF vectorizer under `model/`.

Run from the repository root with `cd backend && python train_model.py`.

## Dependencies

```text
pandas>=2.0.0
scikit-learn>=1.3.0
nltk>=3.8.0
flask>=3.0.0
flask-cors>=4.0.0
```

## Model Artifacts

```text
fake_news_model.pkl  392K
vectorizer.pkl       2.0M
Total                2.4M
```

## Deployment Note

The backend previously loaded the saved model with an scikit-learn version warning: the pickle artifacts were created with scikit-learn 1.8.0, while the runtime had 1.9.1. `requirements.txt` currently allows a broad range (`scikit-learn>=1.3.0`). Pin the training/runtime scikit-learn version consistently, or retrain the artifacts using the deployment version, to avoid pickle compatibility issues.

## Git State

The inspected worktree was clean. The same-origin browser connection fix was pushed to `origin/main` in commit `b45e138` (`Fix browser connection to prediction API`).
