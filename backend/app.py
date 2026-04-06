"""
app.py
======
Flask backend for the Fake News Detection System (v3).

Exposes POST /predict returning:
  {
    "prediction": "Real" | "Fake",
    "confidence": 0.xx,
    "explanation": "human-readable sentence",
    "keywords": ["word1", "word2", …]
  }

The preprocessing function is identical to the one in train_model.py
to guarantee consistency between training and inference.
"""

import os
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# ── NLTK resources ──────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

# ═══════════════════════════════════════════════════════════════════════════
#  Compiled regexes — MUST match train_model.py exactly
# ═══════════════════════════════════════════════════════════════════════════

# Reuters-style dateline: "WASHINGTON (Reuters) -" or "LONDON (Reuters) –"
RE_DATELINE = re.compile(
    r"^[A-Z][A-Za-z\s,/]+\(Reuters\)\s*[-–—]?\s*", re.MULTILINE
)
# Agency name anywhere: "(Reuters)", "(AP)", "(AFP)", etc.
RE_AGENCY = re.compile(
    r"\((?:Reuters|AP|AFP|Associated Press|UPI)\)", re.IGNORECASE
)
# Journalist byline: "By John Smith" at start of text
RE_BYLINE = re.compile(r"^By\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s*", re.MULTILINE)
# "FILE PHOTO:" or "FILE -"
RE_FILE_PHOTO = re.compile(r"FILE\s*(?:PHOTO|IMAGE|VIDEO)?\s*[-:–—]?\s*", re.IGNORECASE)
# Full date patterns: "March 5, 2024" / "Jan 12, 2019"
RE_DATE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b", re.IGNORECASE
)
# City name all-caps prefix: "WASHINGTON -", "NEW YORK -"
RE_CITY_PREFIX = re.compile(r"^[A-Z]{2}[A-Z\s,]+[-–—]\s*")
# "(Reporting by …; Editing by …)" footer
RE_REPORTING = re.compile(r"\((?:Reporting|Writing|Editing|Compiled)\s+by\b[^)]*\)", re.IGNORECASE)
# "21st Century Wire" or similar site names
RE_SITE_NAMES = re.compile(
    r"\b(?:21st\s*Century\s*Wire|InfoWars|Breitbart|Natural\s*News|"
    r"YourNewsWire|The\s*Gateway\s*Pundit|Zero\s*Hedge)\b", re.IGNORECASE
)
# URLs
RE_URL = re.compile(r"https?://\S+")
# Twitter handles
RE_TWITTER = re.compile(r"@\w+")
# "Featured image" captions
RE_FEATURED = re.compile(r"Featured\s+image\s*(?:via|from|credit|source)?[^.]*\.?", re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════
#  Shared preprocessing — MUST be identical to train_model.py
# ═══════════════════════════════════════════════════════════════════════════
def preprocess_text(text: str) -> str:
    """
    Clean and normalise a news article for the TF-IDF pipeline.

    Steps:
      1. Strip source-identifying text (Reuters datelines, agencies,
         bylines, reporting footers, site names, URLs, etc.)
      2. Remove date patterns
      3. Lowercase
      4. Remove everything except letters and whitespace
      5. Remove stopwords
      6. Lemmatize remaining tokens
      7. Collapse extra whitespace
    """
    text = RE_DATELINE.sub("", text)
    text = RE_AGENCY.sub("", text)
    text = RE_BYLINE.sub("", text)
    text = RE_FILE_PHOTO.sub("", text)
    text = RE_DATE.sub("", text)
    text = RE_CITY_PREFIX.sub("", text)
    text = RE_REPORTING.sub("", text)
    text = RE_SITE_NAMES.sub("", text)
    text = RE_URL.sub("", text)
    text = RE_TWITTER.sub("", text)
    text = RE_FEATURED.sub("", text)

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)        # keep only letters + spaces
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# ── Load model & vectorizer ────────────────────────────────────────────────
print("Loading model and vectorizer …")
with open(os.path.join(MODEL_DIR, "fake_news_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
print(f"Model loaded: {type(model).__name__}")


# ═══════════════════════════════════════════════════════════════════════════
#  Explanation helper
# ═══════════════════════════════════════════════════════════════════════════
def get_explanation(features, prediction, top_n=8):
    """
    Explainable AI: extract the most influential keywords and a
    human-readable explanation sentence.

    Uses model `coef_` (available on LR, PA, LinearSVC) multiplied by
    the TF-IDF feature vector to determine per-token importance.
    Positive coefficients push toward class 1 (Real), negative toward
    class 0 (Fake).  We select the direction matching the prediction.
    """
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores  = features.toarray().flatten()

    if hasattr(model, "coef_"):
        coef = model.coef_.flatten()
        importance = tfidf_scores * coef

        if prediction == 1:          # Real → largest positive importance
            top_idx = importance.argsort()[-top_n:][::-1]
        else:                         # Fake → largest negative importance
            top_idx = importance.argsort()[:top_n]
    else:
        top_idx = tfidf_scores.argsort()[-top_n:][::-1]

    # Only keep words actually present in the input
    keywords = [feature_names[i] for i in top_idx if tfidf_scores[i] > 0][:top_n]

    # Human-readable explanation sentence
    if prediction == 1:
        explanation = (
            "The model detected patterns typical of verified news reporting."
        )
    else:
        explanation = (
            "The model detected patterns often found in misleading articles."
        )

    return explanation, keywords


# ── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether a news article is Real or Fake, with explanation.

    Request  JSON: { "text": "article content …" }
    Response JSON: {
        "prediction":  "Real" | "Fake",
        "confidence":  float,
        "explanation": str,
        "keywords":    [str, …]
    }
    """
    data = request.get_json(force=True)
    raw_text = data.get("text", "")

    if not raw_text.strip():
        return jsonify({"error": "No text provided"}), 400

    # Preprocess (same function as training) and vectorize
    cleaned  = preprocess_text(raw_text)
    features = vectorizer.transform([cleaned])

    # Prediction
    prediction = model.predict(features)[0]
    label = "Real" if prediction == 1 else "Fake"

    # Confidence score
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = float(max(proba))
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(features)[0]
        confidence = float(1 / (1 + np.exp(-abs(decision))))
    else:
        confidence = None

    # Explanation
    explanation, keywords = get_explanation(features, prediction, top_n=8)

    return jsonify({
        "prediction":  label,
        "confidence":  round(confidence, 4) if confidence is not None else None,
        "explanation": explanation,
        "keywords":    keywords,
    })


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": type(model).__name__})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
