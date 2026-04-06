"""
train_model.py
==============
Enhanced ML pipeline for Fake News Detection (v3).

Key fixes over v2:
  - Aggressive removal of Reuters / source-identifying text to prevent
    the model from learning formatting cues instead of actual content
  - Strips journalist bylines, city datelines, and agency markers
  - Better TF-IDF: sublinear_tf, tuned min_df/max_df for generalisation
  - class_weight='balanced' for classifiers
  - Sanity-check predictions on real-world sentences after training
"""

import os
import re
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# ── NLTK resources ──────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

# ═══════════════════════════════════════════════════════════════════════════
#  Compiled regexes — aggressively strip source-identifying patterns
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
#  Shared preprocessing — MUST be identical to the one used in app.py
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


# ═══════════════════════════════════════════════════════════════════════════
#  1. Load & label datasets
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  FAKE NEWS DETECTION — MODEL TRAINING (v3)")
print("=" * 60)

print("\n[1/7] Loading datasets …")
fake_df = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
true_df = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Fake articles : {len(fake_df):,}")
print(f"  True articles : {len(true_df):,}")
print(f"  Total         : {len(df):,}")


# ═══════════════════════════════════════════════════════════════════════════
#  2. Combine title + text and preprocess
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2/7] Preprocessing (title + text combined) …")
# Drop 'subject' column entirely — it leaks class information
if "subject" in df.columns:
    df.drop(columns=["subject"], inplace=True)
    print("  ✔  Dropped 'subject' column (prevents data leakage)")

df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["content"] = df["content"].apply(preprocess_text)

# Drop empty rows after preprocessing
empty_mask = df["content"].str.strip() == ""
if empty_mask.any():
    print(f"  ⚠  Dropping {empty_mask.sum()} empty rows after preprocessing")
    df = df[~empty_mask].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  3. TF-IDF vectorization (improved config)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3/7] TF-IDF vectorization …")
tfidf = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.7,
    sublinear_tf=True,         # dampens disproportionately frequent terms
    stop_words="english",
)
X = tfidf.fit_transform(df["content"])
y = df["label"]
print(f"  Features: {X.shape[1]:,}  |  Documents: {X.shape[0]:,}")


# ═══════════════════════════════════════════════════════════════════════════
#  4. Train / test split (80/20, stratified)
# ═══════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
print(f"\n[4/7] Split → Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")


# ═══════════════════════════════════════════════════════════════════════════
#  5. Train & evaluate three classifiers
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5/7] Training classifiers …\n")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced", random_state=42,
    ),
    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=100, C=1.0, class_weight="balanced", random_state=42,
    ),
    "Linear SVC": LinearSVC(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=42,
    ),
}

results = {}

for name, clf in models.items():
    print(f"  ── {name} {'─' * (40 - len(name))}")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted")
    rec  = recall_score(y_test, preds, average="weighted")
    f1   = f1_score(y_test, preds, average="weighted")
    cm   = confusion_matrix(y_test, preds)
    cr   = classification_report(y_test, preds, target_names=["Fake", "Real"])

    results[name] = {"model": clf, "accuracy": acc, "f1": f1}

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    Confusion Matrix:\n{cm}")
    print(f"    Classification Report:\n{cr}")


# ═══════════════════════════════════════════════════════════════════════════
#  6. Select best model and save
# ═══════════════════════════════════════════════════════════════════════════
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
best_acc   = results[best_name]["accuracy"]

print(f"\n[6/7] Best model: {best_name}  (accuracy {best_acc:.4f})")

if best_acc < 0.93:
    print("⚠  WARNING: Best accuracy is below the 93 % target!")
else:
    print("✔  Target accuracy (≥ 93 %) achieved.")

os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "fake_news_model.pkl")
vec_path   = os.path.join(MODEL_DIR, "vectorizer.pkl")

with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
with open(vec_path, "wb") as f:
    pickle.dump(tfidf, f)

print(f"  Model saved      → {model_path}")
print(f"  Vectorizer saved → {vec_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  7. Sanity-check predictions on real-world sentences
# ═══════════════════════════════════════════════════════════════════════════
print("\n[7/7] Sanity-check predictions on sample inputs …\n")

test_sentences = [
    ("Google introduced a new AI model that can translate over 100 languages with improved accuracy.", "Real"),
    ("NASA successfully launched a new satellite to study climate change patterns across the globe.", "Real"),
    ("The Federal Reserve announced an interest rate cut to support the economy.", "Real"),
    ("Scientists confirm the Earth is flat and all governments have been lying to us for decades.", "Fake"),
    ("EXPOSED: Secret illuminati meeting decides the next world president in underground bunker.", "Fake"),
    ("EXPOSED: Hillary Clinton runs secret child trafficking ring from pizza restaurant basement.", "Fake"),
]

correct = 0
for sentence, expected in test_sentences:
    cleaned  = preprocess_text(sentence)
    features = tfidf.transform([cleaned])
    pred     = best_model.predict(features)[0]
    label    = "Real" if pred == 1 else "Fake"
    status   = "✔" if label == expected else "✘"
    if label == expected:
        correct += 1
    print(f"  {status}  [{label:4s}] (expected {expected:4s}) → {sentence[:80]}")

print(f"\n  Sanity check: {correct}/{len(test_sentences)} correct")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
