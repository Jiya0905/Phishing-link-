"""
predict.py
----------
A small command-line script: give it a URL, it prints a single line of
JSON with the prediction, confidence, extracted features, and plain-English
reasons. Handy for quickly testing the model without starting the full
server.

    python predict.py "http://some-url.com"

IMPORTANT: this script must print ONLY the JSON result to stdout. Anything
else (warnings, logs) is sent to stderr instead so it never corrupts the
JSON output.
"""

import sys
import os
import json
import pickle
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features, FEATURE_NAMES, get_matched_suspicious_words
from explain import generate_explanation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No URL provided."}))
        sys.exit(1)

    url = sys.argv[1].strip()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    features = extract_features(url)
    features["matched_suspicious_words"] = get_matched_suspicious_words(url)

    import pandas as pd
    X = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    label = "Phishing" if prediction == 1 else "Legitimate"
    confidence = round(float(max(probabilities)) * 100, 2)

    importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
    reasons = generate_explanation(features, label, importances)

    result = {
        "url": url,
        "label": label,
        "is_phishing": bool(prediction == 1),
        "confidence": confidence,
        "probability_phishing": round(float(probabilities[1]) * 100, 2),
        "probability_legitimate": round(float(probabilities[0]) * 100, 2),
        "features": features,
        "reasons": reasons,
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
