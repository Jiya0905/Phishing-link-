"""
server.py
---------
This is the ENTIRE backend web server - and it's pure Python. No Flask.
No Node.js. Just Python's built-in `http.server` module from the standard
library (something that ships with every Python installation, nothing to
install for it).

Routes:
  POST /api/predict   -> runs the URL through the ML model, saves it to
                          history.json, returns the result as JSON.
  GET  /api/history    -> returns every past scan saved in history.json.

WHY NO CORS SETUP HERE EITHER?
frontend/package.json has a "proxy" field pointing at
http://localhost:5000. React's own dev server (which is just a small
Node/webpack tool used only to preview your React code while building it)
forwards any /api/... call straight to this Python server. The browser
therefore only ever talks to ONE address, so there's no cross-origin
request happening at all - meaning no CORS headers are needed on this
server, from Flask or from anything else.

Run it with:
    python server.py
It starts listening on http://localhost:5000
"""

import http.server
import json
import os
import sys
import threading
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import pickle
import pandas as pd
from feature_extractor import extract_features, FEATURE_NAMES, get_matched_suspicious_words
from explain import generate_explanation

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "history.json")
PORT = 5000

# ---------- Load the trained model ONCE when the server starts ----------
# (loading it fresh on every request would be slow - this keeps it in
# memory and ready to answer instantly)
print("Loading trained ML model...")
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)
IMPORTANCES = dict(zip(FEATURE_NAMES, MODEL.feature_importances_.tolist()))
print("Model loaded successfully.")

history_lock = threading.Lock()  # prevents two requests from writing to
                                  # history.json at the exact same moment


# ---------- history.json helpers ----------
def read_history():
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_to_history(entry):
    with history_lock:
        history = read_history()
        # If this exact URL was already checked before, remove the old
        # entry first - we want the list to show each URL once, updated
        # with its latest result, not a growing pile of duplicates.
        normalized_url = entry["url"].strip().lower()
        history = [h for h in history if h.get("url", "").strip().lower() != normalized_url]
        history.insert(0, entry)          # newest first
        history = history[:200]           # keep the file from growing forever
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


# ---------- the actual ML prediction ----------
def predict_url(url: str) -> dict:
    features = extract_features(url)
    features["matched_suspicious_words"] = get_matched_suspicious_words(url)
    X = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)

    prediction = MODEL.predict(X)[0]
    probabilities = MODEL.predict_proba(X)[0]
    label = "Phishing" if prediction == 1 else "Legitimate"
    confidence = round(float(max(probabilities)) * 100, 2)
    reasons = generate_explanation(features, label, IMPORTANCES)

    return {
        "url": url,
        "label": label,
        "is_phishing": bool(prediction == 1),
        "confidence": confidence,
        "probability_phishing": round(float(probabilities[1]) * 100, 2),
        "probability_legitimate": round(float(probabilities[0]) * 100, 2),
        "features": features,
        "reasons": reasons,
    }


# ---------- the HTTP request handler ----------
class RequestHandler(http.server.BaseHTTPRequestHandler):

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/history":
            self._send_json(read_history())
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        if self.path != "/api/predict":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw_body or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, status=400)
            return

        url = (data.get("url") or "").strip()
        if not url:
            self._send_json({"error": "Please provide a 'url' field."}, status=400)
            return

        try:
            result = predict_url(url)
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            save_to_history(result)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format, *args):
        # Keep the console clean during a live demo.
        # Comment this out (or remove it) if you want to see every request logged.
        pass


def main():
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), RequestHandler)
    print(f"Backend server running at http://localhost:{PORT}")
    print("(Pure Python standard library only - no Flask, no Node.js.)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.shutdown()


if __name__ == "__main__":
    main()
