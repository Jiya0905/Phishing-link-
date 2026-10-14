# Phishing URL & Malicious Web Link Classifier (v3 — Pure Python Backend)

An AI/ML project that detects whether a URL is **Legitimate** or **Phishing/Malicious**,
explains **why**, and remembers every scan you've ever run — even after you close the browser.

- **Machine Learning model:** Python + scikit-learn (Random Forest Classifier)
- **Backend:** Python — the built-in `http.server` module only (no Flask, no Node.js)
- **Frontend:** React
- **Persistent history:** stored in `backend/history.json` on disk

## Why this version has no Flask AND no Node.js

The backend runs entirely on Python's standard library — `http.server`,
which ships with every Python installation, so there's nothing extra to
install for the web server itself (only `pandas` and `scikit-learn` are
needed, and only for the ML model). This means:

- **No Flask** — `server.py` handles HTTP requests directly using
  `http.server.BaseHTTPRequestHandler`.
- **No Flask-CORS, no `cors` npm package, no CORS setup at all** —
  `frontend/package.json` has a `"proxy"` field pointing at
  `http://localhost:5000`. React's own dev server forwards any `/api/...`
  request straight to the Python server for you. The browser only ever
  talks to one address (port 3000), so there's no cross-origin request to
  begin with.
- **No Node.js backend either** — everything (feature extraction, the
  trained model, the explanation engine, and the HTTP server) runs inside
  one single Python process. There's no subprocess call to a different
  language anymore.

---

## Project structure

```
phishing-classifier/
├── backend/
│   ├── server.py               # the ENTIRE backend - pure Python, no Flask
│   ├── feature_extractor.py    # URL -> 20 numeric features
│   ├── explain.py              # features -> plain-English "why" reasons
│   ├── predict.py              # optional CLI tool: test the model from the terminal
│   ├── generate_dataset.py     # builds a SYNTHETIC alternative dataset (optional)
│   ├── prepare_kaggle_dataset.py  # converts a raw url+label CSV into our training format
│   ├── train_model.py          # trains model.pkl (works on any CSV in this format)
│   ├── dataset.csv             # DEFAULT dataset - built from REAL URLs (see below)
│   ├── dataset.xlsx            # same data, Excel format - open this to show faculty
│   ├── dataset_synthetic.csv   # the old rule-generated dataset, kept for comparison
│   ├── kaggle_data/
│   │   ├── real_urls_raw.csv     # the RAW real URLs + label, before feature extraction
│   │   └── real_urls_raw.xlsx    # same, Excel format
│   ├── model.pkl               # already trained on the REAL dataset (82.99% test accuracy)
│   ├── history.json            # persisted scan history (auto-created/updated)
│   └── requirements.txt        # pandas, scikit-learn, openpyxl (for Excel export)
└── frontend/
    ├── public/index.html
    ├── src/App.js
    ├── src/App.css
    ├── src/index.js
    └── package.json            # has "proxy": "http://localhost:5000"
```

---

## 1. HOW IT WORKS

1. You type a URL into the React page and click "Check URL".
2. React sends it to `/api/predict`. The dev-server proxy forwards this to
   `http://localhost:5000/api/predict` — Python's `server.py`.
3. `server.py` (already has the trained model loaded in memory since
   startup) calls `extract_features()` to turn the URL into 20 numbers,
   runs the Random Forest model to get a prediction + confidence, and
   calls `generate_explanation()` from `explain.py` to turn the specific
   features that fired into plain-English reasons — ranked using the
   model's own `feature_importances_`, so the strongest reason shows first.
4. The result is saved (with a timestamp) into `history.json` on disk,
   then sent back to React as JSON.
5. React displays the verdict, confidence bar, and the "Why is this
   flagged...?" reasons list — including the *exact* suspicious keywords
   matched in that specific URL (not a generic example list). It also
   fetches `/api/history` on page load, so every scan you've ever run —
   today, yesterday, last week — shows up grouped by date. If you check a
   URL you've already checked before, its entry is **updated in place**
   (moved to the top with the latest result) instead of creating a
   duplicate row.

---

## 2. HOW TO RUN THIS IN VS CODE

You need **two terminals** open at the same time (backend + frontend).

### Step 1 — Unzip and open
Unzip `phishing-classifier.zip` → open the folder in VS Code.

### Step 2 — Start the backend (Terminal 1)
```bash
cd backend
python -m pip install --upgrade pip
pip install -r requirements.txt
python server.py
```
You should see:
```
Loading trained ML model...
Model loaded successfully.
Backend server running at http://localhost:5000
(Pure Python standard library only - no Flask, no Node.js.)
```
Leave this running.

### Step 3 — Start the frontend (Terminal 2)
```bash
cd frontend
npm install
npm start
```
This opens `http://localhost:3000` automatically.

### Step 4 — Test it
- `https://www.google.com` → **Legitimate**, with reasons like "Uses HTTPS", "Short, simple URL".
- `http://paypal-verify-login.xyz/confirm?user=1` → **Phishing**, with reasons like "Contains suspicious keywords", "Does not use HTTPS", "Domain looks random".
- Check a couple of URLs, close the browser tab entirely, reopen `localhost:3000` — your past scans are still there under "Recent checks", grouped by date.

### (Optional) Test the model directly from the terminal
```bash
cd backend
python predict.py "http://paypal-verify-login.xyz"
```
This prints the same JSON `server.py` would return, without needing the
server running — handy for quickly checking the model during development.

---

## 3. THE DATASET — where it's from, and how it was built

**`dataset.csv` (the default, already shipped) is built from REAL,
human-labeled URLs — not synthetic data.** Here's exactly where it came
from and how, so you can explain it honestly to faculty:

**Source:** a public, widely-cited URL corpus (~420,000 real URLs labeled
`good`/`bad`, originally compiled from blacklists and benign-site crawls —
the same base dataset several published Kaggle phishing-detection datasets
are built from). It is **not scraped from Kaggle directly in this project**
(Kaggle itself isn't reachable from where this project was packaged), but
it is the same real-world, publicly available data that similar Kaggle
datasets are built from — see `kaggle_data/real_urls_raw.csv` for the exact
file used.

**Sampling:** From that larger pool, a **balanced sample of ~5,080 URLs**
was taken (roughly 2,500 legitimate + 2,500 phishing/malicious) — well
over the 2,500-row minimum. This file is `kaggle_data/real_urls_raw.csv`
(and `.xlsx`) — open it in Excel to show the raw `url`, `raw_label`
(`good`/`bad`), and a readable `verdict` column (`Legitimate` /
`Phishing/Malicious`).

**One honest correction worth mentioning if asked:** the source dataset's
"legitimate" URLs are almost all long content pages (forum posts,
articles) — very few are plain root domains like `google.com`. Left
uncorrected, this made the model under-confident on simple real-world
homepages. So ~90 well-known root domains (google.com, github.com,
python.org, etc.) were manually added to the legitimate class to correct
this bias — a standard, documented technique, not something hidden.

**Feature extraction + training:** every one of those ~5,080 URLs was run
through `feature_extractor.py` (the exact same code the live app uses) to
produce the final `dataset.csv` — url + 20 numeric features + label.
`dataset.xlsx` is the same data in Excel format, ready to show faculty.

**Result on the held-out 20% test set:** **82.99% accuracy**, with
balanced precision/recall (~0.83) across both classes. This is a more
honest, defensible number than the ~100% the old synthetic data produced —
real-world data is noisier, and a lower-but-real accuracy is a stronger
answer in a viva than a suspiciously perfect one.

### Reproducing or replacing the dataset yourself

If you want to use a different/larger Kaggle dataset later:

**Step 1 — Download a dataset from Kaggle** (free account required), e.g.
**"Malicious URLs dataset"** by sid321axn (~651,000 rows, columns `url`
and `type`): `kaggle.com/datasets/sid321axn/malicious-urls-dataset`.
Place the CSV inside `backend/kaggle_data/`.

**Step 2 — Convert it into our training format:**
```bash
cd backend
python prepare_kaggle_dataset.py kaggle_data/your_file.csv dataset.csv
```
This auto-detects the URL/label columns, maps whatever label words it
uses onto our standard scheme (0 = Legitimate, 1 = Phishing/Malicious),
prints a preview table of URL → assigned label so you can sanity-check it,
then runs every URL through `feature_extractor.py` and saves the result.

**Step 3 — Retrain:**
```bash
python train_model.py dataset.csv
```
This does the 80/20 train/test split, trains the Random Forest, prints
accuracy + a classification report + confusion matrix, and overwrites
`model.pkl`. `server.py` and `predict.py` pick up the new model
automatically next time they start — no other code changes needed.

### Going back to the synthetic dataset (if you ever want to compare)
```bash
python train_model.py dataset_synthetic.csv
```

---

## 4. LIKELY VIVA / PRESENTATION QUESTIONS & ANSWERS

**Q: Why avoid Flask?**
A: To show that a fully working REST backend doesn't need a third-party
framework — Python's built-in `http.server` module is enough to parse
requests, run the model, and return JSON responses, with nothing extra to
install for the web layer.

**Q: How is CORS handled without Flask-CORS?**
A: It isn't needed at all. `frontend/package.json`'s `"proxy"` field makes
React's dev server forward `/api/...` calls to the Python server itself, so
the browser only ever sees one origin (`localhost:3000`) — there's no
cross-origin request happening, so no CORS headers need to be set anywhere.

**Q: How does the app explain *why* a URL is phishing or legitimate?**
A: `explain.py` holds a set of rules mapping specific feature values (e.g.
"no HTTPS", "high domain entropy", "contains 'verify'") to plain-English
sentences. When a prediction is made, only the rules relevant to the
predicted class are shown, ranked by the trained model's own
`feature_importances_` — so the explanation reflects what the model
actually weighted most heavily, not just a fixed list.

**Q: How is scan history persisted?**
A: Every prediction is appended (with a timestamp) to `backend/history.json`
on disk. The frontend fetches this file's contents via `GET /api/history`
on load, so history survives page refreshes, browser restarts, and even
backend restarts — it's not just in-memory React state.

**Q: What happens if I check the same URL twice?**
A: `server.py`'s `save_to_history()` removes any existing entry for that
exact URL before inserting the new result at the top — so the history
list always shows each URL once, updated with its latest result, rather
than growing a pile of duplicates.

**Q: Is the training data real or synthetic?**
A: Real. `dataset.csv` (the default, already shipped) is built from a
balanced sample of ~5,080 real, human-labeled URLs — not rule-generated
data. `generate_dataset.py` (producing synthetic data) is kept only as an
optional fallback, saved separately as `dataset_synthetic.csv` for
comparison. `prepare_kaggle_dataset.py` can convert any other raw
url+label CSV (e.g. a larger Kaggle dataset) into the same training
format if you want to retrain on more data later.

**Q: What are the limitations?**
A: Still URL-text-based only (no live page-content analysis), real
accuracy is a more modest 82.99% (not the ~100% synthetic data gave,
which is actually a more honest number), `history.json` is a simple flat
file rather than a real database, and `http.server` is single-process —
fine for a classroom prototype, but a production system would use a
proper web framework and
database.

Good luck with your presentation! 🎓
