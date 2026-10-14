"""
prepare_kaggle_dataset.py
--------------------------
Converts a REAL dataset downloaded from Kaggle (a CSV with a column of URLs
and a column saying whether each one is safe or malicious) into the same
format our model training script expects.

WHY THIS EXISTS
Until now, the model was trained on generate_dataset.py's SYNTHETIC data
(URLs we generated with rules, not real ones). This script lets you swap
that out for a real, human-labeled dataset from Kaggle instead, so your
model learns from genuine phishing/legitimate URLs.

HOW TO GET A DATASET FROM KAGGLE (do this once, outside this project)
1. Go to kaggle.com and search "phishing url dataset". Two good, well-known
   options with well over 2,500 rows each:
     - "Phishing Site URLs" by taruntiwarihp
       (~550,000 rows; columns: URL, Label with values Good / Bad)
     - "Malicious URLs dataset" by sid321axn
       (~651,000 rows; columns: url, type with values benign / phishing /
       malware / defacement)
   You do need a free Kaggle account to download. Click "Download" on the
   dataset page (or use the Kaggle API if you have it set up).
2. Unzip the download and place the .csv file somewhere inside this
   `backend` folder - for example: backend/kaggle_data/phishing_urls.csv

USAGE
    python prepare_kaggle_dataset.py <path_to_kaggle_csv> [output_csv]

Example:
    python prepare_kaggle_dataset.py kaggle_data/phishing_urls.csv dataset_kaggle.csv

WHAT IT DOES
1. Loads the CSV and auto-detects which column holds the URL and which
   holds the label (different Kaggle datasets name these differently -
   see URL_COLUMN_CANDIDATES / LABEL_COLUMN_CANDIDATES below).
2. Maps whatever label values that dataset uses (Good/Bad, benign/phishing,
   0/1, -1/1, etc.) onto our standard 0 = Legitimate, 1 = Phishing/Malicious.
3. Prints a preview table of URL -> assigned label, so you can visually
   confirm the mapping is correct before committing to it.
4. Runs every URL through our OWN feature_extractor.py (the same one the
   live app uses) to build the same 20 numeric features.
5. Saves everything to a new CSV (default: dataset_kaggle.csv) in the exact
   same shape as dataset.csv, so train_model.py can train on it directly.

If your specific Kaggle file uses column names or label words this script
doesn't recognize, it will tell you exactly what it saw so you can add
them to the *_CANDIDATES / LABEL_VALUE_MAP lists below.
"""

import sys
import pandas as pd
from feature_extractor import extract_features, FEATURE_NAMES

# Column names this script already knows how to recognize.
URL_COLUMN_CANDIDATES = ["url", "URL", "Website", "website", "link", "Link", "domain", "Domain"]
LABEL_COLUMN_CANDIDATES = ["label", "Label", "type", "Type", "Result", "result", "status", "Status", "class", "Class"]

# Maps whatever text/number a dataset uses -> 0 (Legitimate) or 1 (Phishing/Malicious).
# Multiple malicious sub-types (phishing, malware, defacement, spam) are all
# lumped into 1, since our model is a binary Legitimate-vs-Malicious classifier.
LABEL_VALUE_MAP = {
    "good": 0, "benign": 0, "legitimate": 0, "safe": 0, "0": 0, "-1": 0, 0: 0, -1: 0,
    "bad": 1, "phishing": 1, "malicious": 1, "malware": 1, "defacement": 1,
    "spam": 1, "1": 1, 1: 1,
}


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def map_label(value):
    key = str(value).strip().lower()
    if key in LABEL_VALUE_MAP:
        return LABEL_VALUE_MAP[key]
    if value in LABEL_VALUE_MAP:
        return LABEL_VALUE_MAP[value]
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_kaggle_dataset.py <path_to_kaggle_csv> [output_csv]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "dataset_kaggle.csv"

    print(f"Loading {input_path} ...")
    raw = pd.read_csv(input_path)
    print(f"Loaded {len(raw)} rows. Columns found: {list(raw.columns)}")

    url_col = find_column(raw, URL_COLUMN_CANDIDATES)
    label_col = find_column(raw, LABEL_COLUMN_CANDIDATES)

    if url_col is None or label_col is None:
        print("\nCould not auto-detect the URL and/or label column.")
        print(f"  URL column candidates checked:   {URL_COLUMN_CANDIDATES}")
        print(f"  Label column candidates checked: {LABEL_COLUMN_CANDIDATES}")
        print("Open the CSV, note the real column names, and add them to those")
        print("lists near the top of this script, then run it again.")
        sys.exit(1)

    print(f"Using '{url_col}' as the URL column and '{label_col}' as the label column.")

    data = raw[[url_col, label_col]].dropna()
    data.columns = ["url", "raw_label"]

    data["label"] = data["raw_label"].apply(map_label)
    unmapped = data[data["label"].isna()]
    if len(unmapped) > 0:
        print(f"\nWARNING: {len(unmapped)} rows had a label value this script didn't recognize:")
        print(unmapped["raw_label"].value_counts().head(10).to_string())
        print("These rows will be dropped. Add the missing value to LABEL_VALUE_MAP")
        print("near the top of this script if they should be kept instead.\n")

    data = data.dropna(subset=["label"]).copy()
    data["label"] = data["label"].astype(int)

    print(f"\nUsable rows after cleaning: {len(data)}")
    counts = data["label"].value_counts().rename({0: "Legitimate (0)", 1: "Phishing/Malicious (1)"})
    print(counts.to_string())

    if len(data) < 2500:
        print(f"\nNOTE: only {len(data)} usable rows - for a reliable train/test split,")
        print("2,500+ rows is recommended. Consider a larger Kaggle dataset.")

    # This is the part that answers "show whether it's legitimate or malicious":
    # a preview of real URLs next to the label we assigned them.
    print("\nPreview - URL -> assigned label (first 10 rows):")
    preview = data.head(10).copy()
    preview["label_name"] = preview["label"].map({0: "Legitimate", 1: "Phishing/Malicious"})
    print(preview[["url", "raw_label", "label_name"]].to_string(index=False))

    print(f"\nExtracting our 20 features for {len(data)} URLs (this can take a minute)...")
    rows = []
    for i, (url, label) in enumerate(zip(data["url"], data["label"])):
        try:
            feats = extract_features(str(url))
            rows.append([url] + [feats[name] for name in FEATURE_NAMES] + [label])
        except Exception:
            continue  # skip any malformed row instead of crashing the whole run
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{len(data)} processed")

    out_df = pd.DataFrame(rows, columns=["url"] + FEATURE_NAMES + ["label"])
    out_df.to_csv(output_path, index=False)

    print(f"\nDone. Wrote {len(out_df)} rows to {output_path}")
    print(f"Next step:  python train_model.py {output_path}")


if __name__ == "__main__":
    main()
