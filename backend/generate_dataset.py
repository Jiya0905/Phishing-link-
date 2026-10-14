"""
generate_dataset.py
--------------------
NOTE: this is now the FALLBACK/OPTIONAL dataset generator. The project's
default dataset.csv is built from REAL URLs (see kaggle_data/ and
prepare_kaggle_dataset.py) - this script is kept only so you can still
compare against a synthetic, rule-generated dataset if useful.

This script GENERATES a realistic synthetic dataset by:
  1. Building thousands of "legitimate-style" URLs (short, clean, real-looking
     domains, mostly https, no suspicious words).
  2. Building thousands of "phishing-style" URLs (long, messy, suspicious
     words, IP addresses, hyphens, random subdomains, no https, etc.)

This is a standard, accepted technique to demonstrate an ML pipeline when
a labeled real dataset isn't available - but since this project already
ships with a real dataset, you likely won't need to run this file at all.

Run this file if you want to: it creates dataset_synthetic_new.csv (does
NOT overwrite the real dataset.csv).
"""

import csv
import random
import string
from feature_extractor import extract_features, FEATURE_NAMES, SHORTENING_SERVICES

random.seed(42)

LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "amazon.com",
    "twitter.com", "instagram.com", "linkedin.com", "microsoft.com", "apple.com",
    "github.com", "stackoverflow.com", "reddit.com", "netflix.com", "yahoo.com",
    "nytimes.com", "bbc.com", "cnn.com", "who.int", "mit.edu", "stanford.edu",
    "python.org", "npmjs.com", "mozilla.org", "adobe.com", "spotify.com",
    "dropbox.com", "paypal.com", "ebay.com", "wordpress.com"
]

LEGIT_PATHS = [
    "", "/", "/about", "/contact", "/home", "/products", "/blog/2024/article",
    "/user/profile", "/search?q=laptop", "/docs/getting-started",
    "/news/technology", "/careers", "/support", "/pricing", "/login"
]

SUSPICIOUS_WORDS = [
    "login", "verify", "account", "update", "secure", "bank", "confirm",
    "signin", "password", "pay", "wallet", "gift", "bonus", "free",
    "urgent", "suspend", "limited", "recover", "unlock", "alert"
]

FAKE_TLDS = [".xyz", ".top", ".club", ".online", ".site", ".info", ".click", ".gq", ".tk"]

BRAND_NAMES = ["paypal", "amazon", "netflix", "apple", "microsoft", "google",
               "bankofamerica", "chase", "facebook", "instagram", "wellsfargo"]


def random_string(length):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def make_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)
    scheme = "https"  # legit sites almost always use https
    if random.random() < 0.15:
        # occasionally add a harmless subdomain like www or mail
        domain = random.choice(["www.", "mail.", "docs.", "shop.", ""]) + domain
    return f"{scheme}://{domain}{path}"


def make_ip_domain():
    return ".".join(str(random.randint(1, 255)) for _ in range(4))


def make_shortened_url():
    """A phishing link hidden behind a URL-shortening service."""
    service = random.choice(SHORTENING_SERVICES)
    return f"http://{service}/{random_string(random.choice([6, 7, 8]))}"


def make_phishing_url():
    brand = random.choice(BRAND_NAMES)
    word1 = random.choice(SUSPICIOUS_WORDS)
    word2 = random.choice(SUSPICIOUS_WORDS)
    style = random.randint(1, 6)
    scheme = random.choice(["http", "http", "https"])  # phishing sometimes fakes https too

    if style == 6:
        # shortened URL hiding the real (malicious) destination
        return make_shortened_url()

    if style == 1:
        # brand name mixed with suspicious words in a weird domain
        domain = f"{brand}-{word1}-{word2}{random.choice(FAKE_TLDS)}"
    elif style == 2:
        # raw IP address instead of a domain name
        domain = make_ip_domain()
    elif style == 3:
        # long random subdomain chain
        domain = f"{random_string(8)}.{brand}.{random_string(6)}{random.choice(FAKE_TLDS)}"
    elif style == 4:
        # brand as a fake subdomain of a random domain (classic trick)
        domain = f"{brand}.{word1}.{random_string(5)}.com"
    else:
        domain = f"{word1}-{brand}{random.choice(FAKE_TLDS)}"

    path_options = [
        f"/{word1}/{word2}.php",
        f"/{word1}-{word2}?user=1&id={random.randint(1000,9999)}",
        f"/secure/{brand}/{word1}",
        f"/{random_string(6)}/{word2}.html",
    ]
    path = random.choice(path_options)
    url = f"{scheme}://{domain}{path}"
    if random.random() < 0.2:
        url += f"@{random_string(4)}.com"  # '@' trick redirect
    return url


def build_dataset(n_per_class=1500):
    rows = []
    for _ in range(n_per_class):
        url = make_legit_url()
        feats = extract_features(url)
        rows.append((url, feats, 0))  # 0 = legitimate

    for _ in range(n_per_class):
        url = make_phishing_url()
        feats = extract_features(url)
        rows.append((url, feats, 1))  # 1 = phishing

    random.shuffle(rows)
    return rows


def save_csv(rows, path="dataset_synthetic_new.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url"] + FEATURE_NAMES + ["label"])
        for url, feats, label in rows:
            writer.writerow([url] + [feats[name] for name in FEATURE_NAMES] + [label])


if __name__ == "__main__":
    rows = build_dataset(n_per_class=1500)
    save_csv(rows)
    print(f"dataset_synthetic_new.csv created with {len(rows)} rows (1500 legit + 1500 phishing)")
    print("(dataset.csv, the real-data default, was NOT touched.)")
