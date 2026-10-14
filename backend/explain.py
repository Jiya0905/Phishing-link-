"""
explain.py
----------
This file answers the question: "WHY did the model decide this URL is
Phishing or Legitimate?"

The model itself only outputs numbers (0/1 and probabilities). This file
translates the 20 extracted features into plain-English reasons, and picks
the most relevant ones using the model's own `feature_importances_` (so the
reasons shown are the ones the model actually relied on most, not just a
random subset).

Each "rule" below covers one recognizable URL pattern. A rule can fire in
two directions:
  - as a RED FLAG  (evidence pointing towards "Phishing")
  - as a GOOD SIGN (evidence pointing towards "Legitimate")
Only the rules that make sense for the model's predicted label are shown.
"""

RULES = [
    {
        "keys": ["has_https"],
        "phishing_if": lambda f: f["has_https"] == 0,
        "phishing_text": lambda f: "Does not use HTTPS \u2014 the connection is not encrypted, which is common in phishing pages.",
        "legit_if": lambda f: f["has_https"] == 1,
        "legit_text": lambda f: "Uses HTTPS, meaning the connection is encrypted \u2014 typical of legitimate, trustworthy sites.",
    },
    {
        "keys": ["has_ip_address"],
        "phishing_if": lambda f: f["has_ip_address"] == 1,
        "phishing_text": lambda f: "Uses a raw IP address instead of a readable domain name \u2014 a well-known phishing technique.",
        "legit_if": lambda f: f["has_ip_address"] == 0,
        "legit_text": lambda f: "Uses a proper domain name rather than a raw IP address.",
    },
    {
        "keys": ["has_suspicious_word", "suspicious_word_count"],
        "phishing_if": lambda f: f["has_suspicious_word"] == 1,
        "phishing_text": lambda f: (
            f"Contains {f['suspicious_word_count']} suspicious keyword(s) found in this URL: "
            f"\u2018{', '.join(f.get('matched_suspicious_words', []))}\u2019 \u2014 words commonly used as bait in phishing."
        ),
        "legit_if": lambda f: f["has_suspicious_word"] == 0,
        "legit_text": lambda f: "Does not contain common phishing bait words like \u2018login\u2019, \u2018verify\u2019 or \u2018secure\u2019.",
    },
    {
        "keys": ["num_hyphens"],
        "phishing_if": lambda f: f["num_hyphens"] >= 2,
        "phishing_text": lambda f: f"Contains {f['num_hyphens']} hyphens in the URL \u2014 often used to imitate brand names (e.g. \u2018paypal-secure-login\u2019).",
        "legit_if": lambda f: f["num_hyphens"] == 0,
        "legit_text": lambda f: "Domain does not contain unusual hyphen chains.",
    },
    {
        "keys": ["domain_entropy"],
        "phishing_if": lambda f: f["domain_entropy"] >= 3.6,
        "phishing_text": lambda f: "The domain name looks random / auto-generated (high character randomness), typical of algorithmically created phishing domains.",
        "legit_if": lambda f: f["domain_entropy"] < 3.2,
        "legit_text": lambda f: "The domain name looks like a normal, human-chosen word rather than a random string.",
    },
    {
        "keys": ["num_subdomains"],
        "phishing_if": lambda f: f["num_subdomains"] >= 2,
        "phishing_text": lambda f: f"Uses {f['num_subdomains']} extra subdomains \u2014 a technique used to disguise the real destination domain.",
        "legit_if": lambda f: f["num_subdomains"] <= 1,
        "legit_text": lambda f: "Has a simple, normal subdomain structure.",
    },
    {
        "keys": ["num_at_symbols"],
        "phishing_if": lambda f: f["num_at_symbols"] >= 1,
        "phishing_text": lambda f: "Contains an \u2018@\u2019 symbol \u2014 browsers can be tricked into visiting whatever comes after it, hiding the real destination.",
        "legit_if": lambda f: f["num_at_symbols"] == 0,
        "legit_text": None,  # absence of '@' isn't a distinctive positive signal on its own
    },
    {
        "keys": ["is_shortened"],
        "phishing_if": lambda f: f["is_shortened"] == 1,
        "phishing_text": lambda f: "Uses a URL-shortening service, which hides the real destination link from the user.",
        "legit_if": lambda f: f["is_shortened"] == 0,
        "legit_text": None,
    },
    {
        "keys": ["url_length"],
        "phishing_if": lambda f: f["url_length"] >= 70,
        "phishing_text": lambda f: f"Unusually long URL ({f['url_length']} characters) \u2014 often used to bury malicious parameters or confuse the reader.",
        "legit_if": lambda f: f["url_length"] <= 40,
        "legit_text": lambda f: f"Short, simple URL ({f['url_length']} characters), typical of legitimate sites.",
    },
    {
        "keys": ["digit_letter_ratio"],
        "phishing_if": lambda f: f["digit_letter_ratio"] >= 0.3,
        "phishing_text": lambda f: "The domain contains an unusually high number of digits relative to letters.",
        "legit_if": None,
        "legit_text": None,
    },
    {
        "keys": ["has_double_slash_redirect"],
        "phishing_if": lambda f: f["has_double_slash_redirect"] == 1,
        "phishing_text": lambda f: "Contains a suspicious \u2018//\u2019 redirect pattern inside the URL path.",
        "legit_if": None,
        "legit_text": None,
    },
]


def generate_explanation(features: dict, label: str, importances: dict = None, max_reasons: int = 6):
    """
    Returns a list of plain-English reason strings explaining the prediction.

    features:    dict from feature_extractor.extract_features()
    label:       "Phishing" or "Legitimate" (the model's predicted class)
    importances: optional dict of {feature_name: importance_score} from the
                 trained model, used to rank which reasons matter most.
    """
    importances = importances or {}
    scored = []

    for rule in RULES:
        weight = max((importances.get(k, 0) for k in rule["keys"]), default=0)

        if label == "Phishing" and rule["phishing_if"] and rule["phishing_if"](features):
            scored.append((weight, rule["phishing_text"](features)))
        elif label == "Legitimate" and rule["legit_if"] and rule["legit_text"] and rule["legit_if"](features):
            scored.append((weight, rule["legit_text"](features)))

    scored.sort(key=lambda x: x[0], reverse=True)
    reasons = [text for _, text in scored[:max_reasons]]

    if not reasons:
        reasons = [
            f"The overall combination of URL characteristics matched the typical pattern of {label.lower()} URLs "
            "seen during training, even though no single strong red/green flag stood out."
        ]
    return reasons
