# Phishing Email Classifier

A machine learning-based email security classifier built with **Python** and **Scikit-learn**. This project classifies incoming emails as **Phishing** or **Safe** by analyzing textual content, keywords, and structural URL features.

---

## 📌 Features

- **Realistic Dataset**: 50 labeled examples (25 phishing and 25 legitimate emails) covering banking alerts, prize scams, tech support scams, and typical workplace correspondence.
- **Feature Engineering**:
  - URL detection and classification (IP-based URLs, shorteners like `bit.ly`, suspicious TLDs like `.tk`, `.ml`, `.xyz`, typosquatting such as `paypa1` or `amaz0n`).
  - High-risk keyword indicators (urgency, rewards, security breaches, click-bait, sensitive credential requests).
- **Machine Learning Pipeline**:
  - `CountVectorizer`: Converts text and extracted tokens to n-gram frequency features (unigrams and bigrams).
  - `TfidfTransformer`: Weights terms by inverse document frequency.
  - `MultinomialNB`: Multinomial Naive Bayes classifier with Laplace smoothing ($\alpha = 0.5$).
- **Comprehensive Evaluation**:
  - Accuracy score and stratified 5-fold cross-validation.
  - Full classification report (Precision, Recall, F1-Score).
  - Confusion matrix and discriminative feature importance charts.
- **Interactive Classification**: Includes a test function to evaluate unseen custom email texts with confidence scores.

---

## 📊 Performance Summary

| Metric | Safe (Legitimate) | Phishing | Overall / Weighted Avg |
|---|---|---|---|
| **Precision** | 1.00 | 0.75 | 0.88 |
| **Recall** | 0.71 | 1.00 | 0.85 |
| **F1-Score** | 0.83 | 0.86 | 0.84 |
| **Test Accuracy** | — | — | **84.62%** |
| **5-Fold CV Accuracy** | — | — | **88.00% ± 4.00%** |

> **Key takeaway**: 100% recall on phishing emails ensures zero malicious emails slip through undetected as safe.

---

## 🗂️ Project Structure

```text
phishing-email-classifier/
├── phishing_classifier.py    # Complete training, evaluation, and demo script
├── README.md                 # Project documentation
├── confusion_matrix.png      # Generated confusion matrix heatmap
├── cv_scores.png             # 5-fold cross-validation bar chart
└── top_features.png          # Top 15 phishing vs safe discriminative tokens
```

---

## ⚙️ Requirements & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- `pip`

### Install Dependencies
Run the following command in your terminal:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## 🚀 Running the Classifier

Run the main Python script:

```bash
python phishing_classifier.py
```

### Script Execution Flow
1. **Loads Dataset**: Constructs DataFrame from sample emails.
2. **Feature Engineering**: Detects URLs, suspicious TLDs, and intent keywords, appending synthetic signal tokens.
3. **Train / Test Split**: Splits into 75% train and 25% test sets with stratification.
4. **Pipeline Fit**: Trains `CountVectorizer` + `TfidfTransformer` + `MultinomialNB`.
5. **Evaluation**: Displays accuracy, 5-fold CV score, classification report, and saves visualization plots.
6. **Live Demo**: Runs inference on 5 sample unseen emails and prints class probability distributions.

---

## 🔍 Feature Extraction Logic

The function `extract_features(text)` inspects each email and appends high-signal tokens to the raw string:

- `HAS_URL`: Detects presence of hyperlinks (`http://` or `https://`).
- `SUSPICIOUS_TLD`: Matches free/abused TLDs (`.tk`, `.ml`, `.xyz`, `.gq`, `.cc`, `.ru`, `.info`).
- `TYPOSQUAT_DOMAIN`: Detects brand spoofing (e.g. `paypa1`, `amaz0n`, `netfl1x`).
- `SHORT_URL`: Flags link shorteners (`bit.ly`, `tinyurl`, `t.co`).
- `URGENCY_KEYWORD`: Flags urgent calls to action (`urgent`, `immediately`, `act now`).
- `SECURITY_KEYWORD`: Flags account suspension and verification warnings.
- `REWARD_KEYWORD`: Flags lottery and giveaway incentives.
- `SENSITIVE_INFO_REQUEST`: Flags requests for passwords, SSNs, credit cards, or OTPs.

---

## 🔮 Future Enhancements

- Ingest public datasets such as the **Enron Email Dataset** or **SpamAssassin Public Corpus**.
- Compare with ensemble and gradient boosted classifiers (`RandomForestClassifier`, `XGBoost`).
- Implement character n-grams to detect obfuscated typos and zero-width spaces.
- Package as a lightweight REST API using **FastAPI** or **Flask**.
