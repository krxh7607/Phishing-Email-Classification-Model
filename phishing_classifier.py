# -*- coding: utf-8 -*-
"""
Phishing Email Classifier using Scikit-learn
=============================================
Classifies emails as 'Phishing' or 'Safe' using Naive Bayes on
bag-of-words features (text + URL signals extracted inline).
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import re
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – saves PNG files
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────────────────────

EMAILS = [
    # --- PHISHING (label = 1) ------------------------------------------------
    ("Congratulations! You've won a $1000 gift card. Click here to claim now http://bit.ly/win-prize", 1),
    ("URGENT: Your bank account has been suspended. Verify immediately at http://paypa1.com/verify", 1),
    ("Your PayPal account is at risk. Login now http://secure-paypal-verify.tk/login to avoid suspension", 1),
    ("FREE iPhone 15! Limited offer. Enter your credit card to claim http://freeiphone-offer.xyz", 1),
    ("Account alert: suspicious activity detected. Reset password at http://amaz0n-security.net", 1),
    ("Dear user, your Netflix subscription expired. Update billing http://netflix-billing-update.ml", 1),
    ("You are selected for a $500 reward. Click link to redeem http://reward-claim-now.info/redeem", 1),
    ("IRS TAX REFUND: You are eligible for a $3200 refund. Submit details http://irs-refund-claim.co", 1),
    ("Your Microsoft account will be deactivated. Confirm identity http://microsoft-verify.gq/account", 1),
    ("Exclusive deal! Buy now and get 90% off. Limited slots http://cheap-deals-exclusive.ru/buy", 1),
    ("WINNER NOTIFICATION: You have been selected. Click http://jackpot-winner.tk to claim your prize", 1),
    ("Urgent security breach! Your Apple ID was used in another country http://apple-id-verify.ml", 1),
    ("Your package is on hold. Confirm address to reschedule delivery http://dhl-redelivery.cc/track", 1),
    ("Tax authority notice: Pay outstanding dues to avoid penalty http://tax-penalty-notice.xyz/pay", 1),
    ("Crypto giveaway! Elon Musk is doubling Bitcoin. Send 0.1 BTC to get 0.2 http://crypto-double.tk", 1),
    ("Your email storage is full! Upgrade now or lose access http://email-upgrade-free.gq/upgrade", 1),
    ("Bank verification required: Provide OTP to unlock account http://bankverify-secure.ml/otp", 1),
    ("Job offer: Work from home earn $5000/week. Provide personal info http://work-home-jobs.cc", 1),
    ("ALERT: Unauthorized login detected on your account. Act now http://login-alert-secure.tk", 1),
    ("Update your payment method to continue Netflix service http://netflix-payment-update.ml", 1),
    ("Claim your lottery prize of $50000 http://lottery-prize-claim.xyz/claim send SSN to verify", 1),
    ("Your Dropbox account has exceeded quota. Click to upgrade http://dropbox-upgrade-now.gq", 1),
    ("Security warning: Your Google account password was changed http://google-account-verify.tk", 1),
    ("Get rich quick with our investment scheme, guaranteed returns http://invest-rich-quick.cc", 1),
    ("Your order has been placed for $299. Cancel at http://amazon-order-cancel.ml if not authorized", 1),

    # --- SAFE / LEGITIMATE (label = 0) ---------------------------------------
    ("Hi John, please find the quarterly report attached. Let me know if you have questions.", 0),
    ("Team standup is at 10 AM tomorrow. Please check the meeting agenda on Confluence.", 0),
    ("Your GitHub pull request has been approved. Merge when ready.", 0),
    ("Thanks for subscribing to our newsletter. You can unsubscribe at any time from account settings.", 0),
    ("Reminder: annual performance reviews are scheduled for next week. HR will send calendar invites.", 0),
    ("Your Amazon order #112-3456789 has shipped. Estimated delivery: Monday.", 0),
    ("Meeting notes from today's design review are posted in the shared drive.", 0),
    ("Welcome to the team! Your onboarding documents are attached.", 0),
    ("The sprint planning session will be held on Friday at 2 PM in conference room B.", 0),
    ("Your flight booking confirmation: AA1234 on Nov 20. Check in opens 24 hours before departure.", 0),
    ("Monthly server maintenance is scheduled for Saturday night. Expect brief downtime.", 0),
    ("Please complete the mandatory security training by end of this month.", 0),
    ("Your LinkedIn connection request was accepted. Start a conversation.", 0),
    ("Invoice #2024-089 for $450 is attached. Payment is due within 30 days.", 0),
    ("New comment on your Jira ticket: DEV-4521 – The fix looks good, closing.", 0),
    ("Your salary slip for October is available in the HR portal.", 0),
    ("Code review scheduled for Tuesday. Please ensure unit tests are passing.", 0),
    ("Dr. Smith confirmed your appointment for November 18 at 3:30 PM.", 0),
    ("Your library book renewal was successful. Due date extended by 14 days.", 0),
    ("Zoom meeting link for the client demo: https://zoom.us/j/12345678", 0),
    ("Please review and sign the updated NDA before the project kickoff meeting.", 0),
    ("Your Google Drive storage usage is at 70%. Manage files or upgrade your plan.", 0),
    ("Weekly digest: Top 5 articles in machine learning from our editorial team.", 0),
    ("Budget forecast for Q4 is attached. Finance review is on Thursday.", 0),
    ("Your password was successfully changed. If this was not you, contact support.", 0),
]

df = pd.DataFrame(EMAILS, columns=["email_text", "label"])
df["label_name"] = df["label"].map({1: "Phishing", 0: "Safe"})

print("=" * 65)
print("       PHISHING EMAIL CLASSIFIER  —  Scikit-learn")
print("=" * 65)
print(f"\n📊 Dataset  :  {len(df)} emails  ({df['label'].sum()} phishing, {(df['label']==0).sum()} safe)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (URL & keyword signals appended to text)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(text: str) -> str:
    """Append synthetic tokens that capture URL & keyword signals."""
    extra_tokens = []

    # URL presence
    urls = re.findall(r"http[s]?://\S+", text)
    if urls:
        extra_tokens.append("HAS_URL")
        for url in urls:
            # Suspicious TLDs
            if re.search(r"\.(tk|ml|gq|xyz|cc|ru|info|co\.cc|net\.ru)(/|$)", url):
                extra_tokens.append("SUSPICIOUS_TLD")
            # IP-based URL
            if re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", url):
                extra_tokens.append("IP_URL")
            # URL shorteners
            if re.search(r"bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly", url):
                extra_tokens.append("SHORT_URL")
            # Brand name typosquatting
            if re.search(r"paypa1|amaz0n|g00gle|micros0ft|netfl1x", url, re.I):
                extra_tokens.append("TYPOSQUAT_DOMAIN")

    # Urgency / reward keywords
    text_lower = text.lower()
    if any(w in text_lower for w in ["urgent", "immediately", "act now", "limited"]):
        extra_tokens.append("URGENCY_KEYWORD")
    if any(w in text_lower for w in ["winner", "won", "prize", "reward", "lottery", "gift card"]):
        extra_tokens.append("REWARD_KEYWORD")
    if any(w in text_lower for w in ["verify", "confirm", "suspended", "deactivated", "breach"]):
        extra_tokens.append("SECURITY_KEYWORD")
    if any(w in text_lower for w in ["free", "click here", "claim now", "click link"]):
        extra_tokens.append("CLICK_BAIT")
    if any(w in text_lower for w in ["credit card", "ssn", "otp", "password", "bank account"]):
        extra_tokens.append("SENSITIVE_INFO_REQUEST")

    return text + " " + " ".join(extra_tokens)


df["enriched_text"] = df["email_text"].apply(extract_features)
print("\n✅ Feature engineering complete (URL signals + keyword tokens appended)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

X = df["enriched_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n📂 Train samples : {len(X_train)}   Test samples : {len(X_test)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE  (CountVectorizer → TF-IDF → MultinomialNB)
# ─────────────────────────────────────────────────────────────────────────────

pipeline = Pipeline([
    ("vectorizer", CountVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=5000,
        stop_words="english",
        min_df=1,
    )),
    ("tfidf", TfidfTransformer()),
    ("classifier", MultinomialNB(alpha=0.5)),
])

pipeline.fit(X_train, y_train)
print("\n✅ Model trained  (CountVectorizer → TF-IDF → Multinomial Naive Bayes)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "─" * 65)
print(f"  🎯 Test Accuracy : {accuracy * 100:.2f}%")
print("─" * 65)

# 5-fold cross validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print(f"  📈 5-Fold CV Accuracy : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
print("─" * 65)

print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 6a. Confusion Matrix ---------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Safe", "Phishing"],
    yticklabels=["Safe", "Phishing"],
    ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix — Phishing Email Classifier", fontsize=13, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"💾 Saved  → {cm_path}")

# --- 6b. CV Score bar chart -------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
colors = ["#2196F3" if s >= cv_scores.mean() else "#90CAF9" for s in cv_scores]
bars = ax.bar(folds, cv_scores * 100, color=colors, edgecolor="white", linewidth=1.5)
ax.axhline(cv_scores.mean() * 100, color="#E53935", linestyle="--", linewidth=1.5,
           label=f"Mean: {cv_scores.mean()*100:.1f}%")
ax.set_ylim(0, 115)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
for bar, val in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
cv_path = os.path.join(OUTPUT_DIR, "cv_scores.png")
plt.savefig(cv_path, dpi=150)
plt.close()
print(f"💾 Saved  → {cv_path}")

# --- 6c. Top feature tokens -------------------------------------------------
vectorizer = pipeline.named_steps["vectorizer"]
nb_model   = pipeline.named_steps["classifier"]
feature_names = vectorizer.get_feature_names_out()

# Log-probability difference: phishing (class=1) vs safe (class=0)
log_prob_diff = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]
top_n = 15
top_phishing_idx = np.argsort(log_prob_diff)[-top_n:][::-1]
top_safe_idx     = np.argsort(log_prob_diff)[:top_n]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Phishing top tokens
phishing_features = [feature_names[i] for i in top_phishing_idx]
phishing_scores   = log_prob_diff[top_phishing_idx]
axes[0].barh(phishing_features[::-1], phishing_scores[::-1], color="#EF5350")
axes[0].set_title("Top 15 Phishing Indicators", fontsize=12, fontweight="bold", color="#B71C1C")
axes[0].set_xlabel("Log-Prob Difference (Phishing − Safe)")

# Safe top tokens
safe_features = [feature_names[i] for i in top_safe_idx]
safe_scores   = log_prob_diff[top_safe_idx]
axes[1].barh(safe_features[::-1], np.abs(safe_scores[::-1]), color="#66BB6A")
axes[1].set_title("Top 15 Safe Indicators", fontsize=12, fontweight="bold", color="#1B5E20")
axes[1].set_xlabel("Log-Prob Difference (Safe − Phishing)")

plt.suptitle("Most Discriminative Features", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
feat_path = os.path.join(OUTPUT_DIR, "top_features.png")
plt.savefig(feat_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"💾 Saved  → {feat_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. LIVE DEMO — classify new emails
# ─────────────────────────────────────────────────────────────────────────────

def classify_email(text: str) -> dict:
    """Classify a single email and return label + confidence."""
    enriched = extract_features(text)
    proba = pipeline.predict_proba([enriched])[0]
    pred  = pipeline.predict([enriched])[0]
    label = "🚨 PHISHING" if pred == 1 else "✅ SAFE"
    confidence = proba[pred] * 100
    return {"label": label, "confidence": confidence, "proba_safe": proba[0]*100, "proba_phish": proba[1]*100}


DEMO_EMAILS = [
    "URGENT: Your bank account is suspended! Verify at http://fakeb4nk-secure.tk/verify NOW",
    "Hi Sarah, here are the meeting notes from today's client call. Please review by Friday.",
    "You've won a $500 Amazon gift card! Click http://bit.ly/claim-gift-now to redeem immediately",
    "Team, the CI pipeline is green. Ready to merge to main.",
    "Your Apple ID was used to sign in from Russia. Confirm it was you http://apple-id-verify.ml",
]

print("\n" + "=" * 65)
print("   🔍 LIVE CLASSIFICATION DEMO")
print("=" * 65)
for i, email in enumerate(DEMO_EMAILS, 1):
    result = classify_email(email)
    snippet = email[:70] + "…" if len(email) > 70 else email
    print(f"\n[{i}] \"{snippet}\"")
    print(f"     → {result['label']}  (confidence: {result['confidence']:.1f}%)")
    print(f"        Safe: {result['proba_safe']:.1f}%  |  Phishing: {result['proba_phish']:.1f}%")

print("\n" + "=" * 65)
print("  ✅ All done! Charts saved in the project folder.")
print("=" * 65 + "\n")
