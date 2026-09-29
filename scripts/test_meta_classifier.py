"""
Test meta-classifier to check if it's predicting correctly or biased toward one class.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Load meta-classifier
print("Loading meta-classifier...")
meta_classifier = joblib.load("artifacts/models/meta_classifier.joblib")

# Load meta features
print("Loading meta features...")
meta_df = pd.read_csv("artifacts/prop_datasets/tree_width/meta_features.csv")

# Rename columns if needed
if "post_id" in meta_df.columns and "tweet_id" not in meta_df.columns:
    meta_df = meta_df.rename(columns={"post_id": "tweet_id"})

print(f"\nDataset: {len(meta_df)} samples")
print(f"Columns: {meta_df.columns.tolist()}")
print(f"Ground truth distribution:")
print(f"  Non-propaganda (0): {(meta_df['true_label'] == 0).sum()} ({(meta_df['true_label'] == 0).mean()*100:.1f}%)")
print(f"  Propaganda (1): {(meta_df['true_label'] == 1).sum()} ({(meta_df['true_label'] == 1).mean()*100:.1f}%)")

# Define feature columns
feature_cols = [
    "text_pred",
    "mean_anger",
    "mean_disgust",
    "mean_fear",
    "mean_joy",
    "mean_sadness",
    "mean_surprise",
    "mean_neutral",
    "entropy",
    "variance"
]

# Check for missing features
missing_cols = set(feature_cols) - set(meta_df.columns)
if missing_cols:
    print(f"\n❌ Missing columns: {missing_cols}")
    exit(1)

# Extract features and labels
X = meta_df[feature_cols]
y_true = meta_df["true_label"].astype(int)

print(f"\nFeature statistics:")
print(X.describe())

# Check for NaN or inf
if X.isnull().any().any():
    print("\n⚠️ Warning: NaN values detected in features")
    print(X.isnull().sum())

if np.isinf(X.values).any():
    print("\n⚠️ Warning: Inf values detected in features")

# Predict
print("\n" + "="*50)
print("Running predictions...")
print("="*50)

y_pred_proba = meta_classifier.predict_proba(X)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\n📊 Overall Performance:")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  F1 Score: {f1:.3f}")

# Prediction distribution
print(f"\n📈 Prediction Distribution:")
print(f"  Predicted Non-propaganda (0): {(y_pred == 0).sum()} ({(y_pred == 0).mean()*100:.1f}%)")
print(f"  Predicted Propaganda (1): {(y_pred == 1).sum()} ({(y_pred == 1).mean()*100:.1f}%)")

# Probability distribution
print(f"\n🎲 Probability Statistics:")
print(f"  Mean: {y_pred_proba.mean():.3f}")
print(f"  Std: {y_pred_proba.std():.3f}")
print(f"  Min: {y_pred_proba.min():.3f}")
print(f"  Max: {y_pred_proba.max():.3f}")
print(f"  Median: {np.median(y_pred_proba):.3f}")

# Percentiles
print(f"\n  Percentiles:")
for p in [10, 25, 50, 75, 90]:
    print(f"    {p}th: {np.percentile(y_pred_proba, p):.3f}")

# Confusion matrix
print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)
print(f"\n  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# Classification report
print(f"\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Non-Propaganda", "Propaganda"]))

# Check if model is always predicting one class
unique_preds = np.unique(y_pred)
if len(unique_preds) == 1:
    print("\n❌ PROBLEM DETECTED: Model predicts only one class!")
    print(f"   All predictions are: {unique_preds[0]}")
    print("\n   Possible causes:")
    print("   1. Model threshold issue (try different thresholds)")
    print("   2. Feature scaling mismatch")
    print("   3. Model trained on imbalanced data")
else:
    print("\n✅ Model predicts both classes")

# Test different thresholds
print("\n" + "="*50)
print("Testing different thresholds:")
print("="*50)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
best_f1 = 0
best_threshold = 0.5

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    acc = accuracy_score(y_true, y_pred_thresh)
    f1_thresh = f1_score(y_true, y_pred_thresh)
    print(f"  Threshold {thresh:.1f}: Acc={acc:.3f}, F1={f1_thresh:.3f}, Pred_1={(y_pred_thresh==1).mean()*100:.1f}%")
    
    if f1_thresh > best_f1:
        best_f1 = f1_thresh
        best_threshold = thresh

print(f"\n🏆 Best threshold: {best_threshold} (F1={best_f1:.3f})")

# Sample predictions (show both classes)
print("\n" + "="*50)
print("Sample Predictions:")
print("="*50)

# Show some low probability samples
print("\n🔵 Samples with LOW propaganda probability (<0.3):")
low_prob_idx = np.where(y_pred_proba < 0.3)[0][:5]
for idx in low_prob_idx:
    print(f"  Tweet {meta_df.iloc[idx]['tweet_id']}: prob={y_pred_proba[idx]:.3f}, true={y_true.iloc[idx]}, text_pred={X.iloc[idx]['text_pred']:.3f}")

# Show some medium probability samples
print("\n🟡 Samples with MEDIUM propaganda probability (0.4-0.6):")
med_prob_idx = np.where((y_pred_proba >= 0.4) & (y_pred_proba <= 0.6))[0][:5]
for idx in med_prob_idx:
    print(f"  Tweet {meta_df.iloc[idx]['tweet_id']}: prob={y_pred_proba[idx]:.3f}, true={y_true.iloc[idx]}, text_pred={X.iloc[idx]['text_pred']:.3f}")

# Show some high probability samples
print("\n🔴 Samples with HIGH propaganda probability (>0.7):")
high_prob_idx = np.where(y_pred_proba > 0.7)[0][:5]
for idx in high_prob_idx:
    print(f"  Tweet {meta_df.iloc[idx]['tweet_id']}: prob={y_pred_proba[idx]:.3f}, true={y_true.iloc[idx]}, text_pred={X.iloc[idx]['text_pred']:.3f}")

# Feature importance (if available)
if hasattr(meta_classifier, 'coef_'):
    print("\n" + "="*50)
    print("Feature Coefficients:")
    print("="*50)
    coefs = meta_classifier.coef_[0]
    for feat, coef in sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True):
        sign = "+" if coef > 0 else ""
        print(f"  {feat:20s}: {sign}{coef:7.4f}")

print("\n" + "="*50)
print("Analysis complete!")
print("="*50)
