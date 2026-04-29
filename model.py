import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, auc, classification_report
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("data.csv")   # ✅ FIXED (was clean_data.csv)
df.columns = df.columns.str.strip()

# Auto-detect label column
label_col = None
for col in df.columns:
    if col.lower() in ("label", "class", "target"):
        label_col = col
        break

if label_col is None:
    print("Available columns:", list(df.columns))
    raise ValueError("❌ Could not find label column!")

print(f"Data loaded: {df.shape}")
print(f"Label column: '{label_col}'")

# ===============================
# 2. CLEAN DATA
# ===============================
# Remove duplicates
df = df.drop_duplicates()

# Replace inf and NaN
df = df.replace([np.inf, -np.inf], 0)
df = df.dropna()

print(f"After cleaning: {df.shape}")

# ===============================
# 3. SPLIT FEATURES / LABEL
# ===============================
X = df.drop(columns=[label_col])
y = df[label_col]

# Remove constant columns
nunique = X.nunique()
constant_cols = nunique[nunique <= 1].index.tolist()

if len(constant_cols) > 0:
    print(f"Dropping {len(constant_cols)} constant columns")
    X = X.drop(columns=constant_cols)

print(f"Final features: {X.shape[1]}")

# ===============================
# 4. TRAIN TEST SPLIT (IMPORTANT)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ===============================
# 5. SCALING (NO DATA LEAKAGE)
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 6. BASELINE MODEL
# ===============================
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

print("\n=== LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1:", f1_score(y_test, y_pred_lr))

# ===============================
# 7. RANDOM FOREST MODEL
# ===============================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X_train_scaled, y_train)

# ===============================
# 8. PREDICTIONS
# ===============================
y_pred = rf.predict(X_test_scaled)
y_prob = rf.predict_proba(X_test_scaled)[:, 1]

# ===============================
# 9. METRICS
# ===============================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== RANDOM FOREST METRICS ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# ===============================
# 10. CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("✅ Saved: confusion_matrix.png")

# ===============================
# 11. ROC CURVE
# ===============================
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

print("✅ Saved: roc_curve.png")

# ===============================
# 12. FEATURE IMPORTANCE
# ===============================
importances = rf.feature_importances_
feat_names = X.columns

feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
feat_imp.head(10).plot(kind="barh")
plt.title("Top 10 Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("✅ Saved: feature_importance.png")

# ===============================
# 13. SAVE MODEL
# ===============================
joblib.dump(rf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("✅ Model saved")
print("✅ Scaler saved")
print("✅ Features saved")
print("\n🎯 Pipeline Complete!")