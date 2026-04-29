import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(df, fit=False):
    df = df.copy()

    # Separate features & label
    if "label" in df.columns:
        y = df["label"]
        X = df.drop("label", axis=1)
    else:
        X = df
        y = None

    scaler = StandardScaler()

    if fit:
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, "scaler.pkl")
    else:
        scaler = joblib.load("scaler.pkl")
        X_scaled = scaler.transform(X)

    return X_scaled, y