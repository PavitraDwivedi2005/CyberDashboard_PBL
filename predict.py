import joblib
from utils import preprocess_data

def predict(df):
    model = joblib.load("model.pkl")

    X, _ = preprocess_data(df, fit=False)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities