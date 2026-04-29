import pandas as pd
import joblib
import matplotlib.pyplot as plt

def plot_feature_importance(df):
    model = joblib.load("model.pkl")

    X = df.drop("label", axis=1)

    importance = model.feature_importances_
    features = X.columns

    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    top10 = feat_imp.head(10)

    plt.figure()
    plt.barh(top10["Feature"], top10["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importance")

    plt.savefig("feature_importance.png")

    return plt