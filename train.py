import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def make_synthetic_churn_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 70, size=n)
    credit_score = rng.integers(350, 850, size=n)
    balance = rng.normal(60000, 40000, size=n).clip(0, 250000)
    tenure = rng.integers(0, 10, size=n)
    num_products = rng.integers(1, 5, size=n)
    has_card = rng.integers(0, 2, size=n)
    is_active = rng.integers(0, 2, size=n)
    salary = rng.normal(70000, 50000, size=n).clip(0, 250000)

    # Probabilité churn (règle simple mais réaliste)
    logits = (
        -3.0
        + 0.03 * (age - 40)
        - 0.004 * (credit_score - 600)
        + 0.000015 * (balance - 50000)
        - 0.15 * tenure
        + 0.25 * (num_products - 2)
        + 0.35 * (1 - is_active)
        + 0.25 * (1 - has_card)
        - 0.000005 * (salary - 70000)
    )
    proba = 1 / (1 + np.exp(-logits))
    churn = (rng.random(n) < proba).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "credit_score": credit_score,
            "balance": balance,
            "tenure": tenure,
            "num_products": num_products,
            "has_card": has_card,
            "is_active": is_active,
            "salary": salary,
            "churn": churn,
        }
    )
    return df


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    df = make_synthetic_churn_data(n=2500, seed=42)
    df.to_csv("data/churn.csv", index=False)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, "model/model.pkl")
    print("Saved model to model/model.pkl")


if __name__ == "__main__":
    main()
