import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

def make_dataset(n: int = 2000) -> pd.DataFrame:
    rng = pd.Series(range(n))

    age = (18 + (rng * 7) % 60).astype(int)
    credit_score = (300 + (rng * 13) % 551).astype(int)
    balance = ((rng * 231) % 200000).astype(float)
    tenure = ((rng * 3) % 11).astype(int)
    products = (1 + (rng * 5) % 4).astype(int)
    is_active = ((rng * 17) % 2).astype(int)

    churn = (
        (credit_score < 500).astype(int)
        | ((balance > 120000) & (is_active == 0)).astype(int)
        | ((age > 55) & (products == 1)).astype(int)
    ).astype(int)

    return pd.DataFrame({
        "age": age,
        "credit_score": credit_score,
        "balance": balance,
        "tenure": tenure,
        "products": products,
        "is_active": is_active,
        "churn": churn
    })

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    df = make_dataset(n=2000)
    df.to_csv("data/churn.csv", index=False)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    joblib.dump(model, "model/model.pkl")

    print("✅ Dataset saved to: data/churn.csv")
    print("✅ Model saved to: model/model.pkl")
    print(f"✅ Test accuracy: {acc:.4f}")
    print("✅ Features order:", list(X.columns))

if __name__ == "__main__":
    main()
