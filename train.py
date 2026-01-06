from pathlib import Path
import pandas as pd
import joblib

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "churn.csv"
MODEL_DIR = APP_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"

MODEL_DIR.mkdir(exist_ok=True)


# ---------- Config ----------
FEATURES_ORDER = ["age", "credit_score", "balance", "tenure", "products", "is_active"]

# Cherche automatiquement la colonne target (label)
POSSIBLE_TARGETS = ["churn", "Exited", "exit", "target", "label", "Churn"]


def find_target_column(df: pd.DataFrame) -> str:
    for col in POSSIBLE_TARGETS:
        if col in df.columns:
            return col
    raise ValueError(
        "Impossible de trouver la colonne cible (label) dans churn.csv.\n"
        f"Colonnes trouvées: {list(df.columns)}\n"
        "Ajoute une colonne 'churn' (0/1) ou renomme ta colonne cible."
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 1) Target
    target_col = find_target_column(df)

    # 2) Vérifier que les features existent
    missing = [c for c in FEATURES_ORDER if c not in df.columns]
    if missing:
        raise ValueError(
            "Il manque des colonnes features dans churn.csv:\n"
            f"{missing}\n"
            f"Colonnes actuelles: {list(df.columns)}"
        )

    X = df[FEATURES_ORDER].copy()
    y = df[target_col].copy()

    # (optionnel) s'assurer que y est bien 0/1
    # si y est "Yes/No" par exemple, tu peux adapter ici
    if y.dtype == "object":
        y = y.map({"yes": 1, "no": 0, "Yes": 1, "No": 0}).fillna(y)
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() == 2 else None
    )

    # 3) Modèle (Pipeline = scaler + logistic regression)
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # ---------- MLflow ----------
    mlflow.set_experiment("bank-churn")

    with mlflow.start_run():
        # log params
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("features", ",".join(FEATURES_ORDER))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # ✅ entraîner le modèle
        model.fit(X_train, y_train)

        # ✅ prédire
        y_pred = model.predict(X_test)

        # ✅ métriques
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)

        # roc_auc (si possible)
        if y.nunique() == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("roc_auc", auc)

        # confusion matrix (artifact)
        cm = confusion_matrix(y_test, y_pred)
        cm_path = APP_DIR / "confusion_matrix.txt"
        cm_path.write_text(str(cm), encoding="utf-8")
        mlflow.log_artifact(str(cm_path))

        # log model dans MLflow
        mlflow.sklearn.log_model(model, "model")

        # ✅ sauver le modèle pour l’API (model/model.pkl)
        joblib.dump(model, MODEL_PATH)

        print("✅ Training terminé")
        print(f"✅ Modèle sauvegardé: {MODEL_PATH}")
        print(f"✅ Accuracy: {acc:.4f}")
        if y.nunique() == 2:
            print(f"✅ ROC_AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
