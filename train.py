from pathlib import Path
import pandas as pd
import joblib

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "churn.csv"
MODEL_DIR = APP_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
REPORTS_DIR = APP_DIR  # ou APP_DIR / "reports" si tu veux un dossier dédié

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


def normalize_target(y: pd.Series) -> pd.Series:
    """
    Convertit y en 0/1 si possible.
    Supporte Yes/No, yes/no, True/False, etc.
    """
    if y.dtype == "object":
        mapping = {
            "yes": 1, "no": 0,
            "Yes": 1, "No": 0,
            "true": 1, "false": 0,
            "True": 1, "False": 0,
        }
        y = y.map(mapping).fillna(y)
    return y.astype(int)


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

    # 3) Normaliser y en 0/1
    y = normalize_target(y)

    # 4) Split
    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() == 2 else None
    )

    # 5) Modèle (Pipeline)
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # ---------- MLflow ----------
    mlflow.set_experiment("bank-churn")

    with mlflow.start_run():
        # Params
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("features_order", ",".join(FEATURES_ORDER))
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # AUC (si binaire + predict_proba dispo)
        auc = None
        if y.nunique() == 2 and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("roc_auc", auc)

        # Artifacts: confusion matrix + report
        cm = confusion_matrix(y_test, y_pred)
        cm_path = REPORTS_DIR / "confusion_matrix.txt"
        cm_path.write_text(str(cm), encoding="utf-8")
        mlflow.log_artifact(str(cm_path))

        report = classification_report(y_test, y_pred, zero_division=0)
        report_path = REPORTS_DIR / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        # Log model dans MLflow (Artifacts)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Sauver le modèle pour l’API
        joblib.dump(model, MODEL_PATH)

        print("✅ Training terminé")
        print(f"✅ Modèle sauvegardé: {MODEL_PATH}")
        print(f"✅ Accuracy: {acc:.4f}")
        print(f"✅ F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        if auc is not None:
            print(f"✅ ROC_AUC: {auc:.4f}")
        print("✅ Artifacts générés: confusion_matrix.txt, classification_report.txt")
        print("✅ MLflow: run enregistré (voir mlflow ui)")


if __name__ == "__main__":
    main()
