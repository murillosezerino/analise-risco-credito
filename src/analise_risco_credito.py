"""
Analise de Risco de Credito — estudo comparativo de 3 modelos classicos.

Pre-processamento correto:
- One-hot nas categoricas nominais (Sex, Housing, Saving/Checking account, Purpose)
- Job mantido como ordinal (0 a 3)
- StandardScaler apenas nas numericas continuas (Age, Credit amount, Duration)
- Risk mapeado: good -> 0, bad -> 1

Backend Agg para rodar headless (CI, servidores sem display). As figuras sao
salvas em imgs/ (o script pode ser reexecutado para regenera-las).
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, RocCurveDisplay,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "dados" / "dados_credito.csv"
IMGS_DIR = BASE / "imgs"

NOMINAL = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
NUMERIC = ["Age", "Credit amount", "Duration"]  # Job fica como ordinal


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def preprocess(df: pd.DataFrame):
    """Retorna (X numerico sem NaN, y binario)."""
    df = df.copy()
    df["Risk"] = df["Risk"].map({"good": 0, "bad": 1}).astype(int)

    # NA nas contas vira uma categoria propria antes do one-hot
    for col in ["Saving accounts", "Checking account"]:
        df[col] = df[col].fillna("unknown")

    y = df["Risk"]
    dummies = pd.get_dummies(df[NOMINAL], drop_first=True).astype(int)

    scaler = StandardScaler()
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(df[NUMERIC]), columns=NUMERIC, index=df.index
    )

    X = pd.concat([numeric_scaled, df[["Job"]], dummies], axis=1)
    return X, y


def train_models(X_train, X_test, y_train, y_test) -> dict:
    modelos = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }
    resultados = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        resultados[nome] = {
            "modelo": modelo,
            "acuracia": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1]),
            "pred": y_pred,
        }
    return resultados


def plot_confusion(y_test, y_pred, path: Path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusao - Random Forest")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc(resultados, X_test, y_test, path: Path):
    plt.figure(figsize=(8, 6))
    for nome, res in resultados.items():
        RocCurveDisplay.from_estimator(res["modelo"], X_test, y_test, name=nome, ax=plt.gca())
    plt.title("Curva ROC - Comparacao entre Modelos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_importance(model, feature_names, path: Path):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(np.array(feature_names)[order], importances[order])
    plt.title("Importancia das Features - Random Forest")
    plt.ylabel("Score")
    plt.xlabel("Variaveis")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metrics(resultados, path: Path):
    metricas_df = pd.DataFrame({
        "Modelo": list(resultados.keys()),
        "Acuracia": [r["acuracia"] for r in resultados.values()],
        "AUC": [r["auc"] for r in resultados.values()],
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=metricas_df, x="Modelo", y="Acuracia", ax=axes[0])
    axes[0].set_title("Comparacao de Acuracia")
    axes[0].set_ylim(0, 1)
    sns.barplot(data=metricas_df, x="Modelo", y="AUC", ax=axes[1])
    axes[1].set_title("Comparacao de AUC")
    axes[1].set_ylim(0, 1)
    plt.suptitle("Desempenho dos Modelos de Classificacao")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    IMGS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    logger.info(f"Dataset: {df.shape}")
    logger.info("Distribuicao de Risk:\n%s", df["Risk"].value_counts(normalize=True).to_string())

    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    resultados = train_models(X_train, X_test, y_train, y_test)

    rf = resultados["Random Forest"]
    logger.info("Melhor modelo: Random Forest")
    logger.info(f"Acuracia: {rf['acuracia']:.2f} | AUC: {rf['auc']:.2f}")
    logger.info("Relatorio de Classificacao:\n%s", classification_report(y_test, rf["pred"]))

    plot_confusion(y_test, rf["pred"], IMGS_DIR / "matriz_confusao_rf.png")
    plot_roc(resultados, X_test, y_test, IMGS_DIR / "curva_roc_modelos.png")
    plot_importance(rf["modelo"], X.columns, IMGS_DIR / "importancia_variaveis_rf.png")
    plot_metrics(resultados, IMGS_DIR / "comparacao_metricas.png")

    scores_cv = cross_val_score(rf["modelo"], X, y, cv=5, scoring="accuracy")
    logger.info(f"Acuracia media (CV 5-fold, RF): {scores_cv.mean():.2f}")
    logger.info(f"Figuras salvas em: {IMGS_DIR}")


if __name__ == "__main__":
    main()
