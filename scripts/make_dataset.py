"""
make_dataset.py — Gera um dataset SINTETICO com o schema do German Credit
(valores crus: categoricas em texto, numericas em escala natural, Risk good/bad).

Dado FALSO, para o estudo rodar de forma reprodutivel. O objetivo e demonstrar
o pipeline de pre-processamento e a comparacao de modelos, nao metricas reais.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "dados" / "dados_credito.csv"


def main(n: int = 1000, seed: int = 42):
    rng = np.random.RandomState(seed)

    age = rng.randint(19, 75, n)
    sex = rng.choice(["male", "female"], n, p=[0.69, 0.31])
    job = rng.choice([0, 1, 2, 3], n, p=[0.02, 0.15, 0.63, 0.20])
    housing = rng.choice(["own", "rent", "free"], n, p=[0.71, 0.18, 0.11])
    saving = pd.Series(rng.choice(
        ["little", "moderate", "quite rich", "rich", "unknown"], n,
        p=[0.58, 0.10, 0.06, 0.05, 0.21]))
    checking = pd.Series(rng.choice(
        ["little", "moderate", "rich", "unknown"], n,
        p=[0.27, 0.27, 0.06, 0.40]))
    credit = rng.lognormal(7.6, 0.6, n).astype(int)
    duration = rng.choice(np.arange(6, 73, 6), n)
    purpose = rng.choice(
        ["car", "radio/TV", "furniture/equipment", "business",
         "education", "repairs", "domestic appliances", "vacation/others"],
        n, p=[0.34, 0.28, 0.18, 0.09, 0.05, 0.02, 0.02, 0.02])

    # Risco (bad) correlacionado com sinais plausiveis, com ruido para ser aprendivel
    z = (
        -1.8
        + 0.9 * (saving == "little").to_numpy()
        + 0.8 * (checking == "little").to_numpy()
        + 0.02 * (duration - 20)
        + 0.00004 * (credit - 3000)
        - 0.02 * (age - 35)
        + 0.5 * (housing != "own")
        + rng.normal(0, 0.45, n)
    )
    p_bad = 1 / (1 + np.exp(-z))
    risk = np.where(rng.binomial(1, p_bad) == 1, "bad", "good")

    # NA de verdade nas contas (como no German Credit original)
    saving = saving.replace("unknown", np.nan)
    checking = checking.replace("unknown", np.nan)

    df = pd.DataFrame({
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving,
        "Checking account": checking,
        "Credit amount": credit,
        "Duration": duration,
        "Purpose": purpose,
        "Risk": risk,
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT)  # indice vira a coluna Unnamed: 0 (load usa index_col=0)
    print(f"Wrote {OUT}: {df.shape} | bad={int((risk == 'bad').sum())} ({(risk == 'bad').mean():.0%})")


if __name__ == "__main__":
    main()
