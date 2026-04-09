import pandas as pd
import numpy as np


def _load_dataset():
    return pd.read_csv("dados/dados_credito.csv", index_col=0)


class TestDataset:
    def test_loads_successfully(self):
        df = _load_dataset()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self):
        df = _load_dataset()
        required = {"Age", "Sex", "Job", "Housing", "Credit amount", "Duration", "Risk"}
        assert required.issubset(set(df.columns))

    def test_risk_is_binary(self):
        df = _load_dataset()
        assert set(df["Risk"].unique()).issubset({0, 1})

    def test_no_null_risk(self):
        df = _load_dataset()
        assert df["Risk"].isna().sum() == 0


class TestPreprocessing:
    def test_standard_scaling_applied(self):
        df = _load_dataset()
        # Age should be standardized (mean ~0, std ~1)
        assert abs(df["Age"].mean()) < 0.5
        assert 0.5 < df["Age"].std() < 1.5

    def test_train_test_split_shapes(self):
        from sklearn.model_selection import train_test_split

        df = _load_dataset()
        X = df.drop("Risk", axis=1)
        y = df["Risk"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
