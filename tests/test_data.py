from sklearn.model_selection import train_test_split

from src.analise_risco_credito import load_data, preprocess


class TestRawDataset:
    def test_loads_successfully(self):
        df = load_data()
        assert len(df) > 0

    def test_has_required_columns(self):
        df = load_data()
        required = {
            "Age", "Sex", "Job", "Housing", "Saving accounts",
            "Checking account", "Credit amount", "Duration", "Purpose", "Risk",
        }
        assert required.issubset(set(df.columns))

    def test_risk_is_raw_labels(self):
        df = load_data()
        assert set(df["Risk"].unique()).issubset({"good", "bad"})


class TestPreprocessing:
    def test_target_is_binary(self):
        X, y = preprocess(load_data())
        assert set(y.unique()).issubset({0, 1})

    def test_no_nulls_and_all_numeric(self):
        X, y = preprocess(load_data())
        assert X.isna().sum().sum() == 0
        assert not any(str(dt) == "object" for dt in X.dtypes)

    def test_numeric_features_are_scaled(self):
        X, y = preprocess(load_data())
        # Age padronizada (media ~0, desvio ~1)
        assert abs(X["Age"].mean()) < 0.5
        assert 0.5 < X["Age"].std() < 1.5

    def test_train_test_split_shapes(self):
        X, y = preprocess(load_data())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
