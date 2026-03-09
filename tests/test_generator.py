"""Tests for the synthetic data generator."""


from data.generator import generate_transactions


class TestGenerator:
    def test_generates_correct_count(self):
        df = generate_transactions(n_txns=1000, n_users=100, n_merchants=50, seed=42)
        assert len(df) == 1000

    def test_has_required_columns(self):
        df = generate_transactions(n_txns=100, n_users=20, n_merchants=10, seed=42)
        required = [
            "tx_id", "user_id", "merchant_id", "device_id", "account_id",
            "amount", "currency", "timestamp", "lat", "lon",
            "channel", "mcc", "is_fraud",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_fraud_rate_approximate(self):
        df = generate_transactions(n_txns=10000, fraud_rate=0.05, seed=42)
        rate = df["is_fraud"].mean()
        assert 0.02 < rate < 0.10

    def test_unique_tx_ids(self):
        df = generate_transactions(n_txns=500, seed=42)
        assert df["tx_id"].n_unique() == 500

    def test_amounts_positive(self):
        df = generate_transactions(n_txns=500, seed=42)
        assert (df["amount"] > 0).all()

    def test_channels_valid(self):
        df = generate_transactions(n_txns=500, seed=42)
        valid = {"web", "mobile", "pos"}
        actual = set(df["channel"].unique().to_list())
        assert actual.issubset(valid)

    def test_reproducible_with_seed(self):
        df1 = generate_transactions(n_txns=100, seed=123)
        df2 = generate_transactions(n_txns=100, seed=123)
        assert df1["amount"].to_list() == df2["amount"].to_list()
