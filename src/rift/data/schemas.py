"""Data schemas and column definitions for Rift."""

# Transaction columns (raw)
TX_ID = "tx_id"
USER_ID = "user_id"
MERCHANT_ID = "merchant_id"
DEVICE_ID = "device_id"
ACCOUNT_ID = "account_id"
AMOUNT = "amount"
CURRENCY = "currency"
TIMESTAMP = "timestamp"
LAT = "lat"
LON = "lon"
CHANNEL = "channel"
MCC = "mcc"
IS_FRAUD = "is_fraud"

# Transaction schema
TRANSACTION_COLUMNS = [
    TX_ID,
    USER_ID,
    MERCHANT_ID,
    DEVICE_ID,
    ACCOUNT_ID,
    AMOUNT,
    CURRENCY,
    TIMESTAMP,
    LAT,
    LON,
    CHANNEL,
    MCC,
    IS_FRAUD,
]

# Feature columns (engineered)
FEATURE_COLUMNS = [
    "tx_count_1h",
    "tx_count_24h",
    "tx_count_7d",
    "spend_1h",
    "spend_24h",
    "spend_7d",
    "dist_from_centroid",
    "new_merchants_7d",
    "devices_per_account",
    "merchant_fraud_rate",
    "device_share_degree",
    "time_since_last_tx",
    "amount_zscore",
]

# Graph node/edge types
NODE_TYPES = ["user", "merchant", "device", "transaction", "account"]
EDGE_TYPES = [
    ("user", "transaction"),
    ("transaction", "merchant"),
    ("transaction", "device"),
    ("transaction", "account"),
    ("user", "device"),
    ("user", "merchant"),
    ("account", "device"),
]

# Channels and MCCs for synthetic data
CHANNELS = ["web", "mobile", "pos"]
CURRENCIES = ["USD", "EUR", "GBP"]
MCC_CATEGORIES = [
    "5411",  # Grocery
    "5812",  # Restaurants
    "5912",  # Drugstores
    "5311",  # Department stores
    "5541",  # Gas
    "4121",  # Taxi
    "5942",  # Bookstores
    "5999",  # Miscellaneous
]
