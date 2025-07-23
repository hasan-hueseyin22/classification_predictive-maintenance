# src/config.py

# Paths
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
DATA_PATH = "data/raw/ai4i2020.csv"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "models/model.joblib"

# Feature and target columns
TARGET_COLUMN = "Machine failure"
DROP_COLUMNS = ["UDI", "Product ID"]
CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42