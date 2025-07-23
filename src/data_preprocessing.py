# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import requests

def download_data(url, save_path):
    """Downloads data from a URL and saves it locally."""
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"Data downloaded and saved to {save_path}")

def load_data(path):
    """Loads data from a csv file."""
    return pd.read_csv(path)

def get_preprocessor(numerical_features, categorical_features):
    """Creates a preprocessing pipeline for numerical and categorical features."""
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def preprocess_data(df, target_column, drop_columns, numerical_features, categorical_features, test_size, random_state):
    """Drops unnecessary columns, splits data and applies preprocessing."""
    df = df.drop(columns=drop_columns)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    preprocessor = get_preprocessor(numerical_features, categorical_features)
    
    return X_train, X_test, y_train, y_test, preprocessor