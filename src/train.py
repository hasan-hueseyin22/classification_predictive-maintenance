import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from config import *
from data_preprocessing import download_data, load_data, preprocess_data
from model import create_model_pipeline
import os

def train_and_evaluate():
    """Main function to run the training and evaluation process."""
    # Download data if it doesn't exist
    if not os.path.exists(DATA_PATH):
        download_data(DATA_URL, DATA_PATH)

    # Load and preprocess data
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        df, TARGET_COLUMN, DROP_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TEST_SIZE, RANDOM_STATE
    )

    # Create and train the model
    model_pipeline = create_model_pipeline(preprocessor)
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model_pipeline.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save the trained model
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
