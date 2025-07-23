from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def create_model_pipeline(preprocessor):
    """Creates a full pipeline including preprocessing and the model."""
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])
    return model
