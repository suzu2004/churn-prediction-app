import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

# Define numeric and categorical columns
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = ['Contract', 'InternetService']

def create_preprocessing_pipeline():
    """Create a preprocessing pipeline with OneHotEncoder"""
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num','passthrough', NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'  # Drop any other columns
    )
    
    return preprocessor


def save_pipeline(preprocessor, path="models/preprocessing_pipeline.pkl"):
    """Save the preprocessing pipeline"""
    joblib.dump(preprocessor, path)
    print(f"Pipeline saved to {path}")


def load_pipeline(path="models/preprocessing_pipeline.pkl"):
    """Load the preprocessing pipeline"""
    return joblib.load(path)


def preprocess_data(df, preprocessor):
    """Apply preprocessing pipeline to data"""
    # Select only the features we need
    df_selected = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    
    # Apply the pipeline
    df_transformed = preprocessor.transform(df_selected)
    
    return df_transformed
