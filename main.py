from src.db.fetch_sql import load_data
from src.preprocessing.clean_data import clean
from src.preprocessing.pipeline import create_preprocessing_pipeline, save_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES

from sklearn.model_selection import train_test_split
import pandas as pd

from src.model.train import train_model
from src.model.evaluate import evaluate_model

# 1. Load data
df = load_data()

# 2. Minimal cleaning (without one-hot encoding)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 3. Create and fit preprocessing pipeline
preprocessor = create_preprocessing_pipeline()
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
y = df["Churn"]


# Fit the preprocessor on the data
X_processed = preprocessor.fit_transform(X)

# Debug prints
print("Before preprocessing shape:", X.shape)
print("After preprocessing shape:", X_processed.shape)

# Save the pipeline for later use in API
save_pipeline(preprocessor, "models/preprocessing_pipeline.pkl")

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. Train
model = train_model(X_train, y_train)

# 6. Evaluate
evaluate_model(model, X_test, y_test)