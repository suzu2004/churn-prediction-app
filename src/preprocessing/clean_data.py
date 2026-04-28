import pandas as pd

def clean(df):
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop missing values
    df = df.dropna()

    # Convert target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Convert boolean to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df