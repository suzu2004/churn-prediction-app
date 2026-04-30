import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/churn_model.pkl")

    return model