from src.db.fetch_sql import load_data
from src.preprocessing.clean_data import clean

from sklearn.model_selection import train_test_split

from src.model.train import train_model
from src.model.evaluate import evaluate_model

# 1. Load data
df = load_data()

# 2. Clean data
df = clean(df)

# 3. Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train
model = train_model(X_train, y_train)

# 5. Evaluate
evaluate_model(model, X_test, y_test)