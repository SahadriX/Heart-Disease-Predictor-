import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("heart_disease_clean_5features.csv")
X = df[["age", "sex", "cp", "trestbps", "chol"]]
y = df["target"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "heart_disease_model.pkl")
print("✅ Model retrained and saved locally.")
