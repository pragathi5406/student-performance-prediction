import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# -----------------------------
# 1 Load dataset
# -----------------------------

data = pd.read_csv("data/student_data.csv")

print("Dataset Loaded")
print(data.head())


# -----------------------------
# 2 Split features and target
# -----------------------------

X = data.drop("target", axis=1)
y = data["target"]


# -----------------------------
# 3 Encode categorical feature
# -----------------------------

le = LabelEncoder()
X["highest_education"] = le.fit_transform(X["highest_education"])


# -----------------------------
# 4 Train test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 5 Train models
# -----------------------------

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))


# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))


# XGBoost
xgb_model = XGBClassifier(eval_metric="logloss")
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))


# -----------------------------
# 6 Save best model (XGBoost)
# -----------------------------

os.makedirs("models", exist_ok=True)

with open("models/student_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

print("Model saved successfully")


# -----------------------------
# 7 Save dataset for dashboard
# -----------------------------

os.makedirs("data", exist_ok=True)

final_data = X.copy()
final_data["target"] = y

final_data.to_csv("data/student_data.csv", index=False)

print("student_data.csv saved for dashboard")