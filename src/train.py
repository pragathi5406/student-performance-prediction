import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    best_model = None
    best_score = 0

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        print(name)
        print("Accuracy:", acc)
        print("Recall:", rec)
        print("----------------")

        if rec > best_score:
            best_score = rec
            best_model = model

    return best_model, X.columns


def save_model(model, columns):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/best_model.pkl")
    joblib.dump(columns, "models/feature_columns.pkl")

    print("Model saved!")