import joblib
import pandas as pd

reg_model = joblib.load("models/regression_model.pkl")
clf_model = joblib.load("models/classification_model.pkl")

def predict_student(input_data):

    # Create engineered features
    input_data["Academic_Stability_Score"] = (
        (input_data["studytime"] * 2)
        - (input_data["failures"] * 3)
        - (input_data["absences"] * 0.1)
    )

    input_data["Risk_Index"] = (
        input_data["failures"] + input_data["absences"]
    )

    df = pd.DataFrame([input_data])

    # Apply same dummy encoding
    df = pd.get_dummies(df, drop_first=True)

    # 🔥 VERY IMPORTANT PART
    # Add missing columns and reorder to match training
    df = df.reindex(columns=reg_model.feature_names_in_, fill_value=0)

    grade_prediction = reg_model.predict(df)[0]
    category_prediction = clf_model.predict(df)[0]

    return grade_prediction, category_prediction
