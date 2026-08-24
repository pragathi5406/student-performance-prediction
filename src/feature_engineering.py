import pandas as pd

def create_features(student_info, student_assessment):

    # Convert score column to numeric
    student_assessment["score"] = pd.to_numeric(
        student_assessment["score"], errors="coerce"
    )

    # Create assessment features
    assessment_features = student_assessment.groupby("id_student").agg(
        avg_score=("score", "mean"),
        total_assessments=("score", "count")
    ).reset_index()

    # Merge with student info
    df = pd.merge(student_info, assessment_features, on="id_student", how="left")

    # Fill missing values
    df["avg_score"] = df["avg_score"].fillna(0)
    df["total_assessments"] = df["total_assessments"].fillna(0)

    # Create risk label
    df["risk_label"] = df["final_result"].apply(
        lambda x: 1 if x in ["Fail", "Withdrawn"] else 0
    )

    return df