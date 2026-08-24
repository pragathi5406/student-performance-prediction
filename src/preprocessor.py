import pandas as pd
import os

# Path to raw dataset folder
DATA_PATH = "data/open+university+learning+analytics+dataset (1)"

# Load datasets
student_info = pd.read_csv(f"{DATA_PATH}/studentInfo.csv")
assessments = pd.read_csv(f"{DATA_PATH}/assessments.csv")
student_assessment = pd.read_csv(f"{DATA_PATH}/studentAssessment.csv")

print("Datasets loaded")

# Merge assessment information
assessment_data = pd.merge(
    student_assessment,
    assessments,
    on="id_assessment"
)

# Average score per student
avg_score = assessment_data.groupby("id_student")["score"].mean().reset_index()
avg_score.rename(columns={"score": "avg_score"}, inplace=True)

# Total assessments per student
total_assessments = assessment_data.groupby("id_student")["score"].count().reset_index()
total_assessments.rename(columns={"score": "total_assessments"}, inplace=True)

# Merge with student info
data = student_info.merge(avg_score, on="id_student", how="left")
data = data.merge(total_assessments, on="id_student", how="left")

# Fill missing values
data["avg_score"] = data["avg_score"].fillna(0)
data["total_assessments"] = data["total_assessments"].fillna(0)

# Select features
data = data[[
    "studied_credits",
    "num_of_prev_attempts",
    "highest_education",
    "avg_score",
    "total_assessments",
    "final_result"
]]

# Convert target variable
data["target"] = data["final_result"].map({
    "Pass": 1,
    "Distinction": 1,
    "Fail": 0,
    "Withdrawn": 0
})

# Drop original result column
data.drop("final_result", axis=1, inplace=True)

# Save processed dataset
os.makedirs("data", exist_ok=True)
data.to_csv("data/student_data.csv", index=False)

print("Preprocessing completed")
print("student_data.csv saved")