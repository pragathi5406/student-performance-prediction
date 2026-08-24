import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix


from groq import Groq
import os

st.set_page_config(page_title="Student Performance Prediction", layout="wide")

st.title(" AI Student Performance System")


model = pickle.load(open("models/student_model.pkl","rb"))


client = Groq(api_key="your api key")

# Sidebar
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select Section",
    [
        "Student Prediction",
        "Model Comparison",
        "Confusion Matrix",
        "Study Timer",
        "AI Chatbot"
    ]
)

# ------------------ STUDENT PREDICTION ------------------
if menu == "Student Prediction":

    st.header("Student Performance Prediction")

    studied_credits = st.slider("Studied Credits",0,200,60)
    num_prev_attempts = st.slider("Previous Attempts",0,5,1)
    avg_score = st.slider("Average Score",0,100,50)
    total_assessments = st.slider("Total Assessments",0,10,5)

    education = st.selectbox(
        "Highest Education",
        ["No Formal Education","Lower Than A Level","A Level","HE Qualification","Post Graduate"]
    )

    education_map = {
        "No Formal Education":0,
        "Lower Than A Level":1,
        "A Level":2,
        "HE Qualification":3,
        "Post Graduate":4
    }

    education_encoded = education_map[education]

    if st.button("Predict"):

        input_data = pd.DataFrame({
            "studied_credits":[studied_credits],
            "num_of_prev_attempts":[num_prev_attempts],
            "highest_education":[education_encoded],
            "avg_score":[avg_score],
            "total_assessments":[total_assessments]
        })

        prediction = model.predict(input_data)[0]
        risk_prob = model.predict_proba(input_data)[0][1]

        st.session_state["risk_prob"] = risk_prob
        st.session_state["prediction"] = prediction

        if prediction == 1:
            st.error("⚠ Student is At Risk")
        else:
            st.success("✅ Student Likely to Pass")

        st.write("Risk Probability:",round(risk_prob*100,2),"%")

        st.subheader("AI Study Recommendation")

        if risk_prob > 0.7:
            rec = ["Feynman Technique","Practice Testing","Spaced Repetition","Increase study hours"]
        elif risk_prob > 0.4:
            rec = ["Pomodoro Technique","Active Recall","Revise weak topics"]
        else:
            rec = ["Maintain current study routine","Weekly revision"]

        for r in rec:
            st.write("•",r)

# ------------------ MODEL COMPARISON ------------------
elif menu == "Model Comparison":

    st.header("Model Comparison")

    models = ["Logistic Regression","Random Forest","XGBoost"]
    accuracy = [0.81,0.82,0.85]

    fig,ax = plt.subplots()
    ax.bar(models,accuracy)
    ax.set_ylabel("Accuracy")

    st.pyplot(fig)

# ------------------ CONFUSION MATRIX ------------------
elif menu == "Confusion Matrix":

    st.header("Confusion Matrix")

    y_true = [0,0,1,1,0,1,0,1]
    y_pred = [0,1,1,1,0,1,0,0]

    cm = confusion_matrix(y_true,y_pred)

    fig,ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pass","At Risk"],
        yticklabels=["Pass","At Risk"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

# ------------------ STUDY TIMER ------------------
elif menu == "Study Timer":

    st.header("Learning Technique Timer")

    technique_times = {
        "Pomodoro Technique":25,
        "Active Recall":15,
        "Spaced Repetition":20,
        "Feynman Technique":30,
        "Practice Testing":20
    }

    technique = st.selectbox("Select Technique", list(technique_times.keys()))

    if "timer_running" not in st.session_state:
        st.session_state.timer_running=False
        st.session_state.time_left=0

    if st.button("Start Timer"):
        st.session_state.timer_running=True
        st.session_state.time_left=technique_times[technique]*60

    if st.button("Stop Timer"):
        st.session_state.timer_running=False

    timer_display = st.empty()

    while st.session_state.timer_running and st.session_state.time_left>0:
        mins,secs = divmod(st.session_state.time_left,60)
        timer_display.write(f"{mins:02d}:{secs:02d}")
        time.sleep(1)
        st.session_state.time_left -= 1

    if st.session_state.time_left==0 and st.session_state.timer_running:
        st.success("Study session completed")
        st.session_state.timer_running=False

# ------------------ AI CHATBOT (GROQ) ------------------
elif menu == "AI Chatbot":

    st.header("🤖 AI Study Assistant ")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_prompt = st.chat_input("Ask anything about study, exams, ML...")

    if user_prompt:

        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("user"):
            st.write(user_prompt)

        risk_prob = st.session_state.get("risk_prob", None)

        system_prompt = f"""
You are an AI Study Assistant.

Student Risk Level: {risk_prob if risk_prob else "Unknown"}

- Give study plans
- Help with exams
- Explain ML concepts
- Personalize advice

High risk → strict guidance
Low risk → motivational tone
"""

        try:
            with st.spinner("Thinking..."):

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *st.session_state.messages[-6:]
                    ]
                )

                bot_reply = response.choices[0].message.content

        except Exception as e:
            bot_reply = f"Error: {str(e)}"

        with st.chat_message("assistant"):
            st.write(bot_reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": bot_reply}
        )