# ⚽ EPL Winner Prediction Web App
# Author: Harpreet Singh

import streamlit as st
import pandas as pd
import pickle

# ---- Load Trained Model ----
model = pickle.load(open('epl_model.pkl', 'rb'))

st.set_page_config(page_title="EPL Winner Predictor", page_icon="⚽", layout="centered")

# ---- Title ----
st.title("🏆 English Premier League Winner Predictor")
st.markdown("### Enter the team stats below and predict the chance of being Champion!")

# ---- Sidebar for Info ----
with st.sidebar:
    st.header("About App ⚙️")
    st.markdown("""
    This ML app predicts which team is most likely to win the EPL
    based on performance metrics like goals, wins, and points.

    **Tech Stack:**  
    - Trained in Google Colab (RandomForestClassifier)  
    - Deployed using Streamlit Cloud  
    - Data Source: Kaggle EPL Dataset
    """)

# ---- Input Section ----
col1, col2 = st.columns(2)

with col1:
    wins = st.number_input("Wins", min_value=0, max_value=38, value=20)
    draws = st.number_input("Draws", min_value=0, max_value=38, value=8)
    losses = st.number_input("Losses", min_value=0, max_value=38, value=10)
    goals_scored = st.number_input("Goals Scored", min_value=0, max_value=150, value=70)
    goals_conceded = st.number_input("Goals Conceded", min_value=0, max_value=150, value=35)
    points = st.number_input("Points", min_value=0, max_value=114, value=wins*3 + draws)

with col2:
    shots = st.number_input("Total Shots", min_value=0, max_value=1000, value=500)
    shots_on_target = st.number_input("Shots on Target", min_value=0, max_value=500, value=250)
    corners = st.number_input("Corners", min_value=0, max_value=400, value=200)
    fouls = st.number_input("Fouls Committed", min_value=0, max_value=600, value=300)
    yellow_cards = st.number_input("Yellow Cards", min_value=0, max_value=200, value=60)
    red_cards = st.number_input("Red Cards", min_value=0, max_value=20, value=3)

# ---- Predict Button ----
if st.button("🔮 Predict Champion Probability"):
    features = [[wins, draws, losses, goals_scored, goals_conceded,
                 shots, shots_on_target, corners, fouls,
                 yellow_cards, red_cards, points]]

    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1] * 100  # chance of being Champion

    if prediction[0] == 1:
        st.success(f"🏆 This team has a **{probability:.2f}%** chance of winning the EPL!")
    else:
        st.warning(f"⚽ This team has only a **{probability:.2f}%** chance of winning the EPL.")

# ---- Footer ----
st.markdown("---")
st.caption("Built by Harpreet Singh • Powered by Streamlit + Scikit-learn")
