import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# 1. Načtení modelu přímo ze souboru v repozitáři
@st.cache_resource
def load_local_model():
    return joblib.load('real_data_model.pkl')

model = load_local_model()

st.title("⚽ Live Football Predictor")

# --- Formuář pro vstupy ---
minute = st.slider("Minuta", 0, 90, 45)

col1, col2 = st.columns(2)
with col1:
    h_goals = st.number_input("Domácí góly", 0, 10, 0)
    h_shots = st.number_input("Domácí střely", 0, 30, 2)
    h_sot = st.number_input("Domácí na bránu", 0, 20, 1)
with col2:
    a_goals = st.number_input("Hosté góly", 0, 10, 0)
    a_shots = st.number_input("Hosté střely", 0, 30, 1)
    a_sot = st.number_input("Hosté na bránu", 0, 20, 0)

if st.button("Predikovat"):
    # Příprava dat pro oba týmy
    d_home = pd.DataFrame([{'minute': minute, 'current_goals': h_goals, 'current_shots': h_shots, 'current_sot': h_sot, 'is_home': 1}])
    d_away = pd.DataFrame([{'minute': minute, 'current_goals': a_goals, 'current_shots': a_shots, 'current_sot': a_sot, 'is_home': 0}])
    
    p_home = model.predict(d_home)[0]
    p_away = model.predict(d_away)[0]
    
    st.header(f"Konečné skóre: {round(p_home)} : {round(p_away)}")
    st.write(f"Detailní odhad: {p_home:.2f} - {p_away:.2f}")
