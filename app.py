import streamlit as st
import requests
import pandas as pd
import joblib

# --- TVOJE API ---
API_KEY = "ZDE_VLOZ_SVUJ_KLIC"  # <--- ZKONTROLUJ SI KLÍČ!
API_HOST = "api-football-v1.p.rapidapi.com"
HEADERS = {"x-rapidapi-key": API_KEY, "x-rapidapi-host": API_HOST}

st.set_page_config(layout="wide")
st.title("🕵️ Debugging API")

# --- 1. TEST PŘIPOJENÍ ---
st.subheader("1. Co říká API?")

if st.button("Otestovat API spojení"):
    # Zkusíme stáhnout status účtu nebo live zápasy
    url = f"https://{API_HOST}/v3/fixtures"
    params = {"live": "all"}
    
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        data = response.json()
        
        # Vypíšeme surovou odpověď, abychom viděli chybu
        st.write("Status Code:", response.status_code)
        
        # KONTROLA CHYB
        if "errors" in data and data["errors"]:
            st.error("🚨 API CHYBA:")
            st.json(data["errors"]) # Tady uvidíš "Requests limit exceeded"
        elif "message" in data and data["message"]:
             st.warning(f"Zpráva API: {data['message']}")
        
        # KONTROLA VÝSLEDKŮ
        results = data.get("response", [])
        st.write(f"Počet nalezených zápasů: {len(results)}")
        
        if len(results) > 0:
            st.success("✅ Data tečou! Zde je ukázka prvního zápasu:")
            st.json(results[0])
        else:
            st.warning("API funguje, ale vrátilo prázdný seznam (0 zápasů).")
            
    except Exception as e:
        st.error(f"Kritická chyba Pythonu: {e}")

# --- 2. MODEL CHECK ---
st.subheader("2. Kontrola Modelu")
try:
    model = joblib.load('real_data_model.pkl')
    st.success("✅ Model 'real_data_model.pkl' je načtený správně.")
except:
    st.error("❌ Model 'real_data_model.pkl' chybí!")
