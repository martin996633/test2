import streamlit as st
import requests
import joblib
import pandas as pd
import numpy as np

# --- CONFIG ---
# Zde doplň svůj klíč z RapidAPI
API_KEY = "TVUJ_API_KLIC" 
API_HOST = "api-football-v1.p.rapidapi.com"
HEADERS = {"x-rapidapi-key": API_KEY, "x-rapidapi-host": API_HOST}

st.set_page_config(page_title="Real Match Predictor", layout="wide")
st.title("⚽ Live Predictor (Real Data)")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Ujisti se, že máš soubor 'real_data_model.pkl' v repozitáři
    return joblib.load('real_data_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Chyba při načítání modelu: {e}")
    st.stop()

# --- API FUNCTIONS ---
def get_live_matches():
    url = f"https://{API_HOST}/v3/fixtures"
    try:
        # live=all stáhne všechny zápasy. 
        return requests.get(url, headers=HEADERS, params={"live": "all"}).json().get('response', [])
    except: return []

def get_stats(fixture_id):
    # Stáhne střely a střely na bránu
    url = f"https://{API_HOST}/v3/fixtures/statistics"
    try:
        data = requests.get(url, headers=HEADERS, params={"fixture": fixture_id}).json().get('response', [])
        stats = {'home': {'shots':0, 'sot':0}, 'away': {'shots':0, 'sot':0}}
        
        if not data: return stats
        
        for i, team_data in enumerate(data):
            side = 'home' if i == 0 else 'away'
            for s in team_data['statistics']:
                val = s['value']
                if val is None: val = 0
                if s['type'] == 'Total Shots': stats[side]['shots'] = int(val)
                if s['type'] == 'Shots on Goal': stats[side]['sot'] = int(val)
        return stats
    except:
        return {'home': {'shots':0, 'sot':0}, 'away': {'shots':0, 'sot':0}}

# --- UI ---
st.markdown("Model trénovaný na datech z **Football-Data.co.uk** (Top 5 lig).")

if st.button("Aktualizovat zápasy"):
    with st.spinner("Načítám živá data..."):
        matches = get_live_matches()
        # Filtrujeme zápasy, které už běží alespoň minutu
        active = [m for m in matches if m['fixture']['status']['elapsed'] > 0]
    
    if not active:
        st.info("Právě se nehrají žádné zápasy (nebo API nevrací data).")
    else:
        st.success(f"Nalezeno {len(active)} živých zápasů.")
        for m in active:
            fid = m['fixture']['id']
            home = m['teams']['home']['name']
            away = m['teams']['away']['name']
            g_h = m['goals']['home']
            g_a = m['goals']['away']
            minute = m['fixture']['status']['elapsed']
            league = m['league']['name']
            
            # Získáme statistiky
            stats = get_stats(fid)
            
            # 1. Predikce pro DOMÁCÍ tým
            # Vstup: [minute, current_goals, current_shots, current_sot, is_home=1]
            vec_h = pd.DataFrame([{
                'minute': minute, 
                'current_goals': g_h, 
                'current_shots': stats['home']['shots'], 
                'current_sot': stats['home']['sot'], 
                'is_home': 1
            }])
            pred_h = model.predict(vec_h)[0]
            
            # 2. Predikce pro HOSTUJÍCÍ tým
            # Vstup: [minute, current_goals, current_shots, current_sot, is_home=0]
            vec_a = pd.DataFrame([{
                'minute': minute, 
                'current_goals': g_a, 
                'current_shots': stats['away']['shots'], 
                'current_sot': stats['away']['sot'], 
                'is_home': 0
            }])
            pred_a = model.predict(vec_a)[0]
            
            # Součet
            total_pred = pred_h + pred_a
            current_total = g_h + g_a
            remaining = max(0, total_pred - current_total)
            
            # Zobrazení
            with st.expander(f"{minute}' | {home} {g_h}:{g_a} {away} ({league})"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Domácí (S/SoT)", f"{stats['home']['shots']} ({stats['home']['sot']})")
                c2.metric("Hosté (S/SoT)", f"{stats['away']['shots']} ({stats['away']['sot']})")
                
                # Zelená barva pokud se čeká víc než 0.5 gólu
                delta_color = "normal"
                if remaining > 0.5: delta_color = "inverse"
                
                c3.metric(
                    "AI Predikce (Celkem)", 
                    f"{total_pred:.2f}", 
                    delta=f"+{remaining:.2f} gólů",
                    delta_color=delta_color
                )
