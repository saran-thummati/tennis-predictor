import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- FIX: Teach the app what an EloModel is ---
import __main__
try:
    from train_final import EloModel
    __main__.EloModel = EloModel
except ImportError:
    st.error("🚨 Could not find 'train_final.py'.")
    st.stop()

@st.cache_resource
def load_model_artifacts():
    return joblib.load("tennis_model_artifacts.pkl")

try:
    artifacts = load_model_artifacts()
    all_players = artifacts["all_players"]
    model_elo = artifacts["model_elo"]
    rf_model = artifacts["rf_model"]
    h2h_tracker = artifacts["h2h_tracker"]
    player_stats = artifacts.get("player_stats", {})
except FileNotFoundError:
    st.error("🚨 Could not find 'tennis_model_artifacts.pkl'.")
    st.stop()

# --- APP LAYOUT ---
st.set_page_config(page_title="AI Tennis Predictor", page_icon="🤖", layout="centered")
st.title("🤖 Advanced AI Tennis Predictor")
st.markdown("Powered by a Random Forest Machine Learning Model, Elo Ratings, and historical context.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    p1 = st.selectbox("Select First Player", all_players, key="p1")
with col2:
    default_p2_idx = 1 if len(all_players) > 1 else 0
    p2 = st.selectbox("Select Second Player", all_players, index=default_p2_idx, key="p2")

st.divider()
surface = st.selectbox("Select Court Surface", ["Hard", "Clay", "Grass"])

# --- PREDICTION ENGINE ---
if st.button("🔮 Predict Match Outcome", use_container_width=True):
    if p1 == p2:
        st.warning("Please select two different players.")
    else:
        with st.spinner("Feeding data to Random Forest..."):
            # Extract features exactly as trained
            r1_base, r2_base = model_elo.get_rating(p1), model_elo.get_rating(p2)
            r1_surf, r2_surf = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
            
            p1_wins_h2h = h2h_tracker.get(f"{p1}_vs_{p2}", 0)
            p2_wins_h2h = h2h_tracker.get(f"{p2}_vs_{p1}", 0)
            h2h_adv = p1_wins_h2h - p2_wins_h2h
            
            s1 = player_stats.get(p1, {"matches": 0, "win_rate": 0, "recent_form": 50})
            s2 = player_stats.get(p2, {"matches": 0, "win_rate": 0, "recent_form": 50})
            form_adv = (s1["recent_form"] / 100) - (s2["recent_form"] / 100)

            # Build the feature array for the AI
            features = np.array([[r1_base - r2_base, r1_surf - r2_surf, h2h_adv, form_adv]])
            
            # Ask the AI for the probability
            probabilities = rf_model.predict_proba(features)[0]
            p1_win_prob = probabilities[1] # Probability class 1 (p1 wins)
            
            st.subheader("Random Forest Prediction")
            if p1_win_prob > 0.5:
                st.success(f"🏆 **Predicted Winner:** {p1}")
                st.metric(label=f"{p1} Win Probability", value=f"{p1_win_prob * 100:.1f}%")
            else:
                st.success(f"🏆 **Predicted Winner:** {p2}")
                st.metric(label=f"{p2} Win Probability", value=f"{(1 - p1_win_prob) * 100:.1f}%")

            st.divider()
            
            # --- CONTEXT DASHBOARD ---
            st.subheader("📊 Model Context & Features")
            st.markdown("The Random Forest evaluated these specific advantages to make its decision:")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Head-to-Head**")
                st.write(f"**{p1}:** {p1_wins_h2h} wins")
                st.write(f"**{p2}:** {p2_wins_h2h} wins")
            with c2:
                st.markdown("**Recent Form (Last 10)**")
                st.write(f"**{p1}:** {s1['recent_form']}% win rate")
                st.write(f"**{p2}:** {s2['recent_form']}% win rate")
            with c3:
                st.markdown(f"**{surface} Elo Rating**")
                st.write(f"**{p1}:** {int(r1_surf)}")
                st.write(f"**{p2}:** {int(r2_surf)}")
