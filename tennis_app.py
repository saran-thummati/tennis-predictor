import streamlit as st
import pandas as pd
import joblib

# --- FIX: Teach the app what an EloModel is before opening the file ---
import __main__
try:
    from train_final import EloModel
    __main__.EloModel = EloModel
except ImportError:
    st.error("🚨 Could not find 'train_final.py'. Make sure it is in the exact same folder as this app!")
    st.stop()

# --- 1. LOAD THE SAVED BRAIN INSTANTLY ---
@st.cache_resource
def load_model_artifacts():
    return joblib.load("tennis_model_artifacts.pkl")

# Handle the case where the training script hasn't been run yet
try:
    artifacts = load_model_artifacts()
    all_players = artifacts["all_players"]
    model_elo = artifacts["model_elo"]
    all_surfaces = artifacts["all_surfaces"]
except FileNotFoundError:
    st.error("🚨 Could not find 'tennis_model_artifacts.pkl'.")
    st.info("Please run `python3 train_final.py` in your terminal first to generate the model!")
    st.stop()
except Exception as e:
    st.error(f"🚨 An error occurred loading the model: {e}")
    st.stop()

# --- 2. APP LAYOUT & UI ---
st.set_page_config(page_title="Tennis Match Predictor", page_icon="🎾", layout="centered")

st.title("🎾 Pro Tennis Match Predictor")
st.markdown("Select two players and a court surface to predict the outcome using our fast, pre-trained machine learning model.")

st.divider()

# Layout using columns for a clean look
col1, col2 = st.columns(2)

with col1:
    st.subheader("Player 1")
    p1 = st.selectbox("Select First Player", all_players, key="p1")

with col2:
    st.subheader("Player 2")
    # Default player 2 to a different player if possible
    default_p2_idx = 1 if len(all_players) > 1 else 0
    p2 = st.selectbox("Select Second Player", all_players, index=default_p2_idx, key="p2")

st.divider()

surface = st.selectbox("Select Court Surface", all_surfaces)

# --- 3. PREDICTION ENGINE ---
if st.button("🔮 Predict Match Outcome", use_container_width=True):
    if p1 == p2:
        st.warning("Please select two different players to simulate a match.")
    else:
        with st.spinner("Analyzing matchup..."):
            # Using the fast Elo rating system from your artifacts for instant calculation
            # It factors in the specific court surface (Hard, Clay, Grass, etc.)
            r1 = model_elo.get_rating(p1, surface)
            r2 = model_elo.get_rating(p2, surface)
            
            # Calculate win probability 
            p1_win_prob = model_elo.expected_score(r2, r1)
            
            st.subheader("Prediction Results")
            if p1_win_prob > 0.5:
                st.success(f"🏆 **Predicted Winner:** {p1}")
                st.metric(label=f"{p1} Win Probability", value=f"{p1_win_prob * 100:.1f}%")
            else:
                st.success(f"🏆 **Predicted Winner:** {p2}")
                st.metric(label=f"{p2} Win Probability", value=f"{(1 - p1_win_prob) * 100:.1f}%")
