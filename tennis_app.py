import streamlit as st
import pandas as pd
import numpy as np
import joblib

from elo_model import EloModel 

@st.cache_resource
def load_model_artifacts():
    return joblib.load("tennis_model_artifacts.pkl")

try:
    artifacts = load_model_artifacts()
    all_players = artifacts["all_players"]
    model_elo = artifacts["model_elo"]
    ai_model = artifacts["ai_model"]
    h2h_tracker = artifacts["h2h_tracker"]
    recent_form = artifacts["recent_form"]
    player_bio = artifacts["player_bio"]
    player_stats = artifacts.get("player_stats", {})
except FileNotFoundError:
    st.error("🚨 Could not find 'tennis_model_artifacts.pkl'. Please run train_final.py first!")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# --- APP LAYOUT ---
st.set_page_config(page_title="Ultimate AI Tennis Predictor", page_icon="🎾", layout="centered")
st.title("🎾 Ultimate AI Tennis Predictor")
st.markdown("Powered by a Voting Classifier Ensemble (Random Forest + Gradient Boosting), Elo Ratings, and Biometrics.")
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
        with st.spinner("Board of Directors analyzing data..."):
            r1_base, r2_base = model_elo.get_rating(p1), model_elo.get_rating(p2)
            r1_surf, r2_surf = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
            
            p1_wins_h2h = h2h_tracker.get(f"{p1}_vs_{p2}", 0)
            p2_wins_h2h = h2h_tracker.get(f"{p2}_vs_{p1}", 0)
            h2h_adv = p1_wins_h2h - p2_wins_h2h
            
            form_1 = (sum(recent_form.get(p1, [0.5])) / len(recent_form.get(p1, [0.5]))) * 100
            form_2 = (sum(recent_form.get(p2, [0.5])) / len(recent_form.get(p2, [0.5]))) * 100
            form_adv = (form_1 / 100) - (form_2 / 100)
            
            b1 = player_bio.get(p1, {"age": 25.0, "hand": "R"})
            b2 = player_bio.get(p2, {"age": 25.0, "hand": "R"})
            
            p1_is_lefty = 1 if b1["hand"] == 'L' else 0
            p2_is_lefty = 1 if b2["hand"] == 'L' else 0

            # Rest days removed from features array!
            features = np.array([[
                r1_base - r2_base, 
                r1_surf - r2_surf, 
                h2h_adv, 
                form_adv,
                b1["age"] - b2["age"],
                p1_is_lefty - p2_is_lefty
            ]])
            
            probabilities = ai_model.predict_proba(features)[0]
            p1_win_prob = probabilities[1] 
            
            st.subheader("Ensemble AI Prediction")
            if p1_win_prob > 0.5:
                st.success(f"🏆 **Predicted Winner:** {p1}")
                st.metric(label=f"{p1} Win Probability", value=f"{p1_win_prob * 100:.1f}%")
            else:
                st.success(f"🏆 **Predicted Winner:** {p2}")
                st.metric(label=f"{p2} Win Probability", value=f"{(1 - p1_win_prob) * 100:.1f}%")

            st.divider()
            
            st.subheader("📊 AI Evaluation Matrix")
            st.markdown("Here is the physical and historical context the AI used to adjust the math:")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Biometrics**")
                st.write(f"**{p1}:** Age {int(b1['age'])}, {b1['hand']} Hand")
                st.write(f"**{p2}:** Age {int(b2['age'])}, {b2['hand']} Hand")
            with c2:
                st.markdown("**Recent Form (Last 10)**")
                st.write(f"**{p1}:** {form_1:.1f}% win rate")
                st.write(f"**{p2}:** {form_2:.1f}% win rate")
            with c3:
                st.markdown(f"**{surface} Elo Rating**")
                st.write(f"**{p1}:** {int(r1_surf)}")
                st.write(f"**{p2}:** {int(r2_surf)}")
