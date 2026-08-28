import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Elite Tennis Predictor", layout="centered")

class EloModel:
    """Minimal Elo model for display purposes"""
    def __init__(self):
        self.ratings = {}
        self.surface_ratings = {}
    
    def get_rating(self, player, surface=None):
        if surface:
            return self.surface_ratings.get(surface, {}).get(player, 1500.0)
        return self.ratings.get(player, 1500.0)
    
    def update(self, p1, p2, p1_win, surface, tourney_level='A'):
        pass

@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model artifacts"""
    try:
        if not os.path.exists("tennis_model_artifacts.pkl"):
            raise FileNotFoundError("Model artifact file not found")
        
        artifacts = joblib.load("tennis_model_artifacts.pkl")
        
        # Validate required keys
        required_keys = ["all_players", "model_elo", "ai_model", "h2h_tracker", "player_bio"]
        for key in required_keys:
            if key not in artifacts:
                raise KeyError(f"Missing required key: {key}")
        
        return artifacts
    except FileNotFoundError:
        st.error("❌ Model artifact file not found. Please run the data updater script first.")
        st.info("The GitHub Actions workflow will automatically train the model and upload it.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please clear the cache (⚙️ Settings → Clear Cache) and refresh the page.")
        st.stop()

# Load model
try:
    artifacts = load_model_artifacts()
    all_players = artifacts["all_players"]
    model_elo = artifacts["model_elo"]
    ai_model = artifacts["ai_model"]
    h2h_tracker = artifacts["h2h_tracker"]
    surface_form = artifacts.get("surface_form", {})
    player_bio = artifacts["player_bio"]
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# UI
st.title("🎾 Elite Tennis Predictor")
st.write("Powered by LightGBM & Time-Series Surface Elo")
st.divider()

col1, col2 = st.columns(2)
with col1:
    p1 = st.selectbox("Select First Player", all_players, key="p1")
with col2:
    default_p2_idx = 1 if len(all_players) > 1 else 0
    p2 = st.selectbox("Select Second Player", all_players, index=default_p2_idx, key="p2")

st.divider()
surface = st.selectbox("Select Court Surface", ["Hard", "Clay", "Grass"])

if st.button("Predict Match Outcome", use_container_width=True):
    if p1 == p2:
        st.warning("⚠️ Please select two different players.")
    else:
        with st.spinner("Consulting the Model..."):
            try:
                # Get Elo ratings
                r1_base = model_elo.get_rating(p1)
                r2_base = model_elo.get_rating(p2)
                r1_surf = model_elo.get_rating(p1, surface)
                r2_surf = model_elo.get_rating(p2, surface)
                
                # Head-to-head record
                p1_wins_h2h = h2h_tracker.get(f"{p1}_vs_{p2}", 0)
                p2_wins_h2h = h2h_tracker.get(f"{p2}_vs_{p1}", 0)
                h2h_adv = p1_wins_h2h - p2_wins_h2h
                
                # Surface form (recent performance)
                p1_recent = surface_form.get(p1, {}).get(surface, [0.5])
                p2_recent = surface_form.get(p2, {}).get(surface, [0.5])
                
                p1_form = (sum(p1_recent) / len(p1_recent)) * 100 if p1_recent else 50.0
                p2_form = (sum(p2_recent) / len(p2_recent)) * 100 if p2_recent else 50.0
                form_adv = ((p1_form / 100) - (p2_form / 100)) * 3.0
                
                # Player bio
                b1 = player_bio.get(p1, {})
                b2 = player_bio.get(p2, {})
                
                age1 = b1.get("age", 25.0)
                age2 = b2.get("age", 25.0)
                p1_is_lefty = 1 if b1.get("hand") == 'L' else 0
                p2_is_lefty = 1 if b2.get("hand") == 'L' else 0
                ht1 = b1.get("height", 185.0)
                ht2 = b2.get("height", 185.0)
                rank1 = b1.get("rank", 500.0)
                rank2 = b2.get("rank", 500.0)

                # Build feature vector
                features = np.array([[
                    r1_base - r2_base,
                    r1_surf - r2_surf,
                    h2h_adv,
                    form_adv,
                    age1 - age2,
                    p1_is_lefty - p2_is_lefty,
                    ht1 - ht2,
                    rank2 - rank1
                ]])
                
                # Predict
                probabilities = ai_model.predict_proba(features)[0]
                p1_win_prob = probabilities[1]
                
                # Display results
                st.subheader("Prediction")
                if p1_win_prob > 0.5:
                    st.success(f"✅ **Predicted Winner:** {p1}")
                    st.metric(label=f"{p1} Win Probability", value=f"{p1_win_prob * 100:.1f}%")
                else:
                    st.success(f"✅ **Predicted Winner:** {p2}")
                    st.metric(label=f"{p2} Win Probability", value=f"{(1 - p1_win_prob) * 100:.1f}%")
                
                # Show stats
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("H2H Record", f"{p1_wins_h2h} - {p2_wins_h2h}")
                with col2:
                    st.metric(f"Elo Rating", f"{r1_base:.0f} - {r2_base:.0f}")
                with col3:
                    st.metric(f"Surface Elo", f"{r1_surf:.0f} - {r2_surf:.0f}")
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please ensure the model is properly trained and try again.")