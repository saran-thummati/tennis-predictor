import streamlit as st
import pandas as pd
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
    player_stats = artifacts.get("player_stats", {})
except FileNotFoundError:
    st.error("🚨 Could not find 'tennis_model_artifacts.pkl'.")
    st.stop()

# --- APP LAYOUT ---
st.set_page_config(page_title="Tennis Match Predictor", page_icon="🎾", layout="centered")
st.title("🎾 Pro Tennis Match Predictor")
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
        st.warning("Please select two different players to simulate a match.")
    else:
        with st.spinner("Analyzing matchup..."):
            r1_base, r2_base = model_elo.get_rating(p1), model_elo.get_rating(p2)
            r1_surf, r2_surf = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
            
            p1_win_prob = model_elo.expected_score(r1_surf, r2_surf)
            
            st.subheader("Prediction Results")
            if p1_win_prob > 0.5:
                st.success(f"🏆 **Predicted Winner:** {p1}")
                st.metric(label=f"{p1} Win Probability", value=f"{p1_win_prob * 100:.1f}%")
            else:
                st.success(f"🏆 **Predicted Winner:** {p2}")
                st.metric(label=f"{p2} Win Probability", value=f"{(1 - p1_win_prob) * 100:.1f}%")

            st.divider()
            
            # --- STATS DASHBOARD ---
            st.subheader("📈 Career Statistics (2015-2025)")
            s1 = player_stats.get(p1, {"matches": 0, "win_rate": 0, "avg_dom": 50})
            s2 = player_stats.get(p2, {"matches": 0, "win_rate": 0, "avg_dom": 50})
            
            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.markdown(f"**{p1}**")
                st.write(f"🎾 **Matches Played:** {s1['matches']}")
                st.write(f"🏆 **Win Rate:** {s1['win_rate']}%")
                st.write(f"🔥 **Game Dominance:** {s1['avg_dom']}%")
                
            with stat_col2:
                st.markdown(f"**{p2}**")
                st.write(f"🎾 **Matches Played:** {s2['matches']}")
                st.write(f"🏆 **Win Rate:** {s2['win_rate']}%")
                st.write(f"🔥 **Game Dominance:** {s2['avg_dom']}%")

            st.divider()

            # --- TRANSPARENT ELO BREAKDOWN ---
            st.subheader("🤖 Inside the Algorithm")
            st.markdown("Machine learning doesn't have to be a black box. Here is the exact math the model used to predict this match.")
            
            st.markdown(f"### Step 1: Calculate {surface} Court Ratings")
            st.markdown(f"The model takes their career base rating and applies a modifier based on their historical dominance specifically on **{surface}** courts.")
            
            r1_modifier = int(r1_surf - r1_base)
            r2_modifier = int(r2_surf - r2_base)
            
            c3, c4 = st.columns(2)
            c3.metric(label=f"{p1} Final Rating", value=f"{int(r1_surf)}", delta=f"{r1_modifier} {surface} modifier")
            c4.metric(label=f"{p2} Final Rating", value=f"{int(r2_surf)}", delta=f"{r2_modifier} {surface} modifier")
            
            st.markdown("### Step 2: The Probability Formula")
            st.markdown("The algorithm plugs those final ratings into the standard Elo probability fraction:")
            
            # Draws the algebraic formula beautifully on the screen
            st.latex(r"P(Win) = \frac{1}{1 + 10^{(Rating_2 - Rating_1) / 400}}")
            
            st.markdown("Plugging in the numbers for this exact match:")
            st.latex(fr"P({p1}) = \frac{{1}}{{1 + 10^{{({int(r2_surf)} - {int(r1_surf)}) / 400}}}}")
            
            st.info(f"💡 Solving this fraction gives {p1} a **{p1_win_prob * 100:.1f}%** chance of winning, exactly as predicted above!")
