import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Tennis Match AI Predictor", page_icon="🎾")

@st.cache_resource
def load_artifacts():
    if not os.path.exists("tennis_model_artifacts.pkl"):
        return None
    return joblib.load("tennis_model_artifacts.pkl")

artifacts = load_artifacts()

if artifacts is None:
    st.error("Model file missing! Please run 'python3 train_engine.py' first.")
else:
    all_players = artifacts["all_players"]
    model_elo = artifacts["model_elo"]
    ai_model = artifacts["ai_model"]
    h2h_tracker = artifacts["h2h_tracker"]
    player_bio = artifacts["player_bio"]

    st.title("🎾 ATP Match AI Predictor")
    st.write("Select two players and a surface to predict the outcome using advanced Elo and machine learning.")

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Player 1", all_players, index=0)
    with col2:
        p2 = st.selectbox("Player 2", all_players, index=1)

    surface = st.selectbox("Court Surface", ["Hard", "Clay", "Grass"])

    if st.button("Predict Match Outcome", type="primary"):
        if p1 == p2:
            st.warning("Please select two distinct players.")
        else:
            r1_b = model_elo.get_rating(p1)
            r2_b = model_elo.get_rating(p2)
            r1_s = model_elo.get_rating(p1, surface)
            r2_s = model_elo.get_rating(p2, surface)
            h2h_diff = h2h_tracker.get(f"{p1}_vs_{p2}", 0) - h2h_tracker.get(f"{p2}_vs_{p1}", 0)

            rank1 = player_bio.get(p1, {}).get("rank", 500.0)
            rank2 = player_bio.get(p2, {}).get("rank", 500.0)
            log_rank_adv = np.log(rank2 + 1) - np.log(rank1 + 1)

            features = np.array([[r1_b - r2_b, r1_s - r2_s, h2h_diff, log_rank_adv]])
            probs = ai_model.predict_proba(features)[0]
            p1_prob, p2_prob = float(probs[1]), float(probs[0])

            winner = p1 if p1_prob >= p2_prob else p2

            st.success(f"🏆 Predicted Winner: **{winner}**")

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric(label=f"{p1} Win Probability", value=f"{round(p1_prob * 100, 1)}%")
            metric_col2.metric(label=f"{p2} Win Probability", value=f"{round(p2_prob * 100, 1)}%")

            st.divider()
            st.subheader("📊 Matchup Breakdown")

            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.markdown(f"**{p1}**")
                st.write(f"- Overall Elo: `{round(r1_b, 1)}`")
                st.write(f"- {surface} Elo: `{round(r1_s, 1)}`")
                st.write(f"- Rank: `#{int(rank1)}`")
            with stat_col2:
                st.markdown(f"**{p2}**")
                st.write(f"- Overall Elo: `{round(r2_b, 1)}`")
                st.write(f"- {surface} Elo: `{round(r2_s, 1)}`")
                st.write(f"- Rank: `#{int(rank2)}`")

            h2h_wins_1 = h2h_tracker.get(f"{p1}_vs_{p2}", 0)
            h2h_wins_2 = h2h_tracker.get(f"{p2}_vs_{p1}", 0)
            st.info(f"Head-to-Head Record: **{p1}** ({h2h_wins_1} - {h2h_wins_2}) **{p2}**")
