import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier

@st.cache_data
def load_data():
    df = pd.concat([
        pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2022.csv"),
        pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv"),
        pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv"),
    ]).sort_values("tourney_date").reset_index(drop=True)
    return df

class EloModel:
    def __init__(self, k=32, initial_rating=1500):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings = {}
        self.surface_ratings = {}

    def get_rating(self, player, surface=None):
        if surface:
            return self.surface_ratings.setdefault(surface, {}).get(player, self.initial_rating)
        return self.ratings.get(player, self.initial_rating)

    def expected_score(self, r1, r2):
        return 1 / (1 + 10 ** ((r2 - r1) / 400))

    def update(self, p1, p2, p1_win, surface=None):
        r1, r2 = self.get_rating(p1), self.get_rating(p2)
        exp1 = self.expected_score(r1, r2)
        self.ratings[p1] = r1 + self.k * (p1_win - exp1)
        self.ratings[p2] = r2 + self.k * ((1 - p1_win) - (1 - exp1))
        if surface:
            sr1 = self.get_rating(p1, surface)
            sr2 = self.get_rating(p2, surface)
            exp_s = self.expected_score(sr1, sr2)
            self.surface_ratings[surface][p1] = sr1 + self.k * (p1_win - exp_s)
            self.surface_ratings[surface][p2] = sr2 + self.k * ((1 - p1_win) - (1 - exp_s))

def compute_recent_win_rate(player, match_history, n=20):
    matches = match_history[player][-n:]
    if not matches:
        return 0.5
    return sum(matches) / len(matches)

def compute_h2h(p1, p2, h2h_record):
    key = tuple(sorted([p1, p2]))
    record = h2h_record[key]
    total = record["wins_a"] + record["wins_b"]
    if total == 0:
        return 0.5
    if p1 == key[0]:
        return record["wins_a"] / total
    else:
        return record["wins_b"] / total

@st.cache_resource
def train_model():
    df = load_data()
    np.random.seed(42)
    flip = np.random.rand(len(df)) < 0.5
    df_clean = pd.DataFrame({
        "player1":  np.where(flip, df["loser_name"],  df["winner_name"]),
        "player2":  np.where(flip, df["winner_name"], df["loser_name"]),
        "surface":  df["surface"],
        "p1_win":   np.where(flip, 0, 1),
    })
    df_clean["rank1"] = np.where(flip, df["loser_rank"],  df["winner_rank"])
    df_clean["rank2"] = np.where(flip, df["winner_rank"], df["loser_rank"])

    records = []
    model = EloModel()
    match_history = defaultdict(list)
    h2h_record = defaultdict(lambda: {"wins_a": 0, "wins_b": 0})

    for _, row in df_clean.iterrows():
        p1, p2, surface = row["player1"], row["player2"], row["surface"]
        p1_win = row["p1_win"]
        r1, r2   = model.get_rating(p1), model.get_rating(p2)
        sr1, sr2 = model.get_rating(p1, surface), model.get_rating(p2, surface)
        rank1, rank2 = row.get("rank1", np.nan), row.get("rank2", np.nan)
        rank_diff = rank1 - rank2 if pd.notna(rank1) and pd.notna(rank2) else 0
        wr1 = compute_recent_win_rate(p1, match_history)
        wr2 = compute_recent_win_rate(p2, match_history)
        h2h = compute_h2h(p1, p2, h2h_record)

        records.append({
            "elo_diff":         r1 - r2,
            "surface_elo_diff": sr1 - sr2,
            "rank_diff":        rank_diff,
            "win_rate_diff":    wr1 - wr2,
            "h2h_p1":           h2h,
            "surface":          surface,
            "p1_win":           p1_win,
        })

        model.update(p1, p2, p1_win, surface=surface)
        match_history[p1].append(p1_win)
        match_history[p2].append(1 - p1_win)
        key = tuple(sorted([p1, p2]))
        if p1 == key[0]:
            h2h_record[key]["wins_a"] += p1_win
            h2h_record[key]["wins_b"] += (1 - p1_win)
        else:
            h2h_record[key]["wins_b"] += p1_win
            h2h_record[key]["wins_a"] += (1 - p1_win)

    features_df = pd.DataFrame(records)
    feature_cols = ["elo_diff", "surface_elo_diff", "rank_diff", "win_rate_diff", "h2h_p1"]
    X = pd.get_dummies(features_df[feature_cols + ["surface"]], columns=["surface"])
    y = features_df["p1_win"]

    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3)
    clf.fit(X, y)

    all_players = sorted(set(df_clean["player1"].tolist() + df_clean["player2"].tolist()))
    all_surfaces = features_df["surface"].unique().tolist()

    return model, clf, match_history, h2h_record, all_players, all_surfaces, X.columns.tolist()

# ---- UI ----
st.title("🎾 Tennis Match Predictor")
st.write("Predict the winner of any ATP match using Elo ratings and machine learning.")

with st.spinner("Training model... this takes ~30 seconds on first load"):
    model, clf, match_history, h2h_record, all_players, all_surfaces, feature_cols = train_model()

col1, col2 = st.columns(2)
with col1:
    p1 = st.selectbox("Player 1", all_players, index=0, placeholder="Search for a player...", key="p1")
with col2:
    p2 = st.selectbox("Player 2", all_players, index=1, placeholder="Search for a player...", key="p2")

surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])

if st.button("Predict", type="primary"):
    if p1 == p2:
        st.error("Please select two different players.")
    else:
        r1, r2   = model.get_rating(p1), model.get_rating(p2)
        sr1, sr2 = model.get_rating(p1, surface), model.get_rating(p2, surface)
        wr1 = compute_recent_win_rate(p1, match_history)
        wr2 = compute_recent_win_rate(p2, match_history)
        h2h = compute_h2h(p1, p2, h2h_record)

        row = {
            "elo_diff":         r1 - r2,
            "surface_elo_diff": sr1 - sr2,
            "rank_diff":        0,
            "win_rate_diff":    wr1 - wr2,
            "h2h_p1":           h2h,
        }
        for s in all_surfaces:
            row[f"surface_{s}"] = 1 if surface == s else 0

        input_df = pd.DataFrame([row])[feature_cols]
        prob = clf.predict_proba(input_df)[0][1]

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric(p1, f"{prob * 100:.1f}%")
        with col2:
            st.metric(p2, f"{(1 - prob) * 100:.1f}%")

        winner = p1 if prob > 0.5 else p2
        conf = max(prob, 1 - prob) * 100
        st.success(f"Predicted winner: **{winner}** ({conf:.1f}% confidence)")