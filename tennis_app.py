import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier

@st.cache_data
def load_data():
    years = range(2015, 2025)
    frames = []
    for year in years:
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        frames.append(pd.read_csv(url))
    df = pd.concat(frames).sort_values("tourney_date").reset_index(drop=True)
    return df

class EloModel:
    def __init__(self, initial_rating=1500):
        self.initial_rating = initial_rating
        self.ratings = {}
        self.surface_ratings = {}

    def get_rating(self, player, surface=None):
        if surface:
            return self.surface_ratings.setdefault(surface, {}).get(player, self.initial_rating)
        return self.ratings.get(player, self.initial_rating)

    def expected_score(self, r1, r2):
        return 1 / (1 + 10 ** ((r2 - r1) / 400))

    def update(self, p1, p2, p1_win, surface=None, k=32):
        r1, r2 = self.get_rating(p1), self.get_rating(p2)
        exp1 = self.expected_score(r1, r2)
        self.ratings[p1] = r1 + k * (p1_win - exp1)
        self.ratings[p2] = r2 + k * ((1 - p1_win) - (1 - exp1))
        if surface:
            sr1 = self.get_rating(p1, surface)
            sr2 = self.get_rating(p2, surface)
            exp_s = self.expected_score(sr1, sr2)
            self.surface_ratings[surface][p1] = sr1 + k * (p1_win - exp_s)
            self.surface_ratings[surface][p2] = sr2 + k * ((1 - p1_win) - (1 - exp_s))

def compute_recent_win_rate(player, match_history, n=20):
    matches = match_history[player][-n:]
    if not matches:
        return 0.5
    return sum(matches) / len(matches)

def compute_momentum(player, match_history, n=10):
    matches = match_history[player][-n:]
    if not matches:
        return 0.5
    weights = [i + 1 for i in range(len(matches))]
    return sum(w * m for w, m in zip(weights, matches)) / sum(weights)

def compute_fatigue(player, match_dates, current_date, days=14):
    recent = [d for d in match_dates[player] if (current_date - d).days <= days]
    return len(recent)

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

def compute_serve_score(player, serve_history, n=10):
    recent = serve_history[player][-n:]
    if not recent:
        return 0.5
    return sum(recent) / len(recent)

@st.cache_resource
def train_model():
    df = load_data()

    # Tournament level k-factor
    level_k = {"G": 50, "M": 40, "A": 32, "D": 24, "F": 32}

    np.random.seed(42)
    flip = np.random.rand(len(df)) < 0.5
    df_clean = pd.DataFrame({
        "player1":       np.where(flip, df["loser_name"],     df["winner_name"]),
        "player2":       np.where(flip, df["winner_name"],    df["loser_name"]),
        "surface":       df["surface"],
        "tourney_date":  df["tourney_date"],
        "tourney_level": df["tourney_level"],
        "p1_win":        np.where(flip, 0, 1),
    })
    df_clean["rank1"] = np.where(flip, df["loser_rank"],  df["winner_rank"])
    df_clean["rank2"] = np.where(flip, df["winner_rank"], df["loser_rank"])

    # Serve stats — first serve % won
    df_clean["serve1"] = np.where(flip, df["l_1stWon"] / (df["l_1stIn"] + 1),
                                        df["w_1stWon"] / (df["w_1stIn"] + 1))
    df_clean["serve2"] = np.where(flip, df["w_1stWon"] / (df["w_1stIn"] + 1),
                                        df["l_1stWon"] / (df["l_1stIn"] + 1))

    records = []
    model = EloModel()
    match_history = defaultdict(list)
    match_dates = defaultdict(list)
    h2h_record = defaultdict(lambda: {"wins_a": 0, "wins_b": 0})
    serve_history = defaultdict(list)

    for _, row in df_clean.iterrows():
        p1, p2, surface = row["player1"], row["player2"], row["surface"]
        p1_win = row["p1_win"]
        k = level_k.get(row["tourney_level"], 32)

        try:
            current_date = datetime.strptime(str(int(row["tourney_date"])), "%Y%m%d")
        except:
            current_date = datetime.today()

        # Snapshot all features BEFORE updating
        r1, r2   = model.get_rating(p1), model.get_rating(p2)
        sr1, sr2 = model.get_rating(p1, surface), model.get_rating(p2, surface)
        rank1, rank2 = row.get("rank1", np.nan), row.get("rank2", np.nan)
        rank_diff = rank1 - rank2 if pd.notna(rank1) and pd.notna(rank2) else 0
        wr1 = compute_recent_win_rate(p1, match_history)
        wr2 = compute_recent_win_rate(p2, match_history)
        mom1 = compute_momentum(p1, match_history)
        mom2 = compute_momentum(p2, match_history)
        fat1 = compute_fatigue(p1, match_dates, current_date)
        fat2 = compute_fatigue(p2, match_dates, current_date)
        h2h  = compute_h2h(p1, p2, h2h_record)
        sv1  = compute_serve_score(p1, serve_history)
        sv2  = compute_serve_score(p2, serve_history)

        records.append({
            "elo_diff":         r1 - r2,
            "surface_elo_diff": sr1 - sr2,
            "rank_diff":        rank_diff,
            "win_rate_diff":    wr1 - wr2,
            "momentum_diff":    mom1 - mom2,
            "fatigue_diff":     fat1 - fat2,
            "h2h_p1":           h2h,
            "serve_diff":       sv1 - sv2,
            "surface":          surface,
            "p1_win":           p1_win,
        })

        # Update everything AFTER snapshotting
        model.update(p1, p2, p1_win, surface=surface, k=k)
        match_history[p1].append(p1_win)
        match_history[p2].append(1 - p1_win)
        match_dates[p1].append(current_date)
        match_dates[p2].append(current_date)

        if pd.notna(row["serve1"]):
            serve_history[p1].append(row["serve1"])
        if pd.notna(row["serve2"]):
            serve_history[p2].append(row["serve2"])

        key = tuple(sorted([p1, p2]))
        if p1 == key[0]:
            h2h_record[key]["wins_a"] += p1_win
            h2h_record[key]["wins_b"] += (1 - p1_win)
        else:
            h2h_record[key]["wins_b"] += p1_win
            h2h_record[key]["wins_a"] += (1 - p1_win)

    features_df = pd.DataFrame(records)
    feature_cols = ["elo_diff", "surface_elo_diff", "rank_diff", "win_rate_diff",
                    "momentum_diff", "fatigue_diff", "h2h_p1", "serve_diff"]
    X = pd.get_dummies(features_df[feature_cols + ["surface"]], columns=["surface"])
    y = features_df["p1_win"]

    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3)
    clf.fit(X, y)

    all_players = sorted(set(df_clean["player1"].tolist() + df_clean["player2"].tolist()))
    all_surfaces = features_df["surface"].unique().tolist()

    return model, clf, match_history, match_dates, h2h_record, serve_history, all_players, all_surfaces, X.columns.tolist()

# ---- UI ----
st.title("🎾 Tennis Match Predictor")
st.write("Predict the winner of any ATP match using Elo ratings, momentum, fatigue, serve stats and machine learning.")

with st.spinner("Training model... this takes ~60 seconds on first load"):
    model, clf, match_history, match_dates, h2h_record, serve_history, all_players, all_surfaces, feature_cols = train_model()

col1, col2 = st.columns(2)
with col1:
    p1 = st.selectbox("Player 1", all_players, index=0, key="p1")
with col2:
    p2 = st.selectbox("Player 2", all_players, index=1, key="p2")

surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])

if st.button("Predict", type="primary"):
    if p1 == p2:
        st.error("Please select two different players.")
    else:
        r1, r2   = model.get_rating(p1), model.get_rating(p2)
        sr1, sr2 = model.get_rating(p1, surface), model.get_rating(p2, surface)
        wr1  = compute_recent_win_rate(p1, match_history)
        wr2  = compute_recent_win_rate(p2, match_history)
        mom1 = compute_momentum(p1, match_history)
        mom2 = compute_momentum(p2, match_history)
        fat1 = compute_fatigue(p1, match_dates, datetime.today())
        fat2 = compute_fatigue(p2, match_dates, datetime.today())
        h2h  = compute_h2h(p1, p2, h2h_record)
        sv1  = compute_serve_score(p1, serve_history)
        sv2  = compute_serve_score(p2, serve_history)

        row = {
            "elo_diff":         r1 - r2,
            "surface_elo_diff": sr1 - sr2,
            "rank_diff":        0,
            "win_rate_diff":    wr1 - wr2,
            "momentum_diff":    mom1 - mom2,
            "fatigue_diff":     fat1 - fat2,
            "h2h_p1":           h2h,
            "serve_diff":       sv1 - sv2,
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

        # Show breakdown
        with st.expander("See detailed breakdown"):
            st.write(f"**Overall Elo** — {p1}: {r1:.0f} | {p2}: {r2:.0f}")
            st.write(f"**{surface} Elo** — {p1}: {sr1:.0f} | {p2}: {sr2:.0f}")
            st.write(f"**Recent win rate** — {p1}: {wr1*100:.1f}% | {p2}: {wr2*100:.1f}%")
            st.write(f"**Momentum** — {p1}: {mom1*100:.1f}% | {p2}: {mom2*100:.1f}%")
            st.write(f"**Fatigue (matches last 2 weeks)** — {p1}: {fat1} | {p2}: {fat2}")
            st.write(f"**Serve score** — {p1}: {sv1*100:.1f}% | {p2}: {sv2*100:.1f}%")
            st.write(f"**H2H win rate for {p1}**: {h2h*100:.1f}%")