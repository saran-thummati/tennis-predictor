import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import requests

# ---- Data Loading ----
@st.cache_data(ttl=86400)
def load_data():
    years = range(2015, 2027)
    frames = []
    for year in years:
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        frames.append(pd.read_csv(url, low_memory=False))
    df = pd.concat(frames).sort_values("tourney_date").reset_index(drop=True)
    df = df[df["score"].notna()]
    df = df[~df["score"].str.contains("W/O|RET|DEF", na=False)]
    return df.reset_index(drop=True)

@st.cache_data(ttl=3600)
def fetch_upcoming_matches(api_key):
    today    = datetime.today().strftime("%Y-%m-%d")
    end_date = (datetime.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    url = f"https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey={api_key}&from={today}&to={end_date}&tour_id=2"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "result" in data and data["result"]:
            return data["result"]
        return []
    except:
        return []

# ---- EloModel with decay ----
class EloModel:
    def __init__(self, initial_rating=1500, decay=0.999):
        self.initial_rating = initial_rating
        self.decay          = decay
        self.ratings        = {}
        self.surface_ratings = {}
        self.match_counts   = defaultdict(int)

    def get_rating(self, player, surface=None):
        base = self.initial_rating
        if surface:
            raw = self.surface_ratings.setdefault(surface, {}).get(player, base)
            n   = self.match_counts[player]
            return base + (raw - base) * (self.decay ** max(0, 50 - n))
        raw = self.ratings.get(player, base)
        n   = self.match_counts[player]
        return base + (raw - base) * (self.decay ** max(0, 50 - n))

    def expected_score(self, r1, r2):
        return 1 / (1 + 10 ** ((r2 - r1) / 400))

    def update(self, p1, p2, p1_win, surface=None, k=32):
        r1, r2 = self.get_rating(p1), self.get_rating(p2)
        exp1   = self.expected_score(r1, r2)
        self.ratings[p1] = r1 + k * (p1_win - exp1)
        self.ratings[p2] = r2 + k * ((1 - p1_win) - (1 - exp1))
        self.match_counts[p1] += 1
        self.match_counts[p2] += 1
        if surface:
            sr1 = self.get_rating(p1, surface)
            sr2 = self.get_rating(p2, surface)
            exp_s = self.expected_score(sr1, sr2)
            self.surface_ratings[surface][p1] = sr1 + k * (p1_win - exp_s)
            self.surface_ratings[surface][p2] = sr2 + k * ((1 - p1_win) - (1 - exp_s))

# ---- Feature Functions ----
def compute_recent_win_rate(player, match_history, n=20):
    matches = match_history[player][-n:]
    if not matches:
        return 0.5
    return sum(matches) / len(matches)

def compute_surface_win_rate(player, surface, surface_history, n=20):
    matches = surface_history[player][surface][-n:]
    if not matches:
        return 0.5
    return sum(matches) / len(matches)

def compute_momentum(player, match_history, n=10):
    matches = match_history[player][-n:]
    if not matches:
        return 0.5
    weights = [i + 1 for i in range(len(matches))]
    return sum(w * m for w, m in zip(weights, matches)) / sum(weights)

def compute_streak(player, match_history):
    matches = match_history[player]
    if not matches:
        return 0
    streak = 0
    last   = matches[-1]
    for result in reversed(matches):
        if result == last:
            streak += 1
        else:
            break
    return streak if last == 1 else -streak

def compute_dominance(player, dominance_history, n=10):
    recent = dominance_history[player][-n:]
    if not recent:
        return 0.5
    return sum(recent) / len(recent)

def compute_tiebreak_rate(player, tb_history, n=20):
    recent = tb_history[player][-n:]
    if not recent:
        return 0.5
    return sum(recent) / len(recent)

def compute_round_win_rate(player, round_history, round_num):
    matches = round_history[player][round_num]
    if not matches:
        return 0.5
    return sum(matches) / len(matches)

def compute_h2h(p1, p2, h2h_record):
    key    = tuple(sorted([p1, p2]))
    record = h2h_record[key]
    total  = record["wins_a"] + record["wins_b"]
    if total == 0:
        return 0.5
    if p1 == key[0]:
        return record["wins_a"] / total
    return record["wins_b"] / total

def compute_surface_h2h(p1, p2, surface, h2h_surface):
    key    = tuple(sorted([p1, p2])) + (surface,)
    record = h2h_surface[key]
    total  = record["wins_a"] + record["wins_b"]
    if total == 0:
        return 0.5
    if p1 == tuple(sorted([p1, p2]))[0]:
        return record["wins_a"] / total
    return record["wins_b"] / total

def compute_recent_h2h(p1, p2, h2h_recent, n=5):
    key     = tuple(sorted([p1, p2]))
    matches = h2h_recent[key][-n:]
    if not matches:
        return 0.5
    p1_wins = sum(1 for m in matches if m["winner"] == p1)
    return p1_wins / len(matches)

def compute_serve_score(player, serve_history, n=10):
    recent = serve_history[player][-n:]
    if not recent:
        return 0.5
    return sum(recent) / len(recent)

def compute_upset_rate(player, upset_history):
    total = upset_history[player]["total"]
    wins  = upset_history[player]["wins"]
    if total == 0:
        return 0.5
    return wins / total

def compute_rank_trajectory(player, rank_history, n=10):
    ranks = rank_history[player][-n:]
    if len(ranks) < 2:
        return 0
    return ranks[0] - ranks[-1]

def compute_tournament_win_rate(player, tourney, tourney_history):
    matches = tourney_history[player][tourney]
    if not matches:
        return 0.5
    return sum(matches) / len(matches)

def apply_conditions_adjustment(prob, temp_val, wind_val):
    if temp_val >= 30:
        prob = prob * 0.97 + 0.03 * 0.5
    if wind_val >= 25:
        prob = prob * 0.95 + 0.05 * 0.5
    return prob

def apply_injury_adjustment(prob, p1_injured, p2_injured):
    if p1_injured and not p2_injured:
        prob = prob * 0.75
    elif p2_injured and not p1_injured:
        prob = prob + (1 - prob) * 0.25
    return prob

def apply_streak_adjustment(prob, streak1, streak2):
    diff = streak1 - streak2
    return max(0.05, min(0.95, prob + diff * 0.005))

def round_to_num(round_str):
    mapping = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7, "RR": 3}
    return mapping.get(str(round_str), 3)

def implied_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def confidence_tier(conf):
    if conf >= 70:
        return "🟢 High"
    elif conf >= 60:
        return "🟡 Medium"
    return "🔴 Low"

def parse_score(score_str):
    try:
        sets   = str(score_str).split()
        w_games = l_games = 0
        for s in sets:
            parts = s.replace("(", " ").replace(")", "").split("-")
            if len(parts) >= 2:
                w_games += int(parts[0])
                l_games += int(parts[1].split()[0])
        total = w_games + l_games
        return w_games / total if total > 0 else 0.5
    except:
        return 0.5

def get_prediction(p1, p2, surface, best_of, round_num, tourney,
                   p1_fatigue, p2_fatigue, p1_rest, p2_rest,
                   p1_injured, p2_injured, temp_val, wind_val, indoor,
                   model_elo, clfs, feature_cols, scaler,
                   match_history, surface_history, h2h_record, h2h_surface,
                   h2h_recent, serve_history, ace_history, df_history,
                   bp_history, upset_history, rank_history, tourney_history,
                   round_history, dominance_history, tb_history,
                   player_types, all_surfaces):

    r1, r2   = model_elo.get_rating(p1), model_elo.get_rating(p2)
    sr1, sr2 = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
    wr1  = compute_recent_win_rate(p1, match_history)
    wr2  = compute_recent_win_rate(p2, match_history)
    swr1 = compute_surface_win_rate(p1, surface, surface_history)
    swr2 = compute_surface_win_rate(p2, surface, surface_history)
    mom1 = compute_momentum(p1, match_history)
    mom2 = compute_momentum(p2, match_history)
    str1 = compute_streak(p1, match_history)
    str2 = compute_streak(p2, match_history)
    dom1 = compute_dominance(p1, dominance_history)
    dom2 = compute_dominance(p2, dominance_history)
    tb1  = compute_tiebreak_rate(p1, tb_history)
    tb2  = compute_tiebreak_rate(p2, tb_history)
    rwr1 = compute_round_win_rate(p1, round_history, round_num)
    rwr2 = compute_round_win_rate(p2, round_history, round_num)
    h2h  = compute_h2h(p1, p2, h2h_record)
    sh2h = compute_surface_h2h(p1, p2, surface, h2h_surface)
    rh2h = compute_recent_h2h(p1, p2, h2h_recent)
    sv1  = compute_serve_score(p1, serve_history)
    sv2  = compute_serve_score(p2, serve_history)
    ac1  = compute_serve_score(p1, ace_history)
    ac2  = compute_serve_score(p2, ace_history)
    df1  = compute_serve_score(p1, df_history)
    df2  = compute_serve_score(p2, df_history)
    bp1  = compute_serve_score(p1, bp_history)
    bp2  = compute_serve_score(p2, bp_history)
    up1  = compute_upset_rate(p1, upset_history)
    up2  = compute_upset_rate(p2, upset_history)
    rt1  = compute_rank_trajectory(p1, rank_history)
    rt2  = compute_rank_trajectory(p2, rank_history)
    tw1  = compute_tournament_win_rate(p1, tourney, tourney_history)
    tw2  = compute_tournament_win_rate(p2, tourney, tourney_history)
    pt1  = player_types.get(p1, 0)
    pt2  = player_types.get(p2, 0)

    row = {
        "elo_diff":              r1 - r2,
        "surface_elo_diff":      sr1 - sr2,
        "rank_diff":             0,
        "age_diff":              0,
        "win_rate_diff":         wr1 - wr2,
        "surface_win_rate_diff": swr1 - swr2,
        "momentum_diff":         mom1 - mom2,
        "fatigue_diff":          p1_fatigue - p2_fatigue,
        "rest_diff":             p1_rest - p2_rest,
        "h2h_p1":                h2h,
        "surface_h2h_p1":        sh2h,
        "recent_h2h_p1":         rh2h,
        "serve_diff":            sv1 - sv2,
        "ace_diff":              ac1 - ac2,
        "df_diff":               df1 - df2,
        "bp_diff":               bp1 - bp2,
        "upset_diff":            up1 - up2,
        "rank_traj_diff":        rt1 - rt2,
        "tourney_win_diff":      tw1 - tw2,
        "streak_diff":           str1 - str2,
        "round_win_rate_diff":   rwr1 - rwr2,
        "dominance_diff":        dom1 - dom2,
        "tiebreak_diff":         tb1 - tb2,
        "player_type_diff":      pt1 - pt2,
        "round":                 round_num,
        "best_of":               best_of,
        "indoor":                1 if indoor == "Indoor" else 0,
        "temp":                  temp_val,
        "wind":                  wind_val,
    }
    for s in all_surfaces:
        row[f"surface_{s}"] = 1 if surface == s else 0

    input_df = pd.DataFrame([row])[feature_cols]
    input_sc = scaler.transform(input_df)

    # Ensemble: average predictions from all models
    probs = []
    for name, clf_model in clfs.items():
        try:
            if name == "surface":
                p = clf_model.get(surface, clf_model.get("Hard"))
                if p:
                    probs.append(p.predict_proba(input_df)[0][1])
            elif name in ["mlp"]:
                probs.append(clf_model.predict_proba(input_sc)[0][1])
            else:
                probs.append(clf_model.predict_proba(input_df)[0][1])
        except:
            pass

    prob = np.mean(probs) if probs else 0.5
    prob = apply_injury_adjustment(prob, p1_injured, p2_injured)
    prob = apply_streak_adjustment(prob, str1, str2)
    prob = apply_conditions_adjustment(prob, temp_val, wind_val)

    return (prob, r1, r2, sr1, sr2, wr1, wr2, h2h, sh2h, rh2h,
            mom1, mom2, sv1, sv2, str1, str2, rwr1, rwr2,
            dom1, dom2, tb1, tb2)

# ---- Train Model ----
@st.cache_resource
def train_model():
    df = load_data()
    level_k   = {"G": 50, "M": 40, "A": 32, "D": 24, "F": 32}
    surface_k = {"Hard": 32, "Clay": 32, "Grass": 20, "Carpet": 20}
    np.random.seed(42)
    flip = np.random.rand(len(df)) < 0.5

    df_clean = pd.DataFrame({
        "player1":       np.where(flip, df["loser_name"],  df["winner_name"]),
        "player2":       np.where(flip, df["winner_name"], df["loser_name"]),
        "surface":       df["surface"],
        "tourney_date":  df["tourney_date"],
        "tourney_level": df["tourney_level"],
        "tourney_name":  df["tourney_name"],
        "round":         df["round"].apply(round_to_num),
        "best_of":       df["best_of"],
        "score":         df["score"],
        "p1_win":        np.where(flip, 0, 1),
    })
    df_clean["rank1"]  = np.where(flip, df["loser_rank"],  df["winner_rank"])
    df_clean["rank2"]  = np.where(flip, df["winner_rank"], df["loser_rank"])
    df_clean["age1"]   = np.where(flip, df["loser_age"],   df["winner_age"])
    df_clean["age2"]   = np.where(flip, df["winner_age"],  df["loser_age"])
    df_clean["serve1"] = np.where(flip, df["l_1stWon"] / (df["l_1stIn"] + 1), df["w_1stWon"] / (df["w_1stIn"] + 1))
    df_clean["serve2"] = np.where(flip, df["w_1stWon"] / (df["w_1stIn"] + 1), df["l_1stWon"] / (df["l_1stIn"] + 1))
    df_clean["ace1"]   = np.where(flip, df["l_ace"] / (df["l_svpt"] + 1), df["w_ace"] / (df["w_svpt"] + 1))
    df_clean["ace2"]   = np.where(flip, df["w_ace"] / (df["w_svpt"] + 1), df["l_ace"] / (df["l_svpt"] + 1))
    df_clean["df1"]    = np.where(flip, df["l_df"] / (df["l_svpt"] + 1), df["w_df"] / (df["w_svpt"] + 1))
    df_clean["df2"]    = np.where(flip, df["w_df"] / (df["w_svpt"] + 1), df["l_df"] / (df["l_svpt"] + 1))
    df_clean["bp1"]    = np.where(flip, df["l_bpSaved"] / (df["l_bpFaced"] + 1), df["w_bpSaved"] / (df["w_bpFaced"] + 1))
    df_clean["bp2"]    = np.where(flip, df["w_bpSaved"] / (df["w_bpFaced"] + 1), df["l_bpSaved"] / (df["l_bpFaced"] + 1))

    records          = []
    model_elo        = EloModel()
    match_history    = defaultdict(list)
    surface_history  = defaultdict(lambda: defaultdict(list))
    h2h_record       = defaultdict(lambda: {"wins_a": 0, "wins_b": 0})
    h2h_surface      = defaultdict(lambda: {"wins_a": 0, "wins_b": 0})
    h2h_recent       = defaultdict(list)
    serve_history    = defaultdict(list)
    ace_history      = defaultdict(list)
    df_history       = defaultdict(list)
    bp_history       = defaultdict(list)
    upset_history    = defaultdict(lambda: {"wins": 0, "total": 0})
    rank_history     = defaultdict(list)
    tourney_history  = defaultdict(lambda: defaultdict(list))
    round_history    = defaultdict(lambda: defaultdict(list))
    dominance_history = defaultdict(list)
    tb_history       = defaultdict(list)

    for _, row in df_clean.iterrows():
        p1, p2, surface = row["player1"], row["player2"], row["surface"]
        p1_win  = row["p1_win"]
        tourney = row["tourney_name"]
        rnd     = row["round"]
        k = (level_k.get(row["tourney_level"], 32) + surface_k.get(surface, 32)) / 2

        r1, r2   = model_elo.get_rating(p1), model_elo.get_rating(p2)
        sr1, sr2 = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
        rank1    = row.get("rank1", np.nan)
        rank2    = row.get("rank2", np.nan)
        rank_diff = rank1 - rank2 if pd.notna(rank1) and pd.notna(rank2) else 0
        age1     = row.get("age1", np.nan)
        age2     = row.get("age2", np.nan)
        age_diff = age1 - age2 if pd.notna(age1) and pd.notna(age2) else 0

        str1 = compute_streak(p1, match_history)
        str2 = compute_streak(p2, match_history)
        dom1 = compute_dominance(p1, dominance_history)
        dom2 = compute_dominance(p2, dominance_history)
        tb1  = compute_tiebreak_rate(p1, tb_history)
        tb2  = compute_tiebreak_rate(p2, tb_history)
        rwr1 = compute_round_win_rate(p1, round_history, rnd)
        rwr2 = compute_round_win_rate(p2, round_history, rnd)
        rh2h = compute_recent_h2h(p1, p2, h2h_recent)

        records.append({
            "elo_diff":              r1 - r2,
            "surface_elo_diff":      sr1 - sr2,
            "rank_diff":             rank_diff,
            "age_diff":              age_diff,
            "win_rate_diff":         compute_recent_win_rate(p1, match_history) - compute_recent_win_rate(p2, match_history),
            "surface_win_rate_diff": compute_surface_win_rate(p1, surface, surface_history) - compute_surface_win_rate(p2, surface, surface_history),
            "momentum_diff":         compute_momentum(p1, match_history) - compute_momentum(p2, match_history),
            "fatigue_diff":          0,
            "rest_diff":             0,
            "h2h_p1":                compute_h2h(p1, p2, h2h_record),
            "surface_h2h_p1":        compute_surface_h2h(p1, p2, surface, h2h_surface),
            "recent_h2h_p1":         rh2h,
            "serve_diff":            compute_serve_score(p1, serve_history) - compute_serve_score(p2, serve_history),
            "ace_diff":              compute_serve_score(p1, ace_history) - compute_serve_score(p2, ace_history),
            "df_diff":               compute_serve_score(p1, df_history) - compute_serve_score(p2, df_history),
            "bp_diff":               compute_serve_score(p1, bp_history) - compute_serve_score(p2, bp_history),
            "upset_diff":            compute_upset_rate(p1, upset_history) - compute_upset_rate(p2, upset_history),
            "rank_traj_diff":        compute_rank_trajectory(p1, rank_history) - compute_rank_trajectory(p2, rank_history),
            "tourney_win_diff":      compute_tournament_win_rate(p1, tourney, tourney_history) - compute_tournament_win_rate(p2, tourney, tourney_history),
            "streak_diff":           str1 - str2,
            "round_win_rate_diff":   rwr1 - rwr2,
            "dominance_diff":        dom1 - dom2,
            "tiebreak_diff":         tb1 - tb2,
            "player_type_diff":      0,
            "round":                 rnd,
            "best_of":               row.get("best_of", 3),
            "indoor":                0,
            "temp":                  20,
            "wind":                  10,
            "surface":               surface,
            "p1_win":                p1_win,
        })

        model_elo.update(p1, p2, p1_win, surface=surface, k=k)
        match_history[p1].append(p1_win)
        match_history[p2].append(1 - p1_win)
        surface_history[p1][surface].append(p1_win)
        surface_history[p2][surface].append(1 - p1_win)
        tourney_history[p1][tourney].append(p1_win)
        tourney_history[p2][tourney].append(1 - p1_win)
        round_history[p1][rnd].append(p1_win)
        round_history[p2][rnd].append(1 - p1_win)

        dom_score = parse_score(row["score"])
        dominance_history[p1].append(dom_score if p1_win else 1 - dom_score)
        dominance_history[p2].append(1 - dom_score if p1_win else dom_score)

        key  = tuple(sorted([p1, p2]))
        skey = key + (surface,)
        h2h_recent[key].append({"winner": p1 if p1_win else p2})
        if p1 == key[0]:
            h2h_record[key]["wins_a"]   += p1_win
            h2h_record[key]["wins_b"]   += (1 - p1_win)
            h2h_surface[skey]["wins_a"] += p1_win
            h2h_surface[skey]["wins_b"] += (1 - p1_win)
        else:
            h2h_record[key]["wins_b"]   += p1_win
            h2h_record[key]["wins_a"]   += (1 - p1_win)
            h2h_surface[skey]["wins_b"] += p1_win
            h2h_surface[skey]["wins_a"] += (1 - p1_win)

        if pd.notna(rank1): rank_history[p1].append(rank1)
        if pd.notna(rank2): rank_history[p2].append(rank2)
        if pd.notna(row["serve1"]): serve_history[p1].append(row["serve1"])
        if pd.notna(row["serve2"]): serve_history[p2].append(row["serve2"])
        if pd.notna(row["ace1"]): ace_history[p1].append(row["ace1"])
        if pd.notna(row["ace2"]): ace_history[p2].append(row["ace2"])
        if pd.notna(row["df1"]): df_history[p1].append(row["df1"])
        if pd.notna(row["df2"]): df_history[p2].append(row["df2"])
        if pd.notna(row["bp1"]): bp_history[p1].append(row["bp1"])
        if pd.notna(row["bp2"]): bp_history[p2].append(row["bp2"])

        if pd.notna(rank1) and pd.notna(rank2):
            if rank1 > rank2:
                upset_history[p1]["total"] += 1
                upset_history[p1]["wins"]  += p1_win
            if rank2 > rank1:
                upset_history[p2]["total"] += 1
                upset_history[p2]["wins"]  += (1 - p1_win)

    features_df  = pd.DataFrame(records)
    feature_cols = [
        "elo_diff", "surface_elo_diff", "rank_diff", "age_diff",
        "win_rate_diff", "surface_win_rate_diff", "momentum_diff",
        "fatigue_diff", "rest_diff", "h2h_p1", "surface_h2h_p1",
        "recent_h2h_p1", "serve_diff", "ace_diff", "df_diff", "bp_diff",
        "upset_diff", "rank_traj_diff", "tourney_win_diff",
        "streak_diff", "round_win_rate_diff", "dominance_diff",
        "tiebreak_diff", "player_type_diff", "round", "best_of",
        "indoor", "temp", "wind"
    ]
    X = pd.get_dummies(features_df[feature_cols + ["surface"]], columns=["surface"])
    y = features_df["p1_win"]

    # Player clustering
    all_players = sorted(set(df_clean["player1"].tolist() + df_clean["player2"].tolist()))
    player_stats_data = []
    for p in all_players:
        player_stats_data.append({
            "player":     p,
            "ace_rate":   compute_serve_score(p, ace_history),
            "serve_pct":  compute_serve_score(p, serve_history),
            "upset_rate": compute_upset_rate(p, upset_history),
            "dominance":  compute_dominance(p, dominance_history),
        })
    ps_df = pd.DataFrame(player_stats_data).set_index("player").fillna(0.5)
    scaler_cluster = StandardScaler()
    ps_scaled      = scaler_cluster.fit_transform(ps_df)
    kmeans         = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters       = kmeans.fit_predict(ps_scaled)
    player_types   = dict(zip(ps_df.index, clusters))

    for i, rec in enumerate(records):
        p1 = df_clean.iloc[i]["player1"]
        p2 = df_clean.iloc[i]["player2"]
        features_df.at[i, "player_type_diff"] = player_types.get(p1, 0) - player_types.get(p2, 0)

    X = pd.get_dummies(features_df[feature_cols + ["surface"]], columns=["surface"])

    # Recency weights
    n              = len(features_df)
    sample_weights = np.linspace(0.3, 1.0, n)

    # Scaler for neural network
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # Train/test split
    split     = int(n * 0.8)
    X_train   = X.iloc[:split]
    X_train_sc = X_sc[:split]
    y_train   = y.iloc[:split]
    X_test    = X.iloc[split:]
    X_test_sc  = X_sc[split:]
    y_test    = y.iloc[split:]
    sw_train  = sample_weights[:split]

    # 1. Gradient Boosting
    gb_clf = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, subsample=0.8, min_samples_leaf=5
    )
    gb_clf.fit(X_train, y_train, sample_weight=sw_train)
    gb_cal = CalibratedClassifierCV(gb_clf, method="isotonic", cv="prefit")
    gb_cal.fit(X_train, y_train)

    # 2. Random Forest
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    rf_clf.fit(X_train, y_train, sample_weight=sw_train)
    rf_cal = CalibratedClassifierCV(rf_clf, method="isotonic", cv="prefit")
    rf_cal.fit(X_train, y_train)

    # 3. Logistic Regression
    lr_clf = LogisticRegression(max_iter=1000)
    lr_clf.fit(X_train_sc, y_train, sample_weight=sw_train)

    # 4. Neural Network
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu", learning_rate_init=0.001,
        max_iter=300, random_state=42
    )
    mlp_clf.fit(X_train_sc, y_train)

    # 5. Surface specific models
    surface_models = {}
    for surf in ["Hard", "Clay", "Grass"]:
        mask = features_df["surface"] == surf
        if mask.sum() > 100:
            X_s  = X[mask]
            y_s  = y[mask]
            sw_s = sample_weights[mask.values]
            clf_s = GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=3
            )
            clf_s.fit(X_s, y_s, sample_weight=sw_s)
            surface_models[surf] = clf_s

    clfs = {
        "gb":      gb_cal,
        "rf":      rf_cal,
        "lr":      lr_clf,
        "mlp":     mlp_clf,
        "surface": surface_models,
    }

    # Backtest accuracy
    gb_preds  = gb_cal.predict(X_test)
    backtest_acc = (gb_preds == y_test).mean()

    # Detailed backtest
    all_probs = []
    for i in range(len(X_test)):
        row_df  = X_test.iloc[[i]]
        row_sc  = X_test_sc[[i]]
        p_list  = []
        p_list.append(gb_cal.predict_proba(row_df)[0][1])
        p_list.append(rf_cal.predict_proba(row_df)[0][1])
        p_list.append(lr_clf.predict_proba(row_sc)[0][1])
        p_list.append(mlp_clf.predict_proba(row_sc)[0][1])
        surf_val = features_df["surface"].iloc[split + i]
        if surf_val in surface_models:
            p_list.append(surface_models[surf_val].predict_proba(row_df)[0][1])
        all_probs.append(np.mean(p_list))

    test_df             = features_df.iloc[split:].copy()
    test_df["prob"]     = all_probs
    test_df["conf"]     = test_df["prob"].apply(lambda x: max(x, 1-x))
    test_df["correct"]  = (test_df["prob"] > 0.5) == (test_df["p1_win"] == 1)
    ensemble_acc        = test_df["correct"].mean()
    test_df["tier"]     = test_df["conf"].apply(
        lambda x: "🟢 High (70%+)" if x >= 0.7 else ("🟡 Medium (60-70%)" if x >= 0.6 else "🔴 Low (<60%)")
    )

    all_surfaces = features_df["surface"].unique().tolist()
    all_tourneys = sorted(df_clean["tourney_name"].unique().tolist())

    return (model_elo, clfs, scaler, match_history, surface_history,
            h2h_record, h2h_surface, h2h_recent, serve_history, ace_history,
            df_history, bp_history, upset_history, rank_history,
            tourney_history, round_history, dominance_history, tb_history,
            player_types, all_players, all_surfaces, all_tourneys,
            X.columns.tolist(), backtest_acc, ensemble_acc, test_df, X)

# ---- App ----
st.set_page_config(page_title="Tennis Match Predictor", page_icon="🎾", layout="wide")
st.markdown("""
<style>
.stMetric { background-color: #1e2130; padding: 10px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🎾 Tennis Match Predictor")
st.caption("Professional ATP predictions — Ensemble ML, Neural Network, Surface Models, Player Clustering")

with st.spinner("Training model... ~2 minutes on first load"):
    (model_elo, clfs, scaler, match_history, surface_history,
     h2h_record, h2h_surface, h2h_recent, serve_history, ace_history,
     df_history, bp_history, upset_history, rank_history,
     tourney_history, round_history, dominance_history, tb_history,
     player_types, all_players, all_surfaces, all_tourneys,
     feature_cols, backtest_acc, ensemble_acc, test_df, X_train) = train_model()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Predict", "📅 Calendar", "📊 Odds", "👤 Players", "📈 Backtest", "📋 Track Record"
])

# ======== TAB 1: Predict ========
with tab1:
    st.subheader("Predict a Match")
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Player 1", all_players, index=0, key="p1")
    with col2:
        p2 = st.selectbox("Player 2", all_players, index=1, key="p2")

    surface   = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
    tourney   = st.selectbox("Tournament", ["Unknown"] + all_tourneys)
    best_of   = st.radio("Best of", [3, 5], horizontal=True)
    round_num = st.select_slider("Round", options=[1,2,3,4,5,6,7],
                                  format_func=lambda x: ["R128","R64","R32","R16","QF","SF","F"][x-1])

    st.subheader("Conditions")
    col1, col2, col3 = st.columns(3)
    with col1:
        temp_val = st.number_input("Temperature (°C)", -10, 50, 20)
        st.caption(f"{'🔥 Hot' if temp_val >= 30 else ('❄️ Cold' if temp_val <= 10 else '🌤 Mild')}")
    with col2:
        wind_val = st.number_input("Wind speed (km/h)", 0, 100, 10)
        st.caption(f"{'💨 Windy' if wind_val >= 25 else '🌬 Calm'}")
    with col3:
        indoor = st.selectbox("Venue", ["Outdoor", "Indoor"])

    st.subheader("Fatigue & Rest")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{p1}**")
        p1_fat     = st.number_input("Matches in last 14 days", 0, 15, 0, key="f1")
        p1_rest    = st.number_input("Days since last match",   0, 365, 3, key="r1")
        p1_injured = st.checkbox(f"🚑 {p1} injured", key="inj1")
    with col2:
        st.write(f"**{p2}**")
        p2_fat     = st.number_input("Matches in last 14 days", 0, 15, 0, key="f2")
        p2_rest    = st.number_input("Days since last match",   0, 365, 3, key="r2")
        p2_injured = st.checkbox(f"🚑 {p2} injured", key="inj2")

    if st.button("Predict", type="primary"):
        if p1 == p2:
            st.error("Select two different players.")
        else:
            (prob, r1, r2, sr1, sr2, wr1, wr2, h2h, sh2h, rh2h,
             mom1, mom2, sv1, sv2, str1, str2, rwr1, rwr2,
             dom1, dom2, tb1, tb2) = get_prediction(
                p1, p2, surface, best_of, round_num, tourney,
                p1_fat, p2_fat, p1_rest, p2_rest,
                p1_injured, p2_injured, temp_val, wind_val, indoor,
                model_elo, clfs, feature_cols, scaler,
                match_history, surface_history, h2h_record, h2h_surface,
                h2h_recent, serve_history, ace_history, df_history,
                bp_history, upset_history, rank_history, tourney_history,
                round_history, dominance_history, tb_history,
                player_types, all_surfaces
            )
            winner = p1 if prob > 0.5 else p2
            conf   = max(prob, 1-prob) * 100
            tier   = confidence_tier(conf)

            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1: st.metric(p1, f"{prob*100:.1f}%")
            with col2: st.metric("Confidence", tier)
            with col3: st.metric(p2, f"{(1-prob)*100:.1f}%")
            st.success(f"🏆 Predicted winner: **{winner}** ({conf:.1f}% confidence)")

            if p1_injured: st.warning(f"⚠️ Injury penalty applied for {p1}")
            if p2_injured: st.warning(f"⚠️ Injury penalty applied for {p2}")

            pt_labels = {0: "Big Server", 1: "Grinder", 2: "All-Courter", 3: "Upset Specialist"}
            st.info(f"Player types — {p1}: **{pt_labels.get(player_types.get(p1,0), 'Unknown')}** | {p2}: **{pt_labels.get(player_types.get(p2,0), 'Unknown')}**")

            with st.expander("Detailed breakdown"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{p1}**")
                    st.write(f"Overall Elo: {r1:.0f}")
                    st.write(f"{surface} Elo: {sr1:.0f}")
                    st.write(f"Win rate: {wr1*100:.1f}%")
                    st.write(f"Momentum: {mom1*100:.1f}%")
                    st.write(f"Dominance: {dom1*100:.1f}%")
                    st.write(f"Tiebreak rate: {tb1*100:.1f}%")
                    st.write(f"Serve %: {sv1*100:.1f}%")
                    st.write(f"Streak: {str1:+d}")
                    st.write(f"Round {round_num} win rate: {rwr1*100:.1f}%")
                with col2:
                    st.write(f"**{p2}**")
                    st.write(f"Overall Elo: {r2:.0f}")
                    st.write(f"{surface} Elo: {sr2:.0f}")
                    st.write(f"Win rate: {wr2*100:.1f}%")
                    st.write(f"Momentum: {mom2*100:.1f}%")
                    st.write(f"Dominance: {dom2*100:.1f}%")
                    st.write(f"Tiebreak rate: {tb2*100:.1f}%")
                    st.write(f"Serve %: {sv2*100:.1f}%")
                    st.write(f"Streak: {str2:+d}")
                    st.write(f"Round {round_num} win rate: {rwr2*100:.1f}%")
                st.write(f"**Overall H2H for {p1}**: {h2h*100:.1f}%")
                st.write(f"**{surface} H2H for {p1}**: {sh2h*100:.1f}%")
                st.write(f"**Recent H2H (last 5) for {p1}**: {rh2h*100:.1f}%")

# ======== TAB 2: Calendar ========
with tab2:
    st.subheader("📅 Upcoming ATP Matches")
    api_key = st.secrets.get("TENNIS_API_KEY", "")
    if not api_key:
        api_key = st.text_input("API key from api-tennis.com", type="password", key="cal_api")

    if "manual_matches" not in st.session_state:
        st.session_state.manual_matches = []

    st.subheader("Add Match Manually")
    with st.form("add_match"):
        c1, c2, c3, c4 = st.columns(4)
        with c1: m_date    = st.date_input("Date")
        with c2: m_p1      = st.selectbox("Player 1", all_players, key="mp1")
        with c3: m_p2      = st.selectbox("Player 2", all_players, index=1, key="mp2")
        with c4: m_surface = st.selectbox("Surface", ["Hard","Clay","Grass"], key="msurf")
        m_tourney = st.text_input("Tournament")
        m_bo      = st.radio("Best of", [3, 5], horizontal=True, key="mbo")
        if st.form_submit_button("Add") and m_p1 != m_p2:
            st.session_state.manual_matches.append({
                "date": m_date, "player1": m_p1, "player2": m_p2,
                "surface": m_surface, "tourney": m_tourney or "Unknown", "best_of": m_bo
            })

    all_cal = list(st.session_state.manual_matches)
    if api_key:
        with st.spinner("Fetching..."):
            api_m = fetch_upcoming_matches(api_key)
        for m in api_m:
            try:
                p1n = m.get("event_first_player","")
                p2n = m.get("event_second_player","")
                p1m = next((p for p in all_players if p1n.split()[-1].lower() in p.lower()), None)
                p2m = next((p for p in all_players if p2n.split()[-1].lower() in p.lower()), None)
                if p1m and p2m and p1m != p2m:
                    all_cal.append({
                        "date":    datetime.strptime(m.get("event_date",""), "%Y-%m-%d").date(),
                        "player1": p1m, "player2": p2m,
                        "surface": m.get("event_surface","Hard"),
                        "tourney": m.get("tournament_name","Unknown"), "best_of": 3
                    })
            except:
                continue

    if all_cal:
        by_date = defaultdict(list)
        for m in all_cal:
            by_date[m["date"]].append(m)
        for d in sorted(by_date.keys()):
            st.markdown(f"### 📆 {d.strftime('%A, %B %d %Y') if hasattr(d,'strftime') else d}")
            for m in by_date[d]:
                prob, *_ = get_prediction(
                    m["player1"], m["player2"], m["surface"], m["best_of"], 3, m["tourney"],
                    0, 0, 3, 3, False, False, 20, 10, "Outdoor",
                    model_elo, clfs, feature_cols, scaler,
                    match_history, surface_history, h2h_record, h2h_surface,
                    h2h_recent, serve_history, ace_history, df_history,
                    bp_history, upset_history, rank_history, tourney_history,
                    round_history, dominance_history, tb_history,
                    player_types, all_surfaces
                )
                winner = m["player1"] if prob > 0.5 else m["player2"]
                conf   = max(prob, 1-prob) * 100
                c1, c2, c3, c4 = st.columns([3,3,2,2])
                with c1:
                    st.write(f"🎾 **{m['player1']}** vs **{m['player2']}**")
                    st.caption(f"{m['tourney']} | {m['surface']}")
                with c2:
                    st.write(f"{m['player1']}: **{prob*100:.1f}%** | {m['player2']}: **{(1-prob)*100:.1f}%**")
                with c3:
                    st.write(f"🏆 **{winner}**")
                with c4:
                    st.write(f"{confidence_tier(conf)} | {conf:.1f}%")
                st.divider()
        if st.button("Clear manual matches"):
            st.session_state.manual_matches = []
            st.rerun()

# ======== TAB 3: Odds ========
with tab3:
    st.subheader("📊 Find Value Bets")
    col1, col2 = st.columns(2)
    with col1: op1 = st.selectbox("Player 1", all_players, key="op1")
    with col2: op2 = st.selectbox("Player 2", all_players, index=1, key="op2")

    osurf  = st.selectbox("Surface", ["Hard","Clay","Grass"], key="os")
    obo    = st.radio("Best of", [3, 5], horizontal=True, key="obo")
    oround = st.select_slider("Round", options=[1,2,3,4,5,6,7],
                               format_func=lambda x: ["R128","R64","R32","R16","QF","SF","F"][x-1], key="ornd")

    col1, col2 = st.columns(2)
    with col1: odds1 = st.number_input(f"{op1} odds", value=-150, key="o1")
    with col2: odds2 = st.number_input(f"{op2} odds", value=120,  key="o2")

    col1, col2 = st.columns(2)
    with col1:
        of1 = st.number_input("Matches last 14 days", 0, 15, 0, key="of1")
        or1 = st.number_input("Days rest", 0, 365, 3, key="orr1")
        oi1 = st.checkbox(f"{op1} injured", key="oi1")
    with col2:
        of2 = st.number_input("Matches last 14 days", 0, 15, 0, key="of2")
        or2 = st.number_input("Days rest", 0, 365, 3, key="orr2")
        oi2 = st.checkbox(f"{op2} injured", key="oi2")

    if st.button("Find Value", type="primary"):
        if op1 == op2:
            st.error("Select two different players.")
        else:
            prob, *_ = get_prediction(
                op1, op2, osurf, obo, oround, "Unknown",
                of1, of2, or1, or2, oi1, oi2, 20, 10, "Outdoor",
                model_elo, clfs, feature_cols, scaler,
                match_history, surface_history, h2h_record, h2h_surface,
                h2h_recent, serve_history, ace_history, df_history,
                bp_history, upset_history, rank_history, tourney_history,
                round_history, dominance_history, tb_history,
                player_types, all_surfaces
            )
            imp1  = implied_prob(odds1)
            imp2  = implied_prob(odds2)
            diff1 = prob - imp1
            diff2 = (1-prob) - imp2

            col1, col2 = st.columns(2)
            with col1: st.metric(op1, f"Model: {prob*100:.1f}%", delta=f"Vegas: {imp1*100:.1f}%")
            with col2: st.metric(op2, f"Model: {(1-prob)*100:.1f}%", delta=f"Vegas: {imp2*100:.1f}%")

            if diff1 > 0.05:
                st.success(f"✅ **{op1}** undervalued — edge +{diff1*100:.1f}%")
            elif diff1 < -0.05:
                st.warning(f"⚠️ **{op1}** overvalued by {abs(diff1)*100:.1f}%")
            if diff2 > 0.05:
                st.success(f"✅ **{op2}** undervalued — edge +{diff2*100:.1f}%")
            elif diff2 < -0.05:
                st.warning(f"⚠️ **{op2}** overvalued by {abs(diff2)*100:.1f}%")
            if abs(diff1) <= 0.05 and abs(diff2) <= 0.05:
                st.info("No significant edge detected.")

# ======== TAB 4: Players ========
with tab4:
    st.subheader("👤 Player Profile")
    search = st.selectbox("Search player", all_players, key="search")

    if search:
        r_ov   = model_elo.get_rating(search)
        r_h    = model_elo.get_rating(search, "Hard")
        r_c    = model_elo.get_rating(search, "Clay")
        r_g    = model_elo.get_rating(search, "Grass")
        wr     = compute_recent_win_rate(search, match_history)
        mom    = compute_momentum(search, match_history)
        sv     = compute_serve_score(search, serve_history)
        ac     = compute_serve_score(search, ace_history)
        bp     = compute_serve_score(search, bp_history)
        up     = compute_upset_rate(search, upset_history)
        rt     = compute_rank_trajectory(search, rank_history)
        streak = compute_streak(search, match_history)
        dom    = compute_dominance(search, dominance_history)
        tb     = compute_tiebreak_rate(search, tb_history)
        pt     = player_types.get(search, 0)
        pt_labels = {0: "Big Server", 1: "Grinder", 2: "All-Courter", 3: "Upset Specialist"}

        st.info(f"Player type: **{pt_labels.get(pt, 'Unknown')}**")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Elo", f"{r_ov:.0f}")
        col2.metric("Hard Elo",    f"{r_h:.0f}")
        col3.metric("Clay Elo",    f"{r_c:.0f}")
        col4.metric("Grass Elo",   f"{r_g:.0f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Win Rate",     f"{wr*100:.1f}%")
        col2.metric("Momentum",     f"{mom*100:.1f}%")
        col3.metric("Dominance",    f"{dom*100:.1f}%")
        col4.metric("Tiebreak %",   f"{tb*100:.1f}%")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Serve %",      f"{sv*100:.1f}%")
        col2.metric("Ace Rate",     f"{ac*100:.1f}%")
        col3.metric("Upset Rate",   f"{up*100:.1f}%")
        col4.metric("Streak",       f"{streak:+d}")

        st.subheader("Win rate by round")
        round_labels = {1:"R128",2:"R64",3:"R32",4:"R16",5:"QF",6:"SF",7:"F"}
        round_data   = {round_labels[r]: compute_round_win_rate(search, round_history, r)*100 for r in range(1,8)}
        st.bar_chart(round_data)

        st.subheader("Head to head")
        vs_player = st.selectbox("Compare vs", [p for p in all_players if p != search], key="vs")
        if vs_player:
            h2h_ov  = compute_h2h(search, vs_player, h2h_record)
            h2h_h   = compute_surface_h2h(search, vs_player, "Hard",  h2h_surface)
            h2h_c   = compute_surface_h2h(search, vs_player, "Clay",  h2h_surface)
            h2h_g   = compute_surface_h2h(search, vs_player, "Grass", h2h_surface)
            h2h_rec = compute_recent_h2h(search, vs_player, h2h_recent)
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Overall", f"{h2h_ov*100:.1f}%")
            col2.metric("Hard",    f"{h2h_h*100:.1f}%")
            col3.metric("Clay",    f"{h2h_c*100:.1f}%")
            col4.metric("Grass",   f"{h2h_g*100:.1f}%")
            col5.metric("Recent",  f"{h2h_rec*100:.1f}%")

# ======== TAB 5: Backtest ========
with tab5:
    st.subheader("📈 Model Backtest Report")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Single Model Accuracy", f"{backtest_acc*100:.1f}%")
    col2.metric("Ensemble Accuracy",     f"{ensemble_acc*100:.1f}%")
    col3.metric("Baseline (rank)",       "~65.0%")
    col4.metric("Edge over baseline",    f"+{(ensemble_acc-0.65)*100:.1f}%")

    st.subheader("Accuracy by confidence tier")
    tier_stats = test_df.groupby("tier").agg(
        Predictions=("correct","count"),
        Accuracy=("correct","mean")
    ).reset_index()
    tier_stats["Accuracy"] = (tier_stats["Accuracy"]*100).round(1).astype(str) + "%"
    st.dataframe(tier_stats, use_container_width=True)

    st.subheader("Accuracy by surface")
    surf_stats = test_df.groupby("surface").agg(
        Predictions=("correct","count"),
        Accuracy=("correct","mean")
    ).reset_index()
    surf_stats["Accuracy"] = (surf_stats["Accuracy"]*100).round(1).astype(str) + "%"
    st.dataframe(surf_stats, use_container_width=True)

    st.subheader("Feature importance (Gradient Boosting)")
    gb_model  = clfs["gb"]
    try:
        importances = gb_model.calibrated_classifiers_[0].estimator.feature_importances_
        feat_imp    = pd.DataFrame({
            "Feature":    X_train.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)
        st.bar_chart(feat_imp.set_index("Feature"))
    except:
        st.info("Feature importance unavailable for calibrated model.")

# ======== TAB 6: Track Record ========
with tab6:
    st.subheader("📋 Prediction Track Record")

    if "track_record" not in st.session_state:
        st.session_state.track_record = []

    with st.form("log_pred"):
        c1, c2, c3 = st.columns(3)
        with c1:
            t_date  = st.date_input("Date", key="td")
            t_p1    = st.selectbox("Player 1", all_players, key="tp1")
            t_p2    = st.selectbox("Player 2", all_players, index=1, key="tp2")
        with c2:
            t_surf   = st.selectbox("Surface", ["Hard","Clay","Grass"], key="tsurf")
            t_tourney = st.text_input("Tournament", key="tt")
            t_pick   = st.selectbox("Your pick", ["Player 1","Player 2"], key="tpick")
        with c3:
            t_conf   = st.number_input("Confidence %", 50.0, 100.0, 65.0, key="tc")
            t_result = st.selectbox("Result", ["Pending","Correct","Wrong"], key="tr")
            t_odds   = st.number_input("Vegas odds", value=-110, key="to")

        if st.form_submit_button("Log") and t_p1 != t_p2:
            st.session_state.track_record.append({
                "Date":       str(t_date),
                "Player 1":   t_p1,
                "Player 2":   t_p2,
                "Surface":    t_surf,
                "Tournament": t_tourney,
                "Pick":       t_p1 if t_pick == "Player 1" else t_p2,
                "Confidence": f"{t_conf:.1f}%",
                "Tier":       confidence_tier(t_conf),
                "Odds":       t_odds,
                "Result":     t_result,
            })

    if st.session_state.track_record:
        df_track = pd.DataFrame(st.session_state.track_record)
        st.dataframe(df_track, use_container_width=True)

        decided = df_track[df_track["Result"] != "Pending"]
        if len(decided) > 0:
            correct = (decided["Result"] == "Correct").sum()
            total   = len(decided)
            acc     = correct / total * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("Total",    total)
            col2.metric("Correct",  correct)
            col3.metric("Accuracy", f"{acc:.1f}%")

            high = decided[decided["Tier"] == "🟢 High"]
            med  = decided[decided["Tier"] == "🟡 Medium"]
            if len(high) > 0:
                st.metric("🟢 High confidence accuracy",
                          f"{(high['Result']=='Correct').mean()*100:.1f}%")
            if len(med) > 0:
                st.metric("🟡 Medium confidence accuracy",
                          f"{(med['Result']=='Correct').mean()*100:.1f}%")

        csv = df_track.to_csv(index=False)
        st.download_button("📥 Export CSV", csv, "track_record.csv", "text/csv")
        if st.button("Clear all"):
            st.session_state.track_record = []
            st.rerun()
    else:
        st.info("No predictions logged yet.")
