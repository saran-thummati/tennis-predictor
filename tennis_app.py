import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
import plotly.express as px
import streamlit.components.v1 as components
import google.generativeai as genai
import json

# ==========================================
# 1. DATA LOADING & API SETUP
# ==========================================
@st.cache_data(ttl=86400)
def load_data():
    years = range(2015, 2027)
    frames = []
    for year in years:
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        try:
            frames.append(pd.read_csv(url, low_memory=False))
        except Exception:
            continue # Silently skip years that don't exist yet
            
    if not frames:
        return pd.DataFrame()
        
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

# ==========================================
# 2. GEMINI AI AUTO-FILL ENGINE
# ==========================================
def get_match_context_from_gemini(p1, p2, tourney_hint="Current ATP event"):
    """Uses Gemini to estimate real-world conditions and fatigue for a matchup."""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are a professional tennis analyst. I am simulating a match between {p1} and {p2} at {tourney_hint}.
        Based on your knowledge of tennis, current events, and the typical conditions at this tournament, 
        provide estimated realistic values for the following parameters. 
        
        Return ONLY a valid JSON object with these exact keys and format:
        {{
            "surface": "Hard", // Must be "Hard", "Clay", or "Grass"
            "indoor": "Outdoor", // Must be "Indoor" or "Outdoor"
            "best_of": 3, // 3 or 5
            "round_num": 5, // 1 to 7 (1=R128, 2=R64, 3=R32, 4=R16, 5=QF, 6=SF, 7=Final)
            "temp_celsius": 24, // Integer, estimated temperature
            "wind_kmh": 12, // Integer, estimated wind speed
            "p1_fatigue": 2, // Integer 0-15: Matches played in last 14 days
            "p1_rest_days": 3, // Integer: Days since last match
            "p1_injured": false, // Boolean
            "p2_fatigue": 1, // Integer 0-15: Matches played in last 14 days
            "p2_rest_days": 4, // Integer: Days since last match
            "p2_injured": false // Boolean
        }}
        """
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        return None

# ==========================================
# 3. MACHINE LEARNING & ELO ENGINE
# ==========================================
class EloModel:
    def __init__(self, initial_rating=1500):
        self.initial_rating  = initial_rating
        self.ratings         = {}
        self.surface_ratings = {}
        self.match_counts    = defaultdict(int)

    def get_rating(self, player, surface=None):
        base = self.initial_rating
        if surface: raw = self.surface_ratings.setdefault(surface, {}).get(player, base)
        else: raw = self.ratings.get(player, base)
        n = self.match_counts[player]
        decay = 0.999 ** max(0, 50 - n)
        return base + (raw - base) * decay

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

def compute_recent_win_rate(player, match_history, n=20):
    matches = match_history[player][-n:]
    return sum(matches) / len(matches) if matches else 0.5

def compute_surface_win_rate(player, surface, surface_history, n=20):
    matches = surface_history[player][surface][-n:]
    return sum(matches) / len(matches) if matches else 0.5

def compute_momentum(player, match_history, n=10):
    matches = match_history[player][-n:]
    if not matches: return 0.5
    weights = [i + 1 for i in range(len(matches))]
    return sum(w * m for w, m in zip(weights, matches)) / sum(weights)

def compute_streak(player, match_history):
    matches = match_history[player]
    if not matches: return 0
    streak = 0
    last   = matches[-1]
    for result in reversed(matches):
        if result == last: streak += 1
        else: break
    return streak if last == 1 else -streak

def compute_dominance(player, dominance_history, n=10):
    recent = dominance_history[player][-n:]
    return sum(recent) / len(recent) if recent else 0.5

def compute_tiebreak_rate(player, tb_history, n=20):
    recent = tb_history[player][-n:]
    return sum(recent) / len(recent) if recent else 0.5

def compute_round_win_rate(player, round_history, round_num):
    matches = round_history[player][round_num]
    return sum(matches) / len(matches) if matches else 0.5

def compute_h2h(p1, p2, h2h_record):
    key    = tuple(sorted([p1, p2]))
    record = h2h_record[key]
    total  = record["wins_a"] + record["wins_b"]
    if total == 0: return 0.5
    if p1 == key[0]: return record["wins_a"] / total
    return record["wins_b"] / total

def compute_surface_h2h(p1, p2, surface, h2h_surface):
    key    = tuple(sorted([p1, p2])) + (surface,)
    record = h2h_surface[key]
    total  = record["wins_a"] + record["wins_b"]
    if total == 0: return 0.5
    if p1 == tuple(sorted([p1, p2]))[0]: return record["wins_a"] / total
    return record["wins_b"] / total

def compute_recent_h2h(p1, p2, h2h_recent, n=5):
    key     = tuple(sorted([p1, p2]))
    matches = h2h_recent[key][-n:]
    if not matches: return 0.5
    return sum(1 for m in matches if m["winner"] == p1) / len(matches)

def compute_serve_score(player, serve_history, n=10):
    recent = serve_history[player][-n:]
    return sum(recent) / len(recent) if recent else 0.5

def compute_upset_rate(player, upset_history):
    total = upset_history[player]["total"]
    wins  = upset_history[player]["wins"]
    return wins / total if total > 0 else 0.5

def compute_rank_trajectory(player, rank_history, n=10):
    ranks = rank_history[player][-n:]
    if len(ranks) < 2: return 0
    return ranks[0] - ranks[-1]

def compute_tournament_win_rate(player, tourney, tourney_history):
    matches = tourney_history[player][tourney]
    return sum(matches) / len(matches) if matches else 0.5

def parse_score(score_str):
    try:
        sets = str(score_str).split()
        w_games = l_games = 0
        for s in sets:
            parts = s.replace("(", " ").replace(")", "").split("-")
            if len(parts) >= 2:
                w_games += int(parts[0])
                l_games += int(parts[1].split()[0])
        total = w_games + l_games
        return w_games / total if total > 0 else 0.5
    except: return 0.5

def apply_adjustments(prob, p1_injured, p2_injured, str1, str2, temp_val, wind_val):
    if p1_injured and not p2_injured: prob *= 0.75
    elif p2_injured and not p1_injured: prob = prob + (1 - prob) * 0.25
    prob = max(0.05, min(0.95, prob + (str1 - str2) * 0.005))
    if temp_val >= 30: prob = prob * 0.97 + 0.03 * 0.5
    if wind_val >= 25: prob = prob * 0.95 + 0.05 * 0.5
    return prob

def round_to_num(round_str):
    mapping = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7, "RR": 3}
    return mapping.get(str(round_str), 3)

def implied_prob(odds):
    if odds > 0: return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def confidence_tier(conf):
    if conf >= 70: return "🟢 High"
    elif conf >= 60: return "🟡 Medium"
    return "🔴 Low"

PT_LABELS = {0: "Big Server", 1: "Grinder", 2: "All-Courter", 3: "Upset Specialist"}

def build_row(p1, p2, surface, best_of, round_num, tourney,
              p1_fatigue, p2_fatigue, p1_rest, p2_rest,
              indoor, temp_val, wind_val,
              model_elo, match_history, surface_history,
              h2h_record, h2h_surface, h2h_recent,
              serve_history, ace_history, df_history, bp_history,
              upset_history, rank_history, tourney_history,
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

    stats = {
        "r1": r1, "r2": r2, "sr1": sr1, "sr2": sr2,
        "wr1": wr1, "wr2": wr2, "swr1": swr1, "swr2": swr2,
        "mom1": mom1, "mom2": mom2, "str1": str1, "str2": str2,
        "dom1": dom1, "dom2": dom2, "tb1": tb1, "tb2": tb2,
        "rwr1": rwr1, "rwr2": rwr2, "h2h": h2h, "sh2h": sh2h,
        "rh2h": rh2h, "sv1": sv1, "sv2": sv2, "ac1": ac1, "ac2": ac2,
        "up1": up1, "up2": up2, "pt1": pt1, "pt2": pt2,
    }
    return row, stats

def get_prediction(p1, p2, surface, best_of, round_num, tourney,
                   p1_fatigue, p2_fatigue, p1_rest, p2_rest,
                   p1_injured, p2_injured, temp_val, wind_val, indoor,
                   model_elo, clfs, feature_cols, scaler,
                   match_history, surface_history, h2h_record, h2h_surface,
                   h2h_recent, serve_history, ace_history, df_history,
                   bp_history, upset_history, rank_history, tourney_history,
                   round_history, dominance_history, tb_history,
                   player_types, all_surfaces):

    row, stats = build_row(
        p1, p2, surface, best_of, round_num, tourney,
        p1_fatigue, p2_fatigue, p1_rest, p2_rest,
        indoor, temp_val, wind_val,
        model_elo, match_history, surface_history,
        h2h_record, h2h_surface, h2h_recent,
        serve_history, ace_history, df_history, bp_history,
        upset_history, rank_history, tourney_history,
        round_history, dominance_history, tb_history,
        player_types, all_surfaces
    )

    input_df = pd.DataFrame([row])[feature_cols]
    input_sc = scaler.transform(input_df)

    probs = []
    try: probs.append(clfs["gb"].predict_proba(input_df)[0][1])
    except: pass
    try: probs.append(clfs["rf"].predict_proba(input_df)[0][1])
    except: pass
    try: probs.append(clfs["lr"].predict_proba(input_sc)[0][1])
    except: pass
    try: probs.append(clfs["mlp"].predict_proba(input_sc)[0][1])
    except: pass
    try:
        sm = clfs["surface"].get(surface)
        if sm:
            probs.append(sm.predict_proba(input_df)[0][1])
    except: pass

    prob = np.mean(probs) if probs else 0.5
    prob = apply_adjustments(prob, p1_injured, p2_injured,
                             stats["str1"], stats["str2"], temp_val, wind_val)
    return prob, stats

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
            "recent_h2h_p1":         compute_recent_h2h(p1, p2, h2h_recent),
            "serve_diff":            compute_serve_score(p1, serve_history) - compute_serve_score(p2, serve_history),
            "ace_diff":              compute_serve_score(p1, ace_history) - compute_serve_score(p2, ace_history),
            "df_diff":               compute_serve_score(p1, df_history) - compute_serve_score(p2, df_history),
            "bp_diff":               compute_serve_score(p1, bp_history) - compute_serve_score(p2, bp_history),
            "upset_diff":            compute_upset_rate(p1, upset_history) - compute_upset_rate(p2, upset_history),
            "rank_traj_diff":        compute_rank_trajectory(p1, rank_history) - compute_rank_trajectory(p2, rank_history),
            "tourney_win_diff":      compute_tournament_win_rate(p1, tourney, tourney_history) - compute_tournament_win_rate(p2, tourney, tourney_history),
            "streak_diff":           compute_streak(p1, match_history) - compute_streak(p2, match_history),
            "round_win_rate_diff":   compute_round_win_rate(p1, round_history, rnd) - compute_round_win_rate(p2, round_history, rnd),
            "dominance_diff":        compute_dominance(p1, dominance_history) - compute_dominance(p2, dominance_history),
            "tiebreak_diff":         compute_tiebreak_rate(p1, tb_history) - compute_tiebreak_rate(p2, tb_history),
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

        dom = parse_score(row["score"])
        dominance_history[p1].append(dom if p1_win else 1 - dom)
        dominance_history[p2].append(1 - dom if p1_win else dom)

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
    
    X = pd.get_dummies(features_df[feature_cols + ["surface"]], columns=["surface"], dtype=int)
    y = features_df["p1_win"]

    all_players = sorted(set(df_clean["player1"].tolist() + df_clean["player2"].tolist()))
    ps_data = []
    for p in all_players:
        ps_data.append({
            "player":     p,
            "ace_rate":   compute_serve_score(p, ace_history),
            "serve_pct":  compute_serve_score(p, serve_history),
            "upset_rate": compute_upset_rate(p, upset_history),
            "dominance":  compute_dominance(p, dominance_history),
        })
    ps_df   = pd.DataFrame(ps_data).set_index("player").fillna(0.5)
    sc_cl   = StandardScaler()
    ps_sc   = sc_cl.fit_transform(ps_df)
    kmeans  = KMeans(n_clusters=4, random_state=42, n_init=10)
    player_types = dict(zip(ps_df.index, kmeans.fit_predict(ps_sc)))

    for i in range(len(features_df)):
        p1 = df_clean.iloc[i]["player1"]
        p2 = df_clean.iloc[i]["player2"]
        features_df.at[i, "player_type_diff"] = player_types.get(p1, 0) - player_types.get(p2, 0)

    X = pd.get_dummies(features_df[feature_cols + ["surface"]], columns=["surface"], dtype=int)

    n              = len(features_df)
    sample_weights = np.linspace(0.3, 1.0, n)
    split          = int(n * 0.8)

    X_train    = X.iloc[:split]
    y_train    = y.iloc[:split]
    X_test     = X.iloc[split:]
    y_test     = y.iloc[split:]
    sw_train   = sample_weights[:split]

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, min_samples_leaf=5, random_state=42)
    gb.fit(X_train, y_train, sample_weight=sw_train)
    gb_cal = CalibratedClassifierCV(gb, method="sigmoid", cv=5)
    gb_cal.fit(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X_train, y_train, sample_weight=sw_train)
    rf_cal = CalibratedClassifierCV(rf, method="sigmoid", cv=5)
    rf_cal.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_sc, y_train, sample_weight=sw_train)

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", learning_rate_init=0.001, max_iter=300, random_state=42)
    mlp.fit(X_tr_sc, y_train)

    surface_models = {}
    for surf in ["Hard", "Clay", "Grass"]:
        mask = features_df["surface"].iloc[:split] == surf 
        if mask.sum() > 100:
            X_s  = X_train[mask]
            y_s  = y_train[mask]
            sw_s = sw_train[mask.values]
            clf_s = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
            clf_s.fit(X_s, y_s, sample_weight=sw_s)
            surface_models[surf] = clf_s

    clfs = {"gb": gb_cal, "rf": rf_cal, "lr": lr, "mlp": mlp, "surface": surface_models}

    gb_preds     = gb_cal.predict(X_test)
    backtest_acc = (gb_preds == y_test).mean()

    all_probs = []
    for i in range(len(X_test)):
        row_df = X_test.iloc[[i]]
        row_sc = X_te_sc[[i]]
        p_list = []
        try: p_list.append(gb_cal.predict_proba(row_df)[0][1])
        except: pass
        try: p_list.append(rf_cal.predict_proba(row_df)[0][1])
        except: pass
        try: p_list.append(lr.predict_proba(row_sc)[0][1])
        except: pass
        try: p_list.append(mlp.predict_proba(row_sc)[0][1])
        except: pass
        surf_val = features_df["surface"].iloc[split + i]
        if surf_val in surface_models:
            try: p_list.append(surface_models[surf_val].predict_proba(row_df)[0][1])
            except: pass
        all_probs.append(np.mean(p_list) if p_list else 0.5)

    test_df            = features_df.iloc[split:].copy()
    test_df["prob"]    = all_probs
    test_df["conf"]    = test_df["prob"].apply(lambda x: max(x, 1-x))
    test_df["correct"] = (test_df["prob"] > 0.5) == (test_df["p1_win"] == 1)
    ensemble_acc       = test_df["correct"].mean()
    test_df["tier"]    = test_df["conf"].apply(lambda x: "🟢 High (70%+)" if x >= 0.7 else ("🟡 Medium (60-70%)" if x >= 0.6 else "🔴 Low (<60%)"))

    all_surfaces = features_df["surface"].unique().tolist()
    all_tourneys = sorted(df_clean["tourney_name"].unique().tolist())

    return (model_elo, clfs, scaler, match_history, surface_history, h2h_record, h2h_surface, h2h_recent, serve_history, ace_history, df_history, bp_history, upset_history, rank_history, tourney_history, round_history, dominance_history, tb_history, player_types, all_players, all_surfaces, all_tourneys, feature_cols, backtest_acc, ensemble_acc, test_df, X_train)

# ==========================================
# 4. UI SETUP & CSS
# ==========================================
st.set_page_config(page_title="Tennis Match Predictor", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    /* Global metric styling */
    .stMetric {background-color:#1e2130; padding:10px; border-radius:8px;}
    
    /* Top score strip buttons */
    .stButton > button {
        width: 100%; height: 100%; min-height: 100px;
        background-color: #1e2130; border: 1px solid #333; border-left: 5px solid #4CAF50;
        text-align: left; justify-content: flex-start; padding: 10px;
    }
    
    /* Hide default sidebar arrow and add hamburger */
    [data-testid="collapsedControl"] svg { display: none !important; }
    [data-testid="collapsedControl"]::before {
        content: "☰"; font-size: 26px; color: #FFFFFF; display: block; margin-left: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if "selected_match" not in st.session_state: st.session_state.selected_match = None
if "track_record" not in st.session_state: st.session_state.track_record = []
if "manual_matches" not in st.session_state: st.session_state.manual_matches = []

# Top Global Score Strip (Mocked for layout/visuals)
st.markdown("### 🕒 Live & Upcoming Matches")
score_cols = st.columns(4)
with score_cols[0]:
    if st.button("🎾 LIVE - Set 3\n\nJ. Sinner (6) (4) (3)\nA. Zverev (4) (6) (2)", key="top_m1"):
        st.session_state.selected_match = "Sinner vs Zverev"
with score_cols[1]:
    if st.button("🎾 LIVE - Set 1\n\nC. Alcaraz (5)\nJ. Draper (4)", key="top_m2"):
        st.session_state.selected_match = "Alcaraz vs Draper"
with score_cols[2]:
    if st.button("🕒 Today - 17:30\n\nD. Medvedev\nH. Rune", key="top_m3"):
        st.session_state.selected_match = "Medvedev vs Rune"
with score_cols[3]:
    if st.button("🕒 Tomorrow - 13:00\n\nT. Fritz\nA. de Minaur", key="top_m4"):
        st.session_state.selected_match = "Fritz vs de Minaur"
st.divider()

with st.spinner("Training model... ~2 minutes on first load"):
    (model_elo, clfs, scaler, match_history, surface_history, h2h_record, h2h_surface, h2h_recent, serve_history, ace_history, df_history, bp_history, upset_history, rank_history, tourney_history, round_history, dominance_history, tb_history, player_types, all_players, all_surfaces, all_tourneys, feature_cols, backtest_acc, ensemble_acc, test_df, X_train) = train_model()

# ==========================================
# 5. SIDEBAR TOGGLE
# ==========================================
st.sidebar.title("Settings")
show_extra_features = st.sidebar.checkbox("Show Advanced Tools (Calendar, News, Odds, etc.)", value=False)


# ==========================================
# 6. PREDICTOR (ALWAYS VISIBLE MAIN PAGE)
# ==========================================
st.title("🎾 Tennis Match Predictor")
st.caption("Ensemble ML · Neural Network · Surface Models · Calibrated Probabilities")

col1, col2, col3 = st.columns([2, 2, 2])
with col1: p1 = st.selectbox("Player 1", all_players, key="p1")
with col2: p2 = st.selectbox("Player 2", all_players, index=1, key="p2")
with col3: tourney_hint = st.text_input("Tournament (Optional)", placeholder="e.g. Wimbledon")

# --- THE MAGIC GEMINI BUTTON ---
if st.button("✨ Auto-Fill Conditions with AI", type="secondary"):
    if p1 and p2 and p1 != p2:
        with st.spinner("Gemini is analyzing the matchup and estimating conditions..."):
            try:
                context = get_match_context_from_gemini(p1, p2, tourney_hint)
                if context:
                    st.session_state.surface = context.get("surface", "Hard")
                    st.session_state.indoor = context.get("indoor", "Outdoor")
                    st.session_state.best_of = context.get("best_of", 3)
                    st.session_state.round_num = context.get("round_num", 5)
                    st.session_state.temp = context.get("temp_celsius", 20)
                    st.session_state.wind = context.get("wind_kmh", 10)
                    st.session_state.p1_fat = context.get("p1_fatigue", 0)
                    st.session_state.p1_rest = context.get("p1_rest_days", 3)
                    st.session_state.p1_inj = context.get("p1_injured", False)
                    st.session_state.p2_fat = context.get("p2_fatigue", 0)
                    st.session_state.p2_rest = context.get("p2_rest_days", 3)
                    st.session_state.p2_inj = context.get("p2_injured", False)
                    st.success("✅ Conditions auto-filled successfully!")
                else:
                    st.error("Could not parse AI response. Check your API key or enter manually.")
            except Exception as e:
                st.error(f"Failed to fetch AI context: {e}")

st.divider()

# --- MANUAL OVERRIDES EXPANDER ---
with st.expander("⚙️ View or Edit Match Settings (Auto-filled by AI)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], index=["Hard", "Clay", "Grass"].index(st.session_state.get("surface", "Hard")))
        best_of = st.radio("Best of", [3, 5], index=0 if st.session_state.get("best_of", 3) == 3 else 1, horizontal=True)
    with c2:
        round_num = st.select_slider("Round", options=[1,2,3,4,5,6,7], value=st.session_state.get("round_num", 5), format_func=lambda x: ["R128","R64","R32","R16","QF","SF","F"][x-1])
        tourney = st.selectbox("Tournament", ["Unknown"] + all_tourneys)
    with c3:
        indoor = st.selectbox("Venue", ["Outdoor", "Indoor"], index=0 if st.session_state.get("indoor", "Outdoor") == "Outdoor" else 1)
        temp_val = st.number_input("Temp (°C)", -10, 50, st.session_state.get("temp", 20))
        wind_val = st.number_input("Wind (km/h)", 0, 100, st.session_state.get("wind", 10))

    st.markdown("**Fatigue & Injury**")
    f1, f2 = st.columns(2)
    with f1:
        p1_fat = st.number_input(f"{p1} matches last 14 days", 0, 15, st.session_state.get("p1_fat", 0))
        p1_rest = st.number_input(f"{p1} days rest", 0, 365, st.session_state.get("p1_rest", 3))
        p1_injured = st.checkbox(f"🚑 {p1} injured", value=st.session_state.get("p1_inj", False))
    with f2:
        p2_fat = st.number_input(f"{p2} matches last 14 days", 0, 15, st.session_state.get("p2_fat", 0))
        p2_rest = st.number_input(f"{p2} days rest", 0, 365, st.session_state.get("p2_rest", 3))
        p2_injured = st.checkbox(f"🚑 {p2} injured", value=st.session_state.get("p2_inj", False))


if st.button("🚀 Predict Matchup", type="primary"):
    if p1 == p2:
        st.error("Select two different players.")
    else:
        prob, stats = get_prediction(
            p1, p2, surface, best_of, round_num, tourney,
            p1_fat, p2_fat, p1_rest, p2_rest, p1_injured, p2_injured, temp_val, wind_val, indoor,
            model_elo, clfs, feature_cols, scaler, match_history, surface_history, h2h_record, h2h_surface,
            h2h_recent, serve_history, ace_history, df_history, bp_history, upset_history, rank_history, tourney_history,
            round_history, dominance_history, tb_history, player_types, all_surfaces
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

        st.info(f"Player types — {p1}: **{PT_LABELS.get(stats['pt1'], 'Unknown')}** | {p2}: **{PT_LABELS.get(stats['pt2'], 'Unknown')}**")

        with st.expander("Detailed breakdown"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{p1}**")
                st.write(f"Overall Elo: {stats['r1']:.0f} | {surface} Elo: {stats['sr1']:.0f}")
                st.write(f"Win rate: {stats['wr1']*100:.1f}% | {surface} win rate: {stats['swr1']*100:.1f}%")
                st.write(f"Momentum: {stats['mom1']*100:.1f}% | Serve %: {stats['sv1']*100:.1f}%")
            with col2:
                st.write(f"**{p2}**")
                st.write(f"Overall Elo: {stats['r2']:.0f} | {surface} Elo: {stats['sr2']:.0f}")
                st.write(f"Win rate: {stats['wr2']*100:.1f}% | {surface} win rate: {stats['swr2']*100:.1f}%")
                st.write(f"Momentum: {stats['mom2']*100:.1f}% | Serve %: {stats['sv2']*100:.1f}%")
            st.write(f"**Overall H2H for {p1}**: {stats['h2h']*100:.1f}%")

# ==========================================
# 7. ADVANCED TOOLS (HIDDEN BY DEFAULT)
# ==========================================
if show_extra_features or st.session_state.selected_match:
    st.divider()
    
    tab_calendar, tab_news, tab_odds, tab_players, tab_backtest, tab_history = st.tabs([
        "📅 Matches & Calendar", "📰 Tennis News", "📊 Find Value Bets", "👤 Player Profiles", "📈 Backtest Report", "📋 Session History Tracker"
    ])

    # ======== TAB: CALENDAR & MATCHES ========
    with tab_calendar:
        if st.session_state.selected_match:
            match_title = st.session_state.selected_match
            if st.button("← Back to Calendar"):
                st.session_state.selected_match = None
                st.rerun()
                
            st.title(f"Detailed Breakdown: {match_title}")
            
            st.markdown("### Comparative Statistics")
            html = """
            <table style='width:100%; text-align:center; border-collapse: collapse;'>
                <tr style='border-bottom: 2px solid #333;'><th style='text-align:left; padding:10px;'>Statistic</th><th>Player 1</th><th>Player 2</th></tr>
                <tr style='border-bottom: 1px solid #222;'><td style='text-align:left; padding:10px;'>Recent Win Form %</td><td style='background-color: rgba(76, 175, 80, 0.25); font-weight: bold;'>82.5%</td><td>76.0%</td></tr>
                <tr style='border-bottom: 1px solid #222;'><td style='text-align:left; padding:10px;'>1st Serve Win %</td><td>75.4%</td><td style='background-color: rgba(76, 175, 80, 0.25); font-weight: bold;'>78.2%</td></tr>
            </table><br/>
            """
            st.markdown(html, unsafe_allow_html=True)
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("### Win Probability Timeline")
                time_steps = [f"Set 1 Gm {i}" for i in range(1, 11)]
                df_time = pd.DataFrame({"Match Phase": time_steps, "Player 1 Prob": [50, 55, 52, 60, 65, 62, 70, 75, 72, 80]}).set_index("Match Phase")
                st.line_chart(df_time)
            with c2:
                st.markdown("### Match Projection")
                try:
                    fig = px.pie(values=[80, 20], names=['Player 1', 'Player 2'], color_discrete_sequence=['#4CAF50', '#2a2d3d'], hole=0.5)
                    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                except NameError:
                    st.warning("Plotly not installed. Add 'plotly' to your requirements.txt")

        else:
            st.subheader("📅 Interactive Tournament Calendar")
            st.caption("June 2026")
            days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            cols = st.columns(7)
            for i, day in enumerate(days_of_week):
                cols[i].markdown(f"**{day}**")
            
            day_counter = 1
            for week in range(5):
                grid_cols = st.columns(7)
                for i in range(7):
                    if day_counter <= 30:
                        with grid_cols[i]:
                            st.markdown(f"<div style='border: 1px solid #444; border-radius: 5px; padding: 5px; min-height: 100px;'><b>{day_counter}</b>", unsafe_allow_html=True)
                            if day_counter == 8: 
                                if st.button("Sinner vs Zverev", key="cal_m1"):
                                    st.session_state.selected_match = "Sinner vs Zverev"
                                    st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
                        day_counter += 1

    # ======== TAB: NEWS ========
    with tab_news:
        st.subheader("📰 Tennis News & Social Feed")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image("https://images.unsplash.com/photo-1595435934249-5df7ed86e1c0?q=80&w=1000&auto=format&fit=crop", use_container_width=True)
            st.markdown("### [Alcaraz claims surface speed transitions favor aggressive playstyles](https://www.atptour.com)")
            st.write("An early assessment detailing layout configuration training routines ahead of major schedule turn points as the ATP tour converges on Queen's Club.")
            st.caption("ESPN Tennis • 1 Hour Ago")

        with col2:
            st.subheader("Trending Articles")
            st.markdown("**[Sinner looks to stabilize return metrics after clinical run](https://www.bleacherreport.com)**")
            st.caption("Bleacher Report • 2 Hours Ago")
            st.divider()
            st.markdown("**[Djokovic confirms recovery periods ahead of Wimbledon](https://www.tennis.com)**")
            st.caption("Tennis.com • 4 Hours Ago")
            
        st.divider()
        st.subheader("What's Buzzing on X (Twitter)")
        x_col1, x_col2, x_col3 = st.columns(3)
        with x_col1:
            components.html("""
            <blockquote class="twitter-tweet"><p lang="en" dir="ltr">The grass court season is officially here! 🌱🎾 Let the slides and slices begin. <a href="https://twitter.com/atptour/status/1798361730032644268"></a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
            """, height=400)

    # ======== TAB: VALUE BETS ========
    with tab_odds:
        st.subheader("📊 Find Value Bets")
        col1, col2 = st.columns(2)
        with col1: op1 = st.selectbox("Player 1", all_players, key="op1_bets")
        with col2: op2 = st.selectbox("Player 2", all_players, index=1, key="op2_bets")

        osurf  = st.selectbox("Surface", ["Hard","Clay","Grass"], key="os")
        obo    = st.radio("Best of", [3, 5], horizontal=True, key="obo")
        oround = st.select_slider("Round", options=[1,2,3,4,5,6,7], format_func=lambda x: ["R128","R64","R32","R16","QF","SF","F"][x-1], key="ornd")

        col1, col2 = st.columns(2)
        with col1: odds1 = st.number_input(f"{op1} odds", value=-150, key="o1")
        with col2: odds2 = st.number_input(f"{op2} odds", value=120,  key="o2")

        if st.button("Find Value", type="primary"):
            if op1 == op2:
                st.error("Select two different players.")
            else:
                prob, _ = get_prediction(op1, op2, osurf, obo, oround, "Unknown", 0, 0, 3, 3, False, False, 20, 10, "Outdoor", model_elo, clfs, feature_cols, scaler, match_history, surface_history, h2h_record, h2h_surface, h2h_recent, serve_history, ace_history, df_history, bp_history, upset_history, rank_history, tourney_history, round_history, dominance_history, tb_history, player_types, all_surfaces)
                imp1, imp2  = implied_prob(odds1), implied_prob(odds2)
                diff1, diff2 = prob - imp1, (1-prob) - imp2

                col1, col2 = st.columns(2)
                with col1: st.metric(op1, f"Model: {prob*100:.1f}%", delta=f"Vegas: {imp1*100:.1f}%")
                with col2: st.metric(op2, f"Model: {(1-prob)*100:.1f}%", delta=f"Vegas: {imp2*100:.1f}%")

                if diff1 > 0.05: st.success(f"✅ **{op1}** undervalued — edge +{diff1*100:.1f}%")
                elif diff1 < -0.05: st.warning(f"⚠️ **{op1}** overvalued by {abs(diff1)*100:.1f}%")
                
                if diff2 > 0.05: st.success(f"✅ **{op2}** undervalued — edge +{diff2*100:.1f}%")
                elif diff2 < -0.05: st.warning(f"⚠️ **{op2}** overvalued by {abs(diff2)*100:.1f}%")

    # ======== TAB: PLAYER PROFILE ========
    with tab_players:
        st.subheader("👤 Player Profile")
        search = st.selectbox("Search player", all_players, key="search")
        if search:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall Elo", f"{model_elo.get_rating(search):.0f}")
            col2.metric("Hard Elo",    f"{model_elo.get_rating(search, 'Hard'):.0f}")
            col3.metric("Clay Elo",    f"{model_elo.get_rating(search, 'Clay'):.0f}")
            col4.metric("Grass Elo",   f"{model_elo.get_rating(search, 'Grass'):.0f}")

            wr  = compute_recent_win_rate(search, match_history)
            st_ = compute_streak(search, match_history)
            st.info(f"Player type: **{PT_LABELS.get(player_types.get(search, 0), 'Unknown')}** | Current Streak: {st_:+d} | Recent Win Rate: {wr*100:.1f}%")

    # ======== TAB: BACKTEST ========
    with tab_backtest:
        st.subheader("📈 Backtest Report")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Single Model", f"{backtest_acc*100:.1f}%")
        col2.metric("Ensemble", f"{ensemble_acc*100:.1f}%")
        col3.metric("Baseline", "~65.0%")
        col4.metric("Edge over baseline", f"+{(ensemble_acc-0.65)*100:.1f}%")

    # ======== TAB: HISTORY ========
    with tab_history:
        st.subheader("📋 Session History Tracker")
        with st.form("log_pred"):
            c1, c2, c3 = st.columns(3)
            with c1:
                t_date = st.date_input("Date", key="td")
                t_p1   = st.selectbox("Player 1", all_players, key="tp1_hist")
                t_p2   = st.selectbox("Player 2", all_players, index=1, key="tp2_hist")
            with c2:
                t_surf    = st.selectbox("Surface", ["Hard","Clay","Grass"], key="tsurf")
                t_pick    = st.selectbox("Your pick", ["Player 1","Player 2"], key="tpick")
            with c3:
                t_conf   = st.number_input("Confidence %", 50.0, 100.0, 65.0, key="tc")
                t_result = st.selectbox("Result", ["Pending","Correct","Wrong"], key="tr")

            if st.form_submit_button("Log") and t_p1 != t_p2:
                st.session_state.track_record.append({
                    "Date": str(t_date), "Player 1": t_p1, "Player 2": t_p2, "Surface": t_surf,
                    "Pick": t_p1 if t_pick == "Player 1" else t_p2, "Confidence": f"{t_conf:.1f}%", "Result": t_result,
                })

        if st.session_state.track_record:
            st.dataframe(pd.DataFrame(st.session_state.track_record), use_container_width=True)
            if st.button("Clear all logs"):
                st.session_state.track_record = []
                st.rerun()
