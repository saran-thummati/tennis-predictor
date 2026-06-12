import streamlit as st
import pandas as pd
import ssl
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict # Required for EloModel
import requests
import plotly.express as px
import streamlit.components.v1 as components
import google.generativeai as genai
import json
import joblib

# ==========================================
# 1. ELO MODEL CLASS (Must exist to load the pickle)
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

# ==========================================
# 2. FAST MODEL LOADING
# ==========================================
@st.cache_resource
def load_saved_models():
    """Loads the pre-trained models and data instantly."""
    return joblib.load("tennis_model_artifacts.pkl")

# Load everything at the start of the script
try:
    artifacts = load_saved_models()
    model_elo = artifacts["model_elo"]
    clfs = artifacts["clfs"]
    scaler = artifacts["scaler"]
    match_history = artifacts["match_history"]
    surface_history = artifacts["surface_history"]
    h2h_record = artifacts["h2h_record"]
    h2h_surface = artifacts["h2h_surface"]
    h2h_recent = artifacts["h2h_recent"]
    serve_history = artifacts["serve_history"]
    ace_history = artifacts["ace_history"]
    df_history = artifacts["df_history"]
    bp_history = artifacts["bp_history"]
    upset_history = artifacts["upset_history"]
    rank_history = artifacts["rank_history"]
    tourney_history = artifacts["tourney_history"]
    round_history = artifacts["round_history"]
    dominance_history = artifacts["dominance_history"]
    tb_history = artifacts["tb_history"]
    player_types = artifacts["player_types"]
    all_players = artifacts["all_players"]
    all_surfaces = artifacts["all_surfaces"]
    all_tourneys = artifacts["all_tourneys"]
    feature_cols = artifacts["feature_cols"]
    backtest_acc = artifacts["backtest_acc"]
    ensemble_acc = artifacts["ensemble_acc"]
except FileNotFoundError:
    st.error("Model artifacts not found! Please run 'python train.py' in your terminal first.")
    st.stop()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def compute_recent_win_rate(player, match_history, n=20):
    matches = match_history.get(player, [])[-n:]
    return sum(matches) / len(matches) if matches else 0.5

def compute_surface_win_rate(player, surface, surface_history, n=20):
    matches = surface_history.get(player, {}).get(surface, [])[-n:]
    return sum(matches) / len(matches) if matches else 0.5

def compute_momentum(player, match_history, n=10):
    matches = match_history.get(player, [])[-n:]
    if not matches: return 0.5
    weights = [i + 1 for i in range(len(matches))]
    return sum(w * m for w, m in zip(weights, matches)) / sum(weights)

def compute_streak(player, match_history):
    matches = match_history.get(player, [])
    if not matches: return 0
    streak = 0
    last   = matches[-1]
    for result in reversed(matches):
        if result == last: streak += 1
        else: break
    return streak if last == 1 else -streak

def compute_dominance(player, dominance_history, n=10):
    recent = dominance_history.get(player, [])[-n:]
    return sum(recent) / len(recent) if recent else 0.5

def compute_tiebreak_rate(player, tb_history, n=20):
    recent = tb_history.get(player, [])[-n:]
    return sum(recent) / len(recent) if recent else 0.5

def compute_round_win_rate(player, round_history, round_num):
    matches = round_history.get(player, {}).get(round_num, [])
    return sum(matches) / len(matches) if matches else 0.5

def compute_h2h(p1, p2, h2h_record):
    key = tuple(sorted([p1, p2]))
    record = h2h_record.get(key, {"wins_a": 0, "wins_b": 0})
    total = record["wins_a"] + record["wins_b"]
    if total == 0: return 0.5
    if p1 == key[0]: return record["wins_a"] / total
    return record["wins_b"] / total

def compute_surface_h2h(p1, p2, surface, h2h_surface):
    key = tuple(sorted([p1, p2])) + (surface,)
    record = h2h_surface.get(key, {"wins_a": 0, "wins_b": 0})
    total = record["wins_a"] + record["wins_b"]
    if total == 0: return 0.5
    if p1 == tuple(sorted([p1, p2]))[0]: return record["wins_a"] / total
    return record["wins_b"] / total

def compute_recent_h2h(p1, p2, h2h_recent, n=5):
    key = tuple(sorted([p1, p2]))
    matches = h2h_recent.get(key, [])[-n:]
    if not matches: return 0.5
    return sum(1 for m in matches if m["winner"] == p1) / len(matches)

def compute_serve_score(player, serve_history, n=10):
    recent = serve_history.get(player, [])[-n:]
    return sum(recent) / len(recent) if recent else 0.5

def compute_upset_rate(player, upset_history):
    record = upset_history.get(player, {"wins": 0, "total": 0})
    total, wins = record["total"], record["wins"]
    return wins / total if total > 0 else 0.5

def compute_rank_trajectory(player, rank_history, n=10):
    ranks = rank_history.get(player, [])[-n:]
    if len(ranks) < 2: return 0
    return ranks[0] - ranks[-1]

def compute_tournament_win_rate(player, tourney, tourney_history):
    matches = tourney_history.get(player, {}).get(tourney, [])
    return sum(matches) / len(matches) if matches else 0.5

def apply_adjustments(prob, p1_injured, p2_injured, str1, str2, temp_val, wind_val):
    if p1_injured and not p2_injured: prob *= 0.75
    elif p2_injured and not p1_injured: prob = prob + (1 - prob) * 0.25
    prob = max(0.05, min(0.95, prob + (str1 - str2) * 0.005))
    if temp_val >= 30: prob = prob * 0.97 + 0.03 * 0.5
    if wind_val >= 25: prob = prob * 0.95 + 0.05 * 0.5
    return prob

def implied_prob(odds):
    if odds > 0: return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def confidence_tier(conf):
    if conf >= 70: return "🟢 High"
    elif conf >= 60: return "🟡 Medium"
    return "🔴 Low"

PT_LABELS = {0: "Big Server", 1: "Grinder", 2: "All-Courter", 3: "Upset Specialist"}

# ==========================================
# 4. PREDICTION ENGINE
# ==========================================
def build_row(p1, p2, surface, best_of, round_num, tourney,
              p1_fatigue, p2_fatigue, p1_rest, p2_rest, indoor, temp_val, wind_val):

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
        "elo_diff": r1 - r2, "surface_elo_diff": sr1 - sr2, "rank_diff": 0, "age_diff": 0,
        "win_rate_diff": wr1 - wr2, "surface_win_rate_diff": swr1 - swr2, "momentum_diff": mom1 - mom2,
        "fatigue_diff": p1_fatigue - p2_fatigue, "rest_diff": p1_rest - p2_rest,
        "h2h_p1": h2h, "surface_h2h_p1": sh2h, "recent_h2h_p1": rh2h,
        "serve_diff": sv1 - sv2, "ace_diff": ac1 - ac2, "df_diff": df1 - df2, "bp_diff": bp1 - bp2,
        "upset_diff": up1 - up2, "rank_traj_diff": rt1 - rt2, "tourney_win_diff": tw1 - tw2,
        "streak_diff": str1 - str2, "round_win_rate_diff": rwr1 - rwr2, "dominance_diff": dom1 - dom2,
        "tiebreak_diff": tb1 - tb2, "player_type_diff": pt1 - pt2,
        "round": round_num, "best_of": best_of, "indoor": 1 if indoor == "Indoor" else 0,
        "temp": temp_val, "wind": wind_val,
    }
    for s in all_surfaces:
        row[f"surface_{s}"] = 1 if surface == s else 0

    stats = {
        "r1": r1, "r2": r2, "sr1": sr1, "sr2": sr2, "wr1": wr1, "wr2": wr2, "swr1": swr1, "swr2": swr2,
        "mom1": mom1, "mom2": mom2, "str1": str1, "str2": str2, "dom1": dom1, "dom2": dom2, "tb1": tb1, "tb2": tb2,
        "rwr1": rwr1, "rwr2": rwr2, "h2h": h2h, "sh2h": sh2h, "rh2h": rh2h, "sv1": sv1, "sv2": sv2, "ac1": ac1, "ac2": ac2,
        "up1": up1, "up2": up2, "pt1": pt1, "pt2": pt2,
    }
    return row, stats

def get_prediction(p1, p2, surface, best_of, round_num, tourney, p1_fat, p2_fat, p1_rest, p2_rest, p1_inj, p2_inj, temp, wind, indoor):
    row, stats = build_row(p1, p2, surface, best_of, round_num, tourney, p1_fat, p2_fat, p1_rest, p2_rest, indoor, temp, wind)
    input_df = pd.DataFrame([row])[feature_cols]

    probs = []
    # Stacking Model handles LGBM and RF automatically
    try: probs.append(clfs["stacked"].predict_proba(input_df)[0][1])
    except: pass
    
    # Surface specific
    try:
        sm = clfs["surface"].get(surface)
        if sm: probs.append(sm.predict_proba(input_df)[0][1])
    except: pass

    prob = np.mean(probs) if probs else 0.5
    prob = apply_adjustments(prob, p1_inj, p2_inj, stats["str1"], stats["str2"], temp, wind)
    return prob, stats

# ==========================================
# 5. GEMINI AI AUTO-FILL
# ==========================================
def get_match_context_from_gemini(p1, p2, tourney_hint="Current ATP event"):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are a professional tennis analyst. I am simulating a match between {p1} and {p2} at {tourney_hint}.
        Based on your knowledge of tennis, provide realistic values. Return ONLY a valid JSON object:
        {{
            "surface": "Hard", "indoor": "Outdoor", "best_of": 3, "round_num": 5, "temp_celsius": 24, "wind_kmh": 12,
            "p1_fatigue": 2, "p1_rest_days": 3, "p1_injured": false, "p2_fatigue": 1, "p2_rest_days": 4, "p2_injured": false
        }}
        """
        response = model.generate_content(prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json"))
        return json.loads(response.text)
    except Exception: return None

# ==========================================
# 6. UI SETUP
# ==========================================
st.set_page_config(page_title="Tennis Match Predictor", page_icon="🎾", layout="wide")
st.markdown("""
    <style>
    .stMetric {background-color:#1e2130; padding:10px; border-radius:8px;}
    [data-testid="collapsedControl"] svg { display: none !important; }
    [data-testid="collapsedControl"]::before { content: "☰"; font-size: 26px; color: #FFFFFF; display: block; margin-left: 5px;}
    </style>
""", unsafe_allow_html=True)

if "selected_match" not in st.session_state: st.session_state.selected_match = None
if "track_record" not in st.session_state: st.session_state.track_record = []

st.sidebar.title("Settings")
show_extra_features = st.sidebar.checkbox("Show Advanced Tools (Calendar, News, Odds)", value=False)

st.title("🎾 Tennis Match Predictor")
st.caption("LightGBM Stacking Meta-Learner · Margin-of-Victory Elo · Calibrated Probabilities")

col1, col2, col3 = st.columns([2, 2, 2])
with col1: p1 = st.selectbox("Player 1", all_players, key="p1")
with col2: p2 = st.selectbox("Player 2", all_players, index=1, key="p2")
with col3: tourney_hint = st.text_input("Tournament (Optional)", placeholder="e.g. Wimbledon")

if st.button("✨ Auto-Fill Conditions with AI", type="secondary"):
    if p1 and p2 and p1 != p2:
        with st.spinner("Gemini is analyzing the matchup..."):
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
                st.success("✅ Auto-filled successfully!")
            else: st.error("Failed to parse AI context.")

st.divider()

with st.expander("⚙️ View or Edit Match Settings"):
    c1, c2, c3 = st.columns(3)
    with c1:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], index=["Hard", "Clay", "Grass"].index(st.session_state.get("surface", "Hard")))
        best_of = st.radio("Best of", [3, 5], index=0 if st.session_state.get("best_of", 3) == 3 else 1, horizontal=True)
    with c2:
        round_num = st.select_slider("Round", options=[1,2,3,4,5,6,7], value=st.session_state.get("round_num", 5))
        tourney = st.selectbox("Tournament", ["Unknown"] + all_tourneys)
    with c3:
        indoor = st.selectbox("Venue", ["Outdoor", "Indoor"], index=0 if st.session_state.get("indoor", "Outdoor") == "Outdoor" else 1)
        temp_val = st.number_input("Temp (°C)", -10, 50, st.session_state.get("temp", 20))
        wind_val = st.number_input("Wind (km/h)", 0, 100, st.session_state.get("wind", 10))

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
    if p1 == p2: st.error("Select two different players.")
    else:
        prob, stats = get_prediction(p1, p2, surface, best_of, round_num, tourney, p1_fat, p2_fat, p1_rest, p2_rest, p1_injured, p2_injured, temp_val, wind_val, indoor)
        winner = p1 if prob > 0.5 else p2
        conf   = max(prob, 1-prob) * 100
        
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1: st.metric(p1, f"{prob*100:.1f}%")
        with col2: st.metric("Confidence", confidence_tier(conf))
        with col3: st.metric(p2, f"{(1-prob)*100:.1f}%")
        st.success(f"🏆 Predicted winner: **{winner}** ({conf:.1f}% confidence)")

if show_extra_features:
    st.divider()
    tab_calendar, tab_news, tab_odds, tab_players, tab_backtest, tab_history = st.tabs([
        "📅 Matches", "📰 News", "📊 Value Bets", "👤 Players", "📈 Backtest", "📋 History"
    ])
    
    with tab_calendar: st.info("Calendar functionality here")
    with tab_news: st.info("News feed here")
    with tab_odds: st.info("Odds functionality here")
    with tab_players: st.info("Player search here")
    
    with tab_backtest:
        st.subheader("📈 Backtest Report (Stacked Model)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ensemble Accuracy", f"{ensemble_acc*100:.1f}%")
        c2.metric("Baseline", "~65.0%")
        c3.metric("Edge", f"+{(ensemble_acc-0.65)*100:.1f}%")
        
    with tab_history: st.info("History tracker here")
