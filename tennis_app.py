import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
import requests
import calendar

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
    today     = datetime.today().strftime("%Y-%m-%d")
    end_date  = (datetime.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    url = f"https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey={api_key}&from={today}&to={end_date}&tour_id=2"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "result" in data and data["result"]:
            return data["result"]
        return []
    except:
        return []

# ---- EloModel ----
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

def compute_h2h(p1, p2, h2h_record):
    key = tuple(sorted([p1, p2]))
    record = h2h_record[key]
    total = record["wins_a"] + record["wins_b"]
    if total == 0:
        return 0.5
    if p1 == key[0]:
        return record["wins_a"] / total
    return record["wins_b"] / total

def compute_surface_h2h(p1, p2, surface, h2h_surface):
    key = tuple(sorted([p1, p2])) + (surface,)
    record = h2h_surface[key]
    total = record["wins_a"] + record["wins_b"]
    if total == 0:
        return 0.5
    if p1 == tuple(sorted([p1, p2]))[0]:
        return record["wins_a"] / total
    return record["wins_b"] / total

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

def round_to_num(round_str):
    mapping = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7, "RR": 3}
    return mapping.get(str(round_str), 3)

def implied_prob(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def get_prediction(p1, p2, surface, best_of, round_num, tourney,
                   p1_fatigue, p2_fatigue, p1_rest, p2_rest,
                   model, clf, feature_cols, match_history, surface_history,
                   h2h_record, h2h_surface, serve_history, ace_history,
                   df_history, bp_history, upset_history, rank_history,
                   tourney_history, all_surfaces):

    r1, r2   = model.get_rating(p1), model.get_rating(p2)
    sr1, sr2 = model.get_rating(p1, surface), model.get_rating(p2, surface)
    wr1   = compute_recent_win_rate(p1, match_history)
    wr2   = compute_recent_win_rate(p2, match_history)
    swr1  = compute_surface_win_rate(p1, surface, surface_history)
    swr2  = compute_surface_win_rate(p2, surface, surface_history)
    mom1  = compute_momentum(p1, match_history)
    mom2  = compute_momentum(p2, match_history)
    h2h   = compute_h2h(p1, p2, h2h_record)
    sh2h  = compute_surface_h2h(p1, p2, surface, h2h_surface)
    sv1   = compute_serve_score(p1, serve_history)
    sv2   = compute_serve_score(p2, serve_history)
    ac1   = compute_serve_score(p1, ace_history)
    ac2   = compute_serve_score(p2, ace_history)
    df1   = compute_serve_score(p1, df_history)
    df2   = compute_serve_score(p2, df_history)
    bp1   = compute_serve_score(p1, bp_history)
    bp2   = compute_serve_score(p2, bp_history)
    up1   = compute_upset_rate(p1, upset_history)
    up2   = compute_upset_rate(p2, upset_history)
    rt1   = compute_rank_trajectory(p1, rank_history)
    rt2   = compute_rank_trajectory(p2, rank_history)
    tw1   = compute_tournament_win_rate(p1, tourney, tourney_history)
    tw2   = compute_tournament_win_rate(p2, tourney, tourney_history)

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
        "serve_diff":            sv1 - sv2,
        "ace_diff":              ac1 - ac2,
        "df_diff":               df1 - df2,
        "bp_diff":               bp1 - bp2,
        "upset_diff":            up1 - up2,
        "rank_traj_diff":        rt1 - rt2,
        "tourney_win_diff":      tw1 - tw2,
        "round":                 round_num,
        "best_of":               best_of,
    }
    for s in all_surfaces:
        row[f"surface_{s}"] = 1 if surface == s else 0

    input_df = pd.DataFrame([row])[feature_cols]
    prob     = clf.predict_proba(input_df)[0][1]
    return prob, r1, r2, sr1, sr2, wr1, wr2, h2h, sh2h, mom1, mom2, sv1, sv2

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
        "p1_win":        np.where(flip, 0, 1),
    })
    df_clean["rank1"]  = np.where(flip, df["loser_rank"],  df["winner_rank"])
    df_clean["rank2"]  = np.where(flip, df["winner_rank"], df["loser_rank"])
    df_clean["age1"]   = np.where(flip, df["loser_age"],   df["winner_age"])
    df_clean["age2"]   = np.where(flip, df["winner_age"],  df["loser_age"])
    df_clean["serve1"] = np.where(flip, df["l_1stWon"] / (df["l_1stIn"] + 1),
                                        df["w_1stWon"] / (df["w_1stIn"] + 1))
    df_clean["serve2"] = np.where(flip, df["w_1stWon"] / (df["w_1stIn"] + 1),
                                        df["l_1stWon"] / (df["l_1stIn"] + 1))
    df_clean["ace1"]   = np.where(flip, df["l_ace"] / (df["l_svpt"] + 1),
                                        df["w_ace"] / (df["w_svpt"] + 1))
    df_clean["ace2"]   = np.where(flip, df["w_ace"] / (df["w_svpt"] + 1),
                                        df["l_ace"] / (df["l_svpt"] + 1))
    df_clean["df1"]    = np.where(flip, df["l_df"] / (df["l_svpt"] + 1),
                                        df["w_df"] / (df["w_svpt"] + 1))
    df_clean["df2"]    = np.where(flip, df["w_df"] / (df["w_svpt"] + 1),
                                        df["l_df"] / (df["l_svpt"] + 1))
    df_clean["bp1"]    = np.where(flip, df["l_bpSaved"] / (df["l_bpFaced"] + 1),
                                        df["w_bpSaved"] / (df["w_bpFaced"] + 1))
    df_clean["bp2"]    = np.where(flip, df["w_bpSaved"] / (df["w_bpFaced"] + 1),
                                        df["l_bpSaved"] / (df["l_bpFaced"] + 1))

    records         = []
    model           = EloModel()
    match_history   = defaultdict(list)
    surface_history = defaultdict(lambda: defaultdict(list))
    match_dates     = defaultdict(list)
    h2h_record      = defaultdict(lambda: {"wins_a": 0, "wins_b": 0})
    h2h_surface     = defaultdict(lambda: {"wins_a": 0, "wins_b": 0})
    serve_history   = defaultdict(list)
    ace_history     = defaultdict(list)
    df_history      = defaultdict(list)
    bp_history      = defaultdict(list)
    upset_history   = defaultdict(lambda: {"wins": 0, "total": 0})
    rank_history    = defaultdict(list)
    tourney_history = defaultdict(lambda: defaultdict(list))

    for _, row in df_clean.iterrows():
        p1, p2, surface = row["player1"], row["player2"], row["surface"]
        p1_win  = row["p1_win"]
        tourney = row["tourney_name"]

        level_factor   = level_k.get(row["tourney_level"], 32)
        surface_factor = surface_k.get(surface, 32)
        k = (level_factor + surface_factor) / 2

        try:
            current_date = datetime.strptime(str(int(row["tourney_date"])), "%Y%m%d")
        except:
            current_date = datetime.today()

        r1, r2   = model.get_rating(p1), model.get_rating(p2)
        sr1, sr2 = model.get_rating(p1, surface), model.get_rating(p2, surface)

        rank1     = row.get("rank1", np.nan)
        rank2     = row.get("rank2", np.nan)
        rank_diff = rank1 - rank2 if pd.notna(rank1) and pd.notna(rank2) else 0
        age1      = row.get("age1", np.nan)
        age2      = row.get("age2", np.nan)
        age_diff  = age1 - age2 if pd.notna(age1) and pd.notna(age2) else 0

        wr1  = compute_recent_win_rate(p1, match_history)
        wr2  = compute_recent_win_rate(p2, match_history)
        swr1 = compute_surface_win_rate(p1, surface, surface_history)
        swr2 = compute_surface_win_rate(p2, surface, surface_history)
        mom1 = compute_momentum(p1, match_history)
        mom2 = compute_momentum(p2, match_history)
        h2h  = compute_h2h(p1, p2, h2h_record)
        sh2h = compute_surface_h2h(p1, p2, surface, h2h_surface)
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

        # Use 0 for fatigue/rest during training since we use manual input at prediction
        records.append({
            "elo_diff":              r1 - r2,
            "surface_elo_diff":      sr1 - sr2,
            "rank_diff":             rank_diff,
            "age_diff":              age_diff,
            "win_rate_diff":         wr1 - wr2,
            "surface_win_rate_diff": swr1 - swr2,
            "momentum_diff":         mom1 - mom2,
            "fatigue_diff":          0,
            "rest_diff":             0,
            "h2h_p1":                h2h,
            "surface_h2h_p1":        sh2h,
            "serve_diff":            sv1 - sv2,
            "ace_diff":              ac1 - ac2,
            "df_diff":               df1 - df2,
            "bp_diff":               bp1 - bp2,
            "upset_diff":            up1 - up2,
            "rank_traj_diff":        rt1 - rt2,
            "tourney_win_diff":      tw1 - tw2,
            "round":                 row["round"],
            "best_of":               row.get("best_of", 3),
            "surface":               surface,
            "p1_win":                p1_win,
        })

        model.update(p1, p2, p1_win, surface=surface, k=k)
        match_history[p1].append(p1_win)
        match_history[p2].append(1 - p1_win)
        surface_history[p1][surface].append(p1_win)
        surface_history[p2][surface].append(1 - p1_win)
        match_dates[p1].append(current_date)
        match_dates[p2].append(current_date)
        tourney_history[p1][tourney].append(p1_win)
        tourney_history[p2][tourney].append(1 - p1_win)

        if pd.notna(rank1):
            rank_history[p1].append(rank1)
        if pd.notna(rank2):
            rank_history[p2].append(rank2)
        if pd.notna(row["serve1"]):
            serve_history[p1].append(row["serve1"])
        if pd.notna(row["serve2"]):
            serve_history[p2].append(row["serve2"])
        if pd.notna(row["ace1"]):
            ace_history[p1].append(row["ace1"])
        if pd.notna(row["ace2"]):
            ace_history[p2].append(row["ace2"])
        if pd.notna(row["df1"]):
            df_history[p1].append(row["df1"])
        if pd.notna(row["df2"]):
            df_history[p2].append(row["df2"])
        if pd.notna(row["bp1"]):
            bp_history[p1].append(row["bp1"])
        if pd.notna(row["bp2"]):
            bp_history[p2].append(row["bp2"])

        if pd.notna(rank1) and pd.notna(rank2):
            if rank1 > rank2:
                upset_history[p1]["total"] += 1
                upset_history[p1]["wins"]  += p1_win
            if rank2 > rank1:
                upset_history[p2]["total"] += 1
                upset_history[p2]["wins"]  += (1 - p1_win)

        key  = tuple(sorted([p1, p2]))
        skey = key + (surface,)
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

    features_df  = pd.DataFrame(records)
    feature_cols = [
        "elo_diff", "surface_elo_diff", "rank_diff", "age_diff",
        "win_rate_diff", "surface_win_rate_diff", "momentum_diff",
        "fatigue_diff", "rest_diff", "h2h_p1", "surface_h2h_p1",
        "serve_diff", "ace_diff", "df_diff", "bp_diff",
        "upset_diff", "rank_traj_diff", "tourney_win_diff",
        "round", "best_of"
    ]
    X = pd.get_dummies(features_df[feature_cols + ["surface"]], columns=["surface"])
    y = features_df["p1_win"]

    clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4)
    clf.fit(X, y)

    all_players  = sorted(set(df_clean["player1"].tolist() + df_clean["player2"].tolist()))
    all_surfaces = features_df["surface"].unique().tolist()
    all_tourneys = sorted(df_clean["tourney_name"].unique().tolist())

    return (model, clf, match_history, surface_history, match_dates,
            h2h_record, h2h_surface, serve_history, ace_history,
            df_history, bp_history, upset_history, rank_history,
            tourney_history, all_players, all_surfaces, all_tourneys,
            X.columns.tolist())

# ---- App ----
st.title("🎾 Tennis Match Predictor")

with st.spinner("Training model... ~90 seconds on first load"):
    (model, clf, match_history, surface_history, match_dates,
     h2h_record, h2h_surface, serve_history, ace_history,
     df_history, bp_history, upset_history, rank_history,
     tourney_history, all_players, all_surfaces, all_tourneys,
     feature_cols) = train_model()

tab1, tab2, tab3 = st.tabs(["🔮 Predict Match", "📅 Match Calendar", "📊 Odds Comparison"])

# ======== TAB 1: Predict Match ========
with tab1:
    st.subheader("Predict a Match")

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Player 1", all_players, index=0, key="p1")
    with col2:
        p2 = st.selectbox("Player 2", all_players, index=1, key="p2")

    surface   = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
    tourney   = st.selectbox("Tournament (optional)", ["Unknown"] + all_tourneys)
    best_of   = st.radio("Best of", [3, 5], horizontal=True)
    round_num = st.select_slider("Round", options=[1,2,3,4,5,6,7],
                                  format_func=lambda x: ["R128","R64","R32","R16","QF","SF","F"][x-1])

    st.subheader("Fatigue & Rest (enter manually for accuracy)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{p1}**")
        p1_fatigue = st.number_input(f"Matches played in last 14 days", min_value=0, max_value=15, value=0, key="fat1")
        p1_rest    = st.number_input(f"Days since last match", min_value=0, max_value=365, value=3, key="rest1")
    with col2:
        st.write(f"**{p2}**")
        p2_fatigue = st.number_input(f"Matches played in last 14 days", min_value=0, max_value=15, value=0, key="fat2")
        p2_rest    = st.number_input(f"Days since last match", min_value=0, max_value=365, value=3, key="rest2")

    if st.button("Predict", type="primary"):
        if p1 == p2:
            st.error("Please select two different players.")
        else:
            prob, r1, r2, sr1, sr2, wr1, wr2, h2h, sh2h, mom1, mom2, sv1, sv2 = get_prediction(
                p1, p2, surface, best_of, round_num, tourney,
                p1_fatigue, p2_fatigue, p1_rest, p2_rest,
                model, clf, feature_cols, match_history, surface_history,
                h2h_record, h2h_surface, serve_history, ace_history,
                df_history, bp_history, upset_history, rank_history,
                tourney_history, all_surfaces
            )

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric(p1, f"{prob * 100:.1f}%")
            with col2:
                st.metric(p2, f"{(1-prob) * 100:.1f}%")

            winner = p1 if prob > 0.5 else p2
            conf   = max(prob, 1-prob) * 100
            st.success(f"Predicted winner: **{winner}** ({conf:.1f}% confidence)")

            with st.expander("See detailed breakdown"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{p1}**")
                    st.write(f"Overall Elo: {r1:.0f}")
                    st.write(f"{surface} Elo: {sr1:.0f}")
                    st.write(f"Recent win rate: {wr1*100:.1f}%")
                    st.write(f"Momentum: {mom1*100:.1f}%")
                    st.write(f"Serve %: {sv1*100:.1f}%")
                    st.write(f"Fatigue: {p1_fatigue} matches in 2 weeks")
                    st.write(f"Days rest: {p1_rest}")
                with col2:
                    st.write(f"**{p2}**")
                    st.write(f"Overall Elo: {r2:.0f}")
                    st.write(f"{surface} Elo: {sr2:.0f}")
                    st.write(f"Recent win rate: {wr2*100:.1f}%")
                    st.write(f"Momentum: {mom2*100:.1f}%")
                    st.write(f"Serve %: {sv2*100:.1f}%")
                    st.write(f"Fatigue: {p2_fatigue} matches in 2 weeks")
                    st.write(f"Days rest: {p2_rest}")
                st.write(f"**Overall H2H for {p1}**: {h2h*100:.1f}%")
                st.write(f"**{surface} H2H for {p1}**: {sh2h*100:.1f}%")

# ======== TAB 2: Match Calendar ========
with tab2:
    st.subheader("📅 Upcoming ATP Matches — Next 30 Days")

    api_key = st.secrets.get("TENNIS_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Enter your api-tennis.com API key", type="password", key="api_key_cal")

    if not api_key:
        st.info("Enter your API key from api-tennis.com to see the match calendar.")

        # Manual match entry fallback
        st.subheader("Or add matches manually")
        st.write("Add upcoming matches you want to track:")

        if "manual_matches" not in st.session_state:
            st.session_state.manual_matches = []

        with st.form("add_match"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                m_date = st.date_input("Date")
            with c2:
                m_p1 = st.selectbox("Player 1", all_players, key="m_p1")
            with c3:
                m_p2 = st.selectbox("Player 2", all_players, index=1, key="m_p2")
            with c4:
                m_surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], key="m_surf")
            m_tourney = st.text_input("Tournament name")
            m_best_of = st.radio("Best of", [3, 5], horizontal=True, key="m_bo")
            submitted = st.form_submit_button("Add Match")
            if submitted and m_p1 != m_p2:
                st.session_state.manual_matches.append({
                    "date":    m_date,
                    "player1": m_p1,
                    "player2": m_p2,
                    "surface": m_surface,
                    "tourney": m_tourney or "Unknown",
                    "best_of": m_best_of,
                })

        if st.session_state.manual_matches:
            # Group by date
            by_date = defaultdict(list)
            for m in st.session_state.manual_matches:
                by_date[m["date"]].append(m)

            for match_date in sorted(by_date.keys()):
                st.markdown(f"### 📆 {match_date.strftime('%A, %B %d %Y')}")
                for m in by_date[match_date]:
                    prob, r1, r2, sr1, sr2, wr1, wr2, h2h, sh2h, mom1, mom2, sv1, sv2 = get_prediction(
                        m["player1"], m["player2"], m["surface"],
                        m["best_of"], 3, m["tourney"],
                        0, 0, 3, 3,
                        model, clf, feature_cols, match_history, surface_history,
                        h2h_record, h2h_surface, serve_history, ace_history,
                        df_history, bp_history, upset_history, rank_history,
                        tourney_history, all_surfaces
                    )
                    winner = m["player1"] if prob > 0.5 else m["player2"]
                    conf   = max(prob, 1-prob) * 100
                    with st.container():
                        c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
                        with c1:
                            st.write(f"🎾 **{m['player1']}** vs **{m['player2']}**")
                            st.caption(f"{m['tourney']} | {m['surface']}")
                        with c2:
                            st.write(f"{m['player1']}: **{prob*100:.1f}%**")
                            st.write(f"{m['player2']}: **{(1-prob)*100:.1f}%**")
                        with c3:
                            st.write(f"🏆 **{winner}**")
                        with c4:
                            st.write(f"Confidence: {conf:.1f}%")
                        st.divider()

            if st.button("Clear all matches"):
                st.session_state.manual_matches = []
                st.rerun()
    else:
        with st.spinner("Fetching upcoming matches..."):
            matches = fetch_upcoming_matches(api_key)

        if not matches:
            st.warning("No upcoming matches found from the API.")
        else:
            by_date = defaultdict(list)
            for m in matches:
                try:
                    p1_name = m.get("event_first_player", "")
                    p2_name = m.get("event_second_player", "")
                    surface = m.get("event_surface", "Hard")
                    tourney = m.get("tournament_name", "Unknown")
                    match_date = m.get("event_date", "")

                    if not p1_name or not p2_name:
                        continue

                    p1_match = next((p for p in all_players if p1_name.split()[-1].lower() in p.lower()), None)
                    p2_match = next((p for p in all_players if p2_name.split()[-1].lower() in p.lower()), None)

                    if not p1_match or not p2_match or p1_match == p2_match:
                        continue

                    by_date[match_date].append({
                        "player1": p1_match,
                        "player2": p2_match,
                        "surface": surface,
                        "tourney": tourney,
                    })
                except:
                    continue

            for match_date in sorted(by_date.keys()):
                try:
                    date_obj = datetime.strptime(match_date, "%Y-%m-%d")
                    st.markdown(f"### 📆 {date_obj.strftime('%A, %B %d %Y')}")
                except:
                    st.markdown(f"### 📆 {match_date}")

                for m in by_date[match_date]:
                    prob, r1, r2, sr1, sr2, wr1, wr2, h2h, sh2h, mom1, mom2, sv1, sv2 = get_prediction(
                        m["player1"], m["player2"], m["surface"], 3, 3, m["tourney"],
                        0, 0, 3, 3,
                        model, clf, feature_cols, match_history, surface_history,
                        h2h_record, h2h_surface, serve_history, ace_history,
                        df_history, bp_history, upset_history, rank_history,
                        tourney_history, all_surfaces
                    )
                    winner = m["player1"] if prob > 0.5 else m["player2"]
                    conf   = max(prob, 1-prob) * 100
                    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
                    with c1:
                        st.write(f"🎾 **{m['player1']}** vs **{m['player2']}**")
                        st.caption(f"{m['tourney']} | {m['surface']}")
                    with c2:
                        st.write(f"{m['player1']}: **{prob*100:.1f}%**")
                        st.write(f"{m['player2']}: **{(1-prob)*100:.1f}%**")
                    with c3:
                        st.write(f"🏆 **{winner}**")
                    with c4:
                        st.write(f"Confidence: {conf:.1f}%")
                    st.divider()

# ======== TAB 3: Odds Comparison ========
with tab3:
    st.subheader("📊 Odds Comparison — Find Value Bets")
    st.write("Enter the Vegas odds to see if your model disagrees and where the value is.")

    col1, col2 = st.columns(2)
    with col1:
        op1 = st.selectbox("Player 1", all_players, index=0, key="op1")
    with col2:
        op2 = st.selectbox("Player 2", all_players, index=1, key="op2")

    osurface  = st.selectbox("Surface", ["Hard", "Clay", "Grass"], key="osurf")
    obest_of  = st.radio("Best of", [3, 5], horizontal=True, key="obo")
    oround    = st.select_slider("Round", options=[1,2,3,4,5,6,7],
                                  format_func=lambda x: ["R128","R64","R32","R16","QF","SF","F"][x-1],
                                  key="oround")

    st.subheader("Vegas Odds (American format)")
    col1, col2 = st.columns(2)
    with col1:
        odds1 = st.number_input(f"{op1} odds (e.g. -150 or +120)", value=-150, key="odds1")
    with col2:
        odds2 = st.number_input(f"{op2} odds (e.g. -150 or +120)", value=120, key="odds2")

    col1, col2 = st.columns(2)
    with col1:
        of1 = st.number_input("Matches in last 14 days", min_value=0, max_value=15, value=0, key="of1")
        or1 = st.number_input("Days since last match", min_value=0, max_value=365, value=3, key="or1")
    with col2:
        of2 = st.number_input("Matches in last 14 days", min_value=0, max_value=15, value=0, key="of2")
        or2 = st.number_input("Days since last match", min_value=0, max_value=365, value=3, key="or2")

    if st.button("Compare", type="primary"):
        if op1 == op2:
            st.error("Please select two different players.")
        else:
            prob, r1, r2, sr1, sr2, wr1, wr2, h2h, sh2h, mom1, mom2, sv1, sv2 = get_prediction(
                op1, op2, osurface, obest_of, oround, "Unknown",
                of1, of2, or1, or2,
                model, clf, feature_cols, match_history, surface_history,
                h2h_record, h2h_surface, serve_history, ace_history,
                df_history, bp_history, upset_history, rank_history,
                tourney_history, all_surfaces
            )

            imp1 = implied_prob(odds1)
            imp2 = implied_prob(odds2)

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric(op1, f"Model: {prob*100:.1f}%", delta=f"Vegas: {imp1*100:.1f}%")
            with col2:
                st.metric(op2, f"Model: {(1-prob)*100:.1f}%", delta=f"Vegas: {imp2*100:.1f}%")

            diff1 = prob - imp1
            diff2 = (1-prob) - imp2

            st.divider()
            if abs(diff1) > 0.05 or abs(diff2) > 0.05:
                st.subheader("⚡ Value Detected")
                if diff1 > 0.05:
                    st.success(f"**{op1}** looks undervalued — model gives {prob*100:.1f}% but Vegas implies {imp1*100:.1f}%. Edge: +{diff1*100:.1f}%")
                if diff2 > 0.05:
                    st.success(f"**{op2}** looks undervalued — model gives {(1-prob)*100:.1f}% but Vegas implies {imp2*100:.1f}%. Edge: +{diff2*100:.1f}%")
                if diff1 < -0.05:
                    st.warning(f"**{op1}** looks overvalued — model gives {prob*100:.1f}% but Vegas implies {imp1*100:.1f}%")
                if diff2 < -0.05:
                    st.warning(f"**{op2}** looks overvalued — model gives {(1-prob)*100:.1f}% but Vegas implies {imp2*100:.1f}%")
            else:
                st.info("No significant edge detected — your model and Vegas are in agreement on this match.")
                
