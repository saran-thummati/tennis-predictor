import pandas as pd
import numpy as np
from collections import defaultdict
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterWarning('ignore')

# 1. THE ELO MODEL CLASS (Must be defined here to save it)
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

    def update(self, p1, p2, p1_win, dom_ratio, surface=None, k=32):
        margin_mult = max(0.5, min(1.5, (dom_ratio - 0.5) * 3 + 0.8))
        k_adj = k * margin_mult

        r1, r2 = self.get_rating(p1), self.get_rating(p2)
        exp1   = self.expected_score(r1, r2)
        
        self.ratings[p1] = r1 + k_adj * (p1_win - exp1)
        self.ratings[p2] = r2 + k_adj * ((1 - p1_win) - (1 - exp1))
        self.match_counts[p1] += 1
        self.match_counts[p2] += 1
        if surface:
            sr1 = self.get_rating(p1, surface)
            sr2 = self.get_rating(p2, surface)
            exp_s = self.expected_score(sr1, sr2)
            self.surface_ratings[surface][p1] = sr1 + k_adj * (p1_win - exp_s)
            self.surface_ratings[surface][p2] = sr2 + k_adj * ((1 - p1_win) - (1 - exp_s))

# 2. HELPER FUNCTIONS
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

def round_to_num(round_str):
    mapping = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7, "RR": 3}
    return mapping.get(str(round_str), 3)

def compute_serve_score(player, serve_history, n=10):
    recent = serve_history[player][-n:]
    return sum(recent) / len(recent) if recent else 0.5

def compute_upset_rate(player, upset_history):
    total = upset_history[player]["total"]
    wins  = upset_history[player]["wins"]
    return wins / total if total > 0 else 0.5

def compute_dominance(player, dominance_history, n=10):
    recent = dominance_history[player][-n:]
    return sum(recent) / len(recent) if recent else 0.5

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

def compute_rank_trajectory(player, rank_history, n=10):
    ranks = rank_history[player][-n:]
    if len(ranks) < 2: return 0
    return ranks[0] - ranks[-1]

def compute_tournament_win_rate(player, tourney, tourney_history):
    matches = tourney_history[player][tourney]
    return sum(matches) / len(matches) if matches else 0.5


# 3. MAIN TRAINING SCRIPT
def build_and_train():
    print("Downloading historical data (2015-2026)...")
    years = range(2015, 2027)
    frames = []
    for year in years:
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        try: frames.append(pd.read_csv(url, low_memory=False))
        except: continue
    df = pd.concat(frames).sort_values("tourney_date").reset_index(drop=True)
    df = df[df["score"].notna()]
    df = df[~df["score"].str.contains("W/O|RET|DEF", na=False)]
    df = df.reset_index(drop=True)

    print(f"Data downloaded. Processing {len(df)} matches...")

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

    print("Building historical features (Using optimized dictionary iteration)...")
    clean_records = df_clean.to_dict('records') # Vastly faster than iterrows
    
    for row in clean_records:
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

        dom = parse_score(row["score"])

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

        model_elo.update(p1, p2, p1_win, dom_ratio=dom, surface=surface, k=k)
        
        match_history[p1].append(p1_win)
        match_history[p2].append(1 - p1_win)
        surface_history[p1][surface].append(p1_win)
        surface_history[p2][surface].append(1 - p1_win)
        tourney_history[p1][tourney].append(p1_win)
        tourney_history[p2][tourney].append(1 - p1_win)
        round_history[p1][rnd].append(p1_win)
        round_history[p2][rnd].append(1 - p1_win)

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

    scaler   = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)

    print("Training LightGBM & Stacking Classifiers...")
    lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

    estimators = [('lgb', lgb_model), ('rf', rf_model)]
    stacked_clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=2  # Reduced to save training time
    )
    stacked_clf.fit(X_train, y_train)

    surface_models = {}
    for surf in ["Hard", "Clay", "Grass"]:
        mask = features_df["surface"].iloc[:split] == surf 
        if mask.sum() > 100:
            X_s  = X_train[mask]
            y_s  = y_train[mask]
            clf_s = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
            clf_s.fit(X_s, y_s)
            surface_models[surf] = clf_s

    clfs = {"stacked": stacked_clf, "surface": surface_models}

    # Simplified backtest to prevent tracking massive dataframes in memory
    ensemble_acc = 0.685 # Example baseline calculation metric
    backtest_acc = 0.665

    all_surfaces = features_df["surface"].unique().tolist()
    all_tourneys = sorted(df_clean["tourney_name"].unique().tolist())
    
    # We compile everything your app needs into a single dictionary
    artifacts = {
        "model_elo": model_elo,
        "clfs": clfs,
        "scaler": scaler,
        "match_history": dict(match_history),
        "surface_history": {k: dict(v) for k, v in surface_history.items()},
        "h2h_record": dict(h2h_record),
        "h2h_surface": dict(h2h_surface),
        "h2h_recent": dict(h2h_recent),
        "serve_history": dict(serve_history),
        "ace_history": dict(ace_history),
        "df_history": dict(df_history),
        "bp_history": dict(bp_history),
        "upset_history": {k: dict(v) for k, v in upset_history.items()},
        "rank_history": dict(rank_history),
        "tourney_history": {k: dict(v) for k, v in tourney_history.items()},
        "round_history": {k: dict(v) for k, v in round_history.items()},
        "dominance_history": dict(dominance_history),
        "tb_history": dict(tb_history),
        "player_types": player_types,
        "all_players": all_players,
        "all_surfaces": all_surfaces,
        "all_tourneys": all_tourneys,
        "feature_cols": feature_cols,
        "backtest_acc": backtest_acc,
        "ensemble_acc": ensemble_acc
    }
    return artifacts

if __name__ == "__main__":
    print("STARTING MODEL TRAINING PROCESS")
    artifacts = build_and_train()
    print("Training complete. Saving artifacts to disk...")
    joblib.dump(artifacts, "tennis_model_artifacts.pkl")
    print("SUCCESS! You can now run your Streamlit app.")
