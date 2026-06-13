import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import ssl

# --- BULLETPROOF MAC SSL BYPASS ---
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

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

def build_and_train():
    print("STARTING MODEL TRAINING PROCESS")
    print("Downloading historical data (2015-2025)...")
    years = range(2015, 2026)
    frames = []
    for year in years:
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        try: frames.append(pd.read_csv(url, low_memory=False))
        except: continue

    frames = [f for f in frames if f is not None and not f.empty]
    
    if len(frames) == 0:
        print("Error: No data sheets could be downloaded. Check your internet connection!")
        return

    df = pd.concat(frames).sort_values("tourney_date").reset_index(drop=True)
    df = df[df["score"].notna() & ~df["score"].str.contains("W/O|RET|DEF", na=False)].reset_index(drop=True)

    print(f"Data downloaded successfully. Processing {len(df)} matches...")

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
        "score":         df["score"],
        "p1_win":        np.where(flip, 0, 1),
    })

    model_elo = EloModel()
    
    print("Building features...")
    clean_records = df_clean.to_dict('records')
    
    for row in clean_records:
        p1, p2, surface = row["player1"], row["player2"], row["surface"]
        p1_win  = row["p1_win"]
        k = (level_k.get(row["tourney_level"], 32) + surface_k.get(surface, 32)) / 2
        dom = parse_score(row["score"])
        model_elo.update(p1, p2, p1_win, dom_ratio=dom, surface=surface, k=k)

    # Simplified artifacts save for speed
    all_players = sorted(set(df_clean["player1"].tolist() + df_clean["player2"].tolist()))
    
    artifacts = {
        "model_elo": model_elo,
        "clfs": {}, 
        "all_players": all_players, 
        "all_surfaces": df_clean["surface"].unique().tolist(), 
        "all_tourneys": sorted(df_clean["tourney_name"].unique().tolist()),
    }
    return artifacts

if __name__ == "__main__":
    artifacts = build_and_train()
    if artifacts:
        print("Training complete. Saving artifacts to disk...")
        joblib.dump(artifacts, "tennis_model_artifacts.pkl")
        print("SUCCESS! You can now run your Streamlit app.")
