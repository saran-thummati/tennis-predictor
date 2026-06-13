import pandas as pd
import numpy as np
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
import ssl

# --- IMPORT OUR EXTERNAL BLUEPRINT ---
from elo_model import EloModel 

# --- MAC SSL BYPASS ---
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

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

def build_and_train():
    print("STARTING MACHINE LEARNING PIPELINE")
    print("Downloading historical data (2015-2025)...")
    years = range(2015, 2026)
    frames = []
    for year in years:
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        try: frames.append(pd.read_csv(url, low_memory=False))
        except: continue

    frames = [f for f in frames if f is not None and not f.empty]
    if len(frames) == 0:
        print("Error: No data sheets could be downloaded.")
        return

    df = pd.concat(frames).sort_values("tourney_date").reset_index(drop=True)
    df = df[df["score"].notna() & ~df["score"].str.contains("W/O|RET|DEF", na=False)].reset_index(drop=True)

    print(f"Processing {len(df)} matches to build features...")

    level_k   = {"G": 50, "M": 40, "A": 32, "D": 24, "F": 32}
    surface_k = {"Hard": 32, "Clay": 32, "Grass": 20, "Carpet": 20}
    np.random.seed(42)
    flip = np.random.rand(len(df)) < 0.5

    df_clean = pd.DataFrame({
        "player1": np.where(flip, df["loser_name"],  df["winner_name"]),
        "player2": np.where(flip, df["winner_name"], df["loser_name"]),
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],
        "score": df["score"],
        "p1_win": np.where(flip, 0, 1),
    })

    model_elo = EloModel()
    
    h2h_tracker = defaultdict(int) 
    recent_form = defaultdict(lambda: deque(maxlen=10)) 
    player_stats = defaultdict(lambda: {"matches": 0, "wins": 0, "dom_scores": []})
    
    X_train = []
    y_train = []
    
    clean_records = df_clean.to_dict('records')
    
    for row in clean_records:
        p1, p2, surface = row["player1"], row["player2"], row["surface"]
        p1_win = row["p1_win"]
        k = (level_k.get(row["tourney_level"], 32) + surface_k.get(surface, 32)) / 2
        dom = parse_score(row["score"])
        
        r1_base, r2_base = model_elo.get_rating(p1), model_elo.get_rating(p2)
        r1_surf, r2_surf = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
        
        matchup_hash_p1 = f"{p1}_vs_{p2}"
        matchup_hash_p2 = f"{p2}_vs_{p1}"
        h2h_adv = h2h_tracker[matchup_hash_p1] - h2h_tracker[matchup_hash_p2]
        
        form_1 = sum(recent_form[p1]) / len(recent_form[p1]) if recent_form[p1] else 0.5
        form_2 = sum(recent_form[p2]) / len(recent_form[p2]) if recent_form[p2] else 0.5

        if model_elo.match_counts[p1] > 5 and model_elo.match_counts[p2] > 5:
            X_train.append([
                r1_base - r2_base, 
                r1_surf - r2_surf, 
                h2h_adv,           
                form_1 - form_2    
            ])
            y_train.append(p1_win)
        
        model_elo.update(p1, p2, p1_win, dom_ratio=dom, surface=surface, k=k)
        
        if p1_win == 1:
            h2h_tracker[matchup_hash_p1] += 1
            recent_form[p1].append(1)
            recent_form[p2].append(0)
            player_stats[p1]["wins"] += 1
        else:
            h2h_tracker[matchup_hash_p2] += 1
            recent_form[p1].append(0)
            recent_form[p2].append(1)
            player_stats[p2]["wins"] += 1
            
        player_stats[p1]["matches"] += 1
        player_stats[p2]["matches"] += 1

    print("Training Random Forest Classifier on historical patterns...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rf_model.fit(X_train, y_train)

    print("Formatting final stats...")
    final_stats = {}
    for p, stats in player_stats.items():
        win_rate = (stats["wins"] / stats["matches"]) * 100 if stats["matches"] > 0 else 0
        final_stats[p] = {
            "matches": stats["matches"],
            "win_rate": round(win_rate, 1),
            "recent_form": round((sum(recent_form[p]) / len(recent_form[p]) * 100) if recent_form[p] else 50, 1)
        }

    all_players = sorted(set(df_clean["player1"].tolist() + df_clean["player2"].tolist()))
    
    artifacts = {
        "model_elo": model_elo,
        "rf_model": rf_model, 
        "h2h_tracker": dict(h2h_tracker),
        "all_players": all_players, 
        "player_stats": final_stats
    }
    return artifacts

if __name__ == "__main__":
    artifacts = build_and_train()
    if artifacts:
        print("Saving heavy artifacts to disk...")
        joblib.dump(artifacts, "tennis_model_artifacts.pkl")
        print("SUCCESS! Model is ready for deployment.")
