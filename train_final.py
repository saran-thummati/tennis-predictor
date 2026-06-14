import pandas as pd
import numpy as np
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
import joblib
import warnings
import ssl

from elo_model import EloModel 

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
    print("STARTING ELITE OPTIMIZED PIPELINE...")
    years = range(2015, 2026)
    frames = []
    for year in years:
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        try: frames.append(pd.read_csv(url, low_memory=False))
        except: continue

    df = pd.concat(frames).sort_values("tourney_date").reset_index(drop=True)
    df = df[df["score"].notna() & ~df["score"].str.contains("W/O|RET|DEF", na=False)].reset_index(drop=True)

    np.random.seed(42)
    flip = np.random.rand(len(df)) < 0.5

    df_clean = pd.DataFrame({
        "player1": np.where(flip, df["loser_name"], df["winner_name"]),
        "player2": np.where(flip, df["winner_name"], df["loser_name"]),
        "p1_age": np.where(flip, df["loser_age"], df["winner_age"]),
        "p2_age": np.where(flip, df["winner_age"], df["loser_age"]),
        "p1_hand": np.where(flip, df["loser_hand"], df["winner_hand"]),
        "p2_hand": np.where(flip, df["winner_hand"], df["loser_hand"]),
        "surface": df["surface"],
        "tourney_level": df["tourney_level"], # NEW: To track tournament importance
        "score": df["score"],
        "p1_win": np.where(flip, 0, 1),
    })
    
    df_clean['p1_hand'] = df_clean['p1_hand'].replace({'U': 'R', np.nan: 'R'})
    df_clean['p2_hand'] = df_clean['p2_hand'].replace({'U': 'R', np.nan: 'R'})
    df_clean.fillna({'p1_age': 25.0, 'p2_age': 25.0}, inplace=True)

    model_elo = EloModel()
    h2h_tracker = defaultdict(int) 
    
    # NEW: Surface-Specific Form Tracker!
    surface_form = defaultdict(lambda: defaultdict(lambda: deque(maxlen=10)))
    
    player_bio = {} 
    player_stats = defaultdict(lambda: {"matches": 0, "wins": 0})
    
    X_train, y_train, weights_train = [], [], []
    
    clean_records = df_clean.to_dict('records')
    print(f"Processing {len(clean_records)} matches with Surface Logic and Grand Slam Weighting...")
    
    for row in clean_records:
        p1, p2, surface = row["player1"], row["player2"], row["surface"]
        p1_win, p1_age, p2_age = row["p1_win"], row["p1_age"], row["p2_age"]
        p1_hand, p2_hand = row["p1_hand"], row["p2_hand"]
        t_level = row["tourney_level"]
        
        player_bio[p1] = {"age": p1_age, "hand": p1_hand}
        player_bio[p2] = {"age": p2_age, "hand": p2_hand}
        
        r1_base, r2_base = model_elo.get_rating(p1), model_elo.get_rating(p2)
        r1_surf, r2_surf = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
        h2h_adv = h2h_tracker[f"{p1}_vs_{p2}"] - h2h_tracker[f"{p2}_vs_{p1}"]
        
        # Calculate surface-specific form
        p1_surf_history = surface_form[p1][surface]
        p2_surf_history = surface_form[p2][surface]
        form_1_surf = sum(p1_surf_history) / len(p1_surf_history) if p1_surf_history else 0.5
        form_2_surf = sum(p2_surf_history) / len(p2_surf_history) if p2_surf_history else 0.5
        
        p1_is_lefty = 1 if p1_hand == 'L' else 0
        p2_is_lefty = 1 if p2_hand == 'L' else 0

        if model_elo.match_counts[p1] > 5 and model_elo.match_counts[p2] > 5:
            X_train.append([
                r1_base - r2_base, 
                r1_surf - r2_surf, 
                h2h_adv,           
                form_1_surf - form_2_surf, # Upgraded Feature!
                p1_age - p2_age,   
                p1_is_lefty - p2_is_lefty 
            ])
            y_train.append(p1_win)
            
            # Match Importance Weighting
            if t_level == 'G': weight = 2.0   # Grand Slams matter most
            elif t_level == 'M': weight = 1.5 # Masters 1000s matter a lot
            else: weight = 1.0                # Standard ATP 250/500
            weights_train.append(weight)
        
        dom = parse_score(row["score"])
        model_elo.update(p1, p2, p1_win, dom_ratio=dom, surface=surface)
        
        player_stats[p1]["matches"] += 1
        player_stats[p2]["matches"] += 1
        
        # Update surface-specific trackers
        if p1_win == 1:
            h2h_tracker[f"{p1}_vs_{p2}"] += 1
            surface_form[p1][surface].append(1)
            surface_form[p2][surface].append(0)
            player_stats[p1]["wins"] += 1
        else:
            h2h_tracker[f"{p2}_vs_{p1}"] += 1
            surface_form[p1][surface].append(0)
            surface_form[p2][surface].append(1)
            player_stats[p2]["wins"] += 1

    print("Training the Optimized Ensemble Model...")
    
    # 1. Hyperparameter-Tuned Random Forest
    rf_expert = RandomForestClassifier(
        n_estimators=150, 
        max_depth=9,          # Deeper trees for better logic
        min_samples_split=5,  # Prevents overfitting
        random_state=42
    )
    
    # 2. Hyperparameter-Tuned Gradient Booster
    hgb_expert = HistGradientBoostingClassifier(
        max_iter=200, 
        max_depth=7, 
        learning_rate=0.05,  # Slower learning rate for higher accuracy
        l2_regularization=0.1, # Math trick to keep predictions stable
        random_state=42
    )
    
    # Train both models with our new Match Importance Weights!
    rf_expert.fit(np.array(X_train), np.array(y_train), sample_weight=np.array(weights_train))
    hgb_expert.fit(np.array(X_train), np.array(y_train), sample_weight=np.array(weights_train))
    
    ensemble_model = VotingClassifier(
        estimators=[('Random_Forest', rf_expert), ('Gradient_Boost', hgb_expert)],
        voting='soft'
    )
    # The VotingClassifier just wraps them, since they are already fitted.
    ensemble_model.estimators_ = [rf_expert, hgb_expert]
    ensemble_model.le_ = None
    ensemble_model.classes_ = np.array([0, 1])

    all_players = sorted(set(df_clean["player1"].tolist() + df_clean["player2"].tolist()))
    
    final_stats = {}
    for p, stats in player_stats.items():
        win_rate = (stats["wins"] / stats["matches"]) * 100 if stats["matches"] > 0 else 0
        final_stats[p] = {"matches": stats["matches"], "win_rate": round(win_rate, 1)}
    
    # Convert defaultdicts to regular dicts for safe saving
    clean_surface_form = {p: dict(surfaces) for p, surfaces in surface_form.items()}

    artifacts = {
        "model_elo": model_elo,
        "ai_model": ensemble_model, 
        "h2h_tracker": dict(h2h_tracker),
        "surface_form": clean_surface_form, # NEW!
        "player_bio": player_bio,
        "player_stats": final_stats,
        "all_players": all_players
    }
    return artifacts

if __name__ == "__main__":
    artifacts = build_and_train()
    if artifacts:
        print("Saving Elite artifacts to disk...")
        joblib.dump(artifacts, "tennis_model_artifacts.pkl")
        print("SUCCESS! The Ultimate AI is ready.")
