import pandas as pd
import numpy as np
import io
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
import joblib
import warnings
import requests
import time

from elo_model import EloModel 

warnings.filterwarnings('ignore')

# Disguise as Chrome to avoid blocks
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/vnd.github.v3.raw"
}

def parse_score(score_str):
    """Parse tennis score to extract dominance ratio"""
    try:
        sets = str(score_str).split()
        w_games = l_games = 0
        for s in sets:
            parts = s.replace("(", " ").replace(")", "").split("-")
            if len(parts) >= 2:
                w_games += int(parts[0])
                l_games += int(parts[1].split()[0])
        total = w_games + l_games
        return max(0.51, min(0.99, w_games / total if total > 0 else 0.7))
    except:
        return 0.7

def download_atp_data():
    """Download ATP match data from GitHub with fallback URLs"""
    print("Downloading ATP tennis data...")
    
    years = range(2015, 2026)
    frames = []
    failed_years = []
    
    # Multiple fallback URLs in priority order
    url_templates = [
        "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/main/atp_matches_{year}.csv",
        "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
    ]
    
    for year in years:
        success = False
        
        for url_template in url_templates:
            url = url_template.format(year=year)
            try:
                response = requests.get(url, headers=HEADERS, timeout=15)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text), low_memory=False)
                if len(df) > 0:
                    frames.append(df)
                    print(f"  ✓ {year}: {len(df)} matches")
                    success = True
                    break
            except Exception as e:
                continue
        
        if not success:
            print(f"  ✗ {year}: Not available")
            failed_years.append(year)
        
        time.sleep(1)  # Rate limiting
    
    if not frames:
        raise Exception(f"Failed to download data for any year! Attempted: {failed_years}")
    
    print(f"✓ Successfully downloaded {len(years) - len(failed_years)}/{len(years)} years of data\n")
    return pd.concat(frames, ignore_index=True)

def build_and_train():
    print("STARTING ELITE OPTIMIZED PIPELINE...\n")
    
    # Download data with fallback URLs
    df = download_atp_data()
    
    # Clean data
    df = df.sort_values("tourney_date").reset_index(drop=True)
    df = df[df["score"].notna() & ~df["score"].str.contains("W/O|RET|DEF", na=False, regex=True)].reset_index(drop=True)
    
    print(f"Total valid matches: {len(df)}\n")

    # Randomly flip matches to create balanced training data
    np.random.seed(42)
    flip = np.random.rand(len(df)) < 0.5

    df_clean = pd.DataFrame({
        "player1": np.where(flip, df["loser_name"], df["winner_name"]),
        "player2": np.where(flip, df["winner_name"], df["loser_name"]),
        "p1_age": np.where(flip, df["loser_age"], df["winner_age"]),
        "p2_age": np.where(flip, df["winner_age"], df["loser_age"]),
        "p1_hand": np.where(flip, df["loser_hand"], df["winner_hand"]),
        "p2_hand": np.where(flip, df["winner_hand"], df["loser_hand"]),
        "p1_ht": np.where(flip, df["loser_ht"], df["winner_ht"]),
        "p2_ht": np.where(flip, df["winner_ht"], df["loser_ht"]),
        "p1_rank": np.where(flip, df["loser_rank"], df["winner_rank"]),
        "p2_rank": np.where(flip, df["winner_rank"], df["loser_rank"]),
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],
        "score": df["score"],
        "p1_win": np.where(flip, 0, 1),
    })
    
    # Clean missing values
    df_clean['p1_hand'] = df_clean['p1_hand'].replace({'U': 'R', np.nan: 'R'})
    df_clean['p2_hand'] = df_clean['p2_hand'].replace({'U': 'R', np.nan: 'R'})
    df_clean.fillna({
        'p1_age': 25.0, 'p2_age': 25.0,
        'p1_ht': 185.0, 'p2_ht': 185.0,
        'p1_rank': 500.0, 'p2_rank': 500.0
    }, inplace=True)

    model_elo = EloModel()
    h2h_tracker = defaultdict(int)
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
        p1_ht, p2_ht = row["p1_ht"], row["p2_ht"]
        p1_rank, p2_rank = row["p1_rank"], row["p2_rank"]
        t_level = row["tourney_level"]
        
        player_bio[p1] = {"age": p1_age, "hand": p1_hand, "height": p1_ht, "rank": p1_rank}
        player_bio[p2] = {"age": p2_age, "hand": p2_hand, "height": p2_ht, "rank": p2_rank}
        
        r1_base, r2_base = model_elo.get_rating(p1), model_elo.get_rating(p2)
        r1_surf, r2_surf = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
        h2h_adv = h2h_tracker[f"{p1}_vs_{p2}"] - h2h_tracker[f"{p2}_vs_{p1}"]
        
        # Surface-specific form
        p1_surf_history = surface_form[p1][surface]
        p2_surf_history = surface_form[p2][surface]
        form_1_surf = sum(p1_surf_history) / len(p1_surf_history) if p1_surf_history else 0.5
        form_2_surf = sum(p2_surf_history) / len(p2_surf_history) if p2_surf_history else 0.5
        
        p1_is_lefty = 1 if p1_hand == 'L' else 0
        p2_is_lefty = 1 if p2_hand == 'L' else 0

        # Only train on players with sufficient history
        if model_elo.match_counts[p1] > 5 and model_elo.match_counts[p2] > 5:
            X_train.append([
                r1_base - r2_base, 
                r1_surf - r2_surf, 
                h2h_adv,           
                form_1_surf - form_2_surf,
                p1_age - p2_age,   
                p1_is_lefty - p2_is_lefty,
                p1_ht - p2_ht,
                p2_rank - p1_rank
            ])
            y_train.append(p1_win)
            
            # Match Importance Weighting
            if t_level == 'G': weight = 2.0   # Grand Slams
            elif t_level == 'M': weight = 1.5 # Masters 1000s
            else: weight = 1.0                # Standard ATP
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

    print(f"✓ Features engineered: {len(X_train)} samples\n")
    print("Training the Optimized Ensemble Model...")
    
    # Random Forest Expert
    rf_expert = RandomForestClassifier(
        n_estimators=150, 
        max_depth=9,
        min_samples_split=5,
        random_state=42
    )
    
    # Gradient Boosting Expert
    hgb_expert = HistGradientBoostingClassifier(
        max_iter=200, 
        max_depth=7, 
        learning_rate=0.05,
        l2_regularization=0.1,
        random_state=42
    )
    
    # Train both models with Match Importance Weights
    X_array = np.array(X_train)
    y_array = np.array(y_train)
    w_array = np.array(weights_train)
    
    rf_expert.fit(X_array, y_array, sample_weight=w_array)
    hgb_expert.fit(X_array, y_array, sample_weight=w_array)
    
    # Create ensemble
    ensemble_model = VotingClassifier(
        estimators=[('Random_Forest', rf_expert), ('Gradient_Boost', hgb_expert)],
        voting='soft'
    )
    ensemble_model.estimators_ = [rf_expert, hgb_expert]
    ensemble_model.le_ = None
    ensemble_model.classes_ = np.array([0, 1])

    all_players = sorted(set(df_clean["player1"].tolist() + df_clean["player2"].tolist()))
    
    final_stats = {}
    for p, stats in player_stats.items():
        win_rate = (stats["wins"] / stats["matches"]) * 100 if stats["matches"] > 0 else 0
        final_stats[p] = {"matches": stats["matches"], "win_rate": round(win_rate, 1)}
    
    # Convert defaultdicts to regular dicts
    clean_surface_form = {p: {s: list(h) for s, h in surfaces.items()} 
                          for p, surfaces in surface_form.items()}

    print(f"✓ Total unique players: {len(all_players)}\n")

    artifacts = {
        "model_elo": model_elo,
        "ai_model": ensemble_model, 
        "h2h_tracker": dict(h2h_tracker),
        "surface_form": clean_surface_form,
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
        print("✅ SUCCESS! The Ultimate AI is ready.")