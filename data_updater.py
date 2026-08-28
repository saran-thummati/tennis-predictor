import io
import joblib
import numpy as np
import pandas as pd
import requests
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from elo_model import EloModel

print("1. Downloading Tennis ATP Data with Fallback URLs...")

# Disguise the cloud robot as a real human using Google Chrome to avoid firewalls
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/vnd.github.v3.raw"
}

years = range(2015, 2027)
frames = []
failed_years = []

# Multiple fallback URLs in case GitHub changes branch names
URL_TEMPLATES = [
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/main/atp_matches_{year}.csv",
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
]

for year in years:
    success = False
    
    for url_template in URL_TEMPLATES:
        url = url_template.format(year=year)
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            df = pd.read_csv(io.StringIO(response.text), low_memory=False)
            if len(df) > 0:
                frames.append(df)
                print(f" ✓ Successfully loaded {year} from {url_template.split('/')[5]} branch")
                success = True
                break
        except requests.exceptions.RequestException as e:
            print(f" → Trying alternative source for {year}...")
            continue
        except pd.errors.EmptyDataError:
            print(f" → No data in CSV for {year}")
            continue
    
    if not success:
        print(f" ✗ Skipped {year} (Data not available)")
        failed_years.append(year)
    
    # Rate limiting: 1-second delay prevents GitHub from rate-limiting us
    time.sleep(1)

if not frames:
    print("❌ CRITICAL: Failed to load ANY data from any source!")
    print(f"Failed years: {failed_years}")
    exit(1)

print(f"\n✓ Loaded data for {len(years) - len(failed_years)}/{len(years)} years")
print(f"✓ Total raw records: {sum(len(f) for f in frames)}")

# Sorting by date is CRITICAL for Time-Series Cross Validation
df = pd.concat(frames, ignore_index=True).sort_values("tourney_date").reset_index(drop=True)

# Clean data: remove walkovers, retirements, defaults, and records with missing scores
df = df[df["score"].notna()].copy()
df = df[~df["score"].str.contains("W/O|RET|DEF", na=False, regex=True)].copy()

print(f"✓ After cleaning: {len(df)} valid matches")

if len(df) == 0:
    print("❌ No valid matches after cleaning!")
    exit(1)

print("\n2. Engineering Surface-Specific Elo & Historical Features...")

# Use the imported EloModel from elo_model.py
model_elo = EloModel()
h2h_tracker = {}
surface_form = {}
player_bio = {}
X, y = [], []

for idx, row in df.iterrows():
    p1, p2 = row["winner_name"], row["loser_name"]
    surface = row.get("surface", "Hard")
    tourney_level = row.get("tourney_level", "A")
    
    # Safe extraction of player bio with defaults
    a1 = float(row.get("winner_age", 25)) if pd.notna(row.get("winner_age")) else 25.0
    a2 = float(row.get("loser_age", 25)) if pd.notna(row.get("loser_age")) else 25.0
    h1 = 1 if row.get("winner_hand") == "L" else 0
    h2 = 1 if row.get("loser_hand") == "L" else 0
    ht1 = float(row.get("winner_ht", 185)) if pd.notna(row.get("winner_ht")) else 185.0
    ht2 = float(row.get("loser_ht", 185)) if pd.notna(row.get("loser_ht")) else 185.0
    rank1 = float(row.get("winner_rank", 500)) if pd.notna(row.get("winner_rank")) else 500.0
    rank2 = float(row.get("loser_rank", 500)) if pd.notna(row.get("loser_rank")) else 500.0

    # Store player bio
    player_bio[p1] = {"age": a1, "hand": "L" if h1 else "R", "height": ht1, "rank": rank1}
    player_bio[p2] = {"age": a2, "hand": "L" if h2 else "R", "height": ht2, "rank": rank2}

    # Get Elo ratings
    r1_b = model_elo.get_rating(p1)
    r2_b = model_elo.get_rating(p2)
    r1_s = model_elo.get_rating(p1, surface)
    r2_s = model_elo.get_rating(p2, surface)
    
    # Head-to-head advantage
    h2h_adv = h2h_tracker.get(f"{p1}_vs_{p2}", 0) - h2h_tracker.get(f"{p2}_vs_{p1}", 0)

    # Surface-specific form
    f1 = np.mean(surface_form.get(p1, {}).get(surface, [0.5])) * 100
    f2 = np.mean(surface_form.get(p2, {}).get(surface, [0.5])) * 100
    form_adv = ((f1 / 100) - (f2 / 100)) * 3.0

    # Feature vector for player 1 (winner perspective)
    X.append([r1_b - r2_b, r1_s - r2_s, h2h_adv, form_adv, a1 - a2, h1 - h2, ht1 - ht2, rank2 - rank1])
    y.append(1)

    # Feature vector for player 2 (loser perspective - flipped)
    X.append([r2_b - r1_b, r2_s - r1_s, -h2h_adv, -form_adv, a2 - a1, h2 - h1, ht2 - ht1, rank1 - rank2])
    y.append(0)

    # Update Elo ratings
    try:
        dom_ratio = _parse_score(row.get("score", "6-0 6-0"))
    except:
        dom_ratio = 0.7  # Default to 70-30 if parsing fails
    
    model_elo.update(p1, p2, 1, dom_ratio, surface, k=32)
    
    # Track head-to-head
    h2h_tracker[f"{p1}_vs_{p2}"] = h2h_tracker.get(f"{p1}_vs_{p2}", 0) + 1

    # Track surface form
    if p1 not in surface_form:
        surface_form[p1] = {}
    if p2 not in surface_form:
        surface_form[p2] = {}
    if surface not in surface_form[p1]:
        surface_form[p1][surface] = []
    if surface not in surface_form[p2]:
        surface_form[p2][surface] = []

    surface_form[p1][surface].append(1)
    surface_form[p2][surface].append(0)
    surface_form[p1][surface] = surface_form[p1][surface][-5:]  # Keep last 5
    surface_form[p2][surface] = surface_form[p2][surface][-5:]  # Keep last 5

print(f"✓ Engineered features for {len(X) // 2} matches")

print("\n3. Time-Series Cross-Validation & LightGBM Training...")
X = np.array(X)
y = np.array(y)

print(f"✓ Training set size: {len(X)} samples")

# Time-series split: don't shuffle!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

lgb_model = lgb.LGBMClassifier(
    n_estimators=250, 
    learning_rate=0.05, 
    max_depth=6, 
    num_leaves=31,
    random_state=42, 
    verbose=-1
)

lgb_model.fit(X_train, y_train)

y_pred = lgb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✓ Real-World Accuracy Score: {acc * 100:.2f}%\n")

print("4. Saving Master Artifact File...")
all_players = sorted(list(player_bio.keys()))
print(f"✓ Total unique players: {len(all_players)}")

joblib.dump({
    "all_players": all_players,
    "model_elo": model_elo,
    "ai_model": lgb_model,
    "h2h_tracker": h2h_tracker,
    "surface_form": surface_form,
    "player_bio": player_bio
}, "tennis_model_artifacts.pkl")

print("✅ Automated Update Complete!")


def _parse_score(score_str):
    """Helper function to parse tennis score and extract dominance ratio"""
    try:
        sets = str(score_str).split()
        w_games = l_games = 0
        for s in sets:
            parts = s.replace("(", " ").replace(")", "").split("-")
            if len(parts) >= 2:
                try:
                    w_games += int(parts[0])
                    l_games += int(parts[1].split()[0])
                except:
                    pass
        total = w_games + l_games
        return max(0.51, min(0.99, w_games / total if total > 0 else 0.7))
    except:
        return 0.7