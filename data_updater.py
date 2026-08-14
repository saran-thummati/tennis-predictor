import io
import joblib
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb

print("1. Cloning ATP match data directly via Git...")
# By cloning the repository directly, we completely bypass GitHub's web-scraping WAF blockers
if not os.path.exists("tennis_atp"):
    print(" -> Downloading repository...")
    os.system("git clone https://github.com/JeffSackmann/tennis_atp.git")
    time.sleep(2) # Give the OS a second to finish writing the files to disk

years = range(2018, 2027)
frames = []

for year in years:
    file_path = f"tennis_atp/atp_matches_{year}.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, low_memory=False)
            frames.append(df)
            print(f" -> Successfully loaded {year}")
        except Exception as e:
            print(f" -> Failed to read {year}: {e}")
    else:
        print(f" -> Skipped {year} (File not found in repo)")

if not frames:
    print("❌ Failed to load data.")
    exit(1)

df = pd.concat(frames).sort_values("tourney_date").reset_index(drop=True)
df = df[df["score"].notna()]
df = df[~df["score"].str.contains("W/O|RET|DEF", na=False)]

print(f" -> Total valid matches loaded: {len(df)}")

# --- ELO RATING ENGINE ---
class EloModel:
    def __init__(self):
        self.ratings = {}
        self.surface_ratings = {}

    def get_rating(self, player, surface=None):
        if surface:
            return self.surface_ratings.get(surface, {}).get(player, 1500.0)
        return self.ratings.get(player, 1500.0)

    def update(self, p1, p2, p1_win, surface, tourney_level):
        k = 48 if tourney_level == "G" else 40 if tourney_level == "M" else 32
        
        r1, r2 = self.get_rating(p1), self.get_rating(p2)
        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        self.ratings[p1] = r1 + k * (p1_win - exp1)
        self.ratings[p2] = r2 + k * ((1 - p1_win) - (1 - exp1))

        if surface not in self.surface_ratings:
            self.surface_ratings[surface] = {}
            
        sr1, sr2 = self.get_rating(p1, surface), self.get_rating(p2, surface)
        exps = 1 / (1 + 10 ** ((sr2 - sr1) / 400))
        self.surface_ratings[surface][p1] = sr1 + k * (p1_win - exps)
        self.surface_ratings[surface][p2] = sr2 + k * ((1 - p1_win) - (1 - exps))

print("2. Engineering 10 Advanced Features (Bio, Form, Fatigue, Rest)...")
model_elo = EloModel()
h2h_tracker = {}
surface_form = {}
player_bio = {}
player_last_match = {}
player_recent_matches = {}
X, y = [], []

for _, row in df.iterrows():
    p1, p2 = row["winner_name"], row["loser_name"]
    surface, tourney_level = row["surface"], row["tourney_level"]
    
    # Date Parsing for Schedule Density & Fatigue
    try:
        match_date = datetime.strptime(str(int(row["tourney_date"])), "%Y%m%d")
    except Exception:
        match_date = datetime.now()

    # Calculate Fatigue (Matches in last 7 days) & Rest Days
    recent_p1 = [d for d in player_recent_matches.get(p1, []) if (match_date - d).days <= 7]
    fatigue_p1 = len(recent_p1)
    rest_p1 = min((match_date - player_last_match.get(p1, match_date - pd.Timedelta(days=30))).days, 30)

    recent_p2 = [d for d in player_recent_matches.get(p2, []) if (match_date - d).days <= 7]
    fatigue_p2 = len(recent_p2)
    rest_p2 = min((match_date - player_last_match.get(p2, match_date - pd.Timedelta(days=30))).days, 30)

    # Bio Features
    a1 = row.get("winner_age", 25.0) if not pd.isna(row.get("winner_age")) else 25.0
    a2 = row.get("loser_age", 25.0) if not pd.isna(row.get("loser_age")) else 25.0
    h1 = 1 if row.get("winner_hand") == "L" else 0
    h2 = 1 if row.get("loser_hand") == "L" else 0
    ht1 = row.get("winner_ht", 185.0) if not pd.isna(row.get("winner_ht")) else 185.0
    ht2 = row.get("loser_ht", 185.0) if not pd.isna(row.get("loser_ht")) else 185.0
    rank1 = row.get("winner_rank", 500.0) if not pd.isna(row.get("winner_rank")) else 500.0
    rank2 = row.get("loser_rank", 500.0) if not pd.isna(row.get("loser_rank")) else 500.0

    player_bio[p1] = {"age": a1, "hand": "L" if h1 else "R", "height": ht1, "rank": rank1, "fatigue": fatigue_p1, "rest": rest_p1}
    player_bio[p2] = {"age": a2, "hand": "L" if h2 else "R", "height": ht2, "rank": rank2, "fatigue": fatigue_p2, "rest": rest_p2}

    # Elo & H2H Lookups
    r1_b, r2_b = model_elo.get_rating(p1), model_elo.get_rating(p2)
    r1_s, r2_s = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
    h2h_adv = h2h_tracker.get(f"{p1}_vs_{p2}", 0) - h2h_tracker.get(f"{p2}_vs_{p1}", 0)

    f1 = np.mean(surface_form.get(p1, {}).get(surface, [0.5])) * 100
    f2 = np.mean(surface_form.get(p2, {}).get(surface, [0.5])) * 100
    form_adv = ((f1 / 100) - (f2 / 100)) * 3.0

    # P1 Win Perspective (10 features)
    X.append([r1_b - r2_b, r1_s - r2_s, h2h_adv, form_adv, a1 - a2, h1 - h2, ht1 - ht2, rank2 - rank1, fatigue_p1 - fatigue_p2, rest_p1 - rest_p2])
    y.append(1)

    # P2 Win Perspective (Inverted)
    X.append([r2_b - r1_b, r2_s - r1_s, -h2h_adv, -form_adv, a2 - a1, h2 - h1, ht2 - ht1, rank1 - rank2, fatigue_p2 - fatigue_p1, rest_p2 - rest_p1])
    y.append(0)

    # Update Trackers
    model_elo.update(p1, p2, 1, surface, tourney_level)
    h2h_tracker[f"{p1}_vs_{p2}"] = h2h_tracker.get(f"{p1}_vs_{p2}", 0) + 1

    player_last_match[p1], player_last_match[p2] = match_date, match_date
    recent_p1.append(match_date)
    recent_p2.append(match_date)
    player_recent_matches[p1], player_recent_matches[p2] = recent_p1, recent_p2

    if p1 not in surface_form: surface_form[p1] = {}
    if p2 not in surface_form: surface_form[p2] = {}
    if surface not in surface_form[p1]: surface_form[p1][surface] = []
    if surface not in surface_form[p2]: surface_form[p2][surface] = []

    surface_form[p1][surface].append(1)
    surface_form[p2][surface].append(0)
    surface_form[p1][surface] = surface_form[p1][surface][-5:]
    surface_form[p2][surface] = surface_form[p2][surface][-5:]

print("3. Training Stacking Ensemble Model...")
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multi-model Ensemble Pipeline
estimators = [
    ('xgb', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=42)),
    ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, verbose=-1, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
]

ensemble_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=3
)

ensemble_model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = ensemble_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f" -> 🏆 Ensemble Real-World Accuracy: {acc * 100:.2f}%\n")

print("4. Saving Model Artifacts...")
all_players = sorted(list(player_bio.keys()))

joblib.dump({
    "all_players": all_players,
    "model_elo": model_elo,
    "ai_model": ensemble_model,
    "h2h_tracker": h2h_tracker,
    "surface_form": surface_form,
    "player_bio": player_bio
}, "tennis_model_artifacts.pkl")

print("✅ Automated Update Complete!")
