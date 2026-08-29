import io
import joblib
import numpy as np
import pandas as pd
import requests
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

print("1. PIVOT: Downloading Live Data from Tennis-Data.co.uk (2015-2026)...")

years = range(2015, 2027)
frames = []

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

for year in years:
    # Hitting the betting industry standard database directly
    url = f"http://www.tennis-data.co.uk/{year}/{year}.csv"
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text), low_memory=False)
            
            # Translate Tennis-Data columns to our AI's expected format
            df = df.rename(columns={
                "Winner": "winner_name",
                "Loser": "loser_name",
                "Surface": "surface",
                "Date": "tourney_date",
                "WRank": "winner_rank",
                "LRank": "loser_rank",
                "Series": "tourney_level",
                "Comment": "score"
            })
            frames.append(df)
            print(f" -> ✅ Successfully loaded {year}")
        else:
            print(f" -> ⚠️ Skipped {year} (HTTP {response.status_code})")
    except Exception as e:
        print(f" -> ⚠️ Connection error for {year}: {e}")
        
    time.sleep(1)

if not frames:
    print("❌ CRITICAL: Failed to load any data.")
    exit(1)

df = pd.concat(frames, ignore_index=True)

# 1. Standardize Dates
df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%d/%m/%Y", errors="coerce")
df = df.sort_values("tourney_date").reset_index(drop=True)

# 2. Filter out Walkovers & Retirements
if "score" in df.columns:
    df = df[~df["score"].astype(str).str.contains("Walkover|Retired|Def", case=False, na=False)]

# 3. Standardize Ranks (Some odds sites use text like "N/A" for unranked players)
df["winner_rank"] = pd.to_numeric(df["winner_rank"], errors="coerce").fillna(500.0)
df["loser_rank"] = pd.to_numeric(df["loser_rank"], errors="coerce").fillna(500.0)

# 4. Map Tourney Levels for Elo Math
df["tourney_level"] = df["tourney_level"].map({"Grand Slam": "G", "Masters 1000": "M"}).fillna("A")

print(f" -> Total valid matches loaded: {len(df)}")

print("2. Engineering Surface-Specific Elo & Historical Features...")
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

model_elo = EloModel()
h2h_tracker = {}
surface_form = {}
player_bio = {}
X, y = [], []

for _, row in df.iterrows():
    p1 = row.get("winner_name")
    p2 = row.get("loser_name")
    surface = row.get("surface", "Hard")
    tourney_level = row.get("tourney_level", "A")
    
    if pd.isna(p1) or pd.isna(p2):
        continue
        
    rank1 = row.get("winner_rank", 500.0)
    rank2 = row.get("loser_rank", 500.0)

    # Tennis-Data doesn't track age/height, so we normalize to baseline defaults
    player_bio[p1] = {"age": 25.0, "hand": "R", "height": 185.0, "rank": rank1}
    player_bio[p2] = {"age": 25.0, "hand": "R", "height": 185.0, "rank": rank2}

    r1_b, r2_b = model_elo.get_rating(p1), model_elo.get_rating(p2)
    r1_s, r2_s = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
    h2h_adv = h2h_tracker.get(f"{p1}_vs_{p2}", 0) - h2h_tracker.get(f"{p2}_vs_{p1}", 0)

    f1 = np.mean(surface_form.get(p1, {}).get(surface, [0.5])) * 100
    f2 = np.mean(surface_form.get(p2, {}).get(surface, [0.5])) * 100
    form_adv = ((f1 / 100) - (f2 / 100)) * 3.0

    X.append([r1_b - r2_b, r1_s - r2_s, h2h_adv, form_adv, 0, 0, 0, rank2 - rank1])
    y.append(1)

    X.append([r2_b - r1_b, r2_s - r1_s, -h2h_adv, -form_adv, 0, 0, 0, rank1 - rank2])
    y.append(0)

    model_elo.update(p1, p2, 1, surface, tourney_level)
    h2h_tracker[f"{p1}_vs_{p2}"] = h2h_tracker.get(f"{p1}_vs_{p2}", 0) + 1

    if p1 not in surface_form: surface_form[p1] = {}
    if p2 not in surface_form: surface_form[p2] = {}
    if surface not in surface_form[p1]: surface_form[p1][surface] = []
    if surface not in surface_form[p2]: surface_form[p2][surface] = []

    surface_form[p1][surface].append(1)
    surface_form[p2][surface].append(0)
    surface_form[p1][surface] = surface_form[p1][surface][-5:]
    surface_form[p2][surface] = surface_form[p2][surface][-5:]

print("3. Time-Series Cross-Validation & LightGBM Training...")
X = np.array(X)
y = np.array(y)

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
print(f" -> 🏆 Real-World Accuracy Score: {acc * 100:.2f}%\n")

print("4. Saving Master Artifact File...")
all_players = sorted(list(player_bio.keys()))

joblib.dump({
    "all_players": all_players,
    "model_elo": model_elo,
    "ai_model": lgb_model,
    "h2h_tracker": h2h_tracker,
    "surface_form": surface_form,
    "player_bio": player_bio
}, "tennis_model_artifacts.pkl")

print("✅ Tennis-Data Build Complete!")
