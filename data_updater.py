import io
import joblib
import numpy as np
import pandas as pd
import requests
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

print("1. Downloading Data via Raw CSV URLs...")

# Disguise the cloud robot as a real human using Google Chrome to avoid firewalls
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

years = range(2015, 2027)
frames = []

for year in years:
    # We will try both 'master' and 'main' branches just in case GitHub redirects it
    urls = [
        f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
        f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/main/atp_matches_{year}.csv"
    ]
    
    success = False
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text), low_memory=False)
                frames.append(df)
                print(f" -> Successfully loaded {year}")
                success = True
                break
        except Exception:
            pass
            
    if not success:
        print(f" -> Skipped {year} (Data might not exist yet)")
        
    # A 1-second delay prevents GitHub from rate-limiting us for downloading too fast
    time.sleep(1) 

if not frames:
    print("❌ Failed to load any data.")
    exit(1)

# Sorting by date is CRITICAL for Time-Series Cross Validation
df = pd.concat(frames).sort_values("tourney_date").reset_index(drop=True)
df = df[df["score"].notna()]
df = df[~df["score"].str.contains("W/O|RET|DEF", na=False)]

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
    p1, p2 = row["winner_name"], row["loser_name"]
    surface, tourney_level = row["surface"], row["tourney_level"]
    
    a1 = row.get("winner_age", 25.0) if not pd.isna(row.get("winner_age")) else 25.0
    a2 = row.get("loser_age", 25.0) if not pd.isna(row.get("loser_age")) else 25.0
    h1 = 1 if row.get("winner_hand") == "L" else 0
    h2 = 1 if row.get("loser_hand") == "L" else 0
    ht1 = row.get("winner_ht", 185.0) if not pd.isna(row.get("winner_ht")) else 185.0
    ht2 = row.get("loser_ht", 185.0) if not pd.isna(row.get("loser_ht")) else 185.0
    rank1 = row.get("winner_rank", 500.0) if not pd.isna(row.get("winner_rank")) else 500.0
    rank2 = row.get("loser_rank", 500.0) if not pd.isna(row.get("loser_rank")) else 500.0

    player_bio[p1] = {"age": a1, "hand": "L" if h1 else "R", "height": ht1, "rank": rank1}
    player_bio[p2] = {"age": a2, "hand": "L" if h2 else "R", "height": ht2, "rank": rank2}

    r1_b, r2_b = model_elo.get_rating(p1), model_elo.get_rating(p2)
    r1_s, r2_s = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
    h2h_adv = h2h_tracker.get(f"{p1}_vs_{p2}", 0) - h2h_tracker.get(f"{p2}_vs_{p1}", 0)

    f1 = np.mean(surface_form.get(p1, {}).get(surface, [0.5])) * 100
    f2 = np.mean(surface_form.get(p2, {}).get(surface, [0.5])) * 100
    form_adv = ((f1 / 100) - (f2 / 100)) * 3.0

    X.append([r1_b - r2_b, r1_s - r2_s, h2h_adv, form_adv, a1 - a2, h1 - h2, ht1 - ht2, rank2 - rank1])
    y.append(1)

    X.append([r2_b - r1_b, r2_s - r1_s, -h2h_adv, -form_adv, a2 - a1, h2 - h1, ht2 - ht1, rank1 - rank2])
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

print("✅ Automated Update Complete!")
