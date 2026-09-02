import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("1. Loading ATP Match Records...")

if not os.path.exists("atp_tennis.csv"):
    print("❌ CRITICAL: 'atp_tennis.csv' not found. Please ensure it is in the Tennis_app folder.")
    exit(1)

df = pd.read_csv("atp_tennis.csv", low_memory=False, encoding="latin1")
df.columns = [c.strip() for c in df.columns]
df = df[df["Winner"].notna() & df["Player_1"].notna() & df["Player_2"].notna()].copy()

df["Player_1"] = df["Player_1"].astype(str).str.strip()
df["Player_2"] = df["Player_2"].astype(str).str.strip()
df["Winner"] = df["Winner"].astype(str).str.strip()
df["winner_name"] = df["Winner"]
df["loser_name"] = np.where(df["Winner"] == df["Player_1"], df["Player_2"], df["Player_1"])

df["Rank_1"] = pd.to_numeric(df.get("Rank_1", 500), errors="coerce").fillna(500.0)
df["Rank_2"] = pd.to_numeric(df.get("Rank_2", 500), errors="coerce").fillna(500.0)
df["winner_rank"] = np.where(df["Winner"] == df["Player_1"], df["Rank_1"], df["Rank_2"])
df["loser_rank"] = np.where(df["Winner"] == df["Player_1"], df["Rank_2"], df["Rank_1"])

df["tourney_date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("tourney_date").reset_index(drop=True)
df["surface"] = df.get("Surface", "Hard").fillna("Hard")

print("2. Calculating Elo Ratings & Head-to-Head...")

class TennisElo:
    def __init__(self):
        self.ratings = {}
        self.surface_ratings = {}

    def get_rating(self, player, surface=None):
        if surface:
            return self.surface_ratings.get(surface, {}).get(player, 1500.0)
        return self.ratings.get(player, 1500.0)

    def update(self, winner, loser, surface):
        k = 32
        r_w, r_l = self.get_rating(winner), self.get_rating(loser)
        exp_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
        self.ratings[winner] = r_w + k * (1.0 - exp_w)
        self.ratings[loser] = r_l + k * (0.0 - (1.0 - exp_w))

        if surface not in self.surface_ratings:
            self.surface_ratings[surface] = {}
        sr_w, sr_l = self.get_rating(winner, surface), self.get_rating(loser, surface)
        exp_sw = 1.0 / (1.0 + 10.0 ** ((sr_l - sr_w) / 400.0))
        self.surface_ratings[surface][winner] = sr_w + k * (1.0 - exp_sw)
        self.surface_ratings[surface][loser] = sr_l + k * (0.0 - (1.0 - exp_sw))

model_elo = TennisElo()
h2h_tracker = {}
player_bio = {}
X, y = [], []

for _, row in df.iterrows():
    p1, p2 = row["winner_name"], row["loser_name"]
    surface = row["surface"]
    rank1, rank2 = row["winner_rank"], row["loser_rank"]
    player_bio[p1], player_bio[p2] = {"rank": rank1}, {"rank": rank2}

    r1_b, r2_b = model_elo.get_rating(p1), model_elo.get_rating(p2)
    r1_s, r2_s = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
    h2h_diff = h2h_tracker.get(f"{p1}_vs_{p2}", 0) - h2h_tracker.get(f"{p2}_vs_{p1}", 0)
    log_rank_adv = np.log(rank2 + 1) - np.log(rank1 + 1)

    X.append([r1_b - r2_b, r1_s - r2_s, h2h_diff, log_rank_adv])
    y.append(1)
    X.append([r2_b - r1_b, r2_s - r1_s, -h2h_diff, -log_rank_adv])
    y.append(0)

    model_elo.update(p1, p2, surface)
    h2h_tracker[f"{p1}_vs_{p2}"] = h2h_tracker.get(f"{p1}_vs_{p2}", 0) + 1

print("3. Training Model...")
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.15, shuffle=False)

clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test))
print(f"🏆 Model Accuracy: {acc * 100:.2f}%")

joblib.dump({
    "all_players": sorted(list(player_bio.keys())),
    "model_elo": model_elo,
    "ai_model": clf,
    "h2h_tracker": h2h_tracker,
    "player_bio": player_bio
}, "tennis_model_artifacts.pkl")

print("✅ Saved cleanly to tennis_model_artifacts.pkl!")
