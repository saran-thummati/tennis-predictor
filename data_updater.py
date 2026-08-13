import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import requests
import io

print("1. Downloading latest ATP match data via Requests...")
years = range(2015, 2027)
frames = []
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for year in years:
    url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text), low_memory=False)
            frames.append(df)
            print(f" -> Downloaded {year}")
    except Exception:
        pass

if len(frames) == 0:
    print("❌ Failed to download data.")
    exit()

df = pd.concat(frames).sort_values("tourney_date").reset_index(drop=True)
df = df[df["score"].notna()]
df = df[~df["score"].str.contains("W/O|RET|DEF", na=False)]

print("2. Building Advanced AI Features and Dynamic Elo...")
class EloModel:
    def __init__(self):
        self.ratings = {}
        self.surface_ratings = {}
        
    def get_rating(self, player, surface=None):
        if surface: return self.surface_ratings.get(surface, {}).get(player, 1500)
        return self.ratings.get(player, 1500)
        
    # NEW: The 'k' multiplier now changes based on tournament importance
    def update(self, p1, p2, p1_win, surface, tourney_level):
        if tourney_level == 'G': k = 48    # Grand Slams matter most
        elif tourney_level == 'M': k = 40  # Masters 1000 matter a lot
        else: k = 32                       # Standard ATP tour matches
        
        r1, r2 = self.get_rating(p1), self.get_rating(p2)
        exp1 = 1 / (1 + 10**((r2 - r1) / 400))
        self.ratings[p1] = r1 + k * (p1_win - exp1)
        self.ratings[p2] = r2 + k * ((1 - p1_win) - (1 - exp1))
        
        if surface not in self.surface_ratings: self.surface_ratings[surface] = {}
        sr1, sr2 = self.get_rating(p1, surface), self.get_rating(p2, surface)
        exps = 1 / (1 + 10**((sr2 - sr1) / 400))
        self.surface_ratings[surface][p1] = sr1 + k * (p1_win - exps)
        self.surface_ratings[surface][p2] = sr2 + k * ((1 - p1_win) - (1 - exps))

model_elo = EloModel()
h2h_tracker = {}
surface_form = {}
player_bio = {}
X, y = [] , []

for _, row in df.iterrows():
    p1, p2 = row["winner_name"], row["loser_name"]
    surface, tourney_level = row["surface"], row["tourney_level"]
    
    # Extract Bio, Height, and Rank safely (defaulting if missing)
    a1 = row.get("winner_age", 25.0) if not pd.isna(row.get("winner_age")) else 25.0
    a2 = row.get("loser_age", 25.0) if not pd.isna(row.get("loser_age")) else 25.0
    h1 = 1 if row.get("winner_hand") == 'L' else 0
    h2 = 1 if row.get("loser_hand") == 'L' else 0
    ht1 = row.get("winner_ht", 185.0) if not pd.isna(row.get("winner_ht")) else 185.0
    ht2 = row.get("loser_ht", 185.0) if not pd.isna(row.get("loser_ht")) else 185.0
    rank1 = row.get("winner_rank", 500.0) if not pd.isna(row.get("winner_rank")) else 500.0
    rank2 = row.get("loser_rank", 500.0) if not pd.isna(row.get("loser_rank")) else 500.0
    
    # Store latest stats so our Streamlit app can look them up later
    player_bio[p1] = {"age": a1, "hand": "L" if h1 else "R", "height": ht1, "rank": rank1}
    player_bio[p2] = {"age": a2, "hand": "L" if h2 else "R", "height": ht2, "rank": rank2}

    # Gather AI Features
    r1_b, r2_b = model_elo.get_rating(p1), model_elo.get_rating(p2)
    r1_s, r2_s = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
    h2h_adv = h2h_tracker.get(f"{p1}_vs_{p2}", 0) - h2h_tracker.get(f"{p2}_vs_{p1}", 0)
    
    f1 = np.mean(surface_form.get(p1, {}).get(surface, [0.5])) * 100
    f2 = np.mean(surface_form.get(p2, {}).get(surface, [0.5])) * 100
    form_adv = ((f1 / 100) - (f2 / 100)) * 3.0
    
    # Append 8 features to training data (Player 1 winning)
    # Notice we added Height Advantage (ht1 - ht2) and Rank Advantage (rank2 - rank1)
    X.append([r1_b - r2_b, r1_s - r2_s, h2h_adv, form_adv, a1 - a2, h1 - h2, ht1 - ht2, rank2 - rank1])
    y.append(1)
    
    # Append the inverted perspective (Player 2 winning)
    X.append([r2_b - r1_b, r2_s - r1_s, -h2h_adv, -form_adv, a2 - a1, h2 - h1, ht2 - ht1, rank1 - rank2])
    y.append(0)

    # Update Trackers (passing the tourney_level to Elo now!)
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

print("3. Training the Upgraded AI Model...")
# NEW: Much more powerful AI configuration
ai_model = RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_split=5, random_state=42)
ai_model.fit(X, y)

print("4. Saving Master Artifact File...")
all_players = sorted(list(player_bio.keys()))
joblib.dump({
    "all_players": all_players,
    "model_elo": model_elo,
    "ai_model": ai_model,
    "h2h_tracker": h2h_tracker,
    "surface_form": surface_form,
    "player_bio": player_bio
}, "tennis_model_artifacts.pkl")

print("✅ Update Complete!")
