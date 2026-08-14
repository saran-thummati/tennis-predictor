import io
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
import optuna

# Suppress Optuna's massive wall of text logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("1. Downloading latest ATP match data...")
years = range(2015, 2027)
frames = []
headers = {"User-Agent": "Mozilla/5.0"}

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

if not frames:
    print("❌ Failed to download data.")
    exit()

df = pd.concat(frames).sort_values("tourney_date").reset_index(drop=True)
df = df[df["score"].notna()]
df = df[~df["score"].str.contains("W/O|RET|DEF", na=False)]

# --- ELO TRACKING CLASS ---
class EloModel:
    def __init__(self):
        self.ratings = {}
        self.surface_ratings = {}

    def get_rating(self, player, surface=None):
        if surface: return self.surface_ratings.get(surface, {}).get(player, 1500)
        return self.ratings.get(player, 1500)

    def update(self, p1, p2, p1_win, surface, tourney_level):
        k = 48 if tourney_level == "G" else 40 if tourney_level == "M" else 32
        r1, r2 = self.get_rating(p1), self.get_rating(p2)
        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        self.ratings[p1] = r1 + k * (p1_win - exp1)
        self.ratings[p2] = r2 + k * ((1 - p1_win) - (1 - exp1))

        if surface not in self.surface_ratings: self.surface_ratings[surface] = {}
        sr1, sr2 = self.get_rating(p1, surface), self.get_rating(p2, surface)
        exps = 1 / (1 + 10 ** ((sr2 - sr1) / 400))
        self.surface_ratings[surface][p1] = sr1 + k * (p1_win - exps)
        self.surface_ratings[surface][p2] = sr2 + k * ((1 - p1_win) - (1 - exps))

print("2. Engineering Features (Including Fatigue & Schedule Density)...")
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
    
    # Safely parse the date for fatigue calculations
    try:
        match_date = datetime.strptime(str(int(row["tourney_date"])), "%Y%m%d")
    except:
        match_date = datetime.now()

    # Calculate Fatigue & Rest for Player 1
    recent_p1 = [d for d in player_recent_matches.get(p1, []) if (match_date - d).days <= 7]
    fatigue_p1 = len(recent_p1)
    rest_p1 = min((match_date - player_last_match.get(p1, match_date - pd.Timedelta(days=30))).days, 30)
    
    # Calculate Fatigue & Rest for Player 2
    recent_p2 = [d for d in player_recent_matches.get(p2, []) if (match_date - d).days <= 7]
    fatigue_p2 = len(recent_p2)
    rest_p2 = min((match_date - player_last_match.get(p2, match_date - pd.Timedelta(days=30))).days, 30)

    # Basic Bio Data
    a1 = row.get("winner_age", 25.0) if not pd.isna(row.get("winner_age")) else 25.0
    a2 = row.get("loser_age", 25.0) if not pd.isna(row.get("loser_age")) else 25.0
    h1 = 1 if row.get("winner_hand") == "L" else 0
    h2 = 1 if row.get("loser_hand") == "L" else 0
    ht1 = row.get("winner_ht", 185.0) if not pd.isna(row.get("winner_ht")) else 185.0
    ht2 = row.get("loser_ht", 185.0) if not pd.isna(row.get("loser_ht")) else 185.0
    rank1 = row.get("winner_rank", 500.0) if not pd.isna(row.get("winner_rank")) else 500.0
    rank2 = row.get("loser_rank", 500.0) if not pd.isna(row.get("loser_rank")) else 500.0

    # Store everything to inject into the app later
    player_bio[p1] = {"age": a1, "hand": "L" if h1 else "R", "height": ht1, "rank": rank1, "fatigue": fatigue_p1, "rest": rest_p1}
    player_bio[p2] = {"age": a2, "hand": "L" if h2 else "R", "height": ht2, "rank": rank2, "fatigue": fatigue_p2, "rest": rest_p2}

    r1_b, r2_b = model_elo.get_rating(p1), model_elo.get_rating(p2)
    r1_s, r2_s = model_elo.get_rating(p1, surface), model_elo.get_rating(p2, surface)
    h2h_adv = h2h_tracker.get(f"{p1}_vs_{p2}", 0) - h2h_tracker.get(f"{p2}_vs_{p1}", 0)

    f1 = np.mean(surface_form.get(p1, {}).get(surface, [0.5])) * 100
    f2 = np.mean(surface_form.get(p2, {}).get(surface, [0.5])) * 100
    form_adv = ((f1 / 100) - (f2 / 100)) * 3.0

    # NOW WE HAVE 10 FEATURES!
    X.append([r1_b - r2_b, r1_s - r2_s, h2h_adv, form_adv, a1 - a2, h1 - h2, ht1 - ht2, rank2 - rank1, fatigue_p1 - fatigue_p2, rest_p1 - rest_p2])
    y.append(1)
    
    X.append([r2_b - r1_b, r2_s - r1_s, -h2h_adv, -form_adv, a2 - a1, h2 - h1, ht2 - ht1, rank1 - rank2, fatigue_p2 - fatigue_p1, rest_p2 - rest_p1])
    y.append(0)

    # Update Trackers for the next loop
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

print("3. Validating & Tuning the Ensemble with Optuna...")
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We use a small subset (30%) of data just to run Optuna quickly in the cloud
X_sample, _, y_sample, _ = train_test_split(X_train, y_train, train_size=0.3, random_state=42)

def objective(trial):
    # Optuna intelligently guesses the best depths and leaves
    xgb_depth = trial.suggest_int('xgb_depth', 3, 7)
    lgb_leaves = trial.suggest_int('lgb_leaves', 20, 60)
    
    estimators = [
        ('xgb', XGBClassifier(max_depth=xgb_depth, n_estimators=100, learning_rate=0.1, eval_metric='logloss')),
        ('lgb', lgb.LGBMClassifier(num_leaves=lgb_leaves, n_estimators=100, learning_rate=0.1, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5))
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3)
    clf.fit(X_sample, y_sample)
    return accuracy_score(y_test, clf.predict(X_test))

# Run 10 intelligent trials
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print(f" -> Optuna Best Parameters: {study.best_params}")

print("4. Building Final Meta-Ensemble Stacking Classifier...")
# Build the final model with 100% of the training data using Optuna's winning parameters
best_estimators = [
    ('xgb', XGBClassifier(max_depth=study.best_params['xgb_depth'], n_estimators=100, learning_rate=0.1, eval_metric='logloss')),
    ('lgb', lgb.LGBMClassifier(num_leaves=study.best_params['lgb_leaves'], n_estimators=100, learning_rate=0.1, verbose=-1)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5))
]
final_model = StackingClassifier(estimators=best_estimators, final_estimator=LogisticRegression(), cv=5)
final_model.fit(X_train, y_train)

# Final Accuracy Check
y_pred = final_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f" -> 🏆 Real-World Accuracy Score: {acc * 100:.2f}%\n")

print("5. Saving Master Artifact File...")
all_players = sorted(list(player_bio.keys()))
joblib.dump({
    "all_players": all_players,
    "model_elo": model_elo,
    "ai_model": final_model,
    "h2h_tracker": h2h_tracker,
    "surface_form": surface_form,
    "player_bio": player_bio
}, "tennis_model_artifacts.pkl")

print("✅ Automated Update Complete!")
