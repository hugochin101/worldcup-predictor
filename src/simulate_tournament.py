import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# --- Load data and train the model (same setup as before) ---
df = pd.read_csv("data/processed/matches_with_features.csv")
df['date'] = pd.to_datetime(df['date'])

feature_cols = [
    'home_recent_form', 'away_recent_form', 'h2h_home_win_rate', 'neutral',
    'form_diff', 'home_goal_diff_avg', 'away_goal_diff_avg', 'h2h_matches_played'
]
target_col = 'result'

model_df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

le = LabelEncoder()
model_df['result_encoded'] = le.fit_transform(model_df[target_col])

X = model_df[feature_cols]
y = model_df['result_encoded']

model = HistGradientBoostingClassifier(max_iter=300, max_depth=6, random_state=42)
model.fit(X, y)  # train on ALL data for the simulator (we're not evaluating accuracy here anymore)

class_order = list(le.classes_)  # e.g. ['away_win', 'draw', 'home_win']
print("Model trained. Class order:", class_order)

# --- Build a "current snapshot" of each team's latest stats ---
# For the simulator, we need each team's MOST RECENT form/goal-diff values
# (their latest known state), not historical per-match values.

def get_latest_team_stats(team_name, df):
    # Find this team's most recent match (as home or away) to get their latest rolling stats
    home_matches = df[df['home_team'] == team_name].sort_values('date')
    away_matches = df[df['away_team'] == team_name].sort_values('date')

    latest_form = None
    latest_goal_diff = None

    if len(home_matches) > 0:
        last_home = home_matches.iloc[-1]
        latest_form = last_home['home_recent_form']
        latest_goal_diff = last_home['home_goal_diff_avg']
    if len(away_matches) > 0:
        last_away = away_matches.iloc[-1]
        if latest_form is None or last_away['date'] > home_matches.iloc[-1]['date']:
            latest_form = last_away['away_recent_form']
            latest_goal_diff = last_away['away_goal_diff_avg']

    return latest_form, latest_goal_diff

print("\nTesting latest stats lookup for a few teams:")
for team in ['Brazil', 'Argentina', 'France', 'England']:
    form, gd = get_latest_team_stats(team, model_df)
    print(f"{team}: recent_form={form}, goal_diff_avg={gd}")

# --- Build a feature row for a hypothetical match between two teams ---

def build_match_features(home_team, away_team, df, neutral=True):
    home_form, home_gd = get_latest_team_stats(home_team, df)
    away_form, away_gd = get_latest_team_stats(away_team, df)

    # h2h stats: look up real historical record between these two, if it exists
    matchup = '_'.join(sorted([home_team, away_team]))
    h2h_matches = df[df['matchup'] == matchup]

    if len(h2h_matches) > 0:
        h2h_win_rate = (h2h_matches['result'] == 'home_win').mean() if home_team in h2h_matches['home_team'].values else 0.0
        h2h_count = len(h2h_matches)
    else:
        h2h_win_rate = 0.0  # no history = neutral assumption
        h2h_count = 0

    form_diff = (home_form or 0) - (away_form or 0)

    features = pd.DataFrame([{
        'home_recent_form': home_form or 0,
        'away_recent_form': away_form or 0,
        'h2h_home_win_rate': h2h_win_rate,
        'neutral': neutral,
        'form_diff': form_diff,
        'home_goal_diff_avg': home_gd or 0,
        'away_goal_diff_avg': away_gd or 0,
        'h2h_matches_played': h2h_count
    }])
    return features[feature_cols]

def simulate_match(home_team, away_team, df, model, class_order, knockout=False):
    """Returns the winner of a single simulated match."""
    features = build_match_features(home_team, away_team, df)
    probs = model.predict_proba(features)[0]  # [P(away_win), P(draw), P(home_win)] in class_order

    prob_dict = dict(zip(class_order, probs))

    if knockout:
        # No draws allowed - redistribute draw probability proportionally between win/loss
        p_home = prob_dict['home_win']
        p_away = prob_dict['away_win']
        p_draw = prob_dict['draw']
        total = p_home + p_away
        p_home_adj = p_home + p_draw * (p_home / total)
        p_away_adj = p_away + p_draw * (p_away / total)
        outcome = np.random.choice(['home', 'away'], p=[p_home_adj, p_away_adj])
    else:
        outcome = np.random.choice(
            ['away', 'draw', 'home'],
            p=[prob_dict['away_win'], prob_dict['draw'], prob_dict['home_win']]
        )

    if outcome == 'home':
        return home_team
    elif outcome == 'away':
        return away_team
    else:
        return 'draw'
    
# --- Quick test: check actual probabilities first, then simulate ---
test_features = build_match_features('Brazil', 'Argentina', model_df)
test_probs = model.predict_proba(test_features)[0]
print("\nBrazil (home) vs Argentina (away) - raw probabilities:")
for cls, prob in zip(class_order, test_probs):
    print(f"  {cls}: {prob:.3f}")

print("\nSimulating Brazil vs Argentina (knockout, 10 times):")
for i in range(10):
    winner = simulate_match('Brazil', 'Argentina', model_df, model, class_order, knockout=True)
    print(f"  Simulation {i+1}: {winner} wins")

# --- Full tournament simulation ---

def simulate_knockout_round(teams, df, model, class_order):
    """Takes a list of teams, returns winners of each matchup (pairs them in order)."""
    winners = []
    for i in range(0, len(teams), 2):
        home, away = teams[i], teams[i + 1]
        winner = simulate_match(home, away, df, model, class_order, knockout=True)
        winners.append(winner)
    return winners

def simulate_full_tournament(initial_bracket, df, model, class_order):
    """Simulates a full knockout tournament from a starting bracket of teams (must be power of 2)."""
    round_teams = initial_bracket
    while len(round_teams) > 1:
        round_teams = simulate_knockout_round(round_teams, df, model, class_order)
    return round_teams[0]  # the champion

# --- Example: 8-team knockout bracket ---
bracket = ['Brazil', 'Argentina', 'France', 'England',
           'Spain', 'Germany', 'Portugal', 'Netherlands']

print(f"\nSimulating {len(bracket)}-team tournament 1,000 times...")
print("Bracket:", bracket)

n_simulations = 1000
champion_counts = {team: 0 for team in bracket}

for sim in range(n_simulations):
    champion = simulate_full_tournament(bracket, model_df, model, class_order)
    champion_counts[champion] += 1

print("\n--- Tournament Win Probabilities ---")
results = pd.DataFrame([
    {'team': team, 'win_probability': count / n_simulations}
    for team, count in champion_counts.items()
]).sort_values('win_probability', ascending=False)

print(results.to_string(index=False))