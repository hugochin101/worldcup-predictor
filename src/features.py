import pandas as pd
import numpy as np

# --- Load and clean ---

df = pd.read_csv("data/raw/results.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Remove fixtures that haven't been played yet (no recorded score)
# These exist in the dataset as scheduled/upcoming matches and would
# corrupt the target variable if left in.
before_count = len(df)
df = df.dropna(subset=['home_score', 'away_score']).reset_index(drop=True)
after_count = len(df)
print(f"Removed {before_count - after_count} unplayed/future fixtures")
print(f"Remaining matches: {after_count}")

# --- Target variable: result from the HOME team's perspective ---
# (vectorized version - much faster than .apply() on large datasets)
df['result'] = np.select(
    [df['home_score'] > df['away_score'], df['home_score'] < df['away_score']],
    ['home_win', 'away_win'],
    default='draw'
)

print("\nResult distribution:")
print(df['result'].value_counts())
print("\nAs percentages:")
print(df['result'].value_counts(normalize=True))

# --- Build rolling form (last 5 matches, per team) ---

home_games = df[['date', 'home_team', 'result']].copy()
home_games['team'] = home_games['home_team']
home_games['points'] = home_games['result'].map({'home_win': 3, 'draw': 1, 'away_win': 0})

away_games = df[['date', 'away_team', 'result']].copy()
away_games['team'] = away_games['away_team']
away_games['points'] = away_games['result'].map({'away_win': 3, 'draw': 1, 'home_win': 0})

team_games = pd.concat([
    home_games[['date', 'team', 'points']],
    away_games[['date', 'team', 'points']]
])
team_games = team_games.sort_values(['team', 'date']).reset_index(drop=True)

# shift(1) excludes the current match itself - only past results count
team_games['recent_form'] = (
    team_games.groupby('team')['points']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
)

print("\nSample of rolling form data:")
print(team_games.head(10))
print("\nTotal team-match rows:", len(team_games))

# --- Build head-to-head win rate ---

def matchup_key(row):
    teams = sorted([row['home_team'], row['away_team']])
    return f"{teams[0]}_{teams[1]}"

df['matchup'] = df.apply(matchup_key, axis=1)
df['home_team_won'] = (df['result'] == 'home_win').astype(int)

def calc_h2h_win_rate(group):
    return group['home_team_won'].expanding().mean().shift(1)

df = df.sort_values('date').reset_index(drop=True)
df['h2h_home_win_rate'] = (
    df.groupby(['matchup', 'home_team'], group_keys=False)
    .apply(calc_h2h_win_rate)
)

print("\nSample head-to-head data:")
sample_cols = ['date', 'home_team', 'away_team', 'result', 'h2h_home_win_rate']
print(df[df['matchup'] == df['matchup'].mode()[0]][sample_cols].head(10))

# --- Merge recent_form back onto the main match table ---

form_lookup = team_games[['date', 'team', 'recent_form']].copy()

df = df.merge(
    form_lookup.rename(columns={'team': 'home_team', 'recent_form': 'home_recent_form'}),
    on=['date', 'home_team'],
    how='left'
)

df = df.merge(
    form_lookup.rename(columns={'team': 'away_team', 'recent_form': 'away_recent_form'}),
    on=['date', 'away_team'],
    how='left'
)

print("\nFinal feature table preview:")
final_cols = ['date', 'home_team', 'away_team', 'result',
              'home_recent_form', 'away_recent_form', 'h2h_home_win_rate', 'neutral']
print(df[final_cols].tail(10))

print("\nFinal shape:", df.shape)
print("\nMissing values per feature column:")
print(df[['home_recent_form', 'away_recent_form', 'h2h_home_win_rate']].isna().sum())

# --- Save processed data for the next step (model training) ---
df.to_csv("data/processed/matches_with_features.csv", index=False)
print("\nSaved to data/processed/matches_with_features.csv")