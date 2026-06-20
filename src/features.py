import pandas as pd

# Load the cleaned data
df = pd.read_csv("data/raw/results.csv")

# Convert date column to actual datetime objects (currently just text)
df['date'] = pd.to_datetime(df['date'])

# Sort chronologically - CRITICAL for any "recent form" calculation later
# If this isn't sorted, rolling averages would be calculated in the wrong order
df = df.sort_values('date').reset_index(drop=True)

# Create the target variable: result from the HOME team's perspective
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'home_win'
    elif row['home_score'] < row['away_score']:
        return 'away_win'
    else:
        return 'draw'

df['result'] = df.apply(get_result, axis=1)

print("Result distribution:")
print(df['result'].value_counts())
print("\nAs percentages:")
print(df['result'].value_counts(normalize=True))

# --- Build rolling form ---

# First, we need points per match from EACH team's perspective.
# Right now the data is one row per match (home vs away).
# We'll create a "long" version: one row per TEAM per match.

home_games = df[['date', 'home_team', 'result']].copy()
home_games['team'] = home_games['home_team']
home_games['points'] = home_games['result'].map({'home_win': 3, 'draw': 1, 'away_win': 0})

away_games = df[['date', 'away_team', 'result']].copy()
away_games['team'] = away_games['away_team']
away_games['points'] = away_games['result'].map({'away_win': 3, 'draw': 1, 'home_win': 0})

# Combine into one long table: every team's points from every match they played
team_games = pd.concat([
    home_games[['date', 'team', 'points']],
    away_games[['date', 'team', 'points']]
])
team_games = team_games.sort_values(['team', 'date']).reset_index(drop=True)

# Rolling average of last 5 matches, PER TEAM
# shift(1) is critical: it excludes the current match itself, using only past results
team_games['recent_form'] = (
    team_games.groupby('team')['points']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
)

print("\nSample of rolling form data:")
print(team_games.head(10))
print("\nTotal team-match rows:", len(team_games))

# --- Build head-to-head win rate ---

# Create a consistent matchup identifier regardless of home/away order
def matchup_key(row):
    teams = sorted([row['home_team'], row['away_team']])
    return f"{teams[0]}_{teams[1]}"

df['matchup'] = df.apply(matchup_key, axis=1)

# For head-to-head, we need to know: did the CURRENT home team win this match?
# (1 if yes, 0 if no - draws and losses both count as "not a win" for this purpose)
df['home_team_won'] = (df['result'] == 'home_win').astype(int)

# Now calculate, for each matchup, a rolling win rate from the home team's perspective
# But there's a subtlety: "home team" changes who it refers to between matches!
# So we calculate it per (matchup, home_team) pair instead

def calc_h2h_win_rate(group):
    # Expanding mean = average of all PRIOR matches in this exact matchup
    # shift(1) excludes the current match itself
    return group['home_team_won'].expanding().mean().shift(1)

df = df.sort_values('date').reset_index(drop=True)
df['h2h_home_win_rate'] = (
    df.groupby(['matchup', 'home_team'], group_keys=False)
    .apply(calc_h2h_win_rate)
)

print("\nSample head-to-head data:")
sample_cols = ['date', 'home_team', 'away_team', 'result', 'h2h_home_win_rate']
print(df[df['matchup'] == df['matchup'].mode()[0]][sample_cols].head(10))