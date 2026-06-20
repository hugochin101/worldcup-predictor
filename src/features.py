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