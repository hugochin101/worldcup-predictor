import pandas as pd

# Load the raw data
df = pd.read_csv("data/raw/results.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

print("\nDate range:", df['date'].min(), "to", df['date'].max())
print("\nUnique tournaments:", df['tournament'].nunique())

print("\nWorld Cup matches only:")
wc = df[df['tournament'] == 'FIFA World Cup']
print(wc.shape)