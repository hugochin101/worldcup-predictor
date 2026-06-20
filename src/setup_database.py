import sqlite3
import pandas as pd

# --- Connect to (or create) the database file ---
# This creates a single file: worldcup.db, right in your project folder
conn = sqlite3.connect("data/worldcup.db")
cursor = conn.cursor()

# --- Define the matches table schema ---
# DROP TABLE first so this script is safely re-runnable during development
cursor.execute("DROP TABLE IF EXISTS matches")

cursor.execute("""
CREATE TABLE matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    tournament TEXT,
    city TEXT,
    country TEXT,
    neutral INTEGER
)
""")

print("Created 'matches' table")

# --- Load the raw CSV and insert it into the database ---
df = pd.read_csv("data/raw/results.csv")

df.to_sql("matches", conn, if_exists="append", index=False)

print(f"Inserted {len(df)} rows into matches table")

# --- Verify with a simple SQL query ---
cursor.execute("SELECT COUNT(*) FROM matches")
count = cursor.fetchone()[0]
print(f"\nVerification - total rows in database: {count}")

cursor.execute("SELECT date, home_team, away_team, home_score, away_score FROM matches LIMIT 5")
print("\nSample rows:")
for row in cursor.fetchall():
    print(row)

conn.commit()
conn.close()
print("\nDatabase saved to data/worldcup.db")