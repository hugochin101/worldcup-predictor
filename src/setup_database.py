import sqlite3
import pandas as pd

conn = sqlite3.connect("data/worldcup.db")
cursor = conn.cursor()

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

cursor.execute("DROP TABLE IF EXISTS predictions")
cursor.execute("""
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    predicted_home_win_prob REAL,
    predicted_draw_prob REAL,
    predicted_away_win_prob REAL
)
""")
print("Created 'predictions' table")

df = pd.read_csv("data/raw/results.csv")
df.to_sql("matches", conn, if_exists="append", index=False)
print(f"Inserted {len(df)} rows into matches table")

cursor.execute("SELECT COUNT(*) FROM matches")
print(f"\nVerification - total rows in matches table: {cursor.fetchone()[0]}")
cursor.execute("SELECT COUNT(*) FROM predictions")
print(f"Verification - total rows in predictions table: {cursor.fetchone()[0]}")

conn.commit()
conn.close()
print("\nDatabase saved to data/worldcup.db")