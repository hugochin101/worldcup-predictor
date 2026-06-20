import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

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
model.fit(X, y)

# Save everything the app will need: the model, the label encoder, and the processed data
joblib.dump(model, 'src/model.pkl')
joblib.dump(le, 'src/label_encoder.pkl')
model_df.to_csv('data/processed/model_ready_data.csv', index=False)

print("Saved model.pkl, label_encoder.pkl, and model_ready_data.csv")
print(f"Trained on {len(model_df)} matches")