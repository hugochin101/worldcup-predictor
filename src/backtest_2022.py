import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report

# --- Load processed data ---
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

# --- Train on everything BEFORE 2022 ---
train_df = model_df[model_df['date'] < '2022-01-01']

# --- Test ONLY on the actual 2022 World Cup matches ---
test_df = model_df[
    (model_df['tournament'] == 'FIFA World Cup') &
    (model_df['date'] >= '2022-01-01') &
    (model_df['date'] < '2023-01-01')
]

print(f"Training matches (pre-2022): {len(train_df)}")
print(f"2022 World Cup test matches: {len(test_df)}")

X_train = train_df[feature_cols]
y_train = train_df['result_encoded']
X_test = test_df[feature_cols]
y_test = test_df['result_encoded']

# --- Train the best model so far ---
model = HistGradientBoostingClassifier(max_iter=300, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"\n--- 2022 World Cup Backtest Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Log Loss: {log_loss(y_test, y_pred_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# --- Match-by-match breakdown ---
results_df = test_df[['date', 'home_team', 'away_team', 'result']].copy()
results_df['predicted'] = le.inverse_transform(y_pred)
results_df['correct'] = results_df['result'] == results_df['predicted']

print("\nMatch-by-match predictions:")
print(results_df.to_string(index=False))

print(f"\nOverall: {results_df['correct'].sum()} / {len(results_df)} correct")