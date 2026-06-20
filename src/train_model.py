import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- Load the processed feature table from yesterday ---
df = pd.read_csv("data/processed/matches_with_features.csv")
df['date'] = pd.to_datetime(df['date'])

# --- Select features and target ---
feature_cols = ['home_recent_form', 'away_recent_form', 'h2h_home_win_rate', 'neutral']
target_col = 'result'

# Drop rows with missing feature values (mostly early matches with no history yet)
model_df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

print(f"Rows before dropping NaNs: {len(df)}")
print(f"Rows after dropping NaNs: {len(model_df)}")

# --- Encode target as numbers (model needs numeric labels) ---
le = LabelEncoder()
model_df['result_encoded'] = le.fit_transform(model_df[target_col])
print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- Time-based train/test split ---
# Train on everything before 2018, test on 2018 onward.
# This simulates "would this model have worked on matches it never saw"
# rather than randomly mixing past and future together.
cutoff_date = '2018-01-01'
train_df = model_df[model_df['date'] < cutoff_date]
test_df = model_df[model_df['date'] >= cutoff_date]

print(f"\nTraining set: {len(train_df)} matches (before {cutoff_date})")
print(f"Test set: {len(test_df)} matches (from {cutoff_date} onward)")

X_train = train_df[feature_cols]
y_train = train_df['result_encoded']
X_test = test_df[feature_cols]
y_test = test_df['result_encoded']

# --- Train baseline Logistic Regression ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print("\n--- Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Log Loss: {log_loss(y_test, y_pred_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))