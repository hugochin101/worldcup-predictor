import matplotlib
matplotlib.use('Agg')
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# --- Load data and train the best model (same as before) ---
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

train_df = model_df[model_df['date'] < '2018-01-01']
test_df = model_df[model_df['date'] >= '2018-01-01']

X_train = train_df[feature_cols]
y_train = train_df['result_encoded']
X_test = test_df[feature_cols]

model = HistGradientBoostingClassifier(max_iter=300, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# --- SHAP: explain the model's predictions ---
# Use a sample of test data for speed (SHAP can be slow on large datasets)
X_sample = X_test.sample(n=500, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample, check_additivity=False)

print("SHAP values computed successfully")
print("Shape:", shap_values.shape)

# --- Global summary plot: which features matter most overall ---
# Note: for multi-class, we look at one class at a time (let's start with home_win)
home_win_idx = list(le.classes_).index('home_win')

plt.figure()
shap.summary_plot(
    shap_values[:, :, home_win_idx],
    X_sample,
    show=False
)
plt.title("Feature impact on Home Win predictions")
plt.tight_layout()
plt.savefig("data/processed/shap_summary_home_win.png", dpi=150)
print("\nSaved global summary plot to data/processed/shap_summary_home_win.png")
plt.close()

# --- Local explanation: explain ONE specific match ---
match_idx = 0
print(f"\nExplaining match: {test_df.iloc[X_sample.index[match_idx] - X_test.index[0]][['home_team', 'away_team', 'result']].to_dict() if False else 'see waterfall plot'}")

plt.figure()
shap.plots.waterfall(shap_values[match_idx, :, home_win_idx], show=False)
plt.tight_layout()
plt.savefig("data/processed/shap_waterfall_example.png", dpi=150)
print("Saved single-match explanation to data/processed/shap_waterfall_example.png")
plt.close()