import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="World Cup Predictor", page_icon="⚽", layout="centered")

# --- Load model and data (cached so it only loads once, not on every interaction) ---
@st.cache_resource
def load_model():
    model = joblib.load("src/model.pkl")
    le = joblib.load("src/label_encoder.pkl")
    return model, le

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/model_ready_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

model, le = load_model()
df = load_data()
class_order = list(le.classes_)

feature_cols = [
    'home_recent_form', 'away_recent_form', 'h2h_home_win_rate', 'neutral',
    'form_diff', 'home_goal_diff_avg', 'away_goal_diff_avg', 'h2h_matches_played'
]

teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))

# --- Helper functions (same logic as simulate_tournament.py) ---
def get_latest_team_stats(team_name, df):
    home_matches = df[df['home_team'] == team_name].sort_values('date')
    away_matches = df[df['away_team'] == team_name].sort_values('date')

    latest_form, latest_goal_diff = None, None

    if len(home_matches) > 0:
        last_home = home_matches.iloc[-1]
        latest_form = last_home['home_recent_form']
        latest_goal_diff = last_home['home_goal_diff_avg']
    if len(away_matches) > 0:
        last_away = away_matches.iloc[-1]
        if latest_form is None or last_away['date'] > home_matches.iloc[-1]['date']:
            latest_form = last_away['away_recent_form']
            latest_goal_diff = last_away['away_goal_diff_avg']

    return latest_form, latest_goal_diff

def build_match_features(home_team, away_team, df, neutral=True):
    home_form, home_gd = get_latest_team_stats(home_team, df)
    away_form, away_gd = get_latest_team_stats(away_team, df)

    matchup = '_'.join(sorted([home_team, away_team]))
    h2h_matches = df[df['matchup'] == matchup]

    if len(h2h_matches) > 0:
        h2h_win_rate = (h2h_matches['result'] == 'home_win').mean() if home_team in h2h_matches['home_team'].values else 0.0
        h2h_count = len(h2h_matches)
    else:
        h2h_win_rate = 0.0
        h2h_count = 0

    form_diff = (home_form or 0) - (away_form or 0)

    features = pd.DataFrame([{
        'home_recent_form': home_form or 0,
        'away_recent_form': away_form or 0,
        'h2h_home_win_rate': h2h_win_rate,
        'neutral': neutral,
        'form_diff': form_diff,
        'home_goal_diff_avg': home_gd or 0,
        'away_goal_diff_avg': away_gd or 0,
        'h2h_matches_played': h2h_count
    }])
    return features[feature_cols]

# --- UI ---
st.title("⚽ World Cup Match Predictor")
st.write("Pick two national teams to predict the outcome, based on a model trained on 47,000+ international matches.")

col1, col2 = st.columns(2)
team_a = col1.selectbox("Home Team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
team_b = col2.selectbox("Away Team", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)

if st.button("Predict Outcome", type="primary"):
    if team_a == team_b:
        st.error("Please select two different teams.")
    else:
        features = build_match_features(team_a, team_b, df)
        probs = model.predict_proba(features)[0]
        prob_dict = dict(zip(class_order, probs))

        st.subheader("Prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{team_a} Win", f"{prob_dict['home_win']:.1%}")
        c2.metric("Draw", f"{prob_dict['draw']:.1%}")
        c3.metric(f"{team_b} Win", f"{prob_dict['away_win']:.1%}")

        chart_data = pd.DataFrame({
            'Outcome': [team_a, 'Draw', team_b],
            'Probability': [prob_dict['home_win'], prob_dict['draw'], prob_dict['away_win']]
        }).set_index('Outcome')
        st.bar_chart(chart_data)