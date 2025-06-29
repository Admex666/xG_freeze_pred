#%% Imports, function
import streamlit as st
import numpy as np
import joblib
import pandas as pd
from shapely.geometry import Point, Polygon, LineString

def generate_features(x, y, keeper_x, keeper_y):
    shooter_point = (x, y)
    left_post = (120, 36)
    right_post = (120, 44)
    goal_center = (120, 40)
    # 1. Shot angle
    def shot_angle(shooter, left, right):
        a = np.linalg.norm(np.array(shooter) - np.array(left))
        b = np.linalg.norm(np.array(shooter) - np.array(right))
        c = np.linalg.norm(np.array(left) - np.array(right))
        if a > 0 and b > 0:
            angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
            return np.degrees(angle)
        return 0.0

    # 2. Distance to goal
    distance_to_goal = np.linalg.norm(np.array(shooter_point) - np.array(goal_center))

    # 10. Keeper lateral offset
    keeper_point = Point(keeper_x, keeper_y)
    shot_line = LineString([shooter_point, goal_center])
    keeper_lateral_offset = shot_line.distance(keeper_point)

    # 11. Dummy vars
    feature_dict = {
        "shot_angle": shot_angle(shooter_point, left_post, right_post),
        "distance_to_goal": distance_to_goal,
        "keeper_lateral_offset": keeper_lateral_offset
    }

    return pd.DataFrame([feature_dict])

#%% Load trained model
model = joblib.load("xgb_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Title
st.title("xG Predictor")

st.markdown("Add meg az ismert paramétereket egy lövéshez, és a modell becsli az xG-t.")

# --- Input mezők ---
x = st.slider("Lövő X koordináta", 0.0, 120.0, 102.0)
y = st.slider("Lövő Y koordináta", 0.0, 80.0, 40.0)
keeper_x = st.slider("Kapus X koordináta", 0.0, 120.0, 116.0)
keeper_y = st.slider("Kapus Y koordináta", 0.0, 80.0, 40.0)
num_blockers = st.slider("Blokkoló védők száma", 0, 10, 2)
num_players_in_box = st.slider("Boxban lévő játékosok száma", 0, 20, 6)
num_defenders_in_box = st.slider("Boxban lévő védők száma", 0, 10, 3)
close_defenders = st.slider("Közeli védők (1.5m-en belül)", 0, 5, 1)

# --- Kalkulált változók
shot_angle, distance_to_goal, keeper_lateral_offset = generate_features(x, y, keeper_x, keeper_y).iloc[0]

# --- Dummy encoding ---
row = {
    'x': x,
    'y': y,
    'shot_angle': shot_angle,
    'distance_to_goal': distance_to_goal,
    'num_blockers': num_blockers,
    'num_players_in_box': num_players_in_box,
    'num_defenders_in_box': num_defenders_in_box,
    'keeper_x': keeper_x,
    'keeper_y': keeper_y,
    'close_defenders': close_defenders,
    'keeper_lateral_offset': keeper_lateral_offset,
}

# DataFrame
input_df = pd.DataFrame([row])
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# --- Predikció ---
if st.button("Számítsd ki az xG-t"):
    xg_pred = model.predict(input_df)[0]
    st.success(f"Prediktált xG érték: **{xg_pred:.3f}**")
