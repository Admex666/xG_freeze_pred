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

# App header
st.title("⚽ xG: Expected Goal Calculator")
st.markdown("""
Adjust the shot conditions, and the AI model will calculate the expected goals value (xG).
xG shows what percentage of similar situations typically result in a goal.
""")

# Sidebar for better organization
st.sidebar.header("🎯 Shot Settings")

# --- Shooter Position ---
st.sidebar.subheader("📍 Shooter Position")
distance_from_goal = st.sidebar.slider(
    "Distance from goal (meters)", 
    0.0, 40.0, 18.0, 0.1, 
    help="How far is the shooter from the goal?"
)
# Convert to x coordinate (120 - distance)
x = 120 - distance_from_goal

field_side = st.sidebar.selectbox(
    "Player Position",
    ["Center", "Left Side", "Right Side"],
    help="From which side of the field is the shot taken?"
)

# Convert field side to y coordinate
if field_side == "Center":
    y = 40.0
elif field_side == "Left Side":
    y = st.sidebar.slider("Left Side Position", 0.0, 35.0, 25.0, 0.1)
elif field_side == "Right Side":
    y = st.sidebar.slider("Right Side Position", 45.0, 80.0, 55.0, 0.1)

# --- Goalkeeper Position ---
st.sidebar.subheader("🥅 Goalkeeper Position")
keeper_distance = st.sidebar.slider(
    "Goalkeeper distance from goal line (meters)", 
    0.0, 10.0, 4.0, 0.1,
    help="How far is the goalkeeper from the goal line?"
)
keeper_x = 120 - keeper_distance

keeper_side = st.sidebar.selectbox(
    "Goalkeeper Position",
    ["Goal Center", "Slightly Left", "Slightly Right", "Far Left", "Far Right"]
)

# Convert keeper side to y coordinate
keeper_positions = {
    "Goal Center": 40.0,
    "Slightly Left": 38.0,
    "Slightly Right": 42.0,
    "Far Left": 36.0,
    "Far Right": 44.0
}
keeper_y = keeper_positions[keeper_side]

# --- Defense ---
st.sidebar.subheader("🛡️ Defense")
num_blockers = st.sidebar.slider(
    "Number of blocking defenders", 
    0, 10, 2,
    help="How many defenders are between the shooter and the goal?"
)

num_players_in_box = st.sidebar.slider(
    "Players in penalty area", 
    0, 20, 6,
    help="Total number of players inside the penalty area?"
)

num_defenders_in_box = st.sidebar.slider(
    "Defenders in penalty area", 
    0, 10, 3,
    help="How many defenders are inside the penalty area?"
)

close_defenders = st.sidebar.slider(
    "Close defenders", 
    0, 5, 1,
    help="How many defenders are within 1.5 meters of the shooter?"
)
# Twitter
st.sidebar.markdown("---")
st.sidebar.markdown("Created by [@adamjakus99](https://x.com/adamjakus99)")

# --- Main content area ---
col1, col2 = st.columns([2, 1])

with col1:
    # Calculate features
    features = generate_features(x, y, keeper_x, keeper_y).iloc[0]
    shot_angle = features['shot_angle']
    distance_to_goal = features['distance_to_goal']
    keeper_lateral_offset = features['keeper_lateral_offset']

    # Display calculated metrics
    st.subheader("📊 Calculated Metrics")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            "Shot Angle", 
            f"{shot_angle:.1f}°",
            help="Larger angle = better scoring opportunity"
        )
    
    with metric_col2:
        st.metric(
            "Distance", 
            f"{distance_to_goal:.1f}m",
            help="Shorter distance = better scoring opportunity"
        )
    
    with metric_col3:
        st.metric(
            "Keeper Offset", 
            f"{keeper_lateral_offset:.1f}m",
            help="Greater offset = worse goalkeeper position"
        )

with col2:
    # Prediction button and result
    st.subheader("🎯 Result")
    
    if st.button("🔮 Calculate xG", type="primary"):
        # Prepare data for prediction
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
        
        # Create DataFrame
        input_df = pd.DataFrame([row])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        
        # Make prediction
        xg_pred = model.predict(input_df)[0]
        
        # Display result with color coding
        if xg_pred >= 0.5:
            st.success(f"### 🎯 {xg_pred:.3} xG")
            st.success("**Excellent scoring opportunity!**")
        elif xg_pred >= 0.2:
            st.warning(f"### ⚡ {xg_pred:.3} xG")
            st.warning("**Moderate scoring chance**")
        else:
            st.error(f"### 🔒 {xg_pred:.3} xG")
            st.error("**Low scoring chance**")

# --- Additional info ---
st.markdown("---")
st.markdown("### 📖 How to Interpret xG Values")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🟢 High xG (>0.5)**
    - Excellent scoring opportunity
    - Similar situations often result in goals
    - Shooter is in a good position
    """)

with col2:
    st.markdown("""
    **🟡 Medium xG (0.2-0.5)**
    - Average scoring chance
    - Situation depends on skill
    - Good execution required
    """)

with col3:
    st.markdown("""
    **🔴 Low xG (<0.2)**
    - Difficult scoring situation
    - Rarely results in a goal
    - Requires luck or brilliant shot
    """)

# Footer
st.markdown("---")
st.markdown("""
*The xG model was created using machine learning based on thousands of shot records.*  
📢 Follow me on [Twitter](https://x.com/adamjakus99) for updates!
""")