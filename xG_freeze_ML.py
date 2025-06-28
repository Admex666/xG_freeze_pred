#%% Import, open data
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch, Sbopen, VerticalPitch
import pandas as pd 
from shapely.geometry import Point, Polygon, LineString

# Open statsbomb event data
parser = Sbopen()
# All competitions
df_competitions = parser.competition()
# All games of Euro2024, Leverkusen and PSG
df_matches1 = parser.match(competition_id=55, season_id=282)
df_matches2 = parser.match(competition_id=9, season_id=281)
df_matches3 = parser.match(competition_id=7, season_id=235)

df_matches = pd.concat([df_matches1, df_matches2, df_matches3], ignore_index=True)

#%% Function: get info from freeze frame
def extract_from_freeze(event, freeze_event):
    # 1. Lövés helye (x,y)
    shooter_x, shooter_y = event['x'].iloc[0], event['y'].iloc[0]
    # 2. Lövés típusa
    body_part = event['body_part_name'].iloc[0]
    # 3. Lövés játékszituációja?
    play_pattern = event['play_pattern_name'].iloc[0]
    
    # 4. Védők száma kapu és lövő között (num_blockers_triangle)
    defenders = freeze_event[freeze_event['teammate'] == False]
    ## Kapufa pontok és lövő pont
    left_post = (120, 36)
    right_post = (120, 44)
    shooter_point = (shooter_x, shooter_y)
    ## Háromszög, amit a lövő és a két kapufa alkot
    shooting_triangle = Polygon([shooter_point, left_post, right_post])
    def count_blockers_in_triangle(player_df):
        count = 0
        for _, row in player_df.iterrows():
            player_point = Point(row['x'], row['y'])
            if shooting_triangle.contains(player_point):
                count += 1
        return count
    num_blockers_triangle = count_blockers_in_triangle(defenders)
    
    # 5. Átlagos távolság védőktől
    defenders_x = defenders['x'].to_numpy()
    defenders_y = defenders['y'].to_numpy()
    
    distances = np.sqrt((defenders_x - shooter_x)**2 + (defenders_y - shooter_y)**2)
    avg_def_dist = distances.mean()
    
    ## distance in front
    defenders_in_front = defenders[defenders['x'] > shooter_x]
    distances_if = np.sqrt((defenders_in_front['x'] - shooter_x)**2 + (defenders_in_front['y'] - shooter_y)**2)
    ## Átlagos távolság
    avg_def_front_dist = distances_if.mean() if not distances_if.empty else np.nan
    
    # 6. Boxon belüli játékosok
    BOX_X_MIN = 102
    BOX_X_MAX = 120
    BOX_Y_MIN = 18
    BOX_Y_MAX = 62
    BOX_mask = (freeze_event['x'] >= BOX_X_MIN) & (freeze_event['x'] <= BOX_X_MAX) & (
        freeze_event['y'] >= BOX_Y_MIN) & (freeze_event['y'] <= BOX_Y_MAX)
    ## Boxban lévő játékosok száma a freeze frame-ben
    players_in_box = freeze_event[BOX_mask]
    num_players_in_box = len(players_in_box)
    
    # 7. Védők/játékosok a boxban
    defenders_in_box = freeze_event[BOX_mask & (freeze_event.teammate == False)]
    num_defenders_in_box = len(defenders_in_box)
    
    # 8. Kapus pozíciója
    keeper_x, keeper_y = freeze_event[freeze_event.position_name == 'Goalkeeper'][['x', 'y']].iloc[0]
    
    
    # 9. Lövési szög
    def calculate_shot_angle(shooter, left_post, right_post):
        a = np.linalg.norm(np.array(shooter) - np.array(left_post))
        b = np.linalg.norm(np.array(shooter) - np.array(right_post))
        c = np.linalg.norm(np.array(left_post) - np.array(right_post))
    
        # Cosine rule to get angle at shooter position
        if a > 0 and b > 0:
            angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
            return np.degrees(angle)
        else:
            return 0.0  # if shooter is on top of a post
    shot_angle = calculate_shot_angle(shooter_point, left_post, right_post)
    
    # 10. Távolság a kaputól
    goal_center = (120, 40)
    distance_to_goal = np.sqrt((shooter_x - goal_center[0])**2 + (shooter_y - goal_center[1])**2)
    
    # 11. 1.5 méteres körzetben hány védő van?
    def count_close_defenders(defenders_df, shooter_x, shooter_y, radius=1.5):
        dx = defenders_df['x'] - shooter_x
        dy = defenders_df['y'] - shooter_y
        distances = np.sqrt(dx**2 + dy**2)
        return (distances <= radius).sum()
    num_close_defenders = count_close_defenders(defenders, shooter_x, shooter_y)
    
    # 12. Nyitott lövővonal hossza
    # Lövővonal (vonal a lövő és a kapu közepének között)
    shot_line = LineString([shooter_point, goal_center])

    # Védők pontjai
    defender_points = [Point(xy) for xy in defenders[['x', 'y']].to_numpy()]
    
    # Védők, akik közel vannak a lövővonalhoz (intersect ≈ blokkol)
    blocker_buffer = 0.5  # kb. 0.5m sugarú körrel reprezentáljuk a védőt
    blocked_sections = []
    
    for pt in defender_points:
        if shot_line.distance(pt) <= blocker_buffer:
            blocked_sections.append(pt)
    
    # Ha nincs blokkolás, teljes hossz szabad
    if not blocked_sections:
        open_shot_length = shot_line.length
    else:
        # Ha blokkolva van, akkor 0 szabad rész (egyszerűsített verzió)
        open_shot_length = 0.0
    
    # 13. Kapus elmozdulása a középvonaltól
    # Lövési irányvonal (shooter → kapu közepe)
    shot_line = LineString([shooter_point, goal_center])
    
    # Kapus pozíció pontként
    keeper_point = Point(keeper_x, keeper_y)
    
    # Oldaleltolódás: legkisebb távolság a lövési vonaltól
    keeper_lateral_offset = shot_line.distance(keeper_point)

    
    # Target: xG
    xg = event['shot_statsbomb_xg'].iloc[0]
    
    return {
                'x': shooter_x,
                'y': shooter_y,
                'shot_angle': shot_angle,
                'distance_to_goal': distance_to_goal,
                'num_blockers': num_blockers_triangle,
                'avg_def_dist': avg_def_dist,
                'avg_def_front_dist': avg_def_front_dist,
                'num_players_in_box': num_players_in_box,
                'num_defenders_in_box': num_defenders_in_box,
                'keeper_x': keeper_x,
                'keeper_y': keeper_y,
                'close_defenders': num_close_defenders,
                'body_part': body_part,
                'play_pattern': play_pattern,
                'open_shot_length': open_shot_length,
                'keeper_lateral_offset': keeper_lateral_offset,
                'xg': xg
            }
    
#%% Create dataframe
features = []
for match_id in df_matches.match_id.unique():
    # Specific match event data and freeze frames
    df, related, freeze, tactics = parser.event(match_id)
    
    print([team for team in df.team_name.unique()])
    print(f"Freeze frames in match: {len(freeze.id.unique())}")
    
    # Work with freeze frames
    for event_id in freeze.id.unique():
        freeze_event = freeze[freeze['id'] == event_id]
        event = df[df.id == event_id]
        
        # Build model
        features.append(extract_from_freeze(event, freeze_event))
        
#%% Build ML model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import warnings
warnings.filterwarnings("ignore")

state = 0
# Kategóriák kódolása
features_df = pd.DataFrame(features)
features_df = pd.get_dummies(features_df, columns=['body_part', 'play_pattern'], drop_first=True)

# --- Modellépítés ---
X = features_df.drop(columns='xg')
y = features_df['xg']

model = RandomForestRegressor(n_estimators=100, random_state=state,
                              max_depth= None, max_features = 'sqrt',
                              min_samples_leaf= 1, min_samples_split= 2
                              )

# --- Kiértékelés ---
rmse_scorer = make_scorer(mean_squared_error, squared=False)

cv_results = cross_validate(
    model, X, y,
    cv=5,
    scoring={'r2': 'r2', 'rmse': rmse_scorer},
    return_train_score=False
)

print(f"Mean R²: {cv_results['test_r2'].mean():.3f}")
print(f"Mean RMSE: {cv_results['test_rmse'].mean():.3f}")

#%%
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# --- Adatszétválasztás ---
X = features_df.drop(columns='xg')
y = features_df['xg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state)

# --- Modell definíció ---
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=state,
    n_jobs=-1
)

# --- Illesztés ---
xgb_model.fit(X_train, y_train)

# --- Predikció + Kiértékelés ---
y_pred = xgb_model.predict(X_test)

print(f"R² score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")

import matplotlib.pyplot as plt
import seaborn as sns

importance = xgb_model.feature_importances_
feature_names = X.columns

imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
imp_df = imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=imp_df.head(15))
plt.title('Top 15 fontos jellemző (XGBoost)')
plt.tight_layout()
plt.show()

#%% Save model
import joblib

# Train után
joblib.dump(model, 'xgboost_model.pkl')
