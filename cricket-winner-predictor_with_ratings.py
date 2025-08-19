import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===== Load datasets =====
df_matches = pd.read_csv("E:\\ML datasets\\espncricinfo.csv")
df_ratings = pd.read_csv("E:\\ML datasets\\t20_teams_ranking.csv")

# ===== Merge team ratings =====
# Team 1 rating
df_matches = df_matches.merge(df_ratings[["Team", "Rating"]],
                               left_on="team1", right_on="Team", how="left")
df_matches.rename(columns={"Rating": "team1_rating"}, inplace=True)
df_matches.drop(columns=["Team"], inplace=True)

# Team 2 rating
df_matches = df_matches.merge(df_ratings[["Team", "Rating"]],
                               left_on="team2", right_on="Team", how="left")
df_matches.rename(columns={"Rating": "team2_rating"}, inplace=True)
df_matches.drop(columns=["Team"], inplace=True)

# Rating difference
df_matches["rating_diff"] = df_matches["team1_rating"] - df_matches["team2_rating"]

# ===== Prepare features and target =====
target = "winner"
features = ['team1', 'team2', 'team1_rating', 'team2_rating', 'rating_diff', 'margin', 'ground', 'date']

# One-hot encode categorical features
X = pd.get_dummies(df_matches[features], drop_first=True)

# Encode target
y, winner_classes = pd.factorize(df_matches[target])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ===== Prediction =====
team1 = input("Enter Team 1 name: ")
team2 = input("Enter Team 2 name: ")
ground = input("Enter Ground: ")
match_year = 2025

# Create match row
new_match = pd.DataFrame([{
    "team1": team1,
    "team2": team2,
    "margin": 0,
    "ground": ground,
    "date": f"{match_year}-01-01"
}])

# Merge ratings into prediction data
new_match = new_match.merge(df_ratings[["Team", "Rating"]],
                            left_on="team1", right_on="Team", how="left")
new_match.rename(columns={"Rating": "team1_rating"}, inplace=True)
new_match.drop(columns=["Team"], inplace=True)

new_match = new_match.merge(df_ratings[["Team", "Rating"]],
                            left_on="team2", right_on="Team", how="left")
new_match.rename(columns={"Rating": "team2_rating"}, inplace=True)
new_match.drop(columns=["Team"], inplace=True)

new_match["rating_diff"] = new_match["team1_rating"] - new_match["team2_rating"]

# One-hot encode & align columns with training data
X_new = pd.get_dummies(new_match)
X_new = X_new.reindex(columns=X.columns, fill_value=0)

# Predict winner
pred_code = model.predict(X_new)[0]
predicted_winner = winner_classes[pred_code]

print(f"Predicted Winner: {predicted_winner}")
