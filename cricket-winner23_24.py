import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("E:\\ML datasets\\espncricinfo.csv")

# Target and features
target = "winner"
features = ['team1', 'team2', 'margin', 'ground', 'date'] 

# One-hot encode categorical features
X = pd.get_dummies(df[features], drop_first=True)

# Factorize target & keep mapping
y, winner_classes = pd.factorize(df[target])  # winner_classes will have team names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ======== Prediction ======== #
team1 = input("Enter Team 1 name: ")
team2 = input("Enter Team 2 name: ")
ground = input("Enter Ground: ")
match_year = 2025  # fixed for Asia Cup

new_match = pd.DataFrame([{
    "team1": team1,
    "team2": team2,
    "margin": 0,
    "ground": ground,
    "date": f"{match_year}-01-01"
}])

# One-hot encode & align columns
X_new = pd.get_dummies(new_match)
X_new = X_new.reindex(columns=X.columns, fill_value=0)

# Predict & decode winner name
pred_code = model.predict(X_new)[0]
predicted_winner = winner_classes[pred_code]

print(f"Predicted Winner: {predicted_winner}")
