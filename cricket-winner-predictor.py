import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("E:\\ML datasets\\New folder\\Men ODI Team Match Results - 21st Century.csv")

target = "Result"
features = ['Margin', 'Match', 'Home/Away', 'Ground', 'Match Date',
            'Match Month', 'Match Year', 'Match Period', 'Matches', 'Country']

# One-hot encode categorical features
X = pd.get_dummies(df[features], drop_first=True)
y  = pd.factorize(df[target])[0]  # Encode result into numbers


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# input for prediction
team1 = input("Enter Team 1 name: ")
team2 = input("Enter Team 2 name: ")
ground = input("Enter Ground: ")
home_away = input("Enter Home/Away/Neutral: ")
match_month = int(input("Enter Match Month (1-12): "))
match_year = 2025  # fixed for Asia Cup

# Create match description in same format as dataset
match_name = f"{team1} vs {team2}"

# 8️⃣ Create DataFrame for new match
new_match = pd.DataFrame([{
    "Margin": 0,  # unknown before match
    "Match": match_name,
    "Home/Away": home_away,
    "Ground": ground,
    "Match Date": 1,  # dummy date
    "Match Month": match_month,
    "Match Year": match_year,
    "Match Period": "Asia Cup",
    "Matches": 1,
    "Country": team1  # assume team1's country
}])

# One-hot encode & align with training columns
X_new = pd.get_dummies(new_match)
X_new = X_new.reindex(columns=X.columns, fill_value=0)

# 🔟 Predict winner
pred = model.predict(X_new)[0]
predicted_result = y[pred]

print(f"Predicted Winner: {predicted_result}")