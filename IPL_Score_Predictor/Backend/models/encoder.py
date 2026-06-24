import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv(
"backend/data/processed_data.csv"
)

encoder = LabelEncoder()

df["batting_team"] = encoder.fit_transform(
df["batting_team"]
)

X = df[
[
"batting_team",
"current_score",
"wickets",
"overs",
"run_rate",
"pressure",
"momentum"
]
]

scaler = StandardScaler()

X = scaler.fit_transform(X)

joblib.dump(
scaler,
"backend/model/scaler.pkl"
)

print(
"Scaler Saved"
)