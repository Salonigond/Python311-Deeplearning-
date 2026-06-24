import pandas as pd
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv(
"backend/data/processed_data.csv"
)

team = {
"MI":0,
"RCB":1
}

df["batting_team"] = (
df["batting_team"]
.map(team)
.fillna(0)
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

y = df["pred_feature"]

scaler = joblib.load(
"backend/model/scaler.pkl"
)

X = scaler.transform(X)

model = Sequential()

model.add(
Dense(
128,
activation="relu",
input_shape=(7,)
)
)

model.add(
Dense(
64,
activation="relu"
)
)

model.add(
Dense(1)
)

model.compile(
optimizer="adam",
loss="mse"
)

model.fit(
X,
y,
epochs=50,
batch_size=8
)

model.save(
"backend/model/ann_model.keras"
)

print(
"ANN Saved"
)