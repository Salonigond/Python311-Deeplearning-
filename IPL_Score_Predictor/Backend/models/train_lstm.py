import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

df=pd.read_csv(
"backend/data/processed_data.csv"
)

df["batting_team"]=0

X=df[
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

y=df["pred_feature"]

scaler=joblib.load(
"backend/model/scaler.pkl"
)

X=scaler.transform(X)

X=np.reshape(
X,
(
X.shape[0],
1,
7
)
)

model=Sequential()

model.add(
LSTM(
128,
input_shape=(1,7)
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
epochs=50
)

model.save(
"backend/model/lstm_model.keras"
)

print(
"LSTM Saved"
)